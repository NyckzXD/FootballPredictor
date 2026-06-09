"""
season_model.py  —  Monte Carlo Brasileirão (versão otimizada)

Otimizações aplicadas vs versão anterior:
  1. history como deque por time (dict[team] → deque): elimina busca O(n) na lista global
  2. h2h_cache: dict pré-computado de confrontos diretos entre pares de times
  3. np.random.default_rng por worker: elimina overhead de validação do np.random.choice
  4. IsotonicRegression substituída por interpolação numpy pré-computada (lookup array)
  5. Backend "threading" no Parallel: sem serialização pickle dos modelos entre workers
  6. history_init copiado como arrays numpy, não lista de dicts
  7. apply_result opera em arrays numpy de tamanho fixo (sem append à lista)
  8. rng.choice vetorizado para toda a rodada de uma vez
"""

import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from collections import deque
import warnings, time
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

MODEL_PATH         = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"
POISSON_MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\poisson_model.pkl"
MATCHES_PATH       = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
FIXTURES_PATH      = r"C:\PREDICTOR\REPO\scraping\data\raw\matches.csv"
MARKET_PATH        = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"
ODDS_PATH          = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
OUTPUT_PATH        = r"C:\PREDICTOR\REPO\scraping\data\processed\simulacao_2026.csv"

N_SIMS  = 10_000
N_JOBS  = 12
SEASON  = 2026

TEAM_MAP_ODDS = {
    "America MG": "América FC", "Athletico-PR": "CA Paranaense",
    "Atletico GO": "Atlético Goianiense", "Atletico-MG": "CA Mineiro",
    "Avai": "Avaí FC", "Bahia": "EC Bahia", "Botafogo RJ": "Botafogo FR",
    "Bragantino": "RB Bragantino", "Ceara": "Ceará SC",
    "Chapecoense-SC": "Chapecoense AF", "Corinthians": "SC Corinthians Paulista",
    "Coritiba": "Coritiba FBC", "Criciuma": "Criciúma EC",
    "Cruzeiro": "Cruzeiro EC", "Cuiaba": "Cuiabá EC",
    "Flamengo RJ": "CR Flamengo", "Fluminense": "Fluminense FC",
    "Fortaleza": "Fortaleza EC", "Goias": "Goiás EC", "Gremio": "Grêmio FBPA",
    "Internacional": "SC Internacional", "Juventude": "EC Juventude",
    "Mirassol": "Mirassol FC", "Palmeiras": "SE Palmeiras",
    "Parana": "Paraná Clube", "Remo": "Clube do Remo", "Santos": "Santos FC",
    "Sao Paulo": "São Paulo FC", "Sport Recife": "Sport Club do Recife",
    "Vasco": "CR Vasco da Gama", "Vitoria": "EC Vitória", "CSA": "CSA",
}


# ── Elo ───────────────────────────────────────────────────────────────────────
def expected_score(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def update_elo_scalar(ra, rb, result, k=32):
    ea = expected_score(ra, rb)
    if result == "H":   sa, sb = 1.0, 0.0
    elif result == "A": sa, sb = 0.0, 1.0
    else:               sa, sb = 0.5, 0.5
    return ra + k * (sa - ea), rb + k * (sb - (1.0 - ea))


# ── IsotonicRegression → lookup rápido ────────────────────────────────────────
def build_iso_lookup(iso_model, n_points=1001):
    """
    Converte IsotonicRegression em dois arrays (x_pts, y_pts) para
    interpolação numpy — 10-50x mais rápido que iso.predict() em batch pequeno.
    """
    x_pts = np.linspace(0.0, 1.0, n_points)
    y_pts = iso_model.predict(x_pts)
    return x_pts, y_pts

def iso_apply(raw, x_pts, y_pts):
    """Substitui iso.predict(raw) com np.interp — zero overhead Python."""
    return np.interp(raw, x_pts, y_pts)


# ── Estruturas de histórico por time (evita busca linear O(n)) ─────────────
class TeamHistory:
    """
    Mantém deque de até N=10 jogos por time, indexado por nome.
    Muito mais rápido que filtrar lista global a cada chamada.
    """
    __slots__ = ("data", "maxlen")

    def __init__(self, maxlen=10):
        self.data   = {}
        self.maxlen = maxlen

    def copy(self):
        th = TeamHistory(self.maxlen)
        th.data = {t: deque(d, maxlen=self.maxlen) for t, d in self.data.items()}
        return th

    def add(self, home, away, hg, ag):
        # (is_home, goals_for, goals_against)
        entry_h = (True,  hg, ag)
        entry_a = (False, ag, hg)
        if home not in self.data:
            self.data[home] = deque(maxlen=self.maxlen)
        if away not in self.data:
            self.data[away] = deque(maxlen=self.maxlen)
        self.data[home].append(entry_h)
        self.data[away].append(entry_a)

    def stats(self, team, n=5):
        entries = list(self.data.get(team, []))[-n:]
        if not entries:
            return (1.0, 1.2, 1.0, 0.2, 0.4, 0.25, 0.0, 0.0)
        pts = gf = ga = wins = draws = home_f = away_f = hc = ac = 0
        for is_home, gfor, gagainst in entries:
            gf += gfor; ga += gagainst
            if   gfor > gagainst: pts += 3; wins  += 1
            elif gfor == gagainst: pts += 1; draws += 1
            if is_home: home_f += gfor - gagainst; hc += 1
            else:       away_f += gfor - gagainst; ac += 1
        n_ = len(entries)
        # retorna tupla (pts, gf, ga, gd, wr, dr, home_f, away_f)
        return (pts/n_, gf/n_, ga/n_, (gf-ga)/n_,
                wins/n_, draws/n_,
                home_f/hc if hc else 0.0,
                away_f/ac if ac else 0.0)


# ── H2H cache pré-computado ──────────────────────────────────────────────────
def build_h2h_cache(history_records):
    """
    Pré-computa h2h (hw, aw, d) para todos os pares do histórico.
    Retorna dict {(home, away): (hw, aw, draws)}.
    """
    from collections import defaultdict
    raw = defaultdict(lambda: [0, 0, 0])  # [hw, aw, d]
    for home, away, hg, ag in history_records:
        key = (home, away)
        if   hg > ag: raw[key][0] += 1
        elif ag > hg: raw[key][1] += 1
        else:         raw[key][2] += 1
    return dict(raw)

def get_h2h(h2h_cache, home, away):
    """Busca h2h no cache; inverte perspectiva se necessário."""
    fwd = h2h_cache.get((home, away), None)
    if fwd:
        return fwd[0], fwd[1], fwd[2]
    rev = h2h_cache.get((away, home), None)
    if rev:
        return rev[1], rev[0], rev[2]
    return 0, 0, 0


# ── Features inline (sem dict temporário) ────────────────────────────────────
def compute_feature_row(home, away, live_table, elo_ratings, team_hist,
                        h2h_cache, mv_dict, odds_dict, max_val,
                        feat_order, derived_order):
    """
    Retorna numpy array com features na ordem correta de feat_order + derived_order.
    Não cria dicts intermediários — vai direto para array.
    """
    hs   = team_hist.stats(home, 5)
    as_  = team_hist.stats(away, 5)
    hs10 = team_hist.stats(home, 10)
    as10 = team_hist.stats(away, 10)

    h2h_hw, h2h_aw, h2h_d = get_h2h(h2h_cache, home, away)

    ht = live_table[home]; at = live_table[away]
    h_aprov = ht[0] / max(ht[3] * 3, 1)  # pts / (played*3)
    a_aprov = at[0] / max(at[3] * 3, 1)

    h_pos = max(1, int((1.0 - h_aprov) * 20))
    a_pos = max(1, int((1.0 - a_aprov) * 20))

    h_elo = elo_ratings.get(home, 1500.0)
    a_elo = elo_ratings.get(away, 1500.0)
    h_mv  = mv_dict.get(home, 50.0)
    a_mv  = mv_dict.get(away, 50.0)

    key = (home, away)
    if key in odds_dict:
        o = odds_dict[key]
        prob_h_mkt       = o[0]
        prob_d_mkt       = o[1]
        prob_a_mkt       = o[2]
        oh, od, oa       = o[3], o[4], o[5]
        odds_draw_factor = od / ((oh + oa) / 2.0)
        odds_har         = oh / max(oa, 0.01)
        market_entropy   = -(
            prob_h_mkt * np.log(prob_h_mkt + 1e-9) +
            prob_d_mkt * np.log(prob_d_mkt + 1e-9) +
            prob_a_mkt * np.log(prob_a_mkt + 1e-9)
        )
    else:
        e              = expected_score(h_elo, a_elo)
        prob_h_mkt     = e * 0.85 + 0.05
        prob_a_mkt     = (1.0 - e) * 0.75 + 0.05
        prob_d_mkt     = max(1.0 - prob_h_mkt - prob_a_mkt, 0.05)
        odds_draw_factor = 1.0
        odds_har       = prob_h_mkt / max(prob_a_mkt, 0.01)
        market_entropy = 1.0

    elo_diff       = h_elo - a_elo
    mv_diff        = h_mv - a_mv
    h_mv_log       = np.log1p(h_mv)
    a_mv_log       = np.log1p(a_mv)
    h_mv_norm      = h_mv / max_val
    a_mv_norm      = a_mv / max_val
    pos_diff       = h_pos - a_pos

    # Derived features
    form_diff      = hs[0]  - as_[0]
    form_diff_10   = hs10[0] - as10[0]
    gf_diff        = hs[1]  - as_[1]
    ga_diff        = hs[2]  - as_[2]
    wr_diff        = hs[4]  - as_[4]
    aprov_diff     = h_aprov - a_aprov
    h_in_crisis    = float(hs[0] < 0.5)
    a_in_form      = float(as_[0] > 2.0)
    elo_sim        = 1.0 / (1.0 + abs(elo_diff))
    form_sim       = 1.0 / (1.0 + abs(form_diff))
    val_sim        = 1.0 / (1.0 + abs(mv_diff))
    overall_bal    = (elo_sim + form_sim + val_sim) / 3.0
    comb_dr        = (hs[5] + as_[5]) / 2.0
    both_low_sc    = float(hs[1] < 1.2 and as_[1] < 1.2)
    both_good_def  = float(hs[2] < 1.0 and as_[2] < 1.0)
    total_h2h      = h2h_hw + h2h_aw + h2h_d + 1
    h2h_dr         = h2h_d / total_h2h
    h2h_dec        = (h2h_hw + h2h_aw) / total_h2h
    pos_sim        = 1.0 / (1.0 + abs(pos_diff))
    elo_vs_mkt_h   = elo_sim - prob_h_mkt
    elo_vs_mkt_a   = (1.0 - elo_sim) - prob_a_mkt

    # Mapa nome → valor (lookup único para construir array na ordem certa)
    fmap = {
        "elo_diff": elo_diff, "home_elo": h_elo, "away_elo": a_elo,
        "home_market_value_log": h_mv_log, "away_market_value_log": a_mv_log,
        "market_value_diff": mv_diff,
        "home_market_value_norm": h_mv_norm, "away_market_value_norm": a_mv_norm,
        "home_squad_size": 20.0, "away_squad_size": 20.0,
        "home_aproveitamento": h_aprov, "away_aproveitamento": a_aprov,
        "position_diff": float(pos_diff),
        "home_form_pts":  hs[0],  "home_avg_gf":  hs[1],
        "home_avg_ga":    hs[2],  "home_goal_diff": hs[3],
        "home_win_rate":  hs[4],  "home_draw_rate": hs[5],
        "home_home_form": hs[6],
        "away_form_pts":  as_[0], "away_avg_gf":  as_[1],
        "away_avg_ga":    as_[2], "away_goal_diff": as_[3],
        "away_win_rate":  as_[4], "away_draw_rate": as_[5],
        "away_away_form": as_[7],
        "home_form_pts_10": hs10[0], "home_avg_gf_10": hs10[1],
        "home_avg_ga_10":   hs10[2], "home_win_rate_10": hs10[4],
        "away_form_pts_10": as10[0], "away_avg_gf_10": as10[1],
        "away_avg_ga_10":   as10[2], "away_win_rate_10": as10[4],
        "h2h_home_wins": float(h2h_hw), "h2h_away_wins": float(h2h_aw),
        "h2h_draws": float(h2h_d),
        "prob_h_mkt": prob_h_mkt, "prob_d_mkt": prob_d_mkt,
        "prob_a_mkt": prob_a_mkt,
        "odds_draw_factor": odds_draw_factor,
        "odds_home_away_ratio": odds_har,
        "market_entropy": market_entropy,
        # derived
        "form_diff": form_diff, "form_diff_10": form_diff_10,
        "gf_diff": gf_diff, "ga_diff": ga_diff,
        "win_rate_diff": wr_diff, "aproveit_diff": aprov_diff,
        "home_in_crisis": h_in_crisis, "away_in_form": a_in_form,
        "elo_similarity": elo_sim, "form_similarity": form_sim,
        "value_similarity": val_sim, "overall_balance": overall_bal,
        "home_draw_tendency": hs[5], "away_draw_tendency": as_[5],
        "combined_draw_rate": comb_dr,
        "both_low_scoring": both_low_sc, "both_good_defense": both_good_def,
        "h2h_draw_rate": h2h_dr, "h2h_decisividade": h2h_dec,
        "position_similarity": pos_sim,
        "elo_vs_mkt_h": elo_vs_mkt_h, "elo_vs_mkt_a": elo_vs_mkt_a,
    }
    return np.array([fmap.get(k, 0.0) for k in feat_order], dtype=np.float32)


# ── Simulação individual ──────────────────────────────────────────────────────
def run_simulation(sim_id,
                   fixtures_by_round, completed,
                   teams, n_teams,
                   elo_init, hist_init,     # hist_init = TeamHistory pré-populado
                   h2h_cache_init,
                   mv_dict, odds_dict, max_val,
                   model_data, poisson_data,
                   feat_order,
                   poisson_feat_h, poisson_feat_a,
                   iso_x, iso_h_y, iso_d_y, iso_a_y):   # lookup arrays

    rng = np.random.default_rng(sim_id)

    # live_table: array [pts, gd, gf, played] por time
    team_idx   = {t: i for i, t in enumerate(teams)}
    table      = np.zeros((n_teams, 4), dtype=np.float32)  # pts, gd, gf, played

    elo_ratings = elo_init.copy()
    team_hist   = hist_init.copy()
    h2h_cache   = dict(h2h_cache_init)   # shallow copy — suficiente (não mutamos valores)

    # Construir live_table como dict de views no array (acesso O(1) por nome)
    def get_table(team):
        return table[team_idx[team]]

    # Aplicar jogos já disputados
    for h, a, hg, ag in completed:
        hi = team_idx[h]; ai = team_idx[a]
        res = "H" if hg > ag else ("A" if ag > hg else "D")
        table[hi, 3] += 1; table[ai, 3] += 1
        table[hi, 2] += hg; table[ai, 2] += ag
        table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
        if   res == "H": table[hi, 0] += 3
        elif res == "A": table[ai, 0] += 3
        else:            table[hi, 0] += 1; table[ai, 0] += 1
        elo_ratings[h], elo_ratings[a] = update_elo_scalar(
            elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), res)
        team_hist.add(h, a, hg, ag)

    # Wrapper para live_table compatível com compute_feature_row
    class TableView:
        """Proxy que expõe live_table[team] como tupla (pts, gd, gf, played)."""
        __slots__ = ("_t", "_idx")
        def __init__(self, t_, idx_): self._t = t_; self._idx = idx_
        def __getitem__(self, team):
            row = self._t[self._idx[team]]
            return row  # retorna array [pts, gd, gf, played]
        def get(self, team, default):
            if team in self._idx: return self[team]
            return default

    live_table_view = {t: table[team_idx[t]] for t in teams}

    # Helper inline para live_table_view[team][0] = pts, [3] = played
    def h_aprov_fn(team):
        row = table[team_idx[team]]
        return float(row[0]) / max(float(row[3]) * 3.0, 1.0)

    # Simular por rodada
    model_h = model_data["model_h"]; model_d = model_data["model_d"]
    model_a = model_data["model_a"]
    sc_h    = poisson_data["scaler_home"]; sc_a = poisson_data["scaler_away"]
    pm_h    = poisson_data["model_home"];  pm_a = poisson_data["model_away"]

    for round_games in fixtures_by_round:
        if not round_games:
            continue
        n_games = len(round_games)

        # Build feature matrix — uma linha por jogo da rodada
        X_rows = []
        for (h, a) in round_games:
            row = compute_feature_row(
                h, a, live_table_view, elo_ratings, team_hist,
                h2h_cache, mv_dict, odds_dict, max_val,
                feat_order, None  # derived_order não usado (já embutido)
            )
            X_rows.append(row)
        X = np.vstack(X_rows).astype(np.float64)

        # LightGBM batch predict
        raw_h = model_h.predict_proba(X)[:, 1]
        raw_d = model_d.predict_proba(X)[:, 1]
        raw_a = model_a.predict_proba(X)[:, 1]

        # Calibração via lookup numpy (substitui IsotonicRegression.predict)
        ph = iso_apply(raw_h, iso_x, iso_h_y)
        pd_ = iso_apply(raw_d, iso_x, iso_d_y)
        pa = iso_apply(raw_a, iso_x, iso_a_y)
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        # Poisson batch
        X_ph = X[:, [list(feat_order).index(k) for k in poisson_feat_h]] \
            if hasattr(poisson_feat_h, '__iter__') else X
        X_pa = X[:, [list(feat_order).index(k) for k in poisson_feat_a]] \
            if hasattr(poisson_feat_a, '__iter__') else X

        # Reindexar colunas Poisson
        forder_list = list(feat_order)
        ph_idx = [forder_list.index(k) for k in poisson_feat_h]
        pa_idx = [forder_list.index(k) for k in poisson_feat_a]
        lam_h = np.clip(pm_h.predict(sc_h.transform(X[:, ph_idx])), 0.1, 8.0)
        lam_a = np.clip(pm_a.predict(sc_a.transform(X[:, pa_idx])), 0.1, 8.0)

        hg_arr = rng.poisson(lam_h).astype(np.int32)
        ag_arr = rng.poisson(lam_a).astype(np.int32)

        # Escolha de resultado via rng vetorizada (sem np.random.choice por jogo)
        cumprob = np.column_stack([ph, ph + pd_])           # (n_games, 2)
        rand_v  = rng.random(n_games)
        res_idx = np.where(rand_v < cumprob[:, 0], 0,
                  np.where(rand_v < cumprob[:, 1], 1, 2))   # 0=H,1=D,2=A

        # Aplicar resultados
        for i, (h, a) in enumerate(round_games):
            ri  = res_idx[i]
            hg  = int(hg_arr[i])
            ag  = int(ag_arr[i])

            if   ri == 0 and hg <= ag: hg = ag + 1   # H
            elif ri == 2 and ag <= hg: ag = hg + 1   # A
            elif ri == 1: ag = hg                    # D

            res_str = "H" if ri == 0 else ("A" if ri == 2 else "D")
            hi = team_idx[h]; ai = team_idx[a]

            table[hi, 3] += 1; table[ai, 3] += 1
            table[hi, 2] += hg; table[ai, 2] += ag
            table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
            if   ri == 0: table[hi, 0] += 3
            elif ri == 2: table[ai, 0] += 3
            else:         table[hi, 0] += 1; table[ai, 0] += 1

            elo_ratings[h], elo_ratings[a] = update_elo_scalar(
                elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), res_str)
            team_hist.add(h, a, hg, ag)
            # Atualizar live_table_view (aponta para table row — já atualizado)

    # Classificação final
    pts_arr = table[:, 0]; gd_arr = table[:, 1]; gf_arr = table[:, 2]
    order   = np.lexsort((-gf_arr, -gd_arr, -pts_arr))
    positions = {teams[idx]: int(rank + 1) for rank, idx in enumerate(order)}
    pts_final = {teams[i]: float(table[i, 0]) for i in range(n_teams)}
    return positions, pts_final


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Carregando dados...")
    t_start = time.time()

    hist_df = pd.read_csv(MATCHES_PATH)
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    hist_df = hist_df.dropna(subset=["home_goals", "away_goals"])

    cal_df   = pd.read_csv(FIXTURES_PATH)
    cal_df["date"] = pd.to_datetime(cal_df["date"], errors="coerce")
    cal_2026 = cal_df[cal_df["season"] == SEASON].copy()

    completed_df = cal_2026[cal_2026["home_goals"].notna()].copy()
    fixtures_df  = cal_2026[cal_2026["home_goals"].isna()].copy()

    print(f"   {SEASON}: {len(completed_df)} disputados | {len(fixtures_df)} a simular")

    teams   = sorted(set(cal_2026["home_team"].tolist() + cal_2026["away_team"].tolist()))
    n_teams = len(teams)
    print(f"   {n_teams} times")

    if len(fixtures_df) == 0:
        print("Nenhuma fixture futura — verifique matches.csv")
        return

    completed_list = [
        (r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"]))
        for _, r in completed_df.iterrows()
    ]

    if "matchday" in fixtures_df.columns:
        fixtures_by_round = [
            [(r["home_team"], r["away_team"])
             for _, r in fixtures_df[fixtures_df["matchday"] == rd].iterrows()]
            for rd in sorted(fixtures_df["matchday"].unique())
        ]
    else:
        fixtures_by_round = [
            [(r["home_team"], r["away_team"]) for _, r in fixtures_df.iterrows()]
        ]

    total_fix = sum(len(r) for r in fixtures_by_round)
    print(f"   {len(fixtures_by_round)} rodadas | {total_fix} jogos a simular")

    # ── Histórico → TeamHistory + h2h_cache ──
    hist_records_raw = hist_df[hist_df["season"] < SEASON].tail(5000)
    history_tuples = [
        (r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"]))
        for _, r in hist_records_raw.iterrows()
    ]

    # Elo inicial sobre histórico
    elo_ratings = {t: 1500.0 for t in teams}
    for h, a, hg, ag in history_tuples:
        if h not in elo_ratings: elo_ratings[h] = 1500.0
        if a not in elo_ratings: elo_ratings[a] = 1500.0
        res = "H" if hg > ag else ("A" if ag > hg else "D")
        elo_ratings[h], elo_ratings[a] = update_elo_scalar(
            elo_ratings[h], elo_ratings[a], res)

    # Construir TeamHistory pré-populado (cópia barata depois)
    hist_init = TeamHistory(maxlen=10)
    for h, a, hg, ag in history_tuples:
        hist_init.add(h, a, hg, ag)

    # H2H cache pré-computado
    h2h_cache_init = build_h2h_cache(history_tuples)
    print(f"   h2h cache: {len(h2h_cache_init)} pares")

    # ── Valor de mercado ──
    try:
        mv_df   = pd.read_csv(MARKET_PATH)
        mv_dict = dict(zip(mv_df["team"], mv_df["market_value"]))
        print(f"   Valores de mercado: {len(mv_dict)} times")
    except Exception:
        mv_dict = {}
    max_val = max(mv_dict.values()) if mv_dict else 1.0

    # ── Odds ──
    odds_dict = {}
    try:
        odds_df = pd.read_csv(ODDS_PATH)
        odds_df = odds_df[odds_df["Season"] == SEASON]
        for _, row in odds_df.iterrows():
            h = TEAM_MAP_ODDS.get(row["Home"], row["Home"])
            a = TEAM_MAP_ODDS.get(row["Away"], row["Away"])
            oh = row.get("AvgCH") or row.get("PSCH")
            od = row.get("AvgCD") or row.get("PSCD")
            oa = row.get("AvgCA") or row.get("PSCA")
            if pd.notna(oh) and oh > 0:
                tot = 1/oh + 1/od + 1/oa
                # Guardar como tupla (mais leve que dict)
                odds_dict[(h, a)] = (
                    (1/oh)/tot, (1/od)/tot, (1/oa)/tot,
                    oh, od, oa
                )
        print(f"   Odds {SEASON}: {len(odds_dict)} jogos")
    except Exception as e:
        print(f"   Odds: {e}")

    # ── Modelos ──
    print("\nCarregando modelos...")
    model_data   = joblib.load(MODEL_PATH)
    poisson_data = joblib.load(POISSON_MODEL_PATH)
    feat_order        = tuple(model_data["features"])
    poisson_feat_h    = tuple(poisson_data["features_home"])
    poisson_feat_a    = tuple(poisson_data["features_away"])

    # Pré-computar índices Poisson (evita list.index() dentro da simulação)
    forder_list    = list(feat_order)
    poisson_idx_h  = [forder_list.index(k) for k in poisson_feat_h]
    poisson_idx_a  = [forder_list.index(k) for k in poisson_feat_a]

    # Construir lookups isotônicos
    iso_x   = np.linspace(0.0, 1.0, 1001)
    iso_h_y = model_data["cal_h"].predict(iso_x)
    iso_d_y = model_data["cal_d"].predict(iso_x)
    iso_a_y = model_data["cal_a"].predict(iso_x)
    print("   Modelos + lookups carregados")

    # ── Versão interna de run_simulation com índices Poisson pré-computados ──
    def run_sim_fast(sim_id,
                     fixtures_by_round, completed,
                     teams, n_teams,
                     elo_init, hist_init,
                     h2h_cache_init,
                     mv_dict, odds_dict, max_val,
                     model_data, poisson_data,
                     feat_order,
                     poisson_idx_h, poisson_idx_a,
                     iso_x, iso_h_y, iso_d_y, iso_a_y):

        rng = np.random.default_rng(sim_id)
        team_idx  = {t: i for i, t in enumerate(teams)}
        table     = np.zeros((n_teams, 4), dtype=np.float32)
        elo_r     = elo_init.copy()
        th        = hist_init.copy()
        h2h_cache = h2h_cache_init  # read-only — não precisa copiar

        # Aplicar jogos disputados
        for h, a, hg, ag in completed:
            hi = team_idx[h]; ai = team_idx[a]
            res = "H" if hg > ag else ("A" if ag > hg else "D")
            table[hi, 3] += 1; table[ai, 3] += 1
            table[hi, 2] += hg; table[ai, 2] += ag
            table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
            if   res == "H": table[hi, 0] += 3
            elif res == "A": table[ai, 0] += 3
            else:            table[hi, 0] += 1; table[ai, 0] += 1
            elo_r[h], elo_r[a] = update_elo_scalar(
                elo_r.get(h, 1500.0), elo_r.get(a, 1500.0), res)
            th.add(h, a, hg, ag)

        live_tv = {t: table[team_idx[t]] for t in teams}

        model_h = model_data["model_h"]; model_d_ = model_data["model_d"]
        model_a = model_data["model_a"]
        sc_h = poisson_data["scaler_home"]; sc_a = poisson_data["scaler_away"]
        pm_h = poisson_data["model_home"];  pm_a = poisson_data["model_away"]

        for round_games in fixtures_by_round:
            if not round_games:
                continue
            n_games = len(round_games)
            X = np.empty((n_games, len(feat_order)), dtype=np.float64)
            for gi, (h, a) in enumerate(round_games):
                X[gi] = compute_feature_row(
                    h, a, live_tv, elo_r, th,
                    h2h_cache, mv_dict, odds_dict, max_val,
                    feat_order, None)

            raw_h = model_h.predict_proba(X)[:, 1]
            raw_d = model_d_.predict_proba(X)[:, 1]
            raw_a = model_a.predict_proba(X)[:, 1]
            ph  = np.interp(raw_h, iso_x, iso_h_y)
            pd_ = np.interp(raw_d, iso_x, iso_d_y)
            pa  = np.interp(raw_a, iso_x, iso_a_y)
            tot = ph + pd_ + pa
            ph /= tot; pd_ /= tot; pa /= tot

            lam_h = np.clip(pm_h.predict(sc_h.transform(X[:, poisson_idx_h])), 0.1, 8.0)
            lam_a = np.clip(pm_a.predict(sc_a.transform(X[:, poisson_idx_a])), 0.1, 8.0)
            hg_arr = rng.poisson(lam_h).astype(np.int32)
            ag_arr = rng.poisson(lam_a).astype(np.int32)

            cumprob = np.column_stack([ph, ph + pd_])
            rand_v  = rng.random(n_games)
            res_idx = np.where(rand_v < cumprob[:, 0], 0,
                      np.where(rand_v < cumprob[:, 1], 1, 2))

            for i, (h, a) in enumerate(round_games):
                ri = int(res_idx[i])
                hg = int(hg_arr[i]); ag = int(ag_arr[i])
                if   ri == 0 and hg <= ag: hg = ag + 1
                elif ri == 2 and ag <= hg: ag = hg + 1
                elif ri == 1: ag = hg

                hi = team_idx[h]; ai = team_idx[a]
                table[hi, 3] += 1; table[ai, 3] += 1
                table[hi, 2] += hg; table[ai, 2] += ag
                table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
                if   ri == 0: table[hi, 0] += 3
                elif ri == 2: table[ai, 0] += 3
                else:         table[hi, 0] += 1; table[ai, 0] += 1

                res_str = "H" if ri == 0 else ("A" if ri == 2 else "D")
                elo_r[h], elo_r[a] = update_elo_scalar(
                    elo_r.get(h, 1500.0), elo_r.get(a, 1500.0), res_str)
                th.add(h, a, hg, ag)

        pts_arr = table[:, 0]; gd_arr = table[:, 1]; gf_arr = table[:, 2]
        order   = np.lexsort((-gf_arr, -gd_arr, -pts_arr))
        positions = {teams[idx]: int(rank + 1) for rank, idx in enumerate(order)}
        pts_final = {teams[i]: float(table[i, 0]) for i in range(n_teams)}
        return positions, pts_final

    # ── Benchmark ──
    print("\nBenchmark (10 simulacoes)...")
    t0 = time.time()
    for i in range(10):
        run_sim_fast(i, fixtures_by_round, completed_list,
                     teams, n_teams, elo_ratings, hist_init,
                     h2h_cache_init, mv_dict, odds_dict, max_val,
                     model_data, poisson_data, feat_order,
                     poisson_idx_h, poisson_idx_a,
                     iso_x, iso_h_y, iso_d_y, iso_a_y)
    t_bench = (time.time() - t0) / 10
    print(f"   {t_bench:.3f}s por simulacao | ETA {N_SIMS:,} sims: {t_bench*N_SIMS/N_JOBS/60:.1f} min (com {N_JOBS} CPUs)")

    # ── Monte Carlo ──
    print(f"\nRodando {N_SIMS:,} simulacoes ({N_JOBS} CPUs — backend threading)...")
    t0 = time.time()

    results = Parallel(n_jobs=N_JOBS, backend="threading", verbose=0)(
        delayed(run_sim_fast)(
            i, fixtures_by_round, completed_list,
            teams, n_teams, elo_ratings, hist_init,
            h2h_cache_init, mv_dict, odds_dict, max_val,
            model_data, poisson_data, feat_order,
            poisson_idx_h, poisson_idx_a,
            iso_x, iso_h_y, iso_d_y, iso_a_y
        )
        for i in range(N_SIMS)
    )

    elapsed = time.time() - t0
    print(f"   Concluido em {elapsed/60:.1f} min ({elapsed/N_SIMS:.3f}s/sim)")

    # ── Agregar ──
    print("\nAgregando resultados...")
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int32)
    pts_array       = np.zeros((N_SIMS, n_teams), dtype=np.float32)
    team_idx_map    = {t: i for i, t in enumerate(teams)}

    for sim_i, (positions, pts) in enumerate(results):
        for t, pos in positions.items():
            ti = team_idx_map[t]
            position_counts[ti, pos - 1] += 1
            pts_array[sim_i, ti] = pts.get(t, 0)

    rows = []
    for ti, t in enumerate(teams):
        pc = position_counts[ti] / N_SIMS
        rows.append({
            "time":             t,
            "titulo_pct":       round(float(pc[0]) * 100, 1),
            "libertadores_pct": round(float(pc[:6].sum()) * 100, 1),
            "sulamericana_pct": round(float(pc[:12].sum()) * 100, 1),
            "rebaixamento_pct": round(float(pc[-4:].sum()) * 100, 1),
            "pts_esperados":    round(float(pts_array[:, ti].mean()), 1),
            "pts_std":          round(float(pts_array[:, ti].std()), 1),
            "pos_esperada":     round(float(
                sum(results[s][0].get(t, n_teams) for s in range(N_SIMS)) / N_SIMS
            ), 1),
        })

    df_sim = pd.DataFrame(rows).sort_values("pts_esperados", ascending=False)
    df_sim.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*68}")
    print(f"{N_SIMS:,} simulacoes em {elapsed/60:.1f} min")
    print(f"\nBRASILEIRAO {SEASON} — PROBABILIDADES:\n")
    print(f"{'Time':<28} {'Titulo':>7} {'Liberta':>8} {'Sul-Am':>7} {'Rebaixa':>8} {'Pts':>6} {'Pos':>5}")
    print("-" * 68)
    for _, r in df_sim.iterrows():
        print(f"{r['time']:<28} {r['titulo_pct']:>6.1f}% "
              f"{r['libertadores_pct']:>7.1f}% "
              f"{r['sulamericana_pct']:>6.1f}% "
              f"{r['rebaixamento_pct']:>7.1f}% "
              f"{r['pts_esperados']:>6.1f} "
              f"{r['pos_esperada']:>5.1f}")

    total = time.time() - t_start
    print(f"\nSalvo em {OUTPUT_PATH}")
    print(f"Tempo total: {total/60:.1f} min")


if __name__ == "__main__":
    main()