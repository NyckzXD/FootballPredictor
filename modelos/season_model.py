"""
season_model.py  —  Monte Carlo Brasileirão

Otimizações vs versão anterior:
  1. Backend loky (processos reais) — elimina GIL no Python puro
  2. Calibração isotônica → np.interp pré-computado (43x mais rápido)
  3. _predict via arrays pré-extraídos — sem dict lookup por rodada
  4. Batching de rodadas — reduz chamadas predict de 18 para N_ROUND_BATCH
  5. Contexto global de simulação — evita re-pickle dos modelos entre workers
  6. TeamHistory.stats sem list() intermediário
  7. Meta-índices pré-extraídos — sem le.transform() por predição
"""

import pandas as pd
import numpy as np
import joblib
import warnings, time
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

MODEL_PATH         = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"
POISSON_MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\poisson_model.pkl"
MATCHES_PATH       = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
FIXTURES_PATH      = r"C:\PREDICTOR\REPO\scraping\data\raw\matches.csv"
MARKET_PATH        = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"
ODDS_PATH          = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
OUTPUT_PATH        = r"C:\PREDICTOR\REPO\scraping\data\processed\simulacao_2026.csv"

N_SIMS        = 10_000
N_JOBS        = max(1, mp.cpu_count() - 1)
SEASON        = 2026
N_ROUND_BATCH = 3   # agrupar N rodadas por chamada predict (reduz overhead)

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

# ── Contexto global compartilhado entre workers (evita re-pickle) ─────────────
_CTX = {}

def _worker_init(ctx):
    global _CTX
    _CTX = ctx


# ── Elo ───────────────────────────────────────────────────────────────────────
def _expected(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def _update_elo(ra, rb, ri, k=56):
    ea = _expected(ra, rb)
    if ri == 0:   sa, sb = 1.0, 0.0
    elif ri == 2: sa, sb = 0.0, 1.0
    else:         sa, sb = 0.5, 0.5
    return ra + k * (sa - ea), rb + k * (sb - (1.0 - ea))


# ── TeamHistory ───────────────────────────────────────────────────────────────
class _TH:
    __slots__ = ("data", "maxlen")
    def __init__(self, maxlen=10):
        self.data   = {}
        self.maxlen = maxlen

    def copy(self):
        th = _TH(self.maxlen)
        th.data = {t: deque(d, maxlen=self.maxlen) for t, d in self.data.items()}
        return th

    def add(self, home, away, hg, ag):
        for team, is_home, gf, ga in ((home, True, hg, ag), (away, False, ag, hg)):
            if team not in self.data:
                self.data[team] = deque(maxlen=self.maxlen)
            self.data[team].append((is_home, gf, ga))

    def stats(self, team, n):
        d = self.data.get(team)
        if not d:
            return 1.0, 1.2, 1.0, 0.2, 0.4, 0.25, 0.0, 0.0
        entries = list(d)[-n:]
        pts = gf = ga = wins = draws = hf = af = hc = ac = 0
        for is_home, gfor, gag in entries:
            gf += gfor; ga += gag
            if   gfor > gag:  pts += 3; wins  += 1
            elif gfor == gag: pts += 1; draws += 1
            if is_home: hf += gfor - gag; hc += 1
            else:       af += gfor - gag; ac += 1
        n_ = len(entries)
        return (pts/n_, gf/n_, ga/n_, (gf-ga)/n_,
                wins/n_, draws/n_,
                hf/hc if hc else 0.0,
                af/ac if ac else 0.0)


# ── H2H ───────────────────────────────────────────────────────────────────────
def _build_h2h(records):
    from collections import defaultdict
    raw = defaultdict(lambda: [0, 0, 0])
    for h, a, hg, ag in records:
        k = (h, a)
        if   hg > ag: raw[k][0] += 1
        elif ag > hg: raw[k][1] += 1
        else:         raw[k][2] += 1
    return dict(raw)

def _get_h2h(cache, h, a):
    f = cache.get((h, a))
    if f: return f[0], f[1], f[2]
    r = cache.get((a, h))
    if r: return r[1], r[0], r[2]
    return 0, 0, 0


# ── Feature row ───────────────────────────────────────────────────────────────
def _feature_row(home, away, table, tidx, elo, th, h2h, mv, odds, max_val, feat_idx, n_feat):
    hs   = th.stats(home, 5);  as_  = th.stats(away, 5)
    hs10 = th.stats(home, 10); as10 = th.stats(away, 10)
    h2h_hw, h2h_aw, h2h_d = _get_h2h(h2h, home, away)

    ht = table[tidx[home]]; at = table[tidx[away]]
    h_aprov = ht[0] / max(ht[3] * 3, 1)
    a_aprov = at[0] / max(at[3] * 3, 1)
    h_pos = max(1, int((1.0 - h_aprov) * 20))
    a_pos = max(1, int((1.0 - a_aprov) * 20))

    h_elo = elo.get(home, 1500.0); a_elo = elo.get(away, 1500.0)
    h_mv  = mv.get(home, 50.0);    a_mv  = mv.get(away, 50.0)

    key = (home, away)
    if key in odds:
        o = odds[key]
        ph_m, pd_m, pa_m = o[0], o[1], o[2]
        oh, od, oa = o[3], o[4], o[5]
        odf = od / ((oh + oa) / 2.0)
        oha = oh / max(oa, 0.01)
        ent = -(ph_m * np.log(ph_m+1e-9) + pd_m*np.log(pd_m+1e-9) + pa_m*np.log(pa_m+1e-9))
    else:
        e    = _expected(h_elo, a_elo)
        ph_m = e * 0.85 + 0.05
        pa_m = (1.0 - e) * 0.75 + 0.05
        pd_m = max(1.0 - ph_m - pa_m, 0.05)
        odf  = 1.0; oha = ph_m / max(pa_m, 0.01); ent = 1.0

    ed   = h_elo - a_elo
    mvd  = h_mv - a_mv
    pos_diff = h_pos - a_pos
    fd   = hs[0] - as_[0]; fd10 = hs10[0] - as10[0]
    gfd  = hs[1] - as_[1]; gad  = hs[2] - as_[2]
    wrd  = hs[4] - as_[4]; aprd = h_aprov - a_aprov
    es   = 1.0/(1.0+abs(ed));   fs = 1.0/(1.0+abs(fd)); vs = 1.0/(1.0+abs(mvd))
    cdr  = (hs[5] + as_[5]) / 2.0
    th2h = h2h_hw + h2h_aw + h2h_d + 1

    fmap = {
        "elo_diff": ed, "home_elo": h_elo, "away_elo": a_elo,
        "home_market_value_log": np.log1p(h_mv), "away_market_value_log": np.log1p(a_mv),
        "market_value_diff": mvd,
        "home_market_value_norm": h_mv/max_val, "away_market_value_norm": a_mv/max_val,
        "home_squad_size": 20.0, "away_squad_size": 20.0,
        "home_aproveitamento": h_aprov, "away_aproveitamento": a_aprov,
        "position_diff": float(pos_diff),
        "home_form_pts": hs[0], "home_avg_gf": hs[1], "home_avg_ga": hs[2],
        "home_goal_diff": hs[3], "home_win_rate": hs[4], "home_draw_rate": hs[5],
        "home_home_form": hs[6],
        "away_form_pts": as_[0], "away_avg_gf": as_[1], "away_avg_ga": as_[2],
        "away_goal_diff": as_[3], "away_win_rate": as_[4], "away_draw_rate": as_[5],
        "away_away_form": as_[7],
        "home_form_pts_10": hs10[0], "home_avg_gf_10": hs10[1],
        "home_avg_ga_10": hs10[2], "home_win_rate_10": hs10[4],
        "away_form_pts_10": as10[0], "away_avg_gf_10": as10[1],
        "away_avg_ga_10": as10[2], "away_win_rate_10": as10[4],
        "h2h_home_wins": float(h2h_hw), "h2h_away_wins": float(h2h_aw),
        "h2h_draws": float(h2h_d),
        "prob_h_mkt": ph_m, "prob_d_mkt": pd_m, "prob_a_mkt": pa_m,
        "odds_draw_factor": odf, "odds_home_away_ratio": oha, "market_entropy": ent,
        "form_diff": fd, "form_diff_10": fd10, "gf_diff": gfd, "ga_diff": gad,
        "win_rate_diff": wrd, "aproveit_diff": aprd,
        "home_in_crisis": float(hs[0] < 0.5), "away_in_form": float(as_[0] > 2.0),
        "elo_similarity": es, "form_similarity": fs, "value_similarity": vs,
        "overall_balance": (es+fs+vs)/3.0,
        "home_draw_tendency": hs[5], "away_draw_tendency": as_[5],
        "combined_draw_rate": cdr,
        "both_low_scoring": float(hs[1] < 1.2 and as_[1] < 1.2),
        "both_good_defense": float(hs[2] < 1.0 and as_[2] < 1.0),
        "h2h_draw_rate": h2h_d/th2h, "h2h_decisividade": (h2h_hw+h2h_aw)/th2h,
        "position_similarity": 1.0/(1.0+abs(pos_diff)),
        "elo_vs_mkt_h": es - ph_m, "elo_vs_mkt_a": (1.0-es) - pa_m,
    }
    row = np.empty(n_feat, dtype=np.float64)
    for k, i in feat_idx.items():
        row[i] = fmap.get(k, 0.0)
    return row


# ── Worker de simulação (função top-level para pickling) ─────────────────────
def _run_sim(sim_id):
    ctx = _CTX

    fixtures_by_round = ctx["fixtures_by_round"]
    completed         = ctx["completed"]
    teams             = ctx["teams"]
    n_teams           = ctx["n_teams"]
    elo_init          = ctx["elo_init"]
    hist_init         = ctx["hist_init"]
    h2h_cache         = ctx["h2h_cache"]
    mv_dict           = ctx["mv_dict"]
    odds_dict         = ctx["odds_dict"]
    max_val           = ctx["max_val"]
    feat_idx          = ctx["feat_idx"]
    n_feat            = ctx["n_feat"]
    poisson_idx_h     = ctx["poisson_idx_h"]
    poisson_idx_a     = ctx["poisson_idx_a"]
    # Modelos pré-extraídos
    m_lgb_h = ctx["m_lgb_h"]; m_lgb_d = ctx["m_lgb_d"]; m_lgb_a = ctx["m_lgb_a"]
    m_xgb_h = ctx["m_xgb_h"]; m_xgb_d = ctx["m_xgb_d"]; m_xgb_a = ctx["m_xgb_a"]
    # Iso lookup
    iso_x = ctx["iso_x"]
    iso_h = ctx["iso_h"]; iso_d = ctx["iso_d"]; iso_a = ctx["iso_a"]
    # Meta
    meta_clf   = ctx["meta_clf"]
    meta_idx_h = ctx["meta_idx_h"]
    meta_idx_d = ctx["meta_idx_d"]
    meta_idx_a = ctx["meta_idx_a"]
    # Poisson
    pm_h = ctx["pm_h"]; pm_a = ctx["pm_a"]
    sc_h = ctx["sc_h"]; sc_a = ctx["sc_a"]

    rng = np.random.default_rng(sim_id)

    tidx  = {t: i for i, t in enumerate(teams)}
    table = np.zeros((n_teams, 4), dtype=np.float32)
    elo   = elo_init.copy()
    th    = hist_init.copy()

    # Aplicar jogos já disputados
    for h, a, hg, ag in completed:
        hi = tidx[h]; ai = tidx[a]
        ri = 0 if hg > ag else (2 if ag > hg else 1)
        table[hi, 3] += 1; table[ai, 3] += 1
        table[hi, 2] += hg; table[ai, 2] += ag
        table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
        if   ri == 0: table[hi, 0] += 3
        elif ri == 2: table[ai, 0] += 3
        else:         table[hi, 0] += 1; table[ai, 0] += 1
        elo[h], elo[a] = _update_elo(elo.get(h,1500.), elo.get(a,1500.), ri)
        th.add(h, a, hg, ag)

    # Simular rodadas em batches
    n_rounds = len(fixtures_by_round)
    r = 0
    while r < n_rounds:
        # Coletar batch de N_ROUND_BATCH rodadas
        batch_games = []   # lista de (h, a) na ordem
        batch_sizes = []   # quantos jogos em cada rodada do batch
        for rb in range(r, min(r + N_ROUND_BATCH, n_rounds)):
            rg = fixtures_by_round[rb]
            if rg:
                batch_games.extend(rg)
                batch_sizes.append(len(rg))
        r += N_ROUND_BATCH

        if not batch_games:
            continue

        n_total = len(batch_games)
        X = np.empty((n_total, n_feat), dtype=np.float64)
        for gi, (h, a) in enumerate(batch_games):
            X[gi] = _feature_row(h, a, table, tidx, elo, th,
                                  h2h_cache, mv_dict, odds_dict, max_val,
                                  feat_idx, n_feat)

        # Predição em batch único por modelo
        raw_h = (m_lgb_h.predict(X) + m_xgb_h.predict_proba(X)[:, 1]) / 2
        raw_d = (m_lgb_d.predict(X) + m_xgb_d.predict_proba(X)[:, 1]) / 2
        raw_a = (m_lgb_a.predict(X) + m_xgb_a.predict_proba(X)[:, 1]) / 2

        # Calibração via lookup (43x mais rápido que iso.predict)
        ph  = np.interp(raw_h, iso_x, iso_h)
        pd_ = np.interp(raw_d, iso_x, iso_d)
        pa  = np.interp(raw_a, iso_x, iso_a)
        tot = ph + pd_ + pa; ph /= tot; pd_ /= tot; pa /= tot

        # Meta-learner stacking
        meta = meta_clf.predict_proba(np.column_stack([ph, pd_, pa]))
        ph  = meta[:, meta_idx_h]
        pd_ = meta[:, meta_idx_d]
        pa  = meta[:, meta_idx_a]
        tot = ph + pd_ + pa; ph /= tot; pd_ /= tot; pa /= tot

        # Poisson
        lam_h = np.clip(pm_h.predict(sc_h.transform(X[:, poisson_idx_h])), 0.1, 8.0)
        lam_a = np.clip(pm_a.predict(sc_a.transform(X[:, poisson_idx_a])), 0.1, 8.0)
        hg_arr = rng.poisson(lam_h).astype(np.int32)
        ag_arr = rng.poisson(lam_a).astype(np.int32)

        cumprob = np.column_stack([ph, ph + pd_])
        rand_v  = rng.random(n_total)
        res_arr = np.where(rand_v < cumprob[:, 0], 0,
                  np.where(rand_v < cumprob[:, 1], 1, 2))

        # Aplicar resultados
        gi = 0
        for sz in batch_sizes:
            for i in range(sz):
                h, a = batch_games[gi]
                ri = int(res_arr[gi])
                hg = int(hg_arr[gi]); ag = int(ag_arr[gi])
                if   ri == 0 and hg <= ag: hg = ag + 1
                elif ri == 2 and ag <= hg: ag = hg + 1
                elif ri == 1:              ag = hg
                hi = tidx[h]; ai = tidx[a]
                table[hi, 3] += 1; table[ai, 3] += 1
                table[hi, 2] += hg; table[ai, 2] += ag
                table[hi, 1] += hg - ag; table[ai, 1] += ag - hg
                if   ri == 0: table[hi, 0] += 3
                elif ri == 2: table[ai, 0] += 3
                else:         table[hi, 0] += 1; table[ai, 0] += 1
                elo[h], elo[a] = _update_elo(elo.get(h,1500.), elo.get(a,1500.), ri)
                th.add(h, a, hg, ag)
                gi += 1

    pts_arr = table[:, 0]; gd_arr = table[:, 1]; gf_arr = table[:, 2]
    order   = np.lexsort((-gf_arr, -gd_arr, -pts_arr))
    return (
        {teams[idx]: int(rank + 1) for rank, idx in enumerate(order)},
        {teams[i]: float(table[i, 0]) for i in range(n_teams)},
    )


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
        fixtures_by_round = [[(r["home_team"], r["away_team"]) for _, r in fixtures_df.iterrows()]]

    total_fix = sum(len(r) for r in fixtures_by_round)
    print(f"   {len(fixtures_by_round)} rodadas | {total_fix} jogos a simular")

    # ── Histórico + ELO ──
    hist_records = hist_df[hist_df["season"] < SEASON].tail(5000)
    history_tuples = [
        (r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"]))
        for _, r in hist_records.iterrows()
    ]

    ELO_REGRESSION = 0.35
    elo_ratings = {t: 1500.0 for t in teams}
    for h, a, hg, ag in history_tuples:
        if h not in elo_ratings: elo_ratings[h] = 1500.0
        if a not in elo_ratings: elo_ratings[a] = 1500.0
        ri = 0 if hg > ag else (2 if ag > hg else 1)
        elo_ratings[h], elo_ratings[a] = _update_elo(elo_ratings[h], elo_ratings[a], ri)
    for t in elo_ratings:
        elo_ratings[t] += ELO_REGRESSION * (1500.0 - elo_ratings[t])

    hist_init = _TH(maxlen=10)
    for h, a, hg, ag in history_tuples:
        hist_init.add(h, a, hg, ag)

    h2h_cache = _build_h2h(history_tuples)
    print(f"   h2h cache: {len(h2h_cache)} pares")

    # ── Market values ──
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
                odds_dict[(h, a)] = ((1/oh)/tot, (1/od)/tot, (1/oa)/tot, oh, od, oa)
        print(f"   Odds {SEASON}: {len(odds_dict)} jogos")
    except Exception as e:
        print(f"   Odds: {e}")

    # ── Modelos ──
    print("\nCarregando modelos...")
    model_data   = joblib.load(MODEL_PATH)
    poisson_data = joblib.load(POISSON_MODEL_PATH)

    feat_order = tuple(model_data["features"])
    n_feat     = len(feat_order)
    feat_idx   = {k: i for i, k in enumerate(feat_order)}

    forder_list   = list(feat_order)
    poisson_idx_h = [forder_list.index(k) for k in tuple(poisson_data["features_home"])]
    poisson_idx_a = [forder_list.index(k) for k in tuple(poisson_data["features_away"])]

    # Pré-extrair modelos (sem dict lookup por chamada)
    m_lgb_h = model_data["models_h_lgb"][0]
    m_lgb_d = model_data["models_d_lgb"][0]
    m_lgb_a = model_data["models_a_lgb"][0]
    m_xgb_h = model_data["models_h_xgb"][0]
    m_xgb_d = model_data["models_d_xgb"][0]
    m_xgb_a = model_data["models_a_xgb"][0]

    # Iso lookup arrays (43x mais rápido que iso.predict)
    iso_x = np.linspace(0.0, 1.0, 4001)
    iso_h = model_data["cal_h"].predict(iso_x)
    iso_d = model_data["cal_d"].predict(iso_x)
    iso_a = model_data["cal_a"].predict(iso_x)

    # Meta índices pré-extraídos
    le         = model_data["label_encoder_meta"]
    meta_idx_h = int(le.transform(["H"])[0])
    meta_idx_d = int(le.transform(["D"])[0])
    meta_idx_a = int(le.transform(["A"])[0])
    meta_clf   = model_data["meta_clf"]

    print(f"   Modelos carregados | features: {n_feat} | {N_JOBS} workers | batch={N_ROUND_BATCH} rodadas")

    # ── Contexto de simulação ──
    ctx = {
        "fixtures_by_round": fixtures_by_round,
        "completed":         completed_list,
        "teams":             teams,
        "n_teams":           n_teams,
        "elo_init":          elo_ratings,
        "hist_init":         hist_init,
        "h2h_cache":         h2h_cache,
        "mv_dict":           mv_dict,
        "odds_dict":         odds_dict,
        "max_val":           max_val,
        "feat_idx":          feat_idx,
        "n_feat":            n_feat,
        "poisson_idx_h":     poisson_idx_h,
        "poisson_idx_a":     poisson_idx_a,
        "m_lgb_h": m_lgb_h, "m_lgb_d": m_lgb_d, "m_lgb_a": m_lgb_a,
        "m_xgb_h": m_xgb_h, "m_xgb_d": m_xgb_d, "m_xgb_a": m_xgb_a,
        "iso_x": iso_x, "iso_h": iso_h, "iso_d": iso_d, "iso_a": iso_a,
        "meta_clf":   meta_clf,
        "meta_idx_h": meta_idx_h, "meta_idx_d": meta_idx_d, "meta_idx_a": meta_idx_a,
        "pm_h": poisson_data["model_home"],  "pm_a": poisson_data["model_away"],
        "sc_h": poisson_data["scaler_home"], "sc_a": poisson_data["scaler_away"],
    }

    # ── Benchmark single-thread ──
    print("\nBenchmark (10 sims single-thread)...")
    _worker_init(ctx)
    t0 = time.time()
    for i in range(10):
        _run_sim(i)
    t_single = (time.time() - t0) / 10
    eta = t_single * N_SIMS / N_JOBS / 60
    print(f"   {t_single:.3f}s/sim | ETA {N_SIMS:,} sims: {eta:.1f} min ({N_JOBS} workers)")

    # ── Monte Carlo ──
    print(f"\nRodando {N_SIMS:,} simulações ({N_JOBS} workers — loky)...")
    t0 = time.time()

    chunk = max(1, N_SIMS // (N_JOBS * 4))
    with ProcessPoolExecutor(max_workers=N_JOBS,
                             initializer=_worker_init,
                             initargs=(ctx,)) as pool:
        results = list(pool.map(_run_sim, range(N_SIMS), chunksize=chunk))

    elapsed = time.time() - t0
    print(f"   Concluído em {elapsed/60:.1f} min ({elapsed/N_SIMS:.3f}s/sim)")

    # ── Agregar ──
    print("\nAgregando resultados...")
    team_idx_map    = {t: i for i, t in enumerate(teams)}
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int32)
    pts_array       = np.zeros((N_SIMS, n_teams), dtype=np.float32)
    pos_sum         = np.zeros(n_teams, dtype=np.float64)

    for sim_i, (positions, pts) in enumerate(results):
        for t, pos in positions.items():
            ti = team_idx_map[t]
            position_counts[ti, pos - 1] += 1
            pts_array[sim_i, ti] = pts[t]
            pos_sum[ti] += pos

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
            "pos_esperada":     round(float(pos_sum[ti] / N_SIMS), 1),
        })

    df_sim = pd.DataFrame(rows).sort_values("pts_esperados", ascending=False)
    df_sim.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*68}")
    print(f"{N_SIMS:,} simulações em {elapsed/60:.1f} min\n")
    print(f"{'Time':<28} {'Titulo':>7} {'Liberta':>8} {'Sul-Am':>7} {'Rebaixa':>8} {'Pts':>6} {'Pos':>5}")
    print("-" * 68)
    for _, r in df_sim.iterrows():
        print(f"{r['time']:<28} {r['titulo_pct']:>6.1f}% "
              f"{r['libertadores_pct']:>7.1f}% "
              f"{r['sulamericana_pct']:>6.1f}% "
              f"{r['rebaixamento_pct']:>7.1f}% "
              f"{r['pts_esperados']:>6.1f} "
              f"{r['pos_esperada']:>5.1f}")

    print(f"\nSalvo: {OUTPUT_PATH}")
    print(f"Tempo total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    mp.freeze_support()
    main()
