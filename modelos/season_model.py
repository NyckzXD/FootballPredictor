"""
season_model_v2.py — Simulação Monte Carlo OTIMIZADA
======================================================

GARGALOS DO v1 (por simulação, ~30 rodadas × 10 jogos):
  1. team_stats() escaneia TODA a lista de histórico para achar últimos N jogos
     → 4 chamadas/jogo × 300 jogos × 10K sims = 12M scans em lista crescente
  2. add_derived_dict() calcula Poisson com loop Python por jogo
  3. Dicts de strings (live_table, elo_ratings, history) → lookups lentos

OTIMIZAÇÕES v2:
  1. RING BUFFER por time (deque maxlen=10) → O(1) insert, O(10) stats
     Elimina 100% dos scans em lista de histórico
  2. NUMPY ARRAYS para estado (elo, pts, gd, gf, played) indexados por team_id
     → lookups O(1) por índice, sem hash de string
  3. H2H MATRIX (N×N×3) — lookup O(1) por (home_idx, away_idx)
  4. BATCH features para toda a rodada usando numpy (sem loop Python)
  5. POISSON DRAW PROB vectorizado com numpy (sem math.factorial loop)
  6. PRE-COMPUTE market values e odds como arrays indexados
  7. NUMBA JIT (opcional) para o loop de stats da ring buffer

Speedup esperado: 5-15x vs v1
"""

import pandas as pd
import numpy as np
import math
import joblib
from joblib import Parallel, delayed
from collections import deque
import warnings, time, os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Tentar importar numba (opcional)
try:
    from numba import njit
    HAS_NUMBA = True
    print("   ✅ Numba disponível — JIT ativado")
except ImportError:
    HAS_NUMBA = False
    # Fallback: decorador que não faz nada
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH         = r"C:\PREDICTOR\REPO\modelos\match_model_v2.pkl"
POISSON_MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\poisson_model.pkl"
MATCHES_PATH       = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
FIXTURES_PATH      = r"C:\PREDICTOR\REPO\scraping\data\raw\matches.csv"
MARKET_PATH        = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"
ODDS_PATH          = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
OUTPUT_PATH        = r"C:\PREDICTOR\REPO\scraping\data\processed\simulacao_2026.csv"

N_SIMS  = 10_000
N_JOBS  = 12        # -1 = usar todos os cores
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

# Factorial lookup (pré-computado)
FACT = np.array([math.factorial(g) for g in range(7)], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# RING BUFFER STATS (substitui team_stats + scan do histórico)
# ═══════════════════════════════════════════════════════════════════════════════
def stats_from_ring(ring, n):
    """Calcula stats dos últimos n jogos do ring buffer de um time.
    Ring é um deque de tuples: (is_home, gf, ga)
    """
    games = list(ring)[-n:]
    if not games:
        return 1.0, 1.2, 1.0, 0.2, 0.4, 0.25, 0.0, 0.0
    pts = gf = ga = wins = draws = 0
    home_f = away_f = 0.0
    hc = ac = 0
    for is_home, g_for, g_against in games:
        gf += g_for
        ga += g_against
        if g_for > g_against:
            pts += 3; wins += 1
        elif g_for == g_against:
            pts += 1; draws += 1
        if is_home:
            home_f += g_for - g_against; hc += 1
        else:
            away_f += g_for - g_against; ac += 1
    n_ = len(games)
    return (pts/n_, gf/n_, ga/n_, (gf-ga)/n_,
            wins/n_, draws/n_,
            home_f/hc if hc else 0.0,
            away_f/ac if ac else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# POISSON DRAW PROB — VETORIZADO COM NUMPY
# ═══════════════════════════════════════════════════════════════════════════════
def poisson_draw_prob_batch(lam_h, lam_a):
    """Calcula P(draw) via Poisson para arrays de lambdas."""
    pdraw = np.zeros(len(lam_h))
    for g in range(7):
        pdraw += (np.exp(-lam_h) * lam_h**g / FACT[g] *
                  np.exp(-lam_a) * lam_a**g / FACT[g])
    return pdraw


def poisson_draw_prob_scalar(lam_h, lam_a):
    """Versão escalar para dentro do loop de simulação."""
    pdraw = 0.0
    for g in range(7):
        pdraw += (math.exp(-lam_h) * lam_h**g / FACT[g] *
                  math.exp(-lam_a) * lam_a**g / FACT[g])
    return pdraw


# ═══════════════════════════════════════════════════════════════════════════════
# ELO (vetorizável)
# ═══════════════════════════════════════════════════════════════════════════════
@njit
def elo_expected(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

@njit
def elo_update(ra, rb, sa, k=32.0):
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    return ra + k * (sa - ea), rb + k * ((1.0 - sa) - (1.0 - ea))


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE FEATURES BATCH — para todos os jogos de uma rodada de uma vez
# ═══════════════════════════════════════════════════════════════════════════════
def compute_round_features(games_idx, elo, table_pts, table_gd, table_gf, table_played,
                           rings, h2h_matrix, mv_arr, mv_max,
                           odds_h, odds_d, odds_a, odds_mask, feat_order):
    """
    Calcula features para todos os jogos de uma rodada como numpy array.

    games_idx: array (n_games, 2) com [home_idx, away_idx]
    Retorna: X de shape (n_games, n_features)
    """
    n = len(games_idx)
    feat_dict_list = []

    for i in range(n):
        hi, ai = games_idx[i]

        # Stats dos ring buffers (O(10) max, não O(history_length))
        h5  = stats_from_ring(rings[hi], 5)
        a5  = stats_from_ring(rings[ai], 5)
        h10 = stats_from_ring(rings[hi], 10)
        a10 = stats_from_ring(rings[ai], 10)

        # Elo
        h_elo, a_elo = elo[hi], elo[ai]

        # Tabela
        h_played = max(table_played[hi], 1)
        a_played = max(table_played[ai], 1)
        h_aprov = table_pts[hi] / (h_played * 3)
        a_aprov = table_pts[ai] / (a_played * 3)
        h_pos = max(1, int((1 - h_aprov) * 20))
        a_pos = max(1, int((1 - a_aprov) * 20))

        # Market value
        h_mv, a_mv = mv_arr[hi], mv_arr[ai]

        # Odds (se disponíveis)
        key = hi * 1000 + ai  # unique key
        if odds_mask.get(key, False):
            prob_h_mkt = odds_h[key]
            prob_d_mkt = odds_d[key]
            prob_a_mkt = odds_a[key]
            oh = 1.0 / max(prob_h_mkt, 0.01)
            oa = 1.0 / max(prob_a_mkt, 0.01)
            od = 1.0 / max(prob_d_mkt, 0.01)
            odds_draw_factor = od / ((oh + oa) / 2)
            odds_har = oh / max(oa, 0.01)
            market_entropy = -(
                prob_h_mkt * np.log(prob_h_mkt + 1e-9) +
                prob_d_mkt * np.log(prob_d_mkt + 1e-9) +
                prob_a_mkt * np.log(prob_a_mkt + 1e-9))
        else:
            e = elo_expected(h_elo, a_elo)
            prob_h_mkt = e * 0.85 + 0.05
            prob_a_mkt = (1 - e) * 0.75 + 0.05
            prob_d_mkt = max(1 - prob_h_mkt - prob_a_mkt, 0.05)
            odds_draw_factor = 1.0
            odds_har = prob_h_mkt / max(prob_a_mkt, 0.01)
            market_entropy = 1.0

        # H2H
        h2h_hw = h2h_matrix[hi, ai, 0]
        h2h_aw = h2h_matrix[hi, ai, 1]
        h2h_d  = h2h_matrix[hi, ai, 2]

        # Build feature dict
        f = {
            "elo_diff": h_elo - a_elo, "home_elo": h_elo, "away_elo": a_elo,
            "home_market_value_log": np.log1p(h_mv),
            "away_market_value_log": np.log1p(a_mv),
            "market_value_diff": h_mv - a_mv,
            "home_market_value_norm": h_mv / mv_max,
            "away_market_value_norm": a_mv / mv_max,
            "home_squad_size": 20, "away_squad_size": 20,
            "home_aproveitamento": h_aprov, "away_aproveitamento": a_aprov,
            "position_diff": h_pos - a_pos,
            "home_form_pts": h5[0], "home_avg_gf": h5[1], "home_avg_ga": h5[2],
            "home_goal_diff": h5[3], "home_win_rate": h5[4], "home_draw_rate": h5[5],
            "home_home_form": h5[6],
            "away_form_pts": a5[0], "away_avg_gf": a5[1], "away_avg_ga": a5[2],
            "away_goal_diff": a5[3], "away_win_rate": a5[4], "away_draw_rate": a5[5],
            "away_away_form": a5[7],
            "home_form_pts_10": h10[0], "home_avg_gf_10": h10[1],
            "home_avg_ga_10": h10[2], "home_win_rate_10": h10[4],
            "away_form_pts_10": a10[0], "away_avg_gf_10": a10[1],
            "away_avg_ga_10": a10[2], "away_win_rate_10": a10[4],
            "h2h_home_wins": h2h_hw, "h2h_away_wins": h2h_aw, "h2h_draws": h2h_d,
            "prob_h_mkt": prob_h_mkt, "prob_d_mkt": prob_d_mkt, "prob_a_mkt": prob_a_mkt,
            "odds_draw_factor": odds_draw_factor,
            "odds_home_away_ratio": odds_har,
            "market_entropy": market_entropy,
        }

        # ── Derived features (inline, sem chamada de função) ──
        f["form_diff"]       = f["home_form_pts"]       - f["away_form_pts"]
        f["form_diff_10"]    = f["home_form_pts_10"]    - f["away_form_pts_10"]
        f["gf_diff"]         = f["home_avg_gf"]         - f["away_avg_gf"]
        f["ga_diff"]         = f["home_avg_ga"]         - f["away_avg_ga"]
        f["win_rate_diff"]   = f["home_win_rate"]       - f["away_win_rate"]
        f["aproveit_diff"]   = f["home_aproveitamento"] - f["away_aproveitamento"]
        f["home_in_crisis"]  = int(f["home_form_pts"] < 0.5)
        f["away_in_form"]    = int(f["away_form_pts"] > 2.0)
        f["elo_similarity"]      = 1.0 / (1.0 + abs(f["elo_diff"]))
        f["form_similarity"]     = 1.0 / (1.0 + abs(f["form_diff"]))
        f["value_similarity"]    = 1.0 / (1.0 + abs(f["market_value_diff"]))
        f["overall_balance"]     = (f["elo_similarity"] + f["form_similarity"] + f["value_similarity"]) / 3
        f["home_draw_tendency"]  = f["home_draw_rate"]
        f["away_draw_tendency"]  = f["away_draw_rate"]
        f["combined_draw_rate"]  = (f["home_draw_rate"] + f["away_draw_rate"]) / 2
        f["both_low_scoring"]    = int(f["home_avg_gf"] < 1.2 and f["away_avg_gf"] < 1.2)
        f["both_good_defense"]   = int(f["home_avg_ga"] < 1.0 and f["away_avg_ga"] < 1.0)
        total_h2h = h2h_hw + h2h_aw + h2h_d + 1
        f["h2h_draw_rate"]       = h2h_d / total_h2h
        f["h2h_decisividade"]    = (h2h_hw + h2h_aw) / total_h2h
        f["position_similarity"] = 1.0 / (1.0 + abs(f["position_diff"]))
        f["elo_vs_mkt_h"]        = f["elo_similarity"] - prob_h_mkt
        f["elo_vs_mkt_a"]        = (1 - f["elo_similarity"]) - prob_a_mkt

        # Poisson features
        eg_h = h5[1] * 0.6 + h10[1] * 0.4
        eg_a = a5[1] * 0.6 + a10[1] * 0.4
        ec_h = h5[2] * 0.6 + h10[2] * 0.4
        ec_a = a5[2] * 0.6 + a10[2] * 0.4
        f["expected_goals_h"]   = eg_h
        f["expected_goals_a"]   = eg_a
        f["expected_concede_h"] = ec_h
        f["expected_concede_a"] = ec_a
        lh = (eg_h + ec_a) / 2
        la = (eg_a + ec_h) / 2
        f["lambda_h"]     = lh
        f["lambda_a"]     = la
        f["lambda_diff"]  = lh - la
        f["lambda_total"] = lh + la
        f["poisson_draw_prob"] = poisson_draw_prob_scalar(lh, la)
        f["home_momentum"]      = h5[0] - h10[0]
        f["away_momentum"]      = a5[0] - a10[0]
        f["home_adv_vs_market"] = h5[6] - a5[7]
        f["model_vs_market_d"]  = f["combined_draw_rate"] - prob_d_mkt

        feat_dict_list.append(f)

    # Converter para X numpy de uma vez
    X = np.array([[f.get(k, 0) for k in feat_order] for f in feat_dict_list])
    return X, feat_dict_list


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO OTIMIZADA
# ═══════════════════════════════════════════════════════════════════════════════
def run_simulation(sim_id, fixtures_by_round_idx, completed_idx,
                   n_teams, team_names,
                   elo_init, rings_init, h2h_init,
                   mv_arr, mv_max,
                   odds_h_dict, odds_d_dict, odds_a_dict, odds_mask_dict,
                   model_data, poisson_data,
                   feat_order, poisson_feat_h, poisson_feat_a):
    """Simulação única usando ring buffers e arrays numpy."""

    # Copiar estado mutável
    elo      = elo_init.copy()
    pts      = np.zeros(n_teams, dtype=np.float64)
    gd       = np.zeros(n_teams, dtype=np.float64)
    gf_total = np.zeros(n_teams, dtype=np.float64)
    played   = np.zeros(n_teams, dtype=np.int32)
    h2h_mat  = h2h_init.copy()

    # Deep copy ring buffers (lista de deques)
    rings = [deque(r, maxlen=10) for r in rings_init]

    # ── Aplicar jogos já disputados ──
    for hi, ai, hg, ag in completed_idx:
        # Update table
        played[hi] += 1; played[ai] += 1
        gf_total[hi] += hg; gf_total[ai] += ag
        gd[hi] += hg - ag; gd[ai] += ag - hg
        if hg > ag:
            pts[hi] += 3
        elif ag > hg:
            pts[ai] += 3
        else:
            pts[hi] += 1; pts[ai] += 1

        # Update elo
        sa = 1.0 if hg > ag else (0.0 if ag > hg else 0.5)
        elo[hi], elo[ai] = elo_update(elo[hi], elo[ai], sa)

        # Update ring buffers
        rings[hi].append((1, hg, ag))   # is_home=1
        rings[ai].append((0, ag, hg))   # is_home=0

        # Update H2H
        if hg > ag:
            h2h_mat[hi, ai, 0] += 1
        elif ag > hg:
            h2h_mat[hi, ai, 1] += 1
        else:
            h2h_mat[hi, ai, 2] += 1

    # ── Simular rodadas futuras ──
    for round_games in fixtures_by_round_idx:
        if not round_games:
            continue

        games_arr = np.array(round_games, dtype=np.int32)
        n_games = len(games_arr)

        # Compute features para toda a rodada (BATCH)
        X, feat_dicts = compute_round_features(
            games_arr, elo, pts, gd, gf_total, played,
            rings, h2h_mat, mv_arr, mv_max,
            odds_h_dict, odds_d_dict, odds_a_dict, odds_mask_dict,
            feat_order)

        # ── LightGBM batch predict (ensemble 2 seeds) ──
        ph_arr = np.mean([m.predict_proba(X)[:, 1] for m in model_data["models_h"]], axis=0)
        pd_arr = np.mean([m.predict_proba(X)[:, 1] for m in model_data["models_d"]], axis=0)
        pa_arr = np.mean([m.predict_proba(X)[:, 1] for m in model_data["models_a"]], axis=0)

        ph_arr = model_data["cal_h"].predict(ph_arr)
        pd_arr = model_data["cal_d"].predict(pd_arr)
        pa_arr = model_data["cal_a"].predict(pa_arr)

        tot = ph_arr + pd_arr + pa_arr
        ph_arr /= tot; pd_arr /= tot; pa_arr /= tot

        # ── Poisson goals (batch predict + random) ──
        fh_batch = np.array([[fd.get(k, 0) for k in poisson_feat_h] for fd in feat_dicts])
        fa_batch = np.array([[fd.get(k, 0) for k in poisson_feat_a] for fd in feat_dicts])
        lam_h = np.clip(poisson_data["model_home"].predict(
            poisson_data["scaler_home"].transform(fh_batch)), 0.1, 8.0)
        lam_a = np.clip(poisson_data["model_away"].predict(
            poisson_data["scaler_away"].transform(fa_batch)), 0.1, 8.0)

        hg_arr = np.random.poisson(lam_h)
        ag_arr = np.random.poisson(lam_a)

        # ── Sortear resultados e aplicar (batch random) ──
        rand_vals = np.random.random(n_games)

        for i in range(n_games):
            hi, ai = games_arr[i]
            # Sortear resultado
            if rand_vals[i] < ph_arr[i]:
                res = 0  # H
            elif rand_vals[i] < ph_arr[i] + pd_arr[i]:
                res = 1  # D
            else:
                res = 2  # A

            hg, ag = int(hg_arr[i]), int(ag_arr[i])
            if res == 0 and hg <= ag:
                hg = ag + 1
            elif res == 2 and ag <= hg:
                ag = hg + 1
            elif res == 1:
                ag = hg

            # Update table
            played[hi] += 1; played[ai] += 1
            gf_total[hi] += hg; gf_total[ai] += ag
            gd[hi] += hg - ag; gd[ai] += ag - hg
            if hg > ag:
                pts[hi] += 3
            elif ag > hg:
                pts[ai] += 3
            else:
                pts[hi] += 1; pts[ai] += 1

            # Update elo
            sa = 1.0 if hg > ag else (0.0 if ag > hg else 0.5)
            elo[hi], elo[ai] = elo_update(elo[hi], elo[ai], sa)

            # Update ring
            rings[hi].append((1, hg, ag))
            rings[ai].append((0, ag, hg))

            # Update H2H
            if hg > ag:     h2h_mat[hi, ai, 0] += 1
            elif ag > hg:   h2h_mat[hi, ai, 1] += 1
            else:           h2h_mat[hi, ai, 2] += 1

    # ── Resultado: classificação final ──
    # Critérios: pts desc, gd desc, gf desc
    order = np.lexsort((gf_total, gd, pts))[::-1]  # sort descending
    positions = np.empty(n_teams, dtype=np.int32)
    for rank, team_idx in enumerate(order):
        positions[team_idx] = rank + 1

    return positions, pts.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 68)
    print("  SEASON MODEL v2 — SIMULAÇÃO OTIMIZADA")
    print("=" * 68)

    t_start = time.time()

    # ── Carregar dados ──
    print("\n📂 Carregando dados...")
    hist_df = pd.read_csv(MATCHES_PATH)
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    hist_df = hist_df.dropna(subset=["home_goals", "away_goals"])

    cal_df = pd.read_csv(FIXTURES_PATH)
    cal_df["date"] = pd.to_datetime(cal_df["date"], errors="coerce")
    cal_2026 = cal_df[cal_df["season"] == SEASON].copy()

    completed_df = cal_2026[cal_2026["home_goals"].notna()].copy()
    fixtures_df  = cal_2026[cal_2026["home_goals"].isna()].copy()
    print(f"   {SEASON}: {len(completed_df)} disputados | {len(fixtures_df)} a simular")

    teams = sorted(set(cal_2026["home_team"].tolist() + cal_2026["away_team"].tolist()))
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    print(f"   {n_teams} times")

    if len(fixtures_df) == 0:
        print("⚠️ Sem fixtures"); return

    # ── Converter para índices (elimina lookups de string) ──
    completed_idx = [
        (team_to_idx[r["home_team"]], team_to_idx[r["away_team"]],
         int(r["home_goals"]), int(r["away_goals"]))
        for _, r in completed_df.iterrows()
    ]

    if "matchday" in fixtures_df.columns:
        fixtures_by_round_idx = [
            [(team_to_idx[r["home_team"]], team_to_idx[r["away_team"]])
             for _, r in fixtures_df[fixtures_df["matchday"] == rd].iterrows()]
            for rd in sorted(fixtures_df["matchday"].unique())
        ]
    else:
        fixtures_by_round_idx = [
            [(team_to_idx[r["home_team"]], team_to_idx[r["away_team"]])
             for _, r in fixtures_df.iterrows()]
        ]

    total_fix = sum(len(r) for r in fixtures_by_round_idx)
    print(f"   {len(fixtures_by_round_idx)} rodadas | {total_fix} jogos")

    # ── Histórico → Ring buffers + Elo ──
    print("\n📊 Inicializando estado...")
    hist_records = hist_df[hist_df["season"] < SEASON].tail(5000)

    elo_init = np.full(n_teams, 1500.0)
    rings_init = [deque(maxlen=10) for _ in range(n_teams)]
    h2h_init = np.zeros((n_teams, n_teams, 3), dtype=np.int32)

    for _, r in hist_records.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in team_to_idx or a not in team_to_idx:
            continue
        hi, ai = team_to_idx[h], team_to_idx[a]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])

        sa = 1.0 if hg > ag else (0.0 if ag > hg else 0.5)
        elo_init[hi], elo_init[ai] = elo_update(elo_init[hi], elo_init[ai], sa)

        rings_init[hi].append((1, hg, ag))
        rings_init[ai].append((0, ag, hg))

        if hg > ag:     h2h_init[hi, ai, 0] += 1
        elif ag > hg:   h2h_init[hi, ai, 1] += 1
        else:           h2h_init[hi, ai, 2] += 1

    # ── Market values como array ──
    mv_arr = np.full(n_teams, 50.0)
    try:
        mv_df = pd.read_csv(MARKET_PATH)
        for _, r in mv_df.iterrows():
            if r["team"] in team_to_idx:
                mv_arr[team_to_idx[r["team"]]] = r["market_value"]
        print(f"   Market values: {(mv_arr != 50).sum()} times")
    except Exception:
        pass
    mv_max = max(mv_arr.max(), 1.0)

    # ── Odds como dicts indexados ──
    odds_h_dict = {}
    odds_d_dict = {}
    odds_a_dict = {}
    odds_mask_dict = {}
    try:
        odds_df = pd.read_csv(ODDS_PATH)
        odds_df = odds_df[odds_df["Season"] == SEASON]
        for _, row in odds_df.iterrows():
            h = TEAM_MAP_ODDS.get(row["Home"], row["Home"])
            a = TEAM_MAP_ODDS.get(row["Away"], row["Away"])
            if h not in team_to_idx or a not in team_to_idx:
                continue
            oh = row.get("AvgCH") or row.get("PSCH")
            od = row.get("AvgCD") or row.get("PSCD")
            oa = row.get("AvgCA") or row.get("PSCA")
            if pd.notna(oh) and oh > 0:
                tot = 1/oh + 1/od + 1/oa
                hi, ai = team_to_idx[h], team_to_idx[a]
                key = hi * 1000 + ai
                odds_h_dict[key] = (1/oh) / tot
                odds_d_dict[key] = (1/od) / tot
                odds_a_dict[key] = (1/oa) / tot
                odds_mask_dict[key] = True
        print(f"   Odds: {len(odds_mask_dict)} jogos")
    except Exception as e:
        print(f"   ⚠️ Odds: {e}")

    # ── Modelos ──
    print("\n📂 Carregando modelos v2...")
    model_data   = joblib.load(MODEL_PATH)
    poisson_data = joblib.load(POISSON_MODEL_PATH)
    feat_order    = model_data["features"]
    poisson_feat_h = poisson_data["features_home"]
    poisson_feat_a = poisson_data["features_away"]
    print(f"   ✅ Modelo v{model_data.get('version', '?')} | {len(feat_order)} features")

    # ── Benchmark ──
    print("\n⏱️  Benchmark (10 sims)...")
    t0 = time.time()
    for i in range(10):
        run_simulation(i, fixtures_by_round_idx, completed_idx,
                       n_teams, teams, elo_init, rings_init, h2h_init,
                       mv_arr, mv_max,
                       odds_h_dict, odds_d_dict, odds_a_dict, odds_mask_dict,
                       model_data, poisson_data,
                       feat_order, poisson_feat_h, poisson_feat_a)
    t_bench = (time.time() - t0) / 10
    print(f"   {t_bench:.3f}s/sim | ETA {N_SIMS:,} sims: {t_bench*N_SIMS/60:.1f} min")

    # ── Monte Carlo ──
    print(f"\n🎲 Rodando {N_SIMS:,} simulações ({N_JOBS} jobs)...")
    t0 = time.time()

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_simulation)(
            i, fixtures_by_round_idx, completed_idx,
            n_teams, teams, elo_init, rings_init, h2h_init,
            mv_arr, mv_max,
            odds_h_dict, odds_d_dict, odds_a_dict, odds_mask_dict,
            model_data, poisson_data,
            feat_order, poisson_feat_h, poisson_feat_a
        )
        for i in range(N_SIMS)
    )

    elapsed = time.time() - t0
    print(f"\n   ⏱️  {elapsed/60:.1f} min ({elapsed/N_SIMS:.3f}s/sim)")
    if t_bench > 0:
        print(f"   Speedup vs benchmark single: {t_bench * N_SIMS / elapsed:.1f}x")

    # ── Agregar resultados ──
    print("\n📊 Agregando resultados...")
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int32)
    pts_all = np.zeros((N_SIMS, n_teams))

    for sim_i, (positions, pts_sim) in enumerate(results):
        for t in range(n_teams):
            position_counts[t, positions[t] - 1] += 1
            pts_all[sim_i, t] = pts_sim[t]

    rows = []
    for t in range(n_teams):
        pc = position_counts[t] / N_SIMS
        rows.append({
            "time":             teams[t],
            "titulo_pct":       round(pc[0] * 100, 1),
            "libertadores_pct": round(pc[:6].sum() * 100, 1),
            "sulamericana_pct": round(pc[:12].sum() * 100, 1),
            "rebaixamento_pct": round(pc[-4:].sum() * 100, 1),
            "pts_esperados":    round(pts_all[:, t].mean(), 1),
            "pts_std":          round(pts_all[:, t].std(), 1),
            "pos_esperada":     round(np.mean([r[0][t] for r in results]), 1),
        })

    df_sim = pd.DataFrame(rows).sort_values("pts_esperados", ascending=False)
    df_sim.to_csv(OUTPUT_PATH, index=False)

    # ── Print ──
    print(f"\n{'='*68}")
    print(f"✅ {N_SIMS:,} simulações em {elapsed/60:.1f} min")
    print(f"\n🏆 BRASILEIRÃO {SEASON}:\n")
    print(f"{'Time':<28} {'Título':>7} {'Liberta':>8} {'Sul-Am':>7} {'Rebaixa':>8} {'Pts':>6} {'Pos':>5}")
    print("-" * 68)
    for _, r in df_sim.iterrows():
        print(f"{r['time']:<28} {r['titulo_pct']:>6.1f}% "
              f"{r['libertadores_pct']:>7.1f}% {r['sulamericana_pct']:>6.1f}% "
              f"{r['rebaixamento_pct']:>7.1f}% {r['pts_esperados']:>6.1f} {r['pos_esperada']:>5.1f}")

    total = time.time() - t_start
    print(f"\n✅ Salvo: {OUTPUT_PATH}")
    print(f"⏱️  Total: {total/60:.1f} min")


if __name__ == "__main__":
    main()