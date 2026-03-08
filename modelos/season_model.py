import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

MODEL_PATH         = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"
POISSON_MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\poisson_model.pkl"
FEATURES_PATH      = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
MATCHES_PATH       = r"C:\PREDICTOR\REPO\scraping\data\raw\matches.csv"
MARKET_PATH        = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_team_stats(team: str, history: list) -> dict:
    games = [g for g in history if g["home"] == team or g["away"] == team][-10:]
    if not games:
        return {
            "form_pts": 1.0, "avg_gf": 1.2, "avg_ga": 1.2, "goal_diff": 0.0,
            "win_rate": 0.33, "draw_rate": 0.33, "home_form": 1.0, "away_form": 1.0,
            "form_pts_10": 1.0, "avg_gf_10": 1.2, "avg_ga_10": 1.2, "win_rate_10": 0.33,
        }

    pts, gf, ga = [], [], []
    home_pts, away_pts = [], []
    for g in games:
        is_home = g["home"] == team
        g_gf = g["hg"] if is_home else g["ag"]
        g_ga = g["ag"] if is_home else g["hg"]
        gf.append(g_gf); ga.append(g_ga)
        p = 3 if g_gf > g_ga else 1 if g_gf == g_ga else 0
        pts.append(p)
        if is_home: home_pts.append(p)
        else:       away_pts.append(p)

    n = len(pts)
    return {
        "form_pts":    np.mean(pts[-5:]),
        "avg_gf":      np.mean(gf[-5:]),
        "avg_ga":      np.mean(ga[-5:]),
        "goal_diff":   np.mean(gf[-5:]) - np.mean(ga[-5:]),
        "win_rate":    sum(1 for p in pts[-5:] if p == 3) / min(5, n),
        "draw_rate":   sum(1 for p in pts[-5:] if p == 1) / min(5, n),
        "home_form":   np.mean(home_pts[-5:]) if home_pts else 1.0,
        "away_form":   np.mean(away_pts[-5:]) if away_pts else 1.0,
        "form_pts_10": np.mean(pts),
        "avg_gf_10":   np.mean(gf),
        "avg_ga_10":   np.mean(ga),
        "win_rate_10": sum(1 for p in pts if p == 3) / n,
    }


def get_elo(elo_ratings: dict, team: str) -> float:
    return elo_ratings.get(team, 1500.0)


def update_elo(ratings: dict, home: str, away: str, hg: int, ag: int):
    ra, rb = ratings.get(home, 1500), ratings.get(away, 1500)
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    ratings[home] = ra + 32 * (score - ea)
    ratings[away] = rb + 32 * ((1 - score) - (1 - ea))


def simulate_goals(poisson_features: dict, poisson_saved: dict) -> tuple:
    model_home = poisson_saved["model_home"]
    model_away = poisson_saved["model_away"]
    scaler     = poisson_saved["scaler"]
    features   = poisson_saved["features"]

    X_home = pd.DataFrame([poisson_features])[features]
    X_away = X_home.copy()
    X_away["elo_diff"]      = -X_away["elo_diff"]
    X_away["position_diff"] = -X_away["position_diff"]

    lam_home = max(0.2, model_home.predict(scaler.transform(X_home))[0])
    lam_away = max(0.2, model_away.predict(scaler.transform(X_away))[0])

    return int(np.random.poisson(lam_home)), int(np.random.poisson(lam_away))


def update_table(table: dict, home: str, away: str, hg: int, ag: int):
    for t in [home, away]:
        if t not in table:
            table[t] = {"pts": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0, "played": 0}
    table[home]["gf"] += hg; table[home]["ga"] += ag; table[home]["played"] += 1
    table[away]["gf"] += ag; table[away]["ga"] += hg; table[away]["played"] += 1
    if hg > ag:
        table[home]["pts"] += 3; table[home]["w"] += 1; table[away]["l"] += 1
    elif hg == ag:
        table[home]["pts"] += 1; table[away]["pts"] += 1
        table[home]["d"] += 1;   table[away]["d"] += 1
    else:
        table[away]["pts"] += 3; table[away]["w"] += 1; table[home]["l"] += 1


def update_live_table(live_table: dict, home: str, away: str, hg: int, ag: int):
    for t in [home, away]:
        if t not in live_table:
            live_table[t] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "gd": 0}
    live_table[home]["gf"] += hg; live_table[home]["ga"] += ag
    live_table[away]["gf"] += ag; live_table[away]["ga"] += hg
    live_table[home]["gd"] = live_table[home]["gf"] - live_table[home]["ga"]
    live_table[away]["gd"] = live_table[away]["gf"] - live_table[away]["ga"]
    live_table[home]["played"] += 1; live_table[away]["played"] += 1
    if hg > ag:    live_table[home]["pts"] += 3
    elif hg == ag: live_table[home]["pts"] += 1; live_table[away]["pts"] += 1
    else:          live_table[away]["pts"] += 3


def table_to_df(table: dict) -> pd.DataFrame:
    df = pd.DataFrame(table).T.reset_index()
    df.columns = ["team", "pts", "gf", "ga", "w", "d", "l", "played"]
    df["gd"] = df["gf"] - df["ga"]
    return df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# SIMULAÇÃO ÚNICA
# ──────────────────────────────────────────────

def simulate_season(played: pd.DataFrame, fixtures: pd.DataFrame,
                    model, feature_cols: list, le,
                    poisson_saved: dict, mv_dict: dict) -> pd.DataFrame:
    table       = {}
    history     = []
    elo_ratings = {}
    live_table  = {}

    def get_mv(team):
        return mv_dict.get(team, {
            "market_value_log":  3.0,
            "market_value_norm": 0.3,
            "squad_size":        25,
        })

    def get_position(team):
        if not live_table:
            return {"position": 10, "aproveitamento": 0.33}
        sorted_t = sorted(
            live_table.items(),
            key=lambda x: (x[1]["pts"], x[1].get("gd", 0)),
            reverse=True
        )
        for pos, (t, stats) in enumerate(sorted_t, 1):
            if t == team:
                played_n = stats["played"] or 1
                return {
                    "position":       pos,
                    "aproveitamento": stats["pts"] / (played_n * 3),
                }
        return {"position": 10, "aproveitamento": 0.33}

    # ── Processar jogos já disputados ──
    for _, row in played.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        history.append({"home": home, "away": away, "hg": hg, "ag": ag})
        update_table(table, home, away, hg, ag)
        update_elo(elo_ratings, home, away, hg, ag)
        update_live_table(live_table, home, away, hg, ag)

    # ── Simular jogos futuros ──
    for matchday in sorted(fixtures["matchday"].unique()):
        round_games = fixtures[fixtures["matchday"] == matchday]

        for _, match in round_games.iterrows():
            home, away = match["home_team"], match["away_team"]
            hs  = get_team_stats(home, history)
            as_ = get_team_stats(away, history)

            home_elo = get_elo(elo_ratings, home)
            away_elo = get_elo(elo_ratings, away)

            h_table = get_position(home)
            a_table = get_position(away)
            h_mv    = get_mv(home)
            a_mv    = get_mv(away)

            row = {
                # Elo
                "elo_diff":           home_elo - away_elo,
                "home_elo":           home_elo,
                "away_elo":           away_elo,
                # Valor de mercado
                "home_market_value_log":  h_mv["market_value_log"],
                "home_market_value_norm": h_mv["market_value_norm"],
                "home_squad_size":        h_mv["squad_size"],
                "away_market_value_log":  a_mv["market_value_log"],
                "away_market_value_norm": a_mv["market_value_norm"],
                "away_squad_size":        a_mv["squad_size"],
                "market_value_diff":      h_mv["market_value_log"] - a_mv["market_value_log"],
                # Posição na tabela
                "home_position":       h_table["position"],
                "away_position":       a_table["position"],
                "home_aproveitamento": h_table["aproveitamento"],
                "away_aproveitamento": a_table["aproveitamento"],
                "home_table_gd":       0,
                "away_table_gd":       0,
                "position_diff":       a_table["position"] - h_table["position"],
                # Curto prazo
                "home_form_pts":    hs["form_pts"],
                "home_avg_gf":      hs["avg_gf"],
                "home_avg_ga":      hs["avg_ga"],
                "home_goal_diff":   hs["goal_diff"],
                "home_win_rate":    hs["win_rate"],
                "home_draw_rate":   hs["draw_rate"],
                "home_home_form":   hs["home_form"],
                "away_form_pts":    as_["form_pts"],
                "away_avg_gf":      as_["avg_gf"],
                "away_avg_ga":      as_["avg_ga"],
                "away_goal_diff":   as_["goal_diff"],
                "away_win_rate":    as_["win_rate"],
                "away_draw_rate":   as_["draw_rate"],
                "away_away_form":   as_["away_form"],
                # Longo prazo
                "home_form_pts_10": hs["form_pts_10"],
                "home_avg_gf_10":   hs["avg_gf_10"],
                "home_avg_ga_10":   hs["avg_ga_10"],
                "home_win_rate_10": hs["win_rate_10"],
                "away_form_pts_10": as_["form_pts_10"],
                "away_avg_gf_10":   as_["avg_gf_10"],
                "away_avg_ga_10":   as_["avg_ga_10"],
                "away_win_rate_10": as_["win_rate_10"],
                # H2H
                "h2h_home_wins": 0,
                "h2h_away_wins": 0,
                "h2h_draws":     0,
            }

            X         = pd.DataFrame([row])[feature_cols]
            probs_enc = model.predict_proba(X)[0]
            probs     = {le.classes_[i]: probs_enc[i] for i in range(len(le.classes_))}

            # Sortear resultado
            p = np.array([probs["H"], probs["D"], probs["A"]], dtype=float)
            p = p / p.sum()
            resultado = np.random.choice(["H", "D", "A"], p=p)

            # Features Poisson
            poisson_features = {
                "elo_diff":            home_elo - away_elo,
                "home_elo":            home_elo,
                "away_elo":            away_elo,
                "home_avg_gf":         hs["avg_gf"],
                "home_avg_ga":         hs["avg_ga"],
                "home_goal_diff":      hs["goal_diff"],
                "away_avg_gf":         as_["avg_gf"],
                "away_avg_ga":         as_["avg_ga"],
                "away_goal_diff":      as_["goal_diff"],
                "home_form_pts":       hs["form_pts"],
                "away_form_pts":       as_["form_pts"],
                "home_avg_gf_10":      hs["avg_gf_10"],
                "home_avg_ga_10":      hs["avg_ga_10"],
                "away_avg_gf_10":      as_["avg_gf_10"],
                "away_avg_ga_10":      as_["avg_ga_10"],
                "home_aproveitamento": h_table.get("aproveitamento", 0.33),
                "away_aproveitamento": a_table.get("aproveitamento", 0.33),
                "position_diff":       a_table.get("position", 10) - h_table.get("position", 10),
            }

            hg, ag = simulate_goals(poisson_features, poisson_saved)

            # Ajustar gols com resultado sorteado
            if resultado == "H" and hg <= ag:
                hg = ag + 1
            elif resultado == "D" and hg != ag:
                menor = min(hg, ag); hg = menor; ag = menor
            elif resultado == "A" and ag <= hg:
                ag = hg + 1

            history.append({"home": home, "away": away, "hg": hg, "ag": ag})
            update_table(table, home, away, hg, ag)
            update_elo(elo_ratings, home, away, hg, ag)
            update_live_table(live_table, home, away, hg, ag)

    return table_to_df(table)


# ──────────────────────────────────────────────
# MONTE CARLO PARALELIZADO
# ──────────────────────────────────────────────

def _single_simulation(i, played, fixtures, model, feature_cols, le,
                        poisson_saved, mv_dict):
    table = simulate_season(played, fixtures, model, feature_cols, le,
                            poisson_saved, mv_dict)
    return {
        "title": table.iloc[0]["team"],
        "top4":  list(table.iloc[:4]["team"]),
        "top6":  list(table.iloc[:6]["team"]),
        "rell":  list(table.iloc[-4:]["team"]),
        "pts":   {row["team"]: row["pts"] for _, row in table.iterrows()},
    }


def monte_carlo(played, fixtures, model, feature_cols, le, n=10000):
    poisson_saved = joblib.load(POISSON_MODEL_PATH)
    mv_dict = pd.read_csv(MARKET_PATH).set_index("team").to_dict("index")

    n_jobs = multiprocessing.cpu_count()
    print(f"🎲 Rodando {n} simulações em {n_jobs} threads paralelas...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_single_simulation)(i, played, fixtures, model, feature_cols,
                                    le, poisson_saved, mv_dict)
        for i in range(n)
    )

    teams = pd.concat([
        played["home_team"], played["away_team"],
        fixtures["home_team"], fixtures["away_team"]
    ]).unique()

    title   = {t: 0   for t in teams}
    top4    = {t: 0   for t in teams}
    top6    = {t: 0   for t in teams}
    rell    = {t: 0   for t in teams}
    pts_sum = {t: 0.0 for t in teams}

    for r in results:
        title[r["title"]] += 1
        for t in r["top4"]: top4[t] += 1
        for t in r["top6"]: top6[t] += 1
        for t in r["rell"]: rell[t] += 1
        for t, p in r["pts"].items(): pts_sum[t] += p

    result = pd.DataFrame({
        "time":           teams,
        "titulo_%":       [round(title[t]   / n * 100, 1) for t in teams],
        "libertadores_%": [round(top4[t]    / n * 100, 1) for t in teams],
        "sulamericana_%": [round(top6[t]    / n * 100, 1) for t in teams],
        "rebaixamento_%": [round(rell[t]    / n * 100, 1) for t in teams],
        "pts_esperados":  [round(pts_sum[t] / n, 1)       for t in teams],
    }).sort_values("titulo_%", ascending=False).reset_index(drop=True)

    result.index += 1
    return result


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    saved = joblib.load(MODEL_PATH)
    model, feature_cols, le = saved["model"], saved["features"], saved["label_encoder"]

    raw      = pd.read_csv(MATCHES_PATH)
    df_2026  = raw[raw["season"] == 2026].copy()
    played   = df_2026[df_2026["status"] == "FINISHED"].copy()
    fixtures = df_2026[df_2026["status"] != "FINISHED"].copy()

    print(f"✅ Jogos disputados: {len(played)}")
    print(f"📅 Jogos a simular:  {len(fixtures)}")
    print(f"🏟️  Times:            {df_2026['home_team'].nunique()}")

    # ── Tabela real atual ──
    print("\n" + "="*55)
    print("📊 TABELA REAL — BRASILEIRÃO 2026 (rodadas jogadas)")
    print("="*55)
    real_table = {}
    for _, row in played.iterrows():
        update_table(real_table, row["home_team"], row["away_team"],
                     int(row["home_goals"]), int(row["away_goals"]))
    rt = table_to_df(real_table)
    rt.index += 1
    print(rt[["team", "pts", "w", "d", "l", "gf", "ga", "gd"]].to_string())

    # ── Simulação única ──
    print("\n" + "="*55)
    print("🔮 SIMULAÇÃO ÚNICA — PROJEÇÃO FINAL 2026")
    print("="*55)
    poisson_saved = joblib.load(POISSON_MODEL_PATH)
    mv_dict = pd.read_csv(MARKET_PATH).set_index("team").to_dict("index")
    sim = simulate_season(played, fixtures, model, feature_cols, le,
                          poisson_saved, mv_dict)
    sim.index += 1
    print(sim[["team", "pts", "w", "d", "l", "gf", "ga", "gd"]].to_string())

    # ── Monte Carlo ──
    print("\n" + "="*55)
    print("🎲 MONTE CARLO — 10.000 SIMULAÇÕES")
    print("="*55)
    mc = monte_carlo(played, fixtures, model, feature_cols, le, n=10000)
    print(mc.to_string())

    # Salvar
    out = r"C:\PREDICTOR\scraping\data\processed\simulacao_2026.csv"
    mc.to_csv(out, index=False)
    print(f"\n✅ Resultado salvo em {out}")