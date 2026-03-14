import pandas as pd
import numpy as np
from scipy.stats import poisson
import joblib
from joblib import Parallel, delayed
import warnings
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

N_SIMS  = 1_000
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
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, result, k=32):
    ea = expected_score(ra, rb)
    sa, sb = (1, 0) if result == "H" else ((0, 1) if result == "A" else (0.5, 0.5))
    return ra + k * (sa - ea), rb + k * (sb - ea)


# ── Features ──────────────────────────────────────────────────────────────────
def compute_features(home, away, live_table, elo_ratings, history,
                     mv_dict, odds_dict, all_teams):

    def team_stats(team, n=5):
        games = [g for g in history if g["home"] == team or g["away"] == team][-n:]
        if not games:
            return {"pts": 1.0, "gf": 1.2, "ga": 1.0, "gd": 0.2,
                    "wr": 0.4, "dr": 0.25, "home_f": 0.0, "away_f": 0.0}
        pts, gf, ga, wins, draws = 0, 0, 0, 0, 0
        home_f, away_f, hc, ac = 0, 0, 0, 0
        for g in games:
            if g["home"] == team:
                gf += g["hg"]; ga += g["ag"]
                if g["hg"] > g["ag"]: pts += 3; wins += 1
                elif g["hg"] == g["ag"]: pts += 1; draws += 1
                home_f += g["hg"] - g["ag"]; hc += 1
            else:
                gf += g["ag"]; ga += g["hg"]
                if g["ag"] > g["hg"]: pts += 3; wins += 1
                elif g["ag"] == g["hg"]: pts += 1; draws += 1
                away_f += g["ag"] - g["hg"]; ac += 1
        n_ = len(games)
        return {"pts": pts/n_, "gf": gf/n_, "ga": ga/n_, "gd": (gf-ga)/n_,
                "wr": wins/n_, "dr": draws/n_,
                "home_f": home_f/hc if hc else 0,
                "away_f": away_f/ac if ac else 0}

    hs   = team_stats(home, 5);  as_  = team_stats(away, 5)
    hs10 = team_stats(home, 10); as10 = team_stats(away, 10)

    h2h = [g for g in history
           if (g["home"] == home and g["away"] == away) or
              (g["home"] == away and g["away"] == home)][-10:]
    h2h_hw = sum(1 for g in h2h if g["home"] == home and g["hg"] > g["ag"])
    h2h_aw = sum(1 for g in h2h if g["away"] == away and g["ag"] > g["hg"])
    h2h_d  = sum(1 for g in h2h if g["hg"] == g["ag"])

    ht = live_table.get(home, {}); at = live_table.get(away, {})
    h_aprov = ht.get("pts", 0) / max(ht.get("played", 1) * 3, 1)
    a_aprov = at.get("pts", 0) / max(at.get("played", 1) * 3, 1)

    sorted_teams = sorted(live_table.keys(),
                          key=lambda t: (-live_table[t].get("pts", 0),
                                        -live_table[t].get("gd", 0)))
    h_pos = sorted_teams.index(home) + 1 if home in sorted_teams else 10
    a_pos = sorted_teams.index(away) + 1 if away in sorted_teams else 10

    h_elo = elo_ratings.get(home, 1500)
    a_elo = elo_ratings.get(away, 1500)

    all_vals = [mv_dict.get(t, 50) for t in all_teams]
    max_val  = max(all_vals) if all_vals else 1
    h_mv = mv_dict.get(home, 50); a_mv = mv_dict.get(away, 50)

    # Odds
    key = (home, away)
    if key in odds_dict:
        o = odds_dict[key]
        prob_h_mkt    = o["prob_h"]
        prob_d_mkt    = o["prob_d"]
        prob_a_mkt    = o["prob_a"]
        odds_draw_factor = o["odd_d"] / ((o["odd_h"] + o["odd_a"]) / 2)
        odds_har      = o["odd_h"] / max(o["odd_a"], 0.01)
        market_entropy = -(
            prob_h_mkt * np.log(prob_h_mkt + 1e-9) +
            prob_d_mkt * np.log(prob_d_mkt + 1e-9) +
            prob_a_mkt * np.log(prob_a_mkt + 1e-9)
        )
    else:
        e = expected_score(h_elo, a_elo)
        prob_h_mkt    = e * 0.85 + 0.05
        prob_a_mkt    = (1 - e) * 0.75 + 0.05
        prob_d_mkt    = max(1 - prob_h_mkt - prob_a_mkt, 0.05)
        odds_draw_factor = 1.0
        odds_har      = prob_h_mkt / max(prob_a_mkt, 0.01)
        market_entropy = 1.0

    return {
        "elo_diff": h_elo - a_elo, "home_elo": h_elo, "away_elo": a_elo,
        "home_market_value_log":  np.log1p(h_mv),
        "away_market_value_log":  np.log1p(a_mv),
        "market_value_diff":      h_mv - a_mv,
        "home_market_value_norm": h_mv / max_val,
        "away_market_value_norm": a_mv / max_val,
        "home_squad_size": 20, "away_squad_size": 20,
        "home_aproveitamento": h_aprov, "away_aproveitamento": a_aprov,
        "position_diff": h_pos - a_pos,
        "home_form_pts": hs["pts"],   "home_avg_gf": hs["gf"],
        "home_avg_ga":   hs["ga"],    "home_goal_diff": hs["gd"],
        "home_win_rate": hs["wr"],    "home_draw_rate": hs["dr"],
        "home_home_form": hs["home_f"],
        "away_form_pts": as_["pts"],  "away_avg_gf": as_["gf"],
        "away_avg_ga":   as_["ga"],   "away_goal_diff": as_["gd"],
        "away_win_rate": as_["wr"],   "away_draw_rate": as_["dr"],
        "away_away_form": as_["away_f"],
        "home_form_pts_10": hs10["pts"], "home_avg_gf_10": hs10["gf"],
        "home_avg_ga_10":   hs10["ga"],  "home_win_rate_10": hs10["wr"],
        "away_form_pts_10": as10["pts"], "away_avg_gf_10": as10["gf"],
        "away_avg_ga_10":   as10["ga"],  "away_win_rate_10": as10["wr"],
        "h2h_home_wins": h2h_hw, "h2h_away_wins": h2h_aw, "h2h_draws": h2h_d,
        "prob_h_mkt": prob_h_mkt, "prob_d_mkt": prob_d_mkt, "prob_a_mkt": prob_a_mkt,
        "odds_draw_factor": odds_draw_factor,
        "odds_home_away_ratio": odds_har,
        "market_entropy": market_entropy,
    }


def add_derived(f):
    f = f.copy()
    f["form_diff"]       = f["home_form_pts"]      - f["away_form_pts"]
    f["form_diff_10"]    = f["home_form_pts_10"]   - f["away_form_pts_10"]
    f["gf_diff"]         = f["home_avg_gf"]        - f["away_avg_gf"]
    f["ga_diff"]         = f["home_avg_ga"]        - f["away_avg_ga"]
    f["win_rate_diff"]   = f["home_win_rate"]      - f["away_win_rate"]
    f["aproveit_diff"]   = f["home_aproveitamento"]- f["away_aproveitamento"]
    f["home_in_crisis"]  = int(f["home_form_pts"] < 0.5)
    f["away_in_form"]    = int(f["away_form_pts"] > 2.0)
    f["elo_similarity"]      = 1 / (1 + abs(f["elo_diff"]))
    f["form_similarity"]     = 1 / (1 + abs(f["form_diff"]))
    f["value_similarity"]    = 1 / (1 + abs(f["market_value_diff"]))
    f["overall_balance"]     = (f["elo_similarity"] + f["form_similarity"] + f["value_similarity"]) / 3
    f["home_draw_tendency"]  = f["home_draw_rate"]
    f["away_draw_tendency"]  = f["away_draw_rate"]
    f["combined_draw_rate"]  = (f["home_draw_rate"] + f["away_draw_rate"]) / 2
    f["both_low_scoring"]    = int(f["home_avg_gf"] < 1.2 and f["away_avg_gf"] < 1.2)
    f["both_good_defense"]   = int(f["home_avg_ga"] < 1.0 and f["away_avg_ga"] < 1.0)
    total_h2h                = f["h2h_home_wins"] + f["h2h_away_wins"] + f["h2h_draws"] + 1
    f["h2h_draw_rate"]       = f["h2h_draws"] / total_h2h
    f["h2h_decisividade"]    = (f["h2h_home_wins"] + f["h2h_away_wins"]) / total_h2h
    f["position_similarity"] = 1 / (1 + abs(f["position_diff"]))
    f["elo_vs_mkt_h"]        = f["elo_similarity"] - f["prob_h_mkt"]
    f["elo_vs_mkt_a"]        = (1 - f["elo_similarity"]) - f["prob_a_mkt"]
    return f


# ── Uma simulação completa ────────────────────────────────────────────────────
def run_simulation(sim_id, fixtures, completed, teams, elo_init,
                   history_init, mv_dict, odds_dict, model_data, poisson_data):

    live_table  = {t: {"pts": 0, "gd": 0, "gf": 0, "played": 0} for t in teams}
    elo_ratings = elo_init.copy()
    history     = list(history_init)

    def apply_result(h, a, hg, ag):
        res = "H" if hg > ag else ("A" if ag > hg else "D")
        for t in [h, a]:
            if t not in live_table:
                live_table[t] = {"pts": 0, "gd": 0, "gf": 0, "played": 0}
        live_table[h]["played"] += 1; live_table[a]["played"] += 1
        live_table[h]["gf"] += hg;    live_table[h]["gd"] += hg - ag
        live_table[a]["gf"] += ag;    live_table[a]["gd"] += ag - hg
        if res == "H":   live_table[h]["pts"] += 3
        elif res == "A": live_table[a]["pts"] += 3
        else:            live_table[h]["pts"] += 1; live_table[a]["pts"] += 1
        elo_ratings[h], elo_ratings[a] = update_elo(
            elo_ratings.get(h, 1500), elo_ratings.get(a, 1500), res)
        history.append({"home": h, "away": a, "hg": hg, "ag": ag})

    # Jogos já disputados
    for _, row in completed.iterrows():
        apply_result(row["home_team"], row["away_team"],
                     int(row["home_goals"]), int(row["away_goals"]))

    # Simular jogos futuros
    for _, row in fixtures.iterrows():
        h, a = row["home_team"], row["away_team"]

        feat = compute_features(h, a, live_table, elo_ratings, history,
                                mv_dict, odds_dict, teams)
        feat = add_derived(feat)
        X    = np.array([[feat.get(f, 0) for f in model_data["features"]]])

        ph  = model_data["cal_h"].predict(model_data["model_h"].predict_proba(X)[:, 1])[0]
        pd_ = model_data["cal_d"].predict(model_data["model_d"].predict_proba(X)[:, 1])[0]
        pa  = model_data["cal_a"].predict(model_data["model_a"].predict_proba(X)[:, 1])[0]
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        res = np.random.choice(["H", "D", "A"], p=[ph, pd_, pa])

        # Gols via Poisson
        fh = [feat.get(f, 0) for f in poisson_data["features_home"]]
        fa = [feat.get(f, 0) for f in poisson_data["features_away"]]
        lam_h = min(max(poisson_data["model_home"].predict(
            poisson_data["scaler_home"].transform([fh]))[0], 0.1), 8.0)
        lam_a = min(max(poisson_data["model_away"].predict(
            poisson_data["scaler_away"].transform([fa]))[0], 0.1), 8.0)
        hg = np.random.poisson(lam_h)
        ag = np.random.poisson(lam_a)

        # Consistência resultado/placar
        if res == "H" and hg <= ag: hg = ag + 1
        elif res == "A" and ag <= hg: ag = hg + 1
        elif res == "D": ag = hg

        apply_result(h, a, hg, ag)

    standings = sorted(live_table.items(),
                       key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
    positions = {t: i + 1 for i, (t, _) in enumerate(standings)}
    pts_final = {t: v["pts"] for t, v in live_table.items()}
    return positions, pts_final


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("📂 Carregando dados...")

    # Histórico completo para features (matches_final.csv)
    hist_df = pd.read_csv(MATCHES_PATH)
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    hist_df = hist_df.dropna(subset=["home_goals", "away_goals"])

    # Calendário 2026 (matches.csv — tem jogos disputados + futuros)
    cal_df = pd.read_csv(FIXTURES_PATH)
    cal_df["date"] = pd.to_datetime(cal_df["date"], errors="coerce")
    cal_2026 = cal_df[cal_df["season"] == SEASON].copy()

    completed = cal_2026[cal_2026["home_goals"].notna() &
                         cal_2026["away_goals"].notna()].copy()
    fixtures  = cal_2026[cal_2026["home_goals"].isna()].copy()

    print(f"   {SEASON}: {len(completed)} jogos disputados | {len(fixtures)} a simular")

    # Times da temporada
    teams = sorted(set(cal_2026["home_team"].tolist() + cal_2026["away_team"].tolist()))
    print(f"   {len(teams)} times: {teams}")

    if len(fixtures) == 0:
        print("⚠️  Nenhuma fixture futura encontrada — verifique matches.csv")
        return

    # Histórico para features (sem 2026 para evitar leakage)
    hist_records = hist_df[hist_df["season"] < SEASON].tail(5000)
    history_init = [
        {"home": r["home_team"], "away": r["away_team"],
         "hg": int(r["home_goals"]), "ag": int(r["away_goals"])}
        for _, r in hist_records.iterrows()
    ]

    # Elo inicial a partir do histórico
    elo_ratings = {t: 1500 for t in teams}
    for rec in history_init:
        h, a = rec["home"], rec["away"]
        if h not in elo_ratings: elo_ratings[h] = 1500
        if a not in elo_ratings: elo_ratings[a] = 1500
        res = "H" if rec["hg"] > rec["ag"] else ("A" if rec["ag"] > rec["hg"] else "D")
        elo_ratings[h], elo_ratings[a] = update_elo(elo_ratings[h], elo_ratings[a], res)

    # Valor de mercado
    try:
        mv_df   = pd.read_csv(MARKET_PATH)
        mv_dict = dict(zip(mv_df["team"], mv_df["market_value_eur_m"]))
        print(f"   Valores de mercado: {len(mv_dict)} times")
    except Exception:
        mv_dict = {}
        print("   ⚠️  market_values.csv não encontrado")

    # Odds 2026
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
                odds_dict[(h, a)] = {
                    "odd_h": oh, "odd_d": od, "odd_a": oa,
                    "prob_h": (1/oh)/tot, "prob_d": (1/od)/tot, "prob_a": (1/oa)/tot,
                }
        print(f"   Odds {SEASON}: {len(odds_dict)} jogos carregados")
    except Exception as e:
        print(f"   ⚠️  Odds não carregadas: {e}")

    # Modelos
    print("\n📂 Carregando modelos...")
    model_data   = joblib.load(MODEL_PATH)
    poisson_data = joblib.load(POISSON_MODEL_PATH)
    print("   ✅ match_model + poisson_model carregados")

    # ── Monte Carlo ──
    print(f"\n🎲 Rodando {N_SIMS:,} simulações ({N_JOBS} CPUs)...")
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_simulation)(
            i, fixtures, completed, teams,
            elo_ratings, history_init, mv_dict, odds_dict,
            model_data, poisson_data
        )
        for i in range(N_SIMS)
    )

    # ── Agregar ──
    print("\n📊 Agregando resultados...")
    position_counts = {t: np.zeros(len(teams)) for t in teams}
    pts_list        = {t: [] for t in teams}

    for positions, pts in results:
        for t in teams:
            pos = positions.get(t, len(teams))
            position_counts[t][pos - 1] += 1
            pts_list[t].append(pts.get(t, 0))

    rows = []
    for t in teams:
        pc = position_counts[t] / N_SIMS
        rows.append({
            "time":             t,
            "titulo_pct":       round(pc[0] * 100, 1),
            "libertadores_pct": round(pc[:6].sum() * 100, 1),
            "sulamericana_pct": round(pc[:12].sum() * 100, 1),
            "rebaixamento_pct": round(pc[-4:].sum() * 100, 1),
            "pts_esperados":    round(np.mean(pts_list[t]), 1),
            "pts_std":          round(np.std(pts_list[t]), 1),
            "pos_esperada":     round(np.mean([r[0].get(t, len(teams)) for r in results]), 1),
        })

    df_sim = pd.DataFrame(rows).sort_values("pts_esperados", ascending=False)
    df_sim.to_csv(OUTPUT_PATH, index=False)

    # ── Exibir ──
    print(f"\n{'='*68}")
    print(f"✅ {N_SIMS:,} simulações concluídas!")
    print(f"\n🏆 BRASILEIRÃO {SEASON} — PROBABILIDADES:\n")
    print(f"{'Time':<28} {'Título':>7} {'Liberta':>8} {'Sul-Am':>7} {'Rebaixa':>8} {'Pts':>6} {'Pos':>5}")
    print("-" * 68)
    for _, r in df_sim.iterrows():
        print(f"{r['time']:<28} {r['titulo_pct']:>6.1f}% "
              f"{r['libertadores_pct']:>7.1f}% "
              f"{r['sulamericana_pct']:>6.1f}% "
              f"{r['rebaixamento_pct']:>7.1f}% "
              f"{r['pts_esperados']:>6.1f} "
              f"{r['pos_esperada']:>5.1f}")

    print(f"\n✅ Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
