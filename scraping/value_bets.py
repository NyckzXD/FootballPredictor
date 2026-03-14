import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from odds_api import fetch_odds

FEATURES_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MATCHES_PATH   = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
MODEL_PATH     = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"
MARKET_PATH    = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"
OUTPUT_PATH    = r"C:\PREDICTOR\REPO\scraping\data\external\value_bets.csv"

MIN_VALUE      = 1.05   # prob_modelo * odd >= 1.05
MIN_PROB       = 0.40   # confiança mínima do modelo
KELLY_FRACTION = 0.25   # Kelly fracionado conservador

FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "home_market_value_log", "away_market_value_log", "market_value_diff",
    "home_market_value_norm", "away_market_value_norm",
    "home_squad_size", "away_squad_size",
    "home_aproveitamento", "away_aproveitamento", "position_diff",
    "home_form_pts", "home_avg_gf", "home_avg_ga", "home_goal_diff",
    "home_win_rate", "home_draw_rate", "home_home_form",
    "away_form_pts", "away_avg_gf", "away_avg_ga", "away_goal_diff",
    "away_win_rate", "away_draw_rate", "away_away_form",
    "home_form_pts_10", "home_avg_gf_10", "home_avg_ga_10", "home_win_rate_10",
    "away_form_pts_10", "away_avg_gf_10", "away_avg_ga_10", "away_win_rate_10",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
    "odds_draw_factor", "odds_home_away_ratio", "market_entropy",
]


def add_derived(X_):
    X_ = X_.copy()
    X_["form_diff"]       = X_["home_form_pts"]       - X_["away_form_pts"]
    X_["form_diff_10"]    = X_["home_form_pts_10"]    - X_["away_form_pts_10"]
    X_["gf_diff"]         = X_["home_avg_gf"]         - X_["away_avg_gf"]
    X_["ga_diff"]         = X_["home_avg_ga"]         - X_["away_avg_ga"]
    X_["win_rate_diff"]   = X_["home_win_rate"]       - X_["away_win_rate"]
    X_["aproveit_diff"]   = X_["home_aproveitamento"] - X_["away_aproveitamento"]
    X_["home_in_crisis"]  = (X_["home_form_pts"] < 0.5).astype(int)
    X_["away_in_form"]    = (X_["away_form_pts"] > 2.0).astype(int)
    X_["elo_similarity"]      = 1 / (1 + np.abs(X_["elo_diff"]))
    X_["form_similarity"]     = 1 / (1 + np.abs(X_["form_diff"]))
    X_["value_similarity"]    = 1 / (1 + np.abs(X_["market_value_diff"]))
    X_["overall_balance"]     = (X_["elo_similarity"] + X_["form_similarity"] + X_["value_similarity"]) / 3
    X_["home_draw_tendency"]  = X_["home_draw_rate"]
    X_["away_draw_tendency"]  = X_["away_draw_rate"]
    X_["combined_draw_rate"]  = (X_["home_draw_rate"] + X_["away_draw_rate"]) / 2
    X_["both_low_scoring"]    = ((X_["home_avg_gf"] < 1.2) & (X_["away_avg_gf"] < 1.2)).astype(int)
    X_["both_good_defense"]   = ((X_["home_avg_ga"] < 1.0) & (X_["away_avg_ga"] < 1.0)).astype(int)
    total_h2h                 = X_["h2h_home_wins"] + X_["h2h_away_wins"] + X_["h2h_draws"] + 1
    X_["h2h_draw_rate"]       = X_["h2h_draws"] / total_h2h
    X_["h2h_decisividade"]    = (X_["h2h_home_wins"] + X_["h2h_away_wins"]) / total_h2h
    X_["position_similarity"] = 1 / (1 + np.abs(X_["position_diff"]))
    X_["elo_vs_mkt_h"]        = X_["elo_similarity"] - X_["prob_h_mkt"]
    X_["elo_vs_mkt_a"]        = (1 - X_["elo_similarity"]) - X_["prob_a_mkt"]
    return X_


def get_team_features(team, features_df, matches_df):
    """Extrai últimas features conhecidas de um time."""
    home_games = features_df[features_df["home_team"] == team].tail(1)
    away_games = features_df[features_df["away_team"] == team].tail(1)

    if len(home_games) > 0:
        row = home_games.iloc[0]
        return {
            "form_pts":    row.get("home_form_pts", 1.0),
            "avg_gf":      row.get("home_avg_gf", 1.2),
            "avg_ga":      row.get("home_avg_ga", 1.0),
            "goal_diff":   row.get("home_goal_diff", 0.0),
            "win_rate":    row.get("home_win_rate", 0.4),
            "draw_rate":   row.get("home_draw_rate", 0.25),
            "home_form":   row.get("home_home_form", 0.0),
            "away_form":   row.get("away_away_form", 0.0),
            "form_pts_10": row.get("home_form_pts_10", 1.0),
            "avg_gf_10":   row.get("home_avg_gf_10", 1.2),
            "avg_ga_10":   row.get("home_avg_ga_10", 1.0),
            "win_rate_10": row.get("home_win_rate_10", 0.4),
            "elo":         row.get("home_elo", 1500),
            "mv_log":      row.get("home_market_value_log", 4.0),
            "mv_norm":     row.get("home_market_value_norm", 0.5),
            "aproveit":    row.get("home_aproveitamento", 0.4),
        }
    elif len(away_games) > 0:
        row = away_games.iloc[0]
        return {
            "form_pts":    row.get("away_form_pts", 1.0),
            "avg_gf":      row.get("away_avg_gf", 1.2),
            "avg_ga":      row.get("away_avg_ga", 1.0),
            "goal_diff":   row.get("away_goal_diff", 0.0),
            "win_rate":    row.get("away_win_rate", 0.4),
            "draw_rate":   row.get("away_draw_rate", 0.25),
            "home_form":   row.get("home_home_form", 0.0),
            "away_form":   row.get("away_away_form", 0.0),
            "form_pts_10": row.get("away_form_pts_10", 1.0),
            "avg_gf_10":   row.get("away_avg_gf_10", 1.2),
            "avg_ga_10":   row.get("away_avg_ga_10", 1.0),
            "win_rate_10": row.get("away_win_rate_10", 0.4),
            "elo":         row.get("away_elo", 1500),
            "mv_log":      row.get("away_market_value_log", 4.0),
            "mv_norm":     row.get("away_market_value_norm", 0.5),
            "aproveit":    row.get("away_aproveitamento", 0.4),
        }
    # Fallback com defaults
    return {
        "form_pts": 1.0, "avg_gf": 1.2, "avg_ga": 1.0, "goal_diff": 0.0,
        "win_rate": 0.4, "draw_rate": 0.25, "home_form": 0.0, "away_form": 0.0,
        "form_pts_10": 1.0, "avg_gf_10": 1.2, "avg_ga_10": 1.0, "win_rate_10": 0.4,
        "elo": 1500, "mv_log": 4.0, "mv_norm": 0.5, "aproveit": 0.4,
    }


def get_h2h(home, away, matches_df):
    h2h = matches_df[
        ((matches_df["home_team"] == home) & (matches_df["away_team"] == away)) |
        ((matches_df["home_team"] == away) & (matches_df["away_team"] == home))
    ].tail(10)
    hw = sum(1 for _, r in h2h.iterrows()
             if r["home_team"] == home and r["home_goals"] > r["away_goals"])
    aw = sum(1 for _, r in h2h.iterrows()
             if r["away_team"] == away and r["away_goals"] > r["home_goals"])
    d  = sum(1 for _, r in h2h.iterrows() if r["home_goals"] == r["away_goals"])
    return hw, aw, d


def get_position(team, features_df):
    sub = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].tail(1)
    if len(sub) == 0:
        return 10
    row = sub.iloc[0]
    if row["home_team"] == team:
        aprov = row.get("home_aproveitamento", 0.4)
    else:
        aprov = row.get("away_aproveitamento", 0.4)
    return max(1, int((1 - aprov) * 20))


def build_feature_row(home, away, odd_h, odd_d, odd_a,
                      features_df, matches_df, mv_dict, all_mv):
    h = get_team_features(home, features_df, matches_df)
    a = get_team_features(away, features_df, matches_df)
    hw, aw, d = get_h2h(home, away, matches_df)

    h_pos = get_position(home, features_df)
    a_pos = get_position(away, features_df)

    max_val = max(all_mv) if all_mv else 1
    h_mv = mv_dict.get(home, 50)
    a_mv = mv_dict.get(away, 50)

    p_h = 1/odd_h; p_d = 1/odd_d; p_a = 1/odd_a
    tot = p_h + p_d + p_a
    prob_h_mkt = p_h / tot
    prob_d_mkt = p_d / tot
    prob_a_mkt = p_a / tot
    odds_draw_factor = odd_d / ((odd_h + odd_a) / 2)
    odds_har = odd_h / odd_a
    market_entropy = -(
        prob_h_mkt * np.log(prob_h_mkt + 1e-9) +
        prob_d_mkt * np.log(prob_d_mkt + 1e-9) +
        prob_a_mkt * np.log(prob_a_mkt + 1e-9)
    )

    return {
        "elo_diff":               h["elo"] - a["elo"],
        "home_elo":               h["elo"],
        "away_elo":               a["elo"],
        "home_market_value_log":  np.log1p(h_mv),
        "away_market_value_log":  np.log1p(a_mv),
        "market_value_diff":      h_mv - a_mv,
        "home_market_value_norm": h_mv / max_val,
        "away_market_value_norm": a_mv / max_val,
        "home_squad_size":        20,
        "away_squad_size":        20,
        "home_aproveitamento":    h["aproveit"],
        "away_aproveitamento":    a["aproveit"],
        "position_diff":          h_pos - a_pos,
        "home_form_pts":          h["form_pts"],
        "home_avg_gf":            h["avg_gf"],
        "home_avg_ga":            h["avg_ga"],
        "home_goal_diff":         h["goal_diff"],
        "home_win_rate":          h["win_rate"],
        "home_draw_rate":         h["draw_rate"],
        "home_home_form":         h["home_form"],
        "away_form_pts":          a["form_pts"],
        "away_avg_gf":            a["avg_gf"],
        "away_avg_ga":            a["avg_ga"],
        "away_goal_diff":         a["goal_diff"],
        "away_win_rate":          a["win_rate"],
        "away_draw_rate":         a["draw_rate"],
        "away_away_form":         a["away_form"],
        "home_form_pts_10":       h["form_pts_10"],
        "home_avg_gf_10":         h["avg_gf_10"],
        "home_avg_ga_10":         h["avg_ga_10"],
        "home_win_rate_10":       h["win_rate_10"],
        "away_form_pts_10":       a["form_pts_10"],
        "away_avg_gf_10":         a["avg_gf_10"],
        "away_avg_ga_10":         a["avg_ga_10"],
        "away_win_rate_10":       a["win_rate_10"],
        "h2h_home_wins":          hw,
        "h2h_away_wins":          aw,
        "h2h_draws":              d,
        "prob_h_mkt":             prob_h_mkt,
        "prob_d_mkt":             prob_d_mkt,
        "prob_a_mkt":             prob_a_mkt,
        "odds_draw_factor":       odds_draw_factor,
        "odds_home_away_ratio":   odds_har,
        "market_entropy":         market_entropy,
    }


def kelly_stake(prob, odd, fraction=KELLY_FRACTION, max_pct=0.05):
    edge = prob * odd - 1
    if edge <= 0:
        return 0.0
    k = (prob - (1 - prob) / (odd - 1)) * fraction
    return min(max(k, 0.0), max_pct)


def run_value_bets(odds_df=None):
    print("📂 Carregando dados...")

    if odds_df is None:
        odds_df = fetch_odds()
    if odds_df.empty:
        print("❌ Sem odds disponíveis")
        return pd.DataFrame()

    features_df = pd.read_csv(FEATURES_PATH)
    if "season_x" in features_df.columns:
        features_df = features_df.rename(columns={"season_x": "season"})
    features_df = features_df.sort_values("date")

    matches_df = pd.read_csv(MATCHES_PATH)
    matches_df = matches_df.dropna(subset=["home_goals", "away_goals"])

    try:
        mv_df   = pd.read_csv(MARKET_PATH)
        mv_dict = dict(zip(mv_df["team"], mv_df["market_value_eur_m"]))
    except Exception:
        mv_dict = {}
    all_mv = list(mv_dict.values()) if mv_dict else [50]

    model_data = joblib.load(MODEL_PATH)
    feature_order = model_data["features"]

    print(f"   {len(odds_df)} jogos com odds da Bet365")

    value_bets = []

    for _, game in odds_df.iterrows():
        home  = game["home_team"]
        away  = game["away_team"]
        odd_h = game["odd_h"]
        odd_d = game["odd_d"]
        odd_a = game["odd_a"]

        # Construir features
        feat = build_feature_row(home, away, odd_h, odd_d, odd_a,
                                 features_df, matches_df, mv_dict, all_mv)
        feat = add_derived(pd.DataFrame([feat])).iloc[0].to_dict()
        X    = np.array([[feat.get(f, 0) for f in feature_order]])

        # Probabilidades do modelo
        ph  = model_data["cal_h"].predict(model_data["model_h"].predict_proba(X)[:, 1])[0]
        pd_ = model_data["cal_d"].predict(model_data["model_d"].predict_proba(X)[:, 1])[0]
        pa  = model_data["cal_a"].predict(model_data["model_a"].predict_proba(X)[:, 1])[0]
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        # Resultado mais provável
        probs   = {"H": ph, "D": pd_, "A": pa}
        odds    = {"H": odd_h, "D": odd_d, "A": odd_a}
        labels  = {"H": f"{home} vence", "D": "Empate", "A": f"{away} vence"}
        pred    = max(probs, key=probs.get)

        # Verificar value em todas as opções
        for outcome in ["H", "D", "A"]:
            prob = probs[outcome]
            odd  = odds[outcome]
            value = prob * odd

            if value >= MIN_VALUE and prob >= MIN_PROB:
                kelly_pct = kelly_stake(prob, odd)
                edge_pct  = (value - 1) * 100

                value_bets.append({
                    "date":          game["date"],
                    "time_utc":      game["time_utc"],
                    "home_team":     home,
                    "away_team":     away,
                    "aposta":        labels[outcome],
                    "outcome":       outcome,
                    "prob_modelo":   round(prob, 3),
                    "prob_mercado":  round(game[f"prob_{outcome.lower()}_mkt"], 3),
                    "odd_bet365":    round(odd, 2),
                    "value":         round(value, 3),
                    "edge_pct":      round(edge_pct, 1),
                    "kelly_pct":     round(kelly_pct * 100, 2),
                    "pred_modelo":   labels[pred],
                    "prob_h":        round(ph, 3),
                    "prob_d":        round(pd_, 3),
                    "prob_a":        round(pa, 3),
                    "odd_h":         odd_h,
                    "odd_d":         odd_d,
                    "odd_a":         odd_a,
                    "margin_pct":    round(game["margin"], 2),
                })

    if not value_bets:
        print("   ℹ️  Nenhum value bet encontrado com os filtros atuais")
        print(f"   Ajuste MIN_VALUE ({MIN_VALUE}) ou MIN_PROB ({MIN_PROB})")
        return pd.DataFrame()

    df_vb = pd.DataFrame(value_bets).sort_values("value", ascending=False)
    df_vb.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*65}")
    print(f"🎯 VALUE BETS ENCONTRADOS: {len(df_vb)}")
    print(f"{'='*65}\n")

    for _, r in df_vb.iterrows():
        stars = "⭐" * min(int(r["edge_pct"] / 5) + 1, 5)
        print(f"📅 {r['date']} {r['time_utc']} UTC")
        print(f"   {r['home_team']} vs {r['away_team']}")
        print(f"   🎯 Apostar: {r['aposta']}")
        print(f"   Prob modelo: {r['prob_modelo']:.1%} | Prob mercado: {r['prob_mercado']:.1%}")
        print(f"   Odd Bet365:  {r['odd_bet365']} | Value: {r['value']:.3f} | Edge: +{r['edge_pct']:.1f}% {stars}")
        print(f"   Kelly stake: {r['kelly_pct']:.2f}% do bankroll")
        print(f"   Previsão modelo: {r['pred_modelo']} "
              f"(H:{r['prob_h']:.0%} D:{r['prob_d']:.0%} A:{r['prob_a']:.0%})")
        print()

    print(f"✅ Salvo em {OUTPUT_PATH}")
    return df_vb


if __name__ == "__main__":
    run_value_bets()
