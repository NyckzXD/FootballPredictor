import pandas as pd
import numpy as np
import joblib

MODEL_PATH  = r"C:\PREDICTOR\models\match_model.pkl"
ODDS_PATH   = r"C:\PREDICTOR\scraping\data\external\odds.csv"
FEATURES_PATH = r"C:\PREDICTOR\scraping\data\processed\features.csv"
MARKET_PATH = r"C:\PREDICTOR\scraping\data\external\market_values.csv"
OUTPUT_PATH = r"C:\PREDICTOR\scraping\data\external\value_bets.csv"

# Mapeamento nomes Odds API → football-data.org
TEAM_MAP = {
    "Palmeiras":           "SE Palmeiras",
    "Flamengo":            "CR Flamengo",
    "São Paulo":           "São Paulo FC",
    "Corinthians":         "SC Corinthians Paulista",
    "Atlético Mineiro":    "CA Mineiro",
    "Fluminense":          "Fluminense FC",
    "Botafogo":            "Botafogo FR",
    "Athletico Paranaense":"CA Paranaense",
    "Grêmio":              "Grêmio FBPA",
    "Internacional":       "SC Internacional",
    "Vasco da Gama":       "CR Vasco da Gama",
    "Cruzeiro":            "Cruzeiro EC",
    "Santos":              "Santos FC",
    "Red Bull Bragantino": "RB Bragantino",
    "Bahia":               "EC Bahia",
    "Mirassol":            "Mirassol FC",
    "Remo":                "Clube do Remo",
    "Vitória":             "EC Vitória",
    "Chapecoense":         "Chapecoense AF",
    "Coritiba":            "Coritiba FBC",
}


def predict_probs(home: str, away: str, model, feature_cols, le,
                  features_df, mv_dict) -> dict:
    """Prevê probabilidades para um jogo."""
    def get_mv(team):
        return mv_dict.get(team, {
            "market_value_log": 3.0, "market_value_norm": 0.3, "squad_size": 25
        })

    def get_stats(team):
        rows = features_df[
            (features_df["home_team"] == team) |
            (features_df["away_team"] == team)
        ].tail(1)
        if rows.empty:
            return {k: 1.0 for k in [
                "form_pts","avg_gf","avg_ga","goal_diff","win_rate","draw_rate",
                "home_form","away_form","form_pts_10","avg_gf_10","avg_ga_10",
                "win_rate_10","elo"
            ]}
        r = rows.iloc[0]
        prefix = "home" if r["home_team"] == team else "away"
        return {
            "form_pts":    r.get(f"{prefix}_form_pts", 1.0),
            "avg_gf":      r.get(f"{prefix}_avg_gf", 1.2),
            "avg_ga":      r.get(f"{prefix}_avg_ga", 1.2),
            "goal_diff":   r.get(f"{prefix}_goal_diff", 0.0),
            "win_rate":    r.get(f"{prefix}_win_rate", 0.33),
            "draw_rate":   r.get(f"{prefix}_draw_rate", 0.33),
            "home_form":   r.get("home_home_form", 1.0),
            "away_form":   r.get("away_away_form", 1.0),
            "form_pts_10": r.get(f"{prefix}_form_pts_10", 1.0),
            "avg_gf_10":   r.get(f"{prefix}_avg_gf_10", 1.2),
            "avg_ga_10":   r.get(f"{prefix}_avg_ga_10", 1.2),
            "win_rate_10": r.get(f"{prefix}_win_rate_10", 0.33),
            "elo":         r.get(f"{prefix}_elo", 1500.0),
        }

    hs  = get_stats(home)
    as_ = get_stats(away)
    h_mv = get_mv(home)
    a_mv = get_mv(away)

    row = {
        "elo_diff":           hs["elo"] - as_["elo"],
        "home_elo":           hs["elo"], "away_elo": as_["elo"],
        "home_market_value_log":  h_mv["market_value_log"],
        "home_market_value_norm": h_mv["market_value_norm"],
        "home_squad_size":        h_mv["squad_size"],
        "away_market_value_log":  a_mv["market_value_log"],
        "away_market_value_norm": a_mv["market_value_norm"],
        "away_squad_size":        a_mv["squad_size"],
        "market_value_diff":      h_mv["market_value_log"] - a_mv["market_value_log"],
        "home_position":5,"away_position":10,
        "home_aproveitamento":0.5,"away_aproveitamento":0.4,
        "home_table_gd":0,"away_table_gd":0,"position_diff":5,
        "home_form_pts":   hs["form_pts"],  "home_avg_gf":  hs["avg_gf"],
        "home_avg_ga":     hs["avg_ga"],    "home_goal_diff":hs["goal_diff"],
        "home_win_rate":   hs["win_rate"],  "home_draw_rate":hs["draw_rate"],
        "home_home_form":  hs["home_form"],
        "away_form_pts":   as_["form_pts"], "away_avg_gf":  as_["avg_gf"],
        "away_avg_ga":     as_["avg_ga"],   "away_goal_diff":as_["goal_diff"],
        "away_win_rate":   as_["win_rate"], "away_draw_rate":as_["draw_rate"],
        "away_away_form":  as_["away_form"],
        "home_form_pts_10":hs["form_pts_10"],"home_avg_gf_10":hs["avg_gf_10"],
        "home_avg_ga_10":  hs["avg_ga_10"], "home_win_rate_10":hs["win_rate_10"],
        "away_form_pts_10":as_["form_pts_10"],"away_avg_gf_10":as_["avg_gf_10"],
        "away_avg_ga_10":  as_["avg_ga_10"],"away_win_rate_10":as_["win_rate_10"],
        "h2h_home_wins":0,"h2h_away_wins":0,"h2h_draws":0,
    }

    X         = pd.DataFrame([row])[feature_cols]
    probs_enc = model.predict_proba(X)[0]
    return {le.classes_[i]: probs_enc[i] for i in range(len(le.classes_))}


def find_value_bets(min_edge: float = 0.05) -> pd.DataFrame:
    """
    Encontra value bets onde modelo > prob implícita da odd.
    min_edge: diferença mínima para considerar value (5% padrão)
    """
    saved    = joblib.load(MODEL_PATH)
    model    = saved["model"]
    feat_cols = saved["features"]
    le       = saved["label_encoder"]
    features = pd.read_csv(FEATURES_PATH)
    mv_dict  = pd.read_csv(MARKET_PATH).set_index("team").to_dict("index")
    odds_df  = pd.read_csv(ODDS_PATH)

    if odds_df.empty:
        print("Nenhuma odd disponível.")
        return pd.DataFrame()

    results = []
    for _, row in odds_df.iterrows():
        # Mapear nomes
        home = TEAM_MAP.get(row["home_team"], row["home_team"])
        away = TEAM_MAP.get(row["away_team"], row["away_team"])

        try:
            probs = predict_probs(home, away, model, feat_cols, le, features, mv_dict)
        except Exception as e:
            print(f"Erro {home} vs {away}: {e}")
            continue

        # Calcular edge (vantagem do modelo sobre a odd)
        edge_h = probs.get("H", 0) - row["prob_home"]
        edge_d = probs.get("D", 0) - row["prob_draw"]
        edge_a = probs.get("A", 0) - row["prob_away"]

        # Calcular ROI esperado
        roi_h = (probs.get("H",0) * row["odd_home"]) - 1
        roi_d = (probs.get("D",0) * row["odd_draw"]) - 1
        roi_a = (probs.get("A",0) * row["odd_away"]) - 1

        for outcome, edge, roi, odd, model_prob, mkt_prob in [
            ("H", edge_h, roi_h, row["odd_home"], probs.get("H",0), row["prob_home"]),
            ("D", edge_d, roi_d, row["odd_draw"], probs.get("D",0), row["prob_draw"]),
            ("A", edge_a, roi_a, row["odd_away"], probs.get("A",0), row["prob_away"]),
        ]:
            if edge >= min_edge:  # só value bets
                results.append({
                    "data":         row["date"][:10],
                    "partida":      f"{row['home_team']} vs {row['away_team']}",
                    "casa":         row["bookmaker"],
                    "resultado":    {"H":"Casa","D":"Empate","A":"Fora"}[outcome],
                    "odd":          round(odd, 2),
                    "prob_modelo":  f"{model_prob:.1%}",
                    "prob_mercado": f"{mkt_prob:.1%}",
                    "edge":         f"+{edge:.1%}",
                    "roi_esperado": f"{roi:.1%}",
                    "⭐ valor":     "🔥 FORTE" if edge > 0.10 else "✅ BOM",
                })

    df = pd.DataFrame(results)
    if df.empty:
        print("Nenhum value bet encontrado com edge >= {min_edge:.0%}")
        return df

    df = df.sort_values("edge", ascending=False)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"🎯 {len(df)} VALUE BETS ENCONTRADOS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    find_value_bets(min_edge=0.05)