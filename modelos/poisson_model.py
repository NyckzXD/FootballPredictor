import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

DATA_PATH        = r"C:\PREDICTOR\scraping\data\processed\features.csv"
POISSON_MODEL_PATH = r"C:\PREDICTOR\models\poisson_model.pkl"

GOAL_FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_avg_gf", "home_avg_ga", "home_goal_diff",
    "away_avg_gf", "away_avg_ga", "away_goal_diff",
    "home_form_pts", "away_form_pts",
    "home_avg_gf_10", "home_avg_ga_10",
    "away_avg_gf_10", "away_avg_ga_10",
    "home_aproveitamento", "away_aproveitamento",
    "position_diff",
]

def train_poisson():
    print("📊 Carregando features...")
    df = pd.read_csv(DATA_PATH).dropna(subset=GOAL_FEATURES)
    print(f"   {len(df)} partidas")

    X = df[GOAL_FEATURES]
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🤖 Treinando Poisson para gols mandante...")
    model_home = PoissonRegressor(alpha=0.1, max_iter=300)
    model_home.fit(X_scaled, y_home)

    print("🤖 Treinando Poisson para gols visitante...")
    # Para o visitante, invertemos o elo_diff
    X_away = X.copy()
    X_away["elo_diff"]   = -X_away["elo_diff"]
    X_away["position_diff"] = -X_away["position_diff"]
    X_away_scaled = scaler.transform(X_away)
    model_away = PoissonRegressor(alpha=0.1, max_iter=300)
    model_away.fit(X_away_scaled, y_away)

    # Avaliar
    pred_home = model_home.predict(X_scaled)
    pred_away = model_away.predict(X_away_scaled)
    mae_home = np.mean(np.abs(pred_home - y_home))
    mae_away = np.mean(np.abs(pred_away - y_away))
    print(f"\n📊 MAE gols mandante:  {mae_home:.3f}")
    print(f"📊 MAE gols visitante: {mae_away:.3f}")

    # Distribuição esperada vs real
    print(f"\n📊 Média gols mandante  — real: {y_home.mean():.3f} | previsto: {pred_home.mean():.3f}")
    print(f"📊 Média gols visitante — real: {y_away.mean():.3f} | previsto: {pred_away.mean():.3f}")

    joblib.dump({
        "model_home":   model_home,
        "model_away":   model_away,
        "scaler":       scaler,
        "features":     GOAL_FEATURES,
    }, POISSON_MODEL_PATH)
    print(f"\n✅ Modelos Poisson salvos em {POISSON_MODEL_PATH}")

    return model_home, model_away, scaler


def predict_goals(home_features: dict, away_features: dict) -> tuple:
    """
    Retorna (lambda_home, lambda_away) — taxas de gols esperados.
    Use np.random.poisson(lambda) para sortear gols.
    """
    saved = joblib.load(POISSON_MODEL_PATH)
    model_home = saved["model_home"]
    model_away = saved["model_away"]
    scaler     = saved["scaler"]
    features   = saved["features"]

    X_home = pd.DataFrame([home_features])[features]
    X_away = pd.DataFrame([away_features])[features]
    X_away["elo_diff"]      = -X_away["elo_diff"]
    X_away["position_diff"] = -X_away["position_diff"]

    lam_home = model_home.predict(scaler.transform(X_home))[0]
    lam_away = model_away.predict(scaler.transform(X_away))[0]

    return max(0.2, lam_home), max(0.2, lam_away)


if __name__ == "__main__":
    train_poisson()