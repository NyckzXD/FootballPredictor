import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\poisson_model.pkl"

FEATURE_COLS_HOME = [
    "home_elo", "away_elo", "elo_diff",
    "home_avg_gf", "home_avg_ga", "home_form_pts",
    "away_avg_gf", "away_avg_ga", "away_form_pts",
    "home_avg_gf_10", "home_avg_ga_10",
    "away_avg_gf_10", "away_avg_ga_10",
    "home_aproveitamento", "away_aproveitamento",
    "market_value_diff", "home_market_value_log",
    "home_home_form", "away_away_form",
    "h2h_home_wins", "h2h_draws",
]

FEATURE_COLS_AWAY = [
    "home_elo", "away_elo", "elo_diff",
    "home_avg_gf", "home_avg_ga", "home_form_pts",
    "away_avg_gf", "away_avg_ga", "away_form_pts",
    "home_avg_gf_10", "home_avg_ga_10",
    "away_avg_gf_10", "away_avg_ga_10",
    "home_aproveitamento", "away_aproveitamento",
    "market_value_diff", "away_market_value_log",
    "home_home_form", "away_away_form",
    "h2h_away_wins", "h2h_draws",
]


def train():
    print("📊 Carregando dados...")
    df = pd.read_csv(DATA_PATH)

    # Corrigir colunas do merge
    if "season_x" in df.columns:
        df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns:
        df = df.rename(columns={"result_x": "result"})

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    all_feats = list(set(FEATURE_COLS_HOME + FEATURE_COLS_AWAY))
    df = df.dropna(subset=all_feats + ["home_goals", "away_goals"])
    df = df[df["home_goals"] >= 0]
    df = df[df["away_goals"] >= 0]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas válidas")
    print(f"   Gols mandante: média={df['home_goals'].mean():.3f} | max={df['home_goals'].max():.0f}")
    print(f"   Gols visitante: média={df['away_goals'].mean():.3f} | max={df['away_goals'].max():.0f}")
    print(f"   Temporadas: {sorted(df['season'].unique())}")

    # ── Split temporal — treino 2012-2024, teste 2025-2026 ──
    train_mask = df["season"].isin(range(2012, 2025))
    test_mask  = df["season"].isin([2025, 2026])

    X_home_tr = df.loc[train_mask, FEATURE_COLS_HOME]
    X_away_tr = df.loc[train_mask, FEATURE_COLS_AWAY]
    X_home_te = df.loc[test_mask,  FEATURE_COLS_HOME]
    X_away_te = df.loc[test_mask,  FEATURE_COLS_AWAY]

    y_home_tr = df.loc[train_mask, "home_goals"].values
    y_away_tr = df.loc[train_mask, "away_goals"].values
    y_home_te = df.loc[test_mask,  "home_goals"].values
    y_away_te = df.loc[test_mask,  "away_goals"].values

    print(f"\n   Treino: {train_mask.sum()} jogos (2012-2024)")
    print(f"   Teste:  {test_mask.sum()} jogos (2025-2026)")

    # ── Escalar features ──
    scaler_home = StandardScaler()
    scaler_away = StandardScaler()

    X_home_tr_s = scaler_home.fit_transform(X_home_tr)
    X_home_te_s = scaler_home.transform(X_home_te)
    X_away_tr_s = scaler_away.fit_transform(X_away_tr)
    X_away_te_s = scaler_away.transform(X_away_te)

    # ── Treinar modelos Poisson ──
    print("\n🔧 Treinando modelo Poisson (mandante)...")
    model_home = PoissonRegressor(alpha=0.1, max_iter=1000)
    model_home.fit(X_home_tr_s, y_home_tr)

    print("🔧 Treinando modelo Poisson (visitante)...")
    model_away = PoissonRegressor(alpha=0.1, max_iter=1000)
    model_away.fit(X_away_tr_s, y_away_tr)

    # ── Avaliar ──
    pred_home_tr = model_home.predict(X_home_tr_s)
    pred_away_tr = model_away.predict(X_away_tr_s)
    pred_home_te = model_home.predict(X_home_te_s)
    pred_away_te = model_away.predict(X_away_te_s)

    print(f"\n📊 Resultados no TREINO (2012-2024):")
    print(f"   Mandante — MAE={mean_absolute_error(y_home_tr, pred_home_tr):.3f} | "
          f"média real={y_home_tr.mean():.3f} | média prev={pred_home_tr.mean():.3f}")
    print(f"   Visitante — MAE={mean_absolute_error(y_away_tr, pred_away_tr):.3f} | "
          f"média real={y_away_tr.mean():.3f} | média prev={pred_away_tr.mean():.3f}")

    if test_mask.sum() > 0:
        print(f"\n📊 Resultados no TESTE (2025-2026):")
        print(f"   Mandante — MAE={mean_absolute_error(y_home_te, pred_home_te):.3f} | "
              f"média real={y_home_te.mean():.3f} | média prev={pred_home_te.mean():.3f}")
        print(f"   Visitante — MAE={mean_absolute_error(y_away_te, pred_away_te):.3f} | "
              f"média real={y_away_te.mean():.3f} | média prev={pred_away_te.mean():.3f}")

    # Distribuição de gols previstos
    print(f"\n📐 Distribuição de gols previstos (teste):")
    for g in range(6):
        from scipy.stats import poisson
        p_h = poisson.pmf(g, pred_home_te.mean())
        p_a = poisson.pmf(g, pred_away_te.mean())
        print(f"   {g} gols — mandante: {p_h:.1%} | visitante: {p_a:.1%}")

    # ── Salvar ──
    joblib.dump({
        "model_home":    model_home,
        "model_away":    model_away,
        "scaler_home":   scaler_home,
        "scaler_away":   scaler_away,
        "features_home": FEATURE_COLS_HOME,
        "features_away": FEATURE_COLS_AWAY,
    }, MODEL_PATH)

    print(f"\n✅ Modelo Poisson salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()
