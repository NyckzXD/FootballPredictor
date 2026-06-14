"""
dixon_coles_model.py — Modelo Dixon-Coles para previsão de gols no Brasileirão.

Abordagem híbrida:
  1. Estima λ_h e λ_a via regressão de Poisson (igual ao modelo atual)
  2. Estima o parâmetro ρ que corrige resultados de baixa pontuação (0-0, 1-0, 0-1, 1-1)
  3. Deriva probabilidades H/D/A a partir da distribuição bivariada corrigida

Referência: Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", Applied Statistics 46(2):265–280.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import joblib
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\dixon_coles_model.pkl"

MAX_GOALS = 10  # limite superior para a grade de probabilidades

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


# ─────────────────────────────────────────────────────────────────────────────
# Função de correção τ (Dixon-Coles)
# ─────────────────────────────────────────────────────────────────────────────

def tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """
    Fator de correção para resultados de baixa pontuação.
    τ(0,0) = 1 − λ_h·λ_a·ρ
    τ(1,0) = 1 + λ_a·ρ
    τ(0,1) = 1 + λ_h·ρ
    τ(1,1) = 1 − ρ
    outros = 1
    """
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    """
    Matriz de probabilidades [home_goals × away_goals] com correção DC.
    Shape: (MAX_GOALS+1, MAX_GOALS+1). Soma das entradas ≈ 1.
    """
    g = np.arange(MAX_GOALS + 1)
    # Grade Poisson independente
    mat = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    # Aplicar τ nos 4 cantos de baixa pontuação
    for i in range(2):
        for j in range(2):
            mat[i, j] *= tau(i, j, lh, la, rho)
    # Renormalizar (correção muda levemente a soma)
    mat = np.clip(mat, 0, None)
    mat /= mat.sum()
    return mat


def outcome_probs(lh: float, la: float, rho: float) -> tuple[float, float, float]:
    """Probabilidades (P_H, P_D, P_A) com correção Dixon-Coles."""
    mat = score_matrix(lh, la, rho)
    p_h = float(np.tril(mat, -1).sum())
    p_a = float(np.triu(mat,  1).sum())
    p_d = float(np.trace(mat))
    return p_h, p_d, p_a


# ─────────────────────────────────────────────────────────────────────────────
# Estimação de ρ por MLE
# ─────────────────────────────────────────────────────────────────────────────

def _neg_ll_rho(rho: float, lh_arr: np.ndarray, la_arr: np.ndarray,
                x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Log-verossimilhança negativa para estimar ρ dado os lambdas."""
    total = 0.0
    for lh, la, x, y in zip(lh_arr, la_arr, x_arr, y_arr):
        t = tau(int(x), int(y), lh, la, rho)
        if t <= 0:
            return 1e12
        p = float(poisson.pmf(int(x), lh) * poisson.pmf(int(y), la)) * t
        if p <= 1e-15:
            return 1e12
        total -= np.log(p)
    return total


def estimate_rho(lh_arr: np.ndarray, la_arr: np.ndarray,
                 x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Otimiza ρ via MLE escalar em [-0.5, 0.3]."""
    result = minimize_scalar(
        _neg_ll_rho,
        bounds=(-0.5, 0.3),
        method="bounded",
        args=(lh_arr, la_arr, x_arr, y_arr),
    )
    return float(result.x)


# ─────────────────────────────────────────────────────────────────────────────
# Classe principal
# ─────────────────────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo híbrido: Poisson Regressor (ML) + correção Dixon-Coles (estatística).
    Interface compatível com poisson_model.pkl.
    """

    def __init__(self):
        self.model_home   = PoissonRegressor(alpha=0.1, max_iter=1000)
        self.model_away   = PoissonRegressor(alpha=0.1, max_iter=1000)
        self.scaler_home  = StandardScaler()
        self.scaler_away  = StandardScaler()
        self.rho          = -0.10   # valor inicial típico para futebol
        self.features_home = FEATURE_COLS_HOME
        self.features_away = FEATURE_COLS_AWAY

    def fit(self, df_train: pd.DataFrame) -> None:
        print("🔧 Treinando Poisson (mandante)...")
        Xh = self.scaler_home.fit_transform(df_train[FEATURE_COLS_HOME])
        Xa = self.scaler_away.fit_transform(df_train[FEATURE_COLS_AWAY])
        yh = df_train["home_goals"].values
        ya = df_train["away_goals"].values

        self.model_home.fit(Xh, yh)
        self.model_away.fit(Xa, ya)

        lh = np.clip(self.model_home.predict(Xh), 0.1, 8.0)
        la = np.clip(self.model_away.predict(Xa), 0.1, 8.0)

        print("📐 Estimando ρ (Dixon-Coles)...")
        self.rho = estimate_rho(lh, la, yh, ya)
        print(f"   ρ = {self.rho:.4f}  (negativo = sub-representação de 0-0/1-1 corrigida)")

    def predict_lambdas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xh = self.scaler_home.transform(df[FEATURE_COLS_HOME])
        Xa = self.scaler_away.transform(df[FEATURE_COLS_AWAY])
        lh = np.clip(self.model_home.predict(Xh), 0.1, 8.0)
        la = np.clip(self.model_away.predict(Xa), 0.1, 8.0)
        return lh, la

    def predict_outcome_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        lh, la = self.predict_lambdas(df)
        probs = np.array([outcome_probs(h, a, self.rho) for h, a in zip(lh, la)])
        return pd.DataFrame({
            "prob_h_dc": probs[:, 0],
            "prob_d_dc": probs[:, 1],
            "prob_a_dc": probs[:, 2],
            "lambda_h":  lh,
            "lambda_a":  la,
        })

    def sample_score(self, lh: float, la: float) -> tuple[int, int]:
        """Amostra um placar da distribuição bivariada corrigida."""
        mat  = score_matrix(lh, la, self.rho)
        flat = mat.ravel()
        idx  = np.random.choice(len(flat), p=flat / flat.sum())
        return divmod(idx, MAX_GOALS + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Treinamento & avaliação
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("📊 Carregando dados...")
    df = pd.read_csv(DATA_PATH)
    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    all_feats = list(set(FEATURE_COLS_HOME + FEATURE_COLS_AWAY))
    df = df.dropna(subset=all_feats + ["home_goals", "away_goals"])
    df = df[df["home_goals"] >= 0]
    df = df[df["away_goals"] >= 0]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas | temporadas: {sorted(df['season'].unique())}")

    train_mask = df["season"].isin(range(2012, 2025))
    test_mask  = df["season"].isin([2025, 2026])
    df_train   = df[train_mask].copy()
    df_test    = df[test_mask].copy()

    print(f"   Treino: {len(df_train)} | Teste: {len(df_test)}")

    model = DixonColesModel()
    model.fit(df_train)

    # ── Avaliação no teste ────────────────────────────────────────────────────
    if len(df_test) > 0:
        lh, la = model.predict_lambdas(df_test)
        print(f"\n📊 Resultados no TESTE (2025-2026):")
        print(f"   Mandante — MAE={mean_absolute_error(df_test['home_goals'], lh):.3f} | "
              f"média real={df_test['home_goals'].mean():.3f} | prev={lh.mean():.3f}")
        print(f"   Visitante — MAE={mean_absolute_error(df_test['away_goals'], la):.3f} | "
              f"média real={df_test['away_goals'].mean():.3f} | prev={la.mean():.3f}")

        probs_df = model.predict_outcome_probs(df_test)
        y_pred   = probs_df[["prob_h_dc", "prob_d_dc", "prob_a_dc"]].values.argmax(axis=1)
        y_map    = {"H": 0, "D": 1, "A": 2}
        y_true   = df_test["result"].map(y_map).values
        acc      = (y_pred == y_true).mean()
        print(f"\n   Acurácia DC (argmax): {acc:.2%}")

        print(f"\n   Distribuição prevista vs real:")
        for res, idx in y_map.items():
            pred_mean = probs_df.iloc[:, idx].mean()
            real_frac = (df_test["result"] == res).mean()
            print(f"   {res}: prev={pred_mean:.3f} | real={real_frac:.3f}")

        print(f"\n   Distribuição de gols (Poisson + DC, μ_h={lh.mean():.2f} μ_a={la.mean():.2f}):")
        from scipy.stats import poisson as scipy_poisson
        for g in range(6):
            p_h = scipy_poisson.pmf(g, lh.mean())
            p_a = scipy_poisson.pmf(g, la.mean())
            print(f"   {g} gols — mandante: {p_h:.1%} | visitante: {p_a:.1%}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Modelo Dixon-Coles salvo em {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
