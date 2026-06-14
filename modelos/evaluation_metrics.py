"""
evaluation_metrics.py — Métricas probabilísticas para predição de futebol.

RPS  : Ranked Probability Score (padrão para previsão multi-resultado)
CLV  : Closing Line Value (valor real vs. mercado de fechamento)
Brier: Brier Score multivariado
"""
import numpy as np
import pandas as pd

# Ordem canônica dos resultados
OUTCOME_ORDER = {"H": 0, "D": 1, "A": 2}
IDX_TO_OUTCOME = {0: "H", 1: "D", 2: "A"}


# ─────────────────────────────────────────────────────────────────────────────
# RPS — Ranked Probability Score
# ─────────────────────────────────────────────────────────────────────────────

def rps_match(probs: np.ndarray, outcome: str) -> float:
    """
    RPS para um único jogo (3 resultados: H, D, A — ordem ordinal).
    Intervalo [0, 1]. Menor = melhor. Perfeito = 0.
    """
    o = np.zeros(3)
    o[OUTCOME_ORDER[outcome]] = 1.0
    cum_p = np.cumsum(probs[:2])
    cum_o = np.cumsum(o[:2])
    return float(np.mean((cum_p - cum_o) ** 2))


def rps_serie(probs_hda: np.ndarray, outcomes) -> np.ndarray:
    """RPS para série de jogos. probs_hda: (n, 3)."""
    return np.array([rps_match(p, o) for p, o in zip(probs_hda, outcomes)])


# ─────────────────────────────────────────────────────────────────────────────
# Log-Loss
# ─────────────────────────────────────────────────────────────────────────────

def log_loss_match(probs: np.ndarray, outcome: str, eps: float = 1e-7) -> float:
    idx = OUTCOME_ORDER[outcome]
    return -np.log(float(np.clip(probs[idx], eps, 1.0)))


def log_loss_serie(probs_hda: np.ndarray, outcomes, eps: float = 1e-7) -> np.ndarray:
    return np.array([log_loss_match(p, o, eps) for p, o in zip(probs_hda, outcomes)])


# ─────────────────────────────────────────────────────────────────────────────
# Brier Score
# ─────────────────────────────────────────────────────────────────────────────

def brier_match(probs: np.ndarray, outcome: str) -> float:
    o = np.zeros(3)
    o[OUTCOME_ORDER[outcome]] = 1.0
    return float(np.mean((probs - o) ** 2))


def brier_serie(probs_hda: np.ndarray, outcomes) -> np.ndarray:
    return np.array([brier_match(p, o) for p, o in zip(probs_hda, outcomes)])


# ─────────────────────────────────────────────────────────────────────────────
# Closing Line Value (CLV)
# ─────────────────────────────────────────────────────────────────────────────

def clv(prob_model: float, closing_odd: float, eps: float = 1e-7) -> float:
    """
    CLV = log(prob_model * closing_odd).
    > 0 → modelo tem edge sobre o mercado de fechamento.
    Padrão-ouro para avaliar valor das apostas a longo prazo.
    """
    if closing_odd <= 1.0:
        return 0.0
    prob_closing = 1.0 / closing_odd
    return float(np.log(np.clip(prob_model / prob_closing, eps, None)))


def clv_serie(prob_models: np.ndarray, closing_odds: np.ndarray) -> np.ndarray:
    return np.array([clv(p, o) for p, o in zip(prob_models, closing_odds)])


# ─────────────────────────────────────────────────────────────────────────────
# Baseline de comparação (modelo ingênuo)
# ─────────────────────────────────────────────────────────────────────────────

def naive_baseline_rps(outcomes) -> float:
    """RPS do modelo ingênuo: probabilidades históricas fixas do Brasileirão."""
    freq = {"H": 0.46, "D": 0.25, "A": 0.29}
    probs = np.array([freq["H"], freq["D"], freq["A"]])
    return float(np.mean([rps_match(probs, o) for o in outcomes]))


def market_baseline_rps(probs_mkt: np.ndarray, outcomes) -> float:
    """RPS usando probabilidades de mercado como baseline."""
    return float(rps_serie(probs_mkt, outcomes).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Relatório completo
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_report(
    probs_hda: np.ndarray,
    outcomes,
    label: str = "Teste",
    probs_mkt: np.ndarray = None,
) -> dict:
    outcomes = np.asarray(outcomes)
    rps_vals = rps_serie(probs_hda, outcomes)
    ll_vals  = log_loss_serie(probs_hda, outcomes)
    bs_vals  = brier_serie(probs_hda, outcomes)

    pred_idx = probs_hda.argmax(axis=1)
    real_idx = np.array([OUTCOME_ORDER[o] for o in outcomes])
    acc      = (pred_idx == real_idx).mean()

    print(f"\n{'─'*60}")
    print(f"📊 Métricas probabilísticas — {label}  (n={len(outcomes)})")
    print(f"{'─'*60}")
    print(f"  Acurácia  : {acc:.4f}  ({acc:.2%})")
    print(f"  RPS médio : {rps_vals.mean():.4f}  (↓ menor = melhor)")
    print(f"  Log-Loss  : {ll_vals.mean():.4f}  (↓ menor = melhor)")
    print(f"  Brier     : {bs_vals.mean():.4f}  (↓ menor = melhor)")

    naive_rps = naive_baseline_rps(outcomes)
    print(f"\n  Referência (modelo ingênuo):")
    print(f"  RPS ingênuo: {naive_rps:.4f}  |  Ganho modelo: {naive_rps - rps_vals.mean():+.4f}")

    if probs_mkt is not None:
        mkt_rps = market_baseline_rps(probs_mkt, outcomes)
        print(f"  RPS mercado: {mkt_rps:.4f}  |  Ganho vs mercado: {mkt_rps - rps_vals.mean():+.4f}")

    print(f"\n  Calibração por resultado:")
    for res, idx in OUTCOME_ORDER.items():
        mask = outcomes == res
        print(f"  {res}: prev={probs_hda[:, idx].mean():.3f} | real={mask.mean():.3f} "
              f"| diff={probs_hda[:, idx].mean() - mask.mean():+.3f}")

    return {
        "accuracy":  float(acc),
        "rps_mean":  float(rps_vals.mean()),
        "log_loss":  float(ll_vals.mean()),
        "brier":     float(bs_vals.mean()),
        "naive_rps": float(naive_rps),
    }
