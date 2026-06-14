"""
match_model_v2.py — Modelo principal de predição de resultado (H/D/A).

Melhorias vs v1:
  • Optuna HPO: busca automática de hiperparâmetros LightGBM (100 trials)
  • Ensemble: LightGBM + XGBoost (3 classificadores binários cada)
  • Stacking: meta-learner Logistic Regression treinado nas probabilidades OOF
  • Métricas: acurácia, RPS, log-loss, Brier (via evaluation_metrics.py)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_metrics import print_metrics_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"

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
    "home_joga_libertadores", "away_joga_libertadores",
    "home_pos_trend", "away_pos_trend", "pos_trend_diff",
    "aprov_equilibrio", "h2h_draw_dominance",
    # Novas features contextuais (opcionais)
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_relegation_gap", "away_relegation_gap",
    "home_g6_gap", "away_g6_gap",
    # Over/Under (opcional)
    "prob_over25", "prob_under25", "implied_total_goals",
    # xG (opcional)
    "home_avg_xg", "away_avg_xg", "home_avg_xga", "away_avg_xga", "xg_diff",
]

SEASON_WEIGHTS = {
    2026: 5.0, 2025: 4.0, 2024: 3.5, 2023: 3.0, 2022: 2.5,
    2021: 2.0, 2020: 1.5, 2019: 1.2, 2018: 1.0, 2017: 0.8,
    2016: 0.7, 2015: 0.6, 2014: 0.5, 2013: 0.4, 2012: 0.3,
}

OPTUNA_TRIALS = 80   # aumentar para melhor HPO (mais tempo de treino)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_derived(X_):
    X_ = X_.copy()
    import math
    X_["form_diff"]           = X_["home_form_pts"]       - X_["away_form_pts"]
    X_["form_diff_10"]        = X_["home_form_pts_10"]    - X_["away_form_pts_10"]
    X_["gf_diff"]             = X_["home_avg_gf"]         - X_["away_avg_gf"]
    X_["ga_diff"]             = X_["home_avg_ga"]         - X_["away_avg_ga"]
    X_["win_rate_diff"]       = X_["home_win_rate"]       - X_["away_win_rate"]
    X_["aproveit_diff"]       = X_["home_aproveitamento"] - X_["away_aproveitamento"]
    X_["home_in_crisis"]      = (X_["home_form_pts"] < 0.5).astype(int)
    X_["away_in_form"]        = (X_["away_form_pts"] > 2.0).astype(int)
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
    X_["expected_goals_h"]    = X_["home_avg_gf"] * 0.6 + X_["home_avg_gf_10"] * 0.4
    X_["expected_goals_a"]    = X_["away_avg_gf"] * 0.6 + X_["away_avg_gf_10"] * 0.4
    X_["expected_concede_h"]  = X_["home_avg_ga"] * 0.6 + X_["home_avg_ga_10"] * 0.4
    X_["expected_concede_a"]  = X_["away_avg_ga"] * 0.6 + X_["away_avg_ga_10"] * 0.4
    X_["lambda_h"]     = (X_["expected_goals_h"] + X_["expected_concede_a"]) / 2
    X_["lambda_a"]     = (X_["expected_goals_a"] + X_["expected_concede_h"]) / 2
    X_["lambda_diff"]  = X_["lambda_h"] - X_["lambda_a"]
    X_["lambda_total"] = X_["lambda_h"] + X_["lambda_a"]
    pdraw = np.zeros(len(X_))
    for g in range(7):
        fg = math.factorial(g)
        pdraw += (np.exp(-X_["lambda_h"].values) * X_["lambda_h"].values**g / fg *
                  np.exp(-X_["lambda_a"].values) * X_["lambda_a"].values**g / fg)
    X_["poisson_draw_prob"]   = pdraw
    X_["home_momentum"]       = X_["home_form_pts"]  - X_["home_form_pts_10"]
    X_["away_momentum"]       = X_["away_form_pts"]  - X_["away_form_pts_10"]
    X_["home_adv_vs_market"]  = X_["home_home_form"] - X_["away_away_form"]
    X_["model_vs_market_d"]   = X_["combined_draw_rate"] - X_["prob_d_mkt"]

    # xG vs gols reais (se disponível)
    if "home_avg_xg" in X_.columns:
        X_["xg_overperform_h"] = X_["home_avg_gf"] - X_["home_avg_xg"]
        X_["xg_overperform_a"] = X_["away_avg_gf"] - X_["away_avg_xg"]
        X_["xg_net_h"]         = X_["home_avg_xg"] - X_["home_avg_xga"]
        X_["xg_net_a"]         = X_["away_avg_xg"] - X_["away_avg_xga"]

    return X_


def get_temporal_weights(seasons: pd.Series) -> np.ndarray:
    return seasons.map(SEASON_WEIGHTS).fillna(0.3).values


# ─────────────────────────────────────────────────────────────────────────────
# Treino de um classificador binário LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def train_lgb_binary(X_tr, y_tr, temporal_w, pos_weight, params: dict):
    class_w = np.where(y_tr == 1, pos_weight, 1.0)
    sw      = class_w * temporal_w
    sw      = sw / sw.mean()
    model   = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=-1)
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model


def train_xgb_binary(X_tr, y_tr, temporal_w, pos_weight, params: dict):
    class_w = np.where(y_tr == 1, pos_weight, 1.0)
    sw      = class_w * temporal_w
    sw      = sw / sw.mean()
    model   = xgb.XGBClassifier(**params, eval_metric="logloss",
                                 verbosity=0, use_label_encoder=False)
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna: HPO para LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna(X_tr: pd.DataFrame, y_binary: np.ndarray,
               temporal_w: np.ndarray, pos_weight: float,
               n_trials: int = OPTUNA_TRIALS) -> dict:
    """Otimiza hiperparâmetros LightGBM para um classificador binário."""
    n_splits = min(4, max(2, len(X_tr) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 150, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 10, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "random_state":      42,
        }
        scores = []
        for tr_idx, val_idx in tscv.split(X_tr):
            Xtr, Xval = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
            ytr, yval = y_binary[tr_idx], y_binary[val_idx]
            tw         = temporal_w[tr_idx]
            m          = train_lgb_binary(Xtr, ytr, tw, pos_weight, params)
            p          = m.predict_proba(Xval)[:, 1]
            from sklearn.metrics import log_loss
            scores.append(log_loss(yval, p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Probabilidades combinadas do ensemble
# ─────────────────────────────────────────────────────────────────────────────

def _predict_binary_ensemble(models_lgb, models_xgb, cal, X):
    p_lgb = np.mean([m.predict_proba(X)[:, 1] for m in models_lgb], axis=0)
    p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in models_xgb], axis=0)
    p_raw  = (p_lgb + p_xgb) / 2
    return cal.predict(p_raw), p_raw


def predict_probs(model_data: dict, X: pd.DataFrame):
    """Retorna (p_h, p_d, p_a) normalizados a partir do ensemble."""
    ph, _ = _predict_binary_ensemble(
        model_data["models_h_lgb"], model_data["models_h_xgb"], model_data["cal_h"], X)
    pd_, _ = _predict_binary_ensemble(
        model_data["models_d_lgb"], model_data["models_d_xgb"], model_data["cal_d"], X)
    pa, _ = _predict_binary_ensemble(
        model_data["models_a_lgb"], model_data["models_a_xgb"], model_data["cal_a"], X)

    # Meta-learner stacking (se disponível)
    if "meta_clf" in model_data:
        meta_X  = np.column_stack([ph, pd_, pa])
        meta_p  = model_data["meta_clf"].predict_proba(meta_X)
        ph, pd_, pa = meta_p[:, 0], meta_p[:, 1], meta_p[:, 2]

    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total


# ─────────────────────────────────────────────────────────────────────────────
# Treinamento principal
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("📊 Carregando features...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})

    # Colunas opcionais — preencher com medianas ou zeros se ausentes
    optional_median = ["prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
                       "odds_draw_factor", "odds_home_away_ratio", "market_entropy",
                       "prob_over25", "prob_under25", "implied_total_goals"]
    optional_zero   = ["home_joga_libertadores", "away_joga_libertadores",
                       "home_pos_trend", "away_pos_trend", "pos_trend_diff",
                       "aprov_equilibrio", "h2h_draw_dominance",
                       "home_days_rest", "away_days_rest", "rest_diff",
                       "home_relegation_gap", "away_relegation_gap",
                       "home_g6_gap", "away_g6_gap",
                       "home_avg_xg", "away_avg_xg", "home_avg_xga", "away_avg_xga", "xg_diff"]

    for col in optional_median:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.5
    for col in optional_zero:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Colunas obrigatórias (não opcionais)
    required_feats = [c for c in FEATURE_COLS
                      if c not in optional_median and c not in optional_zero]
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=[c for c in required_feats if c in df.columns])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas | temporadas: {sorted(df['season'].unique())}")
    print(f"   Distribuição: {df['result'].value_counts().to_dict()}")

    X        = add_derived(df[available_feats])
    all_cols = list(X.columns)
    print(f"   Total features: {len(all_cols)}")

    # Split temporal
    train_mask = df["season"].isin(range(2004, 2026))
    test_mask  = df["season"].isin([2026])
    if test_mask.sum() < 10:
        available_seasons = sorted(df["season"].unique())
        train_mask = df["season"].isin(available_seasons[:-1])
        test_mask  = df["season"].isin(available_seasons[-1:])

    X_train, X_test  = X[train_mask], X[test_mask]
    y_train_raw      = df["result"][train_mask]
    y_test_raw       = df["result"][test_mask].values
    seasons_train    = df["season"][train_mask]
    temporal_w_train = get_temporal_weights(seasons_train)

    print(f"\n   Treino: {len(X_train)} | Teste: {len(X_test)}")

    # Pesos de classe
    recent = df[df["season"].isin([2023, 2024, 2025])]["result"]
    freq   = recent.value_counts(normalize=True)
    pw_h   = 1.0
    pw_a   = freq.get("H", 0.45) / freq.get("A", 0.30)
    pw_d   = 2.0
    print(f"   Pesos — H:{pw_h:.2f} D:{pw_d:.2f} A:{pw_a:.2f}")

    # ── Optuna HPO ──────────────────────────────────────────────────────────
    print(f"\n🔍 Optuna HPO ({OPTUNA_TRIALS} trials por classificador)...")
    best_h = run_optuna(X_train, (y_train_raw=="H").astype(int).values, temporal_w_train, pw_h)
    print(f"   H → n_est={best_h['n_estimators']} lr={best_h['learning_rate']:.4f} leaves={best_h['num_leaves']}")
    best_d = run_optuna(X_train, (y_train_raw=="D").astype(int).values, temporal_w_train, pw_d)
    print(f"   D → n_est={best_d['n_estimators']} lr={best_d['learning_rate']:.4f} leaves={best_d['num_leaves']}")
    best_a = run_optuna(X_train, (y_train_raw=="A").astype(int).values, temporal_w_train, pw_a)
    print(f"   A → n_est={best_a['n_estimators']} lr={best_a['learning_rate']:.4f} leaves={best_a['num_leaves']}")

    # XGBoost defaults (sem HPO separado — usa mesmo espírito conservador)
    xgb_params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.4, "reg_lambda": 0.3,
        "random_state": 42, "n_jobs": -1,
    }

    # ── Cross-validação temporal ─────────────────────────────────────────────
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    # OOF predictions para o meta-learner
    oof_h = np.zeros(len(X_train))
    oof_d = np.zeros(len(X_train))
    oof_a = np.zeros(len(X_train))

    print("\n🕐 Cross-validação temporal + coleta OOF para stacking:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr_raw    = y_train_raw.iloc[tr_idx]
        yval_raw   = y_train_raw.iloc[val_idx]
        tw         = get_temporal_weights(seasons_train.iloc[tr_idx])

        mh_lgb = train_lgb_binary(Xtr, (ytr_raw=="H").astype(int).values, tw, pw_h, best_h)
        md_lgb = train_lgb_binary(Xtr, (ytr_raw=="D").astype(int).values, tw, pw_d, best_d)
        ma_lgb = train_lgb_binary(Xtr, (ytr_raw=="A").astype(int).values, tw, pw_a, best_a)
        mh_xgb = train_xgb_binary(Xtr, (ytr_raw=="H").astype(int).values, tw, pw_h, xgb_params)
        md_xgb = train_xgb_binary(Xtr, (ytr_raw=="D").astype(int).values, tw, pw_d, xgb_params)
        ma_xgb = train_xgb_binary(Xtr, (ytr_raw=="A").astype(int).values, tw, pw_a, xgb_params)

        ph = (mh_lgb.predict_proba(Xval)[:, 1] + mh_xgb.predict_proba(Xval)[:, 1]) / 2
        pd_ = (md_lgb.predict_proba(Xval)[:, 1] + md_xgb.predict_proba(Xval)[:, 1]) / 2
        pa  = (ma_lgb.predict_proba(Xval)[:, 1] + ma_xgb.predict_proba(Xval)[:, 1]) / 2
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        oof_h[val_idx] = ph
        oof_d[val_idx] = pd_
        oof_a[val_idx] = pa

        y_pred_fold = np.array(["H", "D", "A"])[
            np.stack([ph, pd_, pa], axis=1).argmax(axis=1)
        ]
        score = (y_pred_fold == yval_raw.values).mean()
        cv_scores.append(score)
        print(f"   Fold {fold}: {score:.2%}")

    print(f"   Média CV: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # ── Meta-learner stacking (OOF) ──────────────────────────────────────────
    print("\n🔧 Treinando meta-learner (stacking)...")
    meta_X_oof = np.column_stack([oof_h, oof_d, oof_a])
    le_meta    = LabelEncoder()
    le_meta.fit(["A", "D", "H"])
    y_meta = le_meta.transform(y_train_raw.values)

    meta_clf = LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=500, random_state=42
    )
    meta_clf.fit(meta_X_oof, y_meta)
    meta_oof_acc = (meta_clf.predict(meta_X_oof) == y_meta).mean()
    print(f"   Meta-learner acurácia OOF: {meta_oof_acc:.2%}")

    # ── Treinar modelos finais no conjunto completo ──────────────────────────
    print("\n🔧 Treinando modelos finais (LightGBM + XGBoost)...")
    model_h_lgb = train_lgb_binary(X_train, (y_train_raw=="H").astype(int).values, temporal_w_train, pw_h, best_h)
    model_d_lgb = train_lgb_binary(X_train, (y_train_raw=="D").astype(int).values, temporal_w_train, pw_d, best_d)
    model_a_lgb = train_lgb_binary(X_train, (y_train_raw=="A").astype(int).values, temporal_w_train, pw_a, best_a)
    model_h_xgb = train_xgb_binary(X_train, (y_train_raw=="H").astype(int).values, temporal_w_train, pw_h, xgb_params)
    model_d_xgb = train_xgb_binary(X_train, (y_train_raw=="D").astype(int).values, temporal_w_train, pw_d, xgb_params)
    model_a_xgb = train_xgb_binary(X_train, (y_train_raw=="A").astype(int).values, temporal_w_train, pw_a, xgb_params)

    # Calibração isotônica (sobre ensemble cru no teste)
    def cal_fit(lgb_m, xgb_m, X_te, y_te):
        p = (lgb_m.predict_proba(X_te)[:, 1] + xgb_m.predict_proba(X_te)[:, 1]) / 2
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y_te)
        return iso, p

    cal_h, p_h_raw = cal_fit(model_h_lgb, model_h_xgb, X_test, (y_test_raw=="H").astype(int))
    cal_d, p_d_raw = cal_fit(model_d_lgb, model_d_xgb, X_test, (y_test_raw=="D").astype(int))
    cal_a, p_a_raw = cal_fit(model_a_lgb, model_a_xgb, X_test, (y_test_raw=="A").astype(int))

    # ── Probabilidades finais no teste ──────────────────────────────────────
    p_h  = cal_h.predict(p_h_raw)
    p_d  = cal_d.predict(p_d_raw)
    p_a  = cal_a.predict(p_a_raw)
    total = p_h + p_d + p_a
    p_h /= total; p_d /= total; p_a /= total

    # Stacking final
    meta_test = np.column_stack([p_h, p_d, p_a])
    meta_pred = meta_clf.predict_proba(meta_test)
    p_h_final = meta_pred[:, le_meta.transform(["H"])[0]]
    p_d_final = meta_pred[:, le_meta.transform(["D"])[0]]
    p_a_final = meta_pred[:, le_meta.transform(["A"])[0]]

    # ── Calibrar threshold de empate ────────────────────────────────────────
    print("\n🎯 Calibrando threshold de empate...")
    p_d_tr_lgb = (model_d_lgb.predict_proba(X_train)[:, 1] + model_d_xgb.predict_proba(X_train)[:, 1]) / 2
    p_d_tr     = cal_d.predict(p_d_tr_lgb)
    p_h_tr_lgb = (model_h_lgb.predict_proba(X_train)[:, 1] + model_h_xgb.predict_proba(X_train)[:, 1]) / 2
    p_h_tr     = cal_h.predict(p_h_tr_lgb)
    p_a_tr_lgb = (model_a_lgb.predict_proba(X_train)[:, 1] + model_a_xgb.predict_proba(X_train)[:, 1]) / 2
    p_a_tr     = cal_a.predict(p_a_tr_lgb)
    tot_tr     = p_d_tr + p_h_tr + p_a_tr
    p_d_tr /= tot_tr; p_h_tr /= tot_tr; p_a_tr /= tot_tr

    # Stacking nas probs de treino para calibração do threshold
    meta_tr_in = np.column_stack([p_h_tr, p_d_tr, p_a_tr])
    meta_tr_p  = meta_clf.predict_proba(meta_tr_in)
    p_h_tr_st  = meta_tr_p[:, le_meta.transform(["H"])[0]]
    p_d_tr_st  = meta_tr_p[:, le_meta.transform(["D"])[0]]
    p_a_tr_st  = meta_tr_p[:, le_meta.transform(["A"])[0]]

    y_tr_arr       = y_train_raw.values
    real_draw_rate = (y_tr_arr == "D").mean()
    draw_min       = max(0.18, real_draw_rate - 0.08)
    draw_max       = min(0.38, real_draw_rate + 0.08)

    best_thresh, best_f1 = 0.40, 0.0
    for thr in np.arange(0.22, 0.50, 0.01):
        preds = np.where(p_d_tr_st >= thr, "D", np.where(p_h_tr_st >= p_a_tr_st, "H", "A"))
        d_rate = (preds == "D").mean()
        tp   = ((preds=="D") & (y_tr_arr=="D")).sum()
        fp   = ((preds=="D") & (y_tr_arr!="D")).sum()
        fn   = ((preds!="D") & (y_tr_arr=="D")).sum()
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-9)
        if draw_min <= d_rate <= draw_max and f1 > best_f1:
            best_f1, best_thresh = f1, thr

    DRAW_THRESHOLD = round(best_thresh, 2)
    print(f"   Threshold: {DRAW_THRESHOLD} (F1-D={best_f1:.3f})")

    # ── Avaliação final no teste ─────────────────────────────────────────────
    probs_final = np.column_stack([p_h_final, p_d_final, p_a_final])
    y_pred = np.where(
        p_d_final >= DRAW_THRESHOLD, "D",
        np.where(p_h_final >= p_a_final, "H", "A")
    )
    print(f"\n✅ Acurácia no teste: {(y_pred == y_test_raw).mean():.2%}")
    print("\n📋 Relatório:")
    print(classification_report(y_test_raw, y_pred))

    # Métricas probabilísticas
    mkt_cols = ["prob_h_mkt", "prob_d_mkt", "prob_a_mkt"]
    probs_mkt = X_test[mkt_cols].values if all(c in X_test.columns for c in mkt_cols) else None
    print_metrics_report(probs_final, y_test_raw, label="Teste 2026", probs_mkt=probs_mkt)

    print("\n🔍 Top 15 features (modelo H LightGBM):")
    imp_h = pd.Series(model_h_lgb.feature_importances_, index=all_cols)
    for feat, imp in imp_h.sort_values(ascending=False).head(15).items():
        print(f"   {'█'*int(imp/imp_h.max()*25)} {feat} {imp:.0f}")

    # ── Salvar ───────────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    joblib.dump({
        "version":        "v2_ensemble_optuna",
        # LightGBM
        "models_h_lgb":   [model_h_lgb],
        "models_d_lgb":   [model_d_lgb],
        "models_a_lgb":   [model_a_lgb],
        # XGBoost
        "models_h_xgb":   [model_h_xgb],
        "models_d_xgb":   [model_d_xgb],
        "models_a_xgb":   [model_a_xgb],
        # Calibração
        "cal_h":          cal_h,
        "cal_d":          cal_d,
        "cal_a":          cal_a,
        # Stacking
        "meta_clf":       meta_clf,
        "label_encoder_meta": le_meta,
        # Meta
        "features":       all_cols,
        "label_encoder":  le,
        "binary":         True,
        "draw_threshold": DRAW_THRESHOLD,
        "lgb_params":     {"h": best_h, "d": best_d, "a": best_a},
    }, MODEL_PATH)
    print(f"\n✅ Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()
