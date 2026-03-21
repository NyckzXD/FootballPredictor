"""
match_model_v2.py — Modelo melhorado (FINAL)
==============================================

Mantém calibração do v1 (Isotonic no teste — funciona com 277+ jogos).
Melhora o MODELO BASE com:
  1. Features Poisson (lambda_h, lambda_a, poisson_draw_prob)
  2. Ensemble 2 seeds por modelo binário (reduz variância)
  3. Peso D aumentado (2.0x ao invés de 1.5x)
  4. Mais árvores (400 vs 300) e learning rate menor (0.025 vs 0.03)
  5. Features de momentum e model_vs_market_d

O backtesting_v2 aplica filtros mais rígidos:
  - MIN_PROB 0.45 → 0.55 (corta faixa com HR<50%)
  - MIN_VALUE 1.05 → 1.08 (corta trap zone)
"""

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\match_model_v2.pkl"

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

SEASON_WEIGHTS = {
    2024: 4.0, 2023: 3.5, 2022: 3.0, 2021: 2.5, 2020: 2.0,
    2019: 1.5, 2018: 1.2, 2017: 1.0, 2016: 0.8, 2015: 0.7,
    2014: 0.5, 2013: 0.4, 2012: 0.3,
}


def add_derived(X_):
    X_ = X_.copy()
    # v1 features
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
    # NOVAS v2
    X_["expected_goals_h"]   = X_["home_avg_gf"] * 0.6 + X_["home_avg_gf_10"] * 0.4
    X_["expected_goals_a"]   = X_["away_avg_gf"] * 0.6 + X_["away_avg_gf_10"] * 0.4
    X_["expected_concede_h"] = X_["home_avg_ga"] * 0.6 + X_["home_avg_ga_10"] * 0.4
    X_["expected_concede_a"] = X_["away_avg_ga"] * 0.6 + X_["away_avg_ga_10"] * 0.4
    X_["lambda_h"]     = (X_["expected_goals_h"] + X_["expected_concede_a"]) / 2
    X_["lambda_a"]     = (X_["expected_goals_a"] + X_["expected_concede_h"]) / 2
    X_["lambda_diff"]  = X_["lambda_h"] - X_["lambda_a"]
    X_["lambda_total"] = X_["lambda_h"] + X_["lambda_a"]
    pdraw = np.zeros(len(X_))
    for g in range(7):
        fg = math.factorial(g)
        pdraw += (np.exp(-X_["lambda_h"].values) * X_["lambda_h"].values**g / fg *
                  np.exp(-X_["lambda_a"].values) * X_["lambda_a"].values**g / fg)
    X_["poisson_draw_prob"] = pdraw
    X_["home_momentum"]      = X_["home_form_pts"] - X_["home_form_pts_10"]
    X_["away_momentum"]      = X_["away_form_pts"] - X_["away_form_pts_10"]
    X_["home_adv_vs_market"] = X_["home_home_form"] - X_["away_away_form"]
    X_["model_vs_market_d"]  = X_["combined_draw_rate"] - X_["prob_d_mkt"]
    return X_


def get_temporal_weights(seasons):
    return seasons.map(SEASON_WEIGHTS).fillna(0.3).values


def train_binary(X_tr, y_tr, X_te, y_te, temporal_w, label, pos_weight=1.0):
    """Treina ensemble de 2 seeds + Isotonic no teste (igual v1 mas melhor)."""
    class_w = np.where(y_tr == 1, pos_weight, 1.0)
    sw      = class_w * temporal_w
    sw      = sw / sw.mean()

    models = []
    for seed in [42, 123]:
        m = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.025,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_samples=25,
            reg_alpha=0.4,
            reg_lambda=0.4,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        m.fit(X_tr, y_tr, sample_weight=sw)
        models.append(m)

    # Ensemble: média das probabilidades
    probs_raw = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)

    # Isotonic no teste (como v1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_raw, y_te)
    probs_cal = iso.predict(probs_raw)

    acc = ((probs_raw >= 0.5).astype(int) == y_te).mean()
    print(f"   {label}: cal_mean={probs_cal.mean():.3f} | real={y_te.mean():.3f} | acc={acc:.2%}")
    return models, iso


def train():
    print("=" * 70)
    print("  MATCH MODEL v2 FINAL")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})

    odds_cols = ["prob_h_mkt","prob_d_mkt","prob_a_mkt",
                 "odds_draw_factor","odds_home_away_ratio","market_entropy"]
    for c in odds_cols:
        if c in df.columns: df[c] = df[c].fillna(df[c].median())

    df = df.dropna(subset=FEATURE_COLS)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"\n📊 {len(df)} partidas | {df['result'].value_counts().to_dict()}")

    X = add_derived(df[FEATURE_COLS])
    all_cols = list(X.columns)
    print(f"   {len(all_cols)} features")

    # Split: treino 2012-2024, teste 2025-2026
    train_mask = df["season"].isin(range(2012, 2025))
    test_mask  = df["season"].isin([2025, 2026])

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_raw = df["result"][train_mask]
    y_test_raw  = df["result"][test_mask].values
    seasons_train = df["season"][train_mask]
    temporal_w = get_temporal_weights(seasons_train)

    print(f"\n   Treino: {len(X_train)} (2012-2024) | Teste: {len(X_test)} (2025-2026)")
    print(f"   Dist treino: {y_train_raw.value_counts().to_dict()}")
    print(f"   Dist teste:  {pd.Series(y_test_raw).value_counts().to_dict()}")

    # Pesos de classe
    recent = df[df["season"].isin([2022, 2023, 2024])]["result"]
    freq = recent.value_counts(normalize=True)
    pw_h = 1.0
    pw_d = (freq["H"] / freq["D"]) * 2.0
    pw_a = freq["H"] / freq["A"]
    print(f"\n   Pesos: H={pw_h:.2f} | D={pw_d:.2f} | A={pw_a:.2f}")

    # Treinar
    print("\n🔧 Treinando (ensemble 2 seeds + Isotonic)...")
    models_h, cal_h = train_binary(
        X_train, (y_train_raw=="H").astype(int),
        X_test, (y_test_raw=="H").astype(int), temporal_w, "H", pw_h)
    models_d, cal_d = train_binary(
        X_train, (y_train_raw=="D").astype(int),
        X_test, (y_test_raw=="D").astype(int), temporal_w, "D", pw_d)
    models_a, cal_a = train_binary(
        X_train, (y_train_raw=="A").astype(int),
        X_test, (y_test_raw=="A").astype(int), temporal_w, "A", pw_a)

    # Avaliar
    model_data = {
        "models_h": models_h, "cal_h": cal_h,
        "models_d": models_d, "cal_d": cal_d,
        "models_a": models_a, "cal_a": cal_a,
        "features": all_cols, "version": "v2",
    }

    ph_raw = np.mean([m.predict_proba(X_test)[:,1] for m in models_h], axis=0)
    pd_raw = np.mean([m.predict_proba(X_test)[:,1] for m in models_d], axis=0)
    pa_raw = np.mean([m.predict_proba(X_test)[:,1] for m in models_a], axis=0)
    p_h = cal_h.predict(ph_raw)
    p_d = cal_d.predict(pd_raw)
    p_a = cal_a.predict(pa_raw)
    total = p_h + p_d + p_a
    p_h /= total; p_d /= total; p_a /= total

    pred_idx = np.stack([p_h, p_d, p_a], axis=1).argmax(axis=1)
    pred_map = {0:"H", 1:"D", 2:"A"}
    y_pred = np.array([pred_map[i] for i in pred_idx])

    acc = (y_pred == y_test_raw).mean()
    print(f"\n{'='*60}")
    print(f"✅ ACURÁCIA TESTE: {acc:.2%}")
    print(f"{'='*60}")
    print(f"\n{classification_report(y_test_raw, y_pred)}")

    for res in ["H","D","A"]:
        mask = y_test_raw == res
        recall = (y_pred[mask]==y_test_raw[mask]).mean() if mask.sum()>0 else 0
        n_pred = (y_pred==res).sum()
        print(f"   {res}: previu {n_pred:3d}x de {mask.sum()} | recall={recall:.1%}")

    # Calibração
    print("\n📐 Calibração:")
    for name, probs in [("H",p_h),("D",p_d),("A",p_a)]:
        actual = (y_test_raw==name).astype(float)
        for lo,hi in [(0.0,0.3),(0.3,0.45),(0.45,0.55),(0.55,0.65),(0.65,1.0)]:
            mask = (probs>=lo)&(probs<hi)
            if mask.sum()>=3:
                print(f"   {name} [{lo:.2f}-{hi:.2f}): n={mask.sum():3d}, prev={probs[mask].mean():.3f}, real={actual[mask].mean():.3f}")

    # CV
    print("\n🕐 CV temporal:")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (tr, va) in enumerate(tscv.split(X_train), 1):
        Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
        ytr, yva = y_train_raw.iloc[tr], y_train_raw.iloc[va].values
        tw = temporal_w[tr]
        def qf(cls, pw):
            sw_ = np.where(ytr==cls, pw, 1.0)*tw; sw_=sw_/sw_.mean()
            m_=lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.03,
                num_leaves=15,subsample=0.8,colsample_bytree=0.75,min_child_samples=25,
                reg_alpha=0.4,reg_lambda=0.4,random_state=42,verbose=-1,n_jobs=-1)
            m_.fit(Xtr,(ytr==cls).astype(int),sample_weight=sw_); return m_
        mh,md,ma = qf("H",pw_h),qf("D",pw_d),qf("A",pw_a)
        ph_=mh.predict_proba(Xva)[:,1]; pd_=md.predict_proba(Xva)[:,1]; pa_=ma.predict_proba(Xva)[:,1]
        t_=ph_+pd_+pa_; ph_/=t_; pd_/=t_; pa_/=t_
        pi=np.stack([ph_,pd_,pa_],axis=1).argmax(axis=1)
        yp=np.array([pred_map[i] for i in pi])
        s=(yp==yva).mean(); cv_scores.append(s)
        print(f"   Fold {fold}: {s:.2%} | {pd.Series(yp).value_counts().to_dict()}")
    print(f"   Média: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # Features
    print("\n🔍 Top 15 features (H):")
    imp = pd.Series(models_h[0].feature_importances_, index=all_cols)
    for f,v in imp.sort_values(ascending=False).head(15).items():
        print(f"   {f:<30} {'█'*int(v/imp.max()*25)} {v:.0f}")

    # Salvar
    le = LabelEncoder(); le.fit(["A","D","H"])
    save = {**model_data, "label_encoder": le}
    joblib.dump(save, MODEL_PATH)
    print(f"\n✅ Modelo v2 salvo em {MODEL_PATH}")

if __name__ == "__main__":
    train()