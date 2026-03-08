import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import joblib

DATA_PATH  = r"C:\PREDICTOR\scraping\data\processed\features.csv"
MODEL_PATH = r"C:\PREDICTOR\models\match_model.pkl"

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
]


def add_derived(X_):
    X_ = X_.copy()
    # Diferenças
    X_["form_diff"]       = X_["home_form_pts"]       - X_["away_form_pts"]
    X_["form_diff_10"]    = X_["home_form_pts_10"]    - X_["away_form_pts_10"]
    X_["gf_diff"]         = X_["home_avg_gf"]         - X_["away_avg_gf"]
    X_["ga_diff"]         = X_["home_avg_ga"]         - X_["away_avg_ga"]
    X_["win_rate_diff"]   = X_["home_win_rate"]       - X_["away_win_rate"]
    X_["aproveit_diff"]   = X_["home_aproveitamento"] - X_["away_aproveitamento"]
    X_["home_in_crisis"]  = (X_["home_form_pts"] < 0.5).astype(int)
    X_["away_in_form"]    = (X_["away_form_pts"] > 2.0).astype(int)

    # Features específicas para empate
    X_["elo_similarity"]      = 1 / (1 + np.abs(X_["elo_diff"]))
    X_["form_similarity"]     = 1 / (1 + np.abs(X_["form_diff"]))
    X_["value_similarity"]    = 1 / (1 + np.abs(X_["market_value_diff"]))
    X_["overall_balance"]     = (
        X_["elo_similarity"] +
        X_["form_similarity"] +
        X_["value_similarity"]
    ) / 3

    X_["home_draw_tendency"]  = X_["home_draw_rate"]
    X_["away_draw_tendency"]  = X_["away_draw_rate"]
    X_["combined_draw_rate"]  = (X_["home_draw_rate"] + X_["away_draw_rate"]) / 2

    X_["both_low_scoring"]    = (
        (X_["home_avg_gf"] < 1.2) & (X_["away_avg_gf"] < 1.2)
    ).astype(int)

    X_["both_good_defense"]   = (
        (X_["home_avg_ga"] < 1.0) & (X_["away_avg_ga"] < 1.0)
    ).astype(int)

    total_h2h                 = X_["h2h_home_wins"] + X_["h2h_away_wins"] + X_["h2h_draws"] + 1
    X_["h2h_draw_rate"]       = X_["h2h_draws"] / total_h2h
    X_["h2h_decisividade"]    = (X_["h2h_home_wins"] + X_["h2h_away_wins"]) / total_h2h
    X_["position_similarity"] = 1 / (1 + np.abs(X_["position_diff"]))

    return X_


def train_binary(X_tr, y_tr, X_te, y_te, label, pos_weight=1.0):
    """Treina modelo binário para uma classe com calibração isotônica."""
    sw = np.where(y_tr == 1, pos_weight, 1.0)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.3,
        reg_lambda=0.3,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=sw)

    # Calibrar com isotonic regression
    probs_raw = model.predict_proba(X_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_raw, y_te)
    probs_cal = iso.predict(probs_raw)

    acc_bin = ((probs_raw >= 0.5).astype(int) == y_te).mean()
    print(f"   {label}: prob média={probs_cal.mean():.3f} | real={y_te.mean():.3f} | acc_bin={acc_bin:.2%}")

    return model, iso


def train():
    print("📊 Carregando features...")
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df)} partidas | distribuição: {df['result'].value_counts().to_dict()}")

    X = add_derived(df[FEATURE_COLS])
    all_cols = list(X.columns)
    print(f"   Total features: {len(all_cols)}")

    # Split temporal
    train_mask = df["season"].isin([2023, 2024])
    test_mask  = df["season"].isin([2025, 2026])
    X_train, X_test = X[train_mask], X[test_mask]
    y_train_raw = df["result"][train_mask]
    y_test_raw  = df["result"][test_mask].values

    print(f"   Treino: {len(X_train)} (2023-2024)")
    print(f"   Teste:  {len(X_test)} (2025-2026)")
    print(f"   Dist treino: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"   Dist teste:  {pd.Series(y_test_raw).value_counts().to_dict()}")

    # Pesos por classe
    freq  = df["result"].value_counts(normalize=True)
    pw_h  = 1.0
    pw_d  = (freq["H"] / freq["D"]) * 1.5   # boost extra empate
    pw_a  = freq["H"] / freq["A"]
    print(f"\n   Pesos — H:{pw_h:.2f} | D:{pw_d:.2f} | A:{pw_a:.2f}")

    # ── Cross-validação temporal para diagnóstico ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    print("\n🕐 Cross-validação temporal:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr_raw   = y_train_raw.iloc[tr_idx]
        yval_raw  = y_train_raw.iloc[val_idx]

        sw_h = np.where(ytr_raw == "H", pw_h, 1.0)
        sw_d = np.where(ytr_raw == "D", pw_d, 1.0)
        sw_a = np.where(ytr_raw == "A", pw_a, 1.0)

        def quick_model(X_t, y_t, sw):
            m = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            m.fit(X_t, (y_t == y_t.unique()[0]).astype(int), sample_weight=sw)
            return m

        mh = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
             num_leaves=15, subsample=0.8, colsample_bytree=0.8,
             min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
             random_state=42, verbose=-1, n_jobs=-1)
        md = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
             num_leaves=15, subsample=0.8, colsample_bytree=0.8,
             min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
             random_state=42, verbose=-1, n_jobs=-1)
        ma = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
             num_leaves=15, subsample=0.8, colsample_bytree=0.8,
             min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
             random_state=42, verbose=-1, n_jobs=-1)

        mh.fit(Xtr, (ytr_raw == "H").astype(int),
               sample_weight=np.where(ytr_raw == "H", pw_h, 1.0))
        md.fit(Xtr, (ytr_raw == "D").astype(int),
               sample_weight=np.where(ytr_raw == "D", pw_d, 1.0))
        ma.fit(Xtr, (ytr_raw == "A").astype(int),
               sample_weight=np.where(ytr_raw == "A", pw_a, 1.0))

        ph = mh.predict_proba(Xval)[:, 1]
        pd_ = md.predict_proba(Xval)[:, 1]
        pa = ma.predict_proba(Xval)[:, 1]
        total = ph + pd_ + pa
        ph /= total; pd_ /= total; pa /= total

        pred_matrix = np.stack([ph, pd_, pa], axis=1)
        pred_idx    = pred_matrix.argmax(axis=1)
        pred_map    = {0: "H", 1: "D", 2: "A"}
        y_pred_fold = np.array([pred_map[i] for i in pred_idx])

        score = (y_pred_fold == yval_raw.values).mean()
        cv_scores.append(score)
        dist  = pd.Series(y_pred_fold).value_counts().to_dict()
        print(f"   Fold {fold}: {score:.2%} | previsões: {dist}")

    print(f"   Média: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # ── Treinar modelos finais ──
    print("\n🔧 Treinando 3 modelos binários finais...")
    model_h, cal_h = train_binary(
        X_train, (y_train_raw == "H").astype(int),
        X_test,  (y_test_raw  == "H").astype(int),
        "H", pw_h
    )
    model_d, cal_d = train_binary(
        X_train, (y_train_raw == "D").astype(int),
        X_test,  (y_test_raw  == "D").astype(int),
        "D", pw_d
    )
    model_a, cal_a = train_binary(
        X_train, (y_train_raw == "A").astype(int),
        X_test,  (y_test_raw  == "A").astype(int),
        "A", pw_a
    )

    # ── Combinar e avaliar ──
    print("\n📐 Combinando probabilidades...")
    p_h  = cal_h.predict(model_h.predict_proba(X_test)[:, 1])
    p_d  = cal_d.predict(model_d.predict_proba(X_test)[:, 1])
    p_a  = cal_a.predict(model_a.predict_proba(X_test)[:, 1])
    total = p_h + p_d + p_a
    p_h /= total; p_d /= total; p_a /= total

    probs_matrix = np.stack([p_h, p_d, p_a], axis=1)
    pred_idx     = probs_matrix.argmax(axis=1)
    pred_map     = {0: "H", 1: "D", 2: "A"}
    y_pred       = np.array([pred_map[i] for i in pred_idx])

    acc = (y_pred == y_test_raw).mean()
    print(f"\n✅ Acurácia no teste (2025-2026): {acc:.2%}")
    print("\n📋 Relatório:")
    print(classification_report(y_test_raw, y_pred))

    print("📊 Acurácia por resultado:")
    for res in ["H", "D", "A"]:
        mask   = y_test_raw == res
        a      = (y_pred[mask] == y_test_raw[mask]).mean()
        n_pred = (y_pred == res).sum()
        print(f"   {res}: acerto={a:.1%} | previu {n_pred}x de {mask.sum()}")

    print("\n📐 Calibração final:")
    print(f"   H: previsto={p_h.mean():.3f} | real={(y_test_raw=='H').mean():.3f}")
    print(f"   D: previsto={p_d.mean():.3f} | real={(y_test_raw=='D').mean():.3f}")
    print(f"   A: previsto={p_a.mean():.3f} | real={(y_test_raw=='A').mean():.3f}")

    print("\n🔍 Top 10 features (modelo H):")
    importances = pd.Series(model_h.feature_importances_, index=all_cols)
    for feat, imp in importances.sort_values(ascending=False).head(10).items():
        bar = "█" * int(imp / importances.max() * 25)
        print(f"   {feat:<28} {bar} {imp:.0f}")

    print("\n🔍 Top 10 features (modelo D):")
    importances_d = pd.Series(model_d.feature_importances_, index=all_cols)
    for feat, imp in importances_d.sort_values(ascending=False).head(10).items():
        bar = "█" * int(imp / importances_d.max() * 25)
        print(f"   {feat:<28} {bar} {imp:.0f}")

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    joblib.dump({
        "model_h": model_h, "cal_h": cal_h,
        "model_d": model_d, "cal_d": cal_d,
        "model_a": model_a, "cal_a": cal_a,
        "features":      all_cols,
        "label_encoder": le,
        "binary":        True,
    }, MODEL_PATH)
    print(f"\n✅ Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()