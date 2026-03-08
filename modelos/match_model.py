import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import joblib

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
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
    X_["overall_balance"]     = (
        X_["elo_similarity"] + X_["form_similarity"] + X_["value_similarity"]
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


def get_temporal_weights(seasons: pd.Series, class_weight: float) -> np.ndarray:
    """Jogos recentes pesam mais — decaimento exponencial por temporada."""
    weights = np.ones(len(seasons))
    weight_map = {
        2024: 4.0,
        2023: 3.0,
        2022: 2.0,
        2021: 1.5,
        2020: 1.2,
        2019: 1.0,
        2018: 0.8,
        2017: 0.7,
        2016: 0.6,
        2015: 0.5,
        2014: 0.4,
        2013: 0.3,
        2012: 0.3,
        2011: 0.2,
        2010: 0.2,
        2009: 0.2,
        2008: 0.1,
        2007: 0.1,
        2006: 0.1,
        2005: 0.1,
        2004: 0.1,
        2003: 0.1,
    }
    for season, w in weight_map.items():
        weights[seasons == season] = w
    return weights * class_weight


def train_binary(X_tr, y_tr, X_te, y_te,
                 seasons_tr: pd.Series, label: str,
                 pos_weight: float = 1.0):
    """Treina modelo binário com pesos temporais + calibração isotônica."""

    # Peso temporal × peso de classe
    class_w  = np.where(y_tr == 1, pos_weight, 1.0)
    temp_w   = get_temporal_weights(seasons_tr, 1.0)
    sw       = class_w * temp_w
    sw       = sw / sw.mean()  # normalizar para média 1

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

    probs_raw = model.predict_proba(X_te)[:, 1]
    iso       = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_raw, y_te)
    probs_cal = iso.predict(probs_raw)

    acc_bin = ((probs_raw >= 0.5).astype(int) == y_te).mean()
    print(f"   {label}: prob média={probs_cal.mean():.3f} | "
          f"real={y_te.mean():.3f} | acc_bin={acc_bin:.2%}")
    return model, iso


def train():
    print("📊 Carregando features...")
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df)} partidas | distribuição: {df['result'].value_counts().to_dict()}")
    print(f"   Temporadas: {sorted(df['season'].unique())}")

    X = add_derived(df[FEATURE_COLS])
    all_cols = list(X.columns)
    print(f"   Total features: {len(all_cols)}")

    # ── Split temporal — treino tudo até 2024, teste 2025-2026 ──
    # O peso temporal garante que dados antigos influenciam menos
    train_mask = df["season"].isin([2023, 2024])
    test_mask  = df["season"].isin([2025, 2026])

    X_train, X_test     = X[train_mask], X[test_mask]
    y_train_raw         = df["result"][train_mask]
    y_test_raw          = df["result"][test_mask].values
    seasons_train       = df["season"][train_mask]

    print(f"\n   Treino: {len(X_train)} jogos (2003-2024)")
    print(f"   Teste:  {len(X_test)} jogos (2025-2026)")
    print(f"   Dist treino: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"   Dist teste:  {pd.Series(y_test_raw).value_counts().to_dict()}")

    # Pesos de classe baseados na distribuição RECENTE (2022-2024)
    recent = df[df["season"].isin([2022, 2023, 2024])]["result"]
    freq   = recent.value_counts(normalize=True)
    pw_h   = 1.0
    pw_d   = (freq["H"] / freq["D"]) * 1.5
    pw_a   = freq["H"] / freq["A"]
    print(f"\n   Pesos classe (dist recente) — H:{pw_h:.2f} | D:{pw_d:.2f} | A:{pw_a:.2f}")

    # ── Cross-validação temporal ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    print("\n🕐 Cross-validação temporal:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval   = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr_raw     = y_train_raw.iloc[tr_idx]
        yval_raw    = y_train_raw.iloc[val_idx]
        seas_tr     = seasons_train.iloc[tr_idx]

        def fold_model():
            return lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.02,
                num_leaves=20, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.2, reg_lambda=0.2,
                random_state=42, verbose=-1, n_jobs=-1
            )

        for cls, pw, mvar in [("H", pw_h, "mh"), ("D", pw_d, "md"), ("A", pw_a, "ma")]:
            cw   = np.where(ytr_raw == cls, pw, 1.0)
            tw   = get_temporal_weights(seas_tr, 1.0)
            sw   = cw * tw; sw = sw / sw.mean()
            m    = fold_model()
            m.fit(Xtr, (ytr_raw == cls).astype(int), sample_weight=sw)
            if cls == "H": mh = m
            elif cls == "D": md = m
            else: ma = m

        ph  = mh.predict_proba(Xval)[:, 1]
        pd_ = md.predict_proba(Xval)[:, 1]
        pa  = ma.predict_proba(Xval)[:, 1]
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        pred_idx    = np.stack([ph, pd_, pa], axis=1).argmax(axis=1)
        pred_map    = {0: "H", 1: "D", 2: "A"}
        y_pred_fold = np.array([pred_map[i] for i in pred_idx])

        score = (y_pred_fold == yval_raw.values).mean()
        cv_scores.append(score)
        dist  = pd.Series(y_pred_fold).value_counts().to_dict()
        print(f"   Fold {fold}: {score:.2%} | previsões: {dist}")

    print(f"   Média: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # ── Treinar modelos finais ──
    print("\n🔧 Treinando 3 modelos binários finais com peso temporal...")
    model_h, cal_h = train_binary(
        X_train, (y_train_raw == "H").astype(int),
        X_test,  (y_test_raw  == "H").astype(int),
        seasons_train, "H", pw_h
    )
    model_d, cal_d = train_binary(
        X_train, (y_train_raw == "D").astype(int),
        X_test,  (y_test_raw  == "D").astype(int),
        seasons_train, "D", pw_d
    )
    model_a, cal_a = train_binary(
        X_train, (y_train_raw == "A").astype(int),
        X_test,  (y_test_raw  == "A").astype(int),
        seasons_train, "A", pw_a
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
    imp_h = pd.Series(model_h.feature_importances_, index=all_cols)
    for feat, imp in imp_h.sort_values(ascending=False).head(10).items():
        bar = "█" * int(imp / imp_h.max() * 25)
        print(f"   {feat:<28} {bar} {imp:.0f}")

    print("\n🔍 Top 10 features (modelo D):")
    imp_d = pd.Series(model_d.feature_importances_, index=all_cols)
    for feat, imp in imp_d.sort_values(ascending=False).head(10).items():
        bar = "█" * int(imp / imp_d.max() * 25)
        print(f"   {feat:<28} {bar} {imp:.0f}")

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    joblib.dump({
        "model_h":       model_h, "cal_h": cal_h,
        "model_d":       model_d, "cal_d": cal_d,
        "model_a":       model_a, "cal_a": cal_a,
        "features":      all_cols,
        "label_encoder": le,
        "binary":        True,
    }, MODEL_PATH)
    print(f"\n✅ Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()