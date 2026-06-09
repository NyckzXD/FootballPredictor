import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import joblib

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
]

SEASON_WEIGHTS = {
    2026: 5.0,
    2025: 4.0,
    2024: 3.5,
    2023: 3.0,
    2022: 2.5,
    2021: 2.0,
    2020: 1.5,
    2019: 1.2,
    2018: 1.0,
    2017: 0.8,
    2016: 0.7,
    2015: 0.6,
    2014: 0.5,
    2013: 0.4,
    2012: 0.3,
}


def add_derived(X_):
    X_ = X_.copy()
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
    return X_


def get_temporal_weights(seasons: pd.Series) -> np.ndarray:
    return seasons.map(SEASON_WEIGHTS).fillna(0.3).values


def train_binary(X_tr, y_tr, X_te, y_te, temporal_w, label, pos_weight=1.0):
    class_w = np.where(y_tr == 1, pos_weight, 1.0)
    sw      = class_w * temporal_w
    sw      = sw / sw.mean()

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=25,
        reg_alpha=0.4,
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
    print(f"   {label}: prob média={probs_cal.mean():.3f} | real={y_te.mean():.3f} | acc_bin={acc_bin:.2%}")
    return model, iso


def train():
    print("📊 Carregando features...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})

    # Preencher odds ausentes com mediana
    for col in ["prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
                "odds_draw_factor", "odds_home_away_ratio", "market_entropy"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Preencher novas features com 0 se não existirem no CSV
    for col in ["home_joga_libertadores", "away_joga_libertadores",
                "home_pos_trend", "away_pos_trend", "pos_trend_diff",
                "aprov_equilibrio", "h2h_draw_dominance"]:
        if col not in df.columns:
            df[col] = 0.0
            print(f"   ⚠️  Coluna '{col}' não encontrada — preenchida com 0")

    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas | distribuição: {df['result'].value_counts().to_dict()}")
    print(f"   Temporadas: {sorted(df['season'].unique())}")

    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    X        = add_derived(df[available_feats])
    all_cols = list(X.columns)
    print(f"   Total features: {len(all_cols)}")

    # Treino: 2004–2025 | Teste: 2026
    train_mask = df["season"].isin(range(2004, 2026))
    test_mask  = df["season"].isin([2026])
    if test_mask.sum() < 10:
        available_seasons = sorted(df["season"].unique())
        train_mask = df["season"].isin(available_seasons[:-1])
        test_mask  = df["season"].isin(available_seasons[-1:])
        print(f"   ⚠️  Fallback — teste: {available_seasons[-1:]}")

    X_train, X_test  = X[train_mask], X[test_mask]
    y_train_raw      = df["result"][train_mask]
    y_test_raw       = df["result"][test_mask].values
    seasons_train    = df["season"][train_mask]
    temporal_w_train = get_temporal_weights(seasons_train)

    print(f"\n   Treino: {len(X_train)} jogos")
    print(f"   Teste:  {len(X_test)} jogos")
    print(f"   Dist treino: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"   Dist teste:  {pd.Series(y_test_raw).value_counts().to_dict()}")

    # ── Pesos de classe ───────────────────────────────────────────────────────
    recent = df[df["season"].isin([2023, 2024, 2025])]["result"]
    freq   = recent.value_counts(normalize=True)
    pw_h   = 1.0
    pw_a   = freq.get("H", 0.45) / freq.get("A", 0.30)
    # pw_d FIXO em 2.0 — independente da distribuição do dataset.
    # A fórmula proporcional gerava ~3.5 com o Brasileirão, causando
    # D dominante nos folds de CV. 2.0 é o ponto equilibrado.
    pw_d   = 2.0
    print(f"\n   Pesos classe — H:{pw_h:.2f} | D:{pw_d:.2f} | A:{pw_a:.2f}")
    print(f"   Peso temporal — 2025={SEASON_WEIGHTS.get(2025,4.0)}x | 2012={SEASON_WEIGHTS.get(2012,0.3)}x")

    # ── Cross-validação temporal ──────────────────────────────────────────────
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    print("\n🕐 Cross-validação temporal:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr_raw   = y_train_raw.iloc[tr_idx]
        yval_raw  = y_train_raw.iloc[val_idx]
        tw        = get_temporal_weights(seasons_train.iloc[tr_idx])

        def fold_fit(cls, pw):
            cw = np.where(ytr_raw == cls, pw, 1.0)
            sw = cw * tw; sw = sw / sw.mean()
            m  = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.02,
                num_leaves=20, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=25, reg_alpha=0.4, reg_lambda=0.3,
                random_state=42, verbose=-1, n_jobs=-1)
            m.fit(Xtr, (ytr_raw == cls).astype(int), sample_weight=sw)
            return m

        mh = fold_fit("H", pw_h)
        md = fold_fit("D", pw_d)
        ma = fold_fit("A", pw_a)

        ph  = mh.predict_proba(Xval)[:, 1]
        pd_ = md.predict_proba(Xval)[:, 1]
        pa  = ma.predict_proba(Xval)[:, 1]
        tot = ph + pd_ + pa
        ph /= tot; pd_ /= tot; pa /= tot

        y_pred_fold = np.array(["H", "D", "A"])[
            np.stack([ph, pd_, pa], axis=1).argmax(axis=1)
        ]
        score = (y_pred_fold == yval_raw.values).mean()
        cv_scores.append(score)
        print(f"   Fold {fold}: {score:.2%} | previsões: {pd.Series(y_pred_fold).value_counts().to_dict()}")

    print(f"   Média CV: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # ── Treinar modelos finais ────────────────────────────────────────────────
    print("\n🔧 Treinando 3 modelos binários finais...")
    model_h, cal_h = train_binary(X_train, (y_train_raw=="H").astype(int),
                                   X_test,  (y_test_raw =="H").astype(int),
                                   temporal_w_train, "H", pw_h)
    model_d, cal_d = train_binary(X_train, (y_train_raw=="D").astype(int),
                                   X_test,  (y_test_raw =="D").astype(int),
                                   temporal_w_train, "D", pw_d)
    model_a, cal_a = train_binary(X_train, (y_train_raw=="A").astype(int),
                                   X_test,  (y_test_raw =="A").astype(int),
                                   temporal_w_train, "A", pw_a)

    # ── Combinar probabilidades ───────────────────────────────────────────────
    print("\n📐 Combinando probabilidades...")
    p_h = cal_h.predict(model_h.predict_proba(X_test)[:, 1])
    p_d = cal_d.predict(model_d.predict_proba(X_test)[:, 1])
    p_a = cal_a.predict(model_a.predict_proba(X_test)[:, 1])
    total = p_h + p_d + p_a
    p_h /= total; p_d /= total; p_a /= total

    # ── Calibrar threshold com restrição de realismo ──────────────────────────
    # Só aceita thresholds que produzem entre (real_rate ± 8%) de D's no treino
    print("\n🎯 Calibrando threshold de empate (com restrição de realismo)...")

    p_d_tr = cal_d.predict(model_d.predict_proba(X_train)[:, 1])
    p_h_tr = cal_h.predict(model_h.predict_proba(X_train)[:, 1])
    p_a_tr = cal_a.predict(model_a.predict_proba(X_train)[:, 1])
    tot_tr = p_d_tr + p_h_tr + p_a_tr
    p_d_tr /= tot_tr; p_h_tr /= tot_tr; p_a_tr /= tot_tr

    y_tr_arr       = y_train_raw.values
    real_draw_rate = (y_tr_arr == "D").mean()
    draw_min       = max(0.18, real_draw_rate - 0.08)
    draw_max       = min(0.38, real_draw_rate + 0.08)
    print(f"   Taxa real de empates no treino: {real_draw_rate:.1%}")
    print(f"   Faixa permitida de D's previstos: {draw_min:.1%} – {draw_max:.1%}")

    best_thresh, best_f1 = 0.40, 0.0
    thresh_report = []
    for thr in np.arange(0.22, 0.50, 0.01):
        preds  = np.where(p_d_tr >= thr, "D", np.where(p_h_tr >= p_a_tr, "H", "A"))
        n_d    = (preds == "D").sum()
        d_rate = n_d / len(y_tr_arr)
        tp     = ((preds=="D") & (y_tr_arr=="D")).sum()
        fp     = ((preds=="D") & (y_tr_arr!="D")).sum()
        fn     = ((preds!="D") & (y_tr_arr=="D")).sum()
        prec   = tp / max(tp+fp, 1)
        rec    = tp / max(tp+fn, 1)
        f1     = 2*prec*rec / max(prec+rec, 1e-9)
        acc    = (preds == y_tr_arr).mean()
        ok     = draw_min <= d_rate <= draw_max
        thresh_report.append((thr, n_d, d_rate, rec, prec, f1, acc, ok))
        if ok and f1 > best_f1:
            best_f1, best_thresh = f1, thr

    print(f"\n   {'Thr':>5} {'#D':>6} {'%D':>6} {'Recall':>7} {'Prec':>7} "
          f"{'F1-D':>7} {'Acc':>7} {'OK?':>5}")
    print(f"   {'-'*58}")
    for thr, n_d, d_rate, rec, prec, f1, acc, ok in thresh_report:
        ok_str = "  ✓" if ok else "  ✗"
        marker = " ◄ ótimo" if abs(thr - best_thresh) < 0.005 else ""
        print(f"   {thr:>5.2f} {n_d:>6} {d_rate:>6.1%} {rec:>7.2%} {prec:>7.2%} "
              f"{f1:>7.3f} {acc:>7.2%}{ok_str}{marker}")

    DRAW_THRESHOLD = round(best_thresh, 2)
    print(f"\n   ✅ Threshold selecionado: {DRAW_THRESHOLD} "
          f"(F1-D={best_f1:.3f} | dentro da faixa realista)")

    # ── Aplicar no teste e analisar curva ────────────────────────────────────
    def apply_threshold(ph, pd_, pa, thr):
        return np.where(pd_ >= thr, "D", np.where(ph >= pa, "H", "A"))

    y_pred_argmax = np.array(["H","D","A"])[
        np.stack([p_h, p_d, p_a], axis=1).argmax(axis=1)
    ]
    y_pred = apply_threshold(p_h, p_d, p_a, DRAW_THRESHOLD)

    print(f"\n   Acurácia argmax puro:          {(y_pred_argmax==y_test_raw).mean():.2%}")
    print(f"   Acurácia threshold (D≥{DRAW_THRESHOLD}):  {(y_pred==y_test_raw).mean():.2%}")

    n_test = len(y_test_raw)
    print(f"\n📈 Análise de threshold no teste "
          f"(n={n_test} | empates reais={(y_test_raw=='D').sum()}):")
    print(f"   {'Thr':>5} {'#D':>7} {'%D':>5} {'Recall-D':>9} "
          f"{'Prec-D':>8} {'F1-D':>7} {'Acc':>7}")
    print(f"   {'-'*55}")
    for thr in np.arange(0.22, 0.50, 0.02):
        preds = apply_threshold(p_h, p_d, p_a, thr)
        tp    = ((preds=="D") & (y_test_raw=="D")).sum()
        fp    = ((preds=="D") & (y_test_raw!="D")).sum()
        fn    = ((preds!="D") & (y_test_raw=="D")).sum()
        prec  = tp / max(tp+fp, 1)
        rec   = tp / max(tp+fn, 1)
        f1    = 2*prec*rec / max(prec+rec, 1e-9)
        n_d   = (preds=="D").sum()
        acc   = (preds==y_test_raw).mean()
        marker = " ◄" if abs(thr - DRAW_THRESHOLD) < 0.005 else ""
        print(f"   {thr:>5.2f} {n_d:>7} {n_d/n_test:>5.1%} {rec:>9.2%} "
              f"{prec:>8.2%} {f1:>7.3f} {acc:>7.2%}{marker}")

    print(f"\n✅ Acurácia no teste: {(y_pred==y_test_raw).mean():.2%}")
    print("\n📋 Relatório:")
    print(classification_report(y_test_raw, y_pred))

    print("📊 Acurácia por resultado:")
    for res in ["H", "D", "A"]:
        mask   = y_test_raw == res
        a      = (y_pred[mask]==y_test_raw[mask]).mean() if mask.sum()>0 else 0
        n_pred = (y_pred==res).sum()
        print(f"   {res}: acerto={a:.1%} | previu {n_pred}x de {mask.sum()}")

    print("\n📐 Calibração final:")
    print(f"   H: previsto={p_h.mean():.3f} | real={(y_test_raw=='H').mean():.3f}")
    print(f"   D: previsto={p_d.mean():.3f} | real={(y_test_raw=='D').mean():.3f}")
    print(f"   A: previsto={p_a.mean():.3f} | real={(y_test_raw=='A').mean():.3f}")

    print("\n🔍 Top 15 features (modelo H):")
    imp_h = pd.Series(model_h.feature_importances_, index=all_cols)
    for feat, imp in imp_h.sort_values(ascending=False).head(15).items():
        print(f"   {'█'*int(imp/imp_h.max()*25)} {feat} {imp:.0f}")

    print("\n🔍 Top 15 features (modelo D):")
    imp_d = pd.Series(model_d.feature_importances_, index=all_cols)
    for feat, imp in imp_d.sort_values(ascending=False).head(15).items():
        print(f"   {'█'*int(imp/imp_d.max()*25)} {feat} {imp:.0f}")

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    joblib.dump({
        "model_h": model_h, "cal_h": cal_h,
        "model_d": model_d, "cal_d": cal_d,
        "model_a": model_a, "cal_a": cal_a,
        "features":       all_cols,
        "label_encoder":  le,
        "binary":         True,
        "draw_threshold": DRAW_THRESHOLD,
    }, MODEL_PATH)
    print(f"\n✅ Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()