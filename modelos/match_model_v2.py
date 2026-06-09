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
    # Odds de mercado
    "prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
    "odds_draw_factor", "odds_home_away_ratio", "market_entropy",
    # MELHORIA 3: Libertadores e tendência de posição
    # Estas colunas serão preenchidas com 0 caso não existam no CSV histórico
    "home_joga_libertadores", "away_joga_libertadores",
    "home_pos_trend", "away_pos_trend", "pos_trend_diff",
    # MELHORIA 5: features de equilíbrio — ajudam o modelo a detectar empates
    "aprov_equilibrio", "h2h_draw_dominance",
]

# MELHORIA 2: temporada 2026 com peso 2x no treino
# O treino usa dados até 2025 (2026 é teste), então o peso mais alto
# vai para 2025, que é o proxy mais próximo do comportamento atual.
SEASON_WEIGHTS = {
    2026: 5.0,   # ← novo: dados de 2026 já disputados valem muito
    2025: 4.0,   # ← aumentado (era implícito como o mais alto)
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

# Times que disputam Libertadores 2026
# Usados para criar a feature no feature_engineering.py
LIBERTADORES_TIMES_2026 = {
    "CR Flamengo", "Fluminense FC", "CA Mineiro",
    "São Paulo FC", "SC Internacional",
}


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
    X_["elo_vs_mkt_h"]        = X_["elo_similarity"] - X_["prob_h_mkt"]
    X_["elo_vs_mkt_a"]        = (1 - X_["elo_similarity"]) - X_["prob_a_mkt"]
    return X_


def get_temporal_weights(seasons: pd.Series) -> np.ndarray:
    return seasons.map(SEASON_WEIGHTS).fillna(0.3).values


def train_binary(X_tr, y_tr, X_te, y_te, temporal_w, label, pos_weight=1.0):
    class_w = np.where(y_tr == 1, pos_weight, 1.0)
    sw      = class_w * temporal_w
    sw      = sw / sw.mean()  # normalizar para média 1

    model = lgb.LGBMClassifier(
        n_estimators=300,        # reduzido de 400 — menos risco de overfit
        max_depth=4,             # reduzido de 5 — mais conservador
        learning_rate=0.02,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=25,    # aumentado de 15 — mais regularização
        reg_alpha=0.4,           # aumentado de 0.2 — mais L1
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
    df = pd.read_csv(DATA_PATH)

    # Corrigir nomes de colunas do merge
    if "season_x" in df.columns:
        df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns:
        df = df.rename(columns={"result_x": "result"})

    # Preencher odds ausentes com mediana (jogos sem odds ainda entram)
    odds_cols = ["prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
                 "odds_draw_factor", "odds_home_away_ratio", "market_entropy"]
    for col in odds_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # MELHORIA 5: Preencher novas features com 0 se não existirem no CSV histórico
    new_feat_cols = [
        "home_joga_libertadores", "away_joga_libertadores",
        "home_pos_trend", "away_pos_trend", "pos_trend_diff",
        "aprov_equilibrio", "h2h_draw_dominance",
    ]
    for col in new_feat_cols:
        if col not in df.columns:
            df[col] = 0.0
            print(f"   ⚠️  Coluna '{col}' não encontrada — preenchida com 0")

    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas | distribuição: {df['result'].value_counts().to_dict()}")
    print(f"   Temporadas: {sorted(df['season'].unique())}")

    # Garantir que todas as colunas de features existem
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    X = add_derived(df[available_feats])
    all_cols = list(X.columns)
    print(f"   Total features: {len(all_cols)}")

    # Split: treino 2004–2025 (inclusive), teste = apenas 2026
    # 2025 entra no treino para maximizar dados recentes disponíveis ao modelo.
    # A CV temporal (5 folds) é a métrica primária de generalização.
    # O teste em 2026 serve como validação final pontual.
    train_mask = df["season"].isin(range(2004, 2026))
    test_mask  = df["season"].isin([2026])

    # Fallback: se não há dados de 2026, usa a última temporada disponível
    if test_mask.sum() < 10:
        available_seasons = sorted(df["season"].unique())
        test_seasons  = available_seasons[-1:]
        train_seasons = available_seasons[:-1]
        train_mask = df["season"].isin(train_seasons)
        test_mask  = df["season"].isin(test_seasons)
        print(f"   ⚠️  Fallback — treino: {train_seasons} | teste: {test_seasons}")

    X_train, X_test  = X[train_mask], X[test_mask]
    y_train_raw      = df["result"][train_mask]
    y_test_raw       = df["result"][test_mask].values
    seasons_train    = df["season"][train_mask]
    temporal_w_train = get_temporal_weights(seasons_train)

    print(f"\n   Treino: {len(X_train)} jogos")
    print(f"   Teste:  {len(X_test)} jogos")
    print(f"   Dist treino: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"   Dist teste:  {pd.Series(y_test_raw).value_counts().to_dict()}")

    # Pesos de classe baseados em temporadas recentes
    recent = df[df["season"].isin([2023, 2024, 2025])]["result"]
    freq   = recent.value_counts(normalize=True)
    pw_h   = 1.0
    # pw_d calibrado em 1.8: intermediário entre 1.5 (subprevia D) e 2.2 (superprevia D)
    pw_d   = (freq.get("H", 0.45) / freq.get("D", 0.25)) * 1.8
    pw_a   = freq.get("H", 0.45) / freq.get("A", 0.30)
    print(f"\n   Pesos classe — H:{pw_h:.2f} | D:{pw_d:.2f} | A:{pw_a:.2f}")
    print(f"   Peso temporal — 2025={SEASON_WEIGHTS.get(2025,4.0)}x | 2012={SEASON_WEIGHTS.get(2012,0.3)}x")

    # ── Cross-validação temporal ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    print("\n🕐 Cross-validação temporal:")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval    = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr_raw      = y_train_raw.iloc[tr_idx]
        yval_raw     = y_train_raw.iloc[val_idx]
        seas_tr      = seasons_train.iloc[tr_idx]
        tw           = get_temporal_weights(seas_tr)

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

        pred_idx    = np.stack([ph, pd_, pa], axis=1).argmax(axis=1)
        pred_map    = {0: "H", 1: "D", 2: "A"}
        y_pred_fold = np.array([pred_map[i] for i in pred_idx])

        score = (y_pred_fold == yval_raw.values).mean()
        cv_scores.append(score)
        dist  = pd.Series(y_pred_fold).value_counts().to_dict()
        print(f"   Fold {fold}: {score:.2%} | previsões: {dist}")

    print(f"   Média CV: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    # ── Treinar modelos finais ──
    print("\n🔧 Treinando 3 modelos binários finais com peso temporal...")
    model_h, cal_h = train_binary(
        X_train, (y_train_raw == "H").astype(int),
        X_test,  (y_test_raw  == "H").astype(int),
        temporal_w_train, "H", pw_h
    )
    model_d, cal_d = train_binary(
        X_train, (y_train_raw == "D").astype(int),
        X_test,  (y_test_raw  == "D").astype(int),
        temporal_w_train, "D", pw_d
    )
    model_a, cal_a = train_binary(
        X_train, (y_train_raw == "A").astype(int),
        X_test,  (y_test_raw  == "A").astype(int),
        temporal_w_train, "A", pw_a
    )

    # ── Combinar e avaliar ──
    print("\n📐 Combinando probabilidades...")
    p_h  = cal_h.predict(model_h.predict_proba(X_test)[:, 1])
    p_d  = cal_d.predict(model_d.predict_proba(X_test)[:, 1])
    p_a  = cal_a.predict(model_a.predict_proba(X_test)[:, 1])
    total = p_h + p_d + p_a
    p_h /= total; p_d /= total; p_a /= total

    # ── Calibrar threshold de empate com restrição de realismo ──────────────
    # Problema anterior: busca sem restrição escolhia threshold que previa
    # 60% dos jogos como empate (irreal — Brasileirão tem ~26-28% de empates).
    # Solução: só considerar thresholds que produzem entre 20% e 35% de D's
    # no treino, mantendo a distribuição compatível com a realidade.
    print("\n🎯 Calibrando threshold de empate (com restrição de realismo)...")

    p_d_train = cal_d.predict(model_d.predict_proba(X_train)[:, 1])
    p_h_train = cal_h.predict(model_h.predict_proba(X_train)[:, 1])
    p_a_train = cal_a.predict(model_a.predict_proba(X_train)[:, 1])
    tot_train  = p_d_train + p_h_train + p_a_train
    p_d_train /= tot_train; p_h_train /= tot_train; p_a_train /= tot_train

    n_train        = len(y_train_raw)
    y_train_arr    = y_train_raw.values
    real_draw_rate = (y_train_arr == "D").mean()

    # Faixa realista: ±8 pontos percentuais em torno da taxa real de empates
    draw_min = max(0.18, real_draw_rate - 0.08)
    draw_max = min(0.38, real_draw_rate + 0.08)
    print(f"   Taxa real de empates no treino: {real_draw_rate:.1%}")
    print(f"   Faixa permitida de D's previstos: {draw_min:.1%} – {draw_max:.1%}")

    best_thresh, best_f1 = 0.35, 0.0   # fallback conservador
    thresh_report = []
    for thr in np.arange(0.22, 0.46, 0.01):
        preds  = np.where(p_d_train >= thr, "D",
                          np.where(p_h_train >= p_a_train, "H", "A"))
        n_d    = (preds == "D").sum()
        d_rate = n_d / n_train

        tp   = ((preds == "D") & (y_train_arr == "D")).sum()
        fp   = ((preds == "D") & (y_train_arr != "D")).sum()
        fn   = ((preds != "D") & (y_train_arr == "D")).sum()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        acc  = (preds == y_train_arr).mean()

        in_range = draw_min <= d_rate <= draw_max
        thresh_report.append((thr, n_d, d_rate, rec, prec, f1, acc, in_range))

        if in_range and f1 > best_f1:
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
    print(f"\n   ✅ Threshold selecionado: {DRAW_THRESHOLD}  "
          f"(F1-D={best_f1:.3f} | dentro da faixa realista)")

    # ── Aplicar threshold no teste ────────────────────────────────────────────
    def apply_threshold(ph, pd_, pa, thr):
        return np.where(pd_ >= thr, "D", np.where(ph >= pa, "H", "A"))

    probs_stack   = np.stack([p_h, p_d, p_a], axis=1)
    y_pred_argmax = np.array(["H", "D", "A"])[probs_stack.argmax(axis=1)]
    y_pred        = apply_threshold(p_h, p_d, p_a, DRAW_THRESHOLD)

    acc_argmax = (y_pred_argmax == y_test_raw).mean()
    acc_thresh = (y_pred == y_test_raw).mean()
    print(f"\n   Acurácia argmax puro:          {acc_argmax:.2%}")
    print(f"   Acurácia threshold (D≥{DRAW_THRESHOLD}):  {acc_thresh:.2%}")

    # ── Análise de curva threshold no teste ───────────────────────────────────
    n_test = len(y_test_raw)
    print(f"\n📈 Análise de threshold no teste "
          f"(n={n_test} | empates reais={( y_test_raw=='D').sum()}):")
    print(f"   {'Thr':>5} {'#D prev':>8} {'%D':>5} {'Recall-D':>9} "
          f"{'Prec-D':>8} {'F1-D':>7} {'Acc':>7}")
    print(f"   {'-'*57}")
    for thr in np.arange(0.22, 0.46, 0.02):
        preds = apply_threshold(p_h, p_d, p_a, thr)
        tp    = ((preds == "D") & (y_test_raw == "D")).sum()
        fp    = ((preds == "D") & (y_test_raw != "D")).sum()
        fn    = ((preds != "D") & (y_test_raw == "D")).sum()
        prec  = tp / max(tp + fp, 1)
        rec   = tp / max(tp + fn, 1)
        f1    = 2 * prec * rec / max(prec + rec, 1e-9)
        n_d   = (preds == "D").sum()
        d_pct = n_d / n_test
        acc   = (preds == y_test_raw).mean()
        marker = " ◄" if abs(thr - DRAW_THRESHOLD) < 0.005 else ""
        print(f"   {thr:>5.2f} {n_d:>8} {d_pct:>5.1%} {rec:>9.2%} "
              f"{prec:>8.2%} {f1:>7.3f} {acc:>7.2%}{marker}")

    acc = (y_pred == y_test_raw).mean()
    print(f"\n✅ Acurácia no teste: {acc:.2%}")
    print("\n📋 Relatório:")
    print(classification_report(y_test_raw, y_pred))

    print("📊 Acurácia por resultado:")
    for res in ["H", "D", "A"]:
        mask   = y_test_raw == res
        a      = (y_pred[mask] == y_test_raw[mask]).mean() if mask.sum() > 0 else 0
        n_pred = (y_pred == res).sum()
        print(f"   {res}: acerto={a:.1%} | previu {n_pred}x de {mask.sum()}")

    print("\n📐 Calibração final:")
    print(f"   H: previsto={p_h.mean():.3f} | real={(y_test_raw=='H').mean():.3f}")
    print(f"   D: previsto={p_d.mean():.3f} | real={(y_test_raw=='D').mean():.3f}")
    print(f"   A: previsto={p_a.mean():.3f} | real={(y_test_raw=='A').mean():.3f}")

    print("\n🔍 Top 15 features (modelo H):")
    imp_h = pd.Series(model_h.feature_importances_, index=all_cols)
    for feat, imp in imp_h.sort_values(ascending=False).head(15).items():
        bar = "█" * int(imp / imp_h.max() * 25)
        print(f"   {feat:<35} {bar} {imp:.0f}")

    print("\n🔍 Top 15 features (modelo D):")
    imp_d = pd.Series(model_d.feature_importances_, index=all_cols)
    for feat, imp in imp_d.sort_values(ascending=False).head(15).items():
        bar = "█" * int(imp / imp_d.max() * 25)
        print(f"   {feat:<35} {bar} {imp:.0f}")

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    joblib.dump({
        "model_h":        model_h, "cal_h": cal_h,
        "model_d":        model_d, "cal_d": cal_d,
        "model_a":        model_a, "cal_a": cal_a,
        "features":       all_cols,
        "label_encoder":  le,
        "binary":         True,
        "draw_threshold": DRAW_THRESHOLD,   # ← salvo para uso no season_model
    }, MODEL_PATH)
    print(f"\n✅ Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()