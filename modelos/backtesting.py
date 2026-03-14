import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\match_model.pkl"
OUTPUT_CSV = r"C:\PREDICTOR\REPO\scraping\data\external\backtesting_results.csv"
OUTPUT_PNG = r"C:\PREDICTOR\REPO\scraping\data\external\backtesting_chart.png"

# ── Configurações ─────────────────────────────────────────────────────────────
BANKROLL_INICIAL = 1000.0   # unidades monetárias
KELLY_FRACTION   = 0.25     # Kelly fracionado (25% do Kelly completo — mais conservador)
MIN_VALUE        = 1.05     # edge mínimo: prob_modelo * odd > 1.05
MIN_PROB         = 0.45     # só apostar se modelo tiver >= 45% de confiança
MAX_BET_FLAT     = 0.02     # flat: máximo 2% do bankroll por aposta
MAX_BET_KELLY    = 0.05     # kelly: máximo 5% do bankroll por aposta
TEST_SEASONS     = [2025, 2026]

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


def predict_probs(model_data, X):
    ph  = model_data["cal_h"].predict(model_data["model_h"].predict_proba(X)[:, 1])
    pd_ = model_data["cal_d"].predict(model_data["model_d"].predict_proba(X)[:, 1])
    pa  = model_data["cal_a"].predict(model_data["model_a"].predict_proba(X)[:, 1])
    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total


def kelly_bet(prob, odd, fraction=KELLY_FRACTION):
    """Kelly Criterion fracionado."""
    edge = prob * odd - 1
    if edge <= 0:
        return 0.0
    k = (prob - (1 - prob) / (odd - 1)) * fraction
    return max(0.0, k)


def run_backtest(df, model_data):
    records = []

    bankroll_flat  = BANKROLL_INICIAL
    bankroll_kelly = BANKROLL_INICIAL
    peak_flat      = BANKROLL_INICIAL
    peak_kelly     = BANKROLL_INICIAL
    max_dd_flat    = 0.0
    max_dd_kelly   = 0.0

    X = add_derived(df[FEATURE_COLS])
    ph_arr, pd_arr, pa_arr = predict_probs(model_data, X)

    for i, (_, row) in enumerate(df.iterrows()):
        ph, pd_, pa = ph_arr[i], pd_arr[i], pa_arr[i]
        result = row["result"]

        odd_h = row.get("odd_h", np.nan)
        odd_d = row.get("odd_d", np.nan)
        odd_a = row.get("odd_a", np.nan)

        # Verificar quais apostas têm value
        bets = []
        for prob, odd, outcome in [(ph, odd_h, "H"), (pd_, odd_d, "D"), (pa, odd_a, "A")]:
            if pd.isna(odd) or odd <= 1.0:
                continue
            value = prob * odd
            if value >= MIN_VALUE and prob >= MIN_PROB:
                bets.append({
                    "outcome": outcome, "prob": prob,
                    "odd": odd, "value": value,
                })

        # Só apostar na melhor aposta por jogo (maior value)
        if bets:
            bet = max(bets, key=lambda x: x["value"])
            won = bet["outcome"] == result

            # Flat
            stake_flat  = min(MAX_BET_FLAT * bankroll_flat, bankroll_flat)
            pl_flat     = stake_flat * (bet["odd"] - 1) if won else -stake_flat
            bankroll_flat += pl_flat
            peak_flat     = max(peak_flat, bankroll_flat)
            dd_flat       = (peak_flat - bankroll_flat) / peak_flat * 100
            max_dd_flat   = max(max_dd_flat, dd_flat)

            # Kelly fracionado
            k_pct       = kelly_bet(bet["prob"], bet["odd"])
            stake_kelly = min(k_pct * bankroll_kelly, MAX_BET_KELLY * bankroll_kelly)
            stake_kelly = max(stake_kelly, 0)
            pl_kelly    = stake_kelly * (bet["odd"] - 1) if won else -stake_kelly
            bankroll_kelly += pl_kelly
            peak_kelly     = max(peak_kelly, bankroll_kelly)
            dd_kelly       = (peak_kelly - bankroll_kelly) / peak_kelly * 100
            max_dd_kelly   = max(max_dd_kelly, dd_kelly)

            records.append({
                "date":           row.get("date", ""),
                "season":         row.get("season", ""),
                "home_team":      row.get("home_team", ""),
                "away_team":      row.get("away_team", ""),
                "result":         result,
                "bet_on":         bet["outcome"],
                "prob_model":     round(bet["prob"], 3),
                "odd":            round(bet["odd"], 2),
                "value":          round(bet["value"], 3),
                "won":            won,
                "stake_flat":     round(stake_flat, 2),
                "pl_flat":        round(pl_flat, 2),
                "bankroll_flat":  round(bankroll_flat, 2),
                "stake_kelly":    round(stake_kelly, 2),
                "pl_kelly":       round(pl_kelly, 2),
                "bankroll_kelly": round(bankroll_kelly, 2),
            })
        else:
            # Sem aposta — registrar bankroll atual
            records.append({
                "date":           row.get("date", ""),
                "season":         row.get("season", ""),
                "home_team":      row.get("home_team", ""),
                "away_team":      row.get("away_team", ""),
                "result":         result,
                "bet_on":         None,
                "prob_model":     None,
                "odd":            None,
                "value":          None,
                "won":            None,
                "stake_flat":     0,
                "pl_flat":        0,
                "bankroll_flat":  round(bankroll_flat, 2),
                "stake_kelly":    0,
                "pl_kelly":       0,
                "bankroll_kelly": round(bankroll_kelly, 2),
            })

    return pd.DataFrame(records), max_dd_flat, max_dd_kelly


def print_summary(df_bt, max_dd_flat, max_dd_kelly):
    bets = df_bt[df_bt["bet_on"].notna()].copy()

    print(f"\n{'='*60}")
    print(f"📊 BACKTESTING — BRASILEIRÃO {TEST_SEASONS}")
    print(f"{'='*60}")
    print(f"   Total de jogos analisados: {len(df_bt)}")
    print(f"   Apostas realizadas:        {len(bets)} ({len(bets)/len(df_bt):.1%})")

    if len(bets) == 0:
        print("   ⚠️  Nenhuma aposta realizada — reduza MIN_VALUE ou MIN_PROB")
        return

    hit_rate = bets["won"].mean()
    print(f"   Hit rate:                  {hit_rate:.1%}")

    # Por tipo de aposta
    print(f"\n   Apostas por resultado:")
    for outcome in ["H", "D", "A"]:
        sub = bets[bets["bet_on"] == outcome]
        if len(sub) == 0: continue
        print(f"   {outcome}: {len(sub)} apostas | hit={sub['won'].mean():.1%} | "
              f"odd média={sub['odd'].mean():.2f} | value médio={sub['value'].mean():.3f}")

    # ── Flat ──
    roi_flat = (df_bt["bankroll_flat"].iloc[-1] - BANKROLL_INICIAL) / BANKROLL_INICIAL * 100
    profit_flat = df_bt["bankroll_flat"].iloc[-1] - BANKROLL_INICIAL
    total_staked_flat = bets["stake_flat"].sum()
    yield_flat = bets["pl_flat"].sum() / total_staked_flat * 100 if total_staked_flat > 0 else 0

    print(f"\n{'─'*60}")
    print(f"🟦 FLAT (2% do bankroll por aposta):")
    print(f"   Bankroll final:  {df_bt['bankroll_flat'].iloc[-1]:.2f} "
          f"({'+'if profit_flat>=0 else ''}{profit_flat:.2f})")
    print(f"   ROI:             {roi_flat:+.2f}%")
    print(f"   Yield:           {yield_flat:+.2f}%")
    print(f"   Max Drawdown:    {max_dd_flat:.2f}%")
    print(f"   Total apostado:  {total_staked_flat:.2f}")

    # ── Kelly ──
    roi_kelly = (df_bt["bankroll_kelly"].iloc[-1] - BANKROLL_INICIAL) / BANKROLL_INICIAL * 100
    profit_kelly = df_bt["bankroll_kelly"].iloc[-1] - BANKROLL_INICIAL
    total_staked_kelly = bets["stake_kelly"].sum()
    yield_kelly = bets["pl_kelly"].sum() / total_staked_kelly * 100 if total_staked_kelly > 0 else 0

    print(f"\n🟩 KELLY FRACIONADO (25% Kelly, máx 5%):")
    print(f"   Bankroll final:  {df_bt['bankroll_kelly'].iloc[-1]:.2f} "
          f"({'+'if profit_kelly>=0 else ''}{profit_kelly:.2f})")
    print(f"   ROI:             {roi_kelly:+.2f}%")
    print(f"   Yield:           {yield_kelly:+.2f}%")
    print(f"   Max Drawdown:    {max_dd_kelly:.2f}%")
    print(f"   Total apostado:  {total_staked_kelly:.2f}")

    print(f"\n{'─'*60}")
    print(f"📌 Parâmetros usados:")
    print(f"   MIN_VALUE={MIN_VALUE} | MIN_PROB={MIN_PROB} | "
          f"KELLY_FRACTION={KELLY_FRACTION}")


def plot_results(df_bt, max_dd_flat=0, max_dd_kelly=0):
    bets = df_bt[df_bt["bet_on"].notna()].copy()
    bets = bets.reset_index(drop=True)

    fig = plt.figure(figsize=(18, 14), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"flat": "#4fc3f7", "kelly": "#81c784", "win": "#66bb6a",
              "loss": "#ef5350", "neutral": "#7986cb"}

    def ax_style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        return ax

    # 1. Evolução do bankroll
    ax1 = fig.add_subplot(gs[0, :2])
    ax_style(ax1, "📈 Evolução do Bankroll")
    idx = range(len(df_bt))
    ax1.plot(idx, df_bt["bankroll_flat"],  color=colors["flat"],  lw=2, label="Flat (2%)")
    ax1.plot(idx, df_bt["bankroll_kelly"], color=colors["kelly"], lw=2, label="Kelly (25%)")
    ax1.axhline(BANKROLL_INICIAL, color="#555577", ls="--", lw=1, alpha=0.7)
    ax1.fill_between(idx, df_bt["bankroll_flat"],  BANKROLL_INICIAL,
                     where=df_bt["bankroll_flat"] >= BANKROLL_INICIAL,
                     alpha=0.15, color=colors["flat"])
    ax1.fill_between(idx, df_bt["bankroll_flat"],  BANKROLL_INICIAL,
                     where=df_bt["bankroll_flat"] < BANKROLL_INICIAL,
                     alpha=0.15, color=colors["loss"])
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax1.set_xlabel("Jogos")
    ax1.set_ylabel("Bankroll")

    # 2. P&L acumulado por aposta (flat)
    ax2 = fig.add_subplot(gs[0, 2])
    ax_style(ax2, "💰 P&L Acumulado (Flat)")
    if len(bets) > 0:
        cum_pl = bets["pl_flat"].cumsum()
        bar_colors = [colors["win"] if v >= 0 else colors["loss"] for v in bets["pl_flat"]]
        ax2.bar(range(len(bets)), bets["pl_flat"], color=bar_colors, alpha=0.7, width=0.8)
        ax2.plot(range(len(bets)), cum_pl, color="white", lw=1.5, label="Acumulado")
        ax2.axhline(0, color="#555577", ls="--", lw=1)
        ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax2.set_xlabel("Nº aposta")
        ax2.set_ylabel("P&L")

    # 3. Distribuição de value por resultado
    ax3 = fig.add_subplot(gs[1, 0])
    ax_style(ax3, "🎯 Value por Resultado")
    if len(bets) > 0:
        for outcome, color in [("H", "#4fc3f7"), ("D", "#ffd54f"), ("A", "#ef9a9a")]:
            sub = bets[bets["bet_on"] == outcome]["value"]
            if len(sub) > 0:
                ax3.hist(sub, bins=15, alpha=0.6, color=color, label=outcome)
        ax3.axvline(MIN_VALUE, color="white", ls="--", lw=1.5, label=f"Min={MIN_VALUE}")
        ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax3.set_xlabel("Value (prob × odd)")
        ax3.set_ylabel("Frequência")

    # 4. Hit rate por faixa de value
    ax4 = fig.add_subplot(gs[1, 1])
    ax_style(ax4, "✅ Hit Rate por Faixa de Value")
    if len(bets) > 0:
        bins = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 2.0]
        labels = ["1.00-1.05","1.05-1.10","1.10-1.15","1.15-1.20","1.20-1.30",">1.30"]
        bets["value_bin"] = pd.cut(bets["value"], bins=bins, labels=labels)
        hr = bets.groupby("value_bin", observed=True)["won"].agg(["mean", "count"])
        hr_labels = [str(l) for l in hr.index]
        bars = ax4.bar(range(len(hr)), hr["mean"] * 100,
                       color=[colors["win"] if v > 0.45 else colors["loss"] for v in hr["mean"]],
                       alpha=0.8)
        for bar, (_, row_) in zip(bars, hr.iterrows()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"n={int(row_['count'])}", ha="center", va="bottom",
                     color="white", fontsize=7)
        ax4.set_xticks(range(len(hr)))
        ax4.set_xticklabels(hr_labels, rotation=30, ha="right", fontsize=7)
        ax4.axhline(50, color="#555577", ls="--", lw=1)
        ax4.set_ylabel("Hit Rate (%)")

    # 5. Drawdown
    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "📉 Drawdown (%)")
    peak_f = df_bt["bankroll_flat"].cummax()
    peak_k = df_bt["bankroll_kelly"].cummax()
    dd_f   = (peak_f - df_bt["bankroll_flat"]) / peak_f * 100
    dd_k   = (peak_k - df_bt["bankroll_kelly"]) / peak_k * 100
    ax5.fill_between(range(len(df_bt)), -dd_f,  0, alpha=0.5, color=colors["flat"],  label="Flat")
    ax5.fill_between(range(len(df_bt)), -dd_k,  0, alpha=0.5, color=colors["kelly"], label="Kelly")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax5.set_xlabel("Jogos")
    ax5.set_ylabel("Drawdown (%)")

    # 6. Calibração do modelo
    ax6 = fig.add_subplot(gs[2, 0])
    ax_style(ax6, "📐 Calibração do Modelo")
    if len(bets) > 0:
        try:
            frac_pos, mean_pred = calibration_curve(
                bets["won"].astype(int), bets["prob_model"], n_bins=8)
            ax6.plot(mean_pred, frac_pos, "o-", color=colors["flat"], lw=2, label="Modelo")
            ax6.plot([0, 1], [0, 1], "--", color="#555577", lw=1, label="Perfeito")
            ax6.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        except Exception:
            ax6.text(0.5, 0.5, "Dados insuficientes", ha="center", va="center", color="white")
        ax6.set_xlabel("Probabilidade prevista")
        ax6.set_ylabel("Frequência real")

    # 7. ROI por temporada
    ax7 = fig.add_subplot(gs[2, 1])
    ax_style(ax7, "📅 ROI por Temporada")
    if len(bets) > 0:
        for season in sorted(bets["season"].unique()):
            sub = bets[bets["season"] == season]
            roi = sub["pl_flat"].sum() / sub["stake_flat"].sum() * 100 if sub["stake_flat"].sum() > 0 else 0
            color = colors["win"] if roi >= 0 else colors["loss"]
            ax7.bar(str(int(season)), roi, color=color, alpha=0.8)
            ax7.text(str(int(season)), roi + (1 if roi >= 0 else -3),
                     f"{roi:+.1f}%", ha="center", color="white", fontsize=9)
        ax7.axhline(0, color="#555577", ls="--", lw=1)
        ax7.set_ylabel("Yield (%)")

    # 8. Resumo estatístico
    ax8 = fig.add_subplot(gs[2, 2])
    ax_style(ax8, "📋 Resumo")
    ax8.axis("off")
    if len(bets) > 0:
        roi_f = (df_bt["bankroll_flat"].iloc[-1]  - BANKROLL_INICIAL) / BANKROLL_INICIAL * 100
        roi_k = (df_bt["bankroll_kelly"].iloc[-1] - BANKROLL_INICIAL) / BANKROLL_INICIAL * 100
        yld_f = bets["pl_flat"].sum()  / bets["stake_flat"].sum()  * 100
        yld_k = bets["pl_kelly"].sum() / bets["stake_kelly"].sum() * 100
        summary = [
            ("Apostas",        f"{len(bets)}"),
            ("Hit Rate",       f"{bets['won'].mean():.1%}"),
            ("Odd média",      f"{bets['odd'].mean():.2f}"),
            ("Value médio",    f"{bets['value'].mean():.3f}"),
            ("", ""),
            ("ROI Flat",       f"{roi_f:+.2f}%"),
            ("Yield Flat",     f"{yld_f:+.2f}%"),
            ("MaxDD Flat",     f"{max_dd_flat:.2f}%"),
            ("", ""),
            ("ROI Kelly",      f"{roi_k:+.2f}%"),
            ("Yield Kelly",    f"{yld_k:+.2f}%"),
            ("MaxDD Kelly",    f"{max_dd_kelly:.2f}%"),
        ]
        for j, (k, v) in enumerate(summary):
            color = "#aaaaaa" if k == "" else ("white" if v.startswith("+") or v == "" else
                    ("#81c784" if "+" in v else "#ef5350" if "-" in v else "#e0e0e0"))
            ax8.text(0.05, 0.95 - j * 0.08, k,   transform=ax8.transAxes,
                     color="#aaaaaa", fontsize=10, va="top")
            ax8.text(0.65, 0.95 - j * 0.08, v,   transform=ax8.transAxes,
                     color=color,     fontsize=10, va="top", fontweight="bold")

    plt.suptitle(f"PREDICTOR — Backtesting Brasileirão {TEST_SEASONS}",
                 color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ Gráfico salvo em {OUTPUT_PNG}")
    plt.close()


def main():
    print("📂 Carregando dados e modelo...")
    df = pd.read_csv(DATA_PATH)

    if "season_x" in df.columns:
        df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns:
        df = df.rename(columns={"result_x": "result"})

    # Preencher odds ausentes com mediana
    odds_cols = ["prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
                 "odds_draw_factor", "odds_home_away_ratio", "market_entropy"]
    for col in odds_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURE_COLS + ["result", "odd_h", "odd_d", "odd_a"])
    df = df.sort_values("date").reset_index(drop=True)

    # Apenas temporadas de teste (nunca vistas no treino)
    df_test = df[df["season"].isin(TEST_SEASONS)].copy().reset_index(drop=True)
    print(f"   {len(df_test)} jogos no período de teste {TEST_SEASONS}")
    print(f"   Jogos com odds válidas: {df_test['odd_h'].notna().sum()}")

    model_data = joblib.load(MODEL_PATH)
    print("   ✅ Modelo carregado")

    print("\n🔄 Rodando backtesting...")
    df_bt, max_dd_flat, max_dd_kelly = run_backtest(df_test, model_data)

    print_summary(df_bt, max_dd_flat, max_dd_kelly)

    df_bt.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Resultados salvos em {OUTPUT_CSV}")

    plot_results(df_bt)


if __name__ == "__main__":
    main()
