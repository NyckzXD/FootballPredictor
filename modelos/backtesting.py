"""
backtesting_v2.py — Backtesting para match_model_v2 FINAL
==========================================================
Modelo base melhorado + filtros de aposta mais rígidos.
"""
import pandas as pd
import numpy as np
import math
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\match_model_v2.pkl"
OUTPUT_CSV = r"C:\PREDICTOR\REPO\scraping\data\external\backtesting_results_v2.csv"

BANKROLL_INICIAL = 1000.0
KELLY_FRACTION   = 0.20
MIN_VALUE        = 1.08
MIN_PROB         = 0.55
MAX_BET_FLAT     = 0.02
MAX_BET_KELLY    = 0.05
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


def predict_probs_v2(model_data, X):
    ph_raw = np.mean([m.predict_proba(X)[:,1] for m in model_data["models_h"]], axis=0)
    pd_raw = np.mean([m.predict_proba(X)[:,1] for m in model_data["models_d"]], axis=0)
    pa_raw = np.mean([m.predict_proba(X)[:,1] for m in model_data["models_a"]], axis=0)
    ph = model_data["cal_h"].predict(ph_raw)
    pd_ = model_data["cal_d"].predict(pd_raw)
    pa = model_data["cal_a"].predict(pa_raw)
    total = ph + pd_ + pa
    return ph/total, pd_/total, pa/total


def kelly_bet(prob, odd, fraction=KELLY_FRACTION):
    edge = prob * odd - 1
    if edge <= 0: return 0.0
    k = (prob - (1-prob)/(odd-1)) * fraction
    return max(0.0, k)


def run_backtest(df, model_data):
    records = []
    bankroll_flat = bankroll_kelly = BANKROLL_INICIAL
    peak_flat = peak_kelly = BANKROLL_INICIAL
    max_dd_flat = max_dd_kelly = 0.0
    X = add_derived(df[FEATURE_COLS])
    ph_arr, pd_arr, pa_arr = predict_probs_v2(model_data, X)

    for i, (_, row) in enumerate(df.iterrows()):
        ph, pd_, pa = ph_arr[i], pd_arr[i], pa_arr[i]
        result = row["result"]
        odd_h, odd_d, odd_a = row.get("odd_h",np.nan), row.get("odd_d",np.nan), row.get("odd_a",np.nan)
        bets = []
        for prob, odd, outcome in [(ph,odd_h,"H"),(pd_,odd_d,"D"),(pa,odd_a,"A")]:
            if pd.isna(odd) or odd<=1.0: continue
            value = prob*odd
            if value >= MIN_VALUE and prob >= MIN_PROB:
                bets.append({"outcome":outcome,"prob":prob,"odd":odd,"value":value})
        if bets:
            bet = max(bets, key=lambda x: x["value"])
            won = bet["outcome"]==result
            sf = min(MAX_BET_FLAT*bankroll_flat, bankroll_flat)
            pf = sf*(bet["odd"]-1) if won else -sf
            bankroll_flat += pf; peak_flat = max(peak_flat, bankroll_flat)
            max_dd_flat = max(max_dd_flat, (peak_flat-bankroll_flat)/peak_flat*100)
            kp = kelly_bet(bet["prob"],bet["odd"])
            sk = min(kp*bankroll_kelly, MAX_BET_KELLY*bankroll_kelly); sk=max(sk,0)
            pk = sk*(bet["odd"]-1) if won else -sk
            bankroll_kelly += pk; peak_kelly = max(peak_kelly, bankroll_kelly)
            max_dd_kelly = max(max_dd_kelly, (peak_kelly-bankroll_kelly)/peak_kelly*100)
            records.append({"date":row.get("date",""),"season":row.get("season",""),
                "home_team":row.get("home_team",""),"away_team":row.get("away_team",""),
                "result":result,"bet_on":bet["outcome"],"prob_model":round(bet["prob"],3),
                "odd":round(bet["odd"],2),"value":round(bet["value"],3),"won":won,
                "stake_flat":round(sf,2),"pl_flat":round(pf,2),"bankroll_flat":round(bankroll_flat,2),
                "stake_kelly":round(sk,2),"pl_kelly":round(pk,2),"bankroll_kelly":round(bankroll_kelly,2)})
        else:
            records.append({"date":row.get("date",""),"season":row.get("season",""),
                "home_team":row.get("home_team",""),"away_team":row.get("away_team",""),
                "result":result,"bet_on":None,"prob_model":None,"odd":None,"value":None,"won":None,
                "stake_flat":0,"pl_flat":0,"bankroll_flat":round(bankroll_flat,2),
                "stake_kelly":0,"pl_kelly":0,"bankroll_kelly":round(bankroll_kelly,2)})
    return pd.DataFrame(records), max_dd_flat, max_dd_kelly


def print_summary(df_bt, max_dd_flat, max_dd_kelly):
    bets = df_bt[df_bt["bet_on"].notna()].copy()
    print(f"\n{'='*65}")
    print(f"📊 BACKTESTING v2 — {TEST_SEASONS}")
    print(f"{'='*65}")
    print(f"   Jogos: {len(df_bt)} | Apostas: {len(bets)} ({len(bets)/len(df_bt):.1%})")
    if len(bets)==0: print("   Nenhuma aposta"); return
    print(f"   Hit rate: {bets['won'].mean():.1%}")
    for o in ["H","D","A"]:
        s = bets[bets["bet_on"]==o]
        if len(s)>0: print(f"   {o}: {len(s):3d} | HR={s['won'].mean():.1%} | odd={s['odd'].mean():.2f} | val={s['value'].mean():.3f}")
    tsf=bets["stake_flat"].sum(); tsk=bets["stake_kelly"].sum()
    rf=(df_bt["bankroll_flat"].iloc[-1]-BANKROLL_INICIAL)/BANKROLL_INICIAL*100
    rk=(df_bt["bankroll_kelly"].iloc[-1]-BANKROLL_INICIAL)/BANKROLL_INICIAL*100
    yf=bets["pl_flat"].sum()/tsf*100 if tsf>0 else 0
    yk=bets["pl_kelly"].sum()/tsk*100 if tsk>0 else 0
    print(f"\n   FLAT:  bankroll={df_bt['bankroll_flat'].iloc[-1]:.0f} | ROI={rf:+.1f}% | Yield={yf:+.1f}% | DD={max_dd_flat:.1f}%")
    print(f"   KELLY: bankroll={df_bt['bankroll_kelly'].iloc[-1]:.0f} | ROI={rk:+.1f}% | Yield={yk:+.1f}% | DD={max_dd_kelly:.1f}%")
    print(f"\n   Por faixa de prob:")
    for lo,hi in [(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.85)]:
        s=bets[(bets["prob_model"]>=lo)&(bets["prob_model"]<hi)]
        if len(s)>0: print(f"   [{lo:.2f}-{hi:.2f}): {len(s):3d} | HR={s['won'].mean():.1%}")
    print(f"\n   Por faixa de value:")
    for lo,hi in [(1.08,1.15),(1.15,1.25),(1.25,1.40),(1.40,3.0)]:
        s=bets[(bets["value"]>=lo)&(bets["value"]<hi)]
        if len(s)>0: print(f"   [{lo:.2f}-{hi:.2f}): {len(s):3d} | HR={s['won'].mean():.1%}")
    print(f"\n📌 MIN_VALUE={MIN_VALUE} | MIN_PROB={MIN_PROB} | KELLY={KELLY_FRACTION}")


def main():
    print("="*65); print("  BACKTESTING v2 FINAL"); print("="*65)
    df = pd.read_csv(DATA_PATH)
    if "season_x" in df.columns: df=df.rename(columns={"season_x":"season"})
    if "result_x" in df.columns: df=df.rename(columns={"result_x":"result"})
    for c in ["prob_h_mkt","prob_d_mkt","prob_a_mkt","odds_draw_factor","odds_home_away_ratio","market_entropy"]:
        if c in df.columns: df[c]=df[c].fillna(df[c].median())
    df = df.dropna(subset=FEATURE_COLS+["result","odd_h","odd_d","odd_a"])
    df = df.sort_values("date").reset_index(drop=True)
    df_test = df[df["season"].isin(TEST_SEASONS)].copy().reset_index(drop=True)
    print(f"\n   {len(df_test)} jogos teste")
    model_data = joblib.load(MODEL_PATH)
    print(f"   Modelo {model_data.get('version','?')} carregado")
    df_bt, mdf, mdk = run_backtest(df_test, model_data)
    print_summary(df_bt, mdf, mdk)
    df_bt.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ CSV: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()