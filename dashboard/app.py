import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = r"C:\PREDICTOR\REPO"
SCRAPING      = os.path.join(BASE, "scraping")
DATA_RAW      = os.path.join(SCRAPING, "data", "raw")
DATA_PROC     = os.path.join(SCRAPING, "data", "processed")
DATA_EXT      = os.path.join(SCRAPING, "data", "external")
MODELOS       = os.path.join(BASE, "modelos")

sys.path.append(SCRAPING)
sys.path.append(MODELOS)

MODEL_PATH         = os.path.join(MODELOS, "match_model.pkl")
POISSON_PATH       = os.path.join(MODELOS, "poisson_model.pkl")
FEATURES_PATH      = os.path.join(DATA_PROC, "features_odds.csv")
MATCHES_PATH       = os.path.join(DATA_RAW,  "matches_final.csv")
MATCHES_CSV        = os.path.join(DATA_RAW,  "matches.csv")
MARKET_PATH        = os.path.join(DATA_EXT,  "market_values.csv")
SIM_PATH           = os.path.join(DATA_PROC, "simulacao_2026.csv")
VALUE_BETS_PATH    = os.path.join(DATA_EXT,  "value_bets.csv")
ODDS_LIVE_PATH     = os.path.join(DATA_EXT,  "odds_live.csv")
BACKTESTING_PATH   = os.path.join(DATA_EXT,  "backtesting_results.csv")

# ── Config ────────────────────────────────────────────────────────────────────

TEAM_LOGOS = {
    "SE Palmeiras":               "https://tmssl.akamaized.net/images/wappen/head/1023.png",
    "CR Flamengo":                "https://tmssl.akamaized.net/images/wappen/head/614.png",
    "São Paulo FC":               "https://tmssl.akamaized.net/images/wappen/head/585.png",
    "SC Corinthians Paulista":    "https://tmssl.akamaized.net/images/wappen/head/199.png",
    "CA Mineiro":                 "https://tmssl.akamaized.net/images/wappen/head/330.png",
    "Fluminense FC":              "https://tmssl.akamaized.net/images/wappen/head/2462.png",
    "Botafogo FR":                "https://tmssl.akamaized.net/images/wappen/head/537.png",
    "CA Paranaense":              "https://tmssl.akamaized.net/images/wappen/head/679.png",
    "Grêmio FBPA":                "https://tmssl.akamaized.net/images/wappen/head/210.png",
    "SC Internacional":           "https://tmssl.akamaized.net/images/wappen/head/6600.png",
    "CR Vasco da Gama":           "https://tmssl.akamaized.net/images/wappen/head/978.png",
    "Cruzeiro EC":                "https://tmssl.akamaized.net/images/wappen/head/609.png",
    "Santos FC":                  "https://tmssl.akamaized.net/images/wappen/head/221.png",
    "RB Bragantino":              "https://tmssl.akamaized.net/images/wappen/head/8793.png",
    "EC Bahia":                   "https://tmssl.akamaized.net/images/wappen/head/10010.png",
    "Mirassol FC":                "https://tmssl.akamaized.net/images/wappen/head/3876.png",
    "Clube do Remo":              "https://tmssl.akamaized.net/images/wappen/head/10997.png",
    "EC Vitória":                 "https://tmssl.akamaized.net/images/wappen/head/2125.png",
    "Chapecoense AF":             "https://tmssl.akamaized.net/images/wappen/head/17776.png",
    "Coritiba FBC":               "https://tmssl.akamaized.net/images/wappen/head/776.png",
    "Atlético Mineiro":           "https://tmssl.akamaized.net/images/wappen/head/330.png",
}

def team_logo_html(team_name, size=28):
    url = TEAM_LOGOS.get(team_name, "")
    if not url:
        return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:#1D2D44;margin-right:8px;flex-shrink:0;"></div>'
    return (
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="object-fit:contain;border-radius:4px;vertical-align:middle;'
        f'margin-right:8px;flex-shrink:0;" '
        f'onerror="this.style.display=\'none\'" />'
    )

st.set_page_config(
    page_title="PREDICTOR — Brasileirão 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* ── Palette ── */
    --ink:       #0D1321;
    --space:     #1D2D44;
    --slate:     #3E5C76;
    --slate-lo:  #2c4260;
    --ice:       #6B9DC0;
    --accent:    #88b4d4;

    /* ── Semantic ── */
    --bg:        #0D1321;
    --surface:   #141d2e;
    --card:      #1D2D44;
    --card-alt:  #223352;
    --border:    #2c4260;
    --border-lo: #1a2740;
    --text:      #dce8f0;
    --muted:     #7a98b2;
    --muted-lo:  #3e5c76;

    /* ── Status colours (kept muted, not neon) ── */
    --green:     #4caf87;
    --green-dim: #3a8a6c;
    --red:       #c0505a;
    --yellow:    #c9a84c;
    --blue:      #5b8db8;
    --orange:    #c47a3d;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--slate-lo); border-radius: 3px; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px;
    color: var(--text) !important;
}

/* ── Section title ── */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 26px;
    letter-spacing: 3px;
    color: var(--text);
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 26px;
    background: linear-gradient(to bottom, var(--accent), var(--slate));
    border-radius: 2px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: 4px 6px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--muted) !important;
    border-radius: 7px;
    padding: 8px 18px;
    transition: all 0.2s ease;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--border), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ── Metric card (native) ── */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--slate), var(--accent));
    opacity: 0.7;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted) !important;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 30px !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Custom metric card ── */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--slate); }
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--slate), var(--accent));
    opacity: 0.6;
}
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 34px;
    color: var(--text);
    line-height: 1;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Value bet card ── */
.vbet-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--slate);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.vbet-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(62,92,118,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.vbet-card:hover {
    border-color: var(--slate);
    border-left-color: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.vbet-card.high  { border-left-color: var(--yellow); }
.vbet-card.high::after { background: linear-gradient(135deg, rgba(201,168,76,0.04) 0%, transparent 60%); }
.vbet-card.top   { border-left-color: var(--orange); }
.vbet-card.top::after { background: linear-gradient(135deg, rgba(196,122,61,0.04) 0%, transparent 60%); }

/* ── Standings row ── */
.standings-row {
    display: flex;
    align-items: center;
    padding: 11px 16px;
    border-radius: 10px;
    margin-bottom: 3px;
    background: var(--card);
    border: 1px solid var(--border-lo);
    font-size: 14px;
    transition: all 0.15s ease;
    cursor: default;
}
.standings-row:hover {
    background: var(--card-alt);
    border-color: var(--border);
}
.pos-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 17px;
    width: 28px;
    text-align: center;
    margin-right: 14px;
    flex-shrink: 0;
}
.pos-liberta { color: var(--blue); }
.pos-sul     { color: var(--yellow); }
.pos-rebaixa { color: var(--red); }
.pos-normal  { color: var(--muted-lo); }

/* ── Badges ── */
.vs-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: var(--muted);
    background: var(--border);
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.edge-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(76,175,135,0.12);
    color: var(--green);
    border: 1px solid rgba(76,175,135,0.25);
    letter-spacing: 0.5px;
}
.edge-badge.high {
    background: rgba(201,168,76,0.12);
    color: var(--yellow);
    border-color: rgba(201,168,76,0.25);
}
.edge-badge.top {
    background: rgba(196,122,61,0.12);
    color: var(--orange);
    border-color: rgba(196,122,61,0.25);
}

/* ── Prob bar ── */
.prob-bar-wrap {
    background: var(--border-lo);
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}
.prob-bar { height: 100%; border-radius: 3px; }

/* ── Info pill (mini stat in value bets) ── */
.info-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 90px;
}
.info-pill .label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin-bottom: 4px;
}
.info-pill .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}

/* ── Mono util ── */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

/* ── Team name ── */
.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 20px;
    letter-spacing: 1px;
}

/* ── Button ── */
.stButton > button {
    background: var(--slate) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--slate-lo) !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
}
.stButton > button:hover {
    background: var(--ice) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stNumberInput > div > div { background: var(--card) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-weight: 600;
    font-size: 13px;
}
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Alert / info ── */
.stAlert { border-radius: 8px !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border-lo) !important; margin: 20px 0 !important; }

/* ── Plotly chart background ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── Legend dots for standings ── */
.legend-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_poisson():
    return joblib.load(POISSON_PATH)

@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_PATH)
    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})
    return df

@st.cache_data
def load_matches():
    return pd.read_csv(MATCHES_PATH)

@st.cache_data
def load_market():
    try: return pd.read_csv(MARKET_PATH)
    except: return pd.DataFrame()

@st.cache_data
def load_sim():
    try: return pd.read_csv(SIM_PATH)
    except: return pd.DataFrame()


# ── Predict helpers ───────────────────────────────────────────────────────────
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


def predict_match(home, away, features_df, model_data,
                  odd_h=None, odd_d=None, odd_a=None):
    hf = features_df[features_df["home_team"] == home].tail(1)
    af = features_df[features_df["away_team"] == away].tail(1)
    if hf.empty or af.empty:
        return None

    hr = hf.iloc[0]; ar = af.iloc[0]

    if odd_h and odd_d and odd_a:
        p_h = 1/odd_h; p_d = 1/odd_d; p_a = 1/odd_a
        tot = p_h + p_d + p_a
        prob_h_mkt = p_h/tot; prob_d_mkt = p_d/tot; prob_a_mkt = p_a/tot
        odds_draw_factor = odd_d / ((odd_h + odd_a) / 2)
        odds_har = odd_h / odd_a
        market_entropy = -(prob_h_mkt*np.log(prob_h_mkt+1e-9) +
                           prob_d_mkt*np.log(prob_d_mkt+1e-9) +
                           prob_a_mkt*np.log(prob_a_mkt+1e-9))
    else:
        prob_h_mkt = hr.get("home_aproveitamento", 0.5)
        prob_d_mkt = 0.27
        prob_a_mkt = ar.get("away_aproveitamento", 0.23)
        odds_draw_factor = 1.0; odds_har = 1.2; market_entropy = 1.0

    feat = {
        "elo_diff": hr.get("home_elo",1500) - ar.get("away_elo",1500),
        "home_elo": hr.get("home_elo",1500), "away_elo": ar.get("away_elo",1500),
        "home_market_value_log": hr.get("home_market_value_log",4),
        "away_market_value_log": ar.get("away_market_value_log",4),
        "market_value_diff": hr.get("home_market_value_log",4) - ar.get("away_market_value_log",4),
        "home_market_value_norm": hr.get("home_market_value_norm",0.5),
        "away_market_value_norm": ar.get("away_market_value_norm",0.5),
        "home_squad_size": 20, "away_squad_size": 20,
        "home_aproveitamento": hr.get("home_aproveitamento",0.4),
        "away_aproveitamento": ar.get("away_aproveitamento",0.4),
        "position_diff": hr.get("position_diff",0),
        "home_form_pts": hr.get("home_form_pts",1),
        "home_avg_gf": hr.get("home_avg_gf",1.2),
        "home_avg_ga": hr.get("home_avg_ga",1.0),
        "home_goal_diff": hr.get("home_goal_diff",0),
        "home_win_rate": hr.get("home_win_rate",0.4),
        "home_draw_rate": hr.get("home_draw_rate",0.25),
        "home_home_form": hr.get("home_home_form",0),
        "away_form_pts": ar.get("away_form_pts",1),
        "away_avg_gf": ar.get("away_avg_gf",1.2),
        "away_avg_ga": ar.get("away_avg_ga",1.0),
        "away_goal_diff": ar.get("away_goal_diff",0),
        "away_win_rate": ar.get("away_win_rate",0.4),
        "away_draw_rate": ar.get("away_draw_rate",0.25),
        "away_away_form": ar.get("away_away_form",0),
        "home_form_pts_10": hr.get("home_form_pts_10",1),
        "home_avg_gf_10": hr.get("home_avg_gf_10",1.2),
        "home_avg_ga_10": hr.get("home_avg_ga_10",1.0),
        "home_win_rate_10": hr.get("home_win_rate_10",0.4),
        "away_form_pts_10": ar.get("away_form_pts_10",1),
        "away_avg_gf_10": ar.get("away_avg_gf_10",1.2),
        "away_avg_ga_10": ar.get("away_avg_ga_10",1.0),
        "away_win_rate_10": ar.get("away_win_rate_10",0.4),
        "h2h_home_wins": hr.get("h2h_home_wins",0),
        "h2h_away_wins": hr.get("h2h_away_wins",0),
        "h2h_draws": hr.get("h2h_draws",0),
        "prob_h_mkt": prob_h_mkt, "prob_d_mkt": prob_d_mkt, "prob_a_mkt": prob_a_mkt,
        "odds_draw_factor": odds_draw_factor,
        "odds_home_away_ratio": odds_har,
        "market_entropy": market_entropy,
    }

    df_feat = add_derived(pd.DataFrame([feat]))
    X = np.array([[df_feat.iloc[0].get(f, 0) for f in model_data["features"]]])

    ph  = model_data["cal_h"].predict(model_data["model_h"].predict_proba(X)[:, 1])[0]
    pd_ = model_data["cal_d"].predict(model_data["model_d"].predict_proba(X)[:, 1])[0]
    pa  = model_data["cal_a"].predict(model_data["model_a"].predict_proba(X)[:, 1])[0]
    tot = ph + pd_ + pa
    return ph/tot, pd_/tot, pa/tot


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:28px 0 20px 0;">
  <div style="margin-bottom:16px;">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                letter-spacing:5px;line-height:1;
                background:linear-gradient(135deg,#dce8f0 40%,#6B9DC0);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">
      PREDICTOR
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   color:#3e5c76;letter-spacing:2px;text-transform:uppercase;">
        Brasileirão 2026
      </span>
      <span style="background:#1D2D44;border:1px solid #2c4260;
                   border-radius:4px;padding:2px 8px;
                   font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#6B9DC0;letter-spacing:1px;">
        v3.0
      </span>
      <span style="background:rgba(76,175,135,0.1);border:1px solid rgba(76,175,135,0.2);
                   border-radius:4px;padding:2px 8px;
                   font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#4caf87;display:flex;align-items:center;gap:4px;">
        <span style="width:5px;height:5px;border-radius:50%;
                     background:#4caf87;display:inline-block;"></span>
        LIVE
      </span>
    </div>
  </div>
  <div style="height:1px;background:linear-gradient(90deg,#3E5C76 0%,#6B9DC0 40%,transparent 100%);
              opacity:0.5;margin-bottom:4px;"></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "SIMULAÇÃO", "PREVER PARTIDA", "VALUE BETS",
    "BACKTESTING", "VALOR DE MERCADO"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>TABELA BRASILEIRÃO 2026</div>", unsafe_allow_html=True)
    df_sim = load_sim()

    if df_sim.empty:
        st.warning("Rode o season_simulator.py para gerar a simulação.")
    else:
        top = df_sim.iloc[0]
        rebaixa_df = df_sim[df_sim["rebaixamento_pct"] > 50]
        rebaixa = rebaixa_df.iloc[0] if len(rebaixa_df) > 0 else df_sim.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Favorito ao Título", top["time"], f"{top['titulo_pct']:.1f}%")
        with c2: st.metric("Maior risco rebaixamento", rebaixa["time"], f"{rebaixa['rebaixamento_pct']:.1f}%")
        with c3: st.metric("Times simulados", len(df_sim))
        with c4: st.metric("Simulações Monte Carlo", "10.000")

        # Legend
        st.markdown("""
        <div style="display:flex;gap:20px;margin:16px 0 10px 0;flex-wrap:wrap;">
          <span style="font-size:11px;color:#7a98b2;display:flex;align-items:center;gap:5px;">
            <span class="legend-dot" style="background:#5b8db8;"></span> Libertadores (1–5)
          </span>
          <span style="font-size:11px;color:#7a98b2;display:flex;align-items:center;gap:5px;">
            <span class="legend-dot" style="background:#4caf87;"></span> Fase Qualificatória (6)
          </span>
          <span style="font-size:11px;color:#7a98b2;display:flex;align-items:center;gap:5px;">
            <span class="legend-dot" style="background:#c9a84c;"></span> Sul-Americana (6–11)
          </span>
          <span style="font-size:11px;color:#7a98b2;display:flex;align-items:center;gap:5px;">
            <span class="legend-dot" style="background:#c0505a;"></span> Rebaixamento (17–20)
          </span>
        </div>
        """, unsafe_allow_html=True)

        for i, row in df_sim.reset_index(drop=True).iterrows():
            pos = i + 1
            if pos <= 4:    pos_class, pos_color, zone_bg = "pos-liberta", "#5b8db8", "rgba(91,141,184,0.05)"
            elif pos == 5:  pos_class, pos_color, zone_bg = "pos-qualify", "#4caf87", "rgba(76,175,135,0.04)"
            elif pos <= 11: pos_class, pos_color, zone_bg = "pos-sul",     "#c9a84c", "rgba(201,168,76,0.04)"
            elif pos <= 16: pos_class, pos_color, zone_bg = "pos-normal",  "#3e5c76", "transparent"
            else:           pos_class, pos_color, zone_bg = "pos-rebaixa", "#c0505a", "rgba(192,80,90,0.05)"

            liberta_bar = min(row["libertadores_pct"], 100)
            logo_html   = team_logo_html(row["time"], size=26)

            st.markdown(
                f"<div class='standings-row' style='background-color:{zone_bg};border-left:2px solid {pos_color if pos<=11 or pos>=17 else 'transparent'};'>"
                f"<span class='pos-badge {pos_class}'>{pos}</span>"
                + logo_html +
                f"<span style='flex:1;font-weight:600;font-size:14px;letter-spacing:0.3px;'>{row['time']}</span>"
                f"<span class='mono' style='width:72px;text-align:right;color:#dce8f0;font-size:13px;'>"
                f"{row['pts_esperados']:.1f} <span style='color:#3e5c76;font-size:10px;'>pts</span></span>"
                f"<div style='width:220px;margin-left:18px;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:9px;color:#3e5c76;"
                f"margin-bottom:4px;font-family:JetBrains Mono,monospace;letter-spacing:0.5px;'>"
                f"<span style='color:#5b8db8;'>Titulo {row['titulo_pct']:.1f}%</span>"
                f"<span style='color:#c9a84c;'>Libertadores {row['libertadores_pct']:.1f}%</span>"
                f"<span style='color:#c0505a;'>Rebaixamento {row['rebaixamento_pct']:.1f}%</span>"
                f"</div>"
                f"<div class='prob-bar-wrap'>"
                f"<div class='prob-bar' style='width:{int(liberta_bar)}%;background:{pos_color};opacity:0.7;'></div>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVER PARTIDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>PREVER PARTIDA</div>", unsafe_allow_html=True)
    features_df = load_features()
    model_data  = load_model()
    all_teams   = sorted(features_df["home_team"].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Mandante", all_teams, key="pred_home")
    with col2:
        away_opts = [t for t in all_teams if t != home]
        away = st.selectbox("Visitante", away_opts, key="pred_away")

    # Matchup header
    home_logo = team_logo_html(home, size=36)
    away_logo = team_logo_html(away, size=36)
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:center;gap:16px;"
        f"padding:16px;background:#1D2D44;border:1px solid #2c4260;border-radius:12px;"
        f"margin:12px 0;'>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        + home_logo +
        f"<span style='font-family:Bebas Neue,sans-serif;font-size:22px;letter-spacing:1px;'>{home}</span>"
        f"</div>"
        f"<span class='vs-badge'>VS</span>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        + away_logo +
        f"<span style='font-family:Bebas Neue,sans-serif;font-size:22px;letter-spacing:1px;'>{away}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Inserir odds (opcional)"):
        oc1, oc2, oc3 = st.columns(3)
        with oc1: oh = st.number_input("Odd Mandante", min_value=1.01, value=2.10, step=0.05)
        with oc2: od = st.number_input("Odd Empate",   min_value=1.01, value=3.30, step=0.05)
        with oc3: oa = st.number_input("Odd Visitante",min_value=1.01, value=3.50, step=0.05)
        use_odds = st.checkbox("Usar odds na predição", value=False)

    if st.button("PREVER PARTIDA", use_container_width=True):
        result = predict_match(
            home, away, features_df, model_data,
            odd_h=oh if use_odds else None,
            odd_d=od if use_odds else None,
            odd_a=oa if use_odds else None,
        )
        if result:
            ph, pd_, pa = result
            outcomes   = {"H": ph, "D": pd_, "A": pa}
            pred       = max(outcomes, key=outcomes.get)
            pred_label = {"H": f"{home} vence", "D": "Empate", "A": f"{away} vence"}[pred]
            pred_color = {"H": "#5b8db8", "D": "#c9a84c", "A": "#88b4d4"}[pred]
            conf       = outcomes[pred]

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric(f"{home}", f"{ph:.1%}")
            with m2: st.metric("Empate", f"{pd_:.1%}")
            with m3: st.metric(f"{away}", f"{pa:.1%}")

            # Confidence bar widths
            bar_h = int(ph * 100)
            bar_d = int(pd_ * 100)
            bar_a = int(pa * 100)

            st.markdown(f"""
            <div style="background:#1D2D44;border:1px solid #2c4260;border-left:3px solid {pred_color};
                        border-radius:12px;padding:24px 28px;margin-top:18px;">
              <div style="font-size:9px;font-weight:700;color:#3e5c76;
                          letter-spacing:2.5px;text-transform:uppercase;margin-bottom:10px;">
                Previsão do Modelo
              </div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:32px;
                          letter-spacing:2px;color:{pred_color};">
                {pred_label}
              </div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                          color:#7a98b2;margin-top:6px;">
                Confiança: <span style="color:{pred_color};font-weight:600;">{conf:.1%}</span>
              </div>
              <div style="margin-top:18px;display:flex;flex-direction:column;gap:8px;">
                <div style="display:flex;align-items:center;gap:10px;">
                  <span style="font-size:10px;color:#7a98b2;width:72px;font-family:JetBrains Mono,monospace;">MAN {ph:.0%}</span>
                  <div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{bar_h}%;background:#5b8db8;"></div></div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                  <span style="font-size:10px;color:#7a98b2;width:72px;font-family:JetBrains Mono,monospace;">EMP {pd_:.0%}</span>
                  <div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{bar_d}%;background:#c9a84c;"></div></div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                  <span style="font-size:10px;color:#7a98b2;width:72px;font-family:JetBrains Mono,monospace;">VIS {pa:.0%}</span>
                  <div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{bar_a}%;background:#88b4d4;"></div></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if use_odds:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:10px;font-weight:700;color:#3e5c76;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>Value vs odds inseridas</div>", unsafe_allow_html=True)
                for outcome, odd, label in [("H", oh, home), ("D", od, "Empate"), ("A", oa, away)]:
                    prob  = outcomes[outcome]
                    value = prob * odd
                    edge  = (value - 1) * 100
                    is_value = value >= 1.05
                    color = "#4caf87" if is_value else "#3e5c76"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                background:#141d2e;border:1px solid {'rgba(76,175,135,0.25)' if is_value else '#2c4260'};
                                border-radius:8px;padding:12px 18px;margin-bottom:6px;">
                      <span style="font-weight:600;font-size:14px;">{label}</span>
                      <span class="mono" style="color:#7a98b2;">odd {odd:.2f}</span>
                      <span style="color:{color};font-family:'JetBrains Mono',monospace;font-weight:700;font-size:14px;">
                        {value:.3f} <span style="font-size:11px;">({edge:+.1f}%)</span>
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Times sem dados suficientes no histórico.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VALUE BETS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>VALUE BETS — PRÓXIMA RODADA</div>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_space = st.columns([1, 1, 4])
    with col_btn1:
        refresh = st.button("Atualizar Odds", use_container_width=True)
    with col_btn2:
        recalc  = st.button("Recalcular Value", use_container_width=True)

    if refresh or recalc:
        try:
            from odds_api import fetch_odds
            from value_bets import run_value_bets
            with st.spinner("Buscando odds e calculando value bets..."):
                if refresh:
                    fetch_odds()
                df_vb = run_value_bets()
            st.cache_data.clear()
            st.success(f"{len(df_vb)} value bets encontrados!")
        except Exception as e:
            st.error(f"Erro: {e}")

    try:
        df_vb = pd.read_csv(VALUE_BETS_PATH)
    except:
        df_vb = pd.DataFrame()

    if df_vb.empty:
        st.info("Nenhum value bet disponível. Clique em Atualizar Odds para buscar.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Value Bets", len(df_vb))
        with m2: st.metric("Edge Médio", f"+{df_vb['edge_pct'].mean():.1f}%")
        with m3: st.metric("Melhor Edge", f"+{df_vb['edge_pct'].max():.1f}%")
        with m4: st.metric("Kelly Médio", f"{df_vb['kelly_pct'].mean():.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        for _, r in df_vb.sort_values("value", ascending=False).iterrows():
            edge = r["edge_pct"]
            if edge >= 50:   card_class, badge_class = "top",  "top"
            elif edge >= 25: card_class, badge_class = "high", "high"
            else:            card_class, badge_class = "",      ""

            stars  = "★" * min(int(edge / 15) + 1, 5)
            ph_w   = int(r["prob_h"] * 100)
            pd_w   = int(r["prob_d"] * 100)
            pa_w   = int(r["prob_a"] * 100)

            home_team    = r["home_team"]
            away_team    = r["away_team"]
            aposta       = r["aposta"]
            date_str     = r["date"]
            time_str     = r["time_utc"]
            odd_str      = f"{r['odd_bet365']:.2f}"
            prob_m_str   = f"{r['prob_modelo']:.1%}"
            prob_mkt_str = f"{r['prob_mercado']:.1%}"
            edge_str     = f"{r['edge_pct']:.1f}"
            kelly_str    = f"{r['kelly_pct']:.2f}"
            ph_str       = f"{r['prob_h']:.0%}"
            pd_str       = f"{r['prob_d']:.0%}"
            pa_str       = f"{r['prob_a']:.0%}"

            home_logo_html = team_logo_html(home_team, size=24)
            away_logo_html = team_logo_html(away_team, size=24)

            html_card = (
                f'<div class="vbet-card {card_class}">'
                # ── Header row: teams + date
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
                '<div style="display:flex;align-items:center;gap:8px;">'
                '<div style="display:flex;align-items:center;">'
                + home_logo_html +
                f'<span class="team-name">{home_team}</span>'
                '</div>'
                '<span class="vs-badge" style="margin:0 10px;">VS</span>'
                '<div style="display:flex;align-items:center;">'
                + away_logo_html +
                f'<span class="team-name">{away_team}</span>'
                '</div>'
                '</div>'
                f'<span class="mono" style="color:#3d4d5e;font-size:11px;">{date_str} · {time_str} UTC</span>'
                '</div>'
                # ── Stats row
                '<div style="display:flex;gap:10px;align-items:stretch;flex-wrap:wrap;margin-bottom:16px;">'
                '<div class="info-pill">'
                '<div class="label">Apostar</div>'
                f'<div class="value" style="font-size:14px;font-family:Inter,sans-serif;font-weight:700;color:#f0f4f8;">{aposta}</div>'
                '</div>'
                '<div class="info-pill">'
                '<div class="label">Odd Pinnacle</div>'
                f'<div class="value" style="font-size:20px;">{odd_str}</div>'
                '</div>'
                '<div class="info-pill">'
                '<div class="label">Prob Modelo</div>'
                f'<div class="value" style="color:#4caf87;">{prob_m_str}</div>'
                '</div>'
                '<div class="info-pill">'
                '<div class="label">Prob Mercado</div>'
                f'<div class="value" style="color:#7a98b2;">{prob_mkt_str}</div>'
                '</div>'
                '<div style="display:flex;flex-direction:column;justify-content:center;gap:8px;margin-left:auto;">'
                f'<span class="edge-badge {badge_class}">EDGE +{edge_str}% {stars}</span>'
                f'<span style="font-size:11px;color:#7a98b2;font-family:JetBrains Mono,monospace;">'
                f'Kelly: <b style="color:#dce8f0;">{kelly_str}%</b></span>'
                '</div>'
                '</div>'
                # ── Prob distribution
                '<div style="border-top:1px solid #1a2740;padding-top:12px;">'
                '<div style="font-size:9px;font-weight:700;color:#3e5c76;letter-spacing:1.5px;'
                'text-transform:uppercase;margin-bottom:8px;">Distribuição do Modelo</div>'
                '<div style="display:flex;gap:8px;align-items:center;">'
                '<span style="font-size:10px;color:#7a98b2;width:28px;font-family:JetBrains Mono,monospace;">MAN</span>'
                f'<div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{ph_w}%;background:#5b8db8;"></div></div>'
                f'<span class="mono" style="font-size:11px;width:36px;color:#7a98b2;">{ph_str}</span>'
                '</div>'
                '<div style="display:flex;gap:8px;align-items:center;margin-top:5px;">'
                '<span style="font-size:10px;color:#7a98b2;width:28px;font-family:JetBrains Mono,monospace;">EMP</span>'
                f'<div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{pd_w}%;background:#c9a84c;"></div></div>'
                f'<span class="mono" style="font-size:11px;width:36px;color:#7a98b2;">{pd_str}</span>'
                '</div>'
                '<div style="display:flex;gap:8px;align-items:center;margin-top:5px;">'
                '<span style="font-size:10px;color:#7a98b2;width:28px;font-family:JetBrains Mono,monospace;">VIS</span>'
                f'<div class="prob-bar-wrap" style="flex:1;"><div class="prob-bar" style="width:{pa_w}%;background:#88b4d4;"></div></div>'
                f'<span class="mono" style="font-size:11px;width:36px;color:#7a98b2;">{pa_str}</span>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(html_card, unsafe_allow_html=True)

        with st.expander("Ver tabela completa"):
            cols_show = ["date", "home_team", "away_team", "aposta",
                         "prob_modelo", "odd_bet365", "value", "edge_pct", "kelly_pct"]
            st.dataframe(
                df_vb[cols_show].sort_values("value", ascending=False),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>BACKTESTING — BRASILEIRÃO 2025/2026</div>", unsafe_allow_html=True)

    try:
        df_bt = pd.read_csv(BACKTESTING_PATH)
        bets  = df_bt[df_bt["bet_on"].notna()].copy()

        if len(bets) == 0:
            st.warning("Nenhuma aposta no backtesting. Rode backtesting.py.")
        else:
            roi_f    = (df_bt["bankroll_flat"].iloc[-1]  - 1000) / 1000 * 100
            roi_k    = (df_bt["bankroll_kelly"].iloc[-1] - 1000) / 1000 * 100
            hit_rate = bets["won"].mean()
            yld_f    = bets["pl_flat"].sum() / bets["stake_flat"].sum() * 100

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: st.metric("Total Apostas",    len(bets))
            with k2: st.metric("Hit Rate",   f"{hit_rate:.1%}")
            with k3: st.metric("ROI Flat",   f"{roi_f:+.1f}%")
            with k4: st.metric("ROI Kelly",  f"{roi_k:+.1f}%")
            with k5: st.metric("Yield",      f"{yld_f:+.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_bt["bankroll_flat"], mode="lines",
                name="Flat (2%)", line=dict(color="#6B9DC0", width=2)))
            fig.add_trace(go.Scatter(
                y=df_bt["bankroll_kelly"], mode="lines",
                name="Kelly (25%)", line=dict(color="#4caf87", width=2)))
            fig.add_hline(y=1000, line_dash="dash",
                          line_color="#2c4260", annotation_text="Bankroll inicial",
                          annotation_font_color="#3e5c76")
            fig.update_layout(
                paper_bgcolor="#0D1321", plot_bgcolor="#1D2D44",
                font=dict(color="#dce8f0", family="Inter"),
                title=dict(text="Evolução do Bankroll", font=dict(family="Bebas Neue", size=20, color="#dce8f0")),
                legend=dict(bgcolor="#1D2D44", bordercolor="#2c4260", font=dict(size=12)),
                xaxis=dict(gridcolor="#1a2740", color="#7a98b2", showline=False),
                yaxis=dict(gridcolor="#1a2740", color="#7a98b2", showline=False),
                height=360, margin=dict(l=0, r=0, t=44, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            colors_pl = ["#4caf87" if v >= 0 else "#c0505a" for v in bets["pl_flat"]]
            fig2.add_trace(go.Bar(
                y=bets["pl_flat"], marker_color=colors_pl,
                name="P&L por aposta", opacity=0.75, marker_line_width=0))
            fig2.add_trace(go.Scatter(
                y=bets["pl_flat"].cumsum(), mode="lines",
                name="P&L acumulado", line=dict(color="#dce8f0", width=2)))
            fig2.add_hline(y=0, line_dash="dash", line_color="#2c4260")
            fig2.update_layout(
                paper_bgcolor="#0D1321", plot_bgcolor="#1D2D44",
                font=dict(color="#dce8f0", family="Inter"),
                title=dict(text="P&L por Aposta (Flat)", font=dict(family="Bebas Neue", size=20, color="#dce8f0")),
                legend=dict(bgcolor="#1D2D44", bordercolor="#2c4260", font=dict(size=12)),
                xaxis=dict(gridcolor="#1a2740", color="#7a98b2", showline=False),
                yaxis=dict(gridcolor="#1a2740", color="#7a98b2", showline=False),
                height=300, margin=dict(l=0, r=0, t=44, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Ver todas as apostas"):
                cols_bt = ["date", "home_team", "away_team", "bet_on",
                           "result", "won", "odd", "prob_model",
                           "value", "pl_flat", "bankroll_flat"]
                st.dataframe(
                    bets[cols_bt].sort_values("date", ascending=False),
                    use_container_width=True, hide_index=True,
                )

    except FileNotFoundError:
        st.warning("Rode `python backtesting.py` para gerar os dados.")
    except Exception as e:
        st.error(f"Erro ao carregar backtesting: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALOR DE MERCADO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>VALOR DE MERCADO — BRASILEIRÃO 2026</div>", unsafe_allow_html=True)
    df_mv = load_market()

    if df_mv.empty:
        st.warning("market_values.csv não encontrado.")
    else:
        df_mv = df_mv.sort_values("market_value", ascending=False)
        total    = df_mv["market_value"].sum()
        top_team = df_mv.iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Time mais valioso", top_team["team"], f"€{top_team['market_value']:.1f}M")
        with c2: st.metric("Total da liga", f"€{total:.1f}M")
        with c3: st.metric("Média por time", f"€{total/len(df_mv):.1f}M")

        st.markdown("<br>", unsafe_allow_html=True)

        import plotly.express as px
        fig = px.bar(
            df_mv.head(20), x="market_value", y="team",
            orientation="h", color="market_value",
            color_continuous_scale=["#1D2D44", "#3E5C76", "#6B9DC0"],
            labels={"market_value": "Valor (€M)", "team": ""},
        )
        fig.update_layout(
            paper_bgcolor="#0D1321", plot_bgcolor="#1D2D44",
            font=dict(color="#dce8f0", family="Inter"),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", color="#7a98b2", gridcolor="#1a2740"),
            xaxis=dict(color="#7a98b2", gridcolor="#1a2740"),
            height=520, margin=dict(l=0, r=0, t=20, b=0),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='font-family:Bebas Neue,sans-serif;font-size:18px;letter-spacing:2px;color:#7a98b2;margin:8px 0 12px 0;'>RANKING DETALHADO</div>", unsafe_allow_html=True)
        max_val = df_mv["market_value"].max()

        for rank_i, (_, mv_row) in enumerate(df_mv.iterrows(), start=1):
            logo_h    = team_logo_html(mv_row["team"], size=26)
            bar_w     = int(mv_row["market_value"] / max_val * 100)
            bar_color = "#5b8db8" if rank_i <= 6 else "#c9a84c" if rank_i <= 12 else "#3e5c76"

            st.markdown(
                f"<div style='display:flex;align-items:center;padding:10px 16px;"
                f"background:#1D2D44;border:1px solid #1a2740;border-radius:10px;"
                f"margin-bottom:3px;gap:10px;transition:all 0.15s;'>"
                f"<span style='font-family:Bebas Neue,sans-serif;font-size:15px;color:#3e5c76;"
                f"width:24px;text-align:center;flex-shrink:0;'>{rank_i}</span>"
                + logo_h +
                f"<span style='flex:1;font-weight:600;font-size:14px;'>{mv_row['team']}</span>"
                f"<div style='width:160px;'>"
                f"<div style='background:#1a2740;border-radius:3px;height:4px;overflow:hidden;'>"
                f"<div style='width:{bar_w}%;height:100%;background:{bar_color};border-radius:3px;'></div>"
                f"</div>"
                f"</div>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:13px;font-weight:600;"
                f"color:#dce8f0;width:72px;text-align:right;'>€{mv_row['market_value']:.1f}M</span>"
                f"<span style='display:flex;align-items:center;justify-content:flex-end;gap:4px;"
                f"width:52px;color:#3e5c76;'>"
                f"<svg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' "
                f"stroke='#3e5c76' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
                f"<circle cx='12' cy='8' r='4'/>"
                f"<path d='M4 20c0-4 3.6-7 8-7s8 3 8 7'/>"
                f"</svg>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;'>"
                f"{int(mv_row.get('squad_size', 0))}</span>"
                f"</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:24px 0;
            border-top:1px solid #1a2740;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3e5c76;
              letter-spacing:1px;">
    © 2026 PREDICTOR
    <span style="margin:0 10px;color:#2c4260;">·</span>
    Todos os direitos reservados
    <span style="margin:0 10px;color:#2c4260;">·</span>
    Desenvolvido por <span style="color:#7a98b2;">Nycolas F. Oliveira</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#2c4260;
              margin-top:6px;letter-spacing:1px;">
    LightGBM · Acurácia 55.96% · Monte Carlo 10.000 simulações
  </div>
</div>
""", unsafe_allow_html=True)
