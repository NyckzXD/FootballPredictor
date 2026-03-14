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
st.set_page_config(
    page_title="PREDICTOR — Brasileirão 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --green:   #00e676;
    --red:     #ff1744;
    --yellow:  #ffd600;
    --blue:    #2979ff;
    --bg:      #080c12;
    --card:    #0d1117;
    --border:  #1c2333;
    --text:    #e6edf3;
    --muted:   #8b949e;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 2px; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    font-size: 13px;
    color: var(--muted) !important;
    border-radius: 6px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: var(--text);
    line-height: 1;
}

.vbet-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.vbet-card:hover { border-left-color: var(--yellow); }
.vbet-card.high  { border-left-color: #ff6d00; }
.vbet-card.top   { border-left-color: var(--red); }

.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    letter-spacing: 1px;
}
.vs-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 4px;
}
.edge-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 5px;
    background: rgba(0,230,118,0.12);
    color: var(--green);
    border: 1px solid rgba(0,230,118,0.25);
}
.edge-badge.high {
    background: rgba(255,109,0,0.12);
    color: #ff6d00;
    border-color: rgba(255,109,0,0.25);
}
.edge-badge.top {
    background: rgba(255,23,68,0.12);
    color: var(--red);
    border-color: rgba(255,23,68,0.25);
}
.prob-bar-wrap {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 4px;
}
.prob-bar { height: 100%; border-radius: 4px; }

.standings-row {
    display: flex;
    align-items: center;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    background: var(--card);
    border: 1px solid var(--border);
    font-size: 14px;
}
.pos-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px;
    width: 32px;
    text-align: center;
    margin-right: 12px;
}
.pos-liberta  { color: var(--blue); }
.pos-sul      { color: var(--yellow); }
.pos-rebaixa  { color: var(--red); }
.pos-normal   { color: var(--muted); }

.mono { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 28px !important;
}

.stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div, .stMultiSelect > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

hr { border-color: var(--border) !important; }
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
<div style='padding: 24px 0 8px 0;'>
  <span style='font-family:Bebas Neue,sans-serif;font-size:42px;letter-spacing:4px;'>
    ⚽ PREDICTOR
  </span>
  <span style='font-family:JetBrains Mono,monospace;font-size:13px;color:#8b949e;
               margin-left:16px;background:#1c2333;padding:4px 10px;border-radius:5px;'>
    BRASILEIRÃO 2026 · v3.0
  </span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Simulação", "🎯 Prever Partida", "📊 Value Bets",
    "📈 Backtesting", "💰 Valor de Mercado"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### TABELA BRASILEIRÃO 2026")
    df_sim = load_sim()

    if df_sim.empty:
        st.warning("Rode o season_simulator.py para gerar a simulação.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        top = df_sim.iloc[0]
        with c1:
            st.metric("🥇 Favorito ao Título", top["time"], f"{top['titulo_pct']:.1f}%")
        with c2:
            rebaixa = df_sim[df_sim["rebaixamento_pct"] > 50].iloc[0] if len(df_sim[df_sim["rebaixamento_pct"] > 50]) > 0 else df_sim.iloc[-1]
            st.metric("⬇️ Maior risco rebaixamento", rebaixa["time"], f"{rebaixa['rebaixamento_pct']:.1f}%")
        with c3:
            st.metric("🏟️ Times simulados", len(df_sim))
        with c4:
            st.metric("🎲 Simulações", "10.000")

        st.markdown("<br>", unsafe_allow_html=True)

        for i, row in df_sim.reset_index(drop=True).iterrows():
            pos = i + 1
            if pos <= 6:    pos_class, pos_color = "pos-liberta", "#2979ff"
            elif pos <= 12: pos_class, pos_color = "pos-sul",     "#ffd600"
            elif pos <= 17: pos_class, pos_color = "pos-normal",  "#8b949e"
            else:           pos_class, pos_color = "pos-rebaixa", "#ff1744"

            titulo_bar  = min(row["titulo_pct"] * 2, 100)
            liberta_bar = min(row["libertadores_pct"], 100)
            rebaixa_bar = min(row["rebaixamento_pct"], 100)

            st.markdown(f"""
            <div class='standings-row'>
              <span class='pos-badge {pos_class}'>{pos}</span>
              <span style='flex:1;font-weight:600;font-size:15px;'>{row['time']}</span>
              <span class='mono' style='width:60px;text-align:right;color:#e6edf3;'>{row['pts_esperados']:.1f} pts</span>
              <div style='width:200px;margin-left:20px;'>
                <div style='display:flex;justify-content:space-between;font-size:10px;color:#8b949e;margin-bottom:2px;'>
                  <span>🏆 {row['titulo_pct']:.1f}%</span>
                  <span>🌎 {row['libertadores_pct']:.1f}%</span>
                  <span>⬇️ {row['rebaixamento_pct']:.1f}%</span>
                </div>
                <div class='prob-bar-wrap'>
                  <div class='prob-bar' style='width:{liberta_bar}%;background:{pos_color};'></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVER PARTIDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### PREVER PARTIDA")
    features_df = load_features()
    model_data  = load_model()
    all_teams   = sorted(features_df["home_team"].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Mandante", all_teams, key="pred_home")
    with col2:
        away_opts = [t for t in all_teams if t != home]
        away = st.selectbox("✈️ Visitante", away_opts, key="pred_away")

    with st.expander("⚙️ Inserir odds (opcional)"):
        oc1, oc2, oc3 = st.columns(3)
        with oc1: oh = st.number_input("Odd Mandante", min_value=1.01, value=2.10, step=0.05)
        with oc2: od = st.number_input("Odd Empate",   min_value=1.01, value=3.30, step=0.05)
        with oc3: oa = st.number_input("Odd Visitante",min_value=1.01, value=3.50, step=0.05)
        use_odds = st.checkbox("Usar odds na predição", value=False)

    if st.button("⚡ PREVER", use_container_width=True):
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
            pred_label = {
                "H": f"🏠 {home} vence",
                "D": "🤝 Empate",
                "A": f"✈️ {away} vence"
            }[pred]

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric(f"🏠 {home}", f"{ph:.1%}")
            with m2: st.metric("🤝 Empate", f"{pd_:.1%}")
            with m3: st.metric(f"✈️ {away}", f"{pa:.1%}")

            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid #1c2333;border-left:4px solid #00e676;
                        border-radius:10px;padding:20px;margin-top:16px;text-align:center;'>
              <div style='font-size:12px;color:#8b949e;text-transform:uppercase;
                          letter-spacing:2px;margin-bottom:8px;'>Previsão do Modelo</div>
              <div style='font-family:Bebas Neue,sans-serif;font-size:28px;
                          letter-spacing:2px;color:#00e676;'>{pred_label}</div>
              <div style='font-size:13px;color:#8b949e;margin-top:6px;'>
                Confiança: {outcomes[pred]:.1%}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if use_odds:
                st.markdown("<br>**Value vs odds inseridas:**", unsafe_allow_html=True)
                for outcome, odd, label in [("H", oh, home), ("D", od, "Empate"), ("A", oa, away)]:
                    prob  = outcomes[outcome]
                    value = prob * odd
                    edge  = (value - 1) * 100
                    color = "#00e676" if value >= 1.05 else "#8b949e"
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;align-items:center;
                                background:#0d1117;border:1px solid #1c2333;border-radius:8px;
                                padding:10px 16px;margin-bottom:6px;'>
                      <span style='font-weight:600;'>{label}</span>
                      <span class='mono'>odd {odd:.2f}</span>
                      <span style='color:{color};font-family:JetBrains Mono,monospace;font-weight:600;'>
                        value {value:.3f} ({edge:+.1f}%)
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Times sem dados suficientes no histórico.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VALUE BETS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### VALUE BETS — PRÓXIMA RODADA")

    col_btn1, col_btn2, col_space = st.columns([1, 1, 4])
    with col_btn1:
        refresh = st.button("🔄 Atualizar Odds", use_container_width=True)
    with col_btn2:
        recalc  = st.button("⚡ Recalcular Value", use_container_width=True)

    if refresh or recalc:
        try:
            from odds_api import fetch_odds
            from value_bets import run_value_bets
            with st.spinner("Buscando odds e calculando value bets..."):
                if refresh:
                    fetch_odds()
                df_vb = run_value_bets()
            st.cache_data.clear()
            st.success(f"✅ {len(df_vb)} value bets encontrados!")
        except Exception as e:
            st.error(f"Erro: {e}")

    # Carregar value bets salvos
    try:
        df_vb = pd.read_csv(VALUE_BETS_PATH)
    except:
        df_vb = pd.DataFrame()

    if df_vb.empty:
        st.info("Nenhum value bet disponível. Clique em **Atualizar Odds** para buscar.")
    else:
        # Métricas resumo
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("🎯 Value Bets", len(df_vb))
        with m2: st.metric("📊 Edge Médio", f"+{df_vb['edge_pct'].mean():.1f}%")
        with m3: st.metric("💎 Melhor Edge", f"+{df_vb['edge_pct'].max():.1f}%")
        with m4: st.metric("🏦 Kelly Médio", f"{df_vb['kelly_pct'].mean():.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        for _, r in df_vb.sort_values("value", ascending=False).iterrows():
            edge = r["edge_pct"]
            if edge >= 50:   card_class, badge_class = "top",  "top"
            elif edge >= 25: card_class, badge_class = "high", "high"
            else:            card_class, badge_class = "",      ""

            stars = "⭐" * min(int(edge / 15) + 1, 5)

            ph_w  = int(r["prob_h"] * 100)
            pd_w  = int(r["prob_d"] * 100)
            pa_w  = int(r["prob_a"] * 100)

            st.markdown(f"""
            <div class='vbet-card {card_class}'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                <div>
                  <span class='team-name'>{r['home_team']}</span>
                  <span class='vs-badge' style='margin:0 10px;'>VS</span>
                  <span class='team-name'>{r['away_team']}</span>
                </div>
                <div style='text-align:right;'>
                  <span class='mono' style='color:#8b949e;font-size:12px;'>
                    📅 {r['date']} · {r['time_utc']} UTC
                  </span>
                </div>
              </div>

              <div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap;'>
                <div style='background:#1c2333;border-radius:8px;padding:10px 16px;'>
                  <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:4px;'>Apostar</div>
                  <div style='font-weight:700;font-size:15px;color:#e6edf3;'>{r['aposta']}</div>
                </div>

                <div style='background:#1c2333;border-radius:8px;padding:10px 16px;'>
                  <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:4px;'>Odd Pinnacle</div>
                  <div class='mono' style='font-size:18px;font-weight:600;
                               color:#e6edf3;'>{r['odd_bet365']:.2f}</div>
                </div>

                <div style='background:#1c2333;border-radius:8px;padding:10px 16px;'>
                  <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:4px;'>Prob Modelo</div>
                  <div class='mono' style='font-size:18px;font-weight:600;
                               color:#e6edf3;'>{r['prob_modelo']:.1%}</div>
                </div>

                <div style='background:#1c2333;border-radius:8px;padding:10px 16px;'>
                  <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:4px;'>Prob Mercado</div>
                  <div class='mono' style='font-size:18px;font-weight:600;
                               color:#8b949e;'>{r['prob_mercado']:.1%}</div>
                </div>

                <div>
                  <span class='edge-badge {badge_class}'>
                    EDGE +{r['edge_pct']:.1f}% {stars}
                  </span>
                  <br><br>
                  <span style='font-size:12px;color:#8b949e;'>
                    Kelly: <b style='color:#e6edf3;'>{r['kelly_pct']:.2f}%</b> do bankroll
                  </span>
                </div>
              </div>

              <div style='margin-top:14px;'>
                <div style='font-size:11px;color:#8b949e;margin-bottom:6px;'>
                  Distribuição de probabilidades do modelo:
                </div>
                <div style='display:flex;gap:6px;align-items:center;'>
                  <span style='font-size:11px;color:#8b949e;width:30px;'>H</span>
                  <div class='prob-bar-wrap' style='flex:1;'>
                    <div class='prob-bar' style='width:{ph_w}%;background:#2979ff;'></div>
                  </div>
                  <span class='mono' style='font-size:11px;width:40px;'>{r['prob_h']:.0%}</span>
                </div>
                <div style='display:flex;gap:6px;align-items:center;margin-top:4px;'>
                  <span style='font-size:11px;color:#8b949e;width:30px;'>D</span>
                  <div class='prob-bar-wrap' style='flex:1;'>
                    <div class='prob-bar' style='width:{pd_w}%;background:#ffd600;'></div>
                  </div>
                  <span class='mono' style='font-size:11px;width:40px;'>{r['prob_d']:.0%}</span>
                </div>
                <div style='display:flex;gap:6px;align-items:center;margin-top:4px;'>
                  <span style='font-size:11px;color:#8b949e;width:30px;'>A</span>
                  <div class='prob-bar-wrap' style='flex:1;'>
                    <div class='prob-bar' style='width:{pa_w}%;background:#ff1744;'></div>
                  </div>
                  <span class='mono' style='font-size:11px;width:40px;'>{r['prob_a']:.0%}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Tabela compacta
        with st.expander("📋 Ver tabela completa"):
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
    st.markdown("### BACKTESTING — BRASILEIRÃO 2025/2026")

    try:
        df_bt = pd.read_csv(BACKTESTING_PATH)
        bets  = df_bt[df_bt["bet_on"].notna()].copy()

        if len(bets) == 0:
            st.warning("Nenhuma aposta no backtesting. Rode backtesting.py.")
        else:
            # KPIs
            roi_f    = (df_bt["bankroll_flat"].iloc[-1]  - 1000) / 1000 * 100
            roi_k    = (df_bt["bankroll_kelly"].iloc[-1] - 1000) / 1000 * 100
            hit_rate = bets["won"].mean()
            yld_f    = bets["pl_flat"].sum() / bets["stake_flat"].sum() * 100

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: st.metric("📊 Apostas",    len(bets))
            with k2: st.metric("✅ Hit Rate",   f"{hit_rate:.1%}")
            with k3: st.metric("💰 ROI Flat",   f"{roi_f:+.1f}%")
            with k4: st.metric("🚀 ROI Kelly",  f"{roi_k:+.1f}%")
            with k5: st.metric("📈 Yield",      f"{yld_f:+.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Evolução bankroll
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_bt["bankroll_flat"], mode="lines",
                name="Flat (2%)", line=dict(color="#4fc3f7", width=2)))
            fig.add_trace(go.Scatter(
                y=df_bt["bankroll_kelly"], mode="lines",
                name="Kelly (25%)", line=dict(color="#81c784", width=2)))
            fig.add_hline(y=1000, line_dash="dash",
                          line_color="#555577", annotation_text="Bankroll inicial")
            fig.update_layout(
                paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", family="DM Sans"),
                title="Evolução do Bankroll",
                legend=dict(bgcolor="#0d1117", bordercolor="#1c2333"),
                xaxis=dict(gridcolor="#1c2333"),
                yaxis=dict(gridcolor="#1c2333"),
                height=350, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # P&L por aposta
            fig2 = go.Figure()
            colors_pl = ["#66bb6a" if v >= 0 else "#ef5350" for v in bets["pl_flat"]]
            fig2.add_trace(go.Bar(
                y=bets["pl_flat"], marker_color=colors_pl,
                name="P&L por aposta", opacity=0.8))
            fig2.add_trace(go.Scatter(
                y=bets["pl_flat"].cumsum(), mode="lines",
                name="P&L acumulado", line=dict(color="white", width=2)))
            fig2.add_hline(y=0, line_dash="dash", line_color="#555577")
            fig2.update_layout(
                paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", family="DM Sans"),
                title="P&L por Aposta (Flat)",
                legend=dict(bgcolor="#0d1117", bordercolor="#1c2333"),
                xaxis=dict(gridcolor="#1c2333"),
                yaxis=dict(gridcolor="#1c2333"),
                height=300, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Tabela de apostas
            with st.expander("📋 Ver todas as apostas"):
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
    st.markdown("### VALOR DE MERCADO — BRASILEIRÃO 2026")
    df_mv = load_market()

    if df_mv.empty:
        st.warning("market_values.csv não encontrado.")
    else:
        df_mv = df_mv.sort_values("market_value", ascending=False)

        total = df_mv["market_value"].sum()
        top_team = df_mv.iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("💎 Time mais valioso", top_team["team"], f"€{top_team['market_value']:.1f}M")
        with c2: st.metric("💰 Total da liga", f"€{total:.1f}M")
        with c3: st.metric("📊 Média por time", f"€{total/len(df_mv):.1f}M")

        st.markdown("<br>", unsafe_allow_html=True)

        import plotly.express as px
        fig = px.bar(
            df_mv.head(20), x="market_value", y="team",
            orientation="h", color="market_value",
            color_continuous_scale=["#1c2333", "#2979ff", "#00e676"],
            labels={"market_value": "Valor (€M)", "team": ""},
        )
        fig.update_layout(
            paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="DM Sans"),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            height=500, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
