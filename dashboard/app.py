import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(r"C:\PREDICTOR\models")
sys.path.append(r"C:\PREDICTOR\processing")

# ── Configuração da página ──
st.set_page_config(
    page_title="Brasileirão 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──
MODEL_PATH         = r"C:\PREDICTOR\models\match_model.pkl"
POISSON_MODEL_PATH = r"C:\PREDICTOR\models\poisson_model.pkl"
MATCHES_PATH       = r"C:\PREDICTOR\scraping\data\raw\matches.csv"
FEATURES_PATH      = r"C:\PREDICTOR\scraping\data\processed\features.csv"
MARKET_PATH        = r"C:\PREDICTOR\scraping\data\external\market_values.csv"
SIMULATION_PATH    = r"C:\PREDICTOR\scraping\data\processed\simulacao_2026.csv"

# ── CSS customizado ──
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d5986;
        text-align: center;
    }
    .title-green  { color: #00c853; font-weight: bold; }
    .title-red    { color: #ff1744; font-weight: bold; }
    .title-yellow { color: #ffd600; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Carregar dados ──
@st.cache_resource
def load_models():
    saved        = joblib.load(MODEL_PATH)
    poisson_saved = joblib.load(POISSON_MODEL_PATH)
    return saved["model"], saved["features"], saved["label_encoder"], poisson_saved

@st.cache_data
def load_data():
    matches  = pd.read_csv(MATCHES_PATH)
    features = pd.read_csv(FEATURES_PATH)
    market   = pd.read_csv(MARKET_PATH)
    sim      = pd.read_csv(SIMULATION_PATH)
    return matches, features, market, sim

model, feature_cols, le, poisson_saved = load_models()
matches, features, market, sim = load_data()

df_2026  = matches[matches["season"] == 2026].copy()
played   = df_2026[df_2026["status"] == "FINISHED"].copy()
fixtures = df_2026[df_2026["status"] != "FINISHED"].copy()
mv_dict  = market.set_index("team").to_dict("index")

# ── Tabela real ──
def build_real_table():
    table = {}
    for _, row in played.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        for t in [home, away]:
            if t not in table:
                table[t] = {"pts":0,"gf":0,"ga":0,"w":0,"d":0,"l":0,"played":0}
        table[home]["gf"]+=hg; table[home]["ga"]+=ag; table[home]["played"]+=1
        table[away]["gf"]+=ag; table[away]["ga"]+=hg; table[away]["played"]+=1
        if hg>ag:   table[home]["pts"]+=3; table[home]["w"]+=1; table[away]["l"]+=1
        elif hg==ag: table[home]["pts"]+=1; table[away]["pts"]+=1; table[home]["d"]+=1; table[away]["d"]+=1
        else:        table[away]["pts"]+=3; table[away]["w"]+=1; table[home]["l"]+=1
    df = pd.DataFrame(table).T.reset_index()
    df.columns = ["team","pts","gf","ga","w","d","l","played"]
    df["gd"] = df["gf"] - df["ga"]
    return df.sort_values(["pts","gd","gf"], ascending=False).reset_index(drop=True)

real_table = build_real_table()
real_table.index += 1

# ── Previsão de partida ──
def predict_match(home, away):
    def get_mv(team):
        return mv_dict.get(team, {"market_value_log":3.0,"market_value_norm":0.3,"squad_size":25})

    def get_team_stats_from_features(team):
        team_feats = features[
            (features["home_team"] == team) | (features["away_team"] == team)
        ].tail(1)
        if team_feats.empty:
            return {
                "form_pts":1.0,"avg_gf":1.2,"avg_ga":1.2,"goal_diff":0.0,
                "win_rate":0.33,"draw_rate":0.33,"home_form":1.0,"away_form":1.0,
                "form_pts_10":1.0,"avg_gf_10":1.2,"avg_ga_10":1.2,"win_rate_10":0.33,
                "elo":1500.0,
            }
        row = team_feats.iloc[0]
        if row["home_team"] == team:
            return {
                "form_pts":   row.get("home_form_pts",1.0),
                "avg_gf":     row.get("home_avg_gf",1.2),
                "avg_ga":     row.get("home_avg_ga",1.2),
                "goal_diff":  row.get("home_goal_diff",0.0),
                "win_rate":   row.get("home_win_rate",0.33),
                "draw_rate":  row.get("home_draw_rate",0.33),
                "home_form":  row.get("home_home_form",1.0),
                "away_form":  row.get("away_away_form",1.0),
                "form_pts_10":row.get("home_form_pts_10",1.0),
                "avg_gf_10":  row.get("home_avg_gf_10",1.2),
                "avg_ga_10":  row.get("home_avg_ga_10",1.2),
                "win_rate_10":row.get("home_win_rate_10",0.33),
                "elo":        row.get("home_elo",1500.0),
            }
        else:
            return {
                "form_pts":   row.get("away_form_pts",1.0),
                "avg_gf":     row.get("away_avg_gf",1.2),
                "avg_ga":     row.get("away_avg_ga",1.2),
                "goal_diff":  row.get("away_goal_diff",0.0),
                "win_rate":   row.get("away_win_rate",0.33),
                "draw_rate":  row.get("away_draw_rate",0.33),
                "home_form":  row.get("home_home_form",1.0),
                "away_form":  row.get("away_away_form",1.0),
                "form_pts_10":row.get("away_form_pts_10",1.0),
                "avg_gf_10":  row.get("away_avg_gf_10",1.2),
                "avg_ga_10":  row.get("away_avg_ga_10",1.2),
                "win_rate_10":row.get("away_win_rate_10",0.33),
                "elo":        row.get("away_elo",1500.0),
            }

    hs  = get_team_stats_from_features(home)
    as_ = get_team_stats_from_features(away)
    h_mv = get_mv(home)
    a_mv = get_mv(away)

    row = {
        "elo_diff":           hs["elo"] - as_["elo"],
        "home_elo":           hs["elo"],
        "away_elo":           as_["elo"],
        "home_market_value_log":  h_mv["market_value_log"],
        "home_market_value_norm": h_mv["market_value_norm"],
        "home_squad_size":        h_mv["squad_size"],
        "away_market_value_log":  a_mv["market_value_log"],
        "away_market_value_norm": a_mv["market_value_norm"],
        "away_squad_size":        a_mv["squad_size"],
        "market_value_diff":      h_mv["market_value_log"] - a_mv["market_value_log"],
        "home_position":      5, "away_position":    10,
        "home_aproveitamento":0.5,"away_aproveitamento":0.4,
        "home_table_gd":      0, "away_table_gd":    0,
        "position_diff":      5,
        "home_form_pts":   hs["form_pts"],   "home_avg_gf":     hs["avg_gf"],
        "home_avg_ga":     hs["avg_ga"],     "home_goal_diff":  hs["goal_diff"],
        "home_win_rate":   hs["win_rate"],   "home_draw_rate":  hs["draw_rate"],
        "home_home_form":  hs["home_form"],
        "away_form_pts":   as_["form_pts"],  "away_avg_gf":     as_["avg_gf"],
        "away_avg_ga":     as_["avg_ga"],    "away_goal_diff":  as_["goal_diff"],
        "away_win_rate":   as_["win_rate"],  "away_draw_rate":  as_["draw_rate"],
        "away_away_form":  as_["away_form"],
        "home_form_pts_10":hs["form_pts_10"],"home_avg_gf_10":  hs["avg_gf_10"],
        "home_avg_ga_10":  hs["avg_ga_10"],  "home_win_rate_10":hs["win_rate_10"],
        "away_form_pts_10":as_["form_pts_10"],"away_avg_gf_10": as_["avg_gf_10"],
        "away_avg_ga_10":  as_["avg_ga_10"], "away_win_rate_10":as_["win_rate_10"],
        "h2h_home_wins":0, "h2h_away_wins":0, "h2h_draws":0,
    }

    X         = pd.DataFrame([row])[feature_cols]
    probs_enc = model.predict_proba(X)[0]
    probs     = {le.classes_[i]: probs_enc[i] for i in range(len(le.classes_))}

    # Gols esperados via Poisson
    pf = {
        "elo_diff":       hs["elo"]-as_["elo"], "home_elo": hs["elo"], "away_elo": as_["elo"],
        "home_avg_gf":    hs["avg_gf"],  "home_avg_ga":    hs["avg_ga"],
        "home_goal_diff": hs["goal_diff"],"away_avg_gf":    as_["avg_gf"],
        "away_avg_ga":    as_["avg_ga"], "away_goal_diff": as_["goal_diff"],
        "home_form_pts":  hs["form_pts"],"away_form_pts":  as_["form_pts"],
        "home_avg_gf_10": hs["avg_gf_10"],"home_avg_ga_10": hs["avg_ga_10"],
        "away_avg_gf_10": as_["avg_gf_10"],"away_avg_ga_10": as_["avg_ga_10"],
        "home_aproveitamento":0.5,"away_aproveitamento":0.4,"position_diff":5,
    }
    pm    = poisson_saved["model_home"]
    pa    = poisson_saved["model_away"]
    sc    = poisson_saved["scaler"]
    feats = poisson_saved["features"]
    Xp    = pd.DataFrame([pf])[feats]
    Xpa   = Xp.copy(); Xpa["elo_diff"] = -Xpa["elo_diff"]; Xpa["position_diff"] = -Xpa["position_diff"]
    lam_h = max(0.2, pm.predict(sc.transform(Xp))[0])
    lam_a = max(0.2, pa.predict(sc.transform(Xpa))[0])

    return probs, lam_h, lam_a


# ════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════

st.title("⚽ Brasileirão 2026 — Predictor & Simulador")
st.caption("Modelo preditivo com LightGBM calibrado + Monte Carlo | Acurácia: 50.3%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tabela & Simulação",
    "🔮 Prever Partida",
    "📈 Análise de Times",
    "💰 Valor de Mercado",
    "💎 Value bets"
])


# ════════════════════════════════════════════
# TAB 1 — TABELA & SIMULAÇÃO
# ════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Tabela Atual")
        def color_table(df):
            colors = []
            for i in range(len(df)):
                if i < 4:   colors.append("background-color: #0d3b0d")    # Libertadores
                elif i < 6: colors.append("background-color: #1a3a0d")    # Sul-Americana
                elif i >= len(df)-4: colors.append("background-color: #3b0d0d")  # Rebaixamento
                else:       colors.append("")
            return colors

        st.dataframe(
            real_table[["team","pts","played","w","d","l","gf","ga","gd"]].rename(columns={
                "team":"Time","pts":"Pts","played":"J","w":"V","d":"E","l":"D",
                "gf":"GM","ga":"GS","gd":"SG"
            }),
            use_container_width=True, height=720, hide_index=False
        )
        st.caption("🟢 Libertadores | 🟡 Sul-Americana | 🔴 Rebaixamento")

    with col2:
        st.subheader("🎲 Probabilidades Monte Carlo")

        sim_display = sim.copy()
        sim_display.index = range(1, len(sim_display)+1)

        # Gráfico título
        fig_title = px.bar(
            sim_display.head(10),
            x="time", y="titulo_%",
            color="titulo_%",
            color_continuous_scale="Greens",
            title="% Probabilidade de Título",
            labels={"titulo_%": "%", "time": ""},
        )
        fig_title.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_title, use_container_width=True)

        # Gráfico rebaixamento
        fig_rell = px.bar(
            sim_display.sort_values("rebaixamento_%", ascending=False).head(8),
            x="time", y="rebaixamento_%",
            color="rebaixamento_%",
            color_continuous_scale="Reds",
            title="% Probabilidade de Rebaixamento",
            labels={"rebaixamento_%": "%", "time": ""},
        )
        fig_rell.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_rell, use_container_width=True)

    # Tabela completa Monte Carlo
    st.subheader("📊 Tabela Completa de Probabilidades")
    st.dataframe(
        sim_display.rename(columns={
            "time":"Time","titulo_%":"Título %","libertadores_%":"Libertadores %",
            "sulamericana_%":"Sul-Americana %","rebaixamento_%":"Rebaixamento %",
            "pts_esperados":"Pts Esperados"
        }),
        use_container_width=True, height=400
    )


# ════════════════════════════════════════════
# TAB 2 — PREVER PARTIDA
# ════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Previsão de Partida")

    teams = sorted(df_2026["home_team"].unique())

    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        home_team = st.selectbox("🏠 Time Mandante", teams, index=0)
    with col2:
        st.markdown("<br><h3 style='text-align:center'>VS</h3>", unsafe_allow_html=True)
    with col3:
        away_team = st.selectbox("✈️ Time Visitante", teams, index=1)

    if home_team == away_team:
        st.warning("Selecione times diferentes!")
    else:
        if st.button("🔮 Prever Partida", type="primary", use_container_width=True):
            with st.spinner("Calculando previsões..."):
                probs, lam_h, lam_a = predict_match(home_team, away_team)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"🏠 {home_team}", f"{probs.get('H',0):.1%}", "Vitória mandante")
            with c2:
                st.metric("🤝 Empate", f"{probs.get('D',0):.1%}", "")
            with c3:
                st.metric(f"✈️ {away_team}", f"{probs.get('A',0):.1%}", "Vitória visitante")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("⚽ Gols esperados mandante", f"{lam_h:.2f}")
            with c2:
                st.metric("⚽ Gols esperados visitante", f"{lam_a:.2f}")

            # Gráfico de probabilidades
            fig = go.Figure(go.Bar(
                x=[home_team, "Empate", away_team],
                y=[probs.get("H",0)*100, probs.get("D",0)*100, probs.get("A",0)*100],
                marker_color=["#00c853", "#ffd600", "#ff1744"],
                text=[f"{probs.get('H',0):.1%}", f"{probs.get('D',0):.1%}", f"{probs.get('A',0):.1%}"],
                textposition="outside"
            ))
            fig.update_layout(
                title="Probabilidades da Partida",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", yaxis_title="%",
                yaxis=dict(range=[0,100])
            )
            st.plotly_chart(fig, use_container_width=True)

            # Matriz de placar mais provável
            st.subheader("🎯 Placares Mais Prováveis")
            from scipy.stats import poisson
            max_goals = 5
            prob_matrix = np.zeros((max_goals+1, max_goals+1))
            for i in range(max_goals+1):
                for j in range(max_goals+1):
                    prob_matrix[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)

            score_probs = []
            for i in range(max_goals+1):
                for j in range(max_goals+1):
                    score_probs.append({
                        "Placar": f"{i} x {j}",
                        "Prob %": round(prob_matrix[i][j]*100, 2)
                    })

            score_df = pd.DataFrame(score_probs).sort_values("Prob %", ascending=False).head(10)
            st.dataframe(score_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 3 — ANÁLISE DE TIMES
# ════════════════════════════════════════════
with tab3:
    st.subheader("📈 Análise de Time")

    team_sel = st.selectbox("Selecione o time", sorted(df_2026["home_team"].unique()))

    team_matches = matches[
        ((matches["home_team"] == team_sel) | (matches["away_team"] == team_sel)) &
        (matches["status"] == "FINISHED")
    ].copy()
    team_matches["date"] = pd.to_datetime(team_matches["date"])
    team_matches = team_matches.sort_values("date")

    # Calcular resultado para o time
    def get_result(row):
        if row["home_team"] == team_sel:
            if row["home_goals"] > row["away_goals"]: return "V"
            if row["home_goals"] < row["away_goals"]: return "D"
            return "E"
        else:
            if row["away_goals"] > row["home_goals"]: return "V"
            if row["away_goals"] < row["home_goals"]: return "D"
            return "E"

    def get_gf(row):
        return row["home_goals"] if row["home_team"] == team_sel else row["away_goals"]

    def get_ga(row):
        return row["away_goals"] if row["home_team"] == team_sel else row["home_goals"]

    team_matches["resultado"] = team_matches.apply(get_result, axis=1)
    team_matches["gf"]        = team_matches.apply(get_gf, axis=1)
    team_matches["ga"]        = team_matches.apply(get_ga, axis=1)
    team_matches["pts"]       = team_matches["resultado"].map({"V":3,"E":1,"D":0})
    team_matches["pts_acc"]   = team_matches["pts"].cumsum()
    team_matches["oponente"]  = team_matches.apply(
        lambda r: r["away_team"] if r["home_team"]==team_sel else r["home_team"], axis=1
    )

    # Métricas gerais
    total = len(team_matches)
    wins  = (team_matches["resultado"]=="V").sum()
    draws = (team_matches["resultado"]=="E").sum()
    losses= (team_matches["resultado"]=="D").sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Jogos", total)
    c2.metric("Vitórias", wins)
    c3.metric("Empates", draws)
    c4.metric("Derrotas", losses)

    # Gráfico de pontos acumulados
    fig_pts = px.line(
        team_matches, x="date", y="pts_acc",
        title=f"Pontos Acumulados — {team_sel}",
        labels={"pts_acc":"Pontos","date":"Data"},
        markers=True
    )
    fig_pts.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white"
    )
    st.plotly_chart(fig_pts, use_container_width=True)

    # Gols por jogo
    fig_gols = go.Figure()
    fig_gols.add_trace(go.Bar(
        x=team_matches["date"], y=team_matches["gf"],
        name="Gols Marcados", marker_color="#00c853"
    ))
    fig_gols.add_trace(go.Bar(
        x=team_matches["date"], y=-team_matches["ga"],
        name="Gols Sofridos", marker_color="#ff1744"
    ))
    fig_gols.update_layout(
        title=f"Gols por Jogo — {team_sel}",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", barmode="relative"
    )
    st.plotly_chart(fig_gols, use_container_width=True)

    # Últimos 10 jogos
    st.subheader("📋 Últimos 10 Jogos")
    last10 = team_matches.tail(10)[["date","oponente","gf","ga","resultado","pts"]].copy()
    last10["date"] = last10["date"].dt.strftime("%d/%m/%Y")
    last10.columns = ["Data","Oponente","GM","GS","Resultado","Pts"]
    st.dataframe(last10, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 4 — VALOR DE MERCADO
# ════════════════════════════════════════════
with tab4:
    st.subheader("💰 Valor de Mercado dos Elencos")

    market_sorted = market.sort_values("market_value", ascending=False)

    fig_mv = px.bar(
        market_sorted,
        x="team", y="market_value",
        color="market_value",
        color_continuous_scale="Blues",
        title="Valor de Mercado por Time (€ Milhões)",
        labels={"market_value":"Valor (€M)","team":""},
        text="market_value"
    )
    fig_mv.update_traces(texttemplate="€%{text:.1f}M", textposition="outside")
    fig_mv.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", coloraxis_showscale=False,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_mv, use_container_width=True)

    # Correlação valor de mercado vs desempenho
    perf = real_table[["team","pts"]].copy()
    mv_perf = perf.merge(market[["team","market_value"]], on="team", how="left")

    fig_corr = px.scatter(
        mv_perf, x="market_value", y="pts",
        text="team", size="market_value",
        color="pts", color_continuous_scale="RdYlGn",
        title="Correlação: Valor de Mercado vs Pontos",
        labels={"market_value":"Valor (€M)","pts":"Pontos"}
    )
    fig_corr.update_traces(textposition="top center")
    fig_corr.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", coloraxis_showscale=False
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("📋 Tabela Completa")
    st.dataframe(
        market_sorted[["team","market_value","squad_size","top_player_value","market_value_norm"]].rename(columns={
            "team":"Time","market_value":"Valor (€M)","squad_size":"Elenco",
            "top_player_value":"Top Jogador (€M)","market_value_norm":"Norm"
        }),
        use_container_width=True, hide_index=True
    )

with tab5:
    st.subheader("💎 Value Bets — Modelo vs Mercado")
    st.caption("Jogos onde o modelo vê probabilidade significativamente maior que as odds implícitas")

    try:
        vb = pd.read_csv(r"C:\PREDICTOR\scraping\data\external\value_bets.csv")

        if vb.empty:
            st.info("Nenhum value bet encontrado no momento.")
        else:
            # Métricas resumo
            fortes = len(vb[vb["⭐ valor"] == "🔥 FORTE"])
            bons   = len(vb[vb["⭐ valor"] == "✅ BOM"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Value Bets", len(vb))
            c2.metric("🔥 Fortes (edge > 10%)", fortes)
            c3.metric("✅ Bons (edge 5-10%)", bons)

            st.markdown("---")

            # Filtros
            col1, col2 = st.columns(2)
            with col1:
                filtro_valor = st.multiselect(
                    "Filtrar por força",
                    ["🔥 FORTE", "✅ BOM"],
                    default=["🔥 FORTE", "✅ BOM"]
                )
            with col2:
                filtro_resultado = st.multiselect(
                    "Filtrar por resultado",
                    ["Casa", "Empate", "Fora"],
                    default=["Casa", "Empate", "Fora"]
                )

            vb_filtered = vb[
                (vb["⭐ valor"].isin(filtro_valor)) &
                (vb["resultado"].isin(filtro_resultado))
            ]

            st.dataframe(
                vb_filtered.rename(columns={
                    "data":         "Data",
                    "partida":      "Partida",
                    "casa":         "Casa de Aposta",
                    "resultado":    "Resultado",
                    "odd":          "Odd",
                    "prob_modelo":  "Prob Modelo",
                    "prob_mercado": "Prob Mercado",
                    "edge":         "Edge",
                    "roi_esperado": "ROI Esperado",
                    "⭐ valor":     "Força",
                }),
                use_container_width=True,
                hide_index=True
            )

            # Gráfico de edges
            fig_edge = px.bar(
                vb_filtered.head(15),
                x="partida", y="edge",
                color="⭐ valor",
                color_discrete_map={"🔥 FORTE": "#ff6b00", "✅ BOM": "#00c853"},
                title="Top Value Bets por Edge",
                labels={"edge": "Edge %", "partida": ""},
            )
            fig_edge.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", xaxis_tickangle=-45
            )
            st.plotly_chart(fig_edge, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Rode primeiro: `python scraping/value_bets.py`")
        st.code("cd C:\\PREDICTOR\\scraping\npython value_bets.py")