import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH   = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
OUTPUT_PATH = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
MARKET_VALUES_PATH = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"

# MELHORIA 3: Times que disputam Libertadores por temporada
# Adicione as listas aqui a cada temporada nova
LIBERTADORES_BY_SEASON = {
    2023: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "RB Bragantino"},
    2024: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "SE Palmeiras"},
    2025: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "SE Palmeiras"},
    2026: {"CR Flamengo", "Fluminense FC", "CA Mineiro", "São Paulo FC", "SC Internacional"},
}

# MELHORIA 3: penalidade de Libertadores (pode ser ajustada)
LIBERTADORES_PENALTY_FEAT = 1   # feature binária — o modelo aprende a magnitude


# ──────────────────────────────────────────────
# ELO com K dinâmico (MELHORIA 1)
# ──────────────────────────────────────────────

K_HISTORICAL = 32
K_CURRENT    = 48
SEASON_REF   = 2026   # temporada mais recente

def expected_elo(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a, k=32):
    ea = expected_elo(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))

def update_elo_dynamic(ra, rb, score_a, season):
    """
    MELHORIA 1: K maior para temporada mais recente.
    Isso faz o ELO responder mais rápido a resultados de 2026.
    """
    k = K_CURRENT if season == SEASON_REF else K_HISTORICAL
    return update_elo(ra, rb, score_a, k=k)

def build_elo_ratings(df):
    ratings = {}
    elo_history = {}
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in ratings: ratings[home] = 1500
        if away not in ratings: ratings[away] = 1500
        elo_history[row["match_id"]] = {
            "home_elo": ratings[home],
            "away_elo": ratings[away],
        }
        if row["home_goals"] > row["away_goals"]:    score = 1.0
        elif row["home_goals"] == row["away_goals"]: score = 0.5
        else:                                         score = 0.0
        # MELHORIA 1: K dinâmico por temporada
        ratings[home], ratings[away] = update_elo_dynamic(
            ratings[home], ratings[away], score, season=int(row["season"])
        )
    return elo_history


# ──────────────────────────────────────────────
# TABELA EM TEMPO REAL
# ──────────────────────────────────────────────

def build_live_table(past_df: pd.DataFrame, season: int) -> dict:
    """
    Constrói a tabela de classificação até o momento,
    apenas com jogos da mesma temporada.
    Retorna dict com posição, pontos, aproveitamento de cada time.
    """
    season_past = past_df[past_df["season"] == season]
    table = {}

    for _, row in season_past.iterrows():
        home, away = row["home_team"], row["away_team"]
        for t in [home, away]:
            if t not in table:
                table[t] = {"pts": 0, "gf": 0, "ga": 0, "played": 0}

        table[home]["gf"] += row["home_goals"]
        table[home]["ga"] += row["away_goals"]
        table[away]["gf"] += row["away_goals"]
        table[away]["ga"] += row["home_goals"]
        table[home]["played"] += 1
        table[away]["played"] += 1

        if row["home_goals"] > row["away_goals"]:
            table[home]["pts"] += 3
        elif row["home_goals"] == row["away_goals"]:
            table[home]["pts"] += 1
            table[away]["pts"] += 1
        else:
            table[away]["pts"] += 3

    if not table:
        return {}

    # Ordenar e calcular posição e aproveitamento
    sorted_teams = sorted(
        table.items(),
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )

    result = {}
    for pos, (team, stats) in enumerate(sorted_teams, 1):
        played = stats["played"] or 1
        result[team] = {
            "position":       pos,
            "pts":            stats["pts"],
            "aproveitamento": stats["pts"] / (played * 3),
            "gd":             stats["gf"] - stats["ga"],
            "played":         played,
        }

    return result


# ──────────────────────────────────────────────
# MELHORIA 4: Tendência de posição
# ──────────────────────────────────────────────

def build_position_trend(team: str, past_df: pd.DataFrame, season: int, n_rounds: int = 5) -> float:
    """
    Calcula a tendência de posição do time nas últimas N rodadas da temporada.
    Retorna:
        negativo → time subindo (bom sinal)
        positivo → time caindo (mau sinal)
        0        → estável ou dados insuficientes
    """
    season_df = past_df[past_df["season"] == season].copy()

    if "matchday" not in season_df.columns or season_df.empty:
        return 0.0

    rounds = sorted(season_df["matchday"].dropna().unique())
    if len(rounds) < 2:
        return 0.0

    # Pegar as últimas N rodadas disputadas
    recent_rounds = rounds[-n_rounds:] if len(rounds) >= n_rounds else rounds

    positions_by_round = []
    for rd in recent_rounds:
        rd_df = season_df[season_df["matchday"] <= rd]
        table = build_live_table(rd_df, season)
        pos = table.get(team, {}).get("position", 10)
        positions_by_round.append(pos)

    if len(positions_by_round) < 2:
        return 0.0

    # Regressão linear simples sobre as posições
    x = np.arange(len(positions_by_round))
    slope = np.polyfit(x, positions_by_round, 1)[0]
    return float(round(slope, 3))   # positivo = caindo, negativo = subindo


# ──────────────────────────────────────────────
# STATS DO TIME
# ──────────────────────────────────────────────

def calc_team_stats(team: str, past_df: pd.DataFrame, n_short=5, n_long=10) -> dict:
    games = past_df[
        (past_df["home_team"] == team) | (past_df["away_team"] == team)
    ]

    def extract(games_slice):
        pts, gf, ga = [], [], []
        home_pts, away_pts = [], []
        wins = draws = losses = 0
        for _, g in games_slice.iterrows():
            is_home = g["home_team"] == team
            g_gf = g["home_goals"] if is_home else g["away_goals"]
            g_ga = g["away_goals"] if is_home else g["home_goals"]
            gf.append(g_gf); ga.append(g_ga)
            if g_gf > g_ga:    p = 3; wins += 1
            elif g_gf == g_ga: p = 1; draws += 1
            else:              p = 0; losses += 1
            pts.append(p)
            if is_home: home_pts.append(p)
            else:       away_pts.append(p)
        n = len(pts) or 1
        return {
            "form_pts":  np.mean(pts) if pts else 1.0,
            "avg_gf":    np.mean(gf)  if gf  else 1.0,
            "avg_ga":    np.mean(ga)  if ga  else 1.0,
            "goal_diff": np.mean(gf) - np.mean(ga) if gf else 0.0,
            "win_rate":  wins / n,
            "draw_rate": draws / n,
            "home_form": np.mean(home_pts) if home_pts else 1.0,
            "away_form": np.mean(away_pts) if away_pts else 1.0,
        }

    short = extract(games.tail(n_short))
    long  = extract(games.tail(n_long))

    return {
        "form_pts":    short["form_pts"],
        "avg_gf":      short["avg_gf"],
        "avg_ga":      short["avg_ga"],
        "goal_diff":   short["goal_diff"],
        "win_rate":    short["win_rate"],
        "draw_rate":   short["draw_rate"],
        "form_pts_10": long["form_pts"],
        "avg_gf_10":   long["avg_gf"],
        "avg_ga_10":   long["avg_ga"],
        "win_rate_10": long["win_rate"],
        "home_form":   short["home_form"],
        "away_form":   short["away_form"],
    }


# ──────────────────────────────────────────────
# BUILD FEATURES
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "FINISHED"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo_history = build_elo_ratings(df)

    # Carregar valores de mercado uma única vez (fora do loop)
    mv = pd.read_csv(MARKET_VALUES_PATH)[["team", "market_value_log", "market_value_norm", "squad_size"]]
    mv_dict = mv.set_index("team").to_dict("index")

    def get_mv(team):
        return mv_dict.get(team, {
            "market_value_log":  3.0,
            "market_value_norm": 0.3,
            "squad_size":        25,
        })

    rows = []
    for idx, row in df.iterrows():
        home, away   = row["home_team"], row["away_team"]
        season       = int(row["season"])
        past         = df.iloc[:idx]

        hs  = calc_team_stats(home, past)
        as_ = calc_team_stats(away, past)

        # Tabela ao vivo da temporada atual
        live_table   = build_live_table(past, season)
        default_pos  = {"position": 10, "aproveitamento": 0.33, "gd": 0, "pts": 0}
        h_table      = live_table.get(home, default_pos)
        a_table      = live_table.get(away, default_pos)

        # H2H
        h2h = past[
            ((past["home_team"] == home) & (past["away_team"] == away)) |
            ((past["home_team"] == away) & (past["away_team"] == home))
        ]
        h2h_hw = len(h2h[
            ((h2h["home_team"] == home) & (h2h["home_goals"] > h2h["away_goals"])) |
            ((h2h["away_team"] == home) & (h2h["away_goals"] > h2h["home_goals"]))
        ])
        h2h_aw = len(h2h[
            ((h2h["home_team"] == away) & (h2h["home_goals"] > h2h["away_goals"])) |
            ((h2h["away_team"] == away) & (h2h["away_goals"] > h2h["home_goals"]))
        ])
        h2h_draws = len(h2h[h2h["home_goals"] == h2h["away_goals"]])

        # Elo
        elos = elo_history.get(row["match_id"], {"home_elo": 1500, "away_elo": 1500})

        # Resultado
        if row["home_goals"] > row["away_goals"]:   result = "H"
        elif row["home_goals"] < row["away_goals"]: result = "A"
        else:                                        result = "D"

        h_mv = get_mv(home)
        a_mv = get_mv(away)

        # MELHORIA 3: feature de Libertadores
        liberta_set = LIBERTADORES_BY_SEASON.get(season, set())
        home_liberta = 1 if home in liberta_set else 0
        away_liberta = 1 if away in liberta_set else 0

        # MELHORIA 4: tendência de posição (slope das últimas 5 rodadas)
        h_trend = build_position_trend(home, past, season, n_rounds=5)
        a_trend = build_position_trend(away, past, season, n_rounds=5)

        # MELHORIA 5: features de equilíbrio para detecção de empates
        # aprov_equilibrio: 1.0 quando aproveitamentos são iguais, cai conforme diferem
        aprov_equilibrio = 1.0 / (
            1 + abs(h_table["aproveitamento"] - a_table["aproveitamento"])
        )
        # h2h_draw_dominance: fração de empates no histórico entre os dois times
        h2h_total = h2h_hw + h2h_aw + h2h_draws
        h2h_draw_dominance = h2h_draws / max(h2h_total, 1)

        rows.append({
            # Identificadores
            "match_id":   row["match_id"],
            "date":       row["date"],
            "matchday":   row.get("matchday"),
            "season":     season,
            "home_team":  home,
            "away_team":  away,
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "result":     result,
            # Elo
            "home_elo":   elos["home_elo"],
            "away_elo":   elos["away_elo"],
            "elo_diff":   elos["home_elo"] - elos["away_elo"],
            # Posição na tabela
            "home_position":      h_table["position"],
            "away_position":      a_table["position"],
            "home_aproveitamento": h_table["aproveitamento"],
            "away_aproveitamento": a_table["aproveitamento"],
            "home_table_gd":      h_table["gd"],
            "away_table_gd":      a_table["gd"],
            "position_diff":      a_table["position"] - h_table["position"],
            # Features curto prazo
            "home_form_pts":   hs["form_pts"],
            "home_avg_gf":     hs["avg_gf"],
            "home_avg_ga":     hs["avg_ga"],
            "home_goal_diff":  hs["goal_diff"],
            "home_win_rate":   hs["win_rate"],
            "home_draw_rate":  hs["draw_rate"],
            "home_home_form":  hs["home_form"],
            "away_form_pts":   as_["form_pts"],
            "away_avg_gf":     as_["avg_gf"],
            "away_avg_ga":     as_["avg_ga"],
            "away_goal_diff":  as_["goal_diff"],
            "away_win_rate":   as_["win_rate"],
            "away_draw_rate":  as_["draw_rate"],
            "away_away_form":  as_["away_form"],
            # Features longo prazo
            "home_form_pts_10": hs["form_pts_10"],
            "home_avg_gf_10":   hs["avg_gf_10"],
            "home_avg_ga_10":   hs["avg_ga_10"],
            "home_win_rate_10": hs["win_rate_10"],
            "away_form_pts_10": as_["form_pts_10"],
            "away_avg_gf_10":   as_["avg_gf_10"],
            "away_avg_ga_10":   as_["avg_ga_10"],
            "away_win_rate_10": as_["win_rate_10"],
            # H2H
            "h2h_home_wins": h2h_hw,
            "h2h_away_wins": h2h_aw,
            "h2h_draws":     h2h_draws,
            # Valor de mercado
            "home_market_value_log":  h_mv["market_value_log"],
            "home_market_value_norm": h_mv["market_value_norm"],
            "home_squad_size":        h_mv["squad_size"],
            "away_market_value_log":  a_mv["market_value_log"],
            "away_market_value_norm": a_mv["market_value_norm"],
            "away_squad_size":        a_mv["squad_size"],
            "market_value_diff":      h_mv["market_value_log"] - a_mv["market_value_log"],
            # MELHORIA 3: Libertadores
            "home_joga_libertadores": home_liberta,
            "away_joga_libertadores": away_liberta,
            # MELHORIA 4: tendência de posição
            "home_pos_trend": h_trend,
            "away_pos_trend": a_trend,
            "pos_trend_diff": h_trend - a_trend,
            # MELHORIA 5: equilíbrio — ajuda na detecção de empates
            "aprov_equilibrio":    aprov_equilibrio,
            "h2h_draw_dominance":  h2h_draw_dominance,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("📊 Lendo dados...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df)} partidas carregadas")

    print("⚙️  Gerando features...")
    features = build_features(df)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features salvas! Shape: {features.shape}")
    print(features[["home_team","away_team","result",
                     "home_position","away_position",
                     "home_aproveitamento","elo_diff",
                     "home_joga_libertadores","home_pos_trend",
                     "aprov_equilibrio","h2h_draw_dominance"]].head(10))