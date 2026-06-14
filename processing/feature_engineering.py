import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional

DATA_PATH          = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"
OUTPUT_PATH        = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
MARKET_VALUES_PATH = r"C:\PREDICTOR\REPO\scraping\data\external\market_values.csv"
XG_PATH            = r"C:\PREDICTOR\REPO\scraping\data\external\xg_data.csv"

# Times que disputam Libertadores por temporada
LIBERTADORES_BY_SEASON = {
    2023: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "RB Bragantino"},
    2024: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "SE Palmeiras"},
    2025: {"CR Flamengo", "CA Mineiro", "Fluminense FC", "SC Internacional", "SE Palmeiras"},
    2026: {"CR Flamengo", "Fluminense FC", "CA Mineiro", "São Paulo FC", "SC Internacional"},
}

K_HISTORICAL = 32
K_CURRENT    = 48
SEASON_REF   = 2026


# ──────────────────────────────────────────────────────────────────────────────
# ELO
# ──────────────────────────────────────────────────────────────────────────────

def expected_elo(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a, k=32):
    ea = expected_elo(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))

def update_elo_dynamic(ra, rb, score_a, season):
    k = K_CURRENT if season == SEASON_REF else K_HISTORICAL
    return update_elo(ra, rb, score_a, k=k)

def build_elo_ratings(df):
    """Calcula ELO de forma incremental — O(n) em vez de O(n²)."""
    ratings     = {}
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
        ratings[home], ratings[away] = update_elo_dynamic(
            ratings[home], ratings[away], score, season=int(row["season"])
        )
    return elo_history


# ──────────────────────────────────────────────────────────────────────────────
# ESTADO INCREMENTAL — substitui df.iloc[:idx] em todo o loop
# Cada estrutura é atualizada jogo a jogo em O(1) amortizado
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalState:
    """
    Mantém todos os estados necessários de forma incremental.
    Elimina o df.iloc[:idx] que era O(n) por jogo → O(n²) total.
    """

    def __init__(self, n_form=10):
        self.n_form = n_form

        # ── Tabela ao vivo por temporada ──────────────────────────────────────
        # season → team → {pts, gf, ga, played}
        self.table: dict[int, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "played": 0})
        )

        # ── Posição por rodada para tendência ────────────────────────────────
        # season → matchday → team → position
        # Mantemos apenas as últimas 6 rodadas por temporada (suficiente para slope-5)
        self.pos_by_round: dict[int, dict[int, dict[str, int]]] = defaultdict(dict)
        self.last_matchday: dict[int, int] = defaultdict(int)

        # ── Form (últimos N jogos) por time — deque O(1) ──────────────────────
        # team → deque de dicts {pts, gf, ga, is_home}
        self.form: dict[str, deque] = defaultdict(lambda: deque(maxlen=n_form))

        # ── H2H por par de times ──────────────────────────────────────────────
        # (teamA, teamB) → {hw: wins de A, aw: wins de B, d: draws}
        # chave sempre ordenada para economizar memória
        self.h2h: dict[tuple, dict] = defaultdict(lambda: {"hw": 0, "aw": 0, "d": 0})

        # ── Última data de jogo por time (para calcular dias de descanso) ─────
        self.last_game_date: dict[str, Optional[pd.Timestamp]] = {}

        # ── xG rolling por time (atualizado externamente se disponível) ───────
        # team → deque de dicts {xg, xga}
        self.xg_form: dict[str, deque] = defaultdict(lambda: deque(maxlen=n_form))

    def _table_key(self, t1, t2):
        return (min(t1, t2), max(t1, t2))

    # ── Snapshots da tabela para posição ──────────────────────────────────────
    def _rank_table(self, season: int) -> dict[str, int]:
        """Retorna {team: position} da temporada atual."""
        t = self.table[season]
        if not t:
            return {}
        ranked = sorted(
            t.items(),
            key=lambda x: (-x[1]["pts"], -(x[1]["gf"] - x[1]["ga"]), -x[1]["gf"])
        )
        return {team: i + 1 for i, (team, _) in enumerate(ranked)}

    def snapshot_position(self, season: int, matchday) -> None:
        """Salva snapshot de posição após cada rodada — chamado uma vez por jogo."""
        if matchday is None or pd.isna(matchday):
            return
        md = int(matchday)
        if md > self.last_matchday[season]:
            self.last_matchday[season] = md
            self.pos_by_round[season][md] = self._rank_table(season)

    # ── Leituras ──────────────────────────────────────────────────────────────
    def get_live_table(self, team: str, season: int) -> dict:
        t = self.table[season].get(team, {})
        played = t.get("played", 0) or 1
        pts    = t.get("pts", 0)
        gf     = t.get("gf", 0)
        ga     = t.get("ga", 0)
        ranked = self._rank_table(season)
        return {
            "position":       ranked.get(team, 10),
            "pts":            pts,
            "aproveitamento": pts / (played * 3),
            "gd":             gf - ga,
            "played":         played,
        }

    def get_position_trend(self, team: str, season: int, n_rounds: int = 5) -> float:
        """Slope das últimas n_rounds posições — O(n_rounds) em vez de O(n²)."""
        rounds_dict = self.pos_by_round.get(season, {})
        if len(rounds_dict) < 2:
            return 0.0
        recent_rounds = sorted(rounds_dict.keys())[-n_rounds:]
        positions = [rounds_dict[rd].get(team, 10) for rd in recent_rounds]
        if len(positions) < 2:
            return 0.0
        x = np.arange(len(positions))
        return float(round(np.polyfit(x, positions, 1)[0], 3))

    def get_team_stats(self, team: str, n: int) -> dict:
        """Estatísticas de forma — O(n) máximo, n ≤ 10."""
        games = list(self.form[team])[-n:] if n < self.n_form else list(self.form[team])
        if not games:
            return {"pts": 1.0, "gf": 1.2, "ga": 1.0, "gd": 0.2,
                    "wr": 0.4, "dr": 0.25, "home_f": 1.0, "away_f": 1.0}
        pts_l, gf_l, ga_l, wins, draws = [], [], [], 0, 0
        home_pts, away_pts = [], []
        for g in games:
            pts_l.append(g["pts"]); gf_l.append(g["gf"]); ga_l.append(g["ga"])
            if g["pts"] == 3: wins += 1
            elif g["pts"] == 1: draws += 1
            if g["is_home"]: home_pts.append(g["pts"])
            else:            away_pts.append(g["pts"])
        n_ = len(games)
        return {
            "pts":    np.mean(pts_l),
            "gf":     np.mean(gf_l),
            "ga":     np.mean(ga_l),
            "gd":     np.mean(gf_l) - np.mean(ga_l),
            "wr":     wins / n_,
            "dr":     draws / n_,
            "home_f": np.mean(home_pts) if home_pts else np.mean(pts_l),
            "away_f": np.mean(away_pts) if away_pts else np.mean(pts_l),
        }

    def get_days_rest(self, team: str, current_date: pd.Timestamp) -> int:
        """Dias desde o último jogo. Padrão 7 se for o primeiro jogo."""
        last = self.last_game_date.get(team)
        if last is None:
            return 7
        return max(0, (current_date - last).days)

    def get_pressure(self, team: str, season: int) -> dict:
        """
        Métricas de pressão situacional:
        - relegation_gap: distância à zona de rebaixamento (pos - 17), positivo = seguro
        - g6_gap: distância ao G6 (6 - pos), positivo = dentro do G6
        - g4_gap: distância ao G4 (4 - pos), positivo = dentro do G4
        """
        ranked = self._rank_table(season)
        pos = ranked.get(team, 10)
        return {
            "relegation_gap": pos - 17,   # negativo = dentro da zona
            "g6_gap":         6   - pos,  # positivo = dentro do G6
            "g4_gap":         4   - pos,  # positivo = dentro do G4
        }

    def get_xg_stats(self, team: str, n: int) -> dict:
        """xG médio dos últimos n jogos. Retorna zeros se não disponível."""
        games = list(self.xg_form[team])[-n:]
        if not games:
            return {"xg": 0.0, "xga": 0.0}
        return {
            "xg":  float(np.mean([g["xg"]  for g in games])),
            "xga": float(np.mean([g["xga"] for g in games])),
        }

    def get_h2h(self, home: str, away: str) -> dict:
        key = self._table_key(home, away)
        h   = self.h2h[key]
        # hw = vitórias de `home`, aw = vitórias de `away`
        if home <= away:
            return {"hw": h["hw"], "aw": h["aw"], "d": h["d"]}
        else:
            return {"hw": h["aw"], "aw": h["hw"], "d": h["d"]}

    # ── Atualização após cada jogo ────────────────────────────────────────────
    def update(self, home: str, away: str, hg: int, ag: int,
               season: int, matchday,
               date: pd.Timestamp = None,
               home_xg: float = None, away_xg: float = None) -> None:
        """Atualiza todos os estados — chamado depois de ler as features do jogo."""

        # Última data de jogo
        if date is not None:
            self.last_game_date[home] = date
            self.last_game_date[away] = date

        # xG rolling (se disponível)
        if home_xg is not None:
            self.xg_form[home].append({"xg": home_xg, "xga": away_xg or 0.0})
            self.xg_form[away].append({"xg": away_xg or 0.0, "xga": home_xg})

        # Tabela
        t = self.table[season]
        for team in [home, away]:
            if team not in t:
                t[team] = {"pts": 0, "gf": 0, "ga": 0, "played": 0}
        t[home]["gf"] += hg; t[home]["ga"] += ag; t[home]["played"] += 1
        t[away]["gf"] += ag; t[away]["ga"] += hg; t[away]["played"] += 1
        if hg > ag:   t[home]["pts"] += 3
        elif hg < ag: t[away]["pts"] += 3
        else:         t[home]["pts"] += 1; t[away]["pts"] += 1

        # Snapshot de posição (uma vez por rodada)
        self.snapshot_position(season, matchday)

        # Form
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if ag == hg else 0)
        self.form[home].append({"pts": h_pts, "gf": hg, "ga": ag, "is_home": True})
        self.form[away].append({"pts": a_pts, "gf": ag, "ga": hg, "is_home": False})

        # H2H
        key = self._table_key(home, away)
        if home <= away:
            if hg > ag:   self.h2h[key]["hw"] += 1
            elif ag > hg: self.h2h[key]["aw"] += 1
            else:         self.h2h[key]["d"]  += 1
        else:
            if hg > ag:   self.h2h[key]["aw"] += 1
            elif ag > hg: self.h2h[key]["hw"] += 1
            else:         self.h2h[key]["d"]  += 1


# ──────────────────────────────────────────────────────────────────────────────
# BUILD FEATURES — loop principal O(n) em vez de O(n²)
# ──────────────────────────────────────────────────────────────────────────────

def _load_xg_lookup(path: str) -> dict:
    """Carrega mapa (date, home_team, away_team) → (home_xg, away_xg) do arquivo xG."""
    xg_path = Path(path)
    if not xg_path.exists():
        print("   ℹ️  xg_data.csv não encontrado — features de xG desativadas")
        return {}
    xg_df = pd.read_csv(xg_path)
    xg_df["date"] = pd.to_datetime(xg_df["date"]).dt.date
    lookup = {}
    for _, r in xg_df.iterrows():
        key = (str(r["date"]), r["home_team"], r["away_team"])
        lookup[key] = (float(r.get("home_xg", 0)), float(r.get("away_xg", 0)))
    print(f"   ✅ xG carregados: {len(lookup)} partidas")
    return lookup


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "FINISHED"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo_history = build_elo_ratings(df)
    xg_lookup   = _load_xg_lookup(XG_PATH)

    mv = pd.read_csv(MARKET_VALUES_PATH)[
        ["team", "market_value_log", "market_value_norm", "squad_size"]
    ]
    mv_dict = mv.set_index("team").to_dict("index")
    def get_mv(team):
        return mv_dict.get(team, {
            "market_value_log": 3.0, "market_value_norm": 0.3, "squad_size": 25
        })

    state = IncrementalState(n_form=10)
    rows  = []
    total = len(df)

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   {idx}/{total} ({idx/total:.0%})...", end="\r")

        home     = row["home_team"]
        away     = row["away_team"]
        season   = int(row["season"])
        hg       = int(row["home_goals"])
        ag       = int(row["away_goals"])
        matchday = row.get("matchday")
        date_ts  = row["date"]

        # xG desta partida (se disponível)
        xg_key = (str(date_ts.date()), home, away)
        home_xg, away_xg = xg_lookup.get(xg_key, (None, None))

        # ── Ler features ANTES de atualizar o estado ──────────────────────────
        hs   = state.get_team_stats(home, 5)
        as_  = state.get_team_stats(away, 5)
        hs10 = state.get_team_stats(home, 10)
        as10 = state.get_team_stats(away, 10)

        h_table = state.get_live_table(home, season)
        a_table = state.get_live_table(away, season)

        h_trend = state.get_position_trend(home, season, n_rounds=5)
        a_trend = state.get_position_trend(away, season, n_rounds=5)

        # Dias de descanso
        h_rest = state.get_days_rest(home, date_ts)
        a_rest = state.get_days_rest(away, date_ts)

        # Pressão situacional
        h_press = state.get_pressure(home, season)
        a_press = state.get_pressure(away, season)

        # xG rolling (se disponível)
        h_xg = state.get_xg_stats(home, 5)
        a_xg = state.get_xg_stats(away, 5)

        h2h = state.get_h2h(home, away)
        h2h_hw    = h2h["hw"]
        h2h_aw    = h2h["aw"]
        h2h_draws = h2h["d"]
        h2h_total = h2h_hw + h2h_aw + h2h_draws

        elos = elo_history.get(row["match_id"], {"home_elo": 1500, "away_elo": 1500})

        h_mv = get_mv(home)
        a_mv = get_mv(away)

        liberta_set  = LIBERTADORES_BY_SEASON.get(season, set())
        home_liberta = 1 if home in liberta_set else 0
        away_liberta = 1 if away in liberta_set else 0

        aprov_equilibrio   = 1.0 / (1 + abs(
            h_table["aproveitamento"] - a_table["aproveitamento"]
        ))
        h2h_draw_dominance = h2h_draws / max(h2h_total, 1)

        result = "H" if hg > ag else ("A" if ag > hg else "D")

        rows.append({
            # Identificadores
            "match_id":   row["match_id"],
            "date":       row["date"],
            "matchday":   matchday,
            "season":     season,
            "home_team":  home,
            "away_team":  away,
            "home_goals": hg,
            "away_goals": ag,
            "result":     result,
            # Elo
            "home_elo":   elos["home_elo"],
            "away_elo":   elos["away_elo"],
            "elo_diff":   elos["home_elo"] - elos["away_elo"],
            # Posição na tabela
            "home_position":       h_table["position"],
            "away_position":       a_table["position"],
            "home_aproveitamento": h_table["aproveitamento"],
            "away_aproveitamento": a_table["aproveitamento"],
            "home_table_gd":       h_table["gd"],
            "away_table_gd":       a_table["gd"],
            "position_diff":       a_table["position"] - h_table["position"],
            # Form curto (5 jogos)
            "home_form_pts":  hs["pts"],  "home_avg_gf":    hs["gf"],
            "home_avg_ga":    hs["ga"],   "home_goal_diff": hs["gd"],
            "home_win_rate":  hs["wr"],   "home_draw_rate": hs["dr"],
            "home_home_form": hs["home_f"],
            "away_form_pts":  as_["pts"], "away_avg_gf":    as_["gf"],
            "away_avg_ga":    as_["ga"],  "away_goal_diff": as_["gd"],
            "away_win_rate":  as_["wr"],  "away_draw_rate": as_["dr"],
            "away_away_form": as_["away_f"],
            # Form longo (10 jogos)
            "home_form_pts_10": hs10["pts"], "home_avg_gf_10": hs10["gf"],
            "home_avg_ga_10":   hs10["ga"],  "home_win_rate_10": hs10["wr"],
            "away_form_pts_10": as10["pts"], "away_avg_gf_10": as10["gf"],
            "away_avg_ga_10":   as10["ga"],  "away_win_rate_10": as10["wr"],
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
            # Libertadores
            "home_joga_libertadores": home_liberta,
            "away_joga_libertadores": away_liberta,
            # Tendência de posição
            "home_pos_trend": h_trend,
            "away_pos_trend": a_trend,
            "pos_trend_diff": h_trend - a_trend,
            # Equilíbrio (detecção de empates)
            "aprov_equilibrio":   aprov_equilibrio,
            "h2h_draw_dominance": h2h_draw_dominance,
            # Dias de descanso
            "home_days_rest": h_rest,
            "away_days_rest": a_rest,
            "rest_diff":      h_rest - a_rest,
            # Pressão situacional (Brasileirão: 20 times, 4 rebaixados, G6=Libertadores)
            "home_relegation_gap": h_press["relegation_gap"],
            "away_relegation_gap": a_press["relegation_gap"],
            "home_g6_gap":         h_press["g6_gap"],
            "away_g6_gap":         a_press["g6_gap"],
            "home_g4_gap":         h_press["g4_gap"],
            "away_g4_gap":         a_press["g4_gap"],
            # xG rolling (0.0 se dados não disponíveis)
            "home_avg_xg":  h_xg["xg"],
            "away_avg_xg":  a_xg["xg"],
            "home_avg_xga": h_xg["xga"],
            "away_avg_xga": a_xg["xga"],
            "xg_diff":      h_xg["xg"]  - a_xg["xg"],
            "xga_diff":     h_xg["xga"] - a_xg["xga"],
        })

        # ── Atualizar estado DEPOIS de salvar as features ─────────────────────
        state.update(home, away, hg, ag, season, matchday,
                     date=date_ts, home_xg=home_xg, away_xg=away_xg)

    print(f"   {total}/{total} (100%) ✓              ")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    print("📊 Lendo dados...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df)} partidas carregadas")

    print("⚙️  Gerando features...")
    t0       = time.time()
    features = build_features(df)
    elapsed  = time.time() - t0

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features salvas! Shape: {features.shape} | Tempo: {elapsed:.1f}s")
    print(features[["home_team", "away_team", "result",
                     "home_position", "away_position",
                     "home_aproveitamento", "elo_diff",
                     "home_joga_libertadores", "home_pos_trend",
                     "aprov_equilibrio", "h2h_draw_dominance"]].head(10))