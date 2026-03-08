import requests
import pandas as pd
import time
from pathlib import Path

API_KEY = "1229cd54e5744eb4bdd23fe36def04c8"  # Registre em football-data.org (gratuito)
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}

# ID da Série A no football-data.org
BRASILEIRAO_ID = 2013

def get_season_matches(season_year: int) -> pd.DataFrame:
    """Coleta todas as partidas de uma temporada."""
    url = f"{BASE_URL}/competitions/{BRASILEIRAO_ID}/matches"
    params = {"season": season_year}
    
    r = requests.get(url, headers=HEADERS, params=params)
    
    # DEBUG - adicione essas 2 linhas:
    print(f"Status: {r.status_code}")
    print(f"Resposta: {r.json()}")
    
    r.raise_for_status()
    
    matches = r.json()["matches"]
    
    rows = []
    for m in matches:
        rows.append({
            "match_id":     m["id"],
            "date":         m["utcDate"],
            "matchday":     m["matchday"],
            "status":       m["status"],
            "home_team":    m["homeTeam"]["name"],
            "away_team":    m["awayTeam"]["name"],
            "home_goals":   m["score"]["fullTime"]["home"],
            "away_goals":   m["score"]["fullTime"]["away"],
            "season":       season_year,
        })
    
    return pd.DataFrame(rows)


def get_standings(season_year: int) -> pd.DataFrame:
    """Coleta a tabela de classificação."""
    url = f"{BASE_URL}/competitions/{BRASILEIRAO_ID}/standings"
    params = {"season": season_year}
    
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    
    table = r.json()["standings"][0]["table"]
    
    rows = []
    for entry in table:
        rows.append({
            "season":       season_year,
            "position":     entry["position"],
            "team":         entry["team"]["name"],
            "played":       entry["playedGames"],
            "won":          entry["won"],
            "draw":         entry["draw"],
            "lost":         entry["lost"],
            "goals_for":    entry["goalsFor"],
            "goals_against":entry["goalsAgainst"],
            "goal_diff":    entry["goalDifference"],
            "points":       entry["points"],
        })
    
    return pd.DataFrame(rows)


def collect_multiple_seasons(seasons: list[int], output_dir: str = "data/raw"):
    """Coleta dados de várias temporadas e salva em CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_matches = []
    all_standings = []
    
    for year in seasons:
        print(f"📥 Coletando temporada {year}...")
        
        matches_df = get_season_matches(year)
        standings_df = get_standings(year)
        
        all_matches.append(matches_df)
        all_standings.append(standings_df)
        
        time.sleep(1)  # Respeitar rate limit
    
    matches = pd.concat(all_matches, ignore_index=True)
    standings = pd.concat(all_standings, ignore_index=True)
    
    matches.to_csv(f"{output_dir}/matches.csv", index=False)
    standings.to_csv(f"{output_dir}/standings.csv", index=False)
    
    print(f"✅ {len(matches)} partidas coletadas!")
    return matches, standings


if __name__ == "__main__":
    # Coleta 2023 2024 e 2025
    matches, standings = collect_multiple_seasons([2023, 2024, 2025, 2026])

def get_season_matches(season_year: int) -> pd.DataFrame:
    url = f"{BASE_URL}/competitions/{BRASILEIRAO_ID}/matches"
    params = {"season": season_year}
    
    r = requests.get(url, headers=HEADERS, params=params)
    
    # Adicione essas 3 linhas para debugar:
    print(f"Status code: {r.status_code}")
    print(f"Resposta: {r.json()}")  # <-- vai mostrar o erro real
    
    r.raise_for_status()
    matches = r.json()["matches"]