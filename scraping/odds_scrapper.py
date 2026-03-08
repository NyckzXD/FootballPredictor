import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Registre em https://the-odds-api.com (gratuito)
API_KEY     = os.getenv("ODDS_API_KEY")
BASE_URL    = "https://api.the-odds-api.com/v4"
OUTPUT_PATH = r"C:\PREDICTOR\REPO\scraping\data\external\odds.csv"

# ID do Brasileirão na Odds API
SPORT = "soccer_brazil_campeonato"

# Casas de apostas disponíveis (gratuito inclui algumas)
BOOKMAKERS = ["betano", "bet365", "pinnacle", "betfair_ex_eu"]


def get_odds() -> pd.DataFrame:
    """Coleta odds para próximos jogos do Brasileirão."""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey":     API_KEY,
        "regions":    "eu",
        "markets":    "h2h",       # 1X2 (casa/empate/fora)
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    r = requests.get(url, params=params)
    print(f"Status: {r.status_code}")
    print(f"Requisições restantes: {r.headers.get('x-requests-remaining', 'N/A')}")

    if r.status_code != 200:
        print(f"Erro: {r.json()}")
        return pd.DataFrame()

    games = r.json()
    if not games:
        print("Nenhum jogo disponível no momento.")
        return pd.DataFrame()

    rows = []
    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        date = game["commence_time"]

        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                odds = {o["name"]: o["price"] for o in market["outcomes"]}

                rows.append({
                    "date":          date,
                    "home_team":     home,
                    "away_team":     away,
                    "bookmaker":     bk_name,
                    "odd_home":      odds.get(home, None),
                    "odd_draw":      odds.get("Draw", None),
                    "odd_away":      odds.get(away, None),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Calcular probabilidade implícita das odds
    # (1/odd normalizado para remover a margem da casa)
    df["imp_home"] = 1 / df["odd_home"]
    df["imp_draw"] = 1 / df["odd_draw"]
    df["imp_away"] = 1 / df["odd_away"]
    df["total"]    = df["imp_home"] + df["imp_draw"] + df["imp_away"]

    # Normalizar (remover overround)
    df["prob_home"] = (df["imp_home"] / df["total"]).round(4)
    df["prob_draw"] = (df["imp_draw"] / df["total"]).round(4)
    df["prob_away"] = (df["imp_away"] / df["total"]).round(4)
    df["margem_%"]  = ((df["total"] - 1) * 100).round(2)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ {len(df)} linhas de odds salvas!")
    print(df[["home_team","away_team","bookmaker",
               "odd_home","odd_draw","odd_away","margem_%"]].to_string(index=False))

    return df


if __name__ == "__main__":
    get_odds()