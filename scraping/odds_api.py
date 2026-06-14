import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY   = os.getenv("ODDS_API_KEY")
BASE_URL  = "https://api.the-odds-api.com/v4"
SPORT     = "soccer_brazil_campeonato"
REGIONS   = "eu"
MARKETS   = "h2h,totals"   # h2h = 1X2, totals = over/under
BOOKMAKER = "pinnacle"
OUTPUT    = r"C:\PREDICTOR\REPO\scraping\data\external\odds_live.csv"

# Nomes The Odds API → nomes internos do PREDICTOR
TEAM_MAP = {
    "Atletico Mineiro":        "CA Mineiro",
    "Atletico Goianiense":     "Atlético Goianiense",
    "Athletico Paranaense":    "CA Paranaense",
    "Flamengo":                "CR Flamengo",
    "Vasco da Gama":           "CR Vasco da Gama",
    "Fluminense":              "Fluminense FC",
    "Botafogo":                "Botafogo FR",
    "Palmeiras":               "SE Palmeiras",
    "Corinthians":             "SC Corinthians Paulista",
    "Santos":                  "Santos FC",
    "Sao Paulo":               "São Paulo FC",
    "Gremio":                  "Grêmio FBPA",
    "Internacional":           "SC Internacional",
    "Cruzeiro":                "Cruzeiro EC",
    "Bahia":                   "EC Bahia",
    "Fortaleza":               "Fortaleza EC",
    "Bragantino":              "RB Bragantino",
    "Red Bull Bragantino":     "RB Bragantino",
    "Ceara":                   "Ceará SC",
    "America Mineiro":         "América FC",
    "Goias":                   "Goiás EC",
    "Coritiba":                "Coritiba FBC",
    "Cuiaba":                  "Cuiabá EC",
    "Juventude":               "EC Juventude",
    "Vitoria":                 "EC Vitória",
    "Chapecoense":             "Chapecoense AF",
    "Mirassol":                "Mirassol FC",
    "Clube do Remo":           "Clube do Remo",
    "Sport Recife":            "Sport Club do Recife",
}


def normalize(name: str) -> str:
    return TEAM_MAP.get(name, name)


def fetch_odds() -> pd.DataFrame:
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey":   API_KEY,
        "regions":  REGIONS,
        "markets":  MARKETS,
        "oddsFormat": "decimal",
    }

    print(f"🌐 Buscando odds da Bet365 para {SPORT}...")
    resp = requests.get(url, params=params, timeout=15)

    # Cotas restantes
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"   API requests usados: {used} | restantes: {remaining}")

    if resp.status_code != 200:
        print(f"   ❌ Erro {resp.status_code}: {resp.text}")
        return pd.DataFrame()

    data = resp.json()
    if not data:
        print(f"   {len(data)} jogos encontrados na API")
        return pd.DataFrame()

    rows = []
    for game in data:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home     = normalize(home_raw)
        away     = normalize(away_raw)
        date     = game["commence_time"][:10]
        time_utc = game["commence_time"][11:16]

        odd_h = odd_d = odd_a = None
        odd_over25 = odd_under25 = None

        for bm in game.get("bookmakers", []):
            if bm["key"] == BOOKMAKER:
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for outcome in mkt["outcomes"]:
                            if outcome["name"] == game["home_team"]:
                                odd_h = outcome["price"]
                            elif outcome["name"] == game["away_team"]:
                                odd_a = outcome["price"]
                            elif outcome["name"] == "Draw":
                                odd_d = outcome["price"]
                    elif mkt["key"] == "totals":
                        for outcome in mkt["outcomes"]:
                            if outcome.get("point") == 2.5:
                                if outcome["name"] == "Over":
                                    odd_over25 = outcome["price"]
                                elif outcome["name"] == "Under":
                                    odd_under25 = outcome["price"]

        if odd_h and odd_d and odd_a:
            # Probabilidades implícitas normalizadas (1X2)
            p_h = 1 / odd_h; p_d = 1 / odd_d; p_a = 1 / odd_a
            tot = p_h + p_d + p_a

            # Over/Under 2.5
            prob_over25 = prob_under25 = None
            if odd_over25 and odd_under25:
                po = 1 / odd_over25; pu = 1 / odd_under25
                tot_ou = po + pu
                prob_over25  = round(po / tot_ou, 4)
                prob_under25 = round(pu / tot_ou, 4)

            rows.append({
                "date":          date,
                "time_utc":      time_utc,
                "home_team":     home,
                "away_team":     away,
                "home_raw":      home_raw,
                "away_raw":      away_raw,
                "odd_h":         round(odd_h, 2),
                "odd_d":         round(odd_d, 2),
                "odd_a":         round(odd_a, 2),
                "prob_h_mkt":    round(p_h / tot, 4),
                "prob_d_mkt":    round(p_d / tot, 4),
                "prob_a_mkt":    round(p_a / tot, 4),
                "margin":        round((tot - 1) * 100, 2),
                "odd_over25":    round(odd_over25,  2) if odd_over25  else None,
                "odd_under25":   round(odd_under25, 2) if odd_under25 else None,
                "prob_over25":   prob_over25,
                "prob_under25":  prob_under25,
            })

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUTPUT, index=False)
    print(f"   ✅ {len(df)} jogos com odds | salvo em {OUTPUT}")
    return df

if __name__ == "__main__":
    df = fetch_odds()
    if not df.empty:
        print(f"\n📋 Jogos disponíveis:\n")
        print(df[["date","time_utc","home_team","away_team",
                   "odd_h","odd_d","odd_a","margin"]].to_string(index=False))