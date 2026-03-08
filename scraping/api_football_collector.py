import requests
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {
    "x-apisports-key":  API_KEY,
}

# Brasileirão Série A na API-Football
LEAGUE_ID = 71
# Temporadas disponíveis
SEASONS   = [2010, 2011, 2012]

OUTPUT_PATH = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_extended.csv"


def get_fixtures(season: int) -> list:
    """Coleta todos os jogos de uma temporada."""
    url    = f"{BASE_URL}/fixtures"
    params = {
        "league": LEAGUE_ID,
        "season": season,
    }
    r = requests.get(url, headers=HEADERS, params=params)

    remaining = r.headers.get("x-ratelimit-requests-remaining", "?")
    print(f"   Requisições restantes hoje: {remaining}")

    if r.status_code != 200:
        print(f"   ❌ Erro {r.status_code}: {r.json()}")
        return []

    data = r.json()
    if data.get("results", 0) == 0:
        print(f"   ⚠️  Nenhum jogo encontrado para {season}")
        return []

    return data["response"]


def parse_fixture(fix: dict) -> dict:
    """Extrai campos relevantes de um jogo."""
    f      = fix["fixture"]
    league = fix["league"]
    teams  = fix["teams"]
    goals  = fix["goals"]
    score  = fix["score"]

    # Status: FT = finalizado, NS = não iniciado, etc
    status = f["status"]["short"]

    return {
        "match_id":   f["id"],
        "date":       f["date"],
        "season":     league["season"],
        "matchday":   league["round"],
        "status":     "FINISHED" if status == "FT" else status,
        "home_team":  teams["home"]["name"],
        "away_team":  teams["away"]["name"],
        "home_goals": goals["home"] if goals["home"] is not None else None,
        "away_goals": goals["away"] if goals["away"] is not None else None,
        # Placar no intervalo
        "home_ht":    score["halftime"]["home"],
        "away_ht":    score["halftime"]["away"],
    }


def collect_all():
    all_matches = []
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"\n📅 Coletando temporada {season}...")
        fixtures = get_fixtures(season)

        if not fixtures:
            continue

        parsed = [parse_fixture(f) for f in fixtures]
        finished = [p for p in parsed if p["status"] == "FINISHED"]

        print(f"   ✅ {len(finished)} jogos finalizados de {len(parsed)} total")
        all_matches.extend(finished)

        # Respeitar limite de 100 req/dia — pausa entre temporadas
        time.sleep(1.5)

    if not all_matches:
        print("❌ Nenhum jogo coletado!")
        return

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"✅ Total: {len(df)} jogos salvos em {OUTPUT_PATH}")
    print(f"📅 Período: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n📊 Jogos por temporada:")
    print(df.groupby("season").size().to_string())
    print(f"\n🏟️  Times únicos: {df['home_team'].nunique()}")


if __name__ == "__main__":
    # Testar com uma temporada antes de coletar tudo
    print("🧪 Testando API com temporada 2024...")
    fixtures_test = get_fixtures(2024)
    if fixtures_test:
        sample = parse_fixture(fixtures_test[0])
        print(f"   Exemplo: {sample}")
        print(f"\n✅ API funcionando! {len(fixtures_test)} jogos encontrados.")
        print("\nRodando coleta completa...")
        collect_all()
    else:
        print("❌ Verifique sua API key!")