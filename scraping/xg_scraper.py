"""
xg_scraper.py — Coleta xG do Brasileirão via API não-oficial do Sofascore.

Sofascore cobre o Brasileirão com xG por partida.
Saída: scraping/data/external/xg_data.csv
  date, season, home_team, away_team, home_xg, away_xg

Sem autenticação necessária — API pública do Sofascore.
"""
import requests
import pandas as pd
import time
from pathlib import Path

OUTPUT_PATH = Path(r"C:\PREDICTOR\REPO\scraping\data\external\xg_data.csv")

BASE_URL     = "https://api.sofascore.com/api/v1"
TOURNAMENT   = 325    # Brasileirão Serie A no Sofascore
DELAY        = 1.0    # segundos entre requisições

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.sofascore.com/",
}

# IDs de temporada do Brasileirão no Sofascore
SEASON_IDS = {
    2020: 36886,
    2021: 40557,
    2022: 43767,
    2023: 47948,
    2024: 57303,
    2025: 63903,
    2026: None,   # descoberto dinamicamente
}

TEAM_MAP = {
    "Atletico Mineiro":      "CA Mineiro",
    "Athletico Paranaense":  "CA Paranaense",
    "Atletico Goianiense":   "Atlético Goianiense",
    "Flamengo":              "CR Flamengo",
    "Vasco da Gama":         "CR Vasco da Gama",
    "Fluminense":            "Fluminense FC",
    "Botafogo":              "Botafogo FR",
    "Palmeiras":             "SE Palmeiras",
    "Corinthians":           "SC Corinthians Paulista",
    "Santos":                "Santos FC",
    "Sao Paulo":             "São Paulo FC",
    "São Paulo":             "São Paulo FC",
    "Gremio":                "Grêmio FBPA",
    "Grêmio":                "Grêmio FBPA",
    "Internacional":         "SC Internacional",
    "Cruzeiro":              "Cruzeiro EC",
    "Bahia":                 "EC Bahia",
    "Fortaleza":             "Fortaleza EC",
    "Bragantino":            "RB Bragantino",
    "Red Bull Bragantino":   "RB Bragantino",
    "Ceara":                 "Ceará SC",
    "Ceará":                 "Ceará SC",
    "America Mineiro":       "América FC",
    "América Mineiro":       "América FC",
    "Goias":                 "Goiás EC",
    "Goiás":                 "Goiás EC",
    "Coritiba":              "Coritiba FBC",
    "Cuiaba":                "Cuiabá EC",
    "Cuiabá":                "Cuiabá EC",
    "Juventude":             "EC Juventude",
    "Vitoria":               "EC Vitória",
    "Vitória":               "EC Vitória",
    "Chapecoense":           "Chapecoense AF",
    "Mirassol":              "Mirassol FC",
    "Sport Recife":          "Sport Club do Recife",
    "Avai":                  "Avaí FC",
    "Avaí":                  "Avaí FC",
    "Criciuma":              "Criciúma EC",
    "Criciúma":              "Criciúma EC",
}


def normalize(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


def get(url: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"   HTTP {r.status_code}: {url}")
    except Exception as e:
        print(f"   Erro: {e}")
    return None


def discover_season_id(year: int) -> int | None:
    data = get(f"{BASE_URL}/tournament/{TOURNAMENT}/seasons")
    if not data:
        return None
    for s in data.get("seasons", []):
        if str(year) in s.get("name", "") or s.get("year") == year:
            print(f"   Temporada {year} → ID {s['id']} ({s.get('name')})")
            return s["id"]
    return None


def get_events_page(season_id: int, page: int) -> tuple[list, bool]:
    data = get(f"{BASE_URL}/tournament/{TOURNAMENT}/season/{season_id}/events/last/{page}")
    if not data:
        return [], False
    events   = data.get("events", [])
    has_next = data.get("hasNextPage", False)
    return events, has_next


def parse_event(event: dict, year: int) -> dict | None:
    status = event.get("status", {}).get("type", "")
    if status != "finished":
        return None

    home = event.get("homeTeam", {}).get("name", "")
    away = event.get("awayTeam", {}).get("name", "")
    date = event.get("startTimestamp", 0)

    # xG fica em homeScore/awayScore com chave "expected"
    h_xg = event.get("homeScore", {}).get("expected")
    a_xg = event.get("awayScore", {}).get("expected")

    if h_xg is None or a_xg is None:
        return None

    import datetime
    date_str = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d") if date else ""

    return {
        "date":      date_str,
        "season":    year,
        "home_team": normalize(home),
        "away_team": normalize(away),
        "home_xg":   float(h_xg),
        "away_xg":   float(a_xg),
    }


def collect_season(year: int, season_id: int) -> pd.DataFrame:
    rows, page = [], 0
    while True:
        events, has_next = get_events_page(season_id, page)
        for ev in events:
            row = parse_event(ev, year)
            if row:
                rows.append(row)
        if not has_next:
            break
        page += 1
        time.sleep(DELAY)

    return pd.DataFrame(rows)


def run(years: list = None) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if years is None:
        years = [y for y in SEASON_IDS]

    if OUTPUT_PATH.exists():
        existing     = pd.read_csv(OUTPUT_PATH)
        done_seasons = set(existing["season"].astype(int).tolist())
        print(f"   Já coletados: {sorted(done_seasons)}")
        all_frames   = [existing]
    else:
        existing, done_seasons, all_frames = pd.DataFrame(), set(), []

    for year in years:
        if year in done_seasons:
            print(f"   [{year}] já coletado — pulando")
            continue

        season_id = SEASON_IDS.get(year)
        if season_id is None:
            print(f"   [{year}] Descobrindo ID de temporada...")
            season_id = discover_season_id(year)
            if season_id is None:
                print(f"   ⚠️  [{year}] Temporada não encontrada")
                continue

        print(f"📅 [{year}] Coletando (season_id={season_id})...")
        df = collect_season(year, season_id)

        if df.empty:
            print(f"   ⚠️  [{year}] Nenhum jogo com xG encontrado")
            continue

        n = len(df)
        print(f"   ✅ [{year}] {n} partidas com xG "
              f"(casa: {df['home_xg'].mean():.2f}, fora: {df['away_xg'].mean():.2f})")

        all_frames.append(df)
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Salvo ({len(combined)} total)")
        time.sleep(2)

    if not all_frames:
        print("\n⚠️  Nenhum xG coletado")
        return pd.DataFrame()

    final = pd.concat(all_frames, ignore_index=True)
    final = final.drop_duplicates(subset=["date", "home_team", "away_team"])
    final = final.sort_values(["season", "date"]).reset_index(drop=True)
    final.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ xG salvo: {len(final)} partidas → {OUTPUT_PATH}")
    print(final.groupby("season").agg(
        partidas=("home_xg", "count"),
        xg_casa=("home_xg", "mean"),
        xg_fora=("away_xg", "mean"),
    ).round(2).to_string())
    return final


if __name__ == "__main__":
    print("⚽ Coletando xG — Brasileirão Serie A via Sofascore\n")
    run()
