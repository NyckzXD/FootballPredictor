import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from pathlib import Path

# URLs do Brasileirão Série A no FBref
# Formato: https://fbref.com/en/comps/24/YYYY/schedule/
LEAGUE_ID = 24  # Brasileirão Série A
SEASONS   = list(range(2012, 2025))  # FBref tem desde ~2012

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def get_season_matches(season: int) -> list:
    url = f"https://fbref.com/en/comps/{LEAGUE_ID}/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"
    
    print(f"   Acessando: {url}")
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"   ❌ Status {r.status_code}")
            return []
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return []

    soup = BeautifulSoup(r.content, "lxml")
    
    # ID da tabela muda por temporada: sched_YYYY_24_1
    table = soup.find("table", {"id": f"sched_{season}_24_1"})
    if not table:
        # Fallback — qualquer tabela stats
        table = soup.find("table", {"id": lambda x: x and "sched" in x})
    if not table:
        print(f"   ⚠️  Tabela não encontrada. IDs disponíveis:")
        for t in soup.find_all("table"):
            print(f"      {t.get('id')}")
        return []

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue

        data = {td.get("data-stat"): td.get_text(strip=True)
                for td in tr.find_all(["td", "th"])}

        score = data.get("score", "")
        if not score or "–" not in score:
            continue

        try:
            hg, ag = score.split("–")
            home_goals = int(hg.strip())
            away_goals = int(ag.strip())
        except ValueError:
            continue

        date_raw = data.get("date", "")
        home     = data.get("home_team", "")
        away     = data.get("away_team", "")

        if not date_raw or not home or not away:
            continue

        rows.append({
            "date":       date_raw,
            "season":     season,
            "matchday":   data.get("gameweek", ""),
            "status":     "FINISHED",
            "home_team":  home,
            "away_team":  away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg":    float(data["home_xg"]) if data.get("home_xg") else None,
            "away_xg":    float(data["away_xg"]) if data.get("away_xg") else None,
        })

    return rows


def normalize_team(name: str) -> str:
    """Normaliza nomes do FBref para padrão do projeto."""
    MAP = {
        "Atlético Mineiro":     "CA Mineiro",
        "Atletico Mineiro":     "CA Mineiro",
        "Corinthians":          "SC Corinthians Paulista",
        "Flamengo":             "CR Flamengo",
        "Vasco da Gama":        "CR Vasco da Gama",
        "Vasco":                "CR Vasco da Gama",
        "Palmeiras":            "SE Palmeiras",
        "São Paulo":            "São Paulo FC",
        "Sao Paulo":            "São Paulo FC",
        "Fluminense":           "Fluminense FC",
        "Botafogo":             "Botafogo FR",
        "Internacional":        "SC Internacional",
        "Grêmio":               "Grêmio FBPA",
        "Gremio":               "Grêmio FBPA",
        "Santos":               "Santos FC",
        "Red Bull Bragantino":  "RB Bragantino",
        "Bragantino":           "RB Bragantino",
        "Bahia":                "EC Bahia",
        "Cruzeiro":             "Cruzeiro EC",
        "Athletico Paranaense": "CA Paranaense",
        "Athletico-PR":         "CA Paranaense",
        "América Mineiro":      "América FC",
        "America Mineiro":      "América FC",
        "Ceará":                "Ceará SC",
        "Ceara":                "Ceará SC",
        "Fortaleza":            "Fortaleza EC",
        "Goiás":                "Goiás EC",
        "Goias":                "Goiás EC",
        "Coritiba":             "Coritiba FBC",
        "Chapecoense":          "Chapecoense AF",
        "Vitória":              "EC Vitória",
        "Vitoria":              "EC Vitória",
        "Mirassol":             "Mirassol FC",
        "Remo":                 "Clube do Remo",
        "Cuiabá":               "Cuiabá EC",
        "Cuiaba":               "Cuiabá EC",
        "Juventude":            "EC Juventude",
        "Sport Recife":         "Sport Club do Recife",
        "Avaí":                 "Avaí FC",
        "Avai":                 "Avaí FC",
        "Criciúma":             "Criciúma EC",
        "Criciuma":             "Criciúma EC",
        "Atlético Goianiense":  "Atlético Goianiense",
        "Atletico Goianiense":  "Atlético Goianiense",
    }
    return MAP.get(name, name)


def collect_all():
    all_matches = []
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"\n📅 Coletando {season}...")
        matches = get_season_matches(season)

        if matches:
            # Normalizar nomes
            for m in matches:
                m["home_team"] = normalize_team(m["home_team"])
                m["away_team"] = normalize_team(m["away_team"])

            print(f"   ✅ {len(matches)} jogos coletados")
            all_matches.extend(matches)
        else:
            print(f"   ⚠️  Sem dados para {season}")

        # FBref bloqueia se requisições forem muito rápidas
        time.sleep(4)

    if not all_matches:
        print("❌ Nenhum jogo coletado!")
        return

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    # Gerar match_id único
    df["match_id"] = range(90000, 90000 + len(df))

    # Extrair número da rodada
    df["matchday"] = pd.to_numeric(
        df["matchday"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    ).fillna(0).astype(int)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"✅ Total: {len(df)} jogos salvos!")
    print(f"📅 Período: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n📊 Jogos por temporada:")
    print(df.groupby("season").size().to_string())
    has_xg = df["home_xg"].notna().sum()
    print(f"\n⚡ Jogos com xG: {has_xg} ({has_xg/len(df):.1%})")


if __name__ == "__main__":
    # Testar com uma temporada primeiro
    print("🧪 Testando com 2023...")
    test = get_season_matches(2023)
    if test:
        print(f"✅ {len(test)} jogos encontrados!")
        print(f"   Exemplo: {test[0]}")
        print("\nIniciando coleta completa...")
        collect_all()
    else:
        print("❌ Problema no scraping — verificar estrutura da página")