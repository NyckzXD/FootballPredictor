import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

OUTPUT_PATH = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_fbref.csv"
LEAGUE_ID   = 24
SEASONS     = list(range(2012, 2026))


def create_driver():
    options = Options()
    options.add_argument("--headless")           # sem abrir janela
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def get_season_matches(driver, season: int) -> list:
    url = (
        f"https://fbref.com/en/comps/{LEAGUE_ID}/{season}"
        f"/schedule/{season}-Serie-A-Scores-and-Fixtures"
    )
    print(f"   Acessando: {url}")

    try:
        driver.get(url)
        # Esperar tabela carregar
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "tbody"))
        )
        time.sleep(2)
    except Exception as e:
        print(f"   ❌ Timeout: {e}")
        return []

    soup   = BeautifulSoup(driver.page_source, "lxml")

    # Tentar achar tabela de schedule
    table = soup.find("table", {"id": f"sched_{season}_24_1"})
    if not table:
        table = soup.find("table", {"id": lambda x: x and "sched" in str(x)})
    if not table:
        print(f"   ⚠️  Tabela não encontrada para {season}")
        tables = [t.get("id") for t in soup.find_all("table")]
        print(f"   Tabelas disponíveis: {tables[:5]}")
        return []

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue

        data = {
            td.get("data-stat"): td.get_text(strip=True)
            for td in tr.find_all(["td", "th"])
        }

        score = data.get("score", "")
        if not score or "–" not in score:
            continue

        try:
            hg, ag   = score.split("–")
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
    MAP = {
        "Atlético Mineiro":     "CA Mineiro",
        "Atletico Mineiro":     "CA Mineiro",
        "Corinthians":          "SC Corinthians Paulista",
        "Flamengo":             "CR Flamengo",
        "Vasco da Gama":        "CR Vasco da Gama",
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
        "Bahia":                "EC Bahia",
        "Cruzeiro":             "Cruzeiro EC",
        "Athletico Paranaense": "CA Paranaense",
        "América Mineiro":      "América FC",
        "Ceará":                "Ceará SC",
        "Fortaleza":            "Fortaleza EC",
        "Goiás":                "Goiás EC",
        "Coritiba":             "Coritiba FBC",
        "Chapecoense":          "Chapecoense AF",
        "Vitória":              "EC Vitória",
        "Mirassol":             "Mirassol FC",
        "Remo":                 "Clube do Remo",
        "Cuiabá":               "Cuiabá EC",
        "Juventude":            "EC Juventude",
        "Sport Recife":         "Sport Club do Recife",
        "Avaí":                 "Avaí FC",
        "Criciúma":             "Criciúma EC",
        "Atlético Goianiense":  "Atlético Goianiense",
    }
    return MAP.get(name, name)


def collect_all():
    all_matches = []
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    print("🌐 Iniciando Chrome...")
    driver = create_driver()

    try:
        for season in SEASONS:
            print(f"\n📅 Coletando {season}...")
            matches = get_season_matches(driver, season)

            if matches:
                for m in matches:
                    m["home_team"] = normalize_team(m["home_team"])
                    m["away_team"] = normalize_team(m["away_team"])
                print(f"   ✅ {len(matches)} jogos coletados")
                all_matches.extend(matches)
            else:
                print(f"   ⚠️  Sem dados para {season}")

            # Pausa entre temporadas para não ser bloqueado
            time.sleep(5)

    finally:
        driver.quit()

    if not all_matches:
        print("❌ Nenhum jogo coletado!")
        return

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = range(90000, 90000 + len(df))
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
    print("🧪 Testando com 2023...")
    driver = create_driver()
    test   = get_season_matches(driver, 2023)
    driver.quit()

    if test:
        print(f"✅ {len(test)} jogos encontrados!")
        print(f"   Exemplo: {test[0]}")
        print("\nIniciando coleta completa...")
        collect_all()
    else:
        print("❌ Problema no scraping")