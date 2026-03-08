import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from pathlib import Path

OUTPUT_PATH = r"C:\PREDICTOR\scraping\data\external\market_values.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Mapeamento times football-data.org → Transfermarkt URL
TEAMS = {
    "SE Palmeiras":           "se-palmeiras/startseite/verein/1023", #OK
    "CR Flamengo":            "cr-flamengo/kader/verein/614/saison_id/2022", #OK
    "São Paulo FC":           "fc-sao-paulo/startseite/verein/585", #OK
    "SC Corinthians Paulista":"sc-corinthians/startseite/verein/199", #OK
    "CA Mineiro":             "clube-atletico-mineiro/startseite/verein/330",#OK
    "Fluminense FC":          "fluminense-rio-de-janeiro/startseite/verein/2462",#OK
    "Botafogo FR":            "botafogo-fr/startseite/verein/537", #OK
    "CA Paranaense":          "athletico-paranaense/startseite/verein/679",#OK
    "Grêmio FBPA":            "gremio-fbpa/startseite/verein/210",#OK
    "SC Internacional":       "sc-internacional/kader/verein/6600/saison_id/2024",#OK
    "CR Vasco da Gama":       "cr-vasco-da-gama/startseite/verein/978", #OK
    "Cruzeiro EC":            "ec-cruzeiro/startseite/verein/609", #OK
    "Santos FC":              "sfc-santos/startseite/verein/221", #OK
    "RB Bragantino":          "red-bull-bragantino/startseite/verein/8793", #OK
    "EC Bahia":               "esporte-clube-bahia/startseite/verein/10010", #OK
    "Mirassol FC":            "mirassol-fc/startseite/verein/3876", #OK
    "Clube do Remo":          "clube-do-remo/startseite/verein/10997", #OK
    "EC Vitória":             "ec-vitoria/startseite/verein/2125",#OK
    "Chapecoense AF":         "chapecoense/startseite/verein/17776",#OK
    "Coritiba FBC":           "coritiba-fc/startseite/verein/776", #OK
}

BASE_URL = "https://www.transfermarkt.com.br"


def parse_value(value_str: str) -> float:
    """Converte '15,73 M €' ou '45,20 mi' para float em milhões."""
    if not value_str or value_str == "-":
        return 0.0
    
    # Limpar texto
    value_str = value_str.replace("\xa0", " ").replace("\n", " ").strip()
    value_str = value_str.lower()
    
    # Determinar multiplicador
    if "bi" in value_str:
        multiplier = 1000.0
    elif "m €" in value_str or " mi" in value_str or "mio" in value_str:
        multiplier = 1.0
    elif "mil" in value_str or "k" in value_str:
        multiplier = 0.001
    else:
        multiplier = 1.0

    # Extrair número
    numbers = re.findall(r"\d+[,.]?\d*", value_str)
    if not numbers:
        return 0.0
    
    num = float(numbers[0].replace(",", "."))
    return round(num * multiplier, 3)


def scrape_team_value(scraper, team_name: str, team_url: str) -> dict:
    url = f"{BASE_URL}/{team_url}"
    try:
        response = scraper.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        total_value = 0.0

        tag = soup.select_one("a.data-header__market-value-wrapper")
        if tag:
            # Remover <p> e <span> filhos para ficar só o número
            for child in tag.select("p, span"):
                unit_text = child.get_text(strip=True)
                child.decompose()

            number_text = tag.get_text(strip=True)  # ex: "92,35"
            full_text = f"{number_text} {unit_text}"  # ex: "92,35 M €"
            total_value = parse_value(full_text)

        # Tamanho do elenco
        rows = soup.select("table.items tbody tr")
        squad_size = len([r for r in rows
                          if "spacer" not in (r.get("class") or [])])

        # Jogador mais valioso
        top_value = max(
            (parse_value(td.get_text(strip=True))
             for td in soup.select("table.items td.rechts.hauptlink")),
            default=0.0
        )

        print(f"   ✅ {team_name}: €{total_value}M | elenco: {squad_size}")
        return {
            "team":             team_name,
            "market_value":     total_value,
            "squad_size":       squad_size,
            "top_player_value": top_value,
        }

    except Exception as e:
        print(f"   ❌ {team_name}: erro — {e}")
        return {
            "team": team_name, "market_value": 0.0,
            "squad_size": 0, "top_player_value": 0.0,
        }


def scrape_all_teams() -> pd.DataFrame:
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    results = []
    total = len(TEAMS)

    for i, (team_name, team_url) in enumerate(TEAMS.items(), 1):
        print(f"[{i}/{total}] Coletando {team_name}...")
        data = scrape_team_value(scraper, team_name, team_url)
        results.append(data)
        time.sleep(3)  # respeitar o site

    df = pd.DataFrame(results)

    # Normalizar valor de mercado (0-1) para usar como feature
    max_val = df["market_value"].max()
    df["market_value_norm"] = (df["market_value"] / max_val).round(4)

    # Log do valor (reduz skewness)
    import numpy as np
    df["market_value_log"] = np.log1p(df["market_value"]).round(4)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Dados salvos em {OUTPUT_PATH}")
    print(df.sort_values("market_value", ascending=False).to_string(index=False))
    return df

if __name__ == "__main__":
    scrape_all_teams()