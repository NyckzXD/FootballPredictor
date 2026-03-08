import pandas as pd
from pathlib import Path

MATCHES_OLD  = r"C:\PREDICTOR\REPO\scraping\data\raw\matches.csv"
MATCHES_NEW  = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_extended.csv"
OUTPUT_PATH  = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_merged.csv"

# Mapeamento de nomes API-Football → padrão do projeto
TEAM_MAP = {
    "Atletico MG":         "CA Mineiro",
    "Atletico-MG":         "CA Mineiro",
    "Atletico Mineiro":    "CA Mineiro",
    "Corinthians":         "SC Corinthians Paulista",
    "Flamengo":            "CR Flamengo",
    "Vasco DA Gama":       "CR Vasco da Gama",
    "Vasco":               "CR Vasco da Gama",
    "Palmeiras":           "SE Palmeiras",
    "Sao Paulo":           "São Paulo FC",
    "São Paulo":           "São Paulo FC",
    "Fluminense":          "Fluminense FC",
    "Botafogo":            "Botafogo FR",
    "Internacional":       "SC Internacional",
    "Gremio":              "Grêmio FBPA",
    "Grêmio":              "Grêmio FBPA",
    "Santos":              "Santos FC",
    "Bragantino":          "RB Bragantino",
    "Red Bull Bragantino": "RB Bragantino",
    "Bahia":               "EC Bahia",
    "Cruzeiro":            "Cruzeiro EC",
    "Athletico Paranaense":"CA Paranaense",
    "Atletico-PR":         "CA Paranaense",
    "Atletico PR":         "CA Paranaense",
    "America MG":          "América FC",
    "America-MG":          "América FC",
    "Ceara":               "Ceará SC",
    "Fortaleza":           "Fortaleza EC",
    "Goias":               "Goiás EC",
    "Sport Recife":        "Sport Club do Recife",
    "Cuiaba":              "Cuiabá EC",
    "Avai":                "Avaí FC",
    "Coritiba":            "Coritiba FBC",
    "Juventude":           "EC Juventude",
    "Chapecoense":         "Chapecoense AF",
    "Mirassol":            "Mirassol FC",
    "Vitoria":             "EC Vitória",
    "Vitória":             "EC Vitória",
    "Criciuma":            "Criciúma EC",
    "Atletico GO":         "Atlético Goianiense",
    "Atletico-GO":         "Atlético Goianiense",
}

def normalize_team(name: str) -> str:
    return TEAM_MAP.get(name, name)

def merge():
    print("📂 Carregando datasets...")
    df_old = pd.read_csv(MATCHES_OLD)
    df_new = pd.read_csv(MATCHES_NEW)

    print(f"   Original (football-data.org): {len(df_old)} jogos")
    print(f"   Novo (api-football):          {len(df_new)} jogos")
    print(f"   Temporadas original: {sorted(df_old['season'].unique())}")
    print(f"   Temporadas novo:     {sorted(df_new['season'].unique())}")

    # Normalizar nomes no df_new
    df_new["home_team"] = df_new["home_team"].map(normalize_team)
    df_new["away_team"] = df_new["away_team"].map(normalize_team)

    # Padronizar colunas
    cols_needed = ["match_id", "date", "season", "matchday",
                   "status", "home_team", "away_team",
                   "home_goals", "away_goals"]

    # Garantir que matchday é string simples
    df_new["matchday"] = df_new["matchday"].astype(str).str.extract(r"(\d+)").astype(int)
    df_old["matchday"] = pd.to_numeric(df_old["matchday"], errors="coerce").fillna(0).astype(int)

    # Filtrar só jogos finalizados
    df_old_fin = df_old[df_old["status"] == "FINISHED"][cols_needed].copy()
    df_new_fin = df_new[df_new["status"] == "FINISHED"][cols_needed].copy()

    # Remover temporadas duplicadas — priorizar football-data.org para 2023+
    df_new_only = df_new_fin[df_new_fin["season"] == 2022].copy()
    print(f"\n   Usando api-football só para 2022: {len(df_new_only)} jogos")
    print(f"   Usando football-data.org para 2023+: {len(df_old_fin)} jogos")

    # Mesclar
    df_merged = pd.concat([df_new_only, df_old_fin], ignore_index=True)
    df_merged["date"] = pd.to_datetime(df_merged["date"], format="mixed", utc=True)
    df_merged["date"] = df_merged["date"].dt.tz_localize(None)  # remover timezone
    df_merged = df_merged.sort_values("date").reset_index(drop=True)

    # Remover duplicatas por match_id
    df_merged = df_merged.drop_duplicates(subset=["home_team","away_team","season","matchday"])

    print(f"\n✅ Dataset mesclado: {len(df_merged)} jogos")
    print(f"📅 Período: {df_merged['date'].min().date()} → {df_merged['date'].max().date()}")
    print(f"\n📊 Jogos por temporada:")
    print(df_merged.groupby("season").size().to_string())
    print(f"\n🏟️  Times únicos: {df_merged['home_team'].nunique()}")

    times_2026 = set(
        df_old[df_old["season"] == 2026]["home_team"].unique()
    )
    print(f"\n🏟️  Times do Brasileirão 2026: {sorted(times_2026)}")

    # Filtrar apenas jogos onde AMBOS os times estão no Brasileirão 2026
    df_merged = df_merged[
        (df_merged["home_team"].isin(times_2026)) &
        (df_merged["away_team"].isin(times_2026))
    ].copy()

    print(f"✅ Após filtro de times 2026: {len(df_merged)} jogos")
    print(df_merged.groupby("season").size().to_string())
    
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    merge()