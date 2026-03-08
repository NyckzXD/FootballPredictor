import pandas as pd
from pathlib import Path

INPUT_PATH  = r"C:\PREDICTOR\REPO\scraping\data\raw\mundo_transfermarkt_competicoes_brasileirao_serie_a.csv"
OUTPUT_PATH = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_transfermarkt.csv"
MERGED_PATH = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_merged.csv"
FINAL_PATH  = r"C:\PREDICTOR\REPO\scraping\data\raw\matches_final.csv"

# Normalizar nomes para padrão do projeto
TEAM_MAP = {
    "Atlético-MG":       "CA Mineiro",
    "Atletico-MG":       "CA Mineiro",
    "Atlético-PR":       "CA Paranaense",
    "Athletico-PR":      "CA Paranaense",
    "Corinthians":       "SC Corinthians Paulista",
    "Flamengo":          "CR Flamengo",
    "Vasco da Gama":     "CR Vasco da Gama",
    "Vasco":             "CR Vasco da Gama",
    "Palmeiras":         "SE Palmeiras",
    "São Paulo":         "São Paulo FC",
    "Sao Paulo":         "São Paulo FC",
    "Fluminense":        "Fluminense FC",
    "Botafogo":          "Botafogo FR",
    "Internacional":     "SC Internacional",
    "Grêmio":            "Grêmio FBPA",
    "Gremio":            "Grêmio FBPA",
    "Santos":            "Santos FC",
    "Red Bull Bragantino":"RB Bragantino",
    "Bragantino":        "RB Bragantino",
    "RB Bragantino":     "RB Bragantino",
    "EC Bahia":          "EC Bahia",
    "Bahia":             "EC Bahia",
    "Cruzeiro":          "Cruzeiro EC",
    "América-MG":        "América FC",
    "America-MG":        "América FC",
    "Ceará SC":          "Ceará SC",
    "Ceara":             "Ceará SC",
    "Fortaleza":         "Fortaleza EC",
    "Goiás":             "Goiás EC",
    "Atlético-GO":       "Atlético Goianiense",
    "Coritiba FC":       "Coritiba FBC",
    "Chapecoense":       "Chapecoense AF",
    "EC Vitória":        "EC Vitória",
    "Vitória":           "EC Vitória",
    "Mirassol":          "Mirassol FC",
    "Remo":              "Clube do Remo",
    "Cuiabá-MT":         "Cuiabá EC",
    "EC Juventude":      "EC Juventude",
    "Avaí FC":           "Avaí FC",
    "Criciúma EC":       "Criciúma EC",
    "Sport Recife":      "Sport Club do Recife",
    "CSA":               "CSA",
    "Paraná":            "Paraná Clube",
    "Figueirense":       "Figueirense FC",
    "Náutico":           "Náutico",
    "Portuguesa":        "Portuguesa",
    "Ponte Preta":       "Ponte Preta",
    "Guarani":           "Guarani FC",
    "Barueri":           "Barueri",
    "Brasiliense-DF":    "Brasiliense FC",
    "América-RN":        "América-RN",
    "Ipatinga":          "Ipatinga FC",
    "Santo André":       "Santo André",
}

def convert():
    print("📂 Carregando dataset Transfermarkt...")
    df = pd.read_csv(INPUT_PATH)
    print(f"   {len(df)} jogos | {df['ano_campeonato'].nunique()} temporadas")

    # Remover jogo sem gols
    df = df.dropna(subset=["gols_mandante", "gols_visitante"])

    # Normalizar nomes
    df["time_mandante"]  = df["time_mandante"].map(lambda x: TEAM_MAP.get(x, x))
    df["time_visitante"] = df["time_visitante"].map(lambda x: TEAM_MAP.get(x, x))

    # Converter para formato padrão
    df_conv = pd.DataFrame({
        "match_id":   range(50000, 50000 + len(df)),
        "date":       pd.to_datetime(df["data"], errors="coerce"),
        "season":     df["ano_campeonato"],
        "matchday":   pd.to_numeric(df["rodada"], errors="coerce").fillna(0).astype(int),
        "status":     "FINISHED",
        "home_team":  df["time_mandante"].values,
        "away_team":  df["time_visitante"].values,
        "home_goals": df["gols_mandante"].astype(int),
        "away_goals": df["gols_visitante"].astype(int),
        # Features extras do Transfermarkt
        "home_shots":      df["chutes_mandante"].values,
        "away_shots":      df["chutes_visitante"].values,
        "home_corners":    df["escanteios_mandante"].values,
        "away_corners":    df["escanteios_visitante"].values,
        "home_squad_value":df["valor_equipe_titular_mandante"].values,
        "away_squad_value":df["valor_equipe_titular_visitante"].values,
        "home_avg_age":    df["idade_media_titular_mandante"].values,
        "away_avg_age":    df["idade_media_titular_visitante"].values,
        "attendance":      df["publico"].values,
    })

    df_conv = df_conv.dropna(subset=["date"])
    df_conv = df_conv.sort_values("date").reset_index(drop=True)

    df_conv.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Transfermarkt convertido: {len(df_conv)} jogos")
    print(df_conv.groupby("season").size().to_string())

    # ── Mesclar com matches_merged.csv (2022-2026) ──
    print("\n📂 Mesclando com dados atuais...")
    df_current = pd.read_csv(MERGED_PATH)
    df_current["date"] = pd.to_datetime(df_current["date"], format="mixed", utc=True)
    df_current["date"] = df_current["date"].dt.tz_localize(None)

    # Usar Transfermarkt para 2006-2021, atual para 2022-2026
    df_hist    = df_conv[df_conv["season"] <= 2021].copy()
    df_recent  = df_current.copy()

    # Colunas comuns
    common_cols = ["match_id","date","season","matchday",
                   "status","home_team","away_team","home_goals","away_goals"]

    df_final = pd.concat([
        df_hist[common_cols],
        df_recent[common_cols]
    ], ignore_index=True)

    df_final = df_final.sort_values("date").reset_index(drop=True)
    df_final = df_final.drop_duplicates(
        subset=["home_team","away_team","season","matchday"]
    )

    df_final.to_csv(FINAL_PATH, index=False)

    print(f"\n🏆 Dataset final: {len(df_final)} jogos")
    print(f"📅 Período: {df_final['date'].min().date()} → {df_final['date'].max().date()}")
    print(f"\n📊 Jogos por temporada:")
    print(df_final.groupby("season").size().to_string())
    print(f"\n🏟️  Times únicos: {df_final['home_team'].nunique()}")


if __name__ == "__main__":
    convert()