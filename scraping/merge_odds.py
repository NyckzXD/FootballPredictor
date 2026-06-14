import pandas as pd
import numpy as np

ODDS_PATH     = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
FEATURES_PATH = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
OUTPUT_PATH   = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"

TEAM_MAP = {
    "America MG":         "América FC",
    "Athletico-PR":       "CA Paranaense",
    "Atletico GO":        "Atlético Goianiense",
    "Atletico-MG":        "CA Mineiro",
    "Atletico MG":        "CA Mineiro",
    "Avai":               "Avaí FC",
    "Bahia":              "EC Bahia",
    "Botafogo RJ":        "Botafogo FR",
    "Bragantino":         "RB Bragantino",
    "Red Bull Bragantino":"RB Bragantino",
    "Ceara":              "Ceará SC",
    "Chapecoense-SC":     "Chapecoense AF",
    "Chapecoense":        "Chapecoense AF",
    "Corinthians":        "SC Corinthians Paulista",
    "Coritiba":           "Coritiba FBC",
    "Criciuma":           "Criciúma EC",
    "Cruzeiro":           "Cruzeiro EC",
    "Cuiaba":             "Cuiabá EC",
    "Flamengo RJ":        "CR Flamengo",
    "Fluminense":         "Fluminense FC",
    "Fortaleza":          "Fortaleza EC",
    "Goias":              "Goiás EC",
    "Gremio":             "Grêmio FBPA",
    "Internacional":      "SC Internacional",
    "Juventude":          "EC Juventude",
    "Mirassol":           "Mirassol FC",
    "Palmeiras":          "SE Palmeiras",
    "Parana":             "Paraná Clube",
    "Remo":               "Clube do Remo",
    "Santos":             "Santos FC",
    "Sao Paulo":          "São Paulo FC",
    "Sport Recife":       "Sport Club do Recife",
    "Vasco":              "CR Vasco da Gama",
    "Vasco DA Gama":      "CR Vasco da Gama",
    "Vitoria":            "EC Vitória",
    "CSA":                "CSA",
    "Portuguesa":         "Portuguesa",
    "Figueirense":        "Figueirense FC",
    "Nautico":            "Náutico",
    "Ponte Preta":        "Ponte Preta",
    "Guarani":            "Guarani FC",
    "Santo Andre":        "Santo André",
    "Brasiliense":        "Brasiliense FC",
    "America RN":         "América-RN",
    "Ipatinga":           "Ipatinga FC",
    "Barueri":            "Barueri",
    "Joinville":          "Joinville-SC",
}


def merge():
    # ── Carregar odds ──
    print("📂 Carregando odds...")
    odds = pd.read_csv(ODDS_PATH)
    odds["season"] = odds["Season"].astype(int)
    odds["home_team"] = odds["Home"].map(lambda x: TEAM_MAP.get(x, x))
    odds["away_team"] = odds["Away"].map(lambda x: TEAM_MAP.get(x, x))

    # Melhor odd disponível: Pinnacle > Average
    def best_odds(row):
        if pd.notna(row.get("PSCH")) and row["PSCH"] > 0:
            return row["PSCH"], row["PSCD"], row["PSCA"]
        elif pd.notna(row.get("AvgCH")) and row["AvgCH"] > 0:
            return row["AvgCH"], row["AvgCD"], row["AvgCA"]
        return np.nan, np.nan, np.nan

    odds[["odd_h", "odd_d", "odd_a"]] = odds.apply(
        lambda r: pd.Series(best_odds(r)), axis=1
    )

    # Probabilidades implícitas normalizadas (sem margem da casa)
    valid = odds["odd_h"].notna()
    odds.loc[valid, "p_h"] = 1.0 / odds.loc[valid, "odd_h"]
    odds.loc[valid, "p_d"] = 1.0 / odds.loc[valid, "odd_d"]
    odds.loc[valid, "p_a"] = 1.0 / odds.loc[valid, "odd_a"]
    tot = odds["p_h"] + odds["p_d"] + odds["p_a"]
    odds["prob_h_mkt"] = odds["p_h"] / tot
    odds["prob_d_mkt"] = odds["p_d"] / tot
    odds["prob_a_mkt"] = odds["p_a"] / tot

    # Over/Under 2.5 histórico (BRA.csv usa colunas BbOU, BbAv>2.5, BbAv<2.5 ou similares)
    def best_ou_odds(row):
        """Extrai odds over/under 2.5 das colunas disponíveis no BRA.csv."""
        for over_col, under_col in [
            ("PSC>2.5", "PSC<2.5"),
            ("BbAv>2.5", "BbAv<2.5"),
            ("Avg>2.5", "Avg<2.5"),
            (">2.5", "<2.5"),
        ]:
            if over_col in row.index and pd.notna(row.get(over_col)) and row[over_col] > 1.0:
                return float(row[over_col]), float(row[under_col])
        return np.nan, np.nan

    odds[["odd_over25", "odd_under25"]] = odds.apply(
        lambda r: pd.Series(best_ou_odds(r)), axis=1
    )

    # Probabilidades over/under normalizadas
    ou_valid = odds["odd_over25"].notna()
    odds.loc[ou_valid, "p_over"]  = 1.0 / odds.loc[ou_valid, "odd_over25"]
    odds.loc[ou_valid, "p_under"] = 1.0 / odds.loc[ou_valid, "odd_under25"]
    tot_ou = odds["p_over"] + odds["p_under"]
    odds["prob_over25"]  = odds["p_over"]  / tot_ou
    odds["prob_under25"] = odds["p_under"] / tot_ou

    # Features derivadas das odds
    odds["odds_draw_factor"]     = odds["odd_d"] / ((odds["odd_h"] + odds["odd_a"]) / 2)
    odds["odds_home_away_ratio"] = odds["odd_h"] / odds["odd_a"]
    odds["market_entropy"]       = -(
        odds["prob_h_mkt"] * np.log(odds["prob_h_mkt"] + 1e-9) +
        odds["prob_d_mkt"] * np.log(odds["prob_d_mkt"] + 1e-9) +
        odds["prob_a_mkt"] * np.log(odds["prob_a_mkt"] + 1e-9)
    )
    # Implied total goals a partir das odds over/under
    # Aproximação: E[gols] ≈ ln(odd_under25) / 0.4  (calibrado empiricamente)
    odds["implied_total_goals"] = np.where(
        ou_valid,
        np.log(odds["odd_under25"].clip(lower=1.01)) / 0.4,
        np.nan
    )

    odds_clean = odds[[
        "season", "home_team", "away_team",
        "odd_h", "odd_d", "odd_a",
        "prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
        "odds_draw_factor", "odds_home_away_ratio", "market_entropy",
        "odd_over25", "odd_under25", "prob_over25", "prob_under25",
        "implied_total_goals",
    ]].dropna(subset=["home_team", "away_team"]).copy()

    odds_clean["season"] = odds_clean["season"].astype(int)
    print(f"   {len(odds_clean)} jogos com odds preparados")

    # ── Carregar features ──
    print("\n📂 Carregando features...")
    features = pd.read_csv(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["season"] = features["season"].astype(int)

    before = len(features)
    features = features.dropna(subset=["home_team", "away_team"])
    print(f"   {len(features)} jogos válidos (removidos {before - len(features)} com nulos)")

    # ── Merge por season + home_team + away_team ──
    print("\n🔗 Fazendo merge por season + home_team + away_team...")
    merged = pd.merge(
        features,
        odds_clean,
        on=["season", "home_team", "away_team"],
        how="left"
    )

    # ── Diagnóstico ──
    matched = merged["prob_h_mkt"].notna().sum()
    print(f"\n✅ Jogos com odds válidas: {matched} ({matched/len(merged):.1%})")
    print(f"\n📊 Match por temporada:")
    print(merged.groupby("season")["prob_h_mkt"].apply(
        lambda x: f"{x.notna().sum():>4}/{len(x)}"
    ).to_string())

    # Times sem match em 2018+
    sem_match = merged[
        merged["prob_h_mkt"].isna() &
        (merged["season"] >= 2018)
    ]
    if len(sem_match) > 0:
        print(f"\n⚠️  Amostra de jogos sem match (2018+):")
        print(sem_match[["season", "home_team", "away_team"]].drop_duplicates().head(10).to_string(index=False))

    # ── Salvar ──
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Salvo em {OUTPUT_PATH}")
    print(f"   Shape: {merged.shape}")
    ou_matched = merged["prob_over25"].notna().sum()
    print(f"   Jogos com over/under 2.5: {ou_matched} ({ou_matched/len(merged):.1%})")
    print(f"   Colunas adicionadas: odd_h/d/a, prob_h/d/a_mkt, draw_factor, entropy, over/under 2.5, implied_total_goals")


if __name__ == "__main__":
    merge()
