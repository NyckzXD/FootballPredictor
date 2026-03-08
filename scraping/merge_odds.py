import pandas as pd
import numpy as np
from pathlib import Path

ODDS_PATH     = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
FEATURES_PATH = r"C:\PREDICTOR\REPO\scraping\data\processed\features.csv"
OUTPUT_PATH   = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"

# Normalizar nomes para padrão do projeto
TEAM_MAP = {
    "Palmeiras":          "SE Palmeiras",
    "Portuguesa":         "Portuguesa",
    "Sport Recife":       "Sport Club do Recife",
    "Flamengo RJ":        "CR Flamengo",
    "Figueirense":        "Figueirense FC",
    "Nautico":            "Náutico",
    "Atletico MG":        "CA Mineiro",
    "Atletico-MG":        "CA Mineiro",
    "Atletico Mineiro":   "CA Mineiro",
    "Atletico-PR":        "CA Paranaense",
    "Athletico-PR":       "CA Paranaense",
    "Corinthians":        "SC Corinthians Paulista",
    "Fluminense":         "Fluminense FC",
    "Botafogo RJ":        "Botafogo FR",
    "Botafogo":           "Botafogo FR",
    "Internacional":      "SC Internacional",
    "Gremio":             "Grêmio FBPA",
    "Santos":             "Santos FC",
    "Bragantino":         "RB Bragantino",
    "Red Bull Bragantino":"RB Bragantino",
    "Bahia":              "EC Bahia",
    "Cruzeiro":           "Cruzeiro EC",
    "America MG":         "América FC",
    "America-MG":         "América FC",
    "Ceara":              "Ceará SC",
    "Fortaleza":          "Fortaleza EC",
    "Goias":              "Goiás EC",
    "Atletico GO":        "Atlético Goianiense",
    "Atletico-GO":        "Atlético Goianiense",
    "Coritiba":           "Coritiba FBC",
    "Chapecoense":        "Chapecoense AF",
    "Vitoria":            "EC Vitória",
    "Vasco DA Gama":      "CR Vasco da Gama",
    "Vasco":              "CR Vasco da Gama",
    "Sao Paulo":          "São Paulo FC",
    "Mirassol":           "Mirassol FC",
    "Remo":               "Clube do Remo",
    "Cuiaba":             "Cuiabá EC",
    "Juventude":          "EC Juventude",
    "Avai":               "Avaí FC",
    "Criciuma":           "Criciúma EC",
    "CSA":                "CSA",
    "Parana":             "Paraná Clube",
    "Barueri":            "Barueri",
    "Ipatinga":           "Ipatinga FC",
    "Santo Andre":        "Santo André",
    "Ponte Preta":        "Ponte Preta",
    "Guarani":            "Guarani FC",
    "Brasiliense":        "Brasiliense FC",
    "America RN":         "América-RN",
}


def odds_to_prob(odd):
    """Converte odd decimal para probabilidade implícita."""
    return 1.0 / odd


def normalize_probs(ph, pd_, pa):
    """Remove margem da casa e normaliza para soma 1."""
    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total


def merge():
    print("📂 Carregando odds...")
    odds = pd.read_csv(ODDS_PATH)
    odds["date"] = pd.to_datetime(odds["Date"], dayfirst=True, errors="coerce")
    odds["home_team"] = odds["Home"].map(lambda x: TEAM_MAP.get(x, x))
    odds["away_team"] = odds["Away"].map(lambda x: TEAM_MAP.get(x, x))
    odds["season"]    = odds["Season"]

    # Usar Pinnacle se disponível, senão Average
    def best_odds(row):
        if pd.notna(row["PSCH"]) and row["PSCH"] > 0:
            return row["PSCH"], row["PSCD"], row["PSCA"]
        elif pd.notna(row["AvgCH"]) and row["AvgCH"] > 0:
            return row["AvgCH"], row["AvgCD"], row["AvgCA"]
        return np.nan, np.nan, np.nan

    odds[["odd_h", "odd_d", "odd_a"]] = odds.apply(
        lambda r: pd.Series(best_odds(r)), axis=1
    )

    # Probabilidades implícitas normalizadas
    valid = odds["odd_h"].notna()
    odds.loc[valid, "prob_h_mkt"] = odds_to_prob(odds.loc[valid, "odd_h"])
    odds.loc[valid, "prob_d_mkt"] = odds_to_prob(odds.loc[valid, "odd_d"])
    odds.loc[valid, "prob_a_mkt"] = odds_to_prob(odds.loc[valid, "odd_a"])

    # Normalizar (tirar margem)
    tot = odds["prob_h_mkt"] + odds["prob_d_mkt"] + odds["prob_a_mkt"]
    odds["prob_h_mkt"] /= tot
    odds["prob_d_mkt"] /= tot
    odds["prob_a_mkt"] /= tot

    # Features derivadas das odds
    odds["odds_draw_factor"]  = odds["odd_d"] / ((odds["odd_h"] + odds["odd_a"]) / 2)
    odds["odds_home_away_ratio"] = odds["odd_h"] / odds["odd_a"]
    odds["market_entropy"]    = -(
        odds["prob_h_mkt"] * np.log(odds["prob_h_mkt"] + 1e-9) +
        odds["prob_d_mkt"] * np.log(odds["prob_d_mkt"] + 1e-9) +
        odds["prob_a_mkt"] * np.log(odds["prob_a_mkt"] + 1e-9)
    )

    odds_clean = odds[[
        "date", "season", "home_team", "away_team",
        "odd_h", "odd_d", "odd_a",
        "prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
        "odds_draw_factor", "odds_home_away_ratio", "market_entropy"
    ]].copy()

    print(f"   {len(odds_clean)} jogos com odds | válidos: {valid.sum()}")

    print("\n📂 Carregando features...")
    features = pd.read_csv(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    print(f"   {len(features)} jogos no features.csv")

    # Merge por data + times (tolerância de 3 dias)
    features = features.sort_values("date").reset_index(drop=True)
    odds_clean = odds_clean.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        features,
        odds_clean,
        on="date",
        by=["home_team", "away_team"],
        tolerance=pd.Timedelta("3 days"),
        direction="nearest"
    )

    # Verificar merge
    matched = merged["prob_h_mkt"].notna().sum()
    print(f"\n✅ Jogos com odds após merge: {matched} ({matched/len(merged):.1%})")
    print(f"   Por temporada:")
    print(merged.groupby("season")["prob_h_mkt"].apply(
        lambda x: f"{x.notna().sum()}/{len(x)}"
    ).to_string())

    # Salvar
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Salvo em {OUTPUT_PATH}")
    print(f"   Shape: {merged.shape}")


if __name__ == "__main__":
    # Copiar BRA.csv para pasta externa
    import shutil
    src = r"C:\PREDICTOR\REPO\scraping\data\external\BRA.csv"
    print(f"Certifique-se que BRA.csv está em:\n{src}\n")
    merge()