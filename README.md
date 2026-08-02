#  PREDICTOR — Brasileirão Série A

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-00A0E4)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)](#avisos)

Sistema de previsão de resultados de futebol para o Campeonato Brasileiro Série A, com identificação de value bets, simulação de temporada via Monte Carlo e dashboard interativo.

---

## Visão Geral

O PREDICTOR é um pipeline completo de machine learning aplicado à análise esportiva. O sistema cobre desde a coleta de dados brutos até a exposição dos resultados em um dashboard web, passando por engenharia de features, treinamento de modelos, ensemble/stacking, detecção de value bets e validação histórica via backtesting.

O modelo principal (`match_model_v2`) combina **LightGBM + XGBoost** com **stacking via meta-learner** e HPO automático (Optuna), calibrado isotonicamente para estimar probabilidades de resultado (Vitória Mandante / Empate / Vitória Visitante). A simulação da temporada é executada com **10.000 iterações de Monte Carlo**, paralelizadas em múltiplos processos, para estimativa probabilística da classificação final.

---

## Arquitetura do Projeto

```
PREDICTOR/
├── dashboard/
│   ├── app.py                    # Dashboard Streamlit (interface principal)
│   └── predictor_logo.png
│
├── modelos/
│   ├── match_model_v2.py         # Modelo de resultado — ensemble LightGBM + XGBoost + stacking
│   ├── dixon_coles_model.py      # Modelo Dixon-Coles para previsão de gols/placar
│   ├── poisson_model.py          # Regressão de Poisson (gols mandante/visitante)
│   ├── lstm_model.py             # Modelo LSTM complementar (padrões sequenciais, requer torch)
│   ├── season_model.py           # Simulação Monte Carlo da temporada
│   ├── backtesting.py            # Avaliação histórica das apostas
│   ├── evaluation_metrics.py     # Métricas probabilísticas (RPS, Log-Loss, Brier, CLV)
│   ├── match_model.pkl           # Modelos serializados
│   ├── match_model_v2.pkl
│   ├── poisson_model.pkl
│   ├── dixon_coles_model.pkl
│   └── lstm_model.pkl
│
├── processing/
│   └── feature_engineering.py    # Construção de features (ELO, forma, H2H, mercado, odds)
│
├── scraping/
│   ├── api_football_collector.py # Coleta via API-Football
│   ├── football_data_api.py      # Coleta via Football-Data.org
│   ├── odds_api.py               # Coleta de odds em tempo real
│   ├── odds_scrapper.py          # Scraping alternativo de odds
│   ├── xg_scraper.py             # Coleta de xG via Sofascore
│   ├── transfermarket_data.py    # Coleta de valores de mercado (Transfermarkt)
│   ├── convert_transfermarket.py # Normalização dos dados do Transfermarkt
│   ├── merge_datasets.py         # Consolidação das fontes de dados
│   ├── merge_odds.py             # Integração das odds ao dataset de features
│   ├── value_bets.py             # Identificação de oportunidades com edge positivo
│   └── data/
│       ├── raw/                  # Partidas, fixtures e resultados brutos
│       ├── processed/            # Features engineered, odds integradas, simulações
│       └── external/             # Valores de mercado, odds ao vivo, backtesting
│
├── .streamlit/                   # Tema e configuração do dashboard
├── .devcontainer/                # Ambiente de desenvolvimento (Codespaces / VS Code)
├── requirements.txt
└── .env.example
```

---

## Modelos

### Modelo de Classificação de Resultado (`modelos/match_model_v2.py`)

Pipeline de **ensemble + stacking**:

1. **Busca de hiperparâmetros** com Optuna (100 trials) para o LightGBM.
2. **Ensemble** de LightGBM e XGBoost, com 3 classificadores binários independentes cada (um por resultado: H, D, A).
3. **Stacking**: meta-learner de Regressão Logística treinado sobre as probabilidades out-of-fold (OOF) dos modelos base.
4. **Calibração isotônica** aplicada às probabilidades finais.
5. Avaliação com **RPS (Ranked Probability Score)**, log-loss e Brier Score via `evaluation_metrics.py`.

O treinamento utiliza **pesos temporais por temporada**, priorizando dados recentes, com **cross-validação temporal (TimeSeriesSplit)** para evitar data leakage. Split de dados: treino em 2012–2024, teste em 2025–2026.

### Modelos de Previsão de Placar

- **`poisson_model.py`** — dois modelos de Regressão de Poisson independentes (gols do mandante e do visitante), com features padronizadas via `StandardScaler`.
- **`dixon_coles_model.py`** — extensão do modelo de Poisson com correção Dixon-Coles (parâmetro ρ) para resultados de baixa pontuação (0-0, 1-0, 0-1, 1-1), conforme Dixon & Coles (1997).

### Modelo Complementar (`modelos/lstm_model.py`)

Rede LSTM (PyTorch, opcional) que constrói sequências dos últimos jogos de cada time para capturar padrões de forma, combinando embeddings sequenciais com features estáticas (ELO, valor de mercado). Pode ser combinado com as probabilidades do `match_model_v2`.

---

## Features (`processing/feature_engineering.py`)

O conjunto de features contempla dezenas de variáveis, entre elas:

- **Ratings ELO** — sistema de rating dinâmico (K=32), calculado progressivamente sobre toda a história de partidas.
- **Forma recente** — pontos, gols marcados/sofridos, aproveitamento e taxa de vitória nas últimas 5 e 10 partidas, com separação de desempenho em casa e fora.
- **Tabela em tempo real** — posição, saldo de gols e aproveitamento acumulado na temporada corrente.
- **Head-to-head** — histórico de confrontos diretos (vitórias, empates, derrotas).
- **Valor de mercado** — valor total do elenco titular em milhões de euros (Transfermarkt), normalizado e em escala logarítmica.
- **Odds de mercado** — probabilidades implícitas (Pinnacle/Bet365) sem margem, entropia de mercado, fator de empate e ratio home/away.
- **Features derivadas** — diferenças entre times (forma, ELO, valor de mercado, aproveitamento), indicadores binários e divergência entre estimativa do modelo e mercado.

---

## Identificação de Value Bets (`scraping/value_bets.py`)

Uma aposta é classificada como value bet quando:

```
prob_modelo * odd_mercado > 1.08   (edge mínimo)
prob_modelo >= 0.55                (confiança mínima do modelo)
```

O dimensionamento da aposta é calculado pelo **Critério de Kelly Fracionado** (20% do Kelly completo):

```
kelly_pct = (prob_modelo * odd - 1) / (odd - 1) * 0.20
```

---

## Backtesting (`modelos/backtesting.py`)

Validação histórica nas temporadas 2025 e 2026, com dois critérios de stake:

- **Flat:** 2% fixo do bankroll por aposta, com cap máximo de 5%.
- **Kelly Fracionado:** stake variável baseada no edge calculado, com cap máximo de 5% do bankroll.

Bankroll inicial de referência: 1000 unidades. Métricas reportadas: ROI, Yield, Hit Rate, RPS, Log-Loss, CLV e evolução do bankroll ao longo do tempo.

---

## Simulação de Temporada (`modelos/season_model.py`)

Simulação de 10.000 temporadas completas em paralelo (`ProcessPoolExecutor`, backend `loky`), com calibração pré-computada e batching de rodadas para performance. Para cada partida não disputada, o modelo estima probabilidades de resultado com base em features atualizadas dinamicamente, e o placar é amostrado da distribuição de Poisson/Dixon-Coles.

Saída por time (`simulacao_2026.csv`):

- Probabilidade de título
- Probabilidade de classificação para a Libertadores (top 6)
- Probabilidade de classificação para a Sul-Americana (7–12)
- Probabilidade de rebaixamento (bottom 4)
- Pontuação esperada e desvio padrão
- Posição média esperada na tabela

---

## Dashboard (`dashboard/app.py`)

Interface web construída em **Streamlit** com múltiplas abas:

| Aba              | Conteúdo                                                                   |
|------------------|------------------------------------------------------------------------------|
| Próximos Jogos   | Previsões para os próximos confrontos, com probabilidades e placar esperado |
| Simulação 2026   | Tabela probabilística da temporada com projeções de classificação          |
| Value Bets       | Apostas com edge positivo identificadas pelo modelo                        |
| Backtesting      | Evolução do bankroll e P&L histórico com gráficos interativos (Plotly)     |
| Valor de Mercado | Ranking dos elencos por valor de mercado (Transfermarkt)                   |

Tema escuro customizado (`.streamlit/config.toml`), pronto para deploy no Streamlit Community Cloud (caminhos relativos, sem dependências absolutas de disco).

---

## Fontes de Dados

| Fonte             | Dados                                                          |
|-------------------|------------------------------------------------------------------|
| API-Football      | Fixtures, resultados e estatísticas de partidas em tempo real   |
| Football-Data.org | Resultados históricos com odds (`BRA.csv`)                       |
| Transfermarkt     | Valor de mercado dos elencos, idade média e escalações           |
| Sofascore         | Estatísticas de xG (Expected Goals) por partida                  |
| Odds API / Scraper| Odds ao vivo para os próximos jogos (Bet365, Pinnacle)           |

O dataset histórico cobre o período de **2012 a 2026**.

---

## Instalação e Execução

### Requisitos

```
Python >= 3.10
streamlit
pandas
numpy
scikit-learn
xgboost
catboost
lightgbm
optuna
joblib
scipy
plotly
pillow
requests
python-dotenv
matplotlib
```

### Instalação

```bash
git clone https://github.com/NyckzXD/FootballPredictor.git
cd FootballPredictor
pip install -r requirements.txt
cp .env.example .env   # preencha DATA_API_KEY e ODDS_API_KEY
```

### Execução do Pipeline Completo

```bash
# 1. Coletar dados
python scraping/api_football_collector.py
python scraping/transfermarket_data.py
python scraping/odds_api.py
python scraping/xg_scraper.py

# 2. Consolidar e processar
python scraping/merge_datasets.py
python processing/feature_engineering.py
python scraping/merge_odds.py

# 3. Treinar modelos
python modelos/match_model_v2.py
python modelos/poisson_model.py
python modelos/dixon_coles_model.py

# 4. Simular temporada
python modelos/season_model.py

# 5. Identificar value bets
python scraping/value_bets.py

# 6. Gerar backtesting
python modelos/backtesting.py

# 7. Iniciar dashboard
streamlit run dashboard/app.py
```

### Deploy

O projeto está pronto para deploy no [Streamlit Community Cloud](https://streamlit.io/cloud): basta apontar para `dashboard/app.py` como arquivo principal. Também há suporte a **GitHub Codespaces / Dev Container** (`.devcontainer/devcontainer.json`), que instala as dependências e sobe o dashboard automaticamente na porta `8501`.

---

## Métricas de Avaliação

O desempenho é medido além da acurácia simples, com métricas específicas para previsão probabilística multi-classe:

| Métrica       | Descrição                                                          |
|---------------|----------------------------------------------------------------------|
| Acurácia      | Taxa de acerto do resultado (H/D/A) mais provável                    |
| RPS           | Ranked Probability Score — penaliza erros distantes na ordem H/D/A   |
| Log-Loss      | Penalização logarítmica sobre as probabilidades previstas            |
| Brier Score   | Erro quadrático médio multivariado das probabilidades                |
| CLV           | Closing Line Value — valor do modelo frente à odd de fechamento      |

Split de avaliação: treino em 2012–2024, teste em 2025–2026 (~640 partidas).

---

## Avisos

Este projeto tem finalidade exclusivamente acadêmica e de pesquisa em machine learning aplicado a dados esportivos. Nenhuma parte deste sistema constitui recomendação financeira ou incentivo à prática de apostas. O autor não se responsabiliza pelo uso dos resultados gerados.

---

## Autor

Desenvolvido por **Nycolas F. Oliveira** — 2026
