# PREDICTOR — Brasileirao Serie A

Sistema de previsao de resultados de futebol para o Campeonato Brasileiro Serie A, com identificacao de value bets, simulacao de temporada via Monte Carlo e dashboard interativo.

---

## Visao Geral

O PREDICTOR e um pipeline completo de machine learning aplicado a analise esportiva. O sistema cobre desde a coleta de dados brutos ate a exposicao dos resultados em um dashboard web, passando por engenharia de features, treinamento de modelos, deteccao de value bets e validacao historica via backtesting.

O modelo principal atinge acuracia de **55.96%** na classificacao de resultados (Vitoria Mandante / Empate / Vitoria Visitante), superando a linha de base naive do mercado. A simulacao da temporada e executada com **10.000 iteracoes de Monte Carlo** para estimativa probabilistica da classificacao final.

---

## Arquitetura do Projeto

```
PREDICTOR/
├── modelos/
│   ├── match_model.pkl          # Modelo LightGBM serializado (3 classificadores binarios)
│   └── poisson_model.pkl        # Regressao de Poisson para previsao de placar
│
├── scraping/
│   ├── data/
│   │   ├── raw/                 # Partidas, fixtures e resultados brutos
│   │   ├── processed/           # Features engineered e odds integradas
│   │   └── external/            # Valores de mercado, odds ao vivo, backtesting
│
├── match_model.py               # Treinamento do modelo de classificacao de resultado
├── poisson_model.py             # Treinamento do modelo de previsao de gols
├── feature_engineering.py       # Construcao de features (ELO, forma, H2H, mercado)
├── season_model.py              # Simulacao Monte Carlo da temporada
├── backtesting.py               # Avaliacao historica das apostas
├── value_bets.py                # Identificacao de oportunidades com edge positivo
├── app.py                       # Dashboard Streamlit
│
├── api_football_collector.py    # Coleta via API-Football
├── football_data_api.py         # Coleta via Football-Data.org
├── odds_api.py                  # Coleta de odds em tempo real
├── odds_scrapper.py             # Scraping alternativo de odds
├── transfermarket_data.py       # Coleta de valores de mercado
├── convert_transfermarket.py    # Normalizacao dos dados do Transfermarkt
├── merge_datasets.py            # Consolidacao das fontes de dados
└── merge_odds.py                # Integracao das odds ao dataset de features
```

---

## Modelos

### Modelo de Classificacao de Resultado (`match_model.py`)

Arquitetura de **3 classificadores binarios independentes** treinados com LightGBM, um para cada resultado possivel (H, D, A). As probabilidades sao normalizadas para somarem 1.

Parametros do LightGBM:

| Parametro         | Valor  |
|-------------------|--------|
| n_estimators      | 300    |
| max_depth         | 4      |
| learning_rate     | 0.03   |
| num_leaves        | 15     |
| subsample         | 0.80   |
| colsample_bytree  | 0.80   |
| reg_alpha         | 0.30   |
| reg_lambda        | 0.30   |

Apos o treinamento, cada classificador passa por **calibracao isotonica** para correcao das probabilidades estimadas.

O treinamento utiliza **pesos temporais por temporada**, dando maior importancia aos dados recentes (2024: peso 4.0x, 2012: peso 0.3x). A avaliacao e feita com **cross-validacao temporal de 5 folds** para evitar data leakage.

Split de dados: treino em 2012-2024, teste em 2025-2026.

### Modelo de Previsao de Placar (`poisson_model.py`)

Dois modelos de **Regressao de Poisson** independentes, um para gols do mandante e outro para gols do visitante. As features de cada modelo sao especificas para o contexto (home/away). Os dados sao padronizados com `StandardScaler` antes do treinamento. Utilizado para gerar a distribuicao de placar esperada nas simulacoes.

---

## Features

O conjunto de features e construido em `feature_engineering.py` e contem 51 variaveis base, expandidas para 68 com features derivadas no momento do treinamento.

As principais categorias sao:

**Ratings ELO** — sistema de rating dinamico com K=32, calculado progressivamente sobre toda a historia de partidas.

**Forma recente** — media de pontos, gols marcados e sofridos, aproveitamento e taxa de vitoria nas ultimas 5 e 10 partidas, com separacao de desempenho em casa e fora.

**Tabela em tempo real** — posicao, saldo de gols e aproveitamento acumulado na temporada corrente no momento de cada partida.

**Head-to-head** — historico de confrontos diretos (vitorias, empates, derrotas).

**Valor de mercado** — valor total do elenco titular em milhoes de euros (fonte: Transfermarkt), normalizado e em escala logaritmica.

**Odds de mercado** — probabilidades implicitas da Pinnacle/Bet365 sem margem, alem de features derivadas como entropia de mercado, fator de empate e ratio home/away.

**Features derivadas** — diferencas entre times (forma, ELO, valor de mercado, aproveitamento), indicadores binarios (crise de forma, time em alta), similaridade entre times e divergencia entre estimativa do modelo e probabilidades de mercado.

---

## Identificacao de Value Bets (`value_bets.py`)

Uma aposta e classificada como value bet quando:

```
prob_modelo * odd_mercado > 1.05  (edge minimo de 5%)
prob_modelo >= 0.40               (confianca minima do modelo)
```

O dimensionamento da aposta e calculado pelo **Criterio de Kelly Fracionado** (25% do Kelly completo), que pondera o edge do modelo em relacao ao retorno oferecido pelo mercado.

```
kelly_pct = (prob_modelo * odd - 1) / (odd - 1) * 0.25
```

---

## Backtesting (`backtesting.py`)

Validacao historica nas temporadas 2025 e 2026 com dois criterios de stake:

**Flat:** 2% fixo do bankroll por aposta, com cap maximo de 5%.

**Kelly Fracionado:** stake variavel baseada no edge calculado, com cap maximo de 5% do bankroll.

O bankroll inicial de referencia e de 1000 unidades. As metricas reportadas sao ROI, Yield, Hit Rate e evolucao do bankroll ao longo do tempo.

---

## Simulacao de Temporada (`season_model.py`)

A simulacao executa 10.000 temporadas completas de forma paralela (`joblib`, 12 workers). Para cada partida nao disputada, o modelo estima as probabilidades de resultado com base nas features atualizadas dinamicamente ao longo da simulacao. O placar esperado e amostrado da distribuicao de Poisson parametrizada pelo modelo de gols.

Saida por time (`simulacao_2026.csv`):

- Probabilidade de titulo
- Probabilidade de classificacao para a Libertadores (top 6)
- Probabilidade de classificacao para a Sul-Americana (7-12)
- Probabilidade de rebaixamento (bottom 4)
- Pontuacao esperada e desvio padrao
- Posicao media esperada na tabela

---

## Dashboard (`app.py`)

Interface web construida em **Streamlit** com cinco abas:

| Aba              | Conteudo                                                                 |
|------------------|--------------------------------------------------------------------------|
| Proximos Jogos   | Previsoes para os proximos confrontos com probabilidades e placar esperado |
| Simulacao 2026   | Tabela probabilistica da temporada com projecoes de classificacao        |
| Value Bets       | Apostas com edge positivo identificadas pelo modelo                      |
| Backtesting      | Evolucao do bankroll e P&L historico com graficos interativos (Plotly)   |
| Valor de Mercado | Ranking dos elencos por valor de mercado (Transfermarkt)                 |

---

## Fontes de Dados

| Fonte             | Dados                                                         |
|-------------------|---------------------------------------------------------------|
| API-Football      | Fixtures, resultados e estatisticas de partidas em tempo real |
| Football-Data.org | Resultados historicos com odds (BRA.csv — 5.357 partidas)    |
| Transfermarkt     | Valor de mercado dos elencos, idade media e escalacoes        |
| Odds API / Scraper| Odds ao vivo para os proximos jogos (Bet365, Pinnacle)        |

O dataset historico cobre o periodo de **2012 a 2026**, totalizando 8.246 partidas com features completas apos o processo de merge e engenharia.

---

## Instalacao e Execucao

### Requisitos

```
Python >= 3.10
lightgbm
scikit-learn
pandas
numpy
scipy
joblib
streamlit
plotly
```

### Instalacao

```bash
git clone https://github.com/seu-usuario/predictor.git
cd predictor
pip install -r requirements.txt
```

### Execucao do Pipeline Completo

```bash
# 1. Coletar dados
python api_football_collector.py
python transfermarket_data.py
python odds_api.py

# 2. Consolidar e processar
python merge_datasets.py
python feature_engineering.py
python merge_odds.py

# 3. Treinar modelos
python match_model.py
python poisson_model.py

# 4. Simular temporada
python season_model.py

# 5. Identificar value bets
python value_bets.py

# 6. Gerar backtesting
python backtesting.py

# 7. Iniciar dashboard
streamlit run app.py
```

---

## Desempenho do Modelo

| Metrica                        | Valor     |
|-------------------------------|-----------|
| Acuracia no conjunto de teste  | 55.96%    |
| Simulacoes Monte Carlo         | 10.000    |
| Partidas no treino             | ~7.600    |
| Partidas no teste (2025-2026)  | ~640      |
| Features totais (com derivadas)| 68        |

---

## Avisos

Este projeto tem finalidade exclusivamente academica e de pesquisa em machine learning aplicado a dados esportivos. Nenhuma parte deste sistema constitui recomendacao financeira ou incentivo a pratica de apostas. O autor nao se responsabiliza pelo uso dos resultados gerados.

---

## Autor

Desenvolvido por **Nycolas F. Oliveira** — 2026