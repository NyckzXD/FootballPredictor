"""
lstm_model.py — Modelo LSTM para captura de padrões sequenciais de resultados.

Arquitetura:
  • Para cada partida, monta sequências dos últimos SEQ_LEN jogos de cada time
  • Duas LSTMs paralelas (casa e visitante) → embeddings de forma
  • Os embeddings são concatenados com features estáticas (ELO, mercado)
  • Saída: probabilidades P(H), P(D), P(A)

Uso como modelo complementar:
  - Treinar e salvar o modelo
  - Usar lstm_predict() para obter probabilidades que podem ser combinadas
    com as probabilidades do match_model_v2

Dependência: torch  (pip install torch)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import defaultdict, deque

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("⚠️  PyTorch não instalado. Execute: pip install torch")
    print("   O modelo LSTM não estará disponível.")

DATA_PATH  = r"C:\PREDICTOR\REPO\scraping\data\processed\features_odds.csv"
MODEL_PATH = r"C:\PREDICTOR\REPO\modelos\lstm_model.pkl"

SEQ_LEN    = 10     # últimos N jogos de cada time
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.3
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
DEVICE     = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"

# Features estáticas usadas junto com os embeddings LSTM
STATIC_FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "prob_h_mkt", "prob_d_mkt", "prob_a_mkt",
    "market_value_diff", "home_market_value_log", "away_market_value_log",
    "home_aproveitamento", "away_aproveitamento",
    "position_diff",
]

# Features por jogo na sequência (input para a LSTM)
# [pts, gf, ga, is_home, elo_pré]
SEQ_FEATURE_DIM = 5
OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Construção das sequências
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame):
    """
    Para cada partida, retorna:
      - seq_home: (SEQ_LEN, SEQ_FEATURE_DIM) — últimos N jogos do mandante
      - seq_away: (SEQ_LEN, SEQ_FEATURE_DIM) — últimos N jogos do visitante
      - X_static: (n_static,) — features estáticas da partida
      - y: 0/1/2 (H/D/A)

    A ordem das features na sequência:
      [pts_normalizados, gf, ga, is_home, elo_normalizado]
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Histórico incremental por time
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))
    elo_ratings: dict[str, float] = defaultdict(lambda: 1500.0)

    seqs_home, seqs_away, statics, labels = [], [], [], []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        result     = row.get("result", None)
        if result not in OUTCOME_MAP:
            continue

        home_seq = list(history[home])
        away_seq = list(history[away])

        def pad_seq(seq):
            arr = np.zeros((SEQ_LEN, SEQ_FEATURE_DIM), dtype=np.float32)
            for i, g in enumerate(seq[-SEQ_LEN:]):
                arr[i] = g
            return arr

        seqs_home.append(pad_seq(home_seq))
        seqs_away.append(pad_seq(away_seq))

        # Features estáticas
        static = []
        for col in STATIC_FEATURES:
            val = row.get(col, 0.0)
            static.append(float(val) if pd.notna(val) else 0.0)
        statics.append(static)
        labels.append(OUTCOME_MAP[result])

        # Atualizar histórico
        hg, ag = int(row.get("home_goals", 0)), int(row.get("away_goals", 0))
        h_pts  = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts  = 3 if ag > hg else (1 if ag == hg else 0)
        h_elo  = float(elo_ratings[home]) / 2000.0
        a_elo  = float(elo_ratings[away]) / 2000.0

        history[home].append([h_pts / 3, min(hg, 5) / 5, min(ag, 5) / 5, 1.0, h_elo])
        history[away].append([a_pts / 3, min(ag, 5) / 5, min(hg, 5) / 5, 0.0, a_elo])

        # Atualizar ELO simplificado
        ea = 1 / (1 + 10 ** ((elo_ratings[away] - elo_ratings[home]) / 400))
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo_ratings[home] += 32 * (score - ea)
        elo_ratings[away] += 32 * ((1 - score) - (1 - ea))

    return (
        np.array(seqs_home, dtype=np.float32),
        np.array(seqs_away, dtype=np.float32),
        np.array(statics, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class MatchDataset(Dataset):
    def __init__(self, seqs_home, seqs_away, statics, labels):
        self.sh = torch.tensor(seqs_home)
        self.sa = torch.tensor(seqs_away)
        self.st = torch.tensor(statics)
        self.y  = torch.tensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sh[idx], self.sa[idx], self.st[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Arquitetura LSTM
# ─────────────────────────────────────────────────────────────────────────────

class FootballLSTM(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int,
                 hidden: int = HIDDEN_DIM, n_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm_home = nn.LSTM(seq_dim, hidden, n_layers,
                                  batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.lstm_away = nn.LSTM(seq_dim, hidden, n_layers,
                                  batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.norm_home = nn.LayerNorm(hidden)
        self.norm_away = nn.LayerNorm(hidden)

        combined_dim = hidden * 2 + static_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, seq_home, seq_away, static_feat):
        _, (h_home, _) = self.lstm_home(seq_home)
        _, (h_away, _) = self.lstm_away(seq_away)
        emb_home = self.norm_home(h_home[-1])
        emb_away = self.norm_away(h_away[-1])
        x = torch.cat([emb_home, emb_away, static_feat], dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def train():
    if not TORCH_OK:
        print("❌ PyTorch não disponível. Execute: pip install torch")
        return

    print(f"🔧 Dispositivo: {DEVICE}")
    print("📊 Carregando dados...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "season_x" in df.columns: df = df.rename(columns={"season_x": "season"})
    if "result_x" in df.columns: df = df.rename(columns={"result_x": "result"})

    for col in STATIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    df = df[df["result"].isin(["H", "D", "A"])]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"   {len(df)} partidas | temporadas: {sorted(df['season'].unique())}")

    print("⚙️  Construindo sequências...")
    seqs_h, seqs_a, statics, labels = build_sequences(df)
    print(f"   Sequências: {seqs_h.shape} | Estáticas: {statics.shape}")

    # Split temporal
    seasons_arr = df["season"].values[:len(labels)]
    train_mask  = seasons_arr < 2025
    test_mask   = seasons_arr >= 2025

    ds_train = MatchDataset(seqs_h[train_mask], seqs_a[train_mask],
                             statics[train_mask], labels[train_mask])
    ds_test  = MatchDataset(seqs_h[test_mask], seqs_a[test_mask],
                             statics[test_mask], labels[test_mask])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n   Treino: {len(ds_train)} | Teste: {len(ds_test)}")

    model    = FootballLSTM(SEQ_FEATURE_DIM, len(STATIC_FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔧 Treinando LSTM ({EPOCHS} épocas)...")
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for sh, sa, st, y in dl_train:
            sh, sa, st, y = sh.to(DEVICE), sa.to(DEVICE), st.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(sh, sa, st)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        scheduler.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for sh, sa, st, y in dl_test:
                sh, sa, st, y = sh.to(DEVICE), sa.to(DEVICE), st.to(DEVICE), y.to(DEVICE)
                out   = model(sh, sa, st)
                loss  = criterion(out, y)
                val_loss += loss.item() * len(y)
                correct  += (out.argmax(1) == y).sum().item()
                total    += len(y)

        tl = train_loss / len(ds_train)
        vl = val_loss   / len(ds_test)
        va = correct    / total

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Época {epoch:3d}/{EPOCHS} | "
                  f"train_loss={tl:.4f} | val_loss={vl:.4f} | val_acc={va:.2%}")

    model.load_state_dict(best_state)
    print(f"\n✅ Melhor val_loss: {best_val_loss:.4f}")

    # Salvar
    joblib.dump({
        "state_dict":     best_state,
        "static_features": STATIC_FEATURES,
        "seq_len":        SEQ_LEN,
        "seq_feature_dim": SEQ_FEATURE_DIM,
        "hidden_dim":     HIDDEN_DIM,
        "num_layers":     NUM_LAYERS,
    }, MODEL_PATH)
    print(f"✅ Modelo LSTM salvo em {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    data  = joblib.load(MODEL_PATH)
    model = FootballLSTM(data["seq_feature_dim"], len(data["static_features"]),
                         data["hidden_dim"], data["num_layers"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


def lstm_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame com prob_h_lstm, prob_d_lstm, prob_a_lstm para cada linha de df.
    df deve ter as mesmas colunas que o conjunto de treino.
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch não instalado")
    model, meta = load_model()
    model.to(DEVICE)

    seqs_h, seqs_a, statics, _ = build_sequences(df)

    with torch.no_grad():
        sh = torch.tensor(seqs_h).to(DEVICE)
        sa = torch.tensor(seqs_a).to(DEVICE)
        st = torch.tensor(statics).to(DEVICE)
        logits = model(sh, sa, st)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    return pd.DataFrame({
        "prob_h_lstm": probs[:, 0],
        "prob_d_lstm": probs[:, 1],
        "prob_a_lstm": probs[:, 2],
    })


if __name__ == "__main__":
    train()
