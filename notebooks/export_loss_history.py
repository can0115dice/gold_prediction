import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "gold_data_enriched.csv"
OUT_FILE = ROOT / "outputs" / "reports" / "dashboard_loss_history.csv"
MAX_TRAIN_SAMPLES = 320
MAX_VAL_SAMPLES = 96

SEQ_LENS = [20, 30, 50]
SCHEMES = {
    "A-宏观(3维)": ["usd_index", "oil_price", "sp500"],
    "B-全部外部(10维)": [
        "usd_index",
        "oil_price",
        "sp500",
        "vix",
        "us10y_yield",
        "silver_price",
        "copper_price",
        "tlt_bond",
        "bitcoin",
        "eur_usd",
    ],
    "C-全部(11维)": [
        "usd_index",
        "oil_price",
        "sp500",
        "vix",
        "us10y_yield",
        "silver_price",
        "copper_price",
        "tlt_bond",
        "bitcoin",
        "eur_usd",
        "gold_price",
    ],
}


def create_window_data(price_df, feature_cols, seq_len=20):
    include_gold_as_feature = "gold_price" in feature_cols
    all_cols = list(dict.fromkeys(feature_cols)) if include_gold_as_feature else feature_cols + ["gold_price"]
    values = price_df[all_cols].values
    gold_idx = all_cols.index("gold_price")
    feat_idx = list(range(len(all_cols))) if include_gold_as_feature else [i for i in range(len(all_cols)) if i != gold_idx]

    xs, ys = [], []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        base = window[0].copy()
        base[base == 0] = 1e-8
        norm_window = window / base
        target_next = values[i + seq_len, gold_idx]
        y_ratio = target_next / base[gold_idx]
        xs.append(norm_window[:, feat_idx])
        ys.append(y_ratio)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class RecurrentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, rnn_type="lstm"):
        super().__init__()
        rnn_type = rnn_type.lower()
        dr = dropout if num_layers > 1 else 0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr)
        else:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_with_losses(x_tr, y_tr, x_val, y_val, input_dim, rnn_type="lstm", epochs=25, lr=0.002, batch_size=256):
    torch.manual_seed(RANDOM_SEED)
    model = RecurrentModel(input_dim, rnn_type=rnn_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    x_tr_t = torch.FloatTensor(x_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    x_val_t = torch.FloatTensor(x_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    dataset = TensorDataset(x_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            train_loss = epoch_loss / len(x_tr)
            val_loss = criterion(model(x_val_t), y_val_t).item()
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {RAW_FILE}")

    raw_data = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True).ffill().bfill()
    n = len(raw_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_prices = raw_data.iloc[:train_end]
    val_prices = raw_data.iloc[train_end:val_end]

    rows = []
    for seq_len in SEQ_LENS:
        for scheme_name, feat_cols in SCHEMES.items():
            x_tr, y_tr = create_window_data(train_prices, feat_cols, seq_len)
            x_val, y_val = create_window_data(val_prices, feat_cols, seq_len)
            if len(x_tr) == 0 or len(x_val) == 0:
                continue
            # Use a fixed subset for fast dashboard loss visualization export.
            x_tr = x_tr[:MAX_TRAIN_SAMPLES]
            y_tr = y_tr[:MAX_TRAIN_SAMPLES]
            x_val = x_val[:MAX_VAL_SAMPLES]
            y_val = y_val[:MAX_VAL_SAMPLES]
            input_dim = x_tr.shape[2]

            for rnn_type in ("lstm", "gru"):
                model_name = rnn_type.upper()
                print(f"Training seq={seq_len}, scheme={scheme_name}, model={model_name}")
                train_losses, val_losses = train_with_losses(
                    x_tr, y_tr, x_val, y_val, input_dim=input_dim, rnn_type=rnn_type, epochs=25, lr=0.002
                )
                for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                    rows.append(
                        {
                            "seq_len": seq_len,
                            "scheme": scheme_name,
                            "model": model_name,
                            "epoch": i,
                            "train_loss": tr,
                            "val_loss": va,
                        }
                    )

    out = pd.DataFrame(rows).sort_values(["seq_len", "scheme", "model", "epoch"]).reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows to: {OUT_FILE}")


if __name__ == "__main__":
    main()
