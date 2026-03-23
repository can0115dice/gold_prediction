import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "gold_data_enriched.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCHEME_C_COLS = [
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
]

SEQ_LENS = [20, 30, 50, 60, 70, 90]


def calc_price_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def create_window_data(price_df, feature_cols, seq_len=20, mode="flat"):
    include_gold_as_feature = "gold_price" in feature_cols

    if include_gold_as_feature:
        all_cols = list(dict.fromkeys(feature_cols))
    else:
        all_cols = feature_cols + ["gold_price"]

    values = price_df[all_cols].values
    gold_idx = all_cols.index("gold_price")

    if include_gold_as_feature:
        feat_idx = list(range(len(all_cols)))
    else:
        feat_idx = [i for i in range(len(all_cols)) if i != gold_idx]

    xs, ys, bases, idxs = [], [], [], []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        base = window[0].copy()
        base[base == 0] = 1e-8
        norm_window = window / base

        target_next = values[i + seq_len, gold_idx]
        y_ratio = target_next / base[gold_idx]

        if mode == "flat":
            last_step = norm_window[-1, feat_idx]
            if not include_gold_as_feature:
                gold_history = norm_window[:, gold_idx]
                x = np.concatenate([last_step, gold_history])
            else:
                x = last_step
        else:
            x = norm_window[:, feat_idx]

        xs.append(x)
        ys.append(y_ratio)
        bases.append(base[gold_idx])
        idxs.append(price_df.index[i + seq_len])

    return (
        np.array(xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        np.array(bases, dtype=np.float32),
        idxs,
    )


class RecurrentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, rnn_type="lstm"):
        super().__init__()
        dr = dropout if num_layers > 1 else 0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dr)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_recurrent_model(
    x_tr, y_tr, x_val, y_val, x_te, input_dim, rnn_type="lstm", epochs=200, lr=0.002, batch_size=64
):
    torch.manual_seed(RANDOM_SEED)
    model = RecurrentModel(input_dim, rnn_type=rnn_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    x_tr_t = torch.FloatTensor(x_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    x_val_t = torch.FloatTensor(x_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    x_te_t = torch.FloatTensor(x_te).to(DEVICE)

    dataset = TensorDataset(x_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

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
            val_loss = criterion(model(x_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_te = model(x_te_t).cpu().numpy()
        y_pred_tr = model(x_tr_t).cpu().numpy()

    return y_pred_te, y_pred_tr, best_epoch


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {RAW_FILE}")

    raw_data = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True)
    data_clean = raw_data.copy().ffill().bfill()
    raw_prices = data_clean.copy()

    n = len(raw_prices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_prices = raw_prices.iloc[:train_end]
    val_prices = raw_prices.iloc[train_end:val_end]
    test_prices = raw_prices.iloc[val_end:]

    results = []

    # Linear baseline (run once as requested)
    seq0 = 20
    x_tr, y_tr, bases_tr, _ = create_window_data(train_prices, SCHEME_C_COLS, seq0, "flat")
    x_te, y_te, bases_te, _ = create_window_data(test_prices, SCHEME_C_COLS, seq0, "flat")
    t0 = time.perf_counter()
    linear = LinearRegression()
    linear.fit(x_tr, y_tr)
    y_pred_te = linear.predict(x_te)
    y_pred_tr = linear.predict(x_tr)
    sec = time.perf_counter() - t0

    y_true_price = y_te * bases_te
    y_pred_price = y_pred_te * bases_te
    y_true_tr_price = y_tr * bases_tr
    y_pred_tr_price = y_pred_tr * bases_tr
    m = calc_price_metrics(y_true_price, y_pred_price)
    train_r2 = 1 - np.sum((y_true_tr_price - y_pred_tr_price) ** 2) / np.sum(
        (y_true_tr_price - np.mean(y_true_tr_price)) ** 2
    )
    results.append(
        {
            "model": "Linear",
            "scheme": "C-全部(11维)",
            "seq_len": seq0,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape": m["mape"],
            "test_r2": m["r2"],
            "train_r2": train_r2,
            "gap": train_r2 - m["r2"],
            "train_time_sec": sec,
            "best_epoch": np.nan,
        }
    )

    for seq_len in SEQ_LENS:
        x_tr, y_tr, bases_tr, _ = create_window_data(train_prices, SCHEME_C_COLS, seq_len, "lstm")
        x_val, y_val, _, _ = create_window_data(val_prices, SCHEME_C_COLS, seq_len, "lstm")
        x_te, y_te, bases_te, _ = create_window_data(test_prices, SCHEME_C_COLS, seq_len, "lstm")

        input_dim = x_tr.shape[2]
        for rnn_type in ("lstm", "gru"):
            start = time.perf_counter()
            y_pred_te, y_pred_tr, best_epoch = train_recurrent_model(
                x_tr,
                y_tr,
                x_val,
                y_val,
                x_te,
                input_dim=input_dim,
                rnn_type=rnn_type,
                epochs=200,
                lr=0.002,
            )
            elapsed = time.perf_counter() - start

            y_true_price = y_te * bases_te
            y_pred_price = y_pred_te * bases_te
            y_true_tr_price = y_tr * bases_tr
            y_pred_tr_price = y_pred_tr * bases_tr
            m = calc_price_metrics(y_true_price, y_pred_price)
            train_r2 = 1 - np.sum((y_true_tr_price - y_pred_tr_price) ** 2) / np.sum(
                (y_true_tr_price - np.mean(y_true_tr_price)) ** 2
            )

            row = {
                "model": rnn_type.upper(),
                "scheme": "C-全部(11维)",
                "seq_len": seq_len,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "test_r2": m["r2"],
                "train_r2": train_r2,
                "gap": train_r2 - m["r2"],
                "train_time_sec": elapsed,
                "best_epoch": best_epoch,
            }
            results.append(row)
            print(
                f"[{row['model']}] seq={seq_len:>2}  "
                f"MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} MAPE={row['mape']:.2f}% "
                f"R2={row['test_r2']:.4f} gap={row['gap']:.4f} time={row['train_time_sec']:.1f}s"
            )

    df = pd.DataFrame(results)
    df = df.sort_values(["model", "seq_len"]).reset_index(drop=True)

    out_csv = REPORT_DIR / "seq_len_sweep_c11_results.csv"
    out_md = REPORT_DIR / "seq_len_sweep_c11_results.md"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_md.write_text(df.to_markdown(index=False), encoding="utf-8")

    print("\n=== Result Table ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
