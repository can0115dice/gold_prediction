import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR


ROOT = Path(r"d:\Ma\HK\school\521 A.I\WC\gold_price_prediction\gold_price_prediction")
RAW_FILE = ROOT / "data" / "raw" / "gold_data_enriched.csv"
OUT_FILE = ROOT / "outputs" / "reports" / "seq_len_sweep_c11_results.csv"
PARTIAL_FILE = ROOT / "outputs" / "reports" / "seq_len_sweep_c11_results_partial.csv"

FEATURE_COLS = [
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


def calc(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return mae, rmse, mape, r2


def create_window_data(price_df, feature_cols, seq_len=20, mode="flat"):
    include_gold = "gold_price" in feature_cols
    all_cols = list(dict.fromkeys(feature_cols)) if include_gold else feature_cols + ["gold_price"]
    values = price_df[all_cols].values
    gold_idx = all_cols.index("gold_price")
    feat_idx = list(range(len(all_cols))) if include_gold else [i for i in range(len(all_cols)) if i != gold_idx]

    xs, ys, bases = [], [], []
    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        base = window[0].copy()
        base[base == 0] = 1e-8
        norm = window / base
        target_next = values[i + seq_len, gold_idx]
        y_ratio = target_next / base[gold_idx]

        if mode == "flat":
            last_step = norm[-1, feat_idx]
            x = last_step if include_gold else np.concatenate([last_step, norm[:, gold_idx]])
        else:
            x = norm[:, feat_idx]

        xs.append(x)
        ys.append(y_ratio)
        bases.append(base[gold_idx])

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(bases, dtype=np.float32)


def main():
    raw = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True).ffill().bfill()
    n = len(raw)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = raw.iloc[:train_end]
    test = raw.iloc[val_end:]

    rows = []

    # Keep existing LSTM/GRU rows from partial/results
    for src in [PARTIAL_FILE, OUT_FILE]:
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if "scheme" in df.columns:
            df = df[df["scheme"] == "C-全部(11维)"]
        if "model" not in df.columns or "seq_len" not in df.columns:
            continue
        for _, r in df.iterrows():
            model = str(r["model"]).upper()
            seq_len = int(r["seq_len"])
            if seq_len not in [20, 30, 50] or model not in ["LSTM", "GRU"]:
                continue
            rows.append(
                {
                    "model": model,
                    "seq_len": seq_len,
                    "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]),
                    "mape": float(r["mape"]),
                    "test_r2": float(r["test_r2"]) if "test_r2" in r else float(r.get("r2", np.nan)),
                    "train_r2": float(r["train_r2"]),
                    "gap": float(r["gap"]),
                    "train_time_sec": float(r["train_time_sec"]),
                }
            )

    # Run missing baselines for each seq_len
    for seq_len in [20, 30, 50]:
        x_tr, y_tr, b_tr = create_window_data(train, FEATURE_COLS, seq_len, "flat")
        x_te, y_te, b_te = create_window_data(test, FEATURE_COLS, seq_len, "flat")

        for name, model in [
            ("Linear", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0)),
            ("SVR", SVR(kernel="rbf", C=10.0, epsilon=0.01)),
        ]:
            t0 = time.perf_counter()
            model.fit(x_tr, y_tr)
            y_hat_te = model.predict(x_te)
            y_hat_tr = model.predict(x_tr)
            sec = time.perf_counter() - t0

            y_true = y_te * b_te
            y_pred = y_hat_te * b_te
            y_true_tr = y_tr * b_tr
            y_pred_tr = y_hat_tr * b_tr
            mae, rmse, mape, r2 = calc(y_true, y_pred)

            ss_res = np.sum((y_true_tr - y_pred_tr) ** 2)
            ss_tot = np.sum((y_true_tr - np.mean(y_true_tr)) ** 2)
            train_r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0

            rows.append(
                {
                    "model": name,
                    "seq_len": seq_len,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "test_r2": r2,
                    "train_r2": train_r2,
                    "gap": train_r2 - r2,
                    "train_time_sec": sec,
                }
            )

    out = pd.DataFrame(rows)
    out = out[out["model"].isin(["Linear", "Ridge", "SVR", "LSTM", "GRU"]) & out["seq_len"].isin([20, 30, 50])]
    out = out.sort_values(["seq_len", "model"]).drop_duplicates(["seq_len", "model"], keep="last").reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nWrote: {OUT_FILE}")


if __name__ == "__main__":
    main()
