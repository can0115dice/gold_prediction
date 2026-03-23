import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seq_len_sweep_c11 import (
    REPORT_DIR,
    RAW_FILE,
    SCHEME_C_COLS,
    calc_price_metrics,
    create_window_data,
    train_recurrent_model,
)


def run_linear_once(train_prices, test_prices):
    from sklearn.linear_model import LinearRegression

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
    return {
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


def main():
    raw_data = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True)
    data_clean = raw_data.copy().ffill().bfill()
    raw_prices = data_clean.copy()

    n = len(raw_prices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_prices = raw_prices.iloc[:train_end]
    val_prices = raw_prices.iloc[train_end:val_end]
    test_prices = raw_prices.iloc[val_end:]

    # Recovered rows from previous run log + linear baseline recomputed.
    results = [
        run_linear_once(train_prices, test_prices),
        {"model": "LSTM", "scheme": "C-全部(11维)", "seq_len": 20, "mae": 49.04, "rmse": 66.05, "mape": 1.51, "test_r2": 0.9874, "train_r2": 0.9933, "gap": 0.0059, "train_time_sec": 690.3, "best_epoch": np.nan},
        {"model": "GRU", "scheme": "C-全部(11维)", "seq_len": 20, "mae": 34.64, "rmse": 47.95, "mape": 1.07, "test_r2": 0.9934, "train_r2": 0.9962, "gap": 0.0028, "train_time_sec": 744.5, "best_epoch": np.nan},
        {"model": "LSTM", "scheme": "C-全部(11维)", "seq_len": 30, "mae": 41.19, "rmse": 57.22, "mape": 1.26, "test_r2": 0.9903, "train_r2": 0.9958, "gap": 0.0055, "train_time_sec": 870.8, "best_epoch": np.nan},
        {"model": "GRU", "scheme": "C-全部(11维)", "seq_len": 30, "mae": 35.81, "rmse": 49.63, "mape": 1.10, "test_r2": 0.9927, "train_r2": 0.9961, "gap": 0.0034, "train_time_sec": 343.4, "best_epoch": np.nan},
        {"model": "LSTM", "scheme": "C-全部(11维)", "seq_len": 50, "mae": 40.89, "rmse": 58.64, "mape": 1.24, "test_r2": 0.9894, "train_r2": 0.9958, "gap": 0.0064, "train_time_sec": 1155.0, "best_epoch": np.nan},
        {"model": "GRU", "scheme": "C-全部(11维)", "seq_len": 50, "mae": 35.79, "rmse": 50.12, "mape": 1.08, "test_r2": 0.9922, "train_r2": 0.9963, "gap": 0.0041, "train_time_sec": 567.5, "best_epoch": np.nan},
        {"model": "LSTM", "scheme": "C-全部(11维)", "seq_len": 60, "mae": 53.03, "rmse": 72.78, "mape": 1.57, "test_r2": 0.9832, "train_r2": 0.9948, "gap": 0.0116, "train_time_sec": 1263.5, "best_epoch": np.nan},
    ]

    pending = [(60, "gru"), (70, "lstm"), (70, "gru"), (90, "lstm"), (90, "gru")]

    for seq_len, rnn_type in pending:
        x_tr, y_tr, bases_tr, _ = create_window_data(train_prices, SCHEME_C_COLS, seq_len, "lstm")
        x_val, y_val, _, _ = create_window_data(val_prices, SCHEME_C_COLS, seq_len, "lstm")
        x_te, y_te, bases_te, _ = create_window_data(test_prices, SCHEME_C_COLS, seq_len, "lstm")
        input_dim = x_tr.shape[2]

        t0 = time.perf_counter()
        y_pred_te, y_pred_tr, best_epoch = train_recurrent_model(
            x_tr, y_tr, x_val, y_val, x_te, input_dim=input_dim, rnn_type=rnn_type, epochs=200, lr=0.002
        )
        elapsed = time.perf_counter() - t0

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

        # Save progress after each run.
        df_progress = pd.DataFrame(results)
        df_progress = df_progress.sort_values(["model", "seq_len"]).reset_index(drop=True)
        df_progress.to_csv(REPORT_DIR / "seq_len_sweep_c11_results_partial.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(results).sort_values(["model", "seq_len"]).reset_index(drop=True)
    out_csv = REPORT_DIR / "seq_len_sweep_c11_results.csv"
    out_md = REPORT_DIR / "seq_len_sweep_c11_results.md"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_md.write_text(df.to_markdown(index=False), encoding="utf-8")
    print("\n=== Result Table ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
