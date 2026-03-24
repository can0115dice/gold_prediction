# Gold Price Prediction

## Project Overview

This project forecasts gold prices using time series prediction and includes traditional machine learning models (Linear, Ridge, SVR) and deep learning models (LSTM, GRU). It compares results with visualizations to identify the best strategy. A Streamlit dashboard is integrated for real-time display.

## Project Structure

- `app_live.py`: Streamlit visualization dashboard.
- `data/raw/gold_data_enriched.csv`: Enriched raw data (after feature engineering).
- `data/processed/`: Train/validation/test split data.
- `notebooks/`:
  - `fill_seq_baselines.py`: Linear/Ridge/SVR baseline experiments.
  - `seq_len_sweep_c11.py`: Sequence length sweep experiments for LSTM/GRU.
  - `export_loss_history.py`: Export training loss history.
- `outputs/`:
  - `reports/seq_len_sweep_c11_results.csv`: Model metric results.
  - `figures/`: Experimental plots.
  - `dashboard_*.csv`: Dashboard data snapshots.
- `requirements.txt`: Python dependency list.

## Environment and Dependencies

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Current `requirements.txt`:

- streamlit==1.55.0
- pandas==2.3.3
- numpy==2.4.1
- plotly==6.6.0

> For full training execution, also install:
> - torch
> - scikit-learn

## Data and Reproduction

1. Ensure `outputs/reports/seq_len_sweep_c11_results.csv` or `seq_len_sweep_c11_results_partial.csv` exists; `app_live.py` loads it automatically.
2. If missing, run:
   - `notebooks/fill_seq_baselines.py`
   - `notebooks/seq_len_sweep_c11.py`
   - `notebooks/export_loss_history.py`
3. Output is saved by default to `outputs/`; adjust the `ROOT` variable in scripts if needed.

## Features

`app_live.py` provides:

- Global results loading and model options (model type, sequence length)
- Metric comparisons: MAE, RMSE, MAPE, R², gap, training time
- Visualizations: actual vs predicted, loss curves, feature correlations, seq_len sweep curves
