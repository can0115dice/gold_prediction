import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# app_live.py is now located inside the gold_price_prediction directory.
ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_CSV = REPORT_DIR / "seq_len_sweep_c11_results_partial.csv"
RESULTS_FALLBACK_CSV = REPORT_DIR / "seq_len_sweep_c11_results.csv"

MODEL_ORDER = ["Linear", "Ridge", "SVR", "LSTM", "GRU"]
SEQ_OPTIONS = [20, 30, 50]
EXPECTED_NOTEBOOK_FIGURES = [
    "feature_correlation.png",
    "cross_experiment.png",
    "best_traditional_prediction.png",
    "lstm_overfit_diagnosis.png",
    "lstm_prediction.png",
]

st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header {font-size: 2.1rem; color: #1f77b4; margin-bottom: .6rem;}
.small-note {color: #777; font-size: .85rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_results_table() -> pd.DataFrame:
    # Prefer complete results file; use partial as supplemental fallback.
    if not RESULTS_FALLBACK_CSV.exists() and not RESULTS_CSV.exists():
        return pd.DataFrame(columns=["model", "seq_len", "mae", "rmse", "mape", "test_r2", "train_r2", "gap", "train_time_sec"])

    frames = []
    if RESULTS_FALLBACK_CSV.exists():
        frames.append(pd.read_csv(RESULTS_FALLBACK_CSV))
    if RESULTS_CSV.exists():
        frames.append(pd.read_csv(RESULTS_CSV))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"r2": "test_r2"})

    required = ["model", "seq_len", "mae", "rmse", "mape", "test_r2", "train_r2", "gap", "train_time_sec"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    df["model"] = df["model"].astype(str).str.upper().replace({"LINEARREG": "LINEAR"})
    df["model"] = df["model"].replace({"LINEAR": "Linear", "RIDGE": "Ridge", "SVR": "SVR", "LSTM": "LSTM", "GRU": "GRU"})
    df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce")

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    # Keep the complete-file rows first, then fill missing from partial.
    df["__priority"] = 1
    if RESULTS_FALLBACK_CSV.exists():
        complete = pd.read_csv(RESULTS_FALLBACK_CSV)
        complete = complete.rename(columns={"r2": "test_r2"})
        complete["model"] = complete["model"].astype(str).str.upper().replace({"LINEARREG": "LINEAR"})
        complete["model"] = complete["model"].replace({"LINEAR": "Linear", "RIDGE": "Ridge", "SVR": "SVR", "LSTM": "LSTM", "GRU": "GRU"})
        complete["seq_len"] = pd.to_numeric(complete["seq_len"], errors="coerce")
        complete = complete[complete["model"].isin(MODEL_ORDER)].copy()
        complete["__priority"] = 0
        df = pd.concat([complete, df], ignore_index=True)

    df = df.sort_values(["seq_len", "model", "__priority"]).drop_duplicates(subset=["seq_len", "model"], keep="first")
    df = df.drop(columns=["__priority"])
    return df.reset_index(drop=True)


def build_metrics_map(df_seq: pd.DataFrame) -> dict:
    m = {}
    for model in MODEL_ORDER:
        row = df_seq[df_seq["model"] == model]
        if row.empty:
            m[model] = None
        else:
            r = row.iloc[0]
            m[model] = {
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "mape": float(r["mape"]),
                "test_r2": float(r["test_r2"]),
                "train_r2": float(r["train_r2"]),
                "gap": float(r["gap"]),
                "train_time_sec": float(r["train_time_sec"]),
            }
    return m


def available_models_for_seq(df_seq: pd.DataFrame):
    if df_seq.empty:
        return []
    return [m for m in MODEL_ORDER if m in set(df_seq["model"].tolist())]


def make_metrics_bar(df_seq: pd.DataFrame):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "MAPE"))
    for i, metric in enumerate(["mae", "rmse", "mape"], start=1):
        fig.add_trace(
            go.Bar(
                x=df_seq["model"],
                y=df_seq[metric],
                marker_color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756"][: len(df_seq)],
                showlegend=False,
            ),
            row=1,
            col=i,
        )
    fig.update_layout(template="plotly_white", height=380, title="Model Metrics Comparison")
    return fig


def placeholder_fig(title: str):
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="gray"))
    fig.update_layout(template="plotly_white", height=360, title=title)
    return fig


def load_prediction_df(seq_len: int) -> pd.DataFrame:
    pred_file = REPORT_DIR / "dashboard_model_predictions.csv"
    if not pred_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(pred_file)
    if "seq_len" in df.columns:
        df = df[df["seq_len"] == seq_len].copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    needed = ["Date", "Actual"] + MODEL_ORDER
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    return df[needed].dropna(subset=["Date"]).reset_index(drop=True)


def make_actual_vs_pred(pred_df: pd.DataFrame, model_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Actual"], mode="lines", name="Actual", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df[model_name], mode="lines", name=model_name, line=dict(color="#1f77b4", width=2, dash="dash")))
    fig.update_layout(template="plotly_white", height=420, title=f"Actual vs {model_name}")
    return fig


def load_loss_df(seq_len: int) -> pd.DataFrame:
    loss_file = REPORT_DIR / "dashboard_loss_history.csv"
    if not loss_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(loss_file)
    if "seq_len" in df.columns:
        df = df[df["seq_len"] == seq_len].copy()
    if not set(["model", "epoch", "train_loss", "val_loss"]).issubset(df.columns):
        return pd.DataFrame()
    return df[df["model"].isin(["LSTM", "GRU"])].copy()


def normalize_scheme_name(s: str) -> str:
    t = str(s).strip().upper()
    if t.startswith("A"):
        return "A-Macro(3D)"
    if t.startswith("B"):
        return "B-Extended(10D)"
    if t.startswith("C"):
        return "C-All(11D)"
    return str(s)


def filter_loss_by_scheme(loss_df: pd.DataFrame, scheme_label: str) -> pd.DataFrame:
    if loss_df.empty:
        return loss_df
    if "scheme" not in loss_df.columns:
        # Backward-compatible: old loss CSV without scheme defaults to C.
        default_scheme = "C-All(11D)"
        return loss_df.copy() if scheme_label == default_scheme else pd.DataFrame(columns=loss_df.columns)
    out = loss_df.copy()
    out["scheme"] = out["scheme"].map(normalize_scheme_name)
    return out[out["scheme"] == scheme_label].copy()


def make_single_model_loss_curve(loss_df: pd.DataFrame, model: str):
    sub = loss_df[loss_df["model"] == model].sort_values("epoch")
    if sub.empty:
        return placeholder_fig(f"{model} Loss Curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["epoch"], y=sub["train_loss"], mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(x=sub["epoch"], y=sub["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(template="plotly_white", height=360, title=f"{model} Loss Curves")
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss")
    return fig


def show_static_fallback_image(candidates, caption_prefix="Static fallback"):
    for name in candidates:
        p = FIG_DIR / name
        if p.exists():
            st.image(str(p), caption=f"{caption_prefix}: {name}", width="stretch")
            return True
    return False


@st.cache_data(ttl=300)
def load_raw_data() -> pd.DataFrame:
    raw_path = ROOT / "data" / "raw" / "gold_data_enriched.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    return df.sort_index()


def make_feature_corr_heatmap(raw_df: pd.DataFrame):
    use_cols = [c for c in ["gold_price", "usd_index", "oil_price", "sp500", "vix", "us10y_yield",
                            "silver_price", "copper_price", "tlt_bond", "bitcoin", "eur_usd"] if c in raw_df.columns]
    if not use_cols:
        return placeholder_fig("Feature Correlation Heatmap")
    corr = raw_df[use_cols].pct_change().dropna().corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} vs %{x}<br>corr=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_white", height=500, title="Feature Correlation (returns)")
    return fig


def make_seq_len_compare(df_all: pd.DataFrame):
    if df_all.empty:
        return placeholder_fig("SEQ_LEN Comparison")
    fig = go.Figure()
    for model, color in [("LSTM", "#d62728"), ("GRU", "#1f77b4"), ("Linear", "#2ca02c"), ("Ridge", "#ff7f0e"), ("SVR", "#9467bd")]:
        sub = df_all[df_all["model"] == model].sort_values("seq_len")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["seq_len"],
                y=sub["test_r2"],
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(template="plotly_white", height=380, title="Test R² vs SEQ_LEN")
    fig.update_xaxes(title="SEQ_LEN")
    fig.update_yaxes(title="Test R²")
    return fig


def make_gap_bar(df_seq: pd.DataFrame):
    if df_seq.empty:
        return placeholder_fig("Train-Test Gap")
    fig = go.Figure(
        go.Bar(
            x=df_seq["model"],
            y=df_seq["gap"],
            marker_color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756"][: len(df_seq)],
        )
    )
    fig.update_layout(template="plotly_white", height=340, title="Train-Test Gap (current SEQ_LEN)")
    fig.update_yaxes(title="Gap = train_r2 - test_r2")
    return fig


def existing_and_missing_files():
    expected = {
        "results_csv_primary": RESULTS_CSV,
        "results_csv_fallback": RESULTS_FALLBACK_CSV,
        "predictions_csv": REPORT_DIR / "dashboard_model_predictions.csv",
        "loss_csv": REPORT_DIR / "dashboard_loss_history.csv",
    }
    for name in EXPECTED_NOTEBOOK_FIGURES:
        expected[f"figure:{name}"] = FIG_DIR / name

    existing = {k: str(p) for k, p in expected.items() if p.exists()}
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return existing, missing


def main():
    st.markdown('<div class="main-header">📈 Gold Price Prediction Dashboard</div>', unsafe_allow_html=True)

    df_all = load_results_table()

    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("**Models:** Linear, Ridge, SVR, LSTM, GRU")
        seq_len = st.selectbox("Select SEQ_LEN", options=SEQ_OPTIONS, index=0)
        st.markdown(f"**Current SEQ_LEN:** {seq_len} days")
        st.markdown("---")

        st.markdown("### 📊 Model Performance")
        df_seq_sidebar = df_all[df_all["seq_len"] == seq_len].copy()
        metrics_map = build_metrics_map(df_seq_sidebar)
        selected_model = st.selectbox("Select model to view metrics", MODEL_ORDER)
        m = metrics_map.get(selected_model)
        if m is None:
            st.warning("No data available for this model at current SEQ_LEN")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("MAE", f"{m['mae']:.2f}")
                st.metric("RMSE", f"{m['rmse']:.2f}")
                st.metric("MAPE", f"{m['mape']:.2f}%")
            with c2:
                st.metric("Test R²", f"{m['test_r2']:.4f}")
                st.metric("Train R²", f"{m['train_r2']:.4f}")
                st.metric("Gap", f"{m['gap']:.4f}")

    st.markdown(f"<div class='small-note'>Current SEQ_LEN = {seq_len} days</div>", unsafe_allow_html=True)

    st.markdown("### 🧾 Results Table")
    df_seq = df_all[df_all["seq_len"] == seq_len].copy()
    if df_seq.empty:
        st.warning("No experimental results for current SEQ_LEN")
    else:
        st.dataframe(
            df_seq[["model", "seq_len", "mae", "rmse", "mape", "test_r2", "train_r2", "gap", "train_time_sec"]].sort_values("model"),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### 📈 Actual vs Predicted")
    pred_df = load_prediction_df(seq_len)
    pred_options = [m for m in MODEL_ORDER if (not pred_df.empty and m in pred_df.columns)]
    if not pred_options:
        pred_options = MODEL_ORDER
    model_choice = st.selectbox("Select prediction model", pred_options, key="pred_model")
    if pred_df.empty or model_choice not in pred_df.columns:
        shown = show_static_fallback_image(
            ["lstm_prediction.png", "best_traditional_prediction.png"],
            caption_prefix=f"Actual vs {model_choice}",
        )
        if not shown:
            st.plotly_chart(placeholder_fig(f"Actual vs {model_choice}"), width="stretch")
            st.caption("Prediction file missing for current SEQ_LEN, showing placeholder")
    else:
        st.plotly_chart(make_actual_vs_pred(pred_df, model_choice), width="stretch")

    st.markdown("### 📉 LSTM / GRU Loss Curves")
    loss_df = load_loss_df(seq_len)
    scheme_options = ["A-Macro(3D)", "B-Extended(10D)", "C-All(11D)"]
    selected_scheme = st.selectbox("Select feature scheme", scheme_options, index=2, key="loss_scheme")
    scheme_loss_df = filter_loss_by_scheme(loss_df, selected_scheme)
    c_lstm, c_gru = st.columns(2)
    with c_lstm:
        st.plotly_chart(make_single_model_loss_curve(scheme_loss_df, "LSTM"), width="stretch")
    with c_gru:
        st.plotly_chart(make_single_model_loss_curve(scheme_loss_df, "GRU"), width="stretch")
    if scheme_loss_df.empty:
        st.caption("No loss detail data for current SEQ_LEN + scheme combination, showing interactive placeholder")

    st.markdown("### 📦 Metrics Bar")
    if df_seq.empty:
        st.plotly_chart(placeholder_fig("Model Metrics Comparison"), width="stretch")
    else:
        # Only chart models with complete metric values.
        bar_df = df_seq.dropna(subset=["mae", "rmse", "mape"])
        if bar_df.empty:
            st.plotly_chart(placeholder_fig("Model Metrics Comparison"), width="stretch")
        else:
            st.plotly_chart(make_metrics_bar(bar_df), width="stretch")

    st.markdown("### 🖼️Notebook Visualizations (Interactive)")
    st.caption("Core notebook visualizations converted to interactive charts (zoom/hover), consistent with Actual vs Predicted")
    raw_df = load_raw_data()
    col_a, col_b = st.columns(2)
    with col_a:
        if raw_df.empty:
            st.plotly_chart(placeholder_fig("Feature Correlation Heatmap"), width="stretch")
        else:
            st.plotly_chart(make_feature_corr_heatmap(raw_df), width="stretch")
    with col_b:
        st.plotly_chart(make_seq_len_compare(df_all), width="stretch")
    st.plotly_chart(make_gap_bar(df_seq), width="stretch")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.8rem;'>"
        "Notebook-aligned models: Linear / Ridge / SVR / LSTM / GRU | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
