"""
backtest.py
===========
Rolling-window backtesting for electricity price forecasting models.

Methodology:
  - Train window: trailing 365 days of data
  - Forecast horizon: next 7 days (168 hours)
  - Step: slide forward by 7 days
  - Evaluation: last 12 months of the dataset (out-of-sample only)

This mirrors real-world operational constraints where a model is re-trained
weekly on the most recent year of data.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import build_feature_matrix, get_feature_cols, TARGET_COL
from src.models import get_all_models, GradientBoostingModel

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TRAIN_WINDOW_DAYS = 365
HORIZON_HOURS = 168  # 7 days
STEP_HOURS = 168


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, MAPE, RMSE, R² for a forecast."""
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return {"mae": round(mae, 2), "mape": round(mape, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def run_rolling_backtest(df_features: pd.DataFrame) -> dict:
    """
    Run rolling-window backtest for all models.

    Parameters
    ----------
    df_features : pd.DataFrame
        Full feature matrix (output of build_feature_matrix).

    Returns
    -------
    dict with keys = model names, values = dict with metrics and predictions DataFrame.
    """
    df = df_features.sort_values("ts").reset_index(drop=True)
    feature_cols = get_feature_cols(df)

    cutoff = df["ts"].max() - pd.Timedelta(days=365)
    eval_start_idx = df[df["ts"] >= cutoff].index[0]

    log.info("Backtest evaluation starts at index %d (ts=%s)", eval_start_idx, df.loc[eval_start_idx, "ts"])
    log.info("Features used: %s", feature_cols)

    models = get_all_models()
    results = {m.name: {"preds": [], "actuals": [], "timestamps": []} for m in models}

    train_hours = TRAIN_WINDOW_DAYS * 24
    start = eval_start_idx

    n_windows = 0
    while start + HORIZON_HOURS <= len(df):
        train_end = start
        train_start = max(0, train_end - train_hours)
        test_end = min(start + HORIZON_HOURS, len(df))

        X_train = df.loc[train_start:train_end - 1, feature_cols]
        y_train = df.loc[train_start:train_end - 1, TARGET_COL]
        X_test = df.loc[start:test_end - 1, feature_cols]
        y_test = df.loc[start:test_end - 1, TARGET_COL]
        ts_test = df.loc[start:test_end - 1, "ts"]

        if len(X_train) < 24 * 30:
            start += STEP_HOURS
            continue

        for model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[model.name]["preds"].extend(preds.tolist())
            results[model.name]["actuals"].extend(y_test.tolist())
            results[model.name]["timestamps"].extend(ts_test.tolist())

        start += STEP_HOURS
        n_windows += 1

    log.info("Backtest complete: %d rolling windows", n_windows)

    summary = {}
    for name, data in results.items():
        y_true = np.array(data["actuals"])
        y_pred = np.array(data["preds"])
        metrics = compute_metrics(y_true, y_pred)
        summary[name] = {
            "metrics": metrics,
            "df": pd.DataFrame({
                "ts": pd.to_datetime(data["timestamps"]),
                "actual": y_true,
                "predicted": y_pred,
            })
        }
        log.info("%s → MAE=%.1f  MAPE=%.1f%%  RMSE=%.1f  R²=%.3f",
                 name, metrics["mae"], metrics["mape"], metrics["rmse"], metrics["r2"])

    return summary


def save_metrics(summary: dict):
    """Save metrics to results/metrics.json."""
    metrics_out = {name: v["metrics"] for name, v in summary.items()}
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics saved to %s", out_path)
    return metrics_out


def plot_sample_week(summary: dict, model_name: str = "GradientBoosting"):
    """Plot actual vs predicted prices for a representative week."""
    df_pred = summary[model_name]["df"]
    sample = df_pred.iloc[:168]  # first week of eval period

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample["ts"], sample["actual"], label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(sample["ts"], sample["predicted"], label=f"Predicted ({model_name})",
            color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title("Day-Ahead Price: Actual vs Predicted (sample week)", fontsize=13)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("DAM Price (TL/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "sample_week_forecast.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved figure → %s", out)
    return str(out)


def plot_error_by_hour(summary: dict, model_name: str = "GradientBoosting"):
    """Plot mean absolute error by hour-of-day."""
    df_pred = summary[model_name]["df"].copy()
    df_pred["hour"] = pd.to_datetime(df_pred["ts"]).dt.hour
    df_pred["abs_error"] = np.abs(df_pred["actual"] - df_pred["predicted"])
    hourly_mae = df_pred.groupby("hour")["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly_mae.index, hourly_mae.values, color="#2196F3", alpha=0.8)
    ax.set_title(f"Mean Absolute Error by Hour-of-Day ({model_name})", fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (TL/MWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "error_by_hour.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved figure → %s", out)
    return str(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    from pathlib import Path
    processed = Path(__file__).parent.parent / "data" / "processed" / "hourly_market.parquet"
    if not processed.exists():
        log.error("Processed data not found. Run src/data_loader.py first.")
        raise SystemExit(1)

    df_raw = pd.read_parquet(processed)
    log.info("Loaded %d rows", len(df_raw))

    df_feat = build_feature_matrix(df_raw)
    log.info("Feature matrix: %d rows × %d cols", *df_feat.shape)

    summary = run_rolling_backtest(df_feat)
    metrics = save_metrics(summary)
    plot_sample_week(summary)
    plot_error_by_hour(summary)

    print("\n=== BACKTEST RESULTS ===")
    for model, m in metrics.items():
        print(f"  {model:20s}  MAE={m['mae']:7.1f}  MAPE={m['mape']:5.1f}%  RMSE={m['rmse']:7.1f}  R²={m['r2']:.3f}")
    print("\nDone.")
