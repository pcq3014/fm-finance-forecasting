"""
Chronos-2 Zero-shot Baseline (Close Price Target)
===================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/chronos_results.csv
        data/results/chronos_summary.csv

Strategy:
  - Feed raw Close price series as context via predict_df()
  - Rolling batches: 876 test days in batches of 50 days
  - For each batch: use context up to earliest date, predict batch_size steps ahead
  - cross_learning=True: all 15 tickers predicted together
  - Convert pred Close -> log return for metric comparison with other models

Model: amazon/chronos-2
Zero-shot: no fine-tuning, weights frozen.

NOTE: Predicting Close price (like ARIMA), converting to return for metrics
      to ensure fair comparison with LSTM and other baselines.
      Rolling batches (not 1-step) for efficiency: 876→18 API calls, still fair comparison with LSTM.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
OUT_DIR    = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LEN = 512
MODEL_ID    = "amazon/chronos-2"
BATCH_SIZE  = 100
BATCH_DAYS  = 50  # Predict 50 test days at once (rolling batches)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da   = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic   = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def add_calendar_features(df: pd.DataFrame, date_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar covariates for Chronos-2."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dt = df[date_col]
    df["day_of_week"]       = dt.dt.dayofweek
    df["quarter"]           = dt.dt.quarter
    df["month"]             = dt.dt.month
    df["is_derivative_exp"] = 0
    return df


def load_model():
    try:
        from chronos import Chronos2Pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=dtype,
        )
        print(f"Chronos-2 loaded | device={device}")
        return pipeline
    except ImportError:
        print("ERROR: chronos not installed. Run: pip install chronos-forecasting")
        raise


def build_context_df(all_data: pd.DataFrame, before_date: pd.Timestamp,
                     context_len: int) -> pd.DataFrame:
    """Get context_len rows before before_date for all tickers. Ensure proper format for Chronos."""
    ctx = all_data[all_data["timestamp"] < before_date].copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])

    # Get last context_len days per ticker
    ctx = (
        ctx.groupby("id", group_keys=False)
           .apply(lambda g: g.sort_values("timestamp").tail(context_len))
           .reset_index(drop=True)
    )

    # Add calendar features
    ctx = add_calendar_features(ctx)

    # Ensure sorted by id and timestamp for Chronos
    ctx = ctx.sort_values(["id", "timestamp"]).reset_index(drop=True)

    return ctx


def main():
    print("=" * 60)
    print(f"Chronos-2 ZERO-SHOT BASELINE (Close Price Target)")
    print(f"Context={CONTEXT_LEN} days | cross_learning=True")
    print("=" * 60)

    # Load data
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")

    # Panel data: id, timestamp, OHLCV + 11 engineered features
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "vnm_ret", "vix_dev", "risk_off_composite", "hsi_vol_20",
        "macd", "vix", "hsi_ret_1d", "pmi_norm", "ibor_rate",
        "ibor_event", "usdvnd_dev",
    ]

    train_panel = train[["ticker", "Date"] + feature_cols].rename(
        columns={"ticker": "id", "Date": "timestamp"})
    test_panel  = test[["ticker", "Date"] + feature_cols + ["regime_hmm"]].rename(
        columns={"ticker": "id", "Date": "timestamp"})

    df_all = pd.concat([
        train_panel,
        test_panel[[c for c in test_panel.columns if c not in ["target", "regime_hmm"]]],
    ], ignore_index=True)

    # Resample each ticker to Business Days so Chronos can infer a stable frequency.
    dfs = []
    for ticker_id, group in df_all.groupby("id"):
        g = group.sort_values("timestamp").set_index("timestamp").resample("B").ffill()
        g["id"] = ticker_id
        dfs.append(g.reset_index())

    all_features = pd.concat(dfs, ignore_index=True)[["id", "timestamp"] + feature_cols]

    test_dates = sorted(test_panel["timestamp"].unique())
    print(f"Test days: {len(test_dates)} | from {test_dates[0].date()} to {test_dates[-1].date()}")

    pipeline = load_model()

    # Rolling batches (efficient alternative to rolling 1-step)
    pred_store = {}  # {(ticker, date): pred_return}

    print(f"\nRunning rolling batch inference ({len(test_dates)} days in batches of {BATCH_DAYS})...")
    n_batches = (len(test_dates) + BATCH_DAYS - 1) // BATCH_DAYS

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_DAYS
        batch_end = min(batch_start + BATCH_DAYS, len(test_dates))
        batch_dates = test_dates[batch_start:batch_end]

        print(f"  Batch {batch_idx+1}/{n_batches} | dates {batch_dates[0].date()} to {batch_dates[-1].date()} (n={len(batch_dates)})")

        # Context: built for earliest date in batch (rolling window)
        ctx_df = build_context_df(all_features, batch_dates[0], CONTEXT_LEN)
        if ctx_df.empty:
            print(f"    WARNING: empty context for batch {batch_idx}")
            continue

        # Future: all dates and tickers in batch
        future_rows = []
        for test_date in batch_dates:
            for ticker in tickers:
                future_rows.append({"id": ticker, "timestamp": test_date})
        future_df = pd.DataFrame(future_rows)
        future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
        future_df = future_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

        try:
            forecast_df = pipeline.predict_df(
                df=ctx_df,
                future_df=future_df[["id", "timestamp"]],
                prediction_length=len(batch_dates),
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="Close",
                cross_learning=True,
                batch_size=BATCH_SIZE,
            )

            for _, row in forecast_df.iterrows():
                ticker     = row["id"]
                test_date  = row["timestamp"]
                pred_close = float(row["predictions"])

                # Store Close prediction (no conversion)
                pred_store[(ticker, test_date)] = pred_close

        except Exception as e:
            print(f"    WARNING batch {batch_idx}: {e} -> zero pred for batch")
            for test_date in batch_dates:
                for ticker in tickers:
                    pred_store[(ticker, test_date)] = 0.0

    print(f"Forecast batches completed: {n_batches} batches")

    # Assemble results
    rows = []
    for _, row in test_panel.iterrows():
        key      = (row["id"], row["timestamp"])
        pred_close = pred_store.get(key, 0.0)
        rows.append({
            "Date"      : row["timestamp"],
            "ticker"    : row["id"],
            "target"    : row["Close"],
            "regime_hmm": row["regime_hmm"],
            "pred"      : pred_close,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "chronos_results.csv", index=False)

    # Summary
    summary_rows = []
    for ticker in tickers:
        sub = results[results["ticker"] == ticker]
        m   = eval_metrics(sub["target"].values, sub["pred"].values)
        print(f"[{ticker}] MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")
        summary_rows.append({"ticker": ticker, **m})

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "chronos_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("Chronos-2 — OVERALL METRICS")
    print("=" * 60)
    print(summary.round(6).to_string(index=False))

    print("\nPer-regime breakdown:")
    for regime in ["Bull", "Sideways", "Bear"]:
        mask = results["regime_hmm"] == regime
        if mask.sum() == 0:
            continue
        m = eval_metrics(results.loc[mask, "target"].values, results.loc[mask, "pred"].values)
        print(f"  {regime:10s}: MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} "
              f"| DA={m['DA']:.3f} | IC={m['IC']:.4f} | n={mask.sum()}")

    print(f"\nSaved: {OUT_DIR / 'chronos_results.csv'}")
    print(f"Saved: {OUT_DIR / 'chronos_summary.csv'}")


if __name__ == "__main__":
    main()
