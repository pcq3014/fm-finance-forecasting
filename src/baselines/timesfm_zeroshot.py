"""
TimesFM 2.0 Zero-shot Baseline (Univariate)
============================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/timesfm_results.csv
        data/results/timesfm_summary.csv

Strategy:
  - TimesFM is UNIVARIATE: only takes Close price series
  - Feed raw Close prices as context (1D array)
  - Predict next-day Close price (1-step ahead)
  - Rolling window: use last CONTEXT_LEN days before each test day

Note: TimesFM cannot process multivariate input (unlike Chronos/Moirai).
      This creates unfair comparison, but TimesFM design is univariate-only.

Zero-shot: no fine-tuning, model weights frozen.
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

CONTEXT_LEN = 512   # days of history fed to model
BATCH_SIZE  = 16    # tickers per batch
BATCH_DAYS  = 5     # predict 5 days at a time (batch rolling, ~5x speedup)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da   = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic   = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def load_model():
    try:
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="torch",
                per_core_batch_size=BATCH_SIZE,
                horizon_len=1,
                context_len=CONTEXT_LEN,
                num_layers=50,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        print("TimesFM 2.0-500M loaded.")
        return tfm
    except Exception as e:
        print(f"ERROR loading TimesFM: {e}")
        raise


def predict_ticker(tfm, close_all: np.ndarray, n_test: int) -> np.ndarray:
    """
    Batch rolling prediction for n_test days.
    Predict BATCH_DAYS at a time, rolling context window (~5x speedup vs rolling 1-step).

    close_all: full Close price series [train + test], shape (T,)
              TimesFM is univariate - only takes 1D input
    Returns: predicted Close prices for test period, shape (n_test,)
    """
    n_total = len(close_all)
    n_train = n_total - n_test
    preds_close = []

    # Batch rolling: predict BATCH_DAYS at a time
    # 876 test days / 5 per batch = 175 batches (vs 876 rolling 1-step calls)
    for batch_start in range(0, n_test, BATCH_DAYS):
        batch_end = min(batch_start + BATCH_DAYS, n_test)
        batch_size = batch_end - batch_start

        # Context: use history up to batch_start (rolling window)
        context = close_all[:n_train + batch_start].astype(np.float32)

        if len(context) < 10:
            preds_close.extend([0.0] * batch_size)
            continue

        # TimesFM: predict batch_size days ahead
        try:
            point_forecast, _ = tfm.forecast(
                inputs=[context],
                freq=[0],  # 0 = daily
                horizon=batch_size,
            )
            # Extract predictions for this batch
            batch_preds = point_forecast[0][:batch_size]
            preds_close.extend(batch_preds.tolist())
        except Exception as e:
            print(f"    WARNING batch predict failed: {e} -> zero preds")
            preds_close.extend([0.0] * batch_size)

    return np.array(preds_close, dtype=np.float64)


def main():
    print("=" * 60)
    print("TimesFM 2.0 ZERO-SHOT BASELINE")
    print(f"Context length: {CONTEXT_LEN} days | Horizon: 1 day")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")

    tfm = load_model()

    all_preds    = []
    summary_rows = []

    for ticker in tickers:
        tr = train[train["ticker"] == ticker].sort_values("Date")
        te = test[test["ticker"]  == ticker].sort_values("Date")

        # TimesFM is univariate: only use Close prices
        close_train = tr["Close"].values.astype(np.float32)
        close_test  = te["Close"].values.astype(np.float32)
        close_all   = np.concatenate([close_train, close_test], axis=0)

        y_test_close = te["Close"].values
        n_test = len(te)

        print(f"\n[{ticker}] train={len(tr)} | test={n_test} | predicting...")
        preds = predict_ticker(tfm, close_all, n_test)

        m = eval_metrics(y_test_close, preds)
        print(f"  -> MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")

        df_pred = te[["Date", "ticker", "regime_hmm"]].copy()
        df_pred["target"] = y_test_close
        df_pred["pred"] = preds
        all_preds.append(df_pred)
        summary_rows.append({"ticker": ticker, **m})

    results = pd.concat(all_preds, ignore_index=True)
    results.to_csv(OUT_DIR / "timesfm_results.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "timesfm_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("TimesFM — OVERALL METRICS")
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

    print(f"\nSaved: {OUT_DIR / 'timesfm_results.csv'}")
    print(f"Saved: {OUT_DIR / 'timesfm_summary.csv'}")


if __name__ == "__main__":
    main()
