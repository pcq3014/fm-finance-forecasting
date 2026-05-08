"""
ARIMA Baseline — per ticker, univariate (Close Price Target)
=============================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/arima_results.csv   (per-row predictions)
        data/results/arima_summary.csv   (per-ticker + overall metrics)

Model: ARIMA(1,0,1) per ticker on Close prices (NOT returns).
       Univariate — only uses past prices, no external features.
       d=0 because returns are stationary, but Close is I(1).

       After forecasting Close prices, convert to returns for metrics:
       pred_ret = ln(pred_close_t / close_{t-1})

Forecast strategy: fit on full train, one-shot forecast over test horizon.
Metrics: Evaluated on return scale (same as LSTM, FM models).
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
OUT_DIR    = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARIMA_ORDER = (1, 0, 1)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da   = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic   = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def forecast_arima_close(train_close: np.ndarray, n_steps: int, ticker: str) -> np.ndarray:
    """
    Train ARIMA on Close price.
    Returns predicted Close prices (not converted to return).
    """
    try:
        # Fit ARIMA on Close prices
        model  = ARIMA(train_close, order=ARIMA_ORDER)
        fitted = model.fit()

        # Forecast next n_steps Close prices
        preds_close = fitted.forecast(steps=n_steps)

        return np.array(preds_close)
    except Exception as e:
        print(f"  [{ticker}] ARIMA failed ({e}) -> zero forecast")
        return np.zeros(n_steps)


def main():
    print("=" * 60)
    print(f"ARIMA BASELINE — ARIMA{ARIMA_ORDER}")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}\n")

    all_preds    = []
    summary_rows = []

    for ticker in tickers:
        tr = train[train["ticker"] == ticker].sort_values("Date")
        te = test[test["ticker"]  == ticker].sort_values("Date")

        train_close = tr["Close"].values  # ← Close prices
        test_close  = te["Close"].values  # ← Actual Close prices for evaluation

        print(f"[{ticker}] train={len(train_close)} | test={len(test_close)}", end="  ->  ")
        preds = forecast_arima_close(train_close, len(test_close), ticker)

        m = eval_metrics(test_close, preds)
        print(f"MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")

        df_pred = te[["Date", "ticker", "Close", "regime_hmm"]].rename(columns={"Close": "target"}).copy()
        df_pred["pred"] = preds
        all_preds.append(df_pred)
        summary_rows.append({"ticker": ticker, **m})

    results = pd.concat(all_preds, ignore_index=True)
    results.to_csv(OUT_DIR / "arima_results.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "arima_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("ARIMA — OVERALL METRICS")
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

    print(f"\nSaved: {OUT_DIR / 'arima_results.csv'}")
    print(f"Saved: {OUT_DIR / 'arima_summary.csv'}")


if __name__ == "__main__":
    main()
