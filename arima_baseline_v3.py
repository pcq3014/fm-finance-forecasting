"""
ARIMA Baseline - V3 Univariate (Panel Data)
===========================================

Input:  data/processed/panel/panel_train_fixed.csv
        data/processed/panel/panel_val_fixed.csv
        data/processed/panel/panel_test_fixed.csv

Output: data/results/arima_summary.csv
        data/results/arima_predictions.csv

Model: ARIMA(1,1,1) per ticker on close_ret_1d (returns)
Univariate: Only uses past returns, no external features
Zero-shot: No feature engineering, no external data
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_train_fixed.csv"
VAL_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_val_fixed.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_test_fixed.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'y_return_1d'
ARIMA_ORDER = (1, 0, 1)  # y_return_1d already stationary, d=1 would double-difference
CONTEXT_LEN = 256


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0

    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    dir_acc = np.mean(dir_true == dir_pred)

    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}


def forecast_arima_rolling(train_val_series: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Rolling 1-step ARIMA forecast.
    Fit once on initial context, then use .append() to update state each step.
    ~20-50x faster than refitting from scratch each step.
    """
    ctx = train_val_series[-CONTEXT_LEN:].astype(np.float64)

    try:
        fitted = ARIMA(ctx, order=ARIMA_ORDER).fit(method='css-mle')
    except Exception:
        try:
            fitted = ARIMA(ctx, order=ARIMA_ORDER).fit()
        except Exception as e:
            print(f"\n  [DEBUG] initial fit failed: {e}")
            return np.zeros(len(y_test))

    preds = []
    for obs in y_test:
        fc = fitted.forecast(steps=1)
        preds.append(float(np.asarray(fc).flat[0]))
        # Fast state update — no full refit
        fitted = fitted.append([obs], refit=False)

    return np.array(preds, dtype=np.float32)


def main():
    print("\n" + "="*80)
    print("ARIMA BASELINE - V3 UNIVARIATE (PANEL DATA)")
    print("="*80)
    print(f"Order: ARIMA{ARIMA_ORDER} | Target: {TARGET}\n")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    tickers = sorted(df_train['ticker'].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Train: {len(df_train):,} rows")
    print(f"Val:   {len(df_val):,} rows")
    print(f"Test:  {len(df_test):,} rows\n")

    all_predictions = []
    all_val_predictions = []
    summary_rows = []

    print("="*80)
    print("Forecasting per ticker (univariate - y_return_1d)...")
    print("="*80 + "\n")

    for ticker in tickers:
        tr  = df_train[df_train['ticker'] == ticker].sort_values('date')
        val = df_val[df_val['ticker'] == ticker].sort_values('date')
        te  = df_test[df_test['ticker'] == ticker].sort_values('date')

        if len(tr) == 0 or len(te) == 0:
            continue

        ret_train  = tr[TARGET].values.astype(np.float32)
        ret_val    = val[TARGET].values.astype(np.float32)
        y_test_arr = te[TARGET].values.astype(np.float32)

        print(f"[{ticker:6s}] train={len(tr):>5d} | val={len(val):>4d} | test={len(te):>4d} -> ", end="", flush=True)

        try:
            # Roll through val+test in one pass using only train as context
            preds_all = forecast_arima_rolling(ret_train, np.concatenate([ret_val, y_test_arr]))
            y_pred_val = preds_all[:len(ret_val)]
            y_pred     = preds_all[len(ret_val):]

            if len(y_pred) != len(y_test_arr):
                print("Forecast failed -> Zero")
                y_pred_val = np.zeros(len(ret_val))
                y_pred     = np.zeros(len(y_test_arr))

            m = eval_metrics(y_test_arr, y_pred)
            print(f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | IC={m['ic']:.4f} | DA={m['dir_acc']:.3f}")

        except Exception as e:
            print(f"Error: {str(e)[:40]}")
            y_pred_val = np.zeros(len(ret_val))
            y_pred     = np.zeros(len(y_test_arr))
            m = eval_metrics(y_test_arr, y_pred)

        all_predictions.append(pd.DataFrame({
            'date': te['date'].values,
            'ticker': ticker,
            'y_true': y_test_arr,
            'y_pred': y_pred,
        }))
        all_val_predictions.append(pd.DataFrame({
            'date': val['date'].values,
            'ticker': ticker,
            'y_true': ret_val,
            'y_pred': y_pred_val,
        }))

        summary_rows.append({
            'ticker': ticker,
            'train_size': len(tr),
            'val_size': len(val),
            'test_size': len(te),
            **m
        })

    if all_predictions:
        predictions_df     = pd.concat(all_predictions, ignore_index=True)
        val_predictions_df = pd.concat(all_val_predictions, ignore_index=True)

        overall_metrics = eval_metrics(predictions_df['y_true'].values, predictions_df['y_pred'].values)
        summary_rows.append({
            'ticker': 'OVERALL',
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test),
            **overall_metrics
        })
        summary_df = pd.DataFrame(summary_rows)

        print("\n" + "="*80)
        print("ARIMA SUMMARY")
        print("="*80)
        print(summary_df.round(6).to_string(index=False))

        predictions_df.to_csv(OUT_DIR / "arima_predictions.csv", index=False)
        val_predictions_df.to_csv(OUT_DIR / "arima_val_predictions.csv", index=False)
        summary_df.to_csv(OUT_DIR / "arima_summary.csv", index=False)

        print(f"\nSaved: {OUT_DIR / 'arima_predictions.csv'}")
        print(f"Saved: {OUT_DIR / 'arima_val_predictions.csv'}")
        print(f"Saved: {OUT_DIR / 'arima_summary.csv'}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
