"""
Chronos-2 Zero-shot Baseline (Log Return Target) - Fixed
========================================================

Input:  data/processed/panel/panel_train_fixed.csv
        data/processed/panel/panel_val_fixed.csv
        data/processed/panel/panel_test_fixed.csv

Output: data/results/chronos2_summary.csv
        data/results/chronos2_predictions.csv

Based on: src/baselines/chronos_2.py
- Rolling 1-step forecast per ticker
- Batch inference for efficiency
- Univariate: close_ret_1d input, y_return_1d target
- Context length: 512 (larger than baseline chronos-t5-small)
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

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_train_fixed.csv"
VAL_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_val_fixed.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_test_fixed.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LEN = 128
BATCH_SIZE = 32       # steps per forward pass
MODEL_ID_MAIN = "amazon/chronos-2"

TARGET = 'y_return_1d'
FEATURES = [
    'vn30_logret',
    'hsi_logret',
    'vn30_volume_log',
    'volume_ratio_20d',
    'trading_value_log',
    'foreign_net_ratio',
    'vnm_ret_lag1',
    'hsi_logret_lag1',
]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # IC: correlation on prediction values
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0

    # Direction Accuracy: sign agreement
    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    dir_acc = np.mean(dir_true == dir_pred)

    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}


def load_model():
    """Load Chronos-2 (Multivariate)"""
    try:
        from chronos.chronos2 import Chronos2Pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading {MODEL_ID_MAIN}...")
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_ID_MAIN,
            device_map=device,
            dtype=dtype,
        )

        print(f"✓ Chronos-2 loaded | device={device}")
        return pipeline

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run: conda activate env_core && pip install chronos-forecasting")
        return None
    except Exception as e:
        print(f"ERROR loading Chronos-2: {e}")
        return None


def predict_ticker_rolling(pipeline, y_all: np.ndarray, X_all: np.ndarray, n_forecast: int) -> np.ndarray:
    """
    Batched rolling 1-step forecast using correct Chronos-2 API:
      pipeline.predict_quantiles(inputs, prediction_length, quantile_levels)

    inputs = list of dicts:
      {"target": 1D array (ctx_len,),
       "past_covariates": {feat_name: 1D array (ctx_len,), ...}}

    output: (quantiles_list, mean_list)
      mean_list[i].shape = (1, prediction_length)  → use [0, 0] for scalar
    """
    n_ctx_start = len(y_all) - n_forecast
    preds = []

    for batch_start in range(0, n_forecast, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_forecast)
        batch_inputs = []

        for step in range(batch_start, batch_end):
            ctx_end = n_ctx_start + step
            start = max(0, ctx_end - CONTEXT_LEN)

            y_ctx = y_all[start:ctx_end].astype(np.float64)
            X_ctx = X_all[start:ctx_end].astype(np.float64)

            if len(y_ctx) == 0:
                batch_inputs.append(None)
                continue

            batch_inputs.append({
                "target": y_ctx,
                "past_covariates": {
                    name: X_ctx[:, i] for i, name in enumerate(FEATURES)
                },
            })

        valid_inputs = [x for x in batch_inputs if x is not None]
        if not valid_inputs:
            preds.extend([0.0] * (batch_end - batch_start))
            continue

        try:
            _, mean_list = pipeline.predict_quantiles(
                valid_inputs,
                prediction_length=1,
                quantile_levels=[0.5],
            )
            vi = 0
            for inp in batch_inputs:
                if inp is None:
                    preds.append(0.0)
                else:
                    # mean_list[i]: (num_variates=1, pred_len=1)
                    val = float(mean_list[vi][0, 0])
                    preds.append(val)
                    vi += 1

        except Exception as e:
            if batch_start == 0:
                print(f"\n  [DEBUG] predict error: {e}")
            preds.extend([0.0] * (batch_end - batch_start))

    return np.array(preds, dtype=np.float32)


def main():
    print("\n" + "="*80)
    print("CHRONOS-2 ZERO-SHOT BASELINE (Fixed - Log Return)")
    print("="*80)
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Target: {TARGET} (log return)")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES[:3])}...\n")

    # Load data
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    tickers = sorted(df_train['ticker'].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Train: {len(df_train):,} rows")
    print(f"Val:   {len(df_val):,} rows")
    print(f"Test:  {len(df_test):,} rows\n")

    # Load model
    pipeline = load_model()
    if pipeline is None:
        return

    print("\n" + "="*80)
    print("Rolling 1-step forecast per ticker...")
    print("="*80 + "\n")

    all_predictions     = []
    all_val_predictions = []
    summary_rows        = []

    for ticker in tickers:
        tr  = df_train[df_train['ticker'] == ticker].sort_values('date')
        val = df_val[df_val['ticker'] == ticker].sort_values('date')
        te  = df_test[df_test['ticker'] == ticker].sort_values('date')

        if len(tr) == 0 or len(te) == 0:
            continue

        y_train     = tr[TARGET].values.astype(np.float64)
        X_train     = tr[FEATURES].values.astype(np.float64)
        y_val       = val[TARGET].values.astype(np.float64)
        X_val       = val[FEATURES].values.astype(np.float64)
        y_test_arr  = te[TARGET].values.astype(np.float32)
        X_test      = te[FEATURES].values.astype(np.float64)

        # Roll through val+test in one pass using only train as context
        y_all    = np.concatenate([y_train, y_val, y_test_arr.astype(np.float64)])
        X_all    = np.concatenate([X_train, X_val, X_test])
        n_forecast = len(y_val) + len(y_test_arr)

        print(f"[{ticker:6s}] train={len(tr):>5d} | val={len(val):>4d} | test={len(te):>4d} -> ", end="", flush=True)

        try:
            preds_all = predict_ticker_rolling(pipeline, y_all, X_all, n_forecast)
            y_pred_val = preds_all[:len(y_val)].astype(np.float32)
            y_pred     = preds_all[len(y_val):].astype(np.float32)

            m = eval_metrics(y_test_arr, y_pred)
            print(f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | IC={m['ic']:.4f} | DA={m['dir_acc']:.3f}")

        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            y_pred_val = np.zeros(len(y_val), dtype=np.float32)
            y_pred     = np.zeros(len(y_test_arr), dtype=np.float32)
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
            'y_true': y_val.astype(np.float32),
            'y_pred': y_pred_val,
        }))

        summary_rows.append({
            'ticker': ticker,
            'train_size': len(tr),
            'val_size': len(val),
            'test_size': len(te),
            **m
        })

    if not all_predictions:
        print("No predictions generated!")
        return

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
    print("CHRONOS-2 SUMMARY")
    print("="*80)
    print(summary_df.round(6).to_string(index=False))

    predictions_df.to_csv(OUT_DIR / "chronos2_predictions_v3.csv", index=False)
    val_predictions_df.to_csv(OUT_DIR / "chronos2_val_predictions.csv", index=False)
    summary_df.to_csv(OUT_DIR / "chronos2_summary_v3.csv", index=False)

    print(f"\nSaved: {OUT_DIR / 'chronos2_predictions_v3.csv'}")
    print(f"Saved: {OUT_DIR / 'chronos2_val_predictions.csv'}")
    print(f"Saved: {OUT_DIR / 'chronos2_summary_v3.csv'}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
