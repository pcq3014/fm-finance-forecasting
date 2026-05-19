"""
TimesFM 2.0 Zero-shot - V3 Fixed Data
====================================

Input:  data/processed/panel/panel_train_fixed.csv
        data/processed/panel/panel_val_fixed.csv
        data/processed/panel/panel_test_fixed.csv

Output: data/results/timesfm_summary.csv
        data/results/timesfm_predictions.csv

Zero-shot: No fine-tuning, model weights frozen
Univariate: Uses y_return_1d time series
"""

import sys
import warnings
import numpy as np
import pandas as pd
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

TARGET = 'y_return_1d'  # Log return
CONTEXT_LEN = 128  # Context window

# ============================================================================
# Metrics
# ============================================================================

def eval_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0

    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    dir_acc = np.mean(dir_true == dir_pred)

    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}

# ============================================================================
# TimesFM Prediction
# ============================================================================

def forecast_timesfm(model, ret_all, n_ctx_start, n_forecast):
    """
    Rolling 1-step forecast using TimesFM.
    ret_all    : concat(train+val, test_actual) — actual values throughout
    n_ctx_start: index where test period begins (= len(train+val))
    """
    preds = []
    for i in range(n_forecast):
        ctx_end = n_ctx_start + i
        ctx = ret_all[max(0, ctx_end - CONTEXT_LEN):ctx_end]

        if len(ctx) == 0:
            preds.append(0.0)
            continue

        try:
            point_forecast, _ = model.forecast(
                inputs=[ctx.astype(np.float32)],
                freq=[0],
            )
            preds.append(float(point_forecast[0][0]))
        except Exception as e:
            print(f"\n  [DEBUG] step {i} error: {str(e)[:50]}")
            preds.append(0.0)

    return np.array(preds)

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("TimesFM 2.0 ZERO-SHOT - V3 FIXED DATA")
    print("="*80)

    # Try to load TimesFM
    try:
        # Monkey-patch snapshot_download to disable symlinks globally (Windows fix)
        import huggingface_hub as _hfhub
        _orig_snapshot = _hfhub.snapshot_download
        def _snapshot_no_symlinks(*args, **kwargs):
            kwargs.setdefault('local_dir_use_symlinks', False)
            return _orig_snapshot(*args, **kwargs)
        _hfhub.snapshot_download = _snapshot_no_symlinks

        import timesfm

        print("Loading TimesFM 1.0-200M...")
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="torch",
                per_core_batch_size=32,
                horizon_len=1,
                context_len=CONTEXT_LEN,
                num_layers=20,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        print("TimesFM loaded successfully!\n")

    except ImportError as e:
        print(f"ERROR: TimesFM not available - {e}")
        print("Install: pip install timesfm")
        return

    # Load data
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    tickers = sorted(df_train['ticker'].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")

    all_predictions     = []
    all_val_predictions = []
    summary_rows        = []

    print("\n" + "="*80)
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

        # Roll through val+test in one pass using only train as context
        ret_all     = np.concatenate([ret_train, ret_val, y_test_arr])
        n_ctx_start = len(ret_train)
        n_forecast  = len(ret_val) + len(y_test_arr)

        print(f"[{ticker:6s}] train={len(tr):>5d} | val={len(val):>4d} | test={len(te):>4d} -> ", end="", flush=True)

        try:
            preds_all = forecast_timesfm(model, ret_all, n_ctx_start, n_forecast)

            if preds_all is None or len(preds_all) != n_forecast:
                print("Forecast failed -> Zero")
                preds_all = np.zeros(n_forecast)

            y_pred_val = preds_all[:len(ret_val)]
            y_pred     = preds_all[len(ret_val):]

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
        print("TimesFM SUMMARY")
        print("="*80)
        print(summary_df.round(6).to_string(index=False))

        predictions_df.to_csv(OUT_DIR / "timesfm_predictions.csv", index=False)
        val_predictions_df.to_csv(OUT_DIR / "timesfm_val_predictions.csv", index=False)
        summary_df.to_csv(OUT_DIR / "timesfm_summary.csv", index=False)

        print(f"\nSaved: {OUT_DIR / 'timesfm_predictions.csv'}")
        print(f"Saved: {OUT_DIR / 'timesfm_val_predictions.csv'}")
        print(f"Saved: {OUT_DIR / 'timesfm_summary.csv'}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
