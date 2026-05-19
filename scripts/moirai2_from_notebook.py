"""
MOIRAI-2 Zero-shot - Converted from Jupyter Notebook
====================================================

Input:  data/processed/panel/panel_train_fixed.csv
        data/processed/panel/panel_val_fixed.csv
        data/processed/panel/panel_test_fixed.csv

Output: data/results/moirai2_predictions_v3.csv
        data/results/moirai2_summary_v3.csv
        data/results/moirai2_val_predictions.csv

Model: Moirai-2.0-R-small (Salesforce)
Multivariate: Uses y_return_1d target with 8 features
Zero-shot: No fine-tuning, weights frozen
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_train_fixed.csv"
VAL_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_val_fixed.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_test_fixed.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Parameters ---
TARGET = 'y_return_1d'
CONTEXT_LEN = 128
PREDICTION_LENGTH = 1
MODEL_NAME = "Salesforce/moirai-2.0-R-small"

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


def eval_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}


def main():
    print("\n" + "="*80)
    print("MOIRAI-2 ZERO-SHOT (CONVERTED FROM NOTEBOOK)")
    print("="*80)

    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        from gluonts.dataset.pandas import PandasDataset
        from gluonts.dataset.split import split
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run: pip install uni2ts gluonts")
        return

    # Load data
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Parse dates
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Prepare GluonTS Dataset - MULTIVARIATE with features
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    dfs = []
    for ticker_id, group in df_all.groupby("ticker"):
        g = group.set_index("date").resample("B").ffill()
        g["ticker"] = ticker_id
        dfs.append(g.reset_index())

    # Include target + features for multivariate
    feature_cols = [TARGET] + FEATURES
    df_ticker = pd.concat(dfs, ignore_index=True)[["ticker", "date"] + feature_cols]

    ds = PandasDataset.from_long_dataframe(
        df_ticker,
        target=TARGET,
        item_id="ticker",
        timestamp="date",
        freq="B",
        past_feat_dynamic_real=FEATURES,
    )

    # B-freq day spans — must use B-days because GluonTS resamples to freq="B"
    val_start  = df_val['date'].min()
    test_start = df_test['date'].min()
    test_end   = df_test['date'].max()

    n_test  = len(pd.bdate_range(test_start, test_end))
    # Roll from start of val through end of test in one pass (train = context only)
    n_total = len(pd.bdate_range(val_start, test_end))
    print(f"Rolling windows: val+test B-freq days = {n_total} (val portion ≈ {n_total - n_test})")

    _, combined_template = split(ds, offset=-n_total)

    combined_data = combined_template.generate_instances(
        prediction_length=PREDICTION_LENGTH,
        windows=n_total,
        distance=PREDICTION_LENGTH,
    )

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(MODEL_NAME),
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LEN,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=len(FEATURES),
    )
    predictor = model.create_predictor(batch_size=64)
    print("Model loaded.\n")

    # Inference — single pass covers val + test
    print("Running zero-shot inference (val + test)...")
    forecast_it = predictor.predict(combined_data.input)
    forecasts_list = list(forecast_it)

    pred_rows = []
    for forecast in forecasts_list:
        pred_rows.append({
            "date":   forecast.start_date.to_timestamp(),
            "ticker": forecast.item_id,
            "pred":   float(forecast.quantile(0.5)[0]),
        })

    df_preds = pd.DataFrame(pred_rows)

    # Split predictions into val and test by date
    df_pred_val  = df_preds[df_preds['date'] <  test_start].copy()
    df_pred_test = df_preds[df_preds['date'] >= test_start].copy()

    # --- Val results ---
    val_meta = df_val[["date", "ticker", TARGET]].copy().rename(columns={TARGET: "y_true"})
    val_results = val_meta.merge(df_pred_val, on=["date", "ticker"], how="left")
    val_results["pred"] = val_results["pred"].fillna(0.0)
    val_results = val_results.rename(columns={"pred": "y_pred"})

    # --- Test results ---
    test_meta = df_test[["date", "ticker", TARGET]].copy().rename(columns={TARGET: "y_true"})
    results = test_meta.merge(df_pred_test, on=["date", "ticker"], how="left")
    results["pred"] = results["pred"].fillna(0.0)

    # Calculate metrics per ticker (test period)
    summary_rows = []
    tickers = sorted(results['ticker'].unique())

    print("\n" + "="*80)
    print("PER-TICKER METRICS (test period)")
    print("="*80)

    for ticker in tickers:
        sub = results[results["ticker"] == ticker]
        m = eval_metrics(sub["y_true"].values, sub["pred"].values)
        print(f"[{ticker}] RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | IC={m['ic']:.4f} | DA={m['dir_acc']:.3f}")
        summary_rows.append({
            "ticker": ticker,
            "train_size": len(df_train[df_train['ticker'] == ticker]),
            "val_size":   len(df_val[df_val['ticker'] == ticker]),
            "test_size":  len(sub),
            **m,
        })

    summary_df = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["y_true"].values, results["pred"].values)
    summary_df = pd.concat([
        summary_df,
        pd.DataFrame([{"ticker": "OVERALL", "train_size": len(df_train),
                       "val_size": len(df_val), "test_size": len(df_test), **overall}])
    ], ignore_index=True)

    # Save results
    results_out = results.rename(columns={"pred": "y_pred"})
    results_out.to_csv(OUT_DIR / "moirai2_predictions_v3.csv", index=False)
    val_results.to_csv(OUT_DIR / "moirai2_val_predictions.csv", index=False)
    summary_df.to_csv(OUT_DIR / "moirai2_summary_v3.csv", index=False)

    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(summary_df.round(6).to_string(index=False))

    print(f"\nSaved: {OUT_DIR / 'moirai2_predictions_v3.csv'}")
    print(f"Saved: {OUT_DIR / 'moirai2_val_predictions.csv'}")
    print(f"Saved: {OUT_DIR / 'moirai2_summary_v3.csv'}")
    print("="*80)


if __name__ == "__main__":
    main()
