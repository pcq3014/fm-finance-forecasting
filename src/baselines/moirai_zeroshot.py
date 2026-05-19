"""
Moirai-2.0 Zero-shot Baseline (Close Price Target)
====================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/moirai_results.csv
        data/results/moirai_summary.csv

Strategy:
  - GluonTS PandasDataset + rolling 1-step sliding window
  - Predict Close price (median quantile) for every test day
  - Convert to log return: pred_ret = ln(pred_close / actual_close_T)
  - Evaluate with MAE, RMSE, DA, IC + per-regime breakdown

Model: Salesforce/moirai-2.0-R-small
Install: pip install uni2ts gluonts
Zero-shot: no fine-tuning, weights frozen.

NOTE: Predicting Close price (like ARIMA/Chronos), converting to return for metrics
      to ensure fair comparison with LSTM and other baselines.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
OUT_DIR    = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME      = "Salesforce/moirai-2.0-R-small"
CONTEXT_LENGTH  = 128   # ngay lich su moi buoc (nhu notebook goc)
PREDICTION_LENGTH = 1   # 1-step ahead


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # 1. MAE và RMSE: Tính trực tiếp trên giá tuyệt đối (Close Price)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 2. DA và IC: BẮT BUỘC phải chuyển sang sai phân (Sự thay đổi giá)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    if len(y_true_diff) > 0:
        # Directional Accuracy: Hôm nay giá tăng/giảm, model có đoán đúng hướng không?
        da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        
        # Information Coefficient: Tương quan giữa biến động thực tế và biến động dự báo
        ic = np.corrcoef(y_true_diff, y_pred_diff)[0, 1] if np.std(y_pred_diff) > 0 else 0.0
    else:
        da, ic = 0.0, 0.0
        
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def main():
    print("=" * 60)
    print(f"Moirai-2.0 ZERO-SHOT BASELINE | model={MODEL_NAME}")
    print(f"Context={CONTEXT_LENGTH} days | Prediction=1 day")
    print("=" * 60)

    # --- Load ---
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")

    # --- Import ---
    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        from gluonts.dataset.pandas import PandasDataset
        from gluonts.dataset.split import split
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run: pip install uni2ts gluonts")
        raise

    # --- Chuan bi panel: ticker + Date + OHLCV + 11 engineered features ---
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "vnm_ret", "vix_dev", "risk_off_composite", "hsi_vol_20",
        "macd", "vix", "hsi_ret_1d", "pmi_norm", "ibor_rate",
        "ibor_event", "usdvnd_dev",
    ]

    df_all = pd.concat([train, test], ignore_index=True)
    dfs = []
    for ticker_id, group in df_all.groupby("ticker"):
        g = group.set_index("Date").resample("B").ffill()
        g["ticker"] = ticker_id
        dfs.append(g.reset_index())

    df_ticker = pd.concat(dfs, ignore_index=True)[["ticker", "Date"] + feature_cols]
    print(f"Panel after resample: {len(df_ticker)} rows | {df_ticker['ticker'].nunique()} tickers")

    # --- GluonTS Dataset ---
    ds = PandasDataset.from_long_dataframe(
        df_ticker,
        target="Close",
        item_id="ticker",
        timestamp="Date",
        freq="B",
    )

    # So ngay test thuc te (per ticker)
    # FIX: Convert to native Python int (GluonTS doesn't accept numpy.int64)
    n_test = int(test.groupby("ticker")["Date"].nunique().max())
    pred_len = int(PREDICTION_LENGTH)
    print(f"Rolling windows (test days): {n_test}")

    train_dataset, test_template = split(ds, offset=-n_test)

    test_data = test_template.generate_instances(
        prediction_length=pred_len,
        windows=n_test,
        distance=pred_len,
    )

    # --- Load model ---
    print(f"\nLoading {MODEL_NAME}...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(MODEL_NAME),
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=64)
    print("Model loaded.")

    # --- Inference ---
    print("\nRunning zero-shot inference...")
    forecast_it    = predictor.predict(test_data.input)
    forecasts_list = list(forecast_it)
    print(f"Total forecasts: {len(forecasts_list)}")

    # --- Parse predictions: pred Close (no conversion) ---
    pred_rows = []
    for forecast in forecasts_list:
        ticker     = forecast.item_id
        pred_date  = forecast.start_date.to_timestamp()
        pred_close = float(forecast.quantile(0.5)[0])

        pred_rows.append({
            "Date"      : pred_date,
            "ticker"    : ticker,
            "pred"      : pred_close,
        })

    df_preds = pd.DataFrame(pred_rows)

    # --- Merge voi actual Close va regime ---
    test_meta = test[["Date", "ticker", "Close", "regime_hmm"]].copy()
    test_meta = test_meta.rename(columns={"Close": "target"})
    results = test_meta.merge(df_preds[["Date", "ticker", "pred"]], on=["Date", "ticker"], how="left")
    results["pred"] = results["pred"].fillna(0.0)
    results.to_csv(OUT_DIR / "moirai_results.csv", index=False)

    # --- Summary ---
    summary_rows = []
    for ticker in tickers:
        sub = results[results["ticker"] == ticker]
        m   = eval_metrics(sub["target"].values, sub["pred"].values)
        print(f"[{ticker}] MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")
        summary_rows.append({"ticker": ticker, **m})

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "moirai_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("Moirai-2.0 — OVERALL METRICS")
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

    print(f"\nSaved: {OUT_DIR / 'moirai_results.csv'}")
    print(f"Saved: {OUT_DIR / 'moirai_summary.csv'}")


if __name__ == "__main__":
    main()
