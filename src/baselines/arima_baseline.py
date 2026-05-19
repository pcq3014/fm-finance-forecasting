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

Forecast strategy: Batch walk-forward validation (like Chronos-2).
                   Chia test set thành batch, mỗi batch fit trên history
                   (train + ngày trước batch), dự báo toàn batch cùng lúc.
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

ARIMA_ORDER = (1, 1, 1)
BATCH_DAYS  = 50  # Batch walk-forward: predict 50 test days at once


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # MAE và RMSE vẫn tính trên giá trị tuyệt đối để đo lường khoảng cách
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Tính DA và IC trên sai phân (sự thay đổi so với bước trước đó)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    if len(y_true_diff) > 0:
        # Hôm nay giá thực tế tăng/giảm, mô hình có đoán đúng hướng đó không?
        da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        
        # Mức độ tương quan giữa biến động thực tế và biến động dự báo
        ic = np.corrcoef(y_true_diff, y_pred_diff)[0, 1] if np.std(y_pred_diff) > 0 else 0.0
    else:
        da, ic = 0.0, 0.0
        
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def forecast_arima_batch_rolling(train_close: np.ndarray, test_close: np.ndarray, 
                                  test_dates: np.ndarray, ticker: str) -> np.ndarray:
    """
    Batch walk-forward validation for ARIMA.
    
    Chia test set thành batch, mỗi batch:
      1. Fit ARIMA trên history (train + ngày trước batch)
      2. Dự báo toàn batch cùng lúc (prediction_length = batch_size)
      3. Batch kế tiếp rebuild history với dữ liệu thực tế trước batch
    """
    n_test = len(test_close)
    preds_all = np.zeros(n_test)
    
    # Get unique test dates to create batches
    unique_dates = sorted(np.unique(test_dates))
    n_batches = (len(unique_dates) + BATCH_DAYS - 1) // BATCH_DAYS
    
    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * BATCH_DAYS
        batch_end_idx = min(batch_start_idx + BATCH_DAYS, len(unique_dates))
        batch_dates = unique_dates[batch_start_idx:batch_end_idx]
        batch_size = len(batch_dates)
        
        # Tìm indices của test set tương ứng với batch_dates này
        batch_test_indices = []
        for date in batch_dates:
            idx_list = np.where(test_dates == date)[0]
            if len(idx_list) > 0:
                batch_test_indices.append(idx_list[0])
        
        if len(batch_test_indices) == 0:
            continue
        
        batch_test_indices = sorted(batch_test_indices)
        
        # Rebuild history: train + các ngày test trước batch
        history = list(train_close)
        for idx in range(len(test_close)):
            if idx < batch_test_indices[0]:
                history.append(test_close[idx])
        
        try:
            # Fit ARIMA trên history hiện có
            model = ARIMA(history, order=ARIMA_ORDER)
            fitted = model.fit()
            
            # Dự báo batch_size bước
            batch_preds = fitted.forecast(steps=batch_size)
            
            # Assign predictions cho các indices trong batch
            for i, idx in enumerate(batch_test_indices):
                if i < len(batch_preds):
                    preds_all[idx] = batch_preds[i]
                    
        except Exception as e:
            print(f"    [{ticker}] ARIMA batch {batch_idx} failed ({e}) -> zero pred")
            for idx in batch_test_indices:
                preds_all[idx] = 0.0
    
    return preds_all


def main():
    print("=" * 60)
    print(f"ARIMA BASELINE — ARIMA{ARIMA_ORDER} (BATCH WALK-FORWARD)")
    print(f"Batch size: {BATCH_DAYS} days")
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
        test_dates  = te["Date"].values   # ← Test dates for batch alignment

        print(f"[{ticker}] train={len(train_close)} | test={len(test_close)}", end="  ->  ")
        
        # Batch walk-forward forecast
        preds = forecast_arima_batch_rolling(train_close, test_close, test_dates, ticker)

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
