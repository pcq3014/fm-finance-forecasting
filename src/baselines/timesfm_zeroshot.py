"""
TimesFM 2.0 Zero-shot Baseline (Univariate) — OPTIMIZED
========================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/timesfm_results.csv
        data/results/timesfm_summary.csv
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

# Tối ưu hóa lại Context và Batch Size cho GPU
CONTEXT_LEN = 256   # Rút gọn context xuống 256 (khoảng 1 năm) để model focus vào xu hướng gần
BATCH_SIZE  = 64    # Tăng số lượng context xử lý cùng lúc (tùy VRAM, nếu lỗi RAM thì giảm xuống 16/32)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # 1. MAE và RMSE: Tính trên giá tuyệt đối
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 2. DA và IC: BẮT BUỘC tính trên sự thay đổi (sai phân)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    if len(y_true_diff) > 0:
        da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        ic = np.corrcoef(y_true_diff, y_pred_diff)[0, 1] if np.std(y_pred_diff) > 0 else 0.0
    else:
        da, ic = 0.0, 0.0
        
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def load_model():
    try:
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="torch",
                per_core_batch_size=BATCH_SIZE,
                horizon_len=1, # Ép model chỉ dự báo 1 bước duy nhất cho mỗi context
                context_len=CONTEXT_LEN,
                num_layers=50,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        print("TimesFM 2.0-500M loaded successfully.")
        return tfm
    except Exception as e:
        print(f"ERROR loading TimesFM: {e}")
        raise


def predict_ticker_close_direct(tfm, close_all: np.ndarray, n_test: int) -> np.ndarray:
    """
    Dự báo TRỰC TIẾP giá Close bằng Rolling 1-Step theo Batch.
    """
    n_total = len(close_all)
    n_train = n_total - n_test

    contexts = []

    # 1. Chuẩn bị TOÀN BỘ các cửa sổ ngữ cảnh (Context Windows) với giá Close
    for day_idx in range(n_test):
        ctx_end = n_train + day_idx
        
        # Đưa trực tiếp giá Close vào làm context
        ctx = close_all[:ctx_end]
        ctx = ctx[-CONTEXT_LEN:] if len(ctx) > CONTEXT_LEN else ctx
        
        contexts.append(ctx.astype(np.float32))

    preds_close = []

    # 2. Suy luận theo lô (Batch Inference)
    for i in range(0, len(contexts), BATCH_SIZE):
        batch_ctx = contexts[i:i + BATCH_SIZE]
        
        try:
            point_forecast, _ = tfm.forecast(
                inputs=batch_ctx,
                freq=[0] * len(batch_ctx), 
                horizon=1,
            )
            
            for j in range(len(batch_ctx)):
                # Lấy trực tiếp kết quả là giá Close dự báo
                pred_close = float(point_forecast[j][0])
                preds_close.append(pred_close)
                
        except Exception as e:
            print(f"    WARNING batch {i} failed: {e} -> fallback to Naive forecast")
            # Fallback: Nếu lỗi, lấy luôn giá Close của ngày hôm trước
            for j in range(len(batch_ctx)):
                last_price = batch_ctx[j][-1]
                preds_close.append(last_price)

    return np.array(preds_close, dtype=np.float64)


def main():
    print("=" * 60)
    print("TimesFM 2.0 BASELINE (DIRECT CLOSE TARGET | 1-Step Rolling Batch)")
    print(f"Context: {CONTEXT_LEN} days")
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

        # Chỉ sử dụng mảng giá Close
        close_train = tr["Close"].values.astype(np.float32)
        close_test  = te["Close"].values.astype(np.float32)
        close_all   = np.concatenate([close_train, close_test], axis=0)

        y_test_close = te["Close"].values
        n_test = len(te)

        print(f"[{ticker}] train={len(tr)} | test={n_test} | predicting...", end=" ")
        
        # Gọi hàm dự báo trực tiếp giá Close
        preds = predict_ticker_close_direct(tfm, close_all, n_test)

        m = eval_metrics(y_test_close, preds)
        print(f"-> MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")

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
    print("TimesFM (DIRECT CLOSE) — OVERALL METRICS")
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