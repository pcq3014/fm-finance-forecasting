"""
Chronos-2 Zero-shot Baseline (Close Price Target) — Fixed
==========================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/chronos_results.csv
        data/results/chronos_summary.csv

Fix vs v1:
  - Bỏ predict_df() / resample("B") gây timestamp mismatch trên VN30 calendar
  - Dùng pipeline() với tensor input trực tiếp (không cần timestamp)
  - Rolling 1-step: mỗi test day dùng context window ngay trước đó
  - Batch tất cả 15 tickers cùng lúc per inference call

Zero-shot: no fine-tuning, weights frozen.
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

CONTEXT_LEN    = 256
MODEL_ID       = "amazon/chronos-t5-small"   # fallback nếu chronos-2 chưa hỗ trợ tensor API
MODEL_ID_MAIN  = "amazon/chronos-2"
NUM_SAMPLES    = 50   # probabilistic samples → lấy median


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


def load_model():
    from chronos import ChronosPipeline, Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    # Thử Chronos-2 trước
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_ID_MAIN,
            device_map=device,
            dtype=dtype,
        )
        print(f"Chronos-2 loaded | device={device}")
        return pipeline
    except Exception as e:
        print(f"Chronos-2 failed ({e}), falling back to chronos-t5-small")
        pipeline = ChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map=device,
            dtype=dtype,
        )
        print(f"Chronos-t5-small loaded | device={device}")
        return pipeline


def predict_all_tickers(pipeline, close_dict: dict, n_test_dict: dict) -> dict:
    """
    Rolling 1-step cho tất cả tickers.
    Mỗi test day: batch 15 tickers cùng lúc → 876 inference calls.
    Dùng tensor input trực tiếp, không cần timestamp.
    """
    tickers  = list(close_dict.keys())
    n_tests  = {t: n_test_dict[t] for t in tickers}
    n_trains = {t: len(close_dict[t]) - n_test_dict[t] for t in tickers}
    n_test_max = max(n_tests.values())

    preds = {t: [] for t in tickers}

    for day_idx in range(n_test_max):
        contexts = []
        active   = []
        for t in tickers:
            if day_idx >= n_tests[t]:
                continue
            ctx_end = n_trains[t] + day_idx
            ctx = close_dict[t][:ctx_end].astype(np.float32)
            ctx = ctx[-CONTEXT_LEN:] if len(ctx) > CONTEXT_LEN else ctx
            contexts.append(torch.tensor(ctx))
            active.append(t)

        if not contexts:
            break

        try:
            # Chronos2Pipeline.predict() nhận inputs và trả list[Tensor]
            forecast = pipeline.predict(
                inputs=contexts,
                prediction_length=1,
                batch_size=len(contexts),
                context_length=CONTEXT_LEN,
            )
            # forecast: list of tensors, mỗi phần tử shape (num_samples, 1) hoặc (1,)
            for i, t in enumerate(active):
                f = forecast[i].cpu().float()
                # Lấy median qua samples nếu có nhiều dim, hoặc lấy trực tiếp
                if f.dim() > 1:
                    val = float(f.median(dim=0).values[0])
                else:
                    val = float(f[0])
                preds[t].append(val)
        except Exception as e:
            print(f"  WARNING day {day_idx}: {e} -> zero pred")
            for t in active:
                preds[t].append(0.0)

        if (day_idx + 1) % 100 == 0:
            print(f"  [{day_idx+1}/{n_test_max} days done]")

    return {t: np.array(preds[t], dtype=np.float64) for t in tickers}


def main():
    print("=" * 60)
    print("Chronos-2 ZERO-SHOT BASELINE (Fixed — tensor input)")
    print(f"Context={CONTEXT_LEN} | rolling 1-step | batch by ticker")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Test days: {test['Date'].nunique()} | "
          f"from {test['Date'].min().date()} to {test['Date'].max().date()}")

    # Chuẩn bị dữ liệu per ticker
    close_dict, n_test_dict, te_dict = {}, {}, {}
    for ticker in tickers:
        tr = train[train["ticker"] == ticker].sort_values("Date")
        te = test[test["ticker"]  == ticker].sort_values("Date")
        close_dict[ticker]  = np.concatenate([tr["Close"].values,
                                               te["Close"].values]).astype(np.float32)
        n_test_dict[ticker] = len(te)
        te_dict[ticker]     = te

    pipeline = load_model()

    print(f"\nRunning rolling inference ({max(n_test_dict.values())} days × {len(tickers)} tickers)...")
    all_ticker_preds = predict_all_tickers(pipeline, close_dict, n_test_dict)

    # Assemble results
    all_rows     = []
    summary_rows = []

    for ticker in tickers:
        te    = te_dict[ticker]
        preds = all_ticker_preds[ticker]
        y     = te["Close"].values

        m = eval_metrics(y, preds)
        print(f"[{ticker}] MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")
        summary_rows.append({"ticker": ticker, **m})

        df_pred = te[["Date", "ticker", "regime_hmm"]].copy()
        df_pred["target"] = y
        df_pred["pred"]   = preds
        all_rows.append(df_pred)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "chronos_results.csv", index=False)

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
        m = eval_metrics(results.loc[mask, "target"].values,
                         results.loc[mask, "pred"].values)
        print(f"  {regime:10s}: MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} "
              f"| DA={m['DA']:.3f} | IC={m['IC']:.4f} | n={mask.sum()}")

    print(f"\nSaved: {OUT_DIR / 'chronos_results.csv'}")
    print(f"Saved: {OUT_DIR / 'chronos_summary.csv'}")


if __name__ == "__main__":
    main()
