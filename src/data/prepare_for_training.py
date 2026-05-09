"""
Prepare for Training — Final data preparation
==============================================
Input:  data/processed/v2/features_selected.csv

Output:
  data/processed/v2/features_train.csv  — Train set, normalized
  data/processed/v2/features_test.csv   — Test set, normalized (dung scaler tu Train)
  data/processed/v2/scaler_params.json  — Mean/Std per feature (de inverse-transform)

Pipeline:
  1. Tao target y = ret_1d.shift(-1) per ticker
  2. Drop NaN rows (warm-up tu rolling windows)
  3. Tach Train (Date <= 2022-06-30) / Test (Date > 2022-06-30)
  4. Z-score normalize: fit StandardScaler tren TRAIN only, apply ca 2

Cot KHONG normalize:
  - Date, ticker (id)
  - Open, High, Low, Close, Volume (raw, khong dung lam input)
  - regime_hmm, regime_vol (categorical)
  - ibor_event (binary 0/1)
  - target (giu nguyen scale return de tinh CRPS/MAE)
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
SEL_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_selected.csv"
OUT_DIR   = ROOT_DIR / "data" / "processed" / "v2"
TRAIN_OUT = OUT_DIR / "features_train.csv"
TEST_OUT  = OUT_DIR / "features_test.csv"
SCALER_OUT = OUT_DIR / "scaler_params.json"

TRAIN_END = "2022-06-30"

# Cot khong normalize
NO_NORM = {
    "Date", "ticker",
    "Open", "High", "Low", "Close", "Volume",
    "regime_hmm", "regime_vol",
    "ibor_event",   # binary
    "target",       # giu nguyen scale return
}


def main():
    print("=" * 60)
    print("PREPARE FOR TRAINING")
    print("=" * 60)

    # 1. Load
    print("\n[1] Load features_selected.csv")
    df = pd.read_csv(SEL_PATH, parse_dates=["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    print(f"   Initial shape: {df.shape}")

    # 2. Tao target y = next-day return
    print("\n[2] Tao target = ret_1d.shift(-1) per ticker")
    if "ret_1d" not in df.columns:
        # ret_1d khong nam trong selected -> compute tu Close
        df["ret_1d_tmp"] = df.groupby("ticker")["Close"].apply(
            lambda s: np.log(s / s.shift(1))
        ).reset_index(level=0, drop=True)
        df["target"] = df.groupby("ticker")["ret_1d_tmp"].shift(-1)
        df = df.drop(columns=["ret_1d_tmp"])
    else:
        df["target"] = df.groupby("ticker")["ret_1d"].shift(-1)
    print(f"   Target NaN (last row per ticker): {df['target'].isna().sum()}")

    # 3. Drop NaN rows
    print("\n[3] Drop NaN rows (warm-up + last-row target)")
    n_before = len(df)
    df_clean = df.dropna()
    n_after = len(df_clean)
    print(f"   Before: {n_before:,} | After: {n_after:,} | Dropped: {n_before-n_after:,}")
    print(f"   Date range after drop: {df_clean['Date'].min().date()} -> {df_clean['Date'].max().date()}")

    # 4. Tach Train / Test
    print(f"\n[4] Split Train/Test (cutoff: {TRAIN_END})")
    train_mask = df_clean["Date"] <= TRAIN_END
    train = df_clean[train_mask].copy().reset_index(drop=True)
    test  = df_clean[~train_mask].copy().reset_index(drop=True)
    print(f"   Train: {len(train):,} rows | dates {train['Date'].nunique()}")
    print(f"   Test:  {len(test):,} rows | dates {test['Date'].nunique()}")

    # 5. Identify features to normalize
    norm_cols = [c for c in train.columns if c not in NO_NORM]
    print(f"\n[5] Z-score normalize {len(norm_cols)} features (fit on TRAIN only)")
    for c in norm_cols:
        print(f"   - {c}")

    # 6. Fit scaler on TRAIN
    scaler = StandardScaler()
    train[norm_cols] = scaler.fit_transform(train[norm_cols])
    test[norm_cols]  = scaler.transform(test[norm_cols])

    # Verify
    print("\n   Train stats after normalize (should be ~mean 0, std 1):")
    stats = train[norm_cols].agg(["mean", "std"]).T.round(3)
    print(stats.to_string())

    print("\n   Test stats after normalize (using Train scaler):")
    stats_test = test[norm_cols].agg(["mean", "std"]).T.round(3)
    print(stats_test.to_string())

    # 7. Save scaler params (JSON for portability)
    scaler_dict = {
        "features": norm_cols,
        "mean": scaler.mean_.tolist(),
        "std":  scaler.scale_.tolist(),
        "train_end_date": TRAIN_END,
        "n_train_rows": len(train),
        "n_test_rows":  len(test),
    }
    with open(SCALER_OUT, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"\n[6] Saved scaler params: {SCALER_OUT}")

    # 8. Save train + test
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"[7] Saved train: {TRAIN_OUT} | Shape: {train.shape}")
    print(f"    Saved test:  {TEST_OUT} | Shape: {test.shape}")

    # 9. Sanity check
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    print(f"   Train tickers: {train['ticker'].nunique()} (expected: 15)")
    print(f"   Test  tickers: {test['ticker'].nunique()} (expected: 15)")
    print(f"   Train target range: [{train['target'].min():.4f}, {train['target'].max():.4f}]")
    print(f"   Train target mean : {train['target'].mean():.6f} | std: {train['target'].std():.4f}")
    print(f"   Test  target mean : {test['target'].mean():.6f} | std: {test['target'].std():.4f}")
    print(f"   Train regime dist :")
    for r, c in train.groupby('Date')['regime_hmm'].first().value_counts().items():
        print(f"     {r:10s}: {c} days")
    print(f"   Test regime dist  :")
    for r, c in test.groupby('Date')['regime_hmm'].first().value_counts().items():
        print(f"     {r:10s}: {c} days")

    print("\n>>> READY FOR BASELINE EXPERIMENTS <<<")


if __name__ == "__main__":
    main()
