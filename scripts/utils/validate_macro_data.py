"""
Validation script để kiểm tra các vấn đề tiềm ẩn trong macro preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/macro")
OUTPUT_DIR = Path("data/processed/macro")

print("=" * 80)
print("VALIDATION: MACRO DATA QUALITY")
print("=" * 80)

# ============================================================================
# 1. DATA AVAILABILITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("1. DATA AVAILABILITY CHECK")
print("=" * 80)

df_vnindex = pd.read_csv(DATA_DIR / "vnindex_2013_2026.csv", parse_dates=["time"])
df_vnm = pd.read_csv(DATA_DIR / "vnm_etf.csv", parse_dates=["Date"])
df_vix = pd.read_csv(DATA_DIR / "vix_daily.csv", parse_dates=["date"])
df_hsi = pd.read_csv(DATA_DIR / "hsi_daily.csv", parse_dates=["date"])
df_vn30 = pd.read_csv(DATA_DIR / "vn30_2013_2026.csv", parse_dates=["time"])
df_ir = pd.read_csv(DATA_DIR / "interest_rate_processed.csv", parse_dates=["month_date"])
df_pmi = pd.read_csv(DATA_DIR / "PMI_Vietnam_processed.csv", parse_dates=["month_date"])

df_foreign = pd.read_csv(DATA_DIR / "foreign.csv", encoding="utf-8")
df_foreign = df_foreign.rename(columns={"Ngày": "date"})
df_foreign["date"] = pd.to_datetime(df_foreign["date"], format="%d/%m/%Y", errors="coerce")

data_availability = {
    "VNINDEX": (df_vnindex["time"].min(), df_vnindex["time"].max()),
    "VNM_ETF": (df_vnm["Date"].min(), df_vnm["Date"].max()),
    "VIX": (df_vix["date"].min(), df_vix["date"].max()),
    "HSI": (df_hsi["date"].min(), df_hsi["date"].max()),
    "VN30": (df_vn30["time"].min(), df_vn30["time"].max()),
    "Interest_Rate": (df_ir["month_date"].min(), df_ir["month_date"].max()),
    "PMI": (df_pmi["month_date"].min(), df_pmi["month_date"].max()),
    "Foreign": (df_foreign["date"].min(), df_foreign["date"].max()),
}

for name, (start, end) in data_availability.items():
    print(f"{name:20s}: {start} to {end}")

# ============================================================================
# 2. FOREIGN DATA RANGE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. FOREIGN DATA RANGE - DETAIL CHECK")
print("=" * 80)

df_foreign_valid = df_foreign.dropna(subset=["date"])
print(f"\nTotal rows in foreign.csv: {len(df_foreign)}")
print(f"Valid rows (with date): {len(df_foreign_valid)}")
print(f"Missing dates: {df_foreign['date'].isna().sum()}")

print(f"\nDate range:")
print(f"  Earliest: {df_foreign_valid['date'].min()}")
print(f"  Latest: {df_foreign_valid['date'].max()}")

print(f"\nComparison with VNINDEX:")
print(f"  VNINDEX starts: {df_vnindex['time'].min()}")
print(f"  Foreign starts: {df_foreign_valid['date'].min()}")
print(f"  Difference: {df_foreign_valid['date'].min() - df_vnindex['time'].min()}")

# ============================================================================
# 3. VOLUME RATIO EDGE CASES
# ============================================================================
print("\n" + "=" * 80)
print("3. VOLUME RATIO EDGE CASES")
print("=" * 80)

# Check if rolling mean could be zero
print("\nVNINDEX volume check:")
print(f"  Min volume: {df_vnindex['volume'].min()}")
print(f"  Max volume: {df_vnindex['volume'].max()}")
print(f"  Zero volume rows: {(df_vnindex['volume'] == 0).sum()}")
print(f"  NaN volume rows: {df_vnindex['volume'].isna().sum()}")

print("\nVN30 volume check:")
print(f"  Min volume: {df_vn30['volume'].min()}")
print(f"  Max volume: {df_vn30['volume'].max()}")
print(f"  Zero volume rows: {(df_vn30['volume'] == 0).sum()}")
print(f"  NaN volume rows: {df_vn30['volume'].isna().sum()}")

# Simulate volume ratio calculation
print("\n\nSimulating volume ratio calculation:")
df_vnindex_sorted = df_vnindex.sort_values("time").reset_index(drop=True)
df_vnindex_sorted["volume_ma20"] = df_vnindex_sorted["volume"].rolling(20).mean()

# Check for potential issues
problematic_rows = df_vnindex_sorted[
    (df_vnindex_sorted["volume_ma20"] < 1e-6) | (df_vnindex_sorted["volume_ma20"].isna())
]

print(f"  Rows with rolling MA20 < 1e-6: {len(problematic_rows)}")
print(f"  Rows with rolling MA20 = NaN: {df_vnindex_sorted['volume_ma20'].isna().sum()}")

# Calculate ratio
df_vnindex_sorted["volume_ratio"] = (
    df_vnindex_sorted["volume"] / (df_vnindex_sorted["volume_ma20"] + 1e-6)
)

print(f"\n  Volume ratio stats:")
print(f"    Min: {df_vnindex_sorted['volume_ratio'].min():.6f}")
print(f"    Max: {df_vnindex_sorted['volume_ratio'].max():.6f}")
print(f"    Mean: {df_vnindex_sorted['volume_ratio'].mean():.6f}")
print(f"    Inf values: {np.isinf(df_vnindex_sorted['volume_ratio']).sum()}")
print(f"    NaN values: {df_vnindex_sorted['volume_ratio'].isna().sum()}")

# ============================================================================
# 4. PMI LAG HANDLING
# ============================================================================
print("\n" + "=" * 80)
print("4. PMI LAG HANDLING - COLUMN CHECK")
print("=" * 80)

print(f"\nPMI columns: {list(df_pmi.columns)}")
print(f"Has pmi_lag1 in file: {'pmi_lag1' in df_pmi.columns}")

if "pmi_lag1" in df_pmi.columns:
    print(f"\n[OK] PMI lag1 already exists in file!")
    print(f"  First few rows:")
    print(df_pmi[["month_date", "pmi", "pmi_lag1"]].head(10))

# ============================================================================
# 5. PROCESSED DATA VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("5. PROCESSED DATA QUALITY CHECK")
print("=" * 80)

if (OUTPUT_DIR / "macro_processed_scaled.csv").exists():
    df_processed = pd.read_csv(OUTPUT_DIR / "macro_processed_scaled.csv")
    print(f"\n[OK] Processed data found: {df_processed.shape}")

    # Check inf values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df_processed[numeric_cols]).sum().sum()
    nan_count = df_processed[numeric_cols].isna().sum().sum()

    print(f"  Inf values: {inf_count}")
    print(f"  NaN values: {nan_count}")

    if inf_count > 0:
        print(f"\n  [WARNING] Found Inf values!")
        for col in numeric_cols:
            inf_in_col = np.isinf(df_processed[col]).sum()
            if inf_in_col > 0:
                print(f"    {col}: {inf_in_col} inf values")
    else:
        print(f"\n  [OK] No Inf values found!")

    # Volume ratio details
    if "vnindex_volume_ratio_20d" in df_processed.columns:
        vol_ratio = df_processed["vnindex_volume_ratio_20d"]
        print(f"\n  Volume ratio stats:")
        print(f"    Count: {vol_ratio.count()}")
        print(f"    Min: {vol_ratio.min():.6f}")
        print(f"    Max: {vol_ratio.max():.6f}")
        print(f"    Mean: {vol_ratio.mean():.6f}")
        print(f"    Std: {vol_ratio.std():.6f}")
else:
    print(f"\n[ERROR] Processed data not found in {OUTPUT_DIR}")

# ============================================================================
# 6. FOREIGN DATA IMPACT ON FILL
# ============================================================================
print("\n" + "=" * 80)
print("6. FOREIGN DATA FILL IMPACT")
print("=" * 80)

print(f"\nVNINDEX data starts: {df_vnindex['time'].min()}")
print(f"Foreign data starts: {df_foreign_valid['date'].min()}")

gap_days = (df_foreign_valid['date'].min() - df_vnindex['time'].min()).days
print(f"\nGap (VNINDEX before Foreign): {gap_days} days")

if gap_days > 0:
    print(f"\n[WARNING] Foreign data starts {gap_days} days AFTER VNINDEX!")
    print(f"   These {gap_days} days will be filled with 0 --> potential bias")
    print(f"   This represents {(gap_days / 252):.1f} trading years of artificial 0s")
    print(f"\n   [NOTE] SOLUTION IN CODE:")
    print(f"      - Code fills NaN with 0 (line 318)")
    print(f"      - This is reasonable if foreign starts late")
    print(f"      - But should document this assumption")
else:
    print(f"\n[OK] Foreign data covers from start - no fill issue")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
