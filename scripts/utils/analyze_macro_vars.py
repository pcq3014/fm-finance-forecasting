import pandas as pd
import numpy as np
from pathlib import Path

# Load all macro files
data_dir = Path("data/macro")
files = {
    "vnm_etf": data_dir / "vnm_etf.csv",
    "vix": data_dir / "vix_daily.csv",
    "interest_rate": data_dir / "interest_rate_processed.csv",
    "pmi": data_dir / "PMI_Vietnam_processed.csv",
    "hsi": data_dir / "hsi_daily.csv",
    "vnindex": data_dir / "vnindex_2013_2026.csv",
    "vn30": data_dir / "vn30_2013_2026.csv",
}

print("=" * 80)
print("MACRO VARIABLES ANALYSIS")
print("=" * 80)

for name, path in files.items():
    print(f"\n{'=' * 80}")
    print(f"{name.upper()}")
    print(f"{'=' * 80}")

    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)

    # Get numeric columns for stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nDescriptive Statistics:")
        print(df[numeric_cols].describe())
        print(f"\nMissing values:")
        print(df[numeric_cols].isnull().sum())

    print("\n")
