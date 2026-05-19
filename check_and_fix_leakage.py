"""
Fix data leakage in target variables.

Problem: y_return_1d[i] = close_ret_1d[i+1] (future information!)
Solution: Target should be future close price, not future return
"""

import pandas as pd
import numpy as np
import os

print("[1] Analyzing Leakage Issue")
print("=" * 80)

# Load one stock to check
df = pd.read_csv('data/processed/panel/panel_train.csv')
vcb = df[df['ticker'] == 'VCB'].head(20).copy()
vcb['date'] = pd.to_datetime(vcb['date'])

print("\nSample data (VCB):\n")
print(vcb[['date', 'Close', 'close_ret_1d', 'y_return_1d']].to_string())

print("\n\n[LEAKAGE ANALYSIS]")
print("-" * 80)

# Check the relationship
print("\nRelationship check:")
print("  Row i:")
print(f"    Close[i]: {vcb.iloc[0]['Close']:.2f}")
print(f"    close_ret_1d[i]: {vcb.iloc[0]['close_ret_1d']:.6f}")
print(f"    y_return_1d[i]: {vcb.iloc[0]['y_return_1d']:.6f}")

print("\n  Row i+1:")
print(f"    Close[i+1]: {vcb.iloc[1]['Close']:.2f}")
print(f"    close_ret_1d[i+1]: {vcb.iloc[1]['close_ret_1d']:.6f}")
print(f"    y_return_1d[i]: {vcb.iloc[0]['y_return_1d']:.6f}")

# Calculate what y_return_1d should be
actual_ret = (vcb.iloc[1]['Close'] / vcb.iloc[0]['Close']) - 1
print(f"\n  CORRECT y_return_1d[i] should be:")
print(f"    = Close[i+1] / Close[i] - 1")
print(f"    = {vcb.iloc[1]['Close']:.2f} / {vcb.iloc[0]['Close']:.2f} - 1")
print(f"    = {actual_ret:.6f}")

print(f"\n  ACTUAL y_return_1d[i] is:")
print(f"    = {vcb.iloc[0]['y_return_1d']:.6f}")

print(f"\n  Does it match? {np.isclose(actual_ret, vcb.iloc[1]['close_ret_1d'])}")

print("\n" + "=" * 80)
print("\nCONCLUSION:")
print("  LEAKAGE FOUND: y_return_1d[i] = close_ret_1d[i+1]")
print("  This means target uses FUTURE feature values!")
print("  Problem: At day t, we know return that happened on day t+1")
print("  This is look-ahead bias!\n")

print("[2] FIXING THE ISSUE")
print("=" * 80)

# Reload full data
print("\nLoading panel data...")
df = pd.read_csv('data/processed/panel/panel_stock_macro.csv')
print(f"Shape before fix: {df.shape}")

# Sort properly
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# Create target properly: shift Close forward, then calculate return
print("\nCreating correct targets...")
df['y_return_1d'] = np.nan
df['y_return_5d'] = np.nan
df['y_direction_1d'] = np.nan

for ticker in df['ticker'].unique():
    mask = df['ticker'] == ticker
    indices = np.where(mask)[0]

    # Get close prices for this ticker
    close_prices = df.loc[indices, 'Close'].values

    # Calculate target: future return
    y_ret_1d = np.full(len(close_prices), np.nan)
    y_ret_5d = np.full(len(close_prices), np.nan)

    # 1-day ahead: log(Close[t+1] / Close[t])
    for i in range(len(close_prices) - 1):
        if close_prices[i] > 0:
            y_ret_1d[i] = np.log(close_prices[i+1] / close_prices[i])

    # 5-day ahead: log(Close[t+5] / Close[t])
    for i in range(len(close_prices) - 5):
        if close_prices[i] > 0:
            y_ret_5d[i] = np.log(close_prices[i+5] / close_prices[i])

    # Direction
    y_dir = np.where(y_ret_1d > 0, 1, 0)

    # Assign back
    df.loc[indices, 'y_return_1d'] = y_ret_1d
    df.loc[indices, 'y_return_5d'] = y_ret_5d
    df.loc[indices, 'y_direction_1d'] = y_dir

print(f"  y_return_1d valid: {df['y_return_1d'].notna().sum()}")
print(f"  y_return_5d valid: {df['y_return_5d'].notna().sum()}")

# Remove NaN targets
df_clean = df[df['y_return_1d'].notna()].copy()
print(f"\nShape after removing NaN targets: {df_clean.shape}")

# Align with macro data date range
START_DATE = "2013-03-01"
df_clean = df_clean[df_clean['date'] >= START_DATE].reset_index(drop=True)
print(f"Shape after date filter: {df_clean.shape}")

# Verify no leakage
print("\n[3] VERIFICATION - No Leakage Check")
print("-" * 80)

vcb_fixed = df_clean[df_clean['ticker'] == 'VCB'].head(10).copy()
print("\nFixed VCB sample:\n")
print(vcb_fixed[['date', 'Close', 'close_ret_1d', 'y_return_1d']].to_string())

# Manual check
print("\n\nManual verification:")
for i in range(3):
    row = vcb_fixed.iloc[i]
    next_row = vcb_fixed.iloc[i+1]

    manual_ret = (next_row['Close'] / row['Close']) - 1
    stored_ret = row['y_return_1d']

    match = np.isclose(manual_ret, stored_ret)
    status = "OK" if match else "ERROR"
    print(f"  Row {i}: manual={manual_ret:.6f}, stored={stored_ret:.6f} [{status}]")

# Check correlation to detect remaining leakage
corr_close_ret = df_clean['close_ret_1d'].corr(df_clean['y_return_1d'])
print(f"\nCorrelation between close_ret_1d[t] and y_return_1d[t]: {corr_close_ret:.6f}")
print(f"(Should be near 0, not 0.95+)")

if corr_close_ret > 0.5:
    print("WARNING: Still potential leakage!")
else:
    print("GOOD: No obvious leakage detected")

# Create train/val/test splits
print("\n[4] Creating Train/Val/Test Splits")
print("-" * 80)

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"

train_df = df_clean[df_clean['date'] <= TRAIN_END].copy()
val_df = df_clean[(df_clean['date'] > TRAIN_END) & (df_clean['date'] <= VAL_END)].copy()
test_df = df_clean[(df_clean['date'] > VAL_END)].copy()

print(f"Train: {len(train_df):,} rows ({len(train_df)/len(df_clean)*100:.1f}%)")
print(f"Val:   {len(val_df):,} rows ({len(val_df)/len(df_clean)*100:.1f}%)")
print(f"Test:  {len(test_df):,} rows ({len(test_df)/len(df_clean)*100:.1f}%)")

print(f"\nTrain targets - y_return_1d:")
print(f"  Mean: {train_df['y_return_1d'].mean():.6f}")
print(f"  Std:  {train_df['y_return_1d'].std():.6f}")
print(f"  % positive: {(train_df['y_return_1d'] > 0).sum() / len(train_df) * 100:.1f}%")

# Save fixed data
print("\n[5] Saving Fixed Data")
print("-" * 80)

OUTPUT_DIR = "data/processed/panel"
train_file = f"{OUTPUT_DIR}/panel_train_fixed.csv"
val_file = f"{OUTPUT_DIR}/panel_val_fixed.csv"
test_file = f"{OUTPUT_DIR}/panel_test_fixed.csv"
full_file = f"{OUTPUT_DIR}/panel_with_targets_fixed.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)
df_clean.to_csv(full_file, index=False)

print(f"Saved: {train_file}")
print(f"Saved: {val_file}")
print(f"Saved: {test_file}")
print(f"Saved: {full_file}")

print("\n[OK] Leakage fixed! Ready for re-running feature selection.")
