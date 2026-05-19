"""
Create target variables and train/val/test split from panel data.
Targets: return_1d, return_5d, direction_1d (optional)
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/processed/panel/panel_stock_macro.csv"
OUTPUT_DIR = "data/processed/panel"
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"
TEST_END = "2025-12-31"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1] Loading panel data...")
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
print(f"    Loaded: {df.shape}")
print(f"    Columns: {len(df.columns)}")

print("\n[2] Creating target variables...")

# Sort by ticker and date to ensure proper shift
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# For each ticker group, create targets (shift backwards to avoid look-ahead bias)
targets_data = []
for ticker in df['ticker'].unique():
    ticker_mask = df['ticker'] == ticker
    ticker_df = df[ticker_mask].copy()

    # Get indices for this ticker
    indices = np.where(ticker_mask)[0]

    # Create target columns (need to shift for future value)
    y_return_1d = np.full(len(ticker_df), np.nan)
    y_return_5d = np.full(len(ticker_df), np.nan)
    y_direction_1d = np.full(len(ticker_df), np.nan, dtype='int')

    for i in range(len(ticker_df) - 1):
        # 1-day ahead return
        if i < len(ticker_df) - 1:
            y_return_1d[i] = ticker_df.iloc[i+1]['close_ret_1d']
            y_direction_1d[i] = 1 if y_return_1d[i] > 0 else 0

    for i in range(len(ticker_df) - 5):
        # 5-day ahead return (Close[t+5] / Close[t] - 1)
        close_t = ticker_df.iloc[i]['Close']
        close_t5 = ticker_df.iloc[i+5]['Close']
        if close_t > 0:
            y_return_5d[i] = (close_t5 / close_t) - 1

    # Store targets
    targets_data.append({
        'ticker': ticker,
        'indices': indices,
        'y_return_1d': y_return_1d,
        'y_return_5d': y_return_5d,
        'y_direction_1d': y_direction_1d
    })

# Add targets to dataframe
df['y_return_1d'] = np.nan
df['y_return_5d'] = np.nan
df['y_direction_1d'] = np.nan

for target_group in targets_data:
    indices = target_group['indices']
    df.loc[indices, 'y_return_1d'] = target_group['y_return_1d']
    df.loc[indices, 'y_return_5d'] = target_group['y_return_5d']
    df.loc[indices, 'y_direction_1d'] = target_group['y_direction_1d']

print(f"    y_return_1d: {df['y_return_1d'].notna().sum()} valid values")
print(f"    y_return_5d: {df['y_return_5d'].notna().sum()} valid values")
print(f"    y_direction_1d: {df['y_direction_1d'].notna().sum()} valid values")

# Remove rows without targets
df_with_targets = df[df['y_return_1d'].notna()].copy()
print(f"\n    After removing NaN targets: {df_with_targets.shape}")

print("\n[3] Creating train/val/test split...")
print(f"    Train: before {TRAIN_END}")
print(f"    Val: {TRAIN_END} to {VAL_END}")
print(f"    Test: {VAL_END} to {TEST_END}")

# Temporal split
train_df = df_with_targets[df_with_targets['date'] <= TRAIN_END].copy()
val_df = df_with_targets[(df_with_targets['date'] > TRAIN_END) & (df_with_targets['date'] <= VAL_END)].copy()
test_df = df_with_targets[(df_with_targets['date'] > VAL_END) & (df_with_targets['date'] <= TEST_END)].copy()

print(f"\n    Train: {train_df.shape[0]:,} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"    Val: {val_df.shape[0]:,} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
print(f"    Test: {test_df.shape[0]:,} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

total = len(df_with_targets)
print(f"\n    Percentages:")
print(f"    Train: {len(train_df)/total*100:.1f}%")
print(f"    Val: {len(val_df)/total*100:.1f}%")
print(f"    Test: {len(test_df)/total*100:.1f}%")

# Per-ticker breakdown
print(f"\n    Per-ticker breakdown (train set):")
for ticker in sorted(train_df['ticker'].unique()):
    count = (train_df['ticker'] == ticker).sum()
    print(f"      {ticker}: {count}")

print("\n[4] Target variable statistics...")
print(f"\n    y_return_1d (train):")
print(f"      Mean: {train_df['y_return_1d'].mean():.6f}")
print(f"      Std: {train_df['y_return_1d'].std():.6f}")
print(f"      Min: {train_df['y_return_1d'].min():.6f}")
print(f"      Max: {train_df['y_return_1d'].max():.6f}")
print(f"      % positive: {(train_df['y_return_1d'] > 0).sum() / len(train_df) * 100:.1f}%")

print(f"\n    y_return_5d (train):")
print(f"      Mean: {train_df['y_return_5d'].mean():.6f}")
print(f"      Std: {train_df['y_return_5d'].std():.6f}")
print(f"      Min: {train_df['y_return_5d'].min():.6f}")
print(f"      Max: {train_df['y_return_5d'].max():.6f}")
print(f"      % positive: {(train_df['y_return_5d'] > 0).sum() / len(train_df) * 100:.1f}%")

print("\n[5] Saving splits...")
train_file = os.path.join(OUTPUT_DIR, "panel_train.csv")
val_file = os.path.join(OUTPUT_DIR, "panel_val.csv")
test_file = os.path.join(OUTPUT_DIR, "panel_test.csv")
full_file = os.path.join(OUTPUT_DIR, "panel_with_targets.csv")

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)
df_with_targets.to_csv(full_file, index=False)

print(f"    Train: {train_file}")
print(f"    Val: {val_file}")
print(f"    Test: {test_file}")
print(f"    Full: {full_file}")

print("\n[6] Feature list for reference...")
feature_cols = [col for col in df.columns
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'date',
                               'trading_value', 'y_return_1d', 'y_return_5d', 'y_direction_1d']]
print(f"\n    Stock-level features (17):")
stock_features = [col for col in feature_cols if col in ['close_ret_1d', 'log_ret_1d', 'close_ret_5d', 'close_ret_20d',
                                                          'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d', 'volume_log',
                                                          'volume_change', 'volume_ma20', 'volume_ratio_20d',
                                                          'trading_value_log', 'trading_value_ma20', 'high_low_range',
                                                          'open_close_ret', 'close_pos']]
for i, col in enumerate(stock_features, 1):
    print(f"      {i:2d}. {col}")

print(f"\n    Macro features (26 - scaled):")
macro_features = [col for col in feature_cols if col not in stock_features]
for i, col in enumerate(macro_features, 1):
    print(f"      {i:2d}. {col}")

print("\n[OK] Target variables and splits created successfully!")
print(f"\nReady for feature selection and baseline training.")
