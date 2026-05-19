"""
Create panel data combining stock OHLCV with macro features.
Output: Date | Ticker | Stock Features | Macro Features
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

# Configuration
STOCK_DIR = "data/raw/selected"
MACRO_SCALED_FILE = "data/processed/macro/macro_processed_scaled_v2.csv"
SCALER_FILE = "data/processed/macro/scaler_macro.pkl"
OUTPUT_DIR = "data/processed/panel"
START_DATE = "2013-03-01"  # Align with macro data start

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1] Loading macro data...")
macro_df = pd.read_csv(MACRO_SCALED_FILE)
macro_df['date'] = pd.to_datetime(macro_df['date'])
print(f"    Macro shape: {macro_df.shape}, dates: {macro_df['date'].min()} to {macro_df['date'].max()}")

# Get macro features (exclude date)
macro_features = [col for col in macro_df.columns if col != 'date']
print(f"    Macro features ({len(macro_features)}): {macro_features[:5]}... (and {len(macro_features)-5} more)")

print("\n[2] Loading stock OHLCV data...")
stock_files = sorted(Path(STOCK_DIR).glob("*.csv"))
print(f"    Found {len(stock_files)} stock files")

# Dictionary to hold stock data
stock_data = {}
for file in stock_files:
    ticker = file.stem.upper()
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    stock_data[ticker] = df
    print(f"    {ticker}: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\n[3] Engineering stock-level features...")

def engineer_stock_features(stock_df, ticker):
    """Engineer features for a single stock"""
    df = stock_df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # 1. Returns
    df['close_ret_1d'] = df['Close'].pct_change()
    df['log_ret_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['close_ret_5d'] = df['Close'].pct_change(5)
    df['close_ret_20d'] = df['Close'].pct_change(20)

    # 2. Volatility (rolling std of log returns)
    df['vol_5d'] = df['log_ret_1d'].rolling(5).std()
    df['vol_10d'] = df['log_ret_1d'].rolling(10).std()
    df['vol_20d'] = df['log_ret_1d'].rolling(20).std()
    df['vol_60d'] = df['log_ret_1d'].rolling(60).std()

    # 3. Volume features
    df['volume_log'] = np.log(df['Volume'] + 1)
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio_20d'] = df['Volume'] / (df['volume_ma20'] + 1e-6)

    # 4. Trading value (Close * Volume)
    df['trading_value'] = df['Close'] * df['Volume']
    df['trading_value_log'] = np.log(df['trading_value'] + 1)
    df['trading_value_ma20'] = df['trading_value'].rolling(20).mean()

    # 5. Liquidity (turnover like)
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_ret'] = (df['Close'] - df['Open']) / df['Open']

    # 6. Price position
    df['close_pos'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])  # 0-1

    # Add ticker column
    df['ticker'] = ticker

    return df

# Engineer features for all stocks
all_stock_features = []
for ticker, df in stock_data.items():
    engineered = engineer_stock_features(df, ticker)
    all_stock_features.append(engineered)
    print(f"    {ticker}: engineered {len([c for c in engineered.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']])} features")

stock_combined = pd.concat(all_stock_features, ignore_index=True)
print(f"\n    Combined stock data: {stock_combined.shape}")

print("\n[4] Merging stock data with macro data...")
# Rename Date column to match macro
stock_combined['date'] = stock_combined['Date']
stock_combined = stock_combined.drop('Date', axis=1)

# Left join: keep all stock dates, merge macro where available
panel_data = stock_combined.merge(macro_df, on='date', how='left')

print(f"    Panel shape after merge: {panel_data.shape}")
print(f"    Missing macro values: {panel_data[macro_features].isna().sum().sum()}")

# Drop rows where macro features are completely missing (before macro data starts)
panel_data = panel_data.dropna(subset=macro_features)
print(f"    Panel shape after dropping NaN macro: {panel_data.shape}")

# Filter to START_DATE
panel_data = panel_data[panel_data['date'] >= START_DATE].reset_index(drop=True)
print(f"    Panel shape after date filter ({START_DATE}): {panel_data.shape}")

print("\n[5] Feature summary...")
stock_feature_cols = [col for col in panel_data.columns
                      if col not in ['Date', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'trading_value']
                      and col not in macro_features]
print(f"    Stock-level features ({len(stock_feature_cols)}): {stock_feature_cols}")
print(f"    Macro features ({len(macro_features)}): {macro_features}")
print(f"    Total features: {len(stock_feature_cols) + len(macro_features)}")

print("\n[6] Data quality checks...")
print(f"    Date range: {panel_data['date'].min().date()} to {panel_data['date'].max().date()}")
print(f"    Tickers: {sorted(panel_data['ticker'].unique())}")
print(f"    Rows per ticker:")
for ticker in sorted(panel_data['ticker'].unique()):
    count = (panel_data['ticker'] == ticker).sum()
    print(f"      {ticker}: {count}")

# Check for NaN in features
print(f"\n    NaN in stock features: {panel_data[stock_feature_cols].isna().sum().sum()}")
print(f"    NaN in macro features: {panel_data[macro_features].isna().sum().sum()}")

# Fill remaining NaNs (rolling window edge cases)
print("\n[7] Handling NaN values...")
before_fill = panel_data.isna().sum().sum()
panel_data = panel_data.bfill().ffill()
after_fill = panel_data.isna().sum().sum()
print(f"    NaN before fill: {before_fill}, after fill: {after_fill}")

print("\n[8] Saving panel data...")
output_file = os.path.join(OUTPUT_DIR, "panel_stock_macro.csv")
panel_data.to_csv(output_file, index=False)
print(f"    Saved: {output_file}")
print(f"    Shape: {panel_data.shape}")

# Summary statistics
print("\n[9] Summary statistics (first 100 rows):")
print(f"    Columns: {len(panel_data.columns)}")
print(f"    Rows: {len(panel_data)}")
print(f"\n    Column list:")
for i, col in enumerate(panel_data.columns):
    print(f"      {i+1}. {col}")

print("\n[10] Sample rows:")
print(panel_data[['date', 'ticker', 'close_ret_1d', 'vol_20d', 'volume_change', 'vnindex_logret', 'foreign_net_value']].head(10))

print("\n[OK] Panel data creation completed!")
print(f"\nOutput file: {output_file}")
print(f"Ready for feature selection and model training.")
