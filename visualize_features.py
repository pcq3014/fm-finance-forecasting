"""
Comprehensive feature visualization and exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
TRAIN_FILE = "data/processed/panel/panel_train.csv"
OUTPUT_DIR = "outputs/feature_visualization"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1] Loading data...")
df_train = pd.read_csv(TRAIN_FILE)
print(f"    Shape: {df_train.shape}")

# Feature groups
stock_features = ['close_ret_1d', 'log_ret_1d', 'close_ret_5d', 'close_ret_20d',
                  'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d', 'volume_log',
                  'volume_change', 'volume_ma20', 'volume_ratio_20d',
                  'trading_value_log', 'trading_value_ma20', 'high_low_range',
                  'open_close_ret', 'close_pos']

macro_features_all = ['vnindex_close', 'vnindex_volume', 'vn30_close', 'vn30_volume',
                  'vnm_ret', 'vnm_close', 'vix', 'hsi_close', 'hsi_logret',
                  'foreign_net_volume', 'foreign_net_value', 'foreign_buy_volume',
                  'foreign_buy_value', 'foreign_sell_volume', 'foreign_sell_value',
                  'month_date', 'interest_rate', 'interest_rate_lag1', 'ir_change',
                  'pmi', 'pmi_change', 'vnindex_logret', 'vn30_logret',
                  'vnindex_vol_5d', 'vn30_vol_5d', 'vix_lag1', 'vix_lag5',
                  'vnm_ret_lag1', 'hsi_logret_lag1', 'vix_log_lag1',
                  'vnindex_volume_log', 'vn30_volume_log', 'vnindex_volume_ratio_20d',
                  'vn30_volume_ratio_20d', 'foreign_net_ratio', 'foreign_buy_ratio',
                  'foreign_net_value_ma5', 'foreign_net_volume_ma5',
                  'foreign_net_value_lag1', 'foreign_net_volume_lag1']

# Filter to numeric only (remove month_date which is string)
macro_features = [f for f in macro_features_all if f in df_train.columns and pd.api.types.is_numeric_dtype(df_train[f])]

all_features = stock_features + macro_features

print(f"\n[2] Feature Summary Statistics...")
print(f"\n{'='*80}")
print(f"STOCK-LEVEL FEATURES ({len(stock_features)})")
print(f"{'='*80}")
summary_stock = df_train[stock_features].describe().T
summary_stock['skew'] = df_train[stock_features].skew()
summary_stock['kurtosis'] = df_train[stock_features].kurtosis()
print(summary_stock[['mean', 'std', 'min', 'max', 'skew']])

print(f"\n{'='*80}")
print(f"MACRO FEATURES ({len(macro_features)})")
print(f"{'='*80}")
summary_macro = df_train[macro_features].describe().T
summary_macro['skew'] = df_train[macro_features].skew()
print(summary_macro[['mean', 'std', 'min', 'max', 'skew']])

print(f"\n{'='*80}")
print(f"TARGET VARIABLES")
print(f"{'='*80}")
targets = df_train[['y_return_1d', 'y_return_5d']].describe().T
targets['skew'] = df_train[['y_return_1d', 'y_return_5d']].skew()
print(targets)

print("\n[3] Creating visualizations...")

# 1. Distribution plots - Stock Features
fig, axes = plt.subplots(6, 3, figsize=(18, 14))
axes = axes.flatten()
for idx, feat in enumerate(stock_features):
    data = df_train[feat].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 0:
        axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{feat}', fontsize=9, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_stock_features_distribution.png", dpi=150, bbox_inches='tight')
print("    [OK] 01_stock_features_distribution.png")
plt.close()

# 2. Distribution plots - Macro Features (top important ones)
fig, axes = plt.subplots(7, 6, figsize=(20, 16))
axes = axes.flatten()
for idx, feat in enumerate(macro_features):
    data = df_train[feat].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 0:
        axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[idx].set_title(f'{feat}', fontsize=8, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_macro_features_distribution.png", dpi=150, bbox_inches='tight')
print("    [OK] 02_macro_features_distribution.png")
plt.close()

# 3. Box plots - Stock Features
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
feature_groups = {
    'Returns': ['close_ret_1d', 'close_ret_5d', 'close_ret_20d', 'log_ret_1d'],
    'Volatility': ['vol_5d', 'vol_10d', 'vol_20d', 'vol_60d'],
    'Volume': ['volume_log', 'volume_change', 'volume_ma20', 'volume_ratio_20d'],
    'Trading Value': ['trading_value_log', 'trading_value_ma20'],
    'Price Features': ['high_low_range', 'open_close_ret', 'close_pos'],
}

for idx, (group_name, features) in enumerate(feature_groups.items()):
    if idx < len(axes):
        data_to_plot = [df_train[f].dropna() for f in features if f in df_train.columns]
        axes[idx].boxplot(data_to_plot, labels=[f.replace('_', '\n') for f in features if f in df_train.columns])
        axes[idx].set_title(f'{group_name}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_stock_features_boxplot.png", dpi=150, bbox_inches='tight')
print("    [OK] 03_stock_features_boxplot.png")
plt.close()

# 4. Correlation heatmap - Stock Features
fig, ax = plt.subplots(figsize=(12, 10))
stock_clean = df_train[stock_features].replace([np.inf, -np.inf], np.nan)
corr_stock = stock_clean.corr()
sns.heatmap(corr_stock, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Stock Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_stock_features_correlation.png", dpi=150, bbox_inches='tight')
print("    [OK] 04_stock_features_correlation.png")
plt.close()

# 5. Correlation heatmap - Macro Features
fig, ax = plt.subplots(figsize=(14, 12))
macro_clean = df_train[macro_features].replace([np.inf, -np.inf], np.nan)
corr_macro = macro_clean.corr()
sns.heatmap(corr_macro, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Macro Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_macro_features_correlation.png", dpi=150, bbox_inches='tight')
print("    [OK] 05_macro_features_correlation.png")
plt.close()

# 6. Target variables distribution and correlation with top features
print("\n[4] Computing feature importance...")

# Use Random Forest to get feature importance for y_return_1d
# Remove features with too many inf values
all_features_clean = [f for f in all_features if not df_train[f].isin([np.inf, -np.inf]).sum() > len(df_train) * 0.1]
X = df_train[all_features_clean].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
y = df_train['y_return_1d'].fillna(0)

rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# 7. Feature Importance plot
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 25
top_features = feature_importance.head(top_n)
colors = ['#FF6B6B' if f in stock_features else '#4ECDC4' for f in top_features['feature']]
ax.barh(range(len(top_features)), top_features['importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax.set_title(f'Feature Importance for y_return_1d (Top {top_n})\nRed=Stock, Teal=Macro',
             fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_feature_importance_rf.png", dpi=150, bbox_inches='tight')
print("    [OK] 06_feature_importance_rf.png")
plt.close()

# 8. Correlation with target
print("\n[5] Computing target correlations...")
temp_df = df_train[all_features_clean + ['y_return_1d']].copy()
temp_df = temp_df.replace([np.inf, -np.inf], np.nan).fillna(temp_df.mean())
corr_with_target = temp_df.corr()['y_return_1d'].drop('y_return_1d').sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top_n = 25
top_corr = corr_with_target.head(top_n)
colors_pos = ['#FF6B6B' if x > 0 else '#95E1D3' for x in top_corr.values]
ax.barh(range(len(top_corr)), top_corr.values, color=colors_pos)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index, fontsize=10)
ax.set_xlabel('Correlation with y_return_1d', fontsize=11, fontweight='bold')
ax.set_title(f'Feature Correlation with Target (Top {top_n})\nRed=Positive, Green=Negative',
             fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_target_correlation.png", dpi=150, bbox_inches='tight')
print("    [OK] 07_target_correlation.png")
plt.close()

# 9. Missing values analysis
fig, ax = plt.subplots(figsize=(14, 6))
missing_count = df_train[all_features].isnull().sum().sort_values(ascending=False)
missing_pct = (missing_count / len(df_train) * 100)
colors_missing = ['red' if x > 0.1 else 'green' for x in missing_pct.values]
ax.bar(range(len(missing_count)), missing_pct.values, color=colors_missing)
ax.set_xticks(range(len(missing_count)))
ax.set_xticklabels(missing_count.index, rotation=90, fontsize=9)
ax.set_ylabel('Missing %', fontsize=11, fontweight='bold')
ax.set_title('Missing Values Analysis', fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_missing_values.png", dpi=150, bbox_inches='tight')
print("    [OK] 08_missing_values.png")
plt.close()

# 10. Summary table - Top features
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

summary_table_data = []
for idx, row in feature_importance.head(20).iterrows():
    feat = row['feature']
    imp = row['importance']
    corr = corr_with_target[feat]
    stat = df_train[feat].describe()
    summary_table_data.append([
        feat,
        f"{imp:.4f}",
        f"{corr:.4f}",
        f"{stat['mean']:.4f}",
        f"{stat['std']:.4f}"
    ])

table = ax.table(cellText=summary_table_data,
                colLabels=['Feature', 'Importance', 'Target Corr', 'Mean', 'Std'],
                cellLoc='center',
                loc='center',
                colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_table_data) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Top 20 Features Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig(f"{OUTPUT_DIR}/09_top_features_summary_table.png", dpi=150, bbox_inches='tight')
print("    [OK] 09_top_features_summary_table.png")
plt.close()

# 11. Scatter plots - Top stock vs macro features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top_stock = feature_importance[feature_importance['feature'].isin(stock_features)].head(3)['feature'].values
top_macro = feature_importance[feature_importance['feature'].isin(macro_features)].head(3)['feature'].values

pairs = list(zip(top_stock, top_macro))
for idx, (stock_f, macro_f) in enumerate(pairs):
    axes[idx].scatter(df_train[stock_f], df_train[macro_f], alpha=0.3, s=10)
    axes[idx].set_xlabel(stock_f, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel(macro_f, fontsize=10, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    corr_val = df_train[stock_f].corr(df_train[macro_f])
    axes[idx].set_title(f'Correlation: {corr_val:.3f}', fontsize=11)

# Target vs top features
for idx, feat in enumerate(top_stock[1:]):
    if idx + 3 < len(axes):
        axes[idx + 3].scatter(df_train[feat], df_train['y_return_1d'], alpha=0.3, s=10, color='red')
        axes[idx + 3].set_xlabel(feat, fontsize=10, fontweight='bold')
        axes[idx + 3].set_ylabel('y_return_1d', fontsize=10, fontweight='bold')
        axes[idx + 3].grid(True, alpha=0.3)
        corr_val = df_train[feat].corr(df_train['y_return_1d'])
        axes[idx + 3].set_title(f'With Target: {corr_val:.3f}', fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_feature_scatter_analysis.png", dpi=150, bbox_inches='tight')
print("    [OK] 10_feature_scatter_analysis.png")
plt.close()

# 12. Time series trend - sample stock average by date
print("\n[6] Analyzing temporal trends...")
df_train['date'] = pd.to_datetime(df_train['date'])
avg_by_date = df_train.groupby('date')[stock_features + macro_features].mean()

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Select representative features
axes[0].plot(avg_by_date.index, avg_by_date['close_ret_1d'], label='Stock Return 1d', linewidth=1.5, alpha=0.8)
axes[0].plot(avg_by_date.index, avg_by_date['vnindex_logret'], label='VNIndex Return', linewidth=1.5, alpha=0.8)
axes[0].set_ylabel('Return', fontsize=10, fontweight='bold')
axes[0].set_title('Average Returns Over Time', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(avg_by_date.index, avg_by_date['vol_20d'], label='Stock Vol 20d', linewidth=1.5, alpha=0.8)
axes[1].plot(avg_by_date.index, avg_by_date['vnindex_vol_5d'], label='VNIndex Vol 5d', linewidth=1.5, alpha=0.8)
axes[1].set_ylabel('Volatility', fontsize=10, fontweight='bold')
axes[1].set_title('Average Volatility Over Time', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(avg_by_date.index, avg_by_date['volume_change'], label='Volume Change', linewidth=1.5, alpha=0.8)
axes[2].plot(avg_by_date.index, avg_by_date['foreign_net_value'], label='Foreign Net Value', linewidth=1.5, alpha=0.8)
axes[2].set_ylabel('Value', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=10, fontweight='bold')
axes[2].set_title('Average Volume & Foreign Flow Over Time', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_temporal_trends.png", dpi=150, bbox_inches='tight')
print("    [OK] 11_temporal_trends.png")
plt.close()

print("\n[OK] Feature visualization completed!")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"\nGenerated files:")
print(f"  01_stock_features_distribution.png")
print(f"  02_macro_features_distribution.png")
print(f"  03_stock_features_boxplot.png")
print(f"  04_stock_features_correlation.png")
print(f"  05_macro_features_correlation.png")
print(f"  06_feature_importance_rf.png")
print(f"  07_target_correlation.png")
print(f"  08_missing_values.png")
print(f"  09_top_features_summary_table.png")
print(f"  10_feature_scatter_analysis.png")
print(f"  11_temporal_trends.png")

# Save feature importance and correlation tables
feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
corr_with_target.to_csv(f"{OUTPUT_DIR}/target_correlation.csv", header=['correlation'])
print(f"\n  feature_importance.csv")
print(f"  target_correlation.csv")

print(f"\nTop 10 Features by Importance:")
print(feature_importance.head(10))
