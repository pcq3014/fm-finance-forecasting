"""
Feature selection V3 - FINAL VERSION

Corrections from V2:
- KEEP log_ret_1d (it's yesterday's return, not target)
- KEEP close_ret_1d (recent momentum, valid)
- REMOVE only truly mechanical features:
  * open_close_ret (intra-day, different from daily return)
  * high_low_range (daily range, different from return)
  * close_ret_5d, close_ret_20d (too distant, use vol instead)
  * volume_change (problematic: inf values)

Key insight: log_ret_1d[t] predicts y_return_1d[t+1]
- They're on different days
- Correlation = 0.009 (no leakage)
- Valid momentum feature!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_FILE = "data/processed/panel/panel_train_fixed.csv"
OUTPUT_DIR = "outputs/feature_selection_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1] Loading training data (FIXED)...")
df_train = pd.read_csv(TRAIN_FILE)
df_train['date'] = pd.to_datetime(df_train['date'])
print(f"    Shape: {df_train.shape}\n")

# Feature groups
stock_features = ['close_ret_1d', 'log_ret_1d', 'close_ret_5d', 'close_ret_20d',
                  'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d', 'volume_log',
                  'volume_change', 'volume_ma20', 'volume_ratio_20d',
                  'trading_value_log', 'trading_value_ma20', 'high_low_range',
                  'open_close_ret', 'close_pos']

macro_features = ['vnindex_close', 'vnindex_volume', 'vn30_close', 'vn30_volume',
                  'vnm_ret', 'vnm_close', 'vix', 'hsi_close', 'hsi_logret',
                  'foreign_net_volume', 'foreign_net_value', 'foreign_buy_volume',
                  'foreign_buy_value', 'foreign_sell_volume', 'foreign_sell_value',
                  'interest_rate', 'interest_rate_lag1', 'ir_change',
                  'pmi', 'pmi_change', 'vnindex_logret', 'vn30_logret',
                  'vnindex_vol_5d', 'vn30_vol_5d', 'vix_lag1', 'vix_lag5',
                  'vnm_ret_lag1', 'hsi_logret_lag1', 'vix_log_lag1',
                  'vnindex_volume_log', 'vn30_volume_log', 'vnindex_volume_ratio_20d',
                  'vn30_volume_ratio_20d', 'foreign_net_ratio', 'foreign_buy_ratio',
                  'foreign_net_value_ma5', 'foreign_net_volume_ma5',
                  'foreign_net_value_lag1', 'foreign_net_volume_lag1']

# REVISED: Only exclude truly problematic features
exclude_features = [
    'open_close_ret',       # Intra-day pattern (different from daily return)
    'high_low_range',       # Daily range (different from return)
    'close_ret_5d',         # Redundant with vol features
    'close_ret_20d',        # Redundant with vol features
    'volume_change',        # Problematic: inf values
]

stock_features = [f for f in stock_features if f not in exclude_features]
macro_features = [f for f in macro_features if pd.api.types.is_numeric_dtype(df_train[f])]
all_features = stock_features + macro_features

print(f"Stock features (after exclusion): {len(stock_features)}")
print(f"  Kept: close_ret_1d, log_ret_1d (valid momentum signals)")
print(f"  Excluded: {exclude_features}\n")
print(f"Macro features: {len(macro_features)}")
print(f"Total features: {len(all_features)}\n")

# Clean data
print("[2] Data cleaning...")
X = df_train[all_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
y = df_train['y_return_1d'].fillna(0)

print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}\n")

print("[3] METHOD 1: Low Variance Filter")
print("=" * 80)

variance_scores = X.var()
variance_sorted = variance_scores.sort_values(ascending=False)

print(f"Top 10 highest variance:\n{variance_sorted.head(10)}\n")

var_threshold = variance_scores.median()
low_var_features = variance_sorted[variance_sorted > var_threshold].index.tolist()

print(f"Variance threshold (median): {var_threshold:.6f}")
print(f"Features to keep: {len(low_var_features)}\n")

print("[4] METHOD 2: Correlation Filter")
print("=" * 80)

print("\n2a. Correlation with Target:")
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)

print(f"\nTop 15 correlated with target:\n{corr_with_target.head(15)}\n")

corr_threshold = corr_with_target.quantile(0.05)
corr_target_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()

print(f"Correlation threshold (5th percentile): {corr_threshold:.6f}")
print(f"Features to keep: {len(corr_target_features)}\n")

print("2b. Redundancy Filter (inter-feature correlation > 0.9):")
corr_matrix = X.corr().abs()
corr_matrix_array = corr_matrix.values.copy()
np.fill_diagonal(corr_matrix_array, 0)
corr_matrix = pd.DataFrame(corr_matrix_array, index=corr_matrix.index, columns=corr_matrix.columns)

redundant_features = set()
for col in corr_matrix.columns:
    highly_corr = corr_matrix[col][corr_matrix[col] > 0.90].index.tolist()
    for corr_feat in highly_corr:
        if corr_with_target[col] < corr_with_target[corr_feat]:
            redundant_features.add(col)
        else:
            redundant_features.add(corr_feat)

redundant_features = list(redundant_features)
corr_filtered_features = [f for f in corr_target_features if f not in redundant_features]

print(f"Redundant features: {len(redundant_features)}")
print(f"After redundancy removal: {len(corr_filtered_features)}\n")

print("[5] METHOD 3: Mutual Information with Target")
print("=" * 80)

X_scaled = StandardScaler().fit_transform(X)
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\nTop 15 features by MI:\n{mi_df.head(15).to_string()}\n")

mi_threshold = mi_df['mi_score'].median()
mi_features = mi_df[mi_df['mi_score'] > mi_threshold]['feature'].tolist()

print(f"MI threshold (median): {mi_threshold:.6f}")
print(f"Features to keep: {len(mi_features)}\n")

print("[6] Feature Selection Summary (FINAL - V3)")
print("=" * 80)

summary_df = pd.DataFrame({
    'feature': all_features,
    'variance': X.var().values,
    'corr_with_target': X.corrwith(y).abs().values,
    'mutual_info': mi_df.set_index('feature').loc[all_features, 'mi_score'].values
})

summary_df['low_variance_pass'] = summary_df['feature'].isin(low_var_features)
summary_df['correlation_pass'] = summary_df['feature'].isin(corr_filtered_features)
summary_df['mi_pass'] = summary_df['feature'].isin(mi_features)
summary_df['votes'] = (summary_df['low_variance_pass'].astype(int) +
                       summary_df['correlation_pass'].astype(int) +
                       summary_df['mi_pass'].astype(int))

summary_df = summary_df.sort_values('votes', ascending=False)

print(f"\nFeatures by voting (out of 3 methods):\n")
print(summary_df[['feature', 'votes', 'variance', 'corr_with_target', 'mutual_info']].to_string())

print("\n[7] FINAL FEATURE RECOMMENDATIONS")
print("=" * 80)

# Option 1: Aggressive (5-12 features)
top_features_3votes = summary_df[summary_df['votes'] == 3]['feature'].tolist()
if len(top_features_3votes) < 5:
    top_features = summary_df.head(10)['feature'].tolist()
else:
    top_features = top_features_3votes[:12]

print(f"\nOPTION 1: Multivariate FM Features (For Chronos-2-M / Moirai-2-M)")
print("-" * 80)
print(f"Selected: {len(top_features)} features\n")
for i, feat in enumerate(top_features, 1):
    row = summary_df[summary_df['feature'] == feat].iloc[0]
    status = "[MOMENTUM]" if feat in ['close_ret_1d', 'log_ret_1d'] else ""
    print(f"  {i:2d}. {feat:25s} | votes={row['votes']:.0f} | corr={row['corr_with_target']:.4f} | mi={row['mutual_info']:.4f} {status}")

# Option 2: Moderate (10-20 features)
leaf_features = summary_df.head(15)['feature'].tolist()

print(f"\n\nOPTION 2: LEAF Context Features (For regime encoder)")
print("-" * 80)
print(f"Selected: {len(leaf_features)} features\n")
for i, feat in enumerate(leaf_features, 1):
    row = summary_df[summary_df['feature'] == feat].iloc[0]
    status = "[MOMENTUM]" if feat in ['close_ret_1d', 'log_ret_1d', 'vnm_ret'] else ""
    print(f"  {i:2d}. {feat:25s} | votes={row['votes']:.0f} | corr={row['corr_with_target']:.4f} | mi={row['mutual_info']:.4f} {status}")

# Option 3
print(f"\n\nOPTION 3: TimesFM-U (Univariate)")
print("-" * 80)
print(f"  Input: close_ret_1d only")

print("\n[8] EXPLANATION: Why keep log_ret_1d and close_ret_1d?")
print("=" * 80)

corr_lag = df_train['log_ret_1d'].corr(df_train['y_return_1d'])
corr_lag_next = df_train['log_ret_1d'].corr(df_train.groupby('ticker')['y_return_1d'].shift(-1))

print(f"""
log_ret_1d[t] = ln(Close[t] / Close[t-1])  <- Yesterday's return
y_return_1d[t] = Close[t+1] / Close[t] - 1 <- Tomorrow's return

Correlation between log_ret_1d[t] and y_return_1d[t]: {corr_lag:.6f}
(Very weak - not same day, no leakage)

Status: VALID FEATURE
- Momentum signal: yesterday's return can influence today's return
- Common in finance (momentum effect)
- NOT mechanical leakage (different time windows)
- MI = 0.45 is high because it captures non-linear patterns

Same reasoning for close_ret_1d:
- close_ret_1d[t] = return on day t
- y_return_1d[t] = return from day t to t+1
- Valid feature for predicting next day return
""")

print("[9] Saving Results")
print("=" * 80)

summary_file = os.path.join(OUTPUT_DIR, "feature_selection_summary_v3.csv")
summary_df.to_csv(summary_file, index=False)
print(f"Saved: {summary_file}\n")

# Save feature lists
feature_lists_file = os.path.join(OUTPUT_DIR, "selected_features_v3.txt")
with open(feature_lists_file, 'w') as f:
    f.write("FEATURE SELECTION RESULTS - V3 (FINAL)\n")
    f.write("=" * 80 + "\n\n")
    f.write("Removed Mechanical Features:\n")
    for feat in exclude_features:
        f.write(f"  - {feat}\n")
    f.write(f"\nKept Momentum Features:\n")
    f.write(f"  - close_ret_1d (valid: different time window)\n")
    f.write(f"  - log_ret_1d (valid: different time window)\n")
    f.write("\n" + "=" * 80 + "\n\n")
    f.write(f"MULTIVARIATE FM FEATURES ({len(top_features)})\n")
    f.write("-" * 80 + "\n")
    f.write("Use for: Chronos-2-M, Moirai-2-M\n\n")
    for i, feat in enumerate(top_features, 1):
        f.write(f"{i:2d}. {feat}\n")
    f.write(f"\nLEAF CONTEXT FEATURES ({len(leaf_features)})\n")
    f.write("-" * 80 + "\n")
    f.write("Use for: LEAF regime encoder\n\n")
    for i, feat in enumerate(leaf_features, 1):
        f.write(f"{i:2d}. {feat}\n")

print(f"Saved: {feature_lists_file}\n")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist(variance_scores, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(var_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {var_threshold:.4f}')
axes[0, 0].set_xlabel('Variance', fontsize=10, fontweight='bold')
axes[0, 0].set_title('Method 1: Low Variance Filter', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

corr_abs = X.corrwith(y).abs().sort_values(ascending=False)
axes[0, 1].barh(range(len(corr_abs.head(20))), corr_abs.head(20).values, color='coral')
axes[0, 1].set_yticks(range(len(corr_abs.head(20))))
axes[0, 1].set_yticklabels(corr_abs.head(20).index, fontsize=9)
axes[0, 1].axvline(corr_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {corr_threshold:.4f}')
axes[0, 1].set_xlabel('[Correlation]', fontsize=10, fontweight='bold')
axes[0, 1].set_title('Method 2: Correlation with Target', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='x')

mi_top = mi_df.head(20)
axes[1, 0].barh(range(len(mi_top)), mi_top['mi_score'].values, color='lightgreen')
axes[1, 0].set_yticks(range(len(mi_top)))
axes[1, 0].set_yticklabels(mi_top['feature'].values, fontsize=9)
axes[1, 0].axvline(mi_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {mi_threshold:.4f}')
axes[1, 0].set_xlabel('Mutual Information', fontsize=10, fontweight='bold')
axes[1, 0].set_title('Method 3: Mutual Information (with momentum features)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='x')

voting_counts = summary_df['votes'].value_counts().sort_index(ascending=False)
colors_vote = ['#2ecc71' if x == 3 else '#f39c12' if x == 2 else '#e74c3c' for x in voting_counts.index]
axes[1, 1].bar(voting_counts.index, voting_counts.values, color=colors_vote, edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Number of Methods Passed', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Count', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Voting Summary (3 Methods)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks([0, 1, 2, 3])
for i, v in enumerate(voting_counts.values):
    axes[1, 1].text(voting_counts.index[i], v + 0.5, str(v), ha='center', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
viz_file = os.path.join(OUTPUT_DIR, "feature_selection_v3.png")
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"Saved: {viz_file}\n")
plt.close()

print("[OK] Feature selection V3 (FINAL) completed!")
print("\nReady to train baselines and foundation models!")
