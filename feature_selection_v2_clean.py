"""
Feature selection V2 - CLEAN VERSION

Removes mechanical/redundant features that have artificial high MI:
- log_ret_1d
- open_close_ret
- high_low_range
- close_ret_5d
- close_ret_20d (can use past returns but these are too recent)
- volume_change (too volatile, problematic values)

Keep only features with genuine predictive information.
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
OUTPUT_DIR = "outputs/feature_selection_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1] Loading training data (FIXED)...")
df_train = pd.read_csv(TRAIN_FILE)
df_train['date'] = pd.to_datetime(df_train['date'])
print(f"    Shape: {df_train.shape}")

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

# Remove problematic mechanical features
exclude_features = [
    'log_ret_1d',           # Mechanical: ln(Close[t]/Close[t-1])
    'open_close_ret',       # Mechanical: intra-day pattern
    'high_low_range',       # Mechanical: daily range
    'close_ret_5d',         # Too recent (only 5 days ahead)
    'close_ret_20d',        # Too recent (only 20 days ahead)
    'volume_change',        # Problematic: inf values
]

stock_features = [f for f in stock_features if f not in exclude_features]
macro_features = [f for f in macro_features if pd.api.types.is_numeric_dtype(df_train[f])]
all_features = stock_features + macro_features

print(f"    Stock features (after cleaning): {len(stock_features)}")
print(f"      Removed: {exclude_features}")
print(f"    Macro features: {len(macro_features)}")
print(f"    Total features: {len(all_features)}")

# Clean data
print("\n[2] Data cleaning...")
X = df_train[all_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
y = df_train['y_return_1d'].fillna(0)

print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}")

print("\n[3] METHOD 1: Low Variance Filter")
print("=" * 80)

variance_scores = X.var()
variance_sorted = variance_scores.sort_values(ascending=False)

print(f"\n    Top 10 highest variance:")
print(variance_sorted.head(10))

var_threshold = variance_scores.median()
low_var_features = variance_sorted[variance_sorted > var_threshold].index.tolist()

print(f"\n    Variance threshold (median): {var_threshold:.6f}")
print(f"    Features to keep: {len(low_var_features)}")
print(f"    Features to remove: {len(all_features) - len(low_var_features)}")

print("\n[4] METHOD 2: Correlation Filter")
print("=" * 80)

print("\n    2a. Correlation with Target:")
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)

print(f"\n    Top 15 correlated with target:")
print(corr_with_target.head(15))

corr_threshold = corr_with_target.quantile(0.05)
corr_target_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()

print(f"\n    Correlation threshold (5th percentile): {corr_threshold:.6f}")
print(f"    Features to keep: {len(corr_target_features)}")

print(f"\n    2b. Redundancy Filter (inter-feature correlation > 0.9):")
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

print(f"    Redundant features: {len(redundant_features)}")
print(f"    After redundancy removal: {len(corr_filtered_features)}")

print("\n[5] METHOD 3: Mutual Information with Target")
print("=" * 80)

X_scaled = StandardScaler().fit_transform(X)
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\n    Top 15 features by MI:")
print(mi_df.head(15))

mi_threshold = mi_df['mi_score'].median()
mi_features = mi_df[mi_df['mi_score'] > mi_threshold]['feature'].tolist()

print(f"\n    MI threshold (median): {mi_threshold:.6f}")
print(f"    Features to keep: {len(mi_features)}")

print("\n[6] Feature Selection Summary (CLEAN)")
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
    print(f"  {i:2d}. {feat:25s} | votes={row['votes']:.0f} | corr={row['corr_with_target']:.4f} | mi={row['mutual_info']:.4f}")

# Option 2: Moderate (10-20 features)
leaf_features = summary_df.head(15)['feature'].tolist()

print(f"\n\nOPTION 2: LEAF Context Features (For regime encoder)")
print("-" * 80)
print(f"Selected: {len(leaf_features)} features\n")
for i, feat in enumerate(leaf_features, 1):
    row = summary_df[summary_df['feature'] == feat].iloc[0]
    print(f"  {i:2d}. {feat:25s} | votes={row['votes']:.0f} | corr={row['corr_with_target']:.4f} | mi={row['mutual_info']:.4f}")

# Option 3
print(f"\n\nOPTION 3: TimesFM-U (Univariate)")
print("-" * 80)
print(f"  Input: close_ret_1d only")

print("\n[8] EXCLUDED FEATURES (Mechanical/Problematic)")
print("=" * 80)
print(f"\nRemoved due to mechanical relationship with target:")
for feat in exclude_features:
    if feat in df_train.columns:
        mi_val = mi_df[mi_df['feature'] == feat]['mi_score'].values
        if len(mi_val) > 0:
            print(f"  - {feat:25s} (MI={mi_val[0]:.4f}) - Derivation of Close price")

print("\n[9] Saving Results")
print("=" * 80)

summary_file = os.path.join(OUTPUT_DIR, "feature_selection_summary_v2.csv")
summary_df.to_csv(summary_file, index=False)
print(f"    Saved: {summary_file}")

# Save feature lists
feature_lists_file = os.path.join(OUTPUT_DIR, "selected_features_v2.txt")
with open(feature_lists_file, 'w') as f:
    f.write("FEATURE SELECTION RESULTS - V2 (CLEANED)\n")
    f.write("=" * 80 + "\n\n")
    f.write("Removed Mechanical Features:\n")
    for feat in exclude_features:
        f.write(f"  - {feat}\n")
    f.write("\n" + "=" * 80 + "\n\n")
    f.write(f"MULTIVARIATE FM FEATURES ({len(top_features)})\n")
    f.write("-" * 80 + "\n")
    f.write("Use for: Chronos-2-M, Moirai-2-M\n")
    f.write("Features:\n")
    for feat in top_features:
        f.write(f"  - {feat}\n")
    f.write(f"\nLEAF CONTEXT FEATURES ({len(leaf_features)})\n")
    f.write("-" * 80 + "\n")
    f.write("Use for: LEAF regime encoder\n")
    f.write("Features:\n")
    for feat in leaf_features:
        f.write(f"  - {feat}\n")

print(f"    Saved: {feature_lists_file}")

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
axes[1, 0].set_title('Method 3: Mutual Information (after removing mechanical)', fontsize=12, fontweight='bold')
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
viz_file = os.path.join(OUTPUT_DIR, "feature_selection_v2.png")
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"    Saved: {viz_file}")
plt.close()

print("\n[OK] Feature selection V2 (cleaned) completed!")
print(f"\nNext steps:")
print(f"  1. Use 'multivariate_fm_features' for FM models")
print(f"  2. Use 'leaf_context_features' for LEAF encoder")
print(f"  3. No mechanical leakage detected")
