"""
Hien thi ket qua Feature Selection V3 theo 3 buoc
"""

import pandas as pd
import os

output_dir = "outputs/feature_selection_v3"
summary_file = os.path.join(output_dir, "feature_selection_summary_v3.csv")

print("\n" + "="*100)
print("FEATURE SELECTION V3 - KET QUA CHI TIET (3 BUOC)")
print("="*100 + "\n")

# Load kết quả
summary_df = pd.read_csv(summary_file)
summary_df = summary_df.sort_values('votes', ascending=False)

print("\n[BAO CAO TONG QUAN]")
print("-"*100)
print(f"Tong so dac trung: {len(summary_df)}")
print(f"  - Votes = 3 (vua qua 3 phuong phap): {(summary_df['votes'] == 3).sum()}")
print(f"  - Votes = 2 (vua qua 2 phuong phap): {(summary_df['votes'] == 2).sum()}")
print(f"  - Votes = 1 (vua qua 1 phuong phap): {(summary_df['votes'] == 1).sum()}")
print(f"  - Votes = 0 (khong vua phuong phap): {(summary_df['votes'] == 0).sum()}")

# BUOC 1: Low Variance
print("\n\n[BUOC 1: LOW VARIANCE FILTER]")
print("-"*100)
var_pass = summary_df[summary_df['low_variance_pass'] == True]
var_fail = summary_df[summary_df['low_variance_pass'] == False]
print(f"Dac trung thua: {len(var_pass)}/51")
print(f"Dac trung loai: {len(var_fail)}/51")
print(f"\nCac dac trung bao nhat co phuong sai (Top 10):")
top_var = summary_df.nlargest(10, 'variance')[['feature', 'variance']]
for idx, row in top_var.iterrows():
    print(f"  {row['feature']:25s} | Variance = {row['variance']:.4e}")

# BUOC 2: Correlation
print("\n\n[BUOC 2: CORRELATION FILTER]")
print("-"*100)
corr_pass = summary_df[summary_df['correlation_pass'] == True]
corr_fail = summary_df[summary_df['correlation_pass'] == False]
print(f"Dac trung thua: {len(corr_pass)}/51")
print(f"Dac trung loai: {len(corr_fail)}/51")
print(f"\nCac dac trung co tuong quan manh nhat voi target (Top 10):")
top_corr = summary_df.nlargest(10, 'corr_with_target')[['feature', 'corr_with_target']]
for idx, row in top_corr.iterrows():
    print(f"  {row['feature']:25s} | Correlation = {row['corr_with_target']:.6f}")

# BUOC 3: Mutual Information
print("\n\n[BUOC 3: MUTUAL INFORMATION FILTER]")
print("-"*100)
mi_pass = summary_df[summary_df['mi_pass'] == True]
mi_fail = summary_df[summary_df['mi_pass'] == False]
print(f"Dac trung thua: {len(mi_pass)}/51")
print(f"Dac trung loai: {len(mi_fail)}/51")
print(f"\nCac dac trung co MI cao nhat voi target (Top 15):")
top_mi = summary_df.nlargest(15, 'mutual_info')[['feature', 'mutual_info']]
for i, (idx, row) in enumerate(top_mi.iterrows(), 1):
    special = "[MOMENTUM]" if row['feature'] in ['log_ret_1d', 'close_ret_1d'] else ""
    print(f"  {i:2d}. {row['feature']:25s} | MI = {row['mutual_info']:.6f} {special}")

# KET QUA CUOI CUNG
print("\n\n[KET QUA CUOI CUNG - HE THONG BINH CHON]")
print("-"*100)

votes_3 = summary_df[summary_df['votes'] == 3]
print(f"\n[PHUONG AN 1: MULTIVARIATE FM - {len(votes_3)} dac trung (CHINH)]")
print("Dung cho: Chronos-2-M, Moirai-2-M, TimesFM-M\n")
print(f"{'STT':>3} | {'Dac trung':25s} | Votes | Corr   | MI     ")
print("-"*70)
for i, (idx, row) in enumerate(votes_3[['feature', 'votes', 'corr_with_target', 'mutual_info']].iterrows(), 1):
    special = "[MOMENTUM]" if row['feature'] in ['log_ret_1d', 'close_ret_1d'] else ""
    print(f"{i:3d} | {row['feature']:25s} |   {int(row['votes'])}   | {row['corr_with_target']:6.4f} | {row['mutual_info']:6.4f} {special}")

# Cac dac trung votes=2 cho LEAF
votes_2 = summary_df[summary_df['votes'] == 2].head(7)
print(f"\n[PHUONG AN 2: LEAF CONTEXT - {len(votes_3) + len(votes_2)} dac trung]")
print("Them 7 dac trung votes=2:\n")
print(f"{'STT':>3} | {'Dac trung':25s} | Votes | Corr   | MI     ")
print("-"*70)
for i, (idx, row) in enumerate(votes_2[['feature', 'votes', 'corr_with_target', 'mutual_info']].iterrows(), 1):
    special = "[MOMENTUM]" if row['feature'] in ['log_ret_1d', 'close_ret_1d'] else ""
    print(f"{i+len(votes_3):3d} | {row['feature']:25s} |   {int(row['votes'])}   | {row['corr_with_target']:6.4f} | {row['mutual_info']:6.4f} {special}")

# Duong dan File
print("\n\n[FILE THAM KHAO]")
print("-"*100)
print(f"Bao cao chi tiet: FEATURE_SELECTION_REPORT_V3.md")
print(f"Tom tat nhanh: FEATURE_SELECTION_SUMMARY_V3.txt")
print(f"Dac trung duoc chon: {os.path.join(output_dir, 'selected_features_v3.txt')}")
print(f"Du lieu CSV: {summary_file}")
print(f"Bieu do: {os.path.join(output_dir, 'feature_selection_v3.png')}")

print("\n" + "="*100)
print("KET LUAN: 8 dac trung votes=3 san sang de train Foundation Models")
print("="*100 + "\n")
