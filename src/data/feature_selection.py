"""
Feature Selection — 4-step pipeline
====================================
Input:  data/processed/v2/features_with_regime.csv
Output: data/processed/v2/features_selected.csv  (panel data with selected cols)
        data/processed/v2/feature_selection_report.csv  (full ranking report)

4 steps (PROJECT_CONTEXT.md Section 4.3):
  1. Spearman + Bonferroni  -> loai feature khong significant
  2. Mutual Information     -> loai MI thap + redundant pairs (corr > 0.95)
  3. SHAP / GBM importance  -> rank theo marginal contribution
  4. Walk-forward stability -> loai feature unstable qua 5 time-folds

Target: y = next-day log return (ret_1d.shift(-1)) per ticker
TRAIN ONLY: 2013-01-02 -> 2022-06-30
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Optional SHAP — fallback dung GBM feature_importances_ neu chua cai
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
FEAT_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_with_regime.csv"
OUT_DIR   = ROOT_DIR / "data" / "processed" / "v2"
SEL_PATH  = OUT_DIR / "features_selected.csv"
REPORT_PATH = OUT_DIR / "feature_selection_report.csv"

TRAIN_END = "2022-06-30"
RANDOM_SEED = 42

# Threshold cho moi buoc
ALPHA_BASE       = 0.05    # Spearman significance
MI_KEEP_RATIO    = 0.70    # Giu top 70% theo MI
REDUNDANCY_CORR  = 0.90    # Pearson > 0.90 = redundant
N_FINAL          = 8       # Cuoi cung giu 8 features
N_FOLDS          = 5       # Walk-forward
GBM_N_ESTIMATORS = 100
GBM_MAX_DEPTH    = 4

# Cot loai khoi candidate (id, raw price, target-related)
EXCLUDE_COLS = {
    "Date", "ticker",
    "Open", "High", "Low", "Close", "Volume",
    "regime_hmm", "regime_vol",  # categorical, handle bo regime encoder rieng
}


# =============================================================
# Load data + setup target
# =============================================================

def load_train_data():
    """
    Load features_with_regime.csv, tao target = ret_1d.shift(-1) per ticker,
    chi giu TRAIN set.
    """
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # Target: next-day return (per ticker)
    df["target"] = df.groupby("ticker")["ret_1d"].shift(-1)

    # Train mask
    train = df[df["Date"] <= TRAIN_END].copy()
    train = train.dropna(subset=["target"])

    # Candidate features = numeric, khong nam trong EXCLUDE
    candidates = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and c != "target"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    # fracdiff_d hoac column hang so -> bo
    constant_cols = [c for c in candidates if train[c].nunique() <= 1]
    candidates = [c for c in candidates if c not in constant_cols]

    # Drop rows con NaN trong bat ky candidate (do rolling windows dau chuoi)
    train_clean = train[candidates + ["target", "ticker", "Date"]].dropna()

    print(f"Train rows: {len(train_clean):,}")
    print(f"Candidate features: {len(candidates)}")
    if constant_cols:
        print(f"Removed constants: {constant_cols}")

    return train_clean, candidates


# =============================================================
# Step 1 — Spearman + Bonferroni
# =============================================================

def step1_spearman(train: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Tinh Spearman corr (X_i vs target) + p-value.
    Bonferroni: alpha_corrected = 0.05 / N
    Loai feature p > alpha_corrected.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Spearman + Bonferroni correction")
    print("=" * 60)

    n = len(features)
    alpha_corrected = ALPHA_BASE / n
    print(f"Bonferroni alpha = {ALPHA_BASE}/{n} = {alpha_corrected:.6f}")

    y = train["target"].values
    rows = []
    for f in features:
        x = train[f].values
        if np.std(x) == 0:
            corr, pval = 0, 1
        else:
            corr, pval = stats.spearmanr(x, y)
            if np.isnan(corr):
                corr, pval = 0, 1
        rows.append({
            "feature": f,
            "spearman_corr": corr,
            "spearman_pval": pval,
            "spearman_pass": pval < alpha_corrected,
        })

    report = pd.DataFrame(rows).sort_values("spearman_pval")
    n_pass = report["spearman_pass"].sum()
    print(f"Pass: {n_pass}/{n} features (p < {alpha_corrected:.2e})")
    return report


# =============================================================
# Step 2 — Mutual Information + Redundancy
# =============================================================

def step2_mutual_info(train: pd.DataFrame, features: list, report: pd.DataFrame) -> pd.DataFrame:
    """
    Tinh MI(X_i, target). Giu top MI_KEEP_RATIO * len.
    Loai redundant pairs: neu corr(X_i, X_j) > REDUNDANCY_CORR -> giu cai MI cao hon.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Mutual Information + Redundancy")
    print("=" * 60)

    # Chi xet features pass step 1
    surviving = report[report["spearman_pass"]]["feature"].tolist()
    if len(surviving) == 0:
        print("WARNING: Khong co feature nao pass Step 1 -> fallback giu top |corr|")
        surviving = report.nlargest(20, "spearman_corr", keep="all")["feature"].tolist()

    X = train[surviving].values
    y = train["target"].values

    print(f"Computing MI for {len(surviving)} features (n={len(y):,} rows)...")
    mi = mutual_info_regression(X, y, random_state=RANDOM_SEED, n_neighbors=5)

    mi_df = pd.DataFrame({"feature": surviving, "mi_score": mi}).sort_values("mi_score", ascending=False)

    # Giu top MI_KEEP_RATIO
    keep_n = max(int(len(surviving) * MI_KEEP_RATIO), N_FINAL)
    top_mi = mi_df.head(keep_n)["feature"].tolist()
    print(f"Top {keep_n} features by MI (ratio={MI_KEEP_RATIO}):")

    # Loai redundant: tinh corr matrix giua top_mi features
    corr_matrix = train[top_mi].corr().abs()
    to_drop = set()
    sorted_by_mi = mi_df.set_index("feature")["mi_score"].to_dict()

    for i, f1 in enumerate(top_mi):
        if f1 in to_drop:
            continue
        for f2 in top_mi[i+1:]:
            if f2 in to_drop:
                continue
            if corr_matrix.loc[f1, f2] > REDUNDANCY_CORR:
                # Giu cai MI cao hon
                if sorted_by_mi[f1] >= sorted_by_mi[f2]:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)
                    break

    final_mi = [f for f in top_mi if f not in to_drop]
    print(f"After dropping redundant (|corr| > {REDUNDANCY_CORR}): {len(final_mi)}")
    if to_drop:
        print(f"  Dropped (redundant): {sorted(to_drop)}")

    # Update report
    report = report.merge(mi_df, on="feature", how="left")
    report["mi_pass"] = report["feature"].isin(final_mi)
    report["mi_score"] = report["mi_score"].fillna(0)
    return report


# =============================================================
# Step 3 — SHAP (or GBM importance) on surrogate model
# =============================================================

def step3_shap_or_importance(train: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """
    Fit GBM tren TRAIN, tinh importance.
    - Neu SHAP available: dung TreeExplainer -> mean(|SHAP|)
    - Else: fallback GBM.feature_importances_ (Gini)
    """
    method = "SHAP (TreeExplainer)" if HAS_SHAP else "GBM Gini importance (SHAP not installed)"
    print("\n" + "=" * 60)
    print(f"STEP 3 — {method}")
    print("=" * 60)

    surviving = report[report["mi_pass"]]["feature"].tolist()
    print(f"Fitting GBM on {len(surviving)} features, {len(train):,} rows...")

    X = train[surviving].values
    y = train["target"].values

    model = GradientBoostingRegressor(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        # Subsample neu data lon de toc do
        n_sample = min(5000, len(X))
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), n_sample, replace=False)
        shap_vals = explainer.shap_values(X[idx])
        importance = np.abs(shap_vals).mean(axis=0)
    else:
        importance = model.feature_importances_

    imp_df = pd.DataFrame({"feature": surviving, "shap_importance": importance})
    imp_df = imp_df.sort_values("shap_importance", ascending=False)
    print(imp_df.head(15).to_string(index=False))

    # Update report
    report = report.merge(imp_df, on="feature", how="left")
    report["shap_importance"] = report["shap_importance"].fillna(0)
    return report


# =============================================================
# Step 4 — Walk-forward stability
# =============================================================

def step4_stability(train: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """
    TimeSeriesSplit n=5 trong TRAIN.
    Moi fold: fit GBM, tinh importance.
    Stability = mean_importance / std_importance qua 5 folds.
    """
    print("\n" + "=" * 60)
    print(f"STEP 4 — Walk-forward Stability ({N_FOLDS} folds)")
    print("=" * 60)

    surviving = report[report["mi_pass"]]["feature"].tolist()

    # Sort by Date trong train (TimeSeriesSplit can theo thoi gian)
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    X = train_sorted[surviving].values
    y = train_sorted["target"].values

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_importance = np.zeros((N_FOLDS, len(surviving)))

    for k, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        model = GradientBoostingRegressor(
            n_estimators=GBM_N_ESTIMATORS,
            max_depth=GBM_MAX_DEPTH,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
        model.fit(X_tr, y_tr)
        fold_importance[k] = model.feature_importances_
        print(f"  Fold {k+1}/{N_FOLDS}: train={len(tr_idx):,} | val={len(val_idx):,}")

    mean_imp = fold_importance.mean(axis=0)
    std_imp  = fold_importance.std(axis=0)
    stability = mean_imp / (std_imp + 1e-10)

    stab_df = pd.DataFrame({
        "feature"       : surviving,
        "stability_mean": mean_imp,
        "stability_std" : std_imp,
        "stability_score": stability,
    }).sort_values("stability_score", ascending=False)

    print("\nTop 15 by stability score:")
    print(stab_df.head(15).round(4).to_string(index=False))

    report = report.merge(stab_df, on="feature", how="left")
    return report


# =============================================================
# Final selection
# =============================================================

def final_select(report: pd.DataFrame, n_final: int = N_FINAL) -> list:
    """
    Combine SHAP importance + stability score thanh final ranking.
    Final score = rank(SHAP) + rank(stability) (lower = better)
    """
    print("\n" + "=" * 60)
    print(f"FINAL SELECTION — Top {n_final} features")
    print("=" * 60)

    # Chi xet features pass step 2 (qua MI)
    candidates = report[report["mi_pass"]].copy()

    # Composite ranking: average of normalized SHAP + stability
    candidates["shap_rank"]  = candidates["shap_importance"].rank(ascending=False)
    candidates["stab_rank"]  = candidates["stability_score"].rank(ascending=False)
    candidates["final_rank"] = (candidates["shap_rank"] + candidates["stab_rank"]) / 2

    candidates = candidates.sort_values("final_rank")
    selected = candidates.head(n_final)["feature"].tolist()

    print("\nFinal selection (combined SHAP + Stability):")
    cols_to_show = ["feature", "spearman_corr", "mi_score", "shap_importance",
                    "stability_score", "final_rank"]
    print(candidates.head(n_final)[cols_to_show].round(4).to_string(index=False))

    # Mark selected in report
    report["selected"] = report["feature"].isin(selected)

    return selected, report


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("FEATURE SELECTION — 4-step pipeline")
    print(f"Train: 2013-01-02 -> {TRAIN_END}")
    print(f"SHAP available: {HAS_SHAP}")
    print("=" * 60 + "\n")

    train, candidates = load_train_data()

    # 4 buoc
    report = step1_spearman(train, candidates)
    report = step2_mutual_info(train, candidates, report)
    report = step3_shap_or_importance(train, report)
    report = step4_stability(train, report)

    # Final
    selected, report = final_select(report, N_FINAL)

    # Save report
    report = report.sort_values("final_rank" if "final_rank" in report.columns else "spearman_pval")
    report.to_csv(REPORT_PATH, index=False)
    print(f"\nReport saved: {REPORT_PATH}")

    # Save selected features list + panel data
    print(f"\n{'=' * 60}")
    print(f"SELECTED ({len(selected)} features):")
    for i, f in enumerate(selected, 1):
        print(f"  {i}. {f}")

    # Save panel data voi cac cot duoc chon
    full_df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    keep_cols = ["Date", "ticker", "Open", "High", "Low", "Close", "Volume",
                 "regime_hmm", "regime_vol"] + selected
    keep_cols = [c for c in keep_cols if c in full_df.columns]
    out = full_df[keep_cols]
    out.to_csv(SEL_PATH, index=False)
    print(f"\nPanel data saved: {SEL_PATH}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()
