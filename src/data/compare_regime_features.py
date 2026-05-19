"""
A/B Comparison: HMM 2-feature vs 3-feature
============================================
Muc dich: Cung cap bang chung thuc nghiem cho claim trong Section 3.5:
  "Bo sung port_ret_ma20 giup HMM phan biet Bear vs HighVol Recovery"

Output:
  data/processed/v2/regime_comparison.csv  (bang so sanh per-state stats)
  Console output: side-by-side table cho paper

Verification:
  - VARIANT A (2 features): port_ret + port_vol_20
    Du doan: Bear state co return DUONG (mixed crash + recovery)
  - VARIANT B (3 features): + port_ret_ma20
    Du doan: Bear state co return AM (true bear market)

Test cases dac biet:
  - COVID crash (3/2020): nen duoc gan Bear in cả 2 phien ban
  - COVID recovery (5/2020): VARIANT A gan Bear (sai), VARIANT B gan Sideways/Bull
  - Bond crisis (10/2022): nen duoc gan Bear
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
FEAT_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_all.csv"
OUT_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "regime_comparison.csv"

TRAIN_END = "2022-06-30"
RANDOM_SEED = 42

# Cac giai doan dac biet de kiem tra
SPECIAL_PERIODS = {
    "COVID crash":     ("2020-03-01", "2020-03-31"),
    "COVID recovery":  ("2020-05-01", "2020-06-30"),
    "Recovery 2021":   ("2021-01-01", "2021-06-30"),
    "Bond crisis":     ("2022-09-15", "2022-11-30"),
    "FTSE upgrade":    ("2025-09-01", "2025-12-31"),
}


def build_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average ret_1d, vol_20d qua tat ca tickers + tinh ret_ma20."""
    port = (
        df.groupby("Date")[["ret_1d", "vol_20d"]]
        .mean()
        .rename(columns={"ret_1d": "port_ret", "vol_20d": "port_vol20"})
    )
    port.index = pd.to_datetime(port.index)
    port = port.sort_index()
    port["port_ret_ma20"] = port["port_ret"].rolling(20).mean()
    return port


def fit_hmm_and_label(port: pd.DataFrame, features: list, variant_name: str) -> tuple:
    """
    Fit HMM, gan nhan theo Sharpe, tra ve (states_series, summary_df).
    """
    p = port.dropna(subset=features).copy()
    train_mask = p.index <= TRAIN_END
    X = p[features].values

    # Scaler tren Train only
    sc = StandardScaler()
    X_tr   = sc.fit_transform(X[train_mask])
    X_full = sc.transform(X)

    # Fit
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
        tol=1e-4,
    )
    model.fit(X_tr)

    print(f"\n{'=' * 70}")
    print(f"VARIANT {variant_name}: {len(features)} features")
    print(f"  Features: {features}")
    print(f"  Days analyzed: {len(p)} | Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood (train): {model.score(X_tr):.2f}")
    print('=' * 70)

    # Predict
    states = model.predict(X_full)

    # Stats per state
    rows = []
    for s in range(3):
        mask = states == s
        if mask.sum() == 0:
            continue
        ret = p["port_ret"].values[mask].mean()
        vol = p["port_ret"].values[mask].std()
        sharpe = ret / (vol + 1e-9)
        rows.append({
            "raw_state": s,
            "n_days": int(mask.sum()),
            "pct": round(mask.mean() * 100, 1),
            "ret_daily": round(ret, 6),
            "ret_annual_pct": round(ret * 252 * 100, 2),
            "vol_annual_pct": round(vol * np.sqrt(252) * 100, 2),
            "sharpe_daily": round(sharpe, 3),
        })

    df_states = pd.DataFrame(rows).sort_values("sharpe_daily", ascending=False).reset_index(drop=True)
    df_states["regime_label"] = ["Bull", "Sideways", "Bear"]
    df_states["variant"] = variant_name

    print("\n  Per-state statistics (sorted by Sharpe):")
    display_cols = ["regime_label", "raw_state", "n_days", "pct",
                    "ret_annual_pct", "vol_annual_pct", "sharpe_daily"]
    print(df_states[display_cols].to_string(index=False))

    # Map states -> labels
    label_map = dict(zip(df_states["raw_state"], df_states["regime_label"]))
    states_labeled = pd.Series([label_map[s] for s in states], index=p.index, name=variant_name)

    return states_labeled, df_states


def compare_special_periods(states_a: pd.Series, states_b: pd.Series):
    """In ra phan loai cua 2 variants tren cac giai doan dac biet."""
    print(f"\n{'=' * 70}")
    print("REGIME LABELS DURING SPECIAL PERIODS")
    print('=' * 70)
    print(f"{'Period':<20s} {'Date range':<25s} {'Variant A (2-feat)':<22s} {'Variant B (3-feat)':<22s}")
    print('-' * 90)

    for name, (start, end) in SPECIAL_PERIODS.items():
        sub_a = states_a.loc[start:end]
        sub_b = states_b.loc[start:end]
        if len(sub_a) == 0:
            continue
        # Mode: nhan pho bien nhat trong period
        mode_a = sub_a.value_counts().index[0] if len(sub_a) else "—"
        mode_b = sub_b.value_counts().index[0] if len(sub_b) else "—"
        # Pct cua mode
        pct_a = sub_a.value_counts().iloc[0] / len(sub_a) * 100 if len(sub_a) else 0
        pct_b = sub_b.value_counts().iloc[0] / len(sub_b) * 100 if len(sub_b) else 0
        print(f"{name:<20s} {start} → {end[:7]}    "
              f"{mode_a} ({pct_a:.0f}%)        "
              f"{mode_b} ({pct_b:.0f}%)")


def main():
    print("=" * 70)
    print("HMM A/B COMPARISON: 2-feature vs 3-feature")
    print(f"Train: 2013-01-02 → {TRAIN_END}")
    print("=" * 70)

    # Load
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    print(f"\nLoaded features_all.csv: {df.shape}")

    # Build portfolio
    port = build_portfolio(df)
    print(f"Portfolio series: {port.shape[0]} days")

    # Run 2 variants
    states_a, summary_a = fit_hmm_and_label(
        port, ["port_ret", "port_vol20"], "A"
    )
    states_b, summary_b = fit_hmm_and_label(
        port, ["port_ret", "port_vol20", "port_ret_ma20"], "B"
    )

    # Compare special periods
    compare_special_periods(states_a, states_b)

    # Final conclusion
    print(f"\n{'=' * 70}")
    print("CONCLUSION (KEY EVIDENCE FOR PAPER)")
    print('=' * 70)
    bear_a = summary_a[summary_a["regime_label"] == "Bear"].iloc[0]
    bear_b = summary_b[summary_b["regime_label"] == "Bear"].iloc[0]
    print(f"\nBear state — annualized return:")
    print(f"  Variant A (2 features): {bear_a['ret_annual_pct']:+.2f}% per year")
    print(f"  Variant B (3 features): {bear_b['ret_annual_pct']:+.2f}% per year")
    diff = bear_a["ret_annual_pct"] - bear_b["ret_annual_pct"]
    print(f"  Difference: {diff:+.2f} percentage points")

    print(f"\nBear state — annualized volatility:")
    print(f"  Variant A: {bear_a['vol_annual_pct']:.2f}%")
    print(f"  Variant B: {bear_b['vol_annual_pct']:.2f}%")

    print(f"\nBear state — % of sample:")
    print(f"  Variant A: {bear_a['pct']:.1f}% ({bear_a['n_days']} days)")
    print(f"  Variant B: {bear_b['pct']:.1f}% ({bear_b['n_days']} days)")

    if bear_b["ret_annual_pct"] < bear_a["ret_annual_pct"]:
        print(f"\n>> CONFIRMED: 3-feature tach Bear sach hon")
        print(f"   - Bear return giam {abs(diff):.2f} pp -> phan loai chinh xac hon")
        if bear_b["ret_annual_pct"] < 0:
            print(f"   - Variant B Bear co return AM -> true Bear market")
        else:
            print(f"   - Variant B Bear van duong nhung thap hon dang ke")
    else:
        print(f"\n>> UNEXPECTED: 3-feature khong cai thien (kiem tra lai data)")

    # Save comparison table
    combined = pd.concat([summary_a, summary_b], ignore_index=True)
    combined = combined[["variant", "regime_label", "n_days", "pct",
                         "ret_annual_pct", "vol_annual_pct", "sharpe_daily"]]
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print("\nUse this table for paper Section 3.5 (Regime Detection methodology).")


if __name__ == "__main__":
    main()
