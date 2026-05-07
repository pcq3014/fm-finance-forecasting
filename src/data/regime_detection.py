"""
Regime Detection — 3-state Gaussian HMM
========================================
Input:  data/processed/v2/features_all.csv
Output: data/processed/v2/regime.csv
        data/processed/v2/features_with_regime.csv

3 regimes (sort by Sharpe-like ratio = ret/vol):
  Bull     — Low volatility, trending up
  Sideways — Medium volatility, directionless
  Bear     — High volatility + negative trend (COVID, bond crisis...)

HMM dung 3 features: port_ret, vol_20d, ret_ma20
  port_ret  : signal ngan han
  vol_20d   : volatility cluster
  ret_ma20  : xu huong 20 ngay (phan biet Bear vs HighVol)

Method:
  - Fit: CHI tren Train (2013-01-02 -> 2022-06-30)
  - Predict: toan bo series
  - Cross-validate: vol-based rule (Bear + High-vol -> HighVol)
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

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
FEAT_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_all.csv"
OUT_DIR    = ROOT_DIR / "data" / "processed" / "v2"
REGIME_PATH = OUT_DIR / "regime.csv"
FINAL_PATH  = OUT_DIR / "features_with_regime.csv"

TRAIN_END  = "2022-06-30"
N_STATES   = 3
N_ITER     = 200
RANDOM_SEED = 42


# =============================================================
# 1. Build portfolio-level daily series
# =============================================================

def build_portfolio_series(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average ret_1d va vol_20d qua tat ca tickers."""
    port = (
        df.groupby("Date")[["ret_1d", "vol_20d"]]
        .mean()
        .rename(columns={"ret_1d": "port_ret", "vol_20d": "port_vol20"})
    )
    port.index = pd.to_datetime(port.index)
    port = port.sort_index()

    # vol_60d cua portfolio (de tinh vol_ratio)
    port["port_vol60"] = port["port_ret"].rolling(60).std()
    port["port_vol_ratio"] = port["port_vol20"] / port["port_vol60"].replace(0, np.nan)

    # ret_ma20 de dung trong rule-based cross-validation
    port["port_ret_ma20"] = port["port_ret"].rolling(20).mean()

    return port


# =============================================================
# 2. Fit 3-state HMM (chi tren Train)
# =============================================================

def fit_hmm(port: pd.DataFrame) -> tuple:
    """
    Fit GaussianHMM 3-state tren Train set.
    3 features: port_ret (short-term), vol_20d (volatility), ret_ma20 (trend direction)
    ret_ma20 giup HMM phan biet Bear (high vol + negative trend) vs HighVol (high vol + positive)
    """
    train_mask = port.index <= TRAIN_END
    # Khong can fillna(0) nua vi da dropna trong main()
    X_full = port[["port_ret", "port_vol20", "port_ret_ma20"]].values
    assert not np.isnan(X_full).any(), "HMM input van con NaN — kiem tra dropna o main()"

    # Scaler fit tren Train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_full[train_mask])
    X_full_scaled  = scaler.transform(X_full)

    # Fit HMM tren Train
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        tol=1e-4,
    )
    lengths_train = [X_train_scaled.shape[0]]
    model.fit(X_train_scaled, lengths=lengths_train)

    print(f"  HMM converged: {model.monitor_.converged}")
    print(f"  Log-likelihood (train): {model.score(X_train_scaled):.2f}")

    # Predict toan bo series (train + test)
    states_full = model.predict(X_full_scaled)

    return model, scaler, states_full


# =============================================================
# 3. Label states: Bull / Sideways / Bear
# =============================================================

def label_states(port: pd.DataFrame, states: np.ndarray) -> dict:
    """
    Gan nhan Bull/Sideways/Bear dua tren Risk-Adjusted Return (Return / Volatility).
    """
    state_ret = {}
    state_vol = {}
    state_sharpe = {}

    for s in range(N_STATES):
        mask = states == s
        if mask.sum() == 0:
            state_ret[s], state_vol[s], state_sharpe[s] = 0, 1, 0
        else:
            ret = port["port_ret"].values[mask].mean()
            vol = port["port_ret"].values[mask].std()
            state_ret[s] = ret
            state_vol[s] = vol
            # Phân loại dựa trên Return/Vol (Sharpe-like ratio)
            state_sharpe[s] = ret / (vol + 1e-6)

    # Sort theo Sharpe (ret/vol): Bull=cao nhat, Bear=thap nhat
    sorted_states = sorted(state_sharpe.keys(), key=lambda s: state_sharpe[s], reverse=True)
    label_map = {
        sorted_states[0]: "Bull",
        sorted_states[1]: "Sideways",
        sorted_states[2]: "Bear",
    }

    print(f"\n  State labeling (Sharpe-sorted: Bull > Sideways > Bear):")
    for s in sorted_states:
        print(f"    State {s} -> {label_map[s]:8s} | "
              f"ret={state_ret[s]:+.4f} | vol={state_vol[s]:.4f} | "
              f"sharpe={state_sharpe[s]:+.3f} | "
              f"n={(states == s).sum()} ({(states==s).mean()*100:.1f}%)")

    return label_map, state_ret, state_vol


# =============================================================
# 4. Economic validation
# =============================================================

def validate_economic(label_map: dict, state_ret: dict, state_vol: dict, states: np.ndarray) -> bool:
    """Kiem tra economic validity cho 3 regimes Bull/Sideways/Bear."""
    bull_s = [s for s, l in label_map.items() if l == "Bull"][0]
    side_s = [s for s, l in label_map.items() if l == "Sideways"][0]
    bear_s = [s for s, l in label_map.items() if l == "Bear"][0]

    n = len(states)
    checks = {
        "Bull Sharpe > Sideways Sharpe": state_ret[bull_s] / (state_vol[bull_s] + 1e-6) > state_ret[side_s] / (state_vol[side_s] + 1e-6),
        "Bull vol < Bear vol"          : state_vol[bull_s] < state_vol[bear_s],
        "Bear vol > Sideways vol"      : state_vol[bear_s] > state_vol[side_s],
        "Bull > 20% sample"            : (states == bull_s).sum() / n > 0.20,
        "Bear > 10% sample"            : (states == bear_s).sum() / n > 0.10,
    }

    print("\n  Economic Validation:")
    all_pass = True
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"    [{status}] {check}")
        if not result:
            all_pass = False

    return all_pass


# =============================================================
# 5. Rule-based cross-validation
# =============================================================

def rule_based_regime(port: pd.DataFrame) -> pd.Series:
    """
    Tinh regime theo rule don gian de cross-validate voi HMM.
    Bull:     vol_ratio < 0.8  AND ret_ma20 > 0
    Bear:     vol_ratio > 1.3  AND ret_ma20 < 0
    Sideways: otherwise (bao gom ca high-vol + positive trend)
    """
    vr  = port["port_vol_ratio"]
    r20 = port["port_ret_ma20"]

    regime = pd.Series("Sideways", index=port.index)
    regime[(vr < 0.8) & (r20 > 0)] = "Bull"
    regime[(vr > 1.3) & (r20 < 0)] = "Bear"

    return regime


def cross_validate(hmm_labels: pd.Series, vol_labels: pd.Series) -> float:
    """Tinh % dong thuan giua HMM va vol-based (Bull/Sideways/Bear)."""
    agree = (hmm_labels == vol_labels).mean()
    print(f"\n  Cross-validation (HMM vs Vol-based): agreement = {agree*100:.1f}%")
    if agree >= 0.70:
        print("  -> OK, dung HMM lam primary")
    else:
        print("  -> WARNING: agreement < 70%")
        print("  -> Vol-based dung nguong cung (0.8/1.3), HMM hoc tu data.")
        print("  -> Van dung HMM lam primary (economically validated).")
    return agree


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("REGIME DETECTION — 3-state Gaussian HMM")
    print(f"Train: 2013-01-02 -> {TRAIN_END} | Test: rest")
    print("=" * 60)

    print("\n[1/5] Loading features_all.csv...")
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    print(f"  Shape: {df.shape} | Tickers: {df['ticker'].nunique()}")

    print("\n[2/5] Building portfolio series (equal-weight)...")
    port = build_portfolio_series(df)
    # Drop ALL HMM input features = NaN -> dam bao input HMM 100% sach (khong co fillna(0) gia tao)
    port = port.dropna(subset=["port_ret", "port_vol20", "port_ret_ma20"])
    print(f"  Portfolio after dropna: {port.shape[0]} days "
          f"| start: {port.index.min().date()} | end: {port.index.max().date()}")
    
    print("\n[3/5] Fitting 3-state Gaussian HMM (train only)...")
    model, scaler, states_arr = fit_hmm(port)

    print("\n[4/5] Labeling states...")
    label_map, state_ret, state_vol = label_states(port, states_arr)

    # Map states -> labels
    port["regime_hmm_raw"] = states_arr
    port["regime_hmm"] = pd.Series(states_arr, index=port.index).map(label_map)

    # Smooth: rolling mode 5 ngày (chống nhiễu chuyển trạng thái liên tục)
    # Phải làm trên numeric (state index) trước, rồi map sang label sau
    smoothed_states = (
        pd.Series(states_arr, index=port.index)
        .rolling(5, min_periods=1)
        .apply(lambda x: pd.Series(x).mode().iloc[0])
        .astype(int)
    )
    port["regime_hmm"] = smoothed_states.map(label_map)

    # Economic validation
    valid = validate_economic(label_map, state_ret, state_vol, states_arr)

    print("\n[5/5] Cross-validating with vol-based rule...")
    port["regime_vol"] = rule_based_regime(port)
    agreement = cross_validate(port["regime_hmm"], port["regime_vol"])

    print("\n--- Mean Return by Regime (HMM) ---")
    summary = port.groupby("regime_hmm")["port_ret"].agg(["mean", "std", "count"])
    summary.columns = ["mean_ret", "std_ret", "n_days"]
    summary["annualized_ret"] = summary["mean_ret"] * 252
    summary["annualized_vol"] = summary["std_ret"] * np.sqrt(252)
    print(summary.round(4).to_string())

    # --- Save regime.csv ---
    regime_out = port[["regime_hmm", "regime_vol"]].copy()
    regime_out.index.name = "Date"
    regime_out.to_csv(REGIME_PATH)
    print(f"\nDa luu regime: {REGIME_PATH}")

    # --- Merge regime vao features_all ---
    print("\nMerging regime vao features_all...")
    df["Date"] = pd.to_datetime(df["Date"])
    regime_out_reset = regime_out.reset_index()
    df_final = df.merge(regime_out_reset, on="Date", how="left")

    df_final["regime_hmm"] = df_final["regime_hmm"].fillna("Sideways")
    df_final["regime_vol"] = df_final["regime_vol"].fillna("Sideways")

    df_final.to_csv(FINAL_PATH, index=False)
    print(f"Da luu: {FINAL_PATH} | Shape: {df_final.shape}")


if __name__ == "__main__":
    main()