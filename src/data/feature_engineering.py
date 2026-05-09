"""
Feature Engineering — VN30-MacroTS
===================================
Input:
  data/raw/selected/[TICKER].csv  — OHLCV, 15 mã
  data/macro/macro_combined.csv   — 6 macro features

Output:
  data/processed/v2/features_all.csv — long format (Date, ticker, ~35 features)

Groups:
  A — Price features (momentum, volatility, trend, volume, mean-reversion)
  B — Fractional Differentiation (close_fd fitted on TRAIN only)
  C — Derived macro features (vix_dev, usdvnd_mom, vnm_ret_z, risk_off...)

Rules:
  - No look-ahead: tất cả rolling windows dùng dữ liệu quá khứ
  - Fractional diff d fitted trên TRAIN (2013-01-02 → 2022-06-30)
  - Macro merge vào trading calendar (left join on price dates)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent
PRICE_DIR = ROOT_DIR / "data" / "raw" / "selected"
MACRO_PATH = ROOT_DIR / "data" / "macro" / "macro_combined.csv"
OUT_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_all.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2022-06-30"
TICKERS = [
    "ACB", "BVH", "CTG", "MBB", "STB", "VCB",
    "SSI", "KDH", "PDR", "VIC", "MSN", "VNM",
    "HPG", "GAS", "FPT",
]


# =============================================================
# Fractional Differentiation (manual, no fracdiff dependency)
# =============================================================

def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
    """Tính trọng số FracDiff theo Lopez de Prado (2018)."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < thresh:
            break
    return np.array(w[::-1])


def fracdiff_series(s: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Áp dụng fractional differentiation cho series (vectorized, safe)."""
    w = _fracdiff_weights(d, len(s), thresh)
    width = len(w)
    arr = s.values.astype(float)
    n = len(arr)
    if n < width:
        return pd.Series(np.nan, index=s.index)
    # sliding_window_view an toàn hơn as_strided (không đọc ngoài bounds)
    windows = np.lib.stride_tricks.sliding_window_view(arr, width)  # shape: (n-width+1, width)
    vals    = windows @ w
    result  = np.full(n, np.nan)
    result[width - 1:] = vals
    return pd.Series(result, index=s.index)


def find_optimal_d(s: pd.Series, train_mask: pd.Series,
                   d_min: float = 0.1, d_max: float = 1.0,
                   step: float = 0.1, p_thresh: float = 0.05) -> float:
    """
    Tìm d nhỏ nhất để ADF p-value < p_thresh trên Train set.
    Giữ tối đa memory (d nhỏ = ít differentiate).
    """
    train_s = s[train_mask].dropna()
    for d in np.arange(d_min, d_max + step, step):
        fd = fracdiff_series(train_s, round(d, 2)).dropna()
        if len(fd) < 50:
            continue
        try:
            pval = adfuller(fd, maxlag=5, autolag=None)[1]
            if pval < p_thresh:
                return round(d, 2)
        except Exception:
            continue
    return 1.0  # fallback: full differentiation


# =============================================================
# Group A — Price Features
# =============================================================

def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính price features từ OHLCV.
    df phải có cột: Open, High, Low, Close, Volume, index=Date
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    feat = pd.DataFrame(index=df.index)

    # --- Momentum ---
    feat["ret_1d"]  = np.log(c / c.shift(1))
    feat["ret_5d"]  = np.log(c / c.shift(5))
    feat["ret_20d"] = np.log(c / c.shift(20))
    feat["ret_60d"] = np.log(c / c.shift(60))

    # --- Volatility ---
    r = feat["ret_1d"]
    feat["vol_20d"]  = r.rolling(20).std()
    feat["vol_60d"]  = r.rolling(60).std()
    feat["vol_ratio"] = feat["vol_20d"] / feat["vol_60d"].replace(0, np.nan)

    # --- Trend ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feat["macd"]         = (ema12 - ema26) / c  # normalized by price

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    feat["rsi_14"] = (100 - 100 / (1 + rs)).mask(loss == 0, 100)

    # --- Mean Reversion ---
    bb_mid   = c.rolling(20).mean()
    bb_upper = bb_mid + 2 * c.rolling(20).std()
    bb_lower = bb_mid - 2 * c.rolling(20).std()
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_position"]   = (c - bb_lower) / bb_range
    feat["dist_52w_high"] = c / c.rolling(252).max() - 1

    # --- Volume ---
    vol_ma20 = v.rolling(20).mean().replace(0, np.nan)
    feat["volume_ratio"] = v / vol_ma20

    # VWAP deviation (20-day rolling VWAP)
    vwap20 = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    feat["vwap_dev"] = c / vwap20 - 1

    # High-Low range normalized
    feat["hl_range"] = (h - l) / c

    # --- Memory (autocorrelation) ---
    feat["autocorr_1"] = r.rolling(20).corr(r.shift(1))

    return feat


# =============================================================
# Group B — Fractional Differentiation
# =============================================================

def compute_fracdiff(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Tính close_fd với d_optimal fitted trên Train."""
    c = df["Close"]
    log_c = np.log(c)

    d_opt = find_optimal_d(log_c, train_mask)
    fd    = fracdiff_series(log_c, d_opt)

    feat = pd.DataFrame(index=df.index)
    feat["close_fd"] = fd
    
    # Đã xóa biến fracdiff_d ở đây

    return feat, d_opt


# =============================================================
# Group C — Derived Macro Features
# =============================================================

def compute_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Tính derived macro features từ macro_combined.
    Input: macro với index=Date (daily calendar)
    Output: DataFrame cùng index
    """
    feat = pd.DataFrame(index=macro.index)

    vix   = macro["vix"]
    usd   = macro["usdvnd"]
    ibor  = macro["ibor_rate"]
    pmi   = macro["pmi"]
    hsi   = macro["hsi_ret"]
    vnm   = macro["vnm_ret"]

    # --- VIX ---
    vix_ma252 = vix.rolling(252, min_periods=60).mean().replace(0, np.nan)
    feat["vix_dev"]    = vix / vix_ma252 - 1       # deviation from 1yr mean
    feat["vix_stress"] = (vix > 30).astype(int)     # binary stress flag

    # --- USD/VND ---
    feat["usdvnd_mom5"]  = np.log(usd / usd.shift(5))   # 5-day momentum
    usd_ma252 = usd.rolling(252, min_periods=60).mean().replace(0, np.nan)
    feat["usdvnd_dev"]   = usd / usd_ma252 - 1          # deviation from 1yr mean

    # --- VNM ETF (proxy foreign flow) ---
    vnm_roll60_mean = vnm.rolling(60, min_periods=10).mean()
    vnm_roll60_std  = vnm.rolling(60, min_periods=10).std().replace(0, np.nan)
    feat["vnm_ret_z"] = (vnm - vnm_roll60_mean) / vnm_roll60_std  # z-score 60d

    # VNM return streak: số ngày liên tiếp dương (+n) hoặc âm (-n), fast numpy loop
    sign = np.sign(vnm).values.astype(float)
    result = sign.copy()
    for i in range(1, len(sign)):
        if sign[i] != 0 and sign[i] == sign[i - 1]:
            result[i] = result[i - 1] + sign[i]
    feat["vnm_ret_streak"] = result

    # --- IBOR ---
    ibor_ma252 = ibor.rolling(252, min_periods=60).mean()
    feat["ibor_dev"]   = ibor - ibor_ma252              # deviation from 1yr mean
    feat["ibor_event"] = ibor.diff().ne(0).astype(int)  # 1 = rate change day

    # --- PMI ---
    feat["pmi_norm"]     = pmi - 50                     # deviation from neutral 50
    feat["pmi_momentum"] = pmi.diff()                   # monthly change (step series)
    # Đã xóa biến pmi_above50

    # --- HSI ---
    feat["hsi_ret_1d"]  = hsi                           # same as raw
    feat["hsi_vol_20"]  = hsi.rolling(20).std()         # 20-day HSI volatility

    # --- Risk-off Composite ---
    # Normalize each component với rolling 252-day z-score
    def roll_z(s, w=252, min_p=60):
        m = s.rolling(w, min_periods=min_p).mean()
        sd = s.rolling(w, min_periods=min_p).std().replace(0, np.nan)
        return (s - m) / sd

    vix_z   = roll_z(vix)
    usd_z   = roll_z(np.log(usd / usd.shift(1)).fillna(0))
    vnm_z_r = roll_z(vnm)

    feat["risk_off_composite"] = (
        vix_z * 0.4 + usd_z * 0.3 - vnm_z_r * 0.3
    )

    return feat


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING — VN30-MacroTS")
    print("=" * 60)

    # Load macro (daily calendar)
    macro = pd.read_csv(MACRO_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    macro_feat = compute_macro_features(macro)
    macro_all  = pd.concat([macro, macro_feat], axis=1)
    print(f"Macro loaded: {macro.shape} | Derived: {macro_feat.shape[1]} features")

    all_frames = []

    for ticker in TICKERS:
        path = PRICE_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"  [{ticker}] MISSING — skip")
            continue

        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        train_mask = df.index <= TRAIN_END

        print(f"  [{ticker}] {df.shape[0]} trading days | "
              f"train={train_mask.sum()} | test={(~train_mask).sum()}")

        # Group A
        price_feat = compute_price_features(df)

        # Group B
        fd_feat, d_opt = compute_fracdiff(df, train_mask)
        print(f"    fracdiff d_opt={d_opt:.2f}")

        # Merge macro onto trading days
        merged_macro = macro_all.reindex(df.index).ffill()

        # Combine
        frame = pd.concat([df, price_feat, fd_feat, merged_macro], axis=1)
        frame["ticker"] = ticker
        frame.index.name = "Date"
        all_frames.append(frame)

    print("\nCombining all tickers...")
    combined = pd.concat(all_frames)
    combined = combined.reset_index()

    # Reorder columns (Đã xóa các cột dư thừa)
    id_cols   = ["Date", "ticker"]
    ohlcv     = ["Open", "High", "Low", "Close", "Volume"]
    group_a   = [c for c in combined.columns if c in [
        "ret_1d","ret_5d","ret_20d","ret_60d",
        "vol_20d","vol_60d","vol_ratio",
        "macd","rsi_14",
        "bb_position","dist_52w_high",
        "volume_ratio","vwap_dev","hl_range","autocorr_1"
    ]]
    group_b   = ["close_fd"]
    macro_raw = ["usdvnd","vix","hsi_ret","ibor_rate","pmi","vnm_ret"]
    group_c   = [c for c in combined.columns if c in [
        "vix_dev","vix_stress",
        "usdvnd_mom5","usdvnd_dev",
        "vnm_ret_z","vnm_ret_streak",
        "ibor_dev","ibor_event",
        "pmi_norm","pmi_momentum",
        "hsi_ret_1d","hsi_vol_20",
        "risk_off_composite"
    ]]

    ordered_cols = id_cols + ohlcv + group_a + group_b + macro_raw + group_c
    # Thêm cột còn lại nếu có
    extra = [c for c in combined.columns if c not in ordered_cols]
    combined = combined[ordered_cols + extra]

    # Stats
    n_features = len(group_a) + len(group_b) + len(macro_raw) + len(group_c)
    print(f"\nShape: {combined.shape}")
    print(f"Tickers: {combined['ticker'].nunique()} | Features: {n_features}")
    print(f"Date range: {combined['Date'].min().date()} -> {combined['Date'].max().date()}")

    # NaN report (bỏ qua NaN đầu chuỗi do rolling windows)
    feat_cols = group_a + group_b + group_c
    nan_pct = combined[feat_cols].isna().mean() * 100
    high_nan = nan_pct[nan_pct > 5]
    if len(high_nan) > 0:
        print(f"\nCảnh báo: {len(high_nan)} features có NaN > 5%:")
        print(high_nan.round(1).to_string())
    else:
        print(f"\nNaN check OK — tất cả features < 5% NaN (chủ yếu đầu chuỗi)")

    # Save
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nĐã lưu: {OUT_PATH}")

    print("\n--- Sample (VCB, 5 rows) ---")
    sample = combined[combined["ticker"] == "VCB"].tail(5)
    display_cols = id_cols + ["Close"] + group_a[:4] + ["close_fd"] + macro_raw[:3]
    print(sample[display_cols].to_string())


if __name__ == "__main__":
    main()