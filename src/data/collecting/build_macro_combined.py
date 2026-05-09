"""
Build macro_combined.csv -- Ket hop 6 Macro Features
=====================================================
Input files:
  data/macro/macro_vn.csv   -> usdvnd (can fix)
  data/macro/vix_daily.csv  -> vix
  data/macro/hsi_daily.csv  -> hsi_ret (log return)
  data/macro/ibor_rate.csv  -> ibor_rate
  data/macro/pmi.csv        -> pmi (monthly -> daily voi lag)
  data/macro/vnm_etf.csv    -> vnm_ret (proxy foreign investor sentiment)

Output: data/macro/macro_combined.csv
  - 6 columns: usdvnd, vix, hsi_ret, ibor_rate, pmi, vnm_ret
  - Daily index: 2013-01-02 -> 2025-12-31
  - Zero NaN

Note: vnm_ret (VanEck Vietnam ETF log return) duoc dung lam proxy cho
      foreign_net_flow vi khong co nguon daily free cho thi truong chung khoan.
      VNM ETF phan anh appetite cua nha dau tu nuoc ngoai voi VN equities.
      Ref: collect_vnm_etf.py
"""

import sys, os
import pandas as pd
import numpy as np
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MACRO    = ROOT_DIR / "data" / "macro"
OUT_PATH = MACRO / "macro_combined.csv"

START = "2013-01-02"
END   = "2025-12-31"
FULL_IDX = pd.date_range(start=START, end=END, freq="D")


# =============================================================
# 1. USD/VND
# =============================================================

def load_usdvnd() -> pd.Series:
    """
    Load va clean USD/VND tu macro_vn.csv.

    Van de:
      - 5 outliers (loi don vi: 21.xx thay vi 21xxx, hoac 3210/3211)
      - ~104 NaN/nam (weekend + holiday) -> ffill

    Nguon: macro_vn.csv (da co san)
    Cross-validate: Yahoo Finance USDVND=X -> corr > 0.999
    """
    df = pd.read_csv(MACRO / "macro_vn.csv", parse_dates=["Date"])
    df = df[["Date", "USD_VND"]].dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    s = pd.to_numeric(df["USD_VND"], errors="coerce")

    # Fix outliers: gia tri < 1000 (loi don vi, chia 1000 ra VND)
    mask_unit = s.between(20, 30)   # 20-30 thay vi 20000-30000
    s[mask_unit] = s[mask_unit] * 1000
    print(f"  [usdvnd] Fix don vi x1000: {mask_unit.sum()} rows")

    # Fix outliers: gia tri bat thuong (3210, 3211) -> interpolate
    mask_bad = s < 5000
    s[mask_bad] = np.nan
    print(f"  [usdvnd] Xoa outlier bat thuong: {mask_bad.sum()} rows -> se interpolate")

    # Reindex + interpolate + ffill
    s = s.reindex(FULL_IDX)
    s = s.interpolate(method="time", limit=5)
    s = s.ffill().bfill()
    s.name = "usdvnd"

    missing = s.isna().sum()
    print(f"  [usdvnd] Final missing: {missing} | range: {s.min():.0f}-{s.max():.0f}")
    return s


# =============================================================
# 2. VIX
# =============================================================

def load_vix() -> pd.Series:
    """
    Load VIX tu vix_daily.csv.
    Nguon: Yahoo Finance ^VIX (da duoc thu thap)
    Xu ly: ffill weekend/holiday
    """
    df = pd.read_csv(MACRO / "vix_daily.csv")
    dc = [c for c in df.columns if "date" in c.lower()][0]
    df[dc] = pd.to_datetime(df[dc])
    df = df.set_index(dc).sort_index()

    vc = [c for c in df.columns if "vix" in c.lower()][0]
    s = pd.to_numeric(df[vc], errors="coerce")
    s = s.reindex(FULL_IDX).ffill().bfill()
    s.name = "vix"

    print(f"  [vix] missing={s.isna().sum()} | range: {s.min():.2f}-{s.max():.2f}")
    return s


# =============================================================
# 3. HSI Return
# =============================================================

def load_hsi_ret() -> pd.Series:
    """
    Load HSI return tu hsi_daily.csv.
    Nguon: Yahoo Finance ^HSI (da duoc thu thap)
    Dung log return (hsi_logret) neu co, neu khong tinh tu close.
    Xu ly: ffill weekend/holiday
    """
    df = pd.read_csv(MACRO / "hsi_daily.csv")
    dc = [c for c in df.columns if "date" in c.lower()][0]
    df[dc] = pd.to_datetime(df[dc])
    df = df.set_index(dc).sort_index()

    # Uu tien log return
    if "hsi_logret" in df.columns:
        s = pd.to_numeric(df["hsi_logret"], errors="coerce")
        print(f"  [hsi_ret] Dung hsi_logret")
    elif "hsi_close" in df.columns:
        close = pd.to_numeric(df["hsi_close"], errors="coerce")
        s = np.log(close / close.shift(1))
        print(f"  [hsi_ret] Tinh log return tu hsi_close")
    else:
        # Fallback: cot dau tien sau date
        vc = [c for c in df.columns][0]
        s = pd.to_numeric(df[vc], errors="coerce").pct_change()

    s = s.reindex(FULL_IDX).fillna(0)  # Weekend/holiday -> 0 return
    s.name = "hsi_ret"

    print(f"  [hsi_ret] missing={s.isna().sum()} | range: {s.min():.4f}-{s.max():.4f}")
    return s


# =============================================================
# 4. IBOR Rate
# =============================================================

def load_ibor() -> pd.Series:
    """
    Load IBOR rate (SBV refinancing rate) tu ibor_rate.csv.
    Nguon: SBV decisions (hardcoded trong collect_ibor_sbv.py)
    Da la step function, khong can xu ly them.
    """
    df = pd.read_csv(MACRO / "ibor_rate.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    s = pd.to_numeric(df["ibor_rate"], errors="coerce")
    s = s.reindex(FULL_IDX).ffill().bfill()
    s.name = "ibor_rate"

    print(f"  [ibor_rate] missing={s.isna().sum()} | range: {s.min():.2f}%-{s.max():.2f}%")
    return s


# =============================================================
# 5. PMI (Monthly -> Daily with publication lag)
# =============================================================

def load_pmi() -> pd.Series:
    """
    Load PMI va convert sang daily voi publication lag 5 ngay.

    Rule (tu Project Context):
      PMI thang T -> chi dung tu ngay 5 thang T+1
      (S&P Global cong bo PMI vao dau thang sau)
      Forward-fill cho den PMI ke tiep co hieu luc.

    Nguon: pmi.csv (da co san, 156 thang 2013-2025)
    """
    df = pd.read_csv(MACRO / "pmi.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    vc = [c for c in df.columns if "pmi" in c.lower()][0]
    s = pd.to_numeric(df[vc], errors="coerce").dropna()

    # Shift index sang ngay 5 thang ke tiep (publication lag)
    new_dates = []
    for dt in s.index:
        m, y = dt.month, dt.year
        if m == 12:
            new_dates.append(pd.Timestamp(f"{y+1}-01-05"))
        else:
            new_dates.append(pd.Timestamp(f"{y}-{m+1:02d}-05"))

    s.index = pd.DatetimeIndex(new_dates)
    s = s[~s.index.duplicated(keep="last")]

    # Reindex daily + ffill
    s = s.reindex(FULL_IDX).ffill()
    # Backfill nhe dau chuan (PMI thang 1/2013 phat hanh 5/2/2013, truoc do dung bfill)
    s = s.bfill(limit=40)
    s.name = "pmi"

    missing = s.isna().sum()
    print(f"  [pmi] missing={missing} | range: {s.min():.1f}-{s.max():.1f} | mean={s.mean():.1f}")
    return s


# =============================================================
# 6. VNM ETF Return (Proxy for Foreign Investor Sentiment)
# =============================================================

def load_vnm_ret() -> pd.Series:
    """
    Load VNM ETF log return tu vnm_etf.csv.
    Nguon: Yahoo Finance "VNM" (VanEck Vietnam ETF, NYSE)
    Proxy cho foreign_net_flow: phan anh appetite nha dau tu nuoc ngoai voi VN equities.
    Xu ly: weekend/holiday da duoc fill = 0 trong collect_vnm_etf.py
    """
    path = MACRO / "vnm_etf.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {path}. "
            "Chay truoc: python src/data/collect_vnm_etf.py"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    s = pd.to_numeric(df["vnm_ret"], errors="coerce")
    s = s.reindex(FULL_IDX).fillna(0)
    s.name = "vnm_ret"

    print(f"  [vnm_ret] missing={s.isna().sum()} | range: {s.min():.4f}-{s.max():.4f}")
    return s


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("BUILD MACRO COMBINED (6 features)")
    print(f"Period: {START} -> {END}")
    print("=" * 60 + "\n")

    series = {}

    print("[1/6] Loading USD/VND...")
    series["usdvnd"] = load_usdvnd()

    print("\n[2/6] Loading VIX...")
    series["vix"] = load_vix()

    print("\n[3/6] Loading HSI Return...")
    series["hsi_ret"] = load_hsi_ret()

    print("\n[4/6] Loading IBOR Rate...")
    series["ibor_rate"] = load_ibor()

    print("\n[5/6] Loading PMI (monthly -> daily)...")
    series["pmi"] = load_pmi()

    print("\n[6/6] Loading VNM ETF Return (foreign sentiment proxy)...")
    series["vnm_ret"] = load_vnm_ret()

    print("\n" + "-" * 60)
    print("Combining all features...")
    macro = pd.concat(series.values(), axis=1)
    macro.index.name = "Date"
    macro = macro.loc[START:END]

    # Validation
    missing = macro.isna().sum()
    print(f"\nMissing values per column:")
    print(missing.to_string())

    if missing.sum() > 0:
        print(f"\nWarning: {missing.sum()} NaN remaining. Attempting final ffill/bfill...")
        macro = macro.ffill().bfill()
        still_missing = macro.isna().sum().sum()
        if still_missing > 0:
            print(f"ERROR: Still {still_missing} NaN after all fills!")
            return

    # Save
    macro.to_csv(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Shape: {macro.shape}")

    print("\n--- Summary Statistics ---")
    print(macro.describe().round(3).to_string())

    print("\n--- Last 5 rows ---")
    print(macro.tail(5).to_string())

    print("\n--- NOTE ---")
    print("vnm_ret: VanEck Vietnam ETF (NYSE:VNM) log return.")
    print("Dung lam proxy cho foreign_net_flow (khong co nguon daily free).")
    print("Academically: 'market-based measure of foreign investor sentiment toward VN equities'.")


if __name__ == "__main__":
    main()
