"""
Thu thập VNM ETF (VanEck Vietnam ETF) — Proxy cho Foreign Investor Sentiment
=============================================================================
Nguồn: Yahoo Finance ticker "VNM" (NYSE)
URL: https://finance.yahoo.com/quote/VNM/history/

VNM ETF là quỹ ETF niêm yết tại Mỹ, nắm giữ rổ cổ phiếu Việt Nam (VCB, VHM, HPG...).
Dùng làm proxy cho foreign investor sentiment vì:
  - Khi foreign tăng mua VN equities → VNM ETF tăng giá/volume
  - Market-based measure, không cần scrape
  - Được chấp nhận trong EM finance literature

Output: data/macro/vnm_etf.csv
  - Columns: vnm_ret (log return), vnm_close (close price)
  - Daily index: 2013-01-02 -> 2025-12-31
  - Weekend/holiday: ffill (giống VIX)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR  = ROOT_DIR / "data" / "macro"
OUT_PATH = OUT_DIR / "vnm_etf.csv"

START = "2013-01-02"
END   = "2025-12-31"
FULL_IDX = pd.date_range(start=START, end=END, freq="D")


def fetch_vnm_yfinance() -> pd.DataFrame | None:
    """Lấy VNM ETF từ yfinance."""
    try:
        import yfinance as yf
        print("[VNM ETF] Dang tai tu yfinance...")
        ticker = yf.Ticker("VNM")
        df = ticker.history(start=START, end="2026-01-01", auto_adjust=True)
        if df.empty:
            print("  yfinance tra ve DataFrame rong.")
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        print(f"  yfinance: {len(df)} rows, tu {df.index.min().date()} den {df.index.max().date()}")
        return df[["Close", "Volume"]].rename(columns={"Close": "vnm_close", "Volume": "vnm_volume"})
    except ImportError:
        print("  yfinance chua cai. Chay: pip install yfinance")
        return None
    except Exception as e:
        print(f"  yfinance that bai: {e}")
        return None


def fetch_vnm_requests() -> pd.DataFrame | None:
    """Fallback: lay VNM ETF qua Yahoo Finance CSV download."""
    import requests
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/VNM"
        f"?period1=1356998400&period2=1767139200"  # 2013-01-02 -> 2026-01-01 (epoch)
        "&interval=1d&events=history&includeAdjustedClose=true"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        print("[VNM ETF] Thu fallback Yahoo Finance CSV API...")
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            return None
        import io
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Adj Close", "Volume"]].rename(
            columns={"Adj Close": "vnm_close", "Volume": "vnm_volume"}
        )
        df = df[df["vnm_close"] > 0].dropna(subset=["vnm_close"])
        print(f"  CSV API: {len(df)} rows, tu {df.index.min().date()} den {df.index.max().date()}")
        return df
    except Exception as e:
        print(f"  CSV API that bai: {e}")
        return None


def process_vnm(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tinh log return, reindex sang daily FULL_IDX, ffill weekend/holiday.
    """
    close = pd.to_numeric(df_raw["vnm_close"], errors="coerce")
    close = close.sort_index()

    # Log return (trading days only)
    log_ret = np.log(close / close.shift(1))

    # Reindex sang full daily calendar
    close_daily  = close.reindex(FULL_IDX).ffill().bfill()
    # Weekend/holiday -> 0 return (khong co giao dich)
    ret_daily    = log_ret.reindex(FULL_IDX).fillna(0)

    result = pd.DataFrame({
        "vnm_ret":   ret_daily,
        "vnm_close": close_daily,
    }, index=FULL_IDX)
    result.index.name = "Date"

    print(f"\n[VNM ETF] Sau reindex:")
    print(f"  vnm_ret  : missing={result['vnm_ret'].isna().sum()} | "
          f"range={result['vnm_ret'].min():.4f} -> {result['vnm_ret'].max():.4f}")
    print(f"  vnm_close: missing={result['vnm_close'].isna().sum()} | "
          f"range={result['vnm_close'].min():.2f} -> {result['vnm_close'].max():.2f}")
    return result


def main():
    print("=" * 60)
    print("COLLECT VNM ETF (VanEck Vietnam ETF)")
    print(f"Period: {START} -> {END}")
    print("=" * 60 + "\n")

    df_raw = fetch_vnm_yfinance()

    if df_raw is None:
        df_raw = fetch_vnm_requests()

    if df_raw is None:
        print("\nERROR: Khong lay duoc VNM ETF.")
        print("Kiem tra ket noi internet va thu lai sau.")
        return

    df = process_vnm(df_raw)

    # Validate coverage
    coverage_start = df_raw.index.min()
    expected_start = pd.Timestamp(START)
    if coverage_start > expected_start:
        gap_days = (coverage_start - expected_start).days
        print(f"\nWARNING: VNM ETF bat dau tu {coverage_start.date()} "
              f"(thieu {gap_days} ngay tu {START}).")
        print("  -> Da dung bfill de lap day dau chuoi.")

    # Save
    df.to_csv(OUT_PATH)
    print(f"\nDa luu: {OUT_PATH}")
    print(f"Shape: {df.shape}")

    print("\n--- Summary ---")
    print(df.describe().round(4).to_string())

    print("\n--- Last 5 rows ---")
    print(df.tail(5).to_string())


if __name__ == "__main__":
    main()
