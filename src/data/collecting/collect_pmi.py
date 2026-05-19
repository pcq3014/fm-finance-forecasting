"""
Thu thập Vietnam Manufacturing PMI — S&P Global / Caixin
=========================================================
Nguồn chính thức:
  [1] S&P Global (IHS Markit) — PMI gốc (có phí license)
      URL: https://www.spglobal.com/marketintelligence/en/mi/products/pmi.html

  [2] Investing.com — Re-publish miễn phí (download CSV thủ công)
      URL: https://www.investing.com/economic-calendar/manufacturing-pmi-1229
      Bước: Click vào biểu đồ → Download Data (CSV)
      Định dạng file: "Vietnam Manufacturing PMI.csv"

  [3] Trading Economics — Alternative miễn phí
      URL: https://tradingeconomics.com/vietnam/manufacturing-pmi
      Bước: Download → CSV

  [4] FRED (Federal Reserve Bank of St. Louis) — Nếu có series
      Có thể có series: "VMPMIM" (Vietnam Manufacturing PMI)
      URL: https://fred.stlouisfed.org (tìm kiếm "Vietnam PMI")

Phương pháp xử lý (theo Project Context):
  - PMI tháng T → chỉ dùng từ ngày 5 tháng T+1 (publication lag)
  - Forward-fill từ ngày 5 tháng T+1 đến ngày 4 tháng T+2
  - Lý do: S&P Global công bố PMI vào ngày làm việc đầu tháng sau

Ví dụ:
  PMI tháng 1/2024 (công bố ~1/2/2024) → dùng từ 5/2/2024 đến 4/3/2024
"""

import os
import io
import requests
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR  = ROOT_DIR / "data" / "macro" / "raw_inputs"
OUT_DIR  = ROOT_DIR / "data" / "macro"
os.makedirs(RAW_DIR, exist_ok=True)

START_DATE = "2013-01-02"
END_DATE   = "2025-12-31"

# ─── Source [4]: Thử FRED API (miễn phí, không cần key) ──────────────────────

FRED_SERIES_CANDIDATES = [
    "VNFMPMIM",   # Vietnam Factory/Manufacturing PMI (Markit)
    "VMPMIM",     # Alternative code
]
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def try_fetch_fred_pmi() -> pd.DataFrame | None:
    """
    Thử lấy PMI từ FRED (Federal Reserve Bank of St. Louis).
    Trả về None nếu series không tồn tại trên FRED.

    Ref: https://fred.stlouisfed.org — Search 'Vietnam Manufacturing PMI'
    """
    for series_id in FRED_SERIES_CANDIDATES:
        try:
            url = f"{FRED_BASE}?id={series_id}"
            print(f"[FRED] Đang thử series: {series_id} ...")
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
            df.columns = ["Date", "pmi"]
            df["pmi"] = pd.to_numeric(df["pmi"], errors="coerce")
            df = df.dropna(subset=["pmi"])
            if len(df) > 12:  # Ít nhất 1 năm dữ liệu
                print(f"  Tìm thấy {len(df)} records từ FRED series {series_id}")
                return df
        except Exception as e:
            print(f"  FRED series {series_id} thất bại: {e}")
    return None


def load_pmi_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load PMI từ CSV download thủ công (Investing.com / Trading Economics).

    Investing.com format:
        Date,Price,Open,High,Low,Change %
        "Oct 01, 2024","48.2",...

    Trading Economics format:
        Date,Value
        2024-10-01,48.2
    """
    df_raw = pd.read_csv(filepath)
    print(f"[PMI CSV] Columns: {df_raw.columns.tolist()}")
    print(df_raw.head(3).to_string())

    # Detect format
    date_col  = df_raw.columns[0]
    value_col = next(
        (c for c in df_raw.columns if c.lower() in ("price", "value", "pmi", "actual")),
        df_raw.columns[1]
    )

    df = df_raw[[date_col, value_col]].copy()
    df.columns = ["Date", "pmi"]

    # Parse dates — thử cả 2 format
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
    df["pmi"] = pd.to_numeric(df["pmi"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna().sort_values("Date")

    print(f"  Parsed {len(df)} records PMI từ {df['Date'].min()} đến {df['Date'].max()}")
    return df


def apply_publication_lag(pmi_monthly: pd.DataFrame,
                           start: str = START_DATE,
                           end: str = END_DATE) -> pd.DataFrame:
    """
    Áp dụng publication lag 5 ngày theo Project Context specification.

    Rule:
      PMI tháng T → available từ ngày 5 tháng T+1
      Forward-fill cho đến PMI tháng T+1 có hiệu lực

    Args:
        pmi_monthly: DataFrame với col 'Date' (ngày gốc trong tháng) và 'pmi'

    Returns:
        DataFrame daily với col 'pmi', index=Date
    """
    df = pmi_monthly.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Dời index sang ngày 5 của tháng kế tiếp (publication lag)
    new_idx = df.index.to_period("M").to_timestamp("M") + pd.Timedelta(days=5)
    # MonthEnd + 5 ngày = ngày 5 tháng sau (approx — dùng offset chuẩn)
    new_idx = pd.to_datetime(
        df.index.year.astype(str) + "-" +
        (df.index.month % 12 + 1).astype(str).str.zfill(2) + "-05"
    )
    # Xử lý tháng 12 → tháng 1 năm sau
    mask_dec = df.index.month == 12
    new_idx_list = []
    for orig_date in df.index:
        m = orig_date.month
        y = orig_date.year
        if m == 12:
            new_idx_list.append(pd.Timestamp(f"{y + 1}-01-05"))
        else:
            new_idx_list.append(pd.Timestamp(f"{y}-{m + 1:02d}-05"))

    df.index = pd.DatetimeIndex(new_idx_list)

    # Reindex sang daily calendar, ffill
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx)
    df["pmi"] = df["pmi"].ffill().bfill(limit=5)
    df.index.name = "Date"

    missing = df["pmi"].isna().sum()
    if missing > 0:
        print(f"  CẢNH BÁO: {missing} ngày còn thiếu PMI (đầu/cuối chuỗi)")

    return df


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  HƯỚNG DẪN LẤY PMI THỦ CÔNG (nếu auto-fetch thất bại)              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Nguồn 1 — Investing.com (KHUYẾN NGHỊ):                             ║
║  1. Vào: https://www.investing.com/economic-calendar/               ║
║           manufacturing-pmi-1229                                    ║
║  2. Click icon Download (mũi tên xuống) góc phải biểu đồ           ║
║  3. Chọn range 2013–2025, download CSV                              ║
║  4. Lưu vào: data/macro/raw_inputs/raw_pmi_investing.csv           ║
║                                                                      ║
║  Nguồn 2 — Trading Economics:                                        ║
║  1. Vào: https://tradingeconomics.com/vietnam/manufacturing-pmi     ║
║  2. Download → CSV                                                   ║
║  3. Lưu vào: data/macro/raw_inputs/raw_pmi_investing.csv           ║
║                                                                      ║
║  Format CSV yêu cầu (tối thiểu 2 cột):                              ║
║    Date        | pmi (hoặc Price/Value/Actual)                      ║
║    2013-01-01  | 51.5                                                ║
║    2013-02-01  | 50.8                                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def main():
    raw_manual_path = RAW_DIR / "raw_pmi_investing.csv"
    raw_auto_path   = RAW_DIR / "raw_pmi_spglobal.csv"
    out_path        = OUT_DIR / "pmi.csv"

    pmi_monthly = None

    # Bước 1: Thử FRED tự động
    pmi_monthly = try_fetch_fred_pmi()

    # Bước 2: Thử file thủ công
    if pmi_monthly is None:
        for candidate in [raw_manual_path, raw_auto_path]:
            if candidate.exists():
                print(f"[PMI] Đang load từ file thủ công: {candidate}")
                pmi_monthly = load_pmi_from_csv(str(candidate))
                break

    # Bước 3: Thất bại → hướng dẫn
    if pmi_monthly is None:
        print_manual_instructions()
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu PMI. "
            "Vui lòng download thủ công theo hướng dẫn trên "
            f"và lưu tại: {raw_manual_path}"
        )

    # Áp dụng publication lag
    pmi_daily = apply_publication_lag(pmi_monthly)

    # Lưu
    pmi_daily.to_csv(out_path, index_label="Date")
    print(f"\nĐã lưu PMI (daily, lagged): {out_path}")
    print(f"Shape: {pmi_daily.shape}")
    print(pmi_daily.head(10))


if __name__ == "__main__":
    main()
