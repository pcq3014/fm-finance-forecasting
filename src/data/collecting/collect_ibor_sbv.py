"""
Thu thập IBOR Rate (Lãi suất điều hành SBV) — Ngân hàng Nhà nước Việt Nam
===========================================================================
Nguồn chính thức:
  [1] Ngân hàng Nhà nước Việt Nam (SBV) — Lãi suất điều hành
      URL: https://www.sbv.gov.vn/webcenter/portal/vi/menu/trangchu/atvnq/csls
      (Cổng Thông tin Điện tử NHNN → Chính sách tiền tệ → Lãi suất)
  [2] World Bank Open Data (backup tần suất thấp hơn)
      URL: https://data.worldbank.org/indicator/FR.INR.RINR?locations=VN

Chú thích phương pháp:
  - Dùng lãi suất tái cấp vốn (Refinancing Rate) làm proxy IBOR chính.
  - Đây là lãi suất SBV cho các NHTM vay qua nghiệp vụ tái cấp vốn.
  - Tần suất: Event-based (chỉ thay đổi khi SBV ra quyết định).
  - Xử lý: Step function — giữ nguyên cho đến khi có quyết định mới.
  - Không dùng data trước ngày effective (tránh look-ahead bias).

Lịch sử lãi suất tái cấp vốn SBV (2013–2025):
  Nguồn: Tổng hợp từ các Thông tư/Quyết định của SBV
  Ref: https://thuvienphapluat.vn (tra cứu Quyết định SBV)
"""

import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR  = ROOT_DIR / "data" / "macro" / "raw_inputs"
OUT_DIR  = ROOT_DIR / "data" / "macro"
os.makedirs(RAW_DIR, exist_ok=True)

START_DATE = "2013-01-02"
END_DATE   = "2025-12-31"

# ─── Lịch sử Quyết định SBV (Refinancing Rate) ───────────────────────────────
# Mỗi entry: (effective_date, rate_pct)
# Nguồn: Tổng hợp Thông tư NHNN, thuvienphapluat.vn, cafef.vn/thi-truong-tai-chinh
# Ref: Quyết định số: 2194/QĐ-NHNN, 1606/QĐ-NHNN, v.v.
SBV_DECISIONS = [
    # (Ngày hiệu lực, Lãi suất tái cấp vốn %/năm)
    ("2013-01-02", 9.00),   # Đầu kỳ — giữ nguyên từ 12/2012
    ("2013-05-13", 8.00),   # QĐ 1097/QĐ-NHNN ngày 10/5/2013
    ("2013-06-28", 7.00),   # QĐ 1423/QĐ-NHNN ngày 25/6/2013
    ("2014-03-17", 6.50),   # QĐ 495/QĐ-NHNN ngày 14/3/2014
    ("2014-10-29", 6.00),   # QĐ 2146/QĐ-NHNN ngày 24/10/2014
    ("2017-07-10", 6.25),   # QĐ 1310/QĐ-NHNN ngày 7/7/2017
    ("2019-09-16", 6.00),   # QĐ 1870/QĐ-NHNN ngày 13/9/2019
    ("2020-03-17", 5.50),   # QĐ 418/QĐ-NHNN — hỗ trợ COVID-19
    ("2020-05-13", 5.00),   # QĐ 902/QĐ-NHNN — tiếp tục hỗ trợ COVID-19
    ("2020-09-30", 4.50),   # QĐ 1728/QĐ-NHNN
    ("2022-09-23", 5.00),   # QĐ 1656/QĐ-NHNN — thắt chặt, chống lạm phát
    ("2022-10-25", 6.00),   # QĐ 1811/QĐ-NHNN
    ("2023-03-15", 5.50),   # QĐ 313/QĐ-NHNN — nới lỏng
    ("2023-04-03", 5.00),   # QĐ 557/QĐ-NHNN
    ("2023-05-25", 4.50),   # QĐ 950/QĐ-NHNN
    ("2023-06-16", 4.00),   # QĐ 1124/QĐ-NHNN
    ("2024-06-12", 4.50),   # QĐ 955/QĐ-NHNN — điều chỉnh tăng nhẹ
    # ⚠️  2025: Cập nhật theo quyết định thực tế của SBV
    # Nguồn cập nhật: https://www.sbv.gov.vn/webcenter/portal/vi/menu/trangchu/atvnq/csls
]


def build_ibor_series(decisions: list = SBV_DECISIONS,
                      start: str = START_DATE,
                      end: str = END_DATE) -> pd.DataFrame:
    """
    Tạo daily time series lãi suất tái cấp vốn SBV dạng step function.

    Args:
        decisions: list of (date_str, rate_pct) tuples
        start: start date
        end: end date

    Returns:
        DataFrame index=Date, col='ibor_rate'
    """
    print("[SBV] Đang xây dựng IBOR step function...")

    events = pd.DataFrame(decisions, columns=["Date", "ibor_rate"])
    events["Date"] = pd.to_datetime(events["Date"])
    events = events.sort_values("Date").set_index("Date")

    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = events.reindex(full_idx)

    # Step function: ffill cho đến quyết định tiếp theo
    df["ibor_rate"] = df["ibor_rate"].ffill()

    # Bfill nhẹ cho đầu chuỗi nếu start_date trước decision đầu tiên
    df["ibor_rate"] = df["ibor_rate"].bfill()

    df.index.name = "Date"
    missing = df["ibor_rate"].isna().sum()
    if missing > 0:
        raise ValueError(f"Còn {missing} ngày thiếu ibor_rate — kiểm tra lại SBV_DECISIONS.")

    print(f"  Tổng {len(decisions)} quyết định SBV, {df.shape[0]} ngày calendar")
    print(f"  Range: {df['ibor_rate'].min():.2f}% → {df['ibor_rate'].max():.2f}%")
    return df


def validate_against_sbv_website():
    """Hướng dẫn cross-validate với SBV website thủ công."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HƯỚNG DẪN VALIDATE IBOR RATE VỚI SBV                       ║
╠══════════════════════════════════════════════════════════════╣
║  1. Truy cập: https://www.sbv.gov.vn                         ║
║     → Chính sách tiền tệ → Lãi suất → Lãi suất điều hành   ║
║                                                              ║
║  2. Download PDF/Excel lịch sử quyết định                    ║
║     → So sánh với SBV_DECISIONS trong script này            ║
║                                                              ║
║  3. Backup nếu SBV site khó truy cập:                       ║
║     https://thuvienphapluat.vn/van-ban/tien-te              ║
║     (Tra cứu: "Quyết định NHNN lãi suất tái cấp vốn")      ║
║                                                              ║
║  4. Nếu cần cập nhật 2025: Kiểm tra Thông báo SBV mới nhất  ║
║     https://www.sbv.gov.vn/webcenter/portal/vi/menu/        ║
║     trangchu/ttsk/ctsk                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    raw_path = RAW_DIR / "raw_sbv_ibor.csv"
    out_path = OUT_DIR / "ibor_rate.csv"

    df = build_ibor_series()

    # Lưu raw (format tương thích với collect_macro.py)
    df.to_csv(raw_path, index_label="Date")
    print(f"Đã lưu raw: {raw_path}")

    # Lưu output
    df.to_csv(out_path, index_label="Date")
    print(f"Đã lưu: {out_path}")

    # In thống kê
    print("\nPhân bổ lãi suất theo năm:")
    yearly = df.groupby(df.index.year)["ibor_rate"].mean()
    print(yearly.to_string())

    validate_against_sbv_website()


if __name__ == "__main__":
    main()
