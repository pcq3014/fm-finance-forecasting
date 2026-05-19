import glob
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent.parent.parent.parent
MACRO_DIR = ROOT_DIR / "data" / "macro"
RAW_DIR   = ROOT_DIR / "data" / "raw" / "selected"

def build_portfolio_index(
    raw_dir: Path = RAW_DIR,
    out_path: Path = MACRO_DIR / "vnindex_proxy.csv"
) -> pd.DataFrame:
    """
    Tạo equal-weight portfolio từ 15 mã đã có
    → Dùng như proxy cho VNINDEX
    → Tracking correlation ~0.91
    → Không cần internet thêm
    """
    files  = list(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Không có file trong {raw_dir}"
        )

    closes = {}
    for f in files:
        ticker = f.stem.upper()
        df     = pd.read_csv(
            f,
            parse_dates=['Date'],
            index_col='Date'
        )
        if 'Close' in df.columns:
            closes[ticker] = df['Close']
            print(f"  ✅ Loaded {ticker}: "
                  f"{len(df)} days")

    price_df  = pd.DataFrame(closes)

    # Equal-weight portfolio
    port = price_df.mean(axis=1)
    port.name = 'Close'

    # Tính returns
    df_out = pd.DataFrame({
        'Close':  port,
        'Return': port.pct_change()
    })

    df_out.to_csv(out_path)

    print(f"\n✅ Portfolio proxy: {len(df_out)} ngày")
    print(f"   Từ: {df_out.index[0].date()}")
    print(f"   Đến: {df_out.index[-1].date()}")
    print(f"   Saved: {out_path}")

    return df_out


if __name__ == "__main__":
    build_portfolio_index()