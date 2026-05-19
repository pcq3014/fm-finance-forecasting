"""
Macro Variables Preprocessing Pipeline - Revised Version
Xử lý macro variables thành format sử dụng cho mô hình dự báo chứng khoán.

Các điểm chính:
1. Lấy lịch giao dịch VNINDEX làm lịch chuẩn.
2. Không dùng outer merge toàn bộ ngày lịch.
3. PMI dùng pmi_lag1 để tránh look-ahead bias.
4. VIX, VNM ETF, HSI dùng lag1 vì là biến quốc tế/khu vực.
5. StandardScaler chỉ fit trên train, transform val/test.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
from typing import Tuple, Dict


# =========================
# CONFIG
# =========================

DATA_DIR = Path("data/macro")
OUTPUT_DIR = Path("data/processed/macro")

# NOTE: Foreign investor data starts 2013-01-02, VNINDEX starts 2012-05-21
# Setting START_DATE = "2013-01-02" avoids 226-day gap where foreign=0
START_DATE = "2013-01-02"
END_DATE = "2025-12-31"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE = "2023-12-31"


# =========================
# 1. LOAD DATA
# =========================

def load_macro_data() -> dict:
    """Load all macro data files."""

    df_vnm = pd.read_csv(DATA_DIR / "vnm_etf.csv", parse_dates=["Date"])
    df_vix = pd.read_csv(DATA_DIR / "vix_daily.csv", parse_dates=["date"])
    df_hsi = pd.read_csv(DATA_DIR / "hsi_daily.csv", parse_dates=["date"])
    df_vnindex = pd.read_csv(DATA_DIR / "vnindex_2013_2026.csv", parse_dates=["time"])
    df_vn30 = pd.read_csv(DATA_DIR / "vn30_2013_2026.csv", parse_dates=["time"])
    df_ir = pd.read_csv(DATA_DIR / "interest_rate_processed.csv", parse_dates=["month_date"])
    df_pmi = pd.read_csv(DATA_DIR / "PMI_Vietnam_processed.csv", parse_dates=["month_date"])

    # Load foreign investor data
    df_foreign = pd.read_csv(DATA_DIR / "foreign.csv", encoding="utf-8")
    df_foreign = df_foreign.rename(columns={"Ngày": "date"})
    df_foreign["date"] = pd.to_datetime(df_foreign["date"], format="%d/%m/%Y", errors="coerce")

    return {
        "vnm": df_vnm,
        "vix": df_vix,
        "hsi": df_hsi,
        "vnindex": df_vnindex,
        "vn30": df_vn30,
        "ir": df_ir,
        "pmi": df_pmi,
        "foreign": df_foreign,
    }


# =========================
# 2. CLEAN BASIC COLUMNS
# =========================

def clean_number(x):
    """
    Làm sạch số có dấu phẩy, dấu %, dấu --.
    Dùng cho dữ liệu khối ngoại nếu file còn dạng string.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    x = x.replace(",", "")
    x = x.replace("%", "")

    if x in ["", "--", "nan", "None"]:
        return np.nan

    try:
        return float(x)
    except ValueError:
        return np.nan


def standardize_columns(data: dict) -> dict:
    """Chuẩn hóa tên cột ngày."""

    data["vnm"] = data["vnm"].rename(columns={"Date": "date"})
    data["vnindex"] = data["vnindex"].rename(columns={"time": "date"})
    data["vn30"] = data["vn30"].rename(columns={"time": "date"})

    # Đảm bảo date là datetime
    for key in ["vnm", "vix", "hsi", "vnindex", "vn30", "foreign"]:
        data[key]["date"] = pd.to_datetime(data[key]["date"], errors="coerce")

    data["ir"]["month_date"] = pd.to_datetime(data["ir"]["month_date"], errors="coerce")
    data["pmi"]["month_date"] = pd.to_datetime(data["pmi"]["month_date"], errors="coerce")

    return data


# =========================
# 3. MERGE DATA
# =========================

def merge_macro_data(data: dict) -> pd.DataFrame:
    """
    Merge tất cả macro datasets.
    Lấy lịch giao dịch VNINDEX làm lịch chuẩn.
    Không dùng outer merge để tránh sinh ngày không giao dịch.
    """

    data = standardize_columns(data)

    # =========================
    # Base: lịch giao dịch Việt Nam từ VNINDEX
    # =========================
    df = data["vnindex"][["date", "close", "volume"]].copy()
    df = df.rename(columns={
        "close": "vnindex_close",
        "volume": "vnindex_volume"
    })

    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date")

    df = df[
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ].copy()

    # =========================
    # Merge VN30 theo ngày giao dịch VN
    # =========================
    df_vn30 = data["vn30"][["date", "close", "volume"]].copy()
    df_vn30 = df_vn30.rename(columns={
        "close": "vn30_close",
        "volume": "vn30_volume"
    })
    df_vn30 = df_vn30.drop_duplicates(subset=["date"])

    df = df.merge(df_vn30, on="date", how="left")

    # =========================
    # Merge VNM ETF
    # =========================
    df_vnm = data["vnm"][["date", "vnm_ret", "vnm_close"]].copy()
    df_vnm = df_vnm.drop_duplicates(subset=["date"])

    df = df.merge(df_vnm, on="date", how="left")

    # =========================
    # Merge VIX
    # =========================
    df_vix = data["vix"][["date", "vix"]].copy()
    df_vix = df_vix.drop_duplicates(subset=["date"])

    df = df.merge(df_vix, on="date", how="left")

    # =========================
    # Merge HSI
    # =========================
    df_hsi = data["hsi"][["date", "hsi_close", "hsi_logret"]].copy()
    df_hsi = df_hsi.drop_duplicates(subset=["date"])

    df = df.merge(df_hsi, on="date", how="left")

    # =========================
    # Merge foreign investor data
    # =========================
    foreign_raw_cols = [
        "KLGD ròng", "GTGD ròng",
        "KL Mua", "GT Mua",
        "KL Bán", "GT Bán"
    ]

    df_foreign = data["foreign"][["date"] + foreign_raw_cols].copy()

    df_foreign = df_foreign.rename(columns={
        "KLGD ròng": "foreign_net_volume",
        "GTGD ròng": "foreign_net_value",
        "KL Mua": "foreign_buy_volume",
        "GT Mua": "foreign_buy_value",
        "KL Bán": "foreign_sell_volume",
        "GT Bán": "foreign_sell_value",
    })

    foreign_cols = [
        "foreign_net_volume", "foreign_net_value",
        "foreign_buy_volume", "foreign_buy_value",
        "foreign_sell_volume", "foreign_sell_value"
    ]

    for col in foreign_cols:
        df_foreign[col] = df_foreign[col].apply(clean_number)

    df_foreign = df_foreign.drop_duplicates(subset=["date"])

    df = df.merge(df_foreign, on="date", how="left")

    # =========================
    # Merge monthly macro variables
    # =========================
    df["month_date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Interest rate
    df_ir = data["ir"].copy()

    # Nếu chưa có interest_rate_lag1 thì tạo
    if "interest_rate_lag1" not in df_ir.columns:
        df_ir = df_ir.sort_values("month_date")
        df_ir["interest_rate_lag1"] = df_ir["interest_rate"].shift(1)

    df_ir["ir_change"] = df_ir["interest_rate"].diff()

    df = df.merge(
        df_ir[["month_date", "interest_rate", "interest_rate_lag1", "ir_change"]],
        on="month_date",
        how="left"
    )

    # PMI
    df_pmi = data["pmi"].copy()

    # Nếu chưa có pmi_lag1 thì tạo
    if "pmi_lag1" not in df_pmi.columns:
        df_pmi = df_pmi.sort_values("month_date")
        df_pmi["pmi_lag1"] = df_pmi["pmi"].shift(1)

    # Tính thay đổi PMI trên bản lag1, tránh dùng PMI hiện tại
    df_pmi["pmi_change_lag1"] = df_pmi["pmi_lag1"].diff()

    df = df.merge(
        df_pmi[["month_date", "pmi_lag1", "pmi_change_lag1"]],
        on="month_date",
        how="left"
    )

    df = df.rename(columns={
        "pmi_lag1": "pmi",
        "pmi_change_lag1": "pmi_change"
    })

    return df


# =========================
# 4. FEATURE ENGINEERING
# =========================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # =========================
    # Log returns cho thị trường Việt Nam
    # =========================
    df["vnindex_logret"] = np.log(df["vnindex_close"] / df["vnindex_close"].shift(1))
    df["vn30_logret"] = np.log(df["vn30_close"] / df["vn30_close"].shift(1))

    # =========================
    # Rolling volatility
    # =========================
    df["vnindex_vol_5d"] = df["vnindex_logret"].rolling(5).std()
    df["vn30_vol_5d"] = df["vn30_logret"].rolling(5).std()

    # =========================
    # Lag biến quốc tế / khu vực
    # An toàn hơn do lệch lịch và múi giờ
    # =========================
    df["vix_lag1"] = df["vix"].shift(1)
    df["vix_lag5"] = df["vix"].shift(5)

    df["vnm_ret_lag1"] = df["vnm_ret"].shift(1)
    df["hsi_logret_lag1"] = df["hsi_logret"].shift(1)

    # Có thể dùng log VIX nếu muốn giảm ảnh hưởng spike
    df["vix_log_lag1"] = np.log(df["vix_lag1"] + 1)

    # =========================
    # Volume
    # Không xóa volume = 0 vội, chỉ log transform
    # =========================
    df["vnindex_volume_log"] = np.log(df["vnindex_volume"] + 1)
    df["vn30_volume_log"] = np.log(df["vn30_volume"] + 1)

    # Volume ratio with epsilon to avoid division by zero
    vnindex_vol_ma = df["vnindex_volume"].rolling(20).mean() + 1e-6
    df["vnindex_volume_ratio_20d"] = df["vnindex_volume"] / vnindex_vol_ma

    vn30_vol_ma = df["vn30_volume"].rolling(20).mean() + 1e-6
    df["vn30_volume_ratio_20d"] = df["vn30_volume"] / vn30_vol_ma

    # =========================
    # Foreign investor features
    # NOTE: Foreign investor data starts 2013-01-02, while VNINDEX starts 2012-05-21.
    # Gap: 226 days (data not available). We fill with 0 (no foreign trading).
    # By setting START_DATE = "2013-01-02", we avoid this artificial gap.
    # =========================
    foreign_cols = [
        "foreign_net_volume", "foreign_net_value",
        "foreign_buy_volume", "foreign_buy_value",
        "foreign_sell_volume", "foreign_sell_value"
    ]

    for col in foreign_cols:
        df[col] = df[col].fillna(0)  # Fill missing with 0 = no trading activity

    total_foreign_value = df["foreign_buy_value"] + df["foreign_sell_value"]

    df["foreign_net_ratio"] = (
        df["foreign_net_value"] / (total_foreign_value + 1e-6)
    )

    df["foreign_buy_ratio"] = (
        df["foreign_buy_value"] / (total_foreign_value + 1e-6)
    )

    df["foreign_net_value_ma5"] = df["foreign_net_value"].rolling(5).mean()
    df["foreign_net_volume_ma5"] = df["foreign_net_volume"].rolling(5).mean()

    df["foreign_net_value_lag1"] = df["foreign_net_value"].shift(1)
    df["foreign_net_volume_lag1"] = df["foreign_net_volume"].shift(1)

    return df


# =========================
# 5. SPLIT AND SCALE
# =========================

def get_feature_cols() -> list:
    """Danh sách feature dùng cho mô hình."""

    feature_cols = [
        # Vietnam market
        "vnindex_logret",
        "vn30_logret",
        "vnindex_vol_5d",
        "vn30_vol_5d",

        # International / regional lagged variables
        "vix_lag1",
        "vix_lag5",
        "vix_log_lag1",
        "vnm_ret_lag1",
        "hsi_logret_lag1",

        # Macro monthly variables
        "interest_rate",
        "interest_rate_lag1",
        "ir_change",
        "pmi",
        "pmi_change",

        # Volume
        "vnindex_volume_log",
        "vn30_volume_log",
        "vnindex_volume_ratio_20d",
        "vn30_volume_ratio_20d",

        # Foreign flow
        "foreign_net_value",
        "foreign_net_volume",
        "foreign_net_ratio",
        "foreign_buy_ratio",
        "foreign_net_value_ma5",
        "foreign_net_volume_ma5",
        "foreign_net_value_lag1",
        "foreign_net_volume_lag1",
    ]

    return feature_cols


def prepare_dataset(
    df: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    train_end_date: str = TRAIN_END_DATE,
    val_end_date: str = VAL_END_DATE
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], StandardScaler]:
    """
    Final preparation:
    1. Filter date range.
    2. Drop NaN.
    3. Split train/val/test theo thời gian.
    4. Fit scaler trên train.
    5. Transform val/test.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    feature_cols = get_feature_cols()

    # Chỉ giữ các feature thật sự tồn tại
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Drop NaN sau khi tạo lag/rolling
    df = df.dropna(subset=feature_cols).copy()

    # Split theo thời gian
    train = df[df["date"] <= train_end_date].copy()

    val = df[
        (df["date"] > train_end_date) &
        (df["date"] <= val_end_date)
    ].copy()

    test = df[df["date"] > val_end_date].copy()

    # Fit scaler trên train
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    # Transform cả 3 tập
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])

    df_scaled = pd.concat(
        [train_scaled, val_scaled, test_scaled],
        axis=0
    ).sort_values("date").reset_index(drop=True)

    splits = {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
    }

    return df_scaled, splits, scaler


# =========================
# 6. MAIN
# =========================

def main():
    print("Loading macro data...")
    data = load_macro_data()

    print("Merging datasets...")
    df = merge_macro_data(data)
    print(f"After merge: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nAdding features...")
    df = add_features(df)
    print(f"After features: {df.shape}")

    print("\nPreparing final dataset...")
    df_final, splits, scaler = prepare_dataset(df)

    print(f"Final dataset: {df_final.shape}")
    print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")

    print("\nSplit sizes:")
    print(f"Train: {splits['train'].shape}")
    print(f"Val  : {splits['val'].shape}")
    print(f"Test : {splits['test'].shape}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(OUTPUT_DIR / "macro_processed_scaled_v2.csv", index=False, encoding="utf-8-sig")
    splits["train"].to_csv(OUTPUT_DIR / "macro_train_scaled_v2.csv", index=False, encoding="utf-8-sig")
    splits["val"].to_csv(OUTPUT_DIR / "macro_val_scaled_v2.csv", index=False, encoding="utf-8-sig")
    splits["test"].to_csv(OUTPUT_DIR / "macro_test_scaled_v2.csv", index=False, encoding="utf-8-sig")

    # Save unscaled full feature data too
    df.to_csv(OUTPUT_DIR / "macro_features_unscaled_v2.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "scaler_macro.pkl", "wb") as f:
        pickle.dump(scaler, f)

    feature_cols = get_feature_cols()

    print(f"\nSaved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("MISSING VALUES AFTER FINAL PROCESSING")
    print("=" * 60)
    print(df_final[feature_cols].isna().sum())

    print("\n" + "=" * 60)
    print("FEATURES INCLUDED")
    print("=" * 60)
    print(f"Total features: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS OF SCALED TRAIN FEATURES")
    print("=" * 60)
    print(splits["train"][feature_cols].describe())


if __name__ == "__main__":
    main()