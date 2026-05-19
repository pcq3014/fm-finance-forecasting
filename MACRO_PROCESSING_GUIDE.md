# Hướng Dẫn Xử Lý Macro Variables

## Tổng Quan Dữ Liệu

| Variable | Tần suất | Khoảng thời gian | Shape | Đặc điểm |
|----------|---------|------------------|-------|----------|
| **vnm_etf** | Daily | 2013-01-02 to 2025-12-31 | (4747, 2) | Return (-11.3% to 12.3%), Close (9.02-20.2) |
| **vix** | Daily | 2013-01-02 to 2026-05-04 | (3354, 2) | Range: 9.1 - 82.7 (volatility spike) |
| **interest_rate** | Monthly | 2013-01 to 2026 (157 obs) | (157, 2) | Range: 4% - 9%, đã có lag1 |
| **pmi** | Monthly | 2013-01 to 2025-12 (156 obs) | (156, 2) | Range: 32.7 - 56.5, đã có lag1 |
| **hsi** | Daily | 2013-01-03 to 2026-05-04 | (3276, 2) | Close (14687-33154), LogReturn |
| **vnindex** | Daily | 2012-05-21 to 2026-05-12 | (3488, 5) | OHLCV, Close (375-1918) |
| **vn30** | Daily | 2012-05-16 to 2026-05-12 | (3488, 5) | OHLCV, Close (441-2111) |

---

## ⚠️ Các Vấn Đề Cần Xử Lý

### 1. **Frequency Mismatch** (Daily vs Monthly)
- **Daily**: vnm_etf, vix, hsi, vnindex, vn30
- **Monthly**: interest_rate, pmi (chỉ 156-157 quan sát)
- **Giải pháp**: 
  - Forward fill monthly vars từ ngày đầu tháng
  - Hoặc interpolate linear giữa các tháng
  - **Khuyến nghị**: Forward fill (giữ nguyên giá trị trong tháng)

### 2. **Date Range Không Liền Mạch**
- vnindex & vn30: Bắt đầu từ 2012-05 nhưng VNM_ETF từ 2013-01
- Interest_rate & PMI: Tới 2025/2026 nhưng chỉ monthly
- **Giải pháp**: Lựa chọn common date range:
  - **Option A** (Dùng hết dữ liệu): 2013-01-02 to 2025-12-31 (daily)
  - **Option B** (Đơn giản): 2013-01 to 2025-12 (monthly)

### 3. **Thang Đo Không Nhất Quán**
- VIX: 9 - 82
- Interest Rate: 4 - 9
- PMI: 32 - 56
- Prices: 375 - 2111
- Returns: -0.11 to 0.12
- Volume: 0 - 2.7e9

**Giải pháp**: Standardization trước khi đưa vào mô hình:
- `(x - mean) / std` (Z-score normalization)
- Hoặc `(x - min) / (max - min)` (Min-Max scaling) nếu muốn [0,1]

---

## 📋 Chiến Lược Xử Lý Từng Biến

### **A. Market Price Variables** (vnindex, vn30, vnm_etf, hsi)

**Dùng gì:**
- `close` price → Log-return: `log(close_t / close_{t-1})`
- Hoặc trực tiếp `close` price (model sẽ học differencing)

**Xử lý:**
```python
# Option 1: Log returns
df['vnindex_logret'] = np.log(df['vnindex_close'] / df['vnindex_close'].shift(1))

# Option 2: Simple returns
df['vnindex_ret'] = df['vnindex_close'].pct_change()

# Option 3: Keep both close price
# (Model sẽ tự handle stationarity via differencing/attention)
```

**Tại sao?**
- Log-returns là stationary (tốt cho modeling)
- vnm_etf đã có `vnm_ret` → dùng luôn
- hsi đã có `hsi_logret` → dùng luôn

**Thêm features:**
- 5-day rolling volatility: `df['vnindex_vol_5d'] = df['vnindex_ret'].rolling(5).std()`
- 20-day rolling mean (trend): `df['vnindex_ma_20'] = df['vnindex_close'].rolling(20).mean()`

---

### **B. Volatility Index** (VIX)

**Hiện trạng:**
- Có spike lên 82 (COVID năm 2020?)
- Thường 13-20, outlier 50+

**Xử lý:**
```python
# Option 1: Dùng nguyên giá trị (model học được spike)
# Option 2: Log transform để giảm outlier impact
df['vix_log'] = np.log(df['vix'] + 1)

# Option 3: Standardize nhưng giữ spike
df['vix_zscore'] = (df['vix'] - df['vix'].mean()) / df['vix'].std()

# Thêm lagged version
df['vix_lag1'] = df['vix'].shift(1)
df['vix_lag5'] = df['vix'].shift(5)
```

**Tại sao?**
- VIX là measure rủi ro → quan trọng cho volatility prediction
- Spike lớn là pattern quan trọng (không nên bỏ)

---

### **C. Macro Indicators** (Interest Rate, PMI)

**Hiện trạng:**
- Monthly frequency → 156-157 observations
- Đã có lag1 feature (tốt!)
- Interest_rate: 4-9%, PMI: 32-56

**Xử lý:**
```python
# Forward fill để matching daily dates
df_daily = pd.DataFrame(index=pd.date_range('2013-01-01', '2025-12-31', freq='D'))

df_ir_monthly = pd.read_csv('interest_rate_processed.csv')
df_ir_monthly['month_date'] = pd.to_datetime(df_ir_monthly['month_date'])
df_ir_monthly = df_ir_monthly.set_index('month_date')

# Forward fill
df_daily['interest_rate'] = df_ir_monthly['interest_rate'].reindex(df_daily.index, method='ffill')
df_daily['pmi'] = df_pmi_monthly['pmi'].reindex(df_daily.index, method='ffill')

# Thêm monthly change
df_daily['ir_change'] = df_daily['interest_rate'].diff()
df_daily['pmi_change'] = df_daily['pmi'].diff()
```

**Tại sao?**
- Forward fill giữ nguyên giá trị trong tháng (realistic)
- Monthly change (`diff`) capture policy changes
- Lagged version giúp avoid look-ahead bias

---

### **D. Volume** (vnindex, vn30)

**Hiện trạng:**
- Rất lớn: 0 - 2.7e9
- 0 volume có nghĩa là holiday hoặc data issue

**Xử lý:**
```python
# Remove zero volume records
df = df[df['volume'] > 0]

# Log-transform để reduce scale
df['vnindex_volume_log'] = np.log(df['vnindex_volume'] + 1)

# Volume as feature (volume changes)
df['vnindex_vol_ratio'] = df['vnindex_volume'] / df['vnindex_volume'].rolling(20).mean()
```

**Tại sao?**
- Volume = trading activity → có tương quan với returns
- Log-transform giúp normalize distribution

---

## 🎯 Recommended Processing Pipeline

### **Step 1: Prepare Individual Variables**

```python
import pandas as pd
import numpy as np

# Load data
df_vnm = pd.read_csv('data/macro/vnm_etf.csv', parse_dates=['Date'])
df_vix = pd.read_csv('data/macro/vix_daily.csv', parse_dates=['date'])
df_hsi = pd.read_csv('data/macro/hsi_daily.csv', parse_dates=['date'])
df_vnindex = pd.read_csv('data/macro/vnindex_2013_2026.csv', parse_dates=['time'])
df_vn30 = pd.read_csv('data/macro/vn30_2013_2026.csv', parse_dates=['time'])
df_ir = pd.read_csv('data/macro/interest_rate_processed.csv', parse_dates=['month_date'])
df_pmi = pd.read_csv('data/macro/PMI_Vietnam_processed.csv', parse_dates=['month_date'])

# Standardize column names
df_vnm.rename(columns={'Date': 'date'}, inplace=True)
df_hsi.rename(columns={'date': 'date'}, inplace=True)
df_vnindex.rename(columns={'time': 'date'}, inplace=True)
df_vn30.rename(columns={'time': 'date'}, inplace=True)
```

### **Step 2: Merge to Common Date Range**

```python
# Create common date index (daily)
date_range = pd.date_range('2013-01-02', '2025-12-31', freq='D')

# Merge all daily data
df = df_vnindex[['date', 'close']].rename(columns={'close': 'vnindex_close'})
df = df.merge(df_vn30[['date', 'close']].rename(columns={'close': 'vn30_close'}), on='date', how='outer')
df = df.merge(df_vnm[['date', 'vnm_ret', 'vnm_close']], on='date', how='outer')
df = df.merge(df_vix[['date', 'vix']], on='date', how='outer')
df = df.merge(df_hsi[['date', 'hsi_close', 'hsi_logret']], on='date', how='outer')

# Forward fill macro vars (monthly)
df_ir_daily = df_ir[['month_date', 'interest_rate']].rename(columns={'month_date': 'date'})
df_pmi_daily = df_pmi[['month_date', 'pmi']].rename(columns={'month_date': 'date'})

df = df.merge(df_ir_daily, on='date', how='outer')
df = df.merge(df_pmi_daily, on='date', how='outer')

# Forward fill monthly vars
df['interest_rate'] = df['interest_rate'].fillna(method='ffill')
df['pmi'] = df['pmi'].fillna(method='ffill')

# Sort by date
df = df.sort_values('date').reset_index(drop=True)
```

### **Step 3: Feature Engineering**

```python
# Log returns
df['vnindex_logret'] = np.log(df['vnindex_close'] / df['vnindex_close'].shift(1))
df['vn30_logret'] = np.log(df['vn30_close'] / df['vn30_close'].shift(1))

# Volatility (5-day rolling)
df['vnindex_vol_5d'] = df['vnindex_logret'].rolling(5).std()
df['vn30_vol_5d'] = df['vn30_logret'].rolling(5).std()

# Macro changes
df['ir_change'] = df['interest_rate'].diff()
df['pmi_change'] = df['pmi'].diff()

# VIX lags
df['vix_lag1'] = df['vix'].shift(1)
df['vix_lag5'] = df['vix'].shift(5)

# Drop first rows with NaN
df = df.dropna()
```

### **Step 4: Standardization (Before Model)**

```python
from sklearn.preprocessing import StandardScaler

# Select numeric columns to scale
cols_to_scale = ['vnindex_logret', 'vn30_logret', 'vnm_ret', 'hsi_logret',
                 'vix', 'interest_rate', 'pmi', 'vnindex_vol_5d', 'vn30_vol_5d',
                 'ir_change', 'pmi_change', 'vix_lag1', 'vix_lag5']

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save scaler for inference
import pickle
pickle.dump(scaler, open('scaler_macro.pkl', 'wb'))
```

---

## 📊 Tóm Tắt Các Features Được Khuyến Nghị

### **Input Features (13-15 features)**
1. `vnindex_logret` - Vietnam index return
2. `vn30_logret` - Large-cap index return  
3. `vnm_ret` - VNM ETF return
4. `hsi_logret` - Hong Kong index return (regional)
5. `vix` - Volatility index (risk sentiment)
6. `vnindex_vol_5d` - VNIndex volatility
7. `vn30_vol_5d` - VN30 volatility
8. `interest_rate` - Policy rate (macro)
9. `interest_rate_change` - Rate change
10. `pmi` - Economic activity (macro)
11. `pmi_change` - PMI change
12. `vix_lag1` - Previous day VIX
13. `vix_lag5` - 5-day lagged VIX

### **Why These Features?**
| Feature | Rationale |
|---------|-----------|
| Log-returns | Stationary, reduces outlier impact |
| Regional index (HSI) | Captures regional risk spillover |
| VIX | Global risk sentiment indicator |
| Volatility | Market uncertainty |
| Interest rate + change | Monetary policy impact |
| PMI | Economic growth expectations |
| Lagged VIX | Momentum/persistence in volatility |

---

## ⚠️ Common Pitfalls

1. **Forward filling macro data**: ✅ Safe (monthly values don't change within month)
2. **Not handling 0 volume**: ❌ Remove rows with 0 volume
3. **Using price levels directly**: ❌ Use returns (non-stationary)
4. **Forgetting standardization**: ❌ Different scales confuse the model
5. **Look-ahead bias**: ✅ Use lagged features for macro vars already in data

---

## 🔍 Next Steps

1. Implement preprocessing pipeline (code above)
2. Validate alignment across all dates
3. Check stationarity of features (ADF test)
4. Split into train/val/test by temporal order
5. Monitor for data leakage in feature engineering

