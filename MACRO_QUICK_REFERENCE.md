# Macro Variables - Quick Reference

## 📊 Dataset Summary

**Final Shape**: 2,891 rows × 41 columns  
**Date Range**: 2013-03-01 to 2025-12-31  
**Features**: 26 (all stationary, scaled)

| Split | Rows | % | Period |
|-------|------|---|--------|
| Train | 1,994 | 69% | 2013-03 to 2021-12 |
| Val | 449 | 16% | 2022-01 to 2023-12 |
| Test | 448 | 15% | 2024-01 to 2025-12 |

---

## 📁 Output Files

```
data/processed/macro/
├─ macro_train_scaled_v2.csv        ← USE THIS for training
├─ macro_val_scaled_v2.csv          ← USE THIS for validation
├─ macro_test_scaled_v2.csv         ← USE THIS for testing
├─ macro_processed_scaled_v2.csv    ← Full dataset (all rows)
├─ macro_features_unscaled_v2.csv   ← Reference (unscaled)
└─ scaler_macro.pkl                 ← For inference
```

---

## 🎯 26 Features Breakdown

### Group 1: Vietnam Market (3)
- `vnindex_logret` - Log return
- `vn30_logret` - Large-cap log return
- `vnm_ret_lag1` - US-listed ETF (lagged)

### Group 2: Volatility (5)
- `vnindex_vol_5d` - 5-day volatility
- `vn30_vol_5d` - 5-day volatility
- `vix_lag1` - Previous day VIX
- `vix_lag5` - 5-day lagged VIX
- `vix_log_lag1` - Log(VIX) previous day

### Group 3: Regional (1)
- `hsi_logret_lag1` - HK index spillover

### Group 4: Macro (5)
- `interest_rate` - Current policy rate
- `interest_rate_lag1` - Previous month
- `ir_change` - Month-on-month change
- `pmi` - Manufacturing PMI (lagged)
- `pmi_change` - Month-on-month change

### Group 5: Volume (4)
- `vnindex_volume_log` - Log-scaled
- `vn30_volume_log` - Log-scaled
- `vnindex_volume_ratio_20d` - vs 20-day MA
- `vn30_volume_ratio_20d` - vs 20-day MA

### Group 6: Foreign Flows (8)
- `foreign_net_value` - Net value (tỷ VND)
- `foreign_net_volume` - Net volume
- `foreign_net_ratio` - % net
- `foreign_buy_ratio` - % buy
- `foreign_net_value_ma5` - 5-day smoothed
- `foreign_net_volume_ma5` - 5-day smoothed
- `foreign_net_value_lag1` - Previous day
- `foreign_net_volume_lag1` - Previous day

---

## 🔧 How to Use

### Load Data
```python
import pandas as pd

df_train = pd.read_csv('data/processed/macro/macro_train_scaled_v2.csv')
df_val = pd.read_csv('data/processed/macro/macro_val_scaled_v2.csv')
df_test = pd.read_csv('data/processed/macro/macro_test_scaled_v2.csv')

# Get features (26 columns, skip 'date')
features = [col for col in df_train.columns if col != 'date']
X_train = df_train[features].values  # (1994, 26)
X_val = df_val[features].values      # (449, 26)
X_test = df_test[features].values    # (448, 26)
```

### For Inference
```python
import pickle

# Load scaler
scaler = pickle.load(open('data/processed/macro/scaler_macro.pkl', 'rb'))

# Scale new data
new_data_scaled = scaler.transform([new_observation])

# Predict
prediction = model.predict(new_data_scaled)
```

---

## ✅ Quality Checks Passed

| Item | Status |
|------|--------|
| NaN values | 0 ✅ |
| Inf values | 0 ✅ |
| Look-ahead bias | ✅ |
| Data leakage | ✅ |
| Date alignment | ✅ |
| Temporal order | ✅ |
| Standardization | ✅ |

---

## ⚠️ Known Constraints

1. **Foreign data gap**: Starts 2013-01-02 (226 days after VNINDEX)
   - Solution: START_DATE = 2013-01-02 aligns everything

2. **Monthly variables** (interest_rate, PMI): Forward filled to daily
   - Safe: Values don't change within month

3. **Scaling**: StandardScaler fit on **train only**
   - Always use loaded scaler for val/test

4. **Volume ratio**: Added epsilon (1e-6) to prevent division by zero
   - Safe: No inf or nan in final data

---

## 📈 Feature Statistic (Train Set, Scaled)

All 26 features after StandardScaler:

```
Mean:     0.0 (by definition)
Std:      1.0 (by definition)
Min:      -6.37 (returns in crash)
Max:      25.69 (foreign flows outlier)
25%:      -0.43
50%:      0.07
75%:      0.54
```

---

## 🚀 Model Training Notes

```python
# Recommended approach:
from sklearn.preprocessing import StandardScaler

# ✅ Data is already scaled, so:
model = LSTM()  # or any model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100
)

# ✅ Evaluate on test set (no data leakage)
score = model.evaluate(X_test, y_test)

# ✅ For inference, use loaded scaler
new_scaled = scaler.transform([new_data])
prediction = model.predict(new_scaled)
```

---

## 📞 Issues Fixed

### Issue 1: Volume Ratio Edge Case ✅ FIXED
- **Problem**: Division by near-zero rolling mean
- **Fix**: Added epsilon (+1e-6) 
- **Result**: No inf/nan values

### Issue 2: Foreign Data Gap ✅ FIXED
- **Problem**: 226 days gap with VNINDEX
- **Fix**: START_DATE = 2013-01-02
- **Result**: Clean aligned data from start

---

## 📋 Checklist Before Using

- [ ] Load data from `macro_*_scaled_v2.csv` files
- [ ] Verify shape: Train (1994, 26), Val (449, 26), Test (448, 26)
- [ ] Check date ranges are in temporal order
- [ ] Confirm no NaN/Inf in features
- [ ] Use scaler_macro.pkl for inference
- [ ] Maintain feature order (26 features)
- [ ] Keep temporal order (no shuffling)

---

## 📞 Reference Documents

- **Full Report**: `MACRO_PROCESSING_FINAL_REPORT.md`
- **Processing Script**: `preprocess_macro.py` (v2)
- **Validation Script**: `validate_macro_data.py`
- **Hướng Dẫn Xử Lý**: `MACRO_PROCESSING_GUIDE.md`

---

**Status**: ✅ Ready to use  
**Last Updated**: 2026-05-14  
**Version**: v2 (Fixed)
