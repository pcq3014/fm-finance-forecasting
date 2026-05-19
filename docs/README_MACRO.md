# Macro Variables Processing - Complete Guide

## 📂 File Structure

```
d:\fm-forecasting\
├─ README_MACRO.md (THIS FILE)
├─ MACRO_COMPLETED.md ⭐ START HERE
├─ MACRO_QUICK_REFERENCE.md (Quick lookup)
├─ MACRO_PROCESSING_FINAL_REPORT.md (Detailed, 500+ lines)
├─ MACRO_PROCESSING_GUIDE.md (Original guide)
├─ preprocess_macro.py (Implementation, v2 fixed)
├─ validate_macro_data.py (Validation script)
├─ analyze_macro_vars.py (Analysis script)
└─ data/processed/macro/
   ├─ macro_train_scaled_v2.csv ⭐ USE FOR TRAINING
   ├─ macro_val_scaled_v2.csv ⭐ USE FOR VALIDATION
   ├─ macro_test_scaled_v2.csv ⭐ USE FOR TESTING
   ├─ macro_processed_scaled_v2.csv (Full dataset)
   ├─ macro_features_unscaled_v2.csv (Reference)
   └─ scaler_macro.pkl ⭐ USE FOR INFERENCE
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Read the Overview
📄 **MACRO_COMPLETED.md** (5 min read)
- What you have
- What was fixed
- Quick usage
- Next steps

### Step 2: Load Your Data
```python
import pandas as pd

df_train = pd.read_csv('data/processed/macro/macro_train_scaled_v2.csv')
df_val = pd.read_csv('data/processed/macro/macro_val_scaled_v2.csv')
df_test = pd.read_csv('data/processed/macro/macro_test_scaled_v2.csv')

features = [col for col in df_train.columns if col != 'date']
X_train = df_train[features].values  # (1994, 26)
X_val = df_val[features].values      # (449, 26)
X_test = df_test[features].values    # (448, 26)
```

### Step 3: Build Your Model
```python
# Data is ready to use!
# - Already standardized (mean=0, std=1)
# - All features stationary
# - No missing values
# - Proper temporal split

model = LSTM(...)  # Your model
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

---

## 📖 Documentation Map

### For Quick Users
- **MACRO_COMPLETED.md** - Everything you need to know (7 KB)
- **MACRO_QUICK_REFERENCE.md** - Feature list + tips (5 KB)

### For Detailed Understanding
- **MACRO_PROCESSING_FINAL_REPORT.md** - Full methodology (16 KB)
- **MACRO_PROCESSING_GUIDE.md** - Processing strategy (10 KB)

### For Implementation
- **preprocess_macro.py** - Well-commented code
- **validate_macro_data.py** - Quality checks
- **analyze_macro_vars.py** - Data analysis

---

## ✅ What's Included

### 26 Engineered Features

| Group | Features | Count |
|-------|----------|-------|
| **Vietnam Market** | vnindex_logret, vn30_logret, vnm_ret_lag1 | 3 |
| **Volatility** | vol_5d × 2, vix variants (3) | 5 |
| **Regional** | hsi_logret_lag1 | 1 |
| **Macro** | interest_rate variants (3), pmi variants (2) | 5 |
| **Volume** | volume_log × 2, volume_ratio_20d × 2 | 4 |
| **Foreign Flows** | net/buy values, ratios, MAs, lags | 8 |
| **TOTAL** | | **26** |

### 3 Ready-to-Use Datasets

```
Training:   macro_train_scaled_v2.csv    1,994 rows (69%)  [2013-2021]
Validation: macro_val_scaled_v2.csv        449 rows (16%)  [2022-2023]
Testing:    macro_test_scaled_v2.csv       448 rows (15%)  [2024-2025]
```

All data:
- ✅ Standardized (mean≈0, std≈1)
- ✅ Stationary (ready for time series)
- ✅ No missing values
- ✅ No look-ahead bias
- ✅ Temporal order preserved

---

## 🔧 What Was Fixed

### Issue #1: Volume Ratio Edge Case
```
Problem: Division by near-zero rolling mean
Fix:     Added epsilon (+1e-6) to denominator
Result:  ✅ 0 Inf/NaN values
```

### Issue #2: Foreign Data Gap
```
Problem: Foreign starts 2013-01-02, VNINDEX starts 2012-05-21
Fix:     SET START_DATE = 2013-01-02 to align
Result:  ✅ Clean aligned data from start
```

---

## 💻 Usage Examples

### Example 1: Train LSTM Model
```python
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df_train = pd.read_csv('data/processed/macro/macro_train_scaled_v2.csv')
df_val = pd.read_csv('data/processed/macro/macro_val_scaled_v2.csv')

features = [c for c in df_train.columns if c != 'date']
X_train = df_train[features].values
X_val = df_val[features].values

# Build model (data already scaled)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, 26)),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
```

### Example 2: Feature Analysis
```python
import shap
import matplotlib.pyplot as plt

# Analyze feature importance
explainer = shap.Explainer(model, X_val)
shap_values = explainer(X_val)

# Plot
shap.summary_plot(shap_values, X_val, feature_names=features)
```

### Example 3: Production Inference
```python
import pickle
import numpy as np

# Load scaler
scaler = pickle.load(open('data/processed/macro/scaler_macro.pkl', 'rb'))

# New observation (raw, unscaled)
new_data = np.array([[...26 features...]])

# Scale using loaded scaler
new_data_scaled = scaler.transform(new_data)

# Predict
forecast = model.predict(new_data_scaled)
```

---

## ⚙️ Technical Details

### Data Preparation Pipeline

```
Raw Data
    ↓
Load 8 sources (VNINDEX, VN30, VNM, VIX, HSI, Interest Rate, PMI, Foreign)
    ↓
Merge on VNINDEX calendar (left join)
    ↓
Feature Engineering
  ├─ Log returns (vnindex, vn30, hsi)
  ├─ Rolling volatility (5-day std)
  ├─ Lagged international vars (VIX, VNM, HSI)
  ├─ Macro policy vars (interest_rate, PMI with proper lags)
  ├─ Volume analysis (log + ratio)
  └─ Foreign investor flows (net, ratios, MA, lags)
    ↓
Drop NaN (rolling windows) → 2,891 rows
    ↓
StandardScaler.fit(train_data)
    ↓
Transform train/val/test
    ↓
Output: 26 standardized, stationary features ✅
```

### Feature Properties

All 26 features are:
- **Stationary**: Ready for time series models
- **Standardized**: mean≈0, std≈1
- **Scaled together**: Using single scaler
- **Properly lagged**: No look-ahead bias
- **Complete**: 0 missing values

### Data Splits

| Set | Rows | Percentage | Period | Purpose |
|-----|------|-----------|--------|---------|
| Train | 1,994 | 69% | 2013-03 to 2021-12 | Model training |
| Val | 449 | 16% | 2022-01 to 2023-12 | Hyperparameter tuning |
| Test | 448 | 15% | 2024-01-01 to 2025-12 | Final evaluation |

Temporal order preserved: Train → Val → Test (no shuffling)

---

## 🎯 Recommended Workflow

1. **Data Loading**
   ```python
   df_train = pd.read_csv('data/processed/macro/macro_train_scaled_v2.csv')
   df_val = pd.read_csv('data/processed/macro/macro_val_scaled_v2.csv')
   df_test = pd.read_csv('data/processed/macro/macro_test_scaled_v2.csv')
   ```

2. **Feature Extraction**
   ```python
   features = [c for c in df_train.columns if c != 'date']
   X_train, X_val, X_test = extract_features(df_train, df_val, df_test, features)
   ```

3. **Model Building**
   ```python
   model = build_your_model(input_dim=26)
   model.fit(X_train, y_train, validation_data=(X_val, y_val))
   ```

4. **Evaluation**
   ```python
   test_score = model.evaluate(X_test, y_test)
   predictions = model.predict(X_test)
   ```

5. **Production Deployment**
   ```python
   scaler = pickle.load(open('scaler_macro.pkl', 'rb'))
   new_scaled = scaler.transform(new_data)
   forecast = model.predict(new_scaled)
   ```

---

## ❓ FAQ

### Q: Can I use old files (without v2)?
A: Yes, but v2 has fixes. Recommend using v2 (epsilon protection + alignment).

### Q: Do I need to scale the data?
A: No! Data is already standardized. Don't re-scale.

### Q: Should I split data differently?
A: No. Temporal order is critical. Use provided train/val/test splits.

### Q: Can I refit the scaler?
A: No. Always use loaded scaler_macro.pkl for consistency.

### Q: Where is the target variable?
A: Not included in macro features. Merge with your target separately.

### Q: How do I handle new data?
A: Load scaler → transform → predict (see Example 3 above).

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Shape mismatch | Check you have 26 features in right order |
| Scaler error | Use loaded pickle, don't re-fit |
| NaN in output | Check input data before scaling |
| Data leakage | Never mix train/val/test |
| Low performance | Check feature importance, try different architecture |

---

## 📊 Data Summary

```
Total Observations: 2,891
Date Range: 2013-03-01 to 2025-12-31 (12.8 years)
Features: 26 (all stationary, standardized)

Feature Statistics (Scaled Train Set):
  Mean:     0.00 (by design)
  Std:      1.00 (by design)
  Min:     -6.37 (market crashes)
  Max:     25.69 (foreign flow outliers)
  
Quality:
  Missing: 0 ✅
  Inf:     0 ✅
  NaN:     0 ✅
  Duplicates: 0 ✅
```

---

## 🎓 Learning Resources

### For Understanding Features
- MACRO_PROCESSING_GUIDE.md: Why each feature matters
- MACRO_PROCESSING_FINAL_REPORT.md: Detailed methodology

### For Implementation
- preprocess_macro.py: See the code
- validate_macro_data.py: See quality checks

### For Usage
- MACRO_QUICK_REFERENCE.md: Feature lookup
- MACRO_COMPLETED.md: Quick start

---

## 📞 Support

### Documents
- **README_MACRO.md** ← You are here
- **MACRO_COMPLETED.md** ← Best overview
- **MACRO_QUICK_REFERENCE.md** ← Feature lookup
- **MACRO_PROCESSING_FINAL_REPORT.md** ← Full details

### Scripts
- **preprocess_macro.py** ← How it's done
- **validate_macro_data.py** ← Quality assurance

### Data
- **data/processed/macro/** ← All ready-to-use files

---

## ✨ Summary

You now have:
- ✅ 2,891 rows of clean macro data
- ✅ 26 engineered features (stationary, scaled)
- ✅ Proper train/val/test split
- ✅ Zero data quality issues
- ✅ Production-ready scaler
- ✅ Full documentation

**Next:** Load data and start building your forecasting model! 🚀

---

**Last Updated**: 2026-05-14  
**Version**: v2 (Fixed & Validated)  
**Status**: ✅ READY TO USE

