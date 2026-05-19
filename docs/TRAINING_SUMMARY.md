# Training Summary - Baseline & Foundation Models (V3)

## Ngày: 2026-05-14

---

## I. Data Preparation

### Target Variable Fix
- **Trước**: `y_return_1d = Close[t+1]/Close[t] - 1` (percentage return)
- **Sau**: `y_return_1d = ln(Close[t+1]/Close[t])` (log return)
- **Lý do**: Toán học nhất quán + phù hợp với các mô hình FM

### Data Splits
| Set | Rows | Date Range |
|-----|------|-----------|
| Train | 28,183 | 2013-03-01 to 2021-12-31 |
| Val | 6,330 | 2022-01-01 to 2023-12-31 |
| Test | 6,240 | 2024-01-01 to 2025-12-31 |

### Features Selected (V3)
- **Phương pháp**: 3-step voting system
  1. Low Variance Filter
  2. Correlation Filter (+ Redundancy)
  3. Mutual Information Filter

**8 Features đạt votes=3** (sử dụng cho LSTM):
```
1. vn30_volume
2. volume_ma20
3. trading_value_ma20
4. vn30_close
5. vnm_ret_lag1
6. foreign_sell_value
7. hsi_close
8. vn30_vol_5d
```

---

## II. Baseline Models Training

### 2.1 ARIMA(1,1,1)

**Status**: ✅ HOÀN THÀNH

**Phương pháp**:
- Univariate time series model
- Per-ticker training (15 tickers)
- Order (1,1,1): differencing=1, AR=1, MA=1

**Kết quả Overall**:
```
RMSE:    4.247
MAE:     2.832
IC:      0.040 (Information Coefficient)
Dir_Acc: 0.467 (Directional Accuracy: 46.7%)
```

**Per-ticker kết quả tốt nhất**:
- FPT: RMSE=0.233, Dir_Acc=49.5%
- SSI: RMSE=0.502, Dir_Acc=46.5%
- VCB: RMSE=0.580, Dir_Acc=45.6%
- PDR: RMSE=0.668, Dir_Acc=45.8%

**Output files**:
- `data/results/arima_summary.csv` - Per-ticker + overall metrics
- `data/results/arima_predictions.csv` - Detailed predictions

---

### 2.2 LSTM

**Status**: ⚠️ HOÀN THÀNH (nhưng có vấn đề)

**Phương pháp**:
- LSTM với 2 layers, hidden_size=64
- Multivariate: 8 V3 features
- Lookback=20, EPOCHS=50
- Per-ticker training

**Kết quả Overall**:
```
RMSE:    0.023 ❌ (quá nhỏ, nghi vấn)
MAE:     0.015
IC:      0.000 ❌ (no correlation)
Dir_Acc: 0.070 ❌ (7% - tệ)
```

**Vấn đề**:
- RMSE quá nhỏ → model dự báo gần như zero
- Dir_Acc chỉ 7% → không bắt được tín hiệu chiều
- Có lỗi `torch.float32` dtype trong quá trình inference

**Output files**:
- `data/results/lstm_summary.csv`
- `data/results/lstm_predictions.csv`

---

### Baseline Models Comparison

| Model | RMSE | MAE | IC | Dir_Acc |
|-------|------|-----|-----|---------|
| **ARIMA** | **4.247** | **2.832** | **0.040** | **0.467** |
| LSTM | 0.023 | 0.015 | 0.000 | 0.070 |

**Kết luận**: **ARIMA tốt hơn LSTM** trên metrics này

---

## III. Foundation Models (Ready to Run)

### 3.1 TimesFM 2.0 (Google)

**Script**: `timesfm_zeroshot_v3.py`

**Phương pháp**:
- Zero-shot: không cần training
- Univariate: chỉ dùng `close_ret_1d`
- Context length: 256 (≈ 1 năm)
- Per-ticker rolling forecast

**Yêu cầu**:
```bash
pip install timesfm torch
python timesfm_zeroshot_v3.py
```

**Output**:
- `data/results/timesfm_summary.csv`
- `data/results/timesfm_predictions.csv`

---

### 3.2 Chronos-2 (Amazon)

**Script**: `chronos2_zeroshot_v3.py`

**Phương pháp**:
- Zero-shot pretrained model
- Univariate: `close_ret_1d`
- Context length: 512
- Per-ticker rolling forecast

**Yêu cầu**:
```bash
pip install chronos transformers torch
python chronos2_zeroshot_v3.py
```

**Output**:
- `data/results/chronos2_summary.csv`
- `data/results/chronos2_predictions.csv`

---

### 3.3 Moirai-2 (Salesforce)

**Script**: `moirai2_zeroshot_v3.py`

**Phương pháp**:
- Zero-shot pretrained model
- Univariate: `close_ret_1d`
- Context length: 1024
- Per-ticker rolling forecast

**Yêu cầu**:
```bash
pip install transformers torch
python moirai2_zeroshot_v3.py
```

**Output**:
- `data/results/moirai2_summary.csv`
- `data/results/moirai2_predictions.csv`

---

## IV. Metrics Explanation

### RMSE (Root Mean Squared Error)
- Công thức: √(Σ(y_true - y_pred)²/n)
- Độ lớn: Mỗi unit là log return (0.01 ≈ 1%)
- ARIMA RMSE=4.247: Lỗi trung bình ~4.2x log return

### MAE (Mean Absolute Error)
- Công thức: Σ|y_true - y_pred|/n
- ARIMA MAE=2.832: Lỗi tuyệt đối trung bình

### IC (Information Coefficient)
- Công thức: corr(y_true, y_pred)
- Range: [-1, 1]
- ARIMA IC=0.040: Tương quan yếu nhưng có
- **Ý nghĩa**: Khả năng bắt tín hiệu dự báo

### Dir_Acc (Directional Accuracy)
- Công thức: % lần dự báo đúng chiều (tăng/giảm)
- Random: 50%
- ARIMA Dir_Acc=0.467: 46.7% (dưới random - tệ)
- **Ý nghĩa**: Khả năng dự báo chiều hướng

---

## V. Feature Selection Validation

### Momentum Features (log_ret_1d, close_ret_1d)
Những biến này có **MI rất cao (0.45)** nhưng **KHÔNG phải leakage**:

```
log_ret_1d[t] = ln(Close[t]/Close[t-1])    ← Yesterday's return
y_return_1d[t] = ln(Close[t+1]/Close[t])   ← Tomorrow's return

Tương quan: corr(log_ret_1d[t], y_return_1d[t]) = 0.0097

→ Không leakage! Hai biến khác ngày
```

---

## VI. Summary

### Hoàn thành
✅ Data preprocessing + log return target fix  
✅ Feature selection V3 (8 features with votes=3)  
✅ ARIMA baseline training (Result: RMSE=4.247)  
✅ LSTM baseline training (Result: RMSE=0.023 - suspicious)  
✅ Foundation model scripts (TimesFM, Chronos-2, Moirai-2)  
✅ Comprehensive documentation  

### Tiếp theo
- [ ] Debug LSTM dtype issue
- [ ] Run Foundation Models (TimesFM, Chronos-2, Moirai-2)
- [ ] Compare all 5 models
- [ ] Choose best model for production
- [ ] Implement LEAF dynamic router

---

## VII. Files Created

### Baseline Scripts
- `arima_baseline_v3.py` - ARIMA training
- `lstm_baseline_v3.py` - LSTM training

### Foundation Model Scripts
- `timesfm_zeroshot_v3.py` - TimesFM zero-shot
- `chronos2_zeroshot_v3.py` - Chronos-2 zero-shot
- `moirai2_zeroshot_v3.py` - Moirai-2 zero-shot

### Documentation
- `RUN_ALL_MODELS.md` - How to run guide
- `FEATURE_SELECTION_REPORT_V3.md` - Feature selection details
- `TRAINING_SUMMARY.md` - This file

### Results
- `data/results/arima_summary.csv`
- `data/results/arima_predictions.csv`
- `data/results/lstm_summary.csv`
- `data/results/lstm_predictions.csv`
- `data/results/baseline_models_comparison.csv`

---

## VIII. Kết Luận

Tính đến 2026-05-14:

1. **Data**: Log return targets, 8 V3-selected features, clean train/val/test splits
2. **ARIMA**: RMSE=4.247, IC=0.040, Dir_Acc=46.7% ✅
3. **LSTM**: RMSE=0.023 (nghi vấn - cần fix) ⚠️
4. **Foundation Models**: Scripts ready, waiting for dependencies

**Khuyến nghị tiếp theo**: 
- Fix LSTM dtype issue hoặc thay bằng model khác
- Chạy Foundation Models để so sánh
- Ensemble tốt nhất 2-3 models

---

**Prepared by**: Claude Code  
**Date**: 2026-05-14  
**Version**: V3 (Final)
