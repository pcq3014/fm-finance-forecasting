# Hướng dẫn Chạy Tất cả Baseline và Foundation Models

## Tóm tắt

Dự án này huấn luyện và so sánh các models dự báo lợi suất cổ phiếu VN30:

### Baselines (2 models)
- **ARIMA(1,1,1)** - Mô hình thời gian đơn biến
- **LSTM** - Mạng nơ-ron hồi quy với features V3

### Foundation Models (3 models - Zero-shot)
- **TimesFM 2.0** - Mô hình univariate từ Google
- **Chronos-2** - Mô hình univariate từ Amazon
- **Moirai-2** - Mô hình univariate từ Salesforce

---

## Dữ liệu

Tất cả models dùng:
- **Input data**: `data/processed/panel/panel_train/val/test_fixed.csv`
- **Target**: `y_return_1d` (log return)
- **Features (cho LSTM)**: V3 selected 8 features
  - vn30_volume, volume_ma20, trading_value_ma20, vn30_close
  - vnm_ret_lag1, foreign_sell_value, hsi_close, vn30_vol_5d

---

## Cách Chạy

### 1. ARIMA Baseline (Khuyên dùng - không cần dependencies phức tạp)

```bash
python arima_baseline_v3.py
```

**Output**:
- `data/results/arima_summary.csv` - Kết quả tổng hợp
- `data/results/arima_predictions.csv` - Dự báo chi tiết

**Kết quả mẫu** (Overall):
```
RMSE: 4.247 | MAE: 2.832 | IC: 0.040 | Dir_Acc: 0.467
```

---

### 2. LSTM Baseline (Cần PyTorch)

```bash
python lstm_baseline_v3.py
```

**Output**:
- `data/results/lstm_summary.csv`
- `data/results/lstm_predictions.csv`

**Yêu cầu**:
```bash
pip install torch numpy pandas scikit-learn
```

**Lưu ý**: LSTM hiện có vấn đề với dtype, RMSE rất nhỏ (0.022)

---

### 3. TimesFM Zero-shot (Optional - Cần GPU mạnh)

```bash
python timesfm_zeroshot_v3.py
```

**Yêu cầu**:
```bash
pip install timesfm torch
```

**Output**:
- `data/results/timesfm_summary.csv`
- `data/results/timesfm_predictions.csv`

---

### 4. Chronos-2 Zero-shot (Optional)

```bash
python chronos2_zeroshot_v3.py
```

**Yêu cầu**:
```bash
pip install chronos transformers torch
```

**Output**:
- `data/results/chronos2_summary.csv`
- `data/results/chronos2_predictions.csv`

---

### 5. Moirai-2 Zero-shot (Optional)

```bash
python moirai2_zeroshot_v3.py
```

**Yêu cầu**:
```bash
pip install transformers torch
```

**Output**:
- `data/results/moirai2_summary.csv`
- `data/results/moirai2_predictions.csv`

---

## Chạy Tất cả Cùng Lúc

### Tổ chức: Chạy Baselines trước

```bash
# Baselines
python arima_baseline_v3.py
python lstm_baseline_v3.py

# Foundation Models (nếu có dependencies)
python timesfm_zeroshot_v3.py
python chronos2_zeroshot_v3.py
python moirai2_zeroshot_v3.py
```

Sau đó chạy script so sánh:

```bash
python compare_all_models.py
```

---

## Kết Quả Hiện Tại

### Baseline Models Comparison (Overall)

| Model | RMSE | MAE | IC | Dir_Acc |
|-------|------|-----|-----|---------|
| **ARIMA(1,1,1)** | 4.247 | 2.832 | 0.040 | 0.467 |
| LSTM | 0.023 | 0.015 | 0.000 | 0.070 |

**Nhận xét**:
- ARIMA: IC=0.040 (tương quan yếu nhưng cao hơn LSTM), Dir_Acc=46.7%
- LSTM: Kết quả RMSE nhỏ phía nghi vấn (có lỗi dtype)

---

## Metrics Giải Thích

- **RMSE**: Root Mean Squared Error - Sai số bình phương trung bình
- **MAE**: Mean Absolute Error - Sai số tuyệt đối trung bình
- **IC** (Information Coefficient): Tương quan Pearson - Khả năng bắt tín hiệu
- **Dir_Acc** (Directional Accuracy): % dự đoán đúng chiều tăng/giảm

---

## Cấu trúc Output

```
data/results/
├── arima_summary.csv              # Tóm tắt ARIMA
├── arima_predictions.csv          # Chi tiết dự báo ARIMA
├── lstm_summary.csv               # Tóm tắt LSTM
├── lstm_predictions.csv           # Chi tiết dự báo LSTM
├── timesfm_summary.csv            # Tóm tắt TimesFM (nếu chạy)
├── timesfm_predictions.csv
├── chronos2_summary.csv           # Tóm tắt Chronos-2 (nếu chạy)
├── chronos2_predictions.csv
├── moirai2_summary.csv            # Tóm tắt Moirai-2 (nếu chạy)
├── moirai2_predictions.csv
└── baseline_models_comparison.csv # So sánh baselines
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'timesfm'"
→ Cài: `pip install timesfm`

### "ModuleNotFoundError: No module named 'chronos'"
→ Cài: `pip install chronos`

### "ModuleNotFoundError: No module named 'torch'"
→ Cài PyTorch: `pip install torch`

### Memory Error khi chạy Foundation Models
→ Giảm `per_core_batch_size` trong code (mặc định 32)

---

## Feature Selection V3 (Được sử dụng)

Báo cáo chi tiết: `FEATURE_SELECTION_REPORT_V3.md`

**8 Đặc trưng được chọn (votes=3)**:
1. vn30_volume - Khối lượng VN30
2. volume_ma20 - MA 20 ngày khối lượng
3. trading_value_ma20 - MA 20 ngày giá trị giao dịch
4. vn30_close - Giá đóng VN30
5. vnm_ret_lag1 - Lợi suất VNM ngày hôm trước
6. foreign_sell_value - Giá trị bán nước ngoài
7. hsi_close - Giá HSI
8. vn30_vol_5d - Biến động 5 ngày VN30

---

## Tiếp Theo

- [ ] Sửa lỗi LSTM dtype
- [ ] Chạy TimesFM, Chronos-2, Moirai-2
- [ ] So sánh tất cả 5 models
- [ ] Chọn mô hình tốt nhất để ensemble
- [ ] Tích hợp LEAF dynamic router

---

**Ngày cập nhật**: 2026-05-14  
**Phiên bản**: V3 (Final)
