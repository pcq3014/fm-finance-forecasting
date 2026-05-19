# FM Forecasting — Session Context
> Cập nhật: 2026-05-16

---

## 1. Tổng quan dự án

Dự án so sánh Foundation Models (FM) với các baseline truyền thống trên bài toán dự báo lợi suất cổ phiếu VN30 (1 ngày tới).

- **Target**: `y_return_1d` — log return 1 ngày
- **Tickers**: 15 mã VN30 (ACB, BVH, CTG, FPT, GAS, HPG, KDH, MBB, MSN, PDR, SSI, STB, VCB, VIC, VNM)
- **Data split**:
  - Train: ~1877 ngày/ticker (~2013–2021)
  - Val: ~422 ngày/ticker (~2021–2022)
  - Test: ~416 ngày/ticker (2024-01-03 → 2025-12-30)
- **Panel data**: `data/processed/panel/panel_train_fixed.csv`, `panel_val_fixed.csv`, `panel_test_fixed.csv`

---

## 2. Models & Files

### Baselines
| Model | File chính | Output |
|---|---|---|
| ARIMA(1,0,1) | `arima_baseline_v3.py` | `data/results/arima_predictions.csv`, `arima_summary.csv` |
| LSTM univariate | `lstm_baseline_v3_fixed.py` | `data/results/lstm_predictions.csv`, `lstm_summary.csv` |
| LSTM multivariate | `lstm_baseline_v3.py` | `data/results/lstm_mv_predictions.csv`, `lstm_mv_summary.csv` |

### Foundation Models (zero-shot)
| Model | File chính | Output |
|---|---|---|
| TimesFM 1.0 | `timesfm_zeroshot_v3.py` | `data/results/timesfm_predictions.csv`, `timesfm_summary.csv` |
| Chronos-2 | `chronos2_zeroshot_fixed.py` | `data/results/chronos2_predictions_v3.csv`, `chronos2_summary_v3.csv` |
| Moirai-2 | `moirai2_from_notebook.py` | `data/results/moirai2_predictions_v3.csv`, `moirai2_summary_v3.csv` |

### Meta-learner
| Model | File chính | Output |
|---|---|---|
| LEAF | `leaf_ensemble.py` | `data/results/leaf_predictions.csv`, `leaf_summary.csv` |

---

## 3. Features đã thống nhất (tất cả multivariate models)

Các biến này đã được cập nhật đồng nhất trong **tất cả** FM và LSTM-MV:

```python
FEATURES = [
    'vn30_logret',       # thay vn30_close (price level → log return)
    'hsi_logret',        # thay hsi_close
    'vn30_volume_log',   # thay vn30_volume (raw → log)
    'volume_ratio_20d',  # thay volume_ma20
    'trading_value_log', # thay trading_value_ma20
    'foreign_net_ratio', # thay foreign_sell_value
    'vnm_ret_lag1',      # giữ nguyên
    'hsi_logret_lag1',   # thêm mới
]
```

**Lý do**: Các biến cũ (`vn30_close`, `hsi_close`, `vn30_volume`...) có scale lớn (hàng nghìn–triệu), gây scale mismatch khi feed vào FM. Chronos-2 cũ có IC = −0.085 do lỗi này. Tất cả biến mới đều stationary và return-scale (~0.01–0.05). **Các biến này đã có sẵn trong dataset**, không cần tính thêm.

---

## 4. Data split logic — chuẩn

### Chuẩn (theo ARIMA `arima_baseline_v3.py`):
```
Context để FM inference: TRAIN + VAL
Predict & evaluate:      TEST only
Rolling:                 dùng actual observed values, không pad zeros
```

### Các file đã fix để đồng nhất:
- `timesfm_zeroshot_v3.py` ✓ — sửa rolling dùng zeros → actual values; context = train+val
- `chronos2_zeroshot_fixed.py` ✓ — data split đúng, chỉ sửa features
- `moirai2_from_notebook.py` ✓ — sửa n_test dùng B-freq days (520) thay trading days (416); past_feat_dynamic_real=FEATURES; past_feat_dynamic_real_dim=len(FEATURES)
- `lstm_baseline_v3.py` ✓ — sửa features, output đổi thành `lstm_mv_*.csv`
- `lstm_baseline_v3_fixed.py` ✓ — đã đúng từ đầu

---

## 5. Kết quả hiện tại (full test set — 416 ngày)

| Model | RMSE | MAE | IC | Dir Acc |
|---|---|---|---|---|
| **ARIMA** | **0.02126** | **0.01359** | **+0.0324** | **0.4793** |
| LSTM-MV | 0.02128 | 0.01375 | −0.0028 | 0.4567 |
| Chronos-2 | 0.02175 | 0.01377 | −0.0557 | 0.4518 |
| TimesFM | 0.02287 | 0.01399 | −0.0566 | 0.4614 |
| Moirai-2 | 0.02446 | 0.01493 | −0.0711 | 0.3822 |

> **Lưu ý Moirai-2**: DA = 38.2% do bug n_test cũ (416 trading days vs 520 B-freq days) → 81 ngày đầu mỗi ticker có pred = 0. Đã sửa code, **cần chạy lại** `moirai2_from_notebook.py`.

**Nhận xét chính**: ARIMA beat tất cả FM zero-shot. FM có IC âm hệ thống vì VN30 return gần random walk và FM bị bias từ pretrain data (energy/retail/traffic). Đây là **finding nghiên cứu hợp lý**, không phải failure.

---

## 6. LEAF — Kiến trúc & Bugs đã fix

### Kiến trúc (trong `leaf_ensemble.py`)
```
60-day context window (8 features trên)
        ↓
RegimeEncoder: Linear(8→64) → LayerNorm → mean_pool → MLP(64→32) → z_regime(32)
        ↓
DynamicRouter: MLP(32→64→3) → softmax → weights (3,)
        ↓
weighted_sum(TimesFM_pred, Chronos2_pred, Moirai2_pred) → final forecast
```

**Novelty**: Dynamic regime-aware routing — weights thay đổi theo market context, không phải static ensemble.

### Bugs đã fix trong `leaf_ensemble.py`
1. ✓ `CONTEXT_FEATURES` — cập nhật 8 biến mới
2. ✓ Context window — xử lý per-ticker (`build_context_tensors`), tránh mix context giữa các mã
3. ✓ StandardScaler — fit trên leaf_train, transform cả train lẫn test
4. ✓ Regime labels — dùng `volume_ratio_20d` thay `vn30_vol_5d`

### LEAF training data — ĐÃ FIX

**Đúng** (đã implement):
```
FM predict VAL+TEST trong 1 lần rolling:
  train_context → predict VAL (422 bước) → tiếp tục → predict TEST (416 bước)
                       ↓                                      ↓
                val_predictions.csv              test_predictions.csv
LEAF:
  train LEAF trên val predictions (422 ngày thực sự out-of-sample)
  eval  LEAF trên test predictions (full 416 ngày = cùng period với ARIMA/LSTM)
```

**Files đã sửa**:
- `arima_baseline_v3.py` ✓ → `arima_val_predictions.csv` + `arima_predictions.csv`
- `timesfm_zeroshot_v3.py` ✓ → `timesfm_val_predictions.csv` + `timesfm_predictions.csv`
- `chronos2_zeroshot_fixed.py` ✓ → `chronos2_val_predictions.csv` + `chronos2_predictions_v3.csv`
- `moirai2_from_notebook.py` ⚠ cần sửa thêm (val predictions chưa có)
- `leaf_ensemble.py` ✓ → load val predictions làm train, test predictions làm eval

---

## 7. Metrics trên cùng period (Jun–Dec 2025 = last 30% test)

Để so sánh fair với LEAF eval period hiện tại:

| Model | RMSE | MAE | IC | DA |
|---|---|---|---|---|
| LEAF | 0.02361 | 0.01663 | +0.0523 | 0.4757 |
| TimesFM | 0.02373 | 0.01678 | +0.0545 | 0.4715 |
| ARIMA | 0.02366 | 0.01651 | +0.0412 | 0.4853 |
| Chronos-2 | 0.02370 | 0.01667 | +0.0293 | 0.4667 |
| LSTM-MV | 0.02381 | 0.01674 | −0.0242 | 0.4528 |

Trên period này FM đều IC dương — do H2 2025 là giai đoạn FM tình cờ tốt hơn. Không nên report riêng period này mà cần full test.

---

## 8. Output files hiện tại

```
data/results/
├── arima_predictions.csv          ✓ test predictions (đúng)
├── arima_summary.csv              ✓
├── lstm_predictions.csv           ✓ univariate
├── lstm_summary.csv               ✓
├── lstm_mv_predictions.csv        ✓ multivariate (features mới)
├── lstm_mv_summary.csv            ✓
├── timesfm_predictions.csv        ✓ test predictions (đã fix)
├── timesfm_summary.csv            ✓
├── chronos2_predictions_v3.csv    ✓ test predictions (features mới)
├── chronos2_summary_v3.csv        ✓
├── moirai2_predictions_v3.csv     ⚠ cần chạy lại (bug n_test đã fix)
├── moirai2_summary_v3.csv         ⚠ cần chạy lại
├── leaf_predictions.csv           ⚠ eval chỉ 30% test, cần fix training data
└── leaf_summary.csv               ⚠ cần fix training data
```

---

## 9. Todo tiếp theo

1. **[Cao] Sửa `moirai2_from_notebook.py` generate val predictions** — để LEAF có đủ 3 FM
2. **[Cao] Chạy lại `moirai2_from_notebook.py`** — bug n_test đã fix (DA=38.2% → ~45%)
3. **[Cao] Chạy lại tất cả FMs + LEAF** — có val predictions xong mới chạy LEAF được
4. **[Trung] Thêm LSTM univariate vào bảng so sánh cuối**
5. **[Trung] Verify Moirai-2 DA sau chạy lại** — kỳ vọng ~45–48%
