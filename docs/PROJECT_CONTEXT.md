# PROJECT CONTEXT — VN30 Foundation Model Research
> Paste document này vào đầu mỗi conversation mới với bất kỳ AI nào
> Update phần "Current Status" mỗi ngày

---

## 1. RESEARCH OVERVIEW

### Đề tài gốc
"Ứng dụng Foundation Model trong dữ liệu tài chính chứng khoán"

### Research Question chính
> "Tại sao Time Series Foundation Model global fail trên thị trường
> chứng khoán Việt Nam (VN30), và liệu Causal Macro Conditioning
> có fix được không?"

### Tên paper đề xuất
"Why Do Time Series Foundation Models Fail on Emerging Market Equity?
Evidence from Vietnam and a Causal Macro-Conditioning Framework"

### Contribution chính (theo thứ tự quan trọng)
1. **VN30-MacroTS Dataset**: Dataset duy nhất capture Frontier →
   Emerging Market transition (FTSE upgrade 10/2025)
2. **Empirical finding**: FM fail có hệ thống trên VN30 trong stress
   regime vì macro signals (không phải price) drive market giai đoạn đó
3. **LEAF Framework**: Learned Economic Adaptive Forecasting —
   ~100K trainable params, causal macro conditioning cho frozen FM ensemble
4. **FTSE Upgrade Natural Experiment**: Pre/post upgrade analysis
   — không paper nào trên thế giới có thể replicate

---

## 2. DATASET — VN30-MacroTS

### Asset Universe
- **Nguồn**: VN30 index composition tháng 7/2022 (HoSE official)
- **Lý do chọn 7/2022**: Ngay trước test period (7/2022), không
  look-ahead bias, pre-bond crisis, pre-FTSE upgrade baseline
- **Universe ban đầu**: 30 mã VN30

### Exclusion Criteria
```
⚠️ QUYẾT ĐỊNH ĐÃ LOCK: Chỉ dùng 3 tiêu chí E1, E2, E3
   E4 (ADF stationarity) đã BỎ vì:
   → Returns của mã pass E1+E2+E3 gần như luôn stationary
   → E4 redundant, không thêm value
   → Thay thế bằng CUSUM nếu cần (nhưng thuộc Section 5)

E1: Lịch sử < 2013-01-02 → loại (IPO muộn)
    Tolerance: 10 ngày (Tết, lễ đầu năm)
    Lý do: Cần balanced panel, tất cả mã thấy cùng market history

E2: Thanh khoản thấp → loại
    Threshold: Adaptive P20 của universe (KHÔNG dùng ngưỡng cứng 30B)
    Lý do: Thị trường 2013 thanh khoản thấp hơn 2022 rất nhiều
           → Ngưỡng cứng 30B loại hầu hết mã (chỉ còn 2 mã)
           → Adaptive P20 = loại 20% mã kém nhất trong universe

E3: Missing data > 15% HOẶC consecutive missing > 30 ngày → loại
    Lý do: Proxy cho suspension, corporate event bất thường
```

### 15 Mã đang sử dụng (candidate, chưa verify final)
```
ACB  - Banking     (Ngân hàng Á Châu)
BVH  - Insurance   (Tập đoàn Bảo Việt)
CTG  - Banking     (VietinBank)
MBB  - Banking     (MB Bank)
STB  - Banking     (Sacombank)
VCB  - Banking     (Vietcombank)
SSI  - Securities  (Chứng khoán SSI)
KDH  - RealEstate  (Khang Điền)
PDR  - RealEstate  (Phát Đạt)
VIC  - RealEstate  (Vingroup)
MSN  - Consumer    (Masan Group)
VNM  - Consumer    (Vinamilk)
HPG  - Materials   (Hòa Phát Group)
GAS  - Energy      (PV Gas)
FPT  - Technology  (FPT Corporation)
```

⚠️ Lưu ý quan trọng:
- PDR, KDH: rủi ro fail E2 (thanh khoản thấp)
- STB: rủi ro có structural break (tái cơ cấu 2015-2019)
- Số mã final phụ thuộc vào kết quả exclusion pipeline thực tế
- Nếu < 10 mã pass → cần xem xét lại ngưỡng E2

### Sector Distribution (dự kiến)
```
Banking:    ACB, CTG, MBB, STB, VCB  (5 mã, ~33%)
RealEstate: KDH, PDR, VIC            (3 mã, ~20%)
Consumer:   MSN, VNM                  (2 mã, ~13%)
Securities: SSI                       (1 mã,  ~7%)
Materials:  HPG                       (1 mã,  ~7%)
Energy:     GAS                       (1 mã,  ~7%)
Technology: FPT                       (1 mã,  ~7%)
Insurance:  BVH                       (1 mã,  ~7%)

Portfolio tracking correlation với VN30: ~0.91
R-squared: ~0.83 → 15 mã giải thích 83% variance VN30
```

### Data Splits
```
⚠️ QUYẾT ĐỊNH ĐÃ LOCK:

Train:  2013-01-02 → 2022-06-30  (~2,340 ngày, 74%)
        Regime: Bull + COVID + Recovery + Early tightening
        → Đủ diverse regime để train LEAF

Test:   2022-07-01 → 2025-12-31  (~840 ngày, 26%)
        Regime: Bond crisis + Post-crisis + FTSE upgrade
        → Chạy DUY NHẤT 1 lần, không tune

KHÔNG có Val set riêng biệt — Val nhúng trong Walk-forward:
→ Walk-forward tạo 5 val sets động trong Train
→ Robust hơn 1 val set cố định
→ Hyperparameter tốt cho MỌI regime, không bias 1 giai đoạn

Walk-forward folds (trong Train):
  Fold 1: Train [2013→2016] | Val [2017]
  Fold 2: Train [2013→2017] | Val [2018]
  Fold 3: Train [2013→2018] | Val [2019]  ← Bull bình thường
  Fold 4: Train [2013→2019] | Val [2020]  ← COVID crash
  Fold 5: Train [2013→2020] | Val [2021]  ← Recovery

Test sub-periods (CHỈ để analyze, KHÔNG tune):
  Test A: 2022-07 → 2022-12  (Bond crisis)
  Test B: 2023-01 → 2024-12  (Post-crisis recovery)
  Test C: 2025-01 → 2025-12  (FTSE upgrade period)
```

### Price Data
- **Nguồn**: VNDirect Chart API (dchart-api.vndirect.com.vn)
- **Adjusted prices**: Đã điều chỉnh cổ tức và chia tách
- **Cross-validated**: Với Yahoo Finance (correlation > 0.999)
- **Lưu tại**: `data/raw/selected/[TICKER].csv`
- **Format**: Date, Open, High, Low, Close, Volume

### Macro Features (6 features)
```
Feature           Nguồn                    Tần suất gốc  Xử lý
──────────────────────────────────────────────────────────────────
vnm_ret           Yahoo Finance (VNM ETF)  Daily (NYSE)  ffill + ret=0 holiday
usdvnd            Yahoo (USDVND=X)         Daily         ffill weekend
ibor_rate         SBV decisions            Event-based   Step function
vix               Yahoo (^VIX)             Daily         ffill holiday
hsi_ret           Yahoo (^HSI)             Daily         ffill holiday
pmi               S&P Global/FRED          Monthly       ffill + pub lag 5d

Note về vnm_ret:
→ foreign_net_flow gốc không lấy được (VNDirect API blocked, HoSE không có daily feed free)
→ Thay bằng VanEck Vietnam ETF (NYSE:VNM) log return làm proxy
→ Justification: "market-based measure of foreign investor sentiment toward VN equities"
→ Được chấp nhận trong EM finance literature
→ Script: src/data/collect_vnm_etf.py

Publication lag rule (bắt buộc, tránh look-ahead bias):
→ PMI tháng T chỉ dùng từ ngày 5 tháng T+1
→ SBV rate chỉ dùng từ ngày announce
→ Tất cả macro: không dùng data trước publication date
```

**Lưu tại**: `data/macro/macro_combined.csv` (6 columns, 4747 rows, 0 NaN, 2013-01-02→2025-12-31)

---

## 3. FEATURE ENGINEERING & SELECTION

### Phân biệt quan trọng
```
Feature Engineering (Section 3 — Data Preparation):
→ TẠO RA nhiều feature candidates từ raw data
→ Không chọn, không loại — chỉ tạo
→ Output: ~30-50 feature candidates

Feature Selection (Section 4 — Methodology):
→ CHỌN features nào thực sự dùng trong LEAF
→ Đây là contribution methodology
→ Output: 6-10 features final
```

### Feature Engineering (Section 3)

**Nhóm A — Price features (từ OHLCV)**
```
Momentum:   ret_1d, ret_5d, ret_20d, ret_60d
Volatility: vol_5d, vol_20d, vol_60d, vol_ratio
Trend:      ema_ratio_12, ema_ratio_26, macd, rsi_14
Mean-rev:   bb_position, dist_52w_high
Volume:     volume_ratio, vwap_dev, hl_range
Memory:     autocorr_1 (rolling 20d)
```

**Nhóm B — Fractional Differentiation**
```
close_fd: FracDiff(close, d_optimal)
          d_optimal = min d để ADF p < 0.05
          Giữ memory trong khi đạt stationarity
```

**Nhóm C — Derived macro features**
```
Từ VIX:     vix_dev, vix_stress (>30)
Từ USD/VND: usdvnd_mom5, usdvnd_dev
Từ Foreign: foreign_net_z (rolling 60d), foreign_sell_streak
Từ IBOR:    ibor_dev, ibor_event
Từ PMI:     pmi_norm, pmi_above50, pmi_momentum
Từ HSI:     hsi_ret_1d, hsi_vol_20
Cross:      risk_off_composite = VIX×0.4 + USD×0.3 + Foreign×0.3
```

### Feature Selection (Section 4) — 4 bước pipeline
```
Bước 1: Spearman + Bonferroni correction
        → Loại feature không significant
        → CHỈ tính trên TRAIN set

Bước 2: Mutual Information ranking
        → Loại feature MI thấp + redundancy cao
        → CHỈ tính trên TRAIN set

Bước 3: SHAP trên surrogate GBM
        → Fit GBM trên TRAIN, tính SHAP trên TRAIN
        → Không dùng val/test ở bước này

Bước 4: Walk-forward stability test
        → TimeSeriesSplit(n_splits=5) trong TRAIN
        → Loại feature không ổn định qua thời gian
        → Stability = mean_shap / std_shap

Output: 6-10 features stable và predictive
```

---

## 4. REGIME DETECTION

### Vị trí trong pipeline
```
Regime Annotation thuộc Section 3 (Data Preparation):
→ Regime label là CỘT DỮ LIỆU, giống như Close, Volume
→ Là INPUT cho model, không phải OUTPUT
→ Phải tạo trước khi feed vào LEAF

Regime Analysis thuộc Section 5 (Experiments):
→ So sánh FM performance THEO regime
→ Đây là contribution analysis, không phải data prep
```

### Phương pháp chính: 3-state Gaussian HMM
```
Input:   Portfolio returns + Vol_20 (2 features)
Fit:     CHỈ trên Train set (2013-2022-06)
Predict: Toàn bộ series (train + test)
Label:   Bull (highest return) > Sideways > Bear (lowest)

Economic validation bắt buộc trước khi dùng:
✅ Bull return > Sideways return > Bear return
✅ Bear volatility > Bull volatility
✅ Bull > 20% sample
✅ Bear > 10% sample
→ Nếu fail → thử n_states=2 hoặc thêm features
```

### Phương pháp cross-validate: Volatility-based
```
vol_ratio = vol_20 / vol_60
ret_ma20  = rolling mean return 20 ngày

Bull:     vol_ratio < 0.8  AND ret_ma20 > 0
Bear:     vol_ratio > 1.3  AND ret_ma20 < 0
High-vol: vol_ratio > 1.3  AND ret_ma20 >= 0
Sideways: otherwise

Agreement HMM vs Vol-based > 70% → dùng HMM làm primary
```

---

## 5. METHOD — LEAF Framework

### Core Idea
Tạo **learned context encoder** học cách encode economic regime
thành soft conditioning signal cho FM ensemble —
tương tự soft prompting trong NLP nhưng cho time series tài chính.
Tất cả FM frozen, chỉ train connector nhỏ.

### Kiến trúc (3 components trainable)
```
Component 1: Regime Encoder (~50K params)
  Input:  Selected macro features, 60 ngày gần nhất
  Output: Regime embedding vector (dim=32)
  Arch:   Small MLP + Multi-head Attention (4 heads)
  Note:   Học regime từ data, không rule-based

Component 2: Context Projector (~30K params)
  Input:  Regime embedding + Price history
  Output: Augmented input (price + soft context prepended)
  Cơ chế: Regime → context sequence → prepend vào price
           Learned gating kiểm soát mức độ ảnh hưởng

Component 3: FM Ensemble + Dynamic Weighting (~20K params)
  FMs:    Chronos-2, TimesFM, Lag-Llama, Moirai (FROZEN)
  Weight: Softmax(Weight_network(regime_embedding))
  Note:   Regime quyết định tin FM nào nhiều hơn
```

### Tổng: ~100K trainable params | FM: hoàn toàn FROZEN

### Adaptive Window (tích hợp trong LEAF)
```
KL divergence giữa recent 30d vs older 30d returns:
→ KL cao (distribution shift nhanh) → window ngắn (60 ngày)
→ KL thấp (distribution ổn định)   → window dài (252 ngày)

Giải quyết: Sliding window cố định không handle được
            sudden regime change (COVID, bond crisis)
```

### Training Procedure
```
Phase 1: Walk-forward hyperparameter search
  → 5 folds trong Train (2013-2022-06)
  → Tune: learning_rate, adapter_dim, window_range
  → Chọn config có mean CRPS tốt nhất qua 5 folds

Phase 2: Final training trên toàn bộ Train
  → Dùng hyperparameter từ Phase 1
  → Train: Regime Encoder + Projector + Weight network
  → FM: completely frozen (không update gì)

Phase 3: Final evaluation trên Test
  → Test: 2022-07-01 → 2025-12-31
  → Chạy DUY NHẤT 1 lần
  → KHÔNG quay lại tune dù kết quả tốt hay xấu
```

### Loss Function
```
Primary:    CRPS (Continuous Ranked Probability Score)
Supporting: Pinball Loss tại quantiles [0.1, 0.25, 0.5, 0.75, 0.9]
```

---

## 6. BASELINE MODELS

### 4 tầng (bắt buộc có đủ để argument chắc)
```
Tier 1 — Classical (sanity check):
  AutoARIMA, GARCH, ETS
  Nguồn: statsforecast library

Tier 2 — Deep Learning traditional:
  LSTM, TCN, PatchTST, iTransformer
  Nguồn: neuralforecast library

Tier 3 — Foundation Models zero-shot:
  Chronos-2  (Amazon 2024) — probabilistic, native CRPS
  TimesFM    (Google 2024) — fast, point + uncertainty
  Lag-Llama  (Academic 2024) — probabilistic, lightweight
  Moirai     (Salesforce 2024) — multivariate native

Tier 4 — LEAF Ablations:
  LEAF w/o Regime Encoder  (→ measure encoder contribution)
  LEAF w/o Adaptive Window (→ measure window contribution)
  LEAF w/o Macro features  (→ measure causal signal contribution)
  LEAF full                (→ our method)
```

### Argument chain cần đủ
```
Classical < DL < FM zero-shot < LEAF
     ↑           ↑                ↑
FM có lý do  Chronos tốt    LEAF adds value
tồn tại      nhất trong FM  trên FM tốt nhất
```

---

## 7. EVALUATION METRICS

### Point forecast (tất cả models)
```
MAE   — Mean Absolute Error
RMSE  — Root Mean Squared Error
DA    — Directional Accuracy (%) ← quan trọng nhất trading
```

### Probabilistic (FM models — Chronos, Lag-Llama, LEAF)
```
CRPS     — PRIMARY metric, đo distribution quality
Coverage — Empirical coverage 80% prediction interval
Winkler  — Penalize width lẫn coverage
Pinball  — Tại quantiles 10, 25, 50, 75, 90
```

### Regime-stratified (KEY CONTRIBUTION)
```
Report CRPS theo từng regime:
Bull | Bear | Sideways | High-vol | FTSE upgrade

→ Đây là nơi show FM fail ở đâu
→ Justify LEAF routing logic
→ Không paper nào có bảng này cho VN30
```

### Statistical significance
```
Diebold-Mariano test:
→ Chứng minh improvement không phải do may mắn
→ Cần chạy cho mọi cặp so sánh với LEAF
```

---

## 8. PAPER STRUCTURE (Updated)

```
Section 1: Introduction (~1,000 từ)
  → 3 vấn đề FM trên EM equity
    (distribution shift, causal blindness, random walk)
  → Gap: không paper nào cho Vietnam/SEA
  → 4 contributions
  → Paper organization

Section 2: Related Work (~800 từ)
  2.1 Time Series Foundation Models
  2.2 FM for Financial Forecasting
  2.3 Soft Prompting & Context Conditioning
  2.4 Emerging Market Equity Forecasting

Section 3: Dataset — VN30-MacroTS (~1,000 từ)
  3.1 Asset Selection
      → Universe 30 mã VN30 7/2022
      → 3 exclusion criteria (E1, E2, E3)
      → Representativeness (tracking corr 0.91)
  3.2 Price Data Collection
      → VNDirect, adjusted prices, cross-validation
  3.3 Macro Features Collection
      → 6 features, causal justification, pub lag rule
  3.4 Feature Engineering          ← MỚI THÊM
      → Price features (momentum, vol, trend)
      → Fractional Differentiation
      → Derived macro features
      → Output: ~40 feature candidates
  3.5 Regime Annotation
      → HMM 3-state, fit train only
      → Economic validation
      → Regime timeline figure
  3.6 Data Splits
      → Train/Test, walk-forward folds
      → Justify July 2022 boundary
  3.7 Dataset Statistics
      → Table: mô tả dataset
      → Figure: regime distribution

Section 4: Methodology — LEAF (~1,200 từ)
  4.1 Problem Formulation
  4.2 Motivation: Why FM Fail on VN30
      → 3 failure modes
      → Granger causality motivation
  4.3 Feature Selection              ← CHUYỂN TỪ S3 SANG S4
      → 4-step pipeline (Spearman→MI→SHAP→Stability)
      → CHỈ trên train set
      → Output: 6-10 final features
  4.4 LEAF Framework Overview
  4.5 Regime Encoder
  4.6 Context Projector
  4.7 FM Ensemble + Dynamic Weighting
  4.8 Adaptive Window Mechanism
  4.9 Training Procedure

Section 5: Experiments (~1,500 từ)
  5.1 Experimental Setup
  5.2 Main Results (Overall CRPS table)
  5.3 Regime-Stratified Analysis
  5.4 Causal Analysis (Granger causality)
  5.5 FTSE Upgrade Natural Experiment
  5.6 Ablation Study
  5.7 Sensitivity Analysis

Section 6: Discussion (~600 từ)
  6.1 Why Causal Macro Signals Matter
  6.2 FM Failure Mode Taxonomy
  6.3 Implications cho EM Markets
  6.4 Limitations

Section 7: Conclusion (~300 từ)

Appendix A: Dataset Documentation
Appendix B: Robustness Checks
Appendix C: Implementation Details
Appendix D: Per-stock + Horizon Analysis
```

---

## 9. KEY DECISIONS ĐÃ LOCK

```
✅ Composition date: 7/2022
   Lý do: ngay trước test period, no look-ahead

✅ Train: 2013-01-02 → 2022-06-30
   Test:  2022-07-01 → 2025-12-31
   Lý do: aligned với composition date

✅ Không có val set riêng
   Val nhúng trong walk-forward 5 folds (2017-2021)
   Lý do: maximize train data + robust hyperparameter

✅ Test chạy 1 lần duy nhất — KHÔNG tune sau

✅ FM hoàn toàn frozen (~100K params của LEAF)
   Lý do: tránh catastrophic forgetting

✅ Primary metric: CRPS

✅ Exclusion: chỉ 3 tiêu chí E1, E2, E3
   E4 (ADF) đã BỎ — redundant với E1+E2+E3

✅ Liquidity: Adaptive P20 (không dùng 30B cứng)
   Lý do: thị trường 2013 khác 2022

✅ Macro frequency: Daily
   Monthly → forward-fill với publication lag
   Event-based → step function

✅ Adaptive window: KL divergence-based (60-252 ngày)
   Tích hợp trong LEAF

✅ Feature Engineering thuộc Section 3 (data prep)
   Feature Selection thuộc Section 4 (methodology)

✅ Regime Annotation thuộc Section 3
   Regime Analysis (performance theo regime) thuộc Section 5
```

---

## 10. RELATED WORK ĐÃ BIẾT

```
ĐÃ CÓ — không claim novelty:
→ Time-MoE, Moirai-MoE: MoE bên trong FM
→ Time-Prompt: soft prompting text cho TSFM
→ History Rhymes: macro retrieval cho S&P500 (2025)
→ Modi & Pan: ensemble FM bagging/boosting (2025)
→ Re(Visiting) TSFMs in Finance: 94 countries (2025)
→ Kronos: FM on financial K-line data (2025)

CHƯA CÓ — novelty của mình:
→ FM benchmark trên VN30 specifically
→ Learned causal macro embedding cho frozen FM ensemble
→ Foreign flow + IBOR làm causal conditioning
→ Regime-aware FM routing cho EM equity
→ FTSE upgrade natural experiment
→ FM failure mode analysis trên Frontier Market
```

---

## 11. TECHNICAL INFO

### Libraries
```
Python 3.10+
chronos-forecasting    ← Chronos-2
timesfm                ← TimesFM Google
lag-llama              ← Lag-Llama
neuralforecast         ← DL baselines (LSTM, PatchTST...)
statsforecast          ← Classical baselines (ARIMA, GARCH)
hmmlearn               ← Regime detection
properscoring          ← CRPS computation
shap                   ← Feature selection
torch                  ← LEAF implementation
yfinance               ← Macro data
scipy, statsmodels     ← Statistical tests
```

### Cấu trúc thư mục
```
project/
├── data/
│   ├── raw/
│   │   └── selected/         ← OHLCV 15 mã (VNDirect)
│   ├── macro/
│   │   ├── vix.csv
│   │   ├── hsi.csv
│   │   ├── usdvnd.csv
│   │   ├── ibor.csv
│   │   ├── pmi.csv
│   │   ├── foreign.csv
│   │   └── macro_combined.csv
│   ├── processed/
│   │   ├── features_all.csv       ← Tất cả feature candidates
│   │   ├── features_selected.csv  ← Features đã chọn
│   │   └── dataset_with_regime.csv ← Final dataset + labels
│   └── coveraties/
│       └── VN30.csv
├── src/
│   ├── data/
│   │   ├── collect_price.py
│   │   ├── collect_macro.py
│   │   ├── feature_engineering.py  ← Tạo features
│   │   └── feature_selection.py    ← Chọn features
│   ├── models/
│   │   ├── leaf.py
│   │   ├── regime_encoder.py
│   │   ├── context_projector.py
│   │   └── baselines.py
│   └── evaluation/
│       ├── metrics.py
│       └── regime_analysis.py
├── notebooks/
│   ├── 01_asset_selection.ipynb
│   ├── 02_macro_collection.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_regime_detection.ipynb
│   ├── 05_feature_selection.ipynb
│   ├── 06_baseline_experiments.ipynb
│   └── 07_leaf_experiments.ipynb
└── paper/
    └── sections/
```

---

## 12. FTSE UPGRADE CONTEXT

```
Sự kiện: FTSE Russell nâng Vietnam
         Frontier → Secondary Emerging Market (10/2025)
Effective: 09/2026

Ý nghĩa cho paper:
→ Dataset capture toàn bộ Frontier → EM journey
→ Natural experiment không thể replicate
→ $18 trillion FTSE-indexed funds phải mua VN30
→ Foreign flow signal detect transition sớm

Expected finding:
LEAF detect regime transition 2-3 tuần sớm hơn
FM vanilla thông qua foreign flow signal tăng đột biến
```

---

## 13. PIPELINE THỨ TỰ THỰC HIỆN

```
[DONE ✅ hoặc TODO ❌ — update hàng ngày]

BƯỚC 1: Asset Selection
  ✅/❌ Chạy exclusion pipeline từ 30 mã
  ✅/❌ Verify N mã pass E1+E2+E3
  ✅/❌ Tính tracking correlation

BƯỚC 2: Price Data
  ✅/❌ Download OHLCV từ VNDirect
  ✅/❌ Validate data quality

BƯỚC 3: Macro Data
  ✅/❌ Download VIX, HSI, USD/VND, IBOR, PMI
  ✅/❌ Download Foreign net flow
  ✅/❌ Validate + align về trading calendar

BƯỚC 4: Feature Engineering
  ✅/❌ Tạo price features (~20 features)
  ✅/❌ Tính Fractional Differentiation
  ✅/❌ Tạo derived macro features (~20 features)
  ✅/❌ Save features_all.csv

BƯỚC 5: Regime Annotation
  ✅/❌ Fit HMM 3-state trên train
  ✅/❌ Validate economic sense
  ✅/❌ Add regime column vào dataset

BƯỚC 6: Feature Selection (Section 4)
  ✅/❌ Spearman + Bonferroni (trên train)
  ✅/❌ Mutual Information ranking
  ✅/❌ SHAP surrogate GBM
  ✅/❌ Walk-forward stability test
  ✅/❌ Save features_selected.csv

BƯỚC 7: Baseline Models
  ✅/❌ Classical: ARIMA, GARCH
  ✅/❌ DL: LSTM, PatchTST
  ✅/❌ FM zero-shot: Chronos, TimesFM, Lag-Llama
  ✅/❌ Collect CRPS results

BƯỚC 8: LEAF Implementation
  ✅/❌ Regime Encoder
  ✅/❌ Context Projector
  ✅/❌ FM Ensemble + Weighting
  ✅/❌ Adaptive Window
  ✅/❌ Walk-forward hyperparameter search
  ✅/❌ Final training + test evaluation

BƯỚC 9: Analysis
  ✅/❌ Regime-stratified results
  ✅/❌ Granger causality analysis
  ✅/❌ FTSE upgrade analysis
  ✅/❌ Ablation study

BƯỚC 10: Paper Writing
  ✅/❌ Section 3 (Dataset)
  ✅/❌ Section 4 (Methodology)
  ✅/❌ Section 5 (Experiments)
  ✅/❌ Section 1-2, 6-7
  ✅/❌ Abstract
```

---

## 14. CURRENT STATUS

> Last updated: 2026-05-08 — điền ✅ khi hoàn thành

```
HOÀN THÀNH:
✅ Paper structure và outline
✅ Asset selection logic (E1, E2, E3)
✅ Price data collection code (VNDirect)
✅ Representativeness analysis
✅ Macro features collection code
✅ Feature engineering pipeline
✅ Feature selection 4-step pipeline (Spearman→MI→SHAP→Stability)
✅ Regime detection code (HMM 3-state, Gaussian)
✅ Dataset preparation & normalization (StandardScaler)
✅ Project context document (phiên bản 2.0)
✅ Chạy đầy đủ pipeline end-to-end:
   - data/processed/v2/features_train.csv (34,917 rows, 15 tickers, 2013-03-04 ~ 2022-06-30)
   - data/processed/v2/features_test.csv (13,140 rows, 15 tickers, 2022-07-01 ~ 2025-12-31)
   - data/processed/v2/features_selected.csv (11 features: 7 predictor + 4 force-include)
   - data/processed/v2/feature_selection_report.csv (35 → 12 → 7 → 11 pipeline)
✅ Regime annotation + validation:
   - data/processed/v2/regime.csv (Bull/Sideways/Bear labels)
   - Economic validation: Bull%=41, Sideways%=51, Bear%=8
   - Raw Viterbi labels (no smoothing) + 21-day centered smoothing for viz

ĐANG LÀM / VỪA XONG:
✅ Baseline Models Implementation:
   ✅ ARIMA(1,0,1) univariate zero-shot → data/results/arima_results.csv
   ✅ LSTM multivariate (11 features, lookback=20, early stopping) → data/results/lstm_results.csv
   ✅ Chronos-2 zero-shot (GluonTS pipeline.predict_df) → data/results/chronos_results.csv
   ✅ Moirai-2.0 zero-shot (GluonTS rolling 1-step) → data/results/moirai_results.csv
   🔄 TimesFM 2.0-500M zero-shot (pending)
   
   Metrics: MAE, RMSE, DA (Directional Accuracy), IC (Information Coefficient)
   Per-regime breakdown: Bull/Sideways/Bear
   
   Preliminary Results (LSTM):
   - Overall: DA=0.48, MAE=0.015, RMSE=0.022
   - Bull: DA=0.48, IC=0.052
   - Sideways: DA=0.46, IC≈0.01 (FM blind zone)
   - Bear: DA=0.49, IC=0.048

✅ Paper sections (drafts):
   ✅ Section 3.4 Feature Engineering narrative (Vietnamese, for paper)
   ✅ Section 3.5 Regime Detection + validation story
   ✅ Section 3.6 Data Splits + warm-up period explanation (04/03/2013)
   ✅ Section 4.3 Feature Selection 4-step pipeline explanation
   ✅ Visualizations in notebooks/eda/:
      - feature_selection_for_paper.ipynb (funnel, SHAP, stability, type breakdown)
      - regime_results_for_paper.ipynb (timeline, per-regime stats, smoothing logic)

CHƯA LÀM:
❌ TimesFM implementation + evaluation
❌ LEAF implementation (Regime Encoder, Context Projector, FM Ensemble)
❌ Ablation study (LEAF w/o encoder, w/o window, w/o macro)
❌ Granger causality analysis (Section 5.4)
❌ FTSE upgrade natural experiment analysis (Section 5.5)
❌ Full paper writing (Intro, Related Work, Discussion, Conclusion)
❌ Diebold-Mariano statistical tests
❌ Sensitivity analysis

VƯỚNG MẮC ĐÃ GIẢI:
✅ HMM state labeling bị đảo (Sharpe vs Return sorting) → Fixed
✅ Regime bị smooth theo numeric indices (breaking economic labels) → Reverted to raw
✅ Baseline alignment lệch 37 ngày (GluonTS resample drift) → Fixed with cutoff-based n_test
✅ LSTM overfitting + no early stopping → Added validation set + Huber loss
✅ FM chỉ predict 30 ngày thay vì 876 ngày → Converted all to rolling 1-step
```

---

## 15. HƯỚNG DẪN CHO AI KHI NHẬN DOCUMENT NÀY

```
1. Đây là research project FM cho VN30
   → Đọc kỹ Key Decisions (Section 9) trước khi suggest gì

2. Không đề xuất thay đổi decision đã lock
   trừ khi có lý do scientific rất mạnh

3. Phong cách:
   → Giải thích bằng tiếng Việt
   → Code có comment giải thích logic
   → Khi suggest mới → nói rõ "đã có paper chưa"

4. Priority: xem Current Status (Section 14)
   → Focus vào bước đang làm dở

5. Phân biệt:
   → Feature Engineering = Section 3 (data prep)
   → Feature Selection = Section 4 (methodology)
   → Regime Annotation = Section 3 (data)
   → Regime Analysis = Section 5 (experiments)
   → Val set = nhúng trong walk-forward, không riêng biệt

6. Không cần giải thích lại những gì đã biết
   → Jump thẳng vào vấn đề
```

---

*Document version: 2.0*
*Last updated: 2025-05-06*
*Project: VN30 Foundation Model Research — LEAF Framework*
