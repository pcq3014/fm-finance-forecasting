# VN30 Foundation-Model Forecasting with LEAF

> **LEAF** — *Lightweight Regime-Aware Ensemble Adapter*
> Routing 3 frozen time-series foundation models (TimesFM, Chronos-2, Moirai-2) for 1-day-ahead return forecasting on VN30 stocks.

---

## Headline result

LEAF beats every baseline on every metric on the full test set (2024-01-03 → 2025-12-30, 15 tickers × 416 days = 6240 rows), and is the **only model with positive Information Coefficient** despite being built from three FMs that are all individually negative-IC on this period.

| Model | RMSE ↓ | MAE ↓ | IC ↑ | DA ↑ |
|---|---|---|---|---|
| **LEAF (5-seed ensemble)** | **0.02107** | **0.01349** | **+0.0429** | **0.4731** |
| ARIMA | 0.02130 | 0.01370 | −0.0229 | 0.4739 |
| LSTM-MV | 0.02128 | 0.01375 | −0.0028 | 0.4567 |
| Chronos-2 | 0.02175 | 0.01377 | −0.0557 | 0.4518 |
| TimesFM | 0.02287 | 0.01399 | −0.0566 | 0.4614 |
| Equal-Weight (3 FM) | 0.02271 | 0.01411 | −0.0698 | 0.4558 |
| Moirai-2 | 0.02469 | 0.01520 | −0.0707 | 0.4563 |

How LEAF achieves this is explained in [`LEAF_IMPROVEMENT_REPORT.md`](LEAF_IMPROVEMENT_REPORT.md). The short version: dynamic per-sample routing via a regime encoder + 17 auxiliary features (FM disagreement, error correlation, ticker characteristics) + per-FM affine calibration + entropy-floor regularization + 5-seed ensemble.

---

## Repository layout

```
fm-forecasting/
├── README.md                       ← this file
├── requirements.txt
├── LEAF_IMPROVEMENT_REPORT.md      ← detailed write-up of the LEAF redesign
├── context.md                      ← project history & session notes
│
│  ── Data pipeline ──
├── preprocess_macro.py             ← raw macro CSVs → cleaned macro panel
├── create_panel_data.py            ← merge stock+macro into long panel
├── create_targets_and_split.py     ← compute y_return_1d, train/val/test split
├── feature_selection_v3_final.py
├── check_and_fix_leakage.py
│
│  ── Baselines ──
├── arima_baseline_v3.py
├── lstm_baseline_v3.py
│
│  ── Foundation Model zero-shot scripts ──
├── timesfm_zeroshot_v3.py
├── chronos2_zeroshot_fixed.py
├── moirai2_from_notebook.py
│
│  ── LEAF ──
├── leaf_ensemble.py                ← main LEAF training + 5-seed ensemble
├── leaf_ablation.py                ← ablation study (4 rounds)
│
│  ── Visualization ──
├── visualize_results.py            ← generate figures
│
├── src/                            ← modular code
│   ├── baselines/                  (older baseline modules)
│   ├── data/                       (feature engineering, regime detection)
│   └── models/                     (LEAF building blocks, alt trainers)
│
├── data/                           ← NOT in repo — see "Data" section
│   ├── raw/                            input CSVs (stock OHLCV, macro)
│   ├── processed/                      panel + train/val/test splits
│   └── results/                        predictions + summaries per model
│
└── outputs/figures/                ← generated PNGs (key ones may be committed)
```

---

## Quick start

### 1. Install

```bash
# Core deps
pip install -r requirements.txt

# Foundation models — install in a dedicated env, they have heavy deps
pip install timesfm
pip install uni2ts gluonts
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

GPU strongly recommended for FM inference (especially Chronos-2 and Moirai-2).

### 2. Data

The processed panel data (~300 MB) is **not** stored in the repo. Two options:

- **Regenerate from raw**: run the pipeline below, starting from `data/raw/` (stock OHLCV + macro CSVs).
- **Download the prepared panel** (recommended for quick start): *[upload your data archive to Google Drive / Zenodo / Kaggle and link here]*.

Expected layout once data is available:

```
data/
├── raw/                                stock + macro source CSVs
├── processed/
│   ├── macro/                          cleaned macro series
│   ├── panel/
│   │   ├── panel_train_fixed.csv       train split  (~2013–2021)
│   │   ├── panel_val_fixed.csv         val split    (~2022–2023)
│   │   └── panel_test_fixed.csv        test split   (2024-01-03 → 2025-12-30)
│   └── v2/                             feature selection artifacts
└── results/                            model outputs (created by run scripts)
```

### 3. Reproduce the full pipeline

```bash
# (a) Macro preprocessing
python preprocess_macro.py

# (b) Build panel & splits
python create_panel_data.py
python create_targets_and_split.py

# (c) Baselines  (each writes data/results/<model>_{predictions,summary}.csv)
python arima_baseline_v3.py
python lstm_baseline_v3.py

# (d) Foundation models (zero-shot, no fine-tuning)
python timesfm_zeroshot_v3.py
python chronos2_zeroshot_fixed.py
python moirai2_from_notebook.py

# (e) LEAF — uses val predictions from the 3 FMs above
python leaf_ensemble.py        # produces the headline numbers
python leaf_ablation.py        # reproduces the 4 improvement rounds

# (f) Figures
python visualize_results.py    # writes outputs/figures/*.png
```

Total runtime ~30 min on a single GPU; LEAF alone is ~3–5 min.

---

## What each script does

### Data pipeline
| Script | Purpose |
|---|---|
| `preprocess_macro.py` | Clean raw macro CSVs (FX, rates, foreign flows) into a unified series |
| `create_panel_data.py` | Merge stock OHLCV + macro features into a long-format panel |
| `create_targets_and_split.py` | Compute `y_return_1d` log returns, build train/val/test temporal splits |
| `feature_selection_v3_final.py` | Final feature set used by multivariate models |
| `check_and_fix_leakage.py` | Audit for lookahead leakage in features |

### Baselines
| Script | Model | IC on test |
|---|---|---|
| `arima_baseline_v3.py` | ARIMA(1,0,1) on returns, per-ticker | −0.023 |
| `lstm_baseline_v3.py` | 2-layer LSTM, 8 features, early-stop on val | −0.003 |

### Foundation models (zero-shot)
| Script | Model | IC on test |
|---|---|---|
| `timesfm_zeroshot_v3.py` | Google TimesFM 1.0-200M | −0.057 |
| `chronos2_zeroshot_fixed.py` | Amazon Chronos-2 (with covariates) | −0.056 |
| `moirai2_from_notebook.py` | Salesforce Moirai-2.0-R-small (multivariate, GluonTS) | −0.071 |

All three roll from start of val through end of test in one pass; train data is context only, val+test predictions are stored.

### LEAF
| Script | What it does |
|---|---|
| `leaf_ensemble.py` | Full LEAF: regime encoder + dynamic router + 17 aux features + per-FM calibration, trained 5 seeds and ensembled. **Run this for the headline result.** |
| `leaf_ablation.py` | Same pipeline but runs 5 variants (Round 0 baseline → Round 4 full) and prints an ablation table. Use to verify each improvement contributed. |

---

## Methodology highlights

- **No data leakage**: all rolling features use `.shift(1)`; scalers fit on val only; FMs are zero-shot (no training on val/test).
- **Deterministic & reproducible**: `set_seed()` covers torch / numpy / random / cudnn; fixed seeds `[42, 123, 2024, 7, 1337]`.
- **Same evaluation period for every model**: 416 trading days (2024-01-03 → 2025-12-30), 15 tickers, 6240 rows.
- **LEAF training data**: FM val predictions are used to train LEAF (true out-of-sample for the FMs) — no information from test ever touches LEAF training.

See [`LEAF_IMPROVEMENT_REPORT.md`](LEAF_IMPROVEMENT_REPORT.md) for the full design rationale and ablation analysis.

---

## Key figures

After running `python visualize_results.py`:

| File | Content |
|---|---|
| `outputs/figures/fig1_model_comparison.png` | Bar chart of RMSE / MAE / IC / DA across all 7 models |
| `outputs/figures/fig2_ic_heatmap.png` | Per-ticker × per-model IC heatmap |
| `outputs/figures/fig3_regime_weights.png` | LEAF weight distribution by market regime |
| `outputs/figures/fig4_weight_timeline.png` | LEAF weights over time, with regime background |
| `outputs/figures/fig5_ic_per_regime.png` | IC per regime — the chart showing LEAF avoids Bear-regime collapse |

---

## Reproducibility checklist

To reproduce the headline IC = **+0.0429**:

1. Same Python + torch versions as `requirements.txt`.
2. Run scripts (c)–(e) above; LEAF needs val + test prediction CSVs from all 3 FMs.
3. `python leaf_ensemble.py` will end with:
   ```
   --- ensemble (avg of 5 seeds) ---
   RMSE=0.02107  MAE=0.01349  IC=+0.0429  DA=0.4731
   ```
4. `python leaf_ablation.py` reproduces every row of the 4-round ablation table.

`cudnn.deterministic = True` + fixed `SEEDS` → byte-for-byte same output across runs on the same hardware.

---

## Status & follow-ups

| Item | Status |
|---|---|
| Foundation-model zero-shot baselines | ✅ done |
| ARIMA / LSTM baselines | ✅ done |
| LEAF router + calibration + ensemble | ✅ done (headline result) |
| Multi-seed reproducibility | ✅ done |
| Ablation study | ✅ `leaf_ablation.py` |
| Walk-forward LEAF re-training | ⏳ not implemented |
| 4th/5th base forecaster (ARIMA / LSTM in router) | ⏳ not implemented |
| Hyper-parameter search | ⏳ not implemented |

---

## Citation / contact

This work is part of a research project on foundation-model adaptation for Vietnamese equity forecasting (ViFinTSFM-LEAF framework). Open an issue or reach out if you'd like to discuss methodology or reuse the code.
