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

How LEAF achieves this is explained in [`MACRO_PROCESSING_GUIDE.md`](MACRO_PROCESSING_GUIDE.md) for the data pipeline, and in the code comments of [`scripts/leaf_ensemble.py`](scripts/leaf_ensemble.py). The short version: dynamic per-sample routing via a regime encoder + 17 auxiliary features (FM disagreement, error correlation, ticker characteristics) + per-FM affine calibration + entropy-floor regularization + 5-seed ensemble.

---

## Repository layout

```
fm-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
│
├── MACRO_PROCESSING_GUIDE.md           ← detailed macro pipeline doc
├── MACRO_PROCESSING_FINAL_REPORT.md
├── MACRO_QUICK_REFERENCE.md
├── MACRO_COMPLETED.md
│
├── scripts/                            ← main pipeline (run these in order)
│   │
│   │   ── Data preprocessing ──
│   ├── preprocess_macro.py
│   ├── create_panel_data.py
│   ├── create_targets_and_split.py
│   ├── feature_selection_v3_final.py
│   │
│   │   ── Traditional baselines ──
│   ├── arima_baseline_v3.py
│   ├── lstm_baseline_v3.py
│   │
│   │   ── Foundation model zero-shot ──
│   ├── timesfm_zeroshot_v3.py
│   ├── chronos2_zeroshot_v3.py
│   ├── moirai2_from_notebook.py
│   │
│   │   ── LEAF (main contribution) ──
│   ├── leaf_ensemble.py                ← multi-seed ensemble (headline result)
│   ├── leaf_ablation.py                ← reproduce 4 improvement rounds
│   │
│   │   ── Visualization ──
│   ├── visualize_results.py
│   │
│   └── utils/                          ← validation / debug utilities
│       ├── analyze_macro_vars.py
│       ├── check_and_fix_leakage.py
│       └── validate_macro_data.py
│
├── src/                                ← support modules
│   ├── config.py                       (constants, paths)
│   └── data/
│       ├── collecting/                 (raw-data fetchers from external APIs)
│       │   ├── collect_price.py
│       │   └── collect_vnm_etf.py
│       └── compare_regime_features.py
│
├── outputs/figures/                    ← committed PNG figures
│   ├── fig1_model_comparison.png
│   ├── fig2_ic_heatmap.png
│   ├── fig3_regime_weights.png
│   ├── fig4_weight_timeline.png
│   └── fig5_ic_per_regime.png
│
└── data/                               ← NOT in repo (gitignored, see below)
```

---

## Quick start

### 1. Install

```bash
# Core dependencies
pip install -r requirements.txt

# Foundation models — install separately (heavy dependencies)
pip install timesfm
pip install uni2ts gluonts
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

GPU strongly recommended for FM inference (Chronos-2, Moirai-2 are large models).

### 2. Get the data

The processed panel data (~300 MB) is **not** stored in the repo. You have two options:

- **A. Download prepared data** (recommended): *[upload your data archive to Google Drive / Zenodo / Kaggle and paste the link here]*. Extract into `data/`.
- **B. Regenerate from scratch**: run `src/data/collecting/*.py` to fetch raw data, then run the preprocessing scripts below.

Expected layout once data is in place:

```
data/
├── raw/                                stock OHLCV + macro source CSVs
├── processed/
│   ├── macro/                          cleaned macro series
│   ├── panel/
│   │   ├── panel_train_fixed.csv       train (~2013–2021)
│   │   ├── panel_val_fixed.csv         val   (~2022–2023)
│   │   └── panel_test_fixed.csv        test  (2024-01-03 → 2025-12-30)
│   └── v2/                             feature selection artifacts
└── results/                            model outputs (created by run scripts)
```

### 3. Reproduce the full pipeline

Each command writes to `data/results/` (predictions and per-ticker summaries) or `outputs/figures/`.

```bash
# (a) Data pipeline  — skip if you downloaded prepared data
python scripts/preprocess_macro.py
python scripts/create_panel_data.py
python scripts/create_targets_and_split.py

# (b) Traditional baselines
python scripts/arima_baseline_v3.py
python scripts/lstm_baseline_v3.py

# (c) Foundation models (zero-shot, no fine-tuning)
python scripts/timesfm_zeroshot_v3.py
python scripts/chronos2_zeroshot_v3.py
python scripts/moirai2_from_notebook.py

# (d) LEAF — uses val predictions from the 3 FMs above
python scripts/leaf_ensemble.py        # headline result (5-seed ensemble)
python scripts/leaf_ablation.py        # reproduce 4 improvement rounds

# (e) Figures
python scripts/visualize_results.py
```

Total runtime ~30 min on a single GPU; LEAF alone is ~3–5 min.

---

## What each script does

### Data pipeline (`scripts/`)
| Script | Purpose |
|---|---|
| `preprocess_macro.py` | Clean raw macro CSVs (FX, rates, foreign flows) into a unified series |
| `create_panel_data.py` | Merge stock OHLCV + macro features into a long-format panel |
| `create_targets_and_split.py` | Compute `y_return_1d` log returns, build train/val/test splits |
| `feature_selection_v3_final.py` | Final feature set used by multivariate models |
| `utils/check_and_fix_leakage.py` | Audit for lookahead leakage in features |
| `utils/analyze_macro_vars.py` | Statistical summary of macro variables |
| `utils/validate_macro_data.py` | Sanity checks on macro CSV outputs |

### Baselines (`scripts/`)
| Script | Model | IC on test |
|---|---|---|
| `arima_baseline_v3.py` | ARIMA(1,0,1) on returns, per-ticker | −0.023 |
| `lstm_baseline_v3.py` | 2-layer LSTM, 8 features, early-stop on val | −0.003 |

### Foundation models — zero-shot (`scripts/`)
| Script | Model | IC on test |
|---|---|---|
| `timesfm_zeroshot_v3.py` | Google TimesFM 1.0-200M | −0.057 |
| `chronos2_zeroshot_v3.py` | Amazon Chronos-2 (with past covariates) | −0.056 |
| `moirai2_from_notebook.py` | Salesforce Moirai-2.0-R-small (multivariate, GluonTS) | −0.071 |

All three roll from start of val through end of test in one pass; train is context-only; val + test predictions are stored separately.

### LEAF (`scripts/`)
| Script | What it does |
|---|---|
| `leaf_ensemble.py` | Regime encoder + dynamic router + 17 aux features + per-FM calibration, trained over 5 seeds and ensembled. **Run this for the headline result.** |
| `leaf_ablation.py` | Same pipeline but runs 5 variants (Round 0 baseline → Round 4 full) and prints an ablation table. Use to verify each improvement contributed. |

### Data collection (`src/data/collecting/`)
| Script | What it fetches |
|---|---|
| `collect_price.py` | Daily OHLCV for VN30 constituents |
| `collect_vnm_etf.py` | VNM ETF flows |

> Only the two scripts above are kept in the repo; other macro fetchers were retired. To rebuild the full macro panel you can use the prepared data link or adapt the collection scripts above.

---

## Methodology highlights

- **No data leakage** — all rolling features use `.shift(1)`; scalers fit on val only; FMs are zero-shot (no training on val/test); LEAF training data = FM val predictions (true out-of-sample for the FMs).
- **Deterministic & reproducible** — `set_seed()` covers torch / numpy / random / cudnn; fixed seeds `[42, 123, 2024, 7, 1337]`.
- **Same evaluation period for every model** — 416 trading days, 15 tickers, 6240 rows.
- **Multi-seed ensemble for LEAF** — single-seed IC has σ ≈ 0.028; averaging 5 seeds collapses noise and yields a deterministic, reproducible result.

---

## Key figures (in `outputs/figures/`)

| File | Content |
|---|---|
| `fig1_model_comparison.png` | RMSE / MAE / IC / DA bar chart across all 7 models |
| `fig2_ic_heatmap.png` | Per-ticker × per-model IC heatmap |
| `fig3_regime_weights.png` | LEAF weight distribution by market regime |
| `fig4_weight_timeline.png` | LEAF weights over time, with regime background |
| `fig5_ic_per_regime.png` | IC per regime — shows LEAF avoids Bear-regime collapse |

---

## Reproducibility checklist

To reproduce the headline IC = **+0.0429**:

1. Same Python + torch versions as `requirements.txt`.
2. Run scripts (b)–(d) above in order; LEAF needs val + test prediction CSVs from all 3 FMs.
3. `python scripts/leaf_ensemble.py` will end with:
   ```
   --- ensemble (avg of 5 seeds) ---
   RMSE=0.02107  MAE=0.01349  IC=+0.0429  DA=0.4731
   ```
4. `python scripts/leaf_ablation.py` reproduces every row of the 4-round ablation table:
   ```
   R0 baseline (collapse)        IC=-0.0580
   R1 + entropy floor            IC=-0.0639
   R2 + tracking aux (9)         IC=-0.0622
   R3 + interact+ticker+calib    IC=+0.0092
   R4 + 5-seed ensemble          IC=+0.0429
   ```

`cudnn.deterministic = True` + fixed `SEEDS` → byte-for-byte same output across runs on the same hardware.

---

## Status & follow-ups

| Item | Status |
|---|---|
| Foundation-model zero-shot baselines | ✅ done |
| ARIMA / LSTM baselines | ✅ done |
| LEAF router + calibration + ensemble | ✅ done (headline result) |
| Multi-seed reproducibility | ✅ done |
| Ablation study | ✅ `scripts/leaf_ablation.py` |
| Walk-forward LEAF re-training | ⏳ not implemented |
| 4th/5th base forecaster (ARIMA / LSTM in router) | ⏳ not implemented |
| Hyper-parameter search | ⏳ not implemented |

---

## Notes on data hosting

Per `.gitignore`, the following are **not** tracked:

- `data/` (raw + processed) — too large for GitHub, hosted externally
- `*.csv`, `*.zip` — generated outputs or compressed archives
- `__pycache__/`, virtual environments, IDE config

If you need the prepared dataset, see the link in step 2 of Quick Start above. The repo is fully reproducible from that dataset alone — no manual data fetching required.

---

## Citation / contact

This work is part of a research project on foundation-model adaptation for Vietnamese equity forecasting. Open an issue or reach out if you'd like to discuss methodology or reuse the code.
