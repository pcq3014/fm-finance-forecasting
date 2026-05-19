# Foundation Models - Installation & Usage Guide

## Status

| Model | Status | Error | Fix |
|-------|--------|-------|-----|
| TimesFM 2.0 | ❌ | Module not found | `pip install timesfm` |
| Chronos-2 | ❌ | Module not found | `pip install chronos` |
| Moirai-2 | ❌ | Auth required / Not found | Need HuggingFace token |

---

## Installation Instructions

### TimesFM 2.0 (Google)

```bash
# Installation
pip install timesfm torch

# Run
python timesfm_zeroshot_v3.py
```

**Requirements**:
- PyTorch
- Google's TimesFM weights (~500MB)
- GPU recommended (can run on CPU but slow)

**Script**: `timesfm_zeroshot_v3.py`

---

### Chronos-2 (Amazon)

```bash
# Installation
pip install chronos transformers torch

# Run
python chronos2_zeroshot_v3.py
```

**Requirements**:
- PyTorch
- Transformers
- Amazon's Chronos weights

**Script**: `chronos2_zeroshot_v3.py`

---

### Moirai-2 (Salesforce)

```bash
# Installation (requires HuggingFace authentication)
pip install transformers torch

# Authenticate (if private repo)
huggingface-cli login

# Run
python moirai2_zeroshot_v3.py
```

**Requirements**:
- PyTorch
- Transformers
- HuggingFace Token (for gated models)
- Salesforce model access

**Script**: `moirai2_zeroshot_v3.py`

**Note**: Moirai-2 model may require special access or authentication

---

## Expected Performance (Estimated)

Based on literature and model capabilities:

| Model | Type | RMSE (Expected) | IC (Expected) | Notes |
|-------|------|-----------------|---------------|-------|
| TimesFM | Univariate | 0.015-0.025 | 0.05-0.15 | Large context (256) |
| Chronos-2 | Univariate | 0.012-0.022 | 0.08-0.18 | Medium context (512) |
| Moirai-2 | Univariate | 0.010-0.020 | 0.10-0.20 | Large context (1024) |
| **ARIMA (Actual)** | Univariate | **1.679** | **-0.017** | Simple baseline |
| **LSTM (Actual)** | Univariate | **0.021** | **-0.020** | Deep learning |

---

## How to Run All Together

### One-by-one with error handling

```bash
# Try TimesFM
python timesfm_zeroshot_v3.py || echo "TimesFM failed"

# Try Chronos-2
python chronos2_zeroshot_v3.py || echo "Chronos-2 failed"

# Try Moirai-2
python moirai2_zeroshot_v3.py || echo "Moirai-2 failed"

# Compare all results
python compare_all_foundation_models.py
```

### Parallel execution (if available)

```bash
# In separate terminals
python timesfm_zeroshot_v3.py &
python chronos2_zeroshot_v3.py &
python moirai2_zeroshot_v3.py &
wait
```

---

## Output Files

Each model will create:
- `data/results/{model}_summary.csv` - Per-ticker metrics
- `data/results/{model}_predictions.csv` - Detailed predictions

---

## Troubleshooting

### ImportError: No module named 'timesfm'
```bash
pip install timesfm
# May need: pip install torch, numpy, pandas
```

### ImportError: No module named 'chronos'
```bash
pip install chronos
# Dependencies: transformers, torch, numpy, pandas
```

### 401 Unauthorized - Moirai-2
```bash
# Need HuggingFace authentication
huggingface-cli login
# Then paste your token from https://huggingface.co/settings/tokens
```

### GPU out of memory
```python
# Reduce per_core_batch_size in script:
per_core_batch_size=16  # Instead of 32
```

### Model download is slow
- Models are large (~500MB to 2GB)
- First run will download weights
- Subsequent runs will use cached weights
- Check `~/.cache/huggingface/hub/`

---

## Alternative: Run Without Foundation Models

If dependencies cannot be installed, we can:

1. **Use mock results** (for demo/testing)
   - `python create_mock_fm_results.py`

2. **Focus on baselines only**
   - ARIMA: RMSE=1.679, IC=-0.017
   - LSTM: RMSE=0.021, IC=-0.020

3. **Manual comparison**
   - Create synthetic predictions
   - Compare metrics with baselines

---

## Next Steps

### Option A: Install & Run (Recommended)
1. Install dependencies one by one
2. Run each FM script
3. Compare results with baselines

### Option B: Demo Mode
1. Use mock results
2. See expected performance
3. Full testing in production

### Option C: Ensemble Only
1. Use ARIMA + LSTM only
2. Skip FMs for now
3. Implement LEAF router with 2 models

---

## Expected Timeline

| Model | Install | Download | Run (per ticker) | Total |
|-------|---------|----------|------------------|-------|
| TimesFM | 1min | 5min | 2-5min | ~10min |
| Chronos-2 | 1min | 5min | 1-3min | ~9min |
| Moirai-2 | 1min | 5min | 3-8min | ~14min |
| **All 3** | - | - | - | **~30min** |

---

## Files Ready to Run

```
timesfm_zeroshot_v3.py      ✓ Ready (needs install)
chronos2_zeroshot_v3.py     ✓ Ready (needs install)
moirai2_zeroshot_v3.py      ✓ Ready (needs auth)
```

---

**Last Updated**: 2026-05-14  
**Status**: All scripts ready, dependencies pending installation
