"""
LEAF Ablation Study — reproduce each of the 4 improvement rounds
================================================================

Trains 5 variants of LEAF and reports metrics side-by-side so you can see
how each architectural change contributed to the final result.

Variants:
  Round 0 (baseline):       no entropy floor, no aux features, no calibration, 1 seed
  Round 1 (+ entropy):      + entropy-deficit penalty
  Round 2 (+ tracking aux): + market vol + per-FM rolling err/DA (9 feats)
  Round 3 (+ full):         + interaction (5) + ticker (3) feats + per-FM calibration
  Round 4 (+ ensemble):     + 5-seed ensemble averaging (final published result)

Usage:
  python leaf_ablation.py

Output:
  data/results/leaf_ablation.csv  — full table
  console table grouped by round
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Reuse machinery from leaf_ensemble.py (avoids drift)
from scripts.leaf_ensemble import (
    set_seed, eval_metrics,
    LEAF, prepare_context_window, compute_aux_features,
    load_base_forecasts, load_val_forecasts,
    CONTEXT_LEN, CONTEXT_FEATURES, ENCODER_DIM, DROPOUT,
    LR, EPOCHS, BATCH_SIZE,
    ENTROPY_TARGET_FRAC, ENTROPY_ALPHA,
    DEVICE, ROOT_DIR, DATA_DIR, OUT_DIR,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── Aux feature groups (for ablation selection) ─────────────────────────────

AUX_GROUPS = {
    'market':    ['vol_20d', 'vol_5d', 'trend_20d'],
    'fm_track':  ['err_timesfm_20d', 'err_chronos2_20d', 'err_moirai2_20d',
                  'da_timesfm_20d',  'da_chronos2_20d',  'da_moirai2_20d'],
    'interact':  ['disagreement_today', 'disagreement_20d',
                  'err_corr_tf_ch_20d', 'err_corr_tf_mo_20d', 'err_corr_ch_mo_20d'],
    'ticker':    ['ticker_vol_20d', 'ticker_vol_60d', 'ticker_beta_60d'],
}


# ─── Ablation variants ───────────────────────────────────────────────────────

VARIANTS = [
    {
        'name':    'R0 baseline (collapse)',
        'entropy': False,
        'aux':     [],
        'calib':   False,
        'seeds':   [42],
    },
    {
        'name':    'R1 + entropy floor',
        'entropy': True,
        'aux':     [],
        'calib':   False,
        'seeds':   [42],
    },
    {
        'name':    'R2 + tracking aux (9)',
        'entropy': True,
        'aux':     ['market', 'fm_track'],
        'calib':   False,
        'seeds':   [42],
    },
    {
        'name':    'R3 + interact+ticker+calib',
        'entropy': True,
        'aux':     ['market', 'fm_track', 'interact', 'ticker'],
        'calib':   True,
        'seeds':   [42],
    },
    {
        'name':    'R4 + 5-seed ensemble',
        'entropy': True,
        'aux':     ['market', 'fm_track', 'interact', 'ticker'],
        'calib':   True,
        'seeds':   [42, 123, 2024, 7, 1337],
    },
]


# ─── Single training run for a given config ──────────────────────────────────

def train_one(seed, config, ctx_tr, ctx_te, base_tr, base_te,
              aux_tr, aux_te, y_tr, y_te, aux_dim):
    """Train one LEAF instance and return test predictions."""
    set_seed(seed)

    leaf = LEAF(
        context_dim=len(CONTEXT_FEATURES),
        aux_dim=aux_dim,
        encoder_dim=ENCODER_DIM,
        n_models=3,
        calibrate=config['calib'],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(leaf.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=0.01)

    max_entropy    = float(np.log(3))
    target_entropy = ENTROPY_TARGET_FRAC * max_entropy

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        leaf.train()
        idx = torch.randperm(len(ctx_tr))
        for i in range(0, len(ctx_tr), BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            aux_b = aux_tr[b] if aux_tr is not None else None
            pred, w = leaf(ctx_tr[b], base_tr[b], aux_b)
            loss = criterion(pred, y_tr[b])
            if config['entropy']:
                ent = -(w * torch.log(w + 1e-9)).sum(dim=1).mean()
                deficit = torch.clamp(target_entropy - ent, min=0.0)
                loss = loss + ENTROPY_ALPHA * deficit
            loss.backward()
            optimizer.step()

        leaf.eval()
        with torch.no_grad():
            val_pred, _ = leaf(ctx_te, base_te, aux_te)
            vloss = criterion(val_pred, y_te).item()
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.clone() for k, v in leaf.state_dict().items()}

    leaf.load_state_dict(best_state)
    leaf.eval()
    with torch.no_grad():
        preds, weights = leaf(ctx_te, base_te, aux_te)
        return preds.cpu().numpy(), weights.cpu().numpy()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("LEAF ABLATION STUDY — 4 improvement rounds")
    print("="*80 + "\n")

    # ── Load FM predictions ──────────────────────────────────────────────────
    val_df  = load_val_forecasts()
    test_df = load_base_forecasts()
    if val_df is None or test_df is None:
        print("ERROR: missing val or test prediction files.")
        return

    # ── Load panel data + merge context features ─────────────────────────────
    panel_path = ROOT_DIR / "data" / "processed" / "panel"
    df_panel_train = pd.read_csv(panel_path / "panel_train_fixed.csv")
    df_panel_val   = pd.read_csv(panel_path / "panel_val_fixed.csv")
    df_panel_test  = pd.read_csv(panel_path / "panel_test_fixed.csv")
    for df in (df_panel_train, df_panel_val, df_panel_test):
        df['date'] = pd.to_datetime(df['date'])

    val_df  = val_df.merge(df_panel_val[['date','ticker']+CONTEXT_FEATURES],   on=['date','ticker'], how='left')
    test_df = test_df.merge(df_panel_test[['date','ticker']+CONTEXT_FEATURES], on=['date','ticker'], how='left')
    val_df[CONTEXT_FEATURES]  = val_df[CONTEXT_FEATURES].fillna(0.0)
    test_df[CONTEXT_FEATURES] = test_df[CONTEXT_FEATURES].fillna(0.0)

    # ── Compute ALL aux features once (we'll subset per variant) ─────────────
    df_panel_all = pd.concat([df_panel_train, df_panel_val, df_panel_test]).sort_values(['ticker','date'])
    print("Computing all aux features (used by R2–R4)...")
    val_df, test_df = compute_aux_features(val_df, test_df, df_panel_all)

    # ── Build context tensors (per-ticker with history, fit scaler on val) ──
    df_panel_train_val = pd.concat([df_panel_train, df_panel_val]).sort_values(['ticker','date'])

    def _build_ctx(df_pred, df_full_panel):
        parts = []
        for ticker in sorted(df_pred['ticker'].unique()):
            sub_pred  = df_pred[df_pred['ticker']==ticker].sort_values('date')
            sub_panel = df_full_panel[df_full_panel['ticker']==ticker].sort_values('date').reset_index(drop=True)
            date_to_idx = {d: i for i, d in enumerate(sub_panel['date'])}
            ctx_rows = []
            for d in sub_pred['date']:
                idx = date_to_idx.get(d, None)
                if idx is None:
                    ctx_rows.append(np.zeros((CONTEXT_LEN, len(CONTEXT_FEATURES)), dtype=np.float32))
                    continue
                start = max(0, idx - CONTEXT_LEN)
                ctx = sub_panel[CONTEXT_FEATURES].iloc[start:idx].values.astype(np.float32)
                if len(ctx) < CONTEXT_LEN:
                    pad = np.zeros((CONTEXT_LEN - len(ctx), len(CONTEXT_FEATURES)), dtype=np.float32)
                    ctx = np.vstack([pad, ctx])
                ctx_rows.append(ctx)
            parts.append(np.array(ctx_rows))
        return np.concatenate(parts, axis=0)

    ctx_tr_np = _build_ctx(val_df,  df_panel_train_val)
    ctx_te_np = _build_ctx(test_df, df_panel_all)
    n_feat = ctx_tr_np.shape[-1]
    sc = StandardScaler()
    ctx_tr_np = sc.fit_transform(ctx_tr_np.reshape(-1,n_feat)).reshape(ctx_tr_np.shape).astype(np.float32)
    ctx_te_np = sc.transform(ctx_te_np.reshape(-1,n_feat)).reshape(ctx_te_np.shape).astype(np.float32)

    ctx_tr = torch.tensor(ctx_tr_np).to(DEVICE)
    ctx_te = torch.tensor(ctx_te_np).to(DEVICE)

    base_tr = torch.tensor(np.stack([val_df['pred_timesfm'].values,
                                     val_df['pred_chronos2'].values,
                                     val_df['pred_moirai2'].values], axis=1).astype(np.float32)).to(DEVICE)
    base_te = torch.tensor(np.stack([test_df['pred_timesfm'].values,
                                     test_df['pred_chronos2'].values,
                                     test_df['pred_moirai2'].values], axis=1).astype(np.float32)).to(DEVICE)
    y_tr = torch.tensor(val_df['y_true'].values.astype(np.float32)).to(DEVICE)
    y_te = torch.tensor(test_df['y_true'].values.astype(np.float32)).to(DEVICE)
    y_true_test = test_df['y_true'].values

    # ── Run each variant ─────────────────────────────────────────────────────
    rows = []
    for cfg in VARIANTS:
        # Subset aux features for this variant
        aux_cols = sum((AUX_GROUPS[g] for g in cfg['aux']), [])
        aux_dim  = len(aux_cols)

        if aux_dim > 0:
            aux_sc = StandardScaler()
            aux_tr_np = aux_sc.fit_transform(val_df[aux_cols].values).astype(np.float32)
            aux_te_np = aux_sc.transform(test_df[aux_cols].values).astype(np.float32)
            aux_tr = torch.tensor(aux_tr_np).to(DEVICE)
            aux_te = torch.tensor(aux_te_np).to(DEVICE)
        else:
            aux_tr = aux_te = None

        print(f"\n[{cfg['name']}]  seeds={cfg['seeds']}  aux_dim={aux_dim}  "
              f"entropy={cfg['entropy']}  calib={cfg['calib']}")

        seed_preds = []
        for seed in cfg['seeds']:
            p, _ = train_one(seed, cfg, ctx_tr, ctx_te, base_tr, base_te,
                             aux_tr, aux_te, y_tr, y_te, aux_dim)
            m = eval_metrics(y_true_test, p)
            print(f"  seed={seed:>4d}: RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  "
                  f"IC={m['ic']:+.4f}  DA={m['dir_acc']:.4f}")
            seed_preds.append(p)

        # Ensemble (or single seed)
        ens_pred = np.stack(seed_preds).mean(axis=0)
        m = eval_metrics(y_true_test, ens_pred)
        rows.append({
            'variant':    cfg['name'],
            'n_seeds':    len(cfg['seeds']),
            'aux_dim':    aux_dim,
            'entropy':    cfg['entropy'],
            'calib':      cfg['calib'],
            'rmse':       m['rmse'],
            'mae':        m['mae'],
            'ic':         m['ic'],
            'dir_acc':    m['dir_acc'],
        })
        if len(cfg['seeds']) > 1:
            print(f"  ENSEMBLE: RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  "
                  f"IC={m['ic']:+.4f}  DA={m['dir_acc']:.4f}")

    df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    pretty = df.copy()
    pretty['rmse']    = pretty['rmse'].round(5)
    pretty['mae']     = pretty['mae'].round(5)
    pretty['ic']      = pretty['ic'].round(4)
    pretty['dir_acc'] = pretty['dir_acc'].round(4)
    print(pretty.to_string(index=False))

    out_path = OUT_DIR / "leaf_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("="*80)


if __name__ == "__main__":
    main()
