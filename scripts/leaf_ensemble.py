"""
LEAF: Lightweight Regime-Aware Ensemble Adapter
===============================================

Combines frozen foundation model forecasts (TimesFM, Chronos-2, Moirai-2)
using dynamic regime-aware weights based on market-macro context.

Framework: ViFinTSFM-LEAF for Vietnamese stock market forecasting

Input:
- Base forecasts from 3 frozen foundation models
- 60-day market-macro context window
- Market regime labels (bull/bear/sideway, low/high volatility)

Output:
- Dynamic weights for each model
- Final LEAF ensemble forecast
- Weight visualization and regime analysis
"""

import os
import random
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    """Set all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "results"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LEAF Configuration
CONTEXT_LEN = 60  # 60-day context window
CONTEXT_FEATURES = [
    'vn30_logret',
    'hsi_logret',
    'vn30_volume_log',
    'volume_ratio_20d',
    'trading_value_log',
    'foreign_net_ratio',
    'vnm_ret_lag1',
    'hsi_logret_lag1',
]
# Auxiliary scalar features fed directly into the router (see compute_aux_features)
AUX_FEATURES = [
    # Market state
    'vol_20d', 'vol_5d', 'trend_20d',
    # Per-FM rolling tracking
    'err_timesfm_20d', 'err_chronos2_20d', 'err_moirai2_20d',
    'da_timesfm_20d',  'da_chronos2_20d',  'da_moirai2_20d',
    # FM interaction / disagreement
    'disagreement_today', 'disagreement_20d',
    'err_corr_tf_ch_20d', 'err_corr_tf_mo_20d', 'err_corr_ch_mo_20d',
    # Ticker characteristics
    'ticker_vol_20d', 'ticker_vol_60d', 'ticker_beta_60d',
]
ROLL_WINDOW = 20
ENCODER_DIM = 32
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
SEEDS = [42, 123, 2024, 7, 1337]   # multiple seeds for variance estimation
# Entropy floor: penalize router collapse only when entropy drops below
# TARGET (* log(3)). Avoids pushing weights to fully uniform like a raw bonus does.
ENTROPY_TARGET_FRAC = 0.85       # target ≈ 0.85 * log(3) ≈ 0.935
ENTROPY_ALPHA       = 0.005      # multiplier on the deficit penalty
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegimeEncoder(nn.Module):
    """Encode market-macro context into regime embedding"""

    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim

        self.projection = nn.Linear(input_dim, 64)
        self.norm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, output_dim),
        )

    def forward(self, context):
        """
        Args:
            context: (batch, context_len, context_features)
        Returns:
            regime_embedding: (batch, output_dim)
        """
        # Project per timestep
        x = self.projection(context)  # (batch, context_len, 64)
        x = self.norm(x)

        # Temporal pooling: mean across context window
        x = x.mean(dim=1)  # (batch, 64)

        # MLP projection to regime embedding
        z_regime = self.mlp(x)  # (batch, output_dim)
        return z_regime


class DynamicRouter(nn.Module):
    """Route base forecasts based on regime embedding + auxiliary scalar features."""

    def __init__(self, regime_dim: int, aux_dim: int = 0, n_models: int = 3):
        super().__init__()
        self.n_models = n_models
        self.aux_dim = aux_dim

        in_dim = regime_dim + aux_dim
        self.router = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, n_models),
        )

    def forward(self, z_regime, aux=None):
        """
        Args:
            z_regime: (batch, regime_dim)
            aux:      (batch, aux_dim) or None
        Returns:
            weights: (batch, n_models), softmax normalized
        """
        x = z_regime if aux is None else torch.cat([z_regime, aux], dim=1)
        logits = self.router(x)
        weights = torch.softmax(logits, dim=1)
        return weights


class LEAF(nn.Module):
    """
    LEAF: Lightweight Regime-Aware Ensemble Adapter

    Architecture:
    context_window → RegimeEncoder → z_regime ─┐
                                               ├─→ DynamicRouter → weights
    aux scalar features ───────────────────────┘
    base_forecasts ──→ per-FM affine calibration (scale, bias) ──→ weighted sum

    Calibration decouples the magnitude problem (handled by scale/bias) from
    the ranking problem (handled by router) — allows both RMSE and IC to be
    optimized via the same Huber loss without trade-off.
    """

    def __init__(self, context_dim: int, aux_dim: int = 0,
                 encoder_dim: int = 32, n_models: int = 3, calibrate: bool = True):
        super().__init__()
        self.encoder = RegimeEncoder(context_dim, encoder_dim)
        self.router  = DynamicRouter(encoder_dim, aux_dim, n_models)
        self.calibrate = calibrate
        if calibrate:
            # Per-FM affine: y_i = scale_i * pred_i + bias_i (identity init)
            self.scale = nn.Parameter(torch.ones(n_models))
            self.bias  = nn.Parameter(torch.zeros(n_models))

    def forward(self, context, base_forecasts, aux=None):
        z_regime = self.encoder(context)
        weights  = self.router(z_regime, aux)
        if self.calibrate:
            base_forecasts = self.scale * base_forecasts + self.bias
        final_forecast = (base_forecasts * weights).sum(dim=1)
        return final_forecast, weights


def prepare_context_window(df: pd.DataFrame, context_features: list, context_len: int = 60):
    """Prepare rolling 60-day context windows for a SINGLE ticker df"""
    contexts = []
    for i in range(len(df)):
        start_idx = max(0, i - context_len)
        ctx = df[context_features].iloc[start_idx:i].values.astype(np.float32)
        if len(ctx) < context_len:
            pad = np.zeros((context_len - len(ctx), ctx.shape[1]), dtype=np.float32)
            ctx = np.vstack([pad, ctx])
        contexts.append(ctx)
    return np.array(contexts)


def build_context_tensors(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Build context tensors per-ticker, scaler fit on train only.
    Returns train/test tensors + fitted scaler.
    """
    def _stack_per_ticker(df):
        parts = []
        for ticker in sorted(df['ticker'].unique()):
            sub = df[df['ticker'] == ticker].reset_index(drop=True)
            parts.append(prepare_context_window(sub, CONTEXT_FEATURES, CONTEXT_LEN))
        return np.concatenate(parts, axis=0)  # (N, ctx_len, n_feat)

    ctx_tr = _stack_per_ticker(df_train)
    ctx_te = _stack_per_ticker(df_test)

    # Fit scaler on train only, transform both
    n_feat = ctx_tr.shape[-1]
    scaler = StandardScaler()
    ctx_tr = scaler.fit_transform(ctx_tr.reshape(-1, n_feat)).reshape(ctx_tr.shape)
    ctx_te = scaler.transform(ctx_te.reshape(-1, n_feat)).reshape(ctx_te.shape)

    return ctx_tr.astype(np.float32), ctx_te.astype(np.float32), scaler


FM_FILES = {
    'timesfm':  DATA_DIR / 'timesfm_predictions.csv',
    'chronos2': DATA_DIR / 'chronos2_predictions_v3.csv',
    'moirai2':  DATA_DIR / 'moirai2_predictions_v3.csv',
}

FM_VAL_FILES = {
    'timesfm':  DATA_DIR / 'timesfm_val_predictions.csv',
    'chronos2': DATA_DIR / 'chronos2_val_predictions.csv',
    'moirai2':  DATA_DIR / 'moirai2_val_predictions.csv',
}


def _merge_fm_dfs(dfs: dict) -> pd.DataFrame:
    """Merge three FM DataFrames on (date, ticker) inner join."""
    base = dfs['timesfm'].rename(columns={'y_pred': 'pred_timesfm', 'y_true': 'y_true'})
    base = base.merge(
        dfs['chronos2'][['date', 'ticker', 'y_pred']].rename(columns={'y_pred': 'pred_chronos2'}),
        on=['date', 'ticker'], how='inner'
    )
    base = base.merge(
        dfs['moirai2'][['date', 'ticker', 'y_pred']].rename(columns={'y_pred': 'pred_moirai2'}),
        on=['date', 'ticker'], how='inner'
    )
    return base.sort_values(['ticker', 'date']).reset_index(drop=True)


def _load_fm_set(file_dict: dict, label: str) -> pd.DataFrame | None:
    """Load and align predictions from a set of FM files."""
    dfs = {}
    for name, path in file_dict.items():
        if not path.exists():
            print(f"WARNING: {name} {label} predictions not found at {path}")
            return None
        df = pd.read_csv(path, parse_dates=['date'])
        if 'pred' in df.columns and 'y_pred' not in df.columns:
            df = df.rename(columns={'pred': 'y_pred'})
        dfs[name] = df[['date', 'ticker', 'y_pred', 'y_true']].copy()

    merged = _merge_fm_dfs(dfs)
    print(f"  {label}: {len(merged)} rows | {merged['ticker'].nunique()} tickers | "
          f"{merged['date'].min().date()} → {merged['date'].max().date()}")
    return merged


def load_base_forecasts():
    """Load test predictions from 3 FMs, return aligned DataFrame."""
    print("Loading test predictions...")
    return _load_fm_set(FM_FILES, 'test')


def load_val_forecasts():
    """Load val predictions from 3 FMs (used to train LEAF)."""
    print("Loading val predictions...")
    return _load_fm_set(FM_VAL_FILES, 'val')


def eval_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}


def compute_aux_features(val_df: pd.DataFrame, test_df: pd.DataFrame, panel_all: pd.DataFrame):
    """
    Compute auxiliary scalar features for the router (no leakage; shift by 1).

      market state:   vol_20d, vol_5d, trend_20d from vn30_logret
      FM tracking:    20d rolling MAE + DA per FM
      FM interaction: today's pred std, 20d pred-std rolling mean, pairwise err-corr
      ticker chars:   own ret vol (20d, 60d), beta to vn30 (60d)
    """
    # ── Market vol/trend + ticker characteristics from panel ─────────────────
    panel = panel_all.sort_values(['ticker', 'date']).copy()
    g_mkt = panel.groupby('ticker')['vn30_logret']
    panel['vol_20d']   = g_mkt.transform(lambda x: x.rolling(20, min_periods=5).std().shift(1))
    panel['vol_5d']    = g_mkt.transform(lambda x: x.rolling(5,  min_periods=2).std().shift(1))
    panel['trend_20d'] = g_mkt.transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))

    # Ticker characteristics from own return series (use shifted y_return_1d to avoid leak)
    g_own = panel.groupby('ticker')['y_return_1d']
    panel['ticker_vol_20d'] = g_own.transform(lambda x: x.rolling(20, min_periods=5).std().shift(1))
    panel['ticker_vol_60d'] = g_own.transform(lambda x: x.rolling(60, min_periods=20).std().shift(1))

    # Rolling beta = Cov(ret, vn30) / Var(vn30) per ticker, window=60
    panel = panel.reset_index(drop=True)
    beta_col = pd.Series(index=panel.index, dtype=float)
    for ticker, gdf in panel.groupby('ticker', sort=False):
        cov = gdf['y_return_1d'].rolling(60, min_periods=20).cov(gdf['vn30_logret'])
        var = gdf['vn30_logret'].rolling(60, min_periods=20).var()
        beta_col.loc[gdf.index] = (cov / var.replace(0, np.nan)).shift(1).values
    panel['ticker_beta_60d'] = beta_col

    panel_cols = ['date', 'ticker',
                  'vol_20d', 'vol_5d', 'trend_20d',
                  'ticker_vol_20d', 'ticker_vol_60d', 'ticker_beta_60d']

    # ── Rolling FM error/DA + disagreement from val+test predictions ─────────
    val_df  = val_df.copy();  val_df['_split']  = 'val'
    test_df = test_df.copy(); test_df['_split'] = 'test'
    combined = pd.concat([val_df, test_df], ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)

    pred_cols = ['pred_timesfm', 'pred_chronos2', 'pred_moirai2']

    # Per-FM rolling MAE & DA
    for fm in ['timesfm', 'chronos2', 'moirai2']:
        pred_col = f'pred_{fm}'
        combined[f'_err_{fm}']  = (combined[pred_col] - combined['y_true']).abs()
        combined[f'_dacc_{fm}'] = (np.sign(combined[pred_col]) == np.sign(combined['y_true'])).astype(float)
        combined[f'err_{fm}_20d'] = (
            combined.groupby('ticker')[f'_err_{fm}']
            .transform(lambda x: x.rolling(ROLL_WINDOW, min_periods=1).mean().shift(1))
        )
        combined[f'da_{fm}_20d'] = (
            combined.groupby('ticker')[f'_dacc_{fm}']
            .transform(lambda x: x.rolling(ROLL_WINDOW, min_periods=1).mean().shift(1))
        )

    # Disagreement: today's std across 3 FM preds
    combined['disagreement_today'] = combined[pred_cols].std(axis=1)
    combined['disagreement_20d']   = (
        combined.groupby('ticker')['disagreement_today']
        .transform(lambda x: x.rolling(ROLL_WINDOW, min_periods=5).mean().shift(1))
    )

    # Pairwise error correlations (rolling 20d)
    pairs = [('timesfm', 'chronos2', 'tf_ch'),
             ('timesfm', 'moirai2',  'tf_mo'),
             ('chronos2', 'moirai2', 'ch_mo')]
    for fm1, fm2, name in pairs:
        col = pd.Series(index=combined.index, dtype=float)
        for ticker, gdf in combined.groupby('ticker', sort=False):
            c = gdf[f'_err_{fm1}'].rolling(ROLL_WINDOW, min_periods=5).corr(gdf[f'_err_{fm2}']).shift(1)
            col.loc[gdf.index] = c.values
        combined[f'err_corr_{name}_20d'] = col

    # Merge panel features
    combined['date'] = pd.to_datetime(combined['date'])
    panel['date']    = pd.to_datetime(panel['date'])
    combined = combined.merge(panel[panel_cols], on=['date', 'ticker'], how='left')

    # Fill NaNs with neutral values
    fill_values = {
        'vol_20d': 0.0, 'vol_5d': 0.0, 'trend_20d': 0.0,
        'err_timesfm_20d': 0.0, 'err_chronos2_20d': 0.0, 'err_moirai2_20d': 0.0,
        'da_timesfm_20d': 0.5, 'da_chronos2_20d': 0.5, 'da_moirai2_20d': 0.5,
        'disagreement_today': 0.0, 'disagreement_20d': 0.0,
        'err_corr_tf_ch_20d': 0.0, 'err_corr_tf_mo_20d': 0.0, 'err_corr_ch_mo_20d': 0.0,
        'ticker_vol_20d': 0.0, 'ticker_vol_60d': 0.0, 'ticker_beta_60d': 1.0,
    }
    combined = combined.fillna(fill_values)

    val_out  = combined[combined['_split'] == 'val' ].drop(columns=['_split']).reset_index(drop=True)
    test_out = combined[combined['_split'] == 'test'].drop(columns=['_split']).reset_index(drop=True)
    return val_out, test_out


def create_regime_labels(df: pd.DataFrame):
    """
    Data-driven regime labels.
    Trend: vnm_ret_lag1 rolling 20d mean (momentum proxy)
    Vol:   volume_ratio_20d (volume activity proxy)
    """
    # Trend regime — 33rd/67th percentile of rolling 20d mean return
    trend_col = 'vnm_ret_lag1' if 'vnm_ret_lag1' in df.columns else None
    trend_signal = (
        df[trend_col].rolling(20, min_periods=1).mean().values
        if trend_col else np.zeros(len(df))
    )
    q33, q67 = np.nanpercentile(trend_signal, 33), np.nanpercentile(trend_signal, 67)
    trend_regime = np.where(
        trend_signal > q67, 'Bull',
        np.where(trend_signal < q33, 'Bear', 'Sideway')
    ).tolist()

    # Vol regime — volume_ratio_20d (>1 = high activity)
    vol_col = 'volume_ratio_20d' if 'volume_ratio_20d' in df.columns else None
    vol_signal = df[vol_col].values if vol_col else np.ones(len(df))
    vol_p66 = np.nanpercentile(vol_signal[np.isfinite(vol_signal)], 66)
    vol_regime = np.where(vol_signal > vol_p66, 'HighVol', 'LowVol').tolist()

    return trend_regime, vol_regime


def main():
    print("\n" + "="*80)
    print("LEAF: Lightweight Regime-Aware Ensemble Adapter")
    print("="*80)
    print(f"Context length: {CONTEXT_LEN} days")
    print(f"Encoder dimension: {ENCODER_DIM}")
    print(f"Device: {DEVICE}\n")

    # Load FM predictions: val for LEAF training, test for LEAF evaluation
    val_df  = load_val_forecasts()
    test_df = load_base_forecasts()
    if val_df is None or test_df is None:
        print("\nERROR: Run all FM scripts first to generate val and test predictions.")
        return

    # Load panel data for context features
    panel_path = ROOT_DIR / "data" / "processed" / "panel"
    df_panel_train = pd.read_csv(panel_path / "panel_train_fixed.csv")
    df_panel_val   = pd.read_csv(panel_path / "panel_val_fixed.csv")
    df_panel_test  = pd.read_csv(panel_path / "panel_test_fixed.csv")
    for df in (df_panel_train, df_panel_val, df_panel_test):
        df['date'] = pd.to_datetime(df['date'])

    # Attach context features to val predictions (LEAF train)
    val_df = val_df.merge(
        df_panel_val[['date', 'ticker'] + CONTEXT_FEATURES],
        on=['date', 'ticker'], how='left'
    )
    val_df[CONTEXT_FEATURES] = val_df[CONTEXT_FEATURES].fillna(0.0)

    # Attach context features to test predictions (LEAF eval)
    test_df = test_df.merge(
        df_panel_test[['date', 'ticker'] + CONTEXT_FEATURES],
        on=['date', 'ticker'], how='left'
    )
    test_df[CONTEXT_FEATURES] = test_df[CONTEXT_FEATURES].fillna(0.0)

    # LEAF train = val period (out-of-sample for FMs, no leakage)
    # LEAF eval  = test period (full 416 trading days, comparable to ARIMA/LSTM)
    df_leaf_train = val_df.reset_index(drop=True)
    df_leaf_test  = test_df.reset_index(drop=True)
    print(f"\nLEAF train (val period): {len(df_leaf_train)} rows")
    print(f"LEAF eval  (test period): {len(df_leaf_test)} rows\n")

    # ── Compute auxiliary scalar features (rolling FM tracking + market vol) ──
    df_panel_all = pd.concat([df_panel_train, df_panel_val, df_panel_test]).sort_values(['ticker', 'date'])
    print("Computing rolling FM-tracking & market-vol aux features...")
    df_leaf_train, df_leaf_test = compute_aux_features(df_leaf_train, df_leaf_test, df_panel_all)

    def to_base_tensor(df):
        return torch.tensor(
            np.stack([df['pred_timesfm'].values,
                      df['pred_chronos2'].values,
                      df['pred_moirai2'].values], axis=1).astype(np.float32),
            dtype=torch.float32
        ).to(DEVICE)

    def to_aux_tensor(df):
        return torch.tensor(df[AUX_FEATURES].values.astype(np.float32), dtype=torch.float32).to(DEVICE)

    # Build context windows per-ticker; include train history so val rows
    # can look back beyond the val period boundary.
    df_panel_train_val = pd.concat([df_panel_train, df_panel_val]).sort_values(['ticker', 'date'])

    def _build_with_history(df_pred, df_full_panel):
        """
        For each ticker, build context windows using full historical panel
        (so the first rows of the period don't get all-zero context).
        Returns (N, CONTEXT_LEN, n_feat) array aligned to df_pred rows.
        """
        parts = []
        for ticker in sorted(df_pred['ticker'].unique()):
            sub_pred  = df_pred[df_pred['ticker'] == ticker].sort_values('date')
            sub_panel = df_full_panel[df_full_panel['ticker'] == ticker].sort_values('date').reset_index(drop=True)

            # Map each prediction date to its panel index
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

    print("Preparing tensors (per-ticker context with full history, scaler fit on val)...")
    ctx_tr_np = _build_with_history(df_leaf_train, df_panel_train_val)
    ctx_te_np = _build_with_history(df_leaf_test,  df_panel_all)

    # Fit scaler on val (LEAF train) context, transform test
    n_feat = ctx_tr_np.shape[-1]
    scaler = StandardScaler()
    ctx_tr_np = scaler.fit_transform(ctx_tr_np.reshape(-1, n_feat)).reshape(ctx_tr_np.shape).astype(np.float32)
    ctx_te_np = scaler.transform(ctx_te_np.reshape(-1, n_feat)).reshape(ctx_te_np.shape).astype(np.float32)

    ctx_tr  = torch.tensor(ctx_tr_np, dtype=torch.float32).to(DEVICE)
    ctx_te  = torch.tensor(ctx_te_np, dtype=torch.float32).to(DEVICE)
    base_tr = to_base_tensor(df_leaf_train)
    base_te = to_base_tensor(df_leaf_test)

    # Aux scalar features — scaled with same fit-on-train logic
    aux_scaler = StandardScaler()
    aux_tr_np = aux_scaler.fit_transform(df_leaf_train[AUX_FEATURES].values).astype(np.float32)
    aux_te_np = aux_scaler.transform(df_leaf_test[AUX_FEATURES].values).astype(np.float32)
    aux_tr  = torch.tensor(aux_tr_np, dtype=torch.float32).to(DEVICE)
    aux_te  = torch.tensor(aux_te_np, dtype=torch.float32).to(DEVICE)

    y_tr    = torch.tensor(df_leaf_train['y_true'].values.astype(np.float32)).to(DEVICE)
    y_te    = torch.tensor(df_leaf_test['y_true'].values.astype(np.float32)).to(DEVICE)

    # ── Train & evaluate over multiple seeds ─────────────────────────────────
    max_entropy    = float(np.log(3))
    target_entropy = ENTROPY_TARGET_FRAC * max_entropy
    print(f"Training LEAF over {len(SEEDS)} seeds (entropy_target={target_entropy:.3f}/{max_entropy:.3f}, "
          f"α={ENTROPY_ALPHA}, aux_dim={len(AUX_FEATURES)})...")

    per_seed_preds   = []   # (n_seeds, N_test) — for averaging
    per_seed_weights = []   # (n_seeds, N_test, 3)
    per_seed_metrics = []   # list of dicts

    for seed in SEEDS:
        set_seed(seed)

        leaf = LEAF(
            context_dim=len(CONTEXT_FEATURES),
            aux_dim=len(AUX_FEATURES),
            encoder_dim=ENCODER_DIM,
            n_models=3,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(leaf.parameters(), lr=LR)
        criterion = nn.HuberLoss(delta=0.01)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            leaf.train()
            idx = torch.randperm(len(ctx_tr))
            for i in range(0, len(ctx_tr), BATCH_SIZE):
                b = idx[i:i+BATCH_SIZE]
                optimizer.zero_grad()
                pred, w = leaf(ctx_tr[b], base_tr[b], aux_tr[b])
                huber = criterion(pred, y_tr[b])
                entropy = -(w * torch.log(w + 1e-9)).sum(dim=1).mean()
                deficit = torch.clamp(target_entropy - entropy, min=0.0)
                loss = huber + ENTROPY_ALPHA * deficit
                loss.backward()
                optimizer.step()

            leaf.eval()
            with torch.no_grad():
                val_pred, _ = leaf(ctx_te, base_te, aux_te)
                val_loss = criterion(val_pred, y_te).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in leaf.state_dict().items()}

        leaf.load_state_dict(best_state)
        leaf.eval()
        with torch.no_grad():
            seed_preds, seed_w = leaf(ctx_te, base_te, aux_te)
            seed_preds = seed_preds.cpu().numpy()
            seed_w     = seed_w.cpu().numpy()

        m = eval_metrics(df_leaf_test['y_true'].values, seed_preds)
        per_seed_preds.append(seed_preds)
        per_seed_weights.append(seed_w)
        per_seed_metrics.append(m)
        print(f"  seed={seed:>4d}: RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  "
              f"IC={m['ic']:+.4f}  DA={m['dir_acc']:.4f}")

    per_seed_preds   = np.stack(per_seed_preds)     # (S, N)
    per_seed_weights = np.stack(per_seed_weights)   # (S, N, 3)

    # Ensemble across seeds — averaged predictions are far more stable than any single seed
    final_preds = per_seed_preds.mean(axis=0)
    weights_all = per_seed_weights.mean(axis=0)

    # Print seed-variance summary
    print("\n" + "-"*80)
    print(f"Variance across {len(SEEDS)} seeds:")
    for k in ['rmse', 'mae', 'ic', 'dir_acc']:
        vals = np.array([m[k] for m in per_seed_metrics])
        print(f"  {k:<8s}: mean={vals.mean():+.5f}  std={vals.std():.5f}  "
              f"min={vals.min():+.5f}  max={vals.max():+.5f}")
    ens_m = eval_metrics(df_leaf_test['y_true'].values, final_preds)
    print(f"  --- ensemble (avg of {len(SEEDS)} seeds) ---")
    print(f"  RMSE={ens_m['rmse']:.5f}  MAE={ens_m['mae']:.5f}  "
          f"IC={ens_m['ic']:+.4f}  DA={ens_m['dir_acc']:.4f}")
    print("-"*80 + "\n")

    y_true_eval = df_leaf_test['y_true'].values
    base_preds_eval = np.stack([
        df_leaf_test['pred_timesfm'].values,
        df_leaf_test['pred_chronos2'].values,
        df_leaf_test['pred_moirai2'].values,
    ], axis=1)

    equal_preds = base_preds_eval.mean(axis=1)
    trend_regimes, vol_regimes = create_regime_labels(df_leaf_test)

    # ── Per-ticker summary ────────────────────────────────────────────────────
    ticker_rows = []
    for ticker in sorted(df_leaf_test['ticker'].unique()):
        mask = df_leaf_test['ticker'].values == ticker
        row = {'ticker': ticker}
        for label, preds in [
            ('leaf', final_preds[mask]),
            ('equal', equal_preds[mask]),
            ('timesfm', base_preds_eval[mask, 0]),
            ('chronos2', base_preds_eval[mask, 1]),
            ('moirai2', base_preds_eval[mask, 2]),
        ]:
            m = eval_metrics(y_true_eval[mask], preds)
            row[f'rmse_{label}'] = m['rmse']
            row[f'mae_{label}']  = m['mae']
            row[f'ic_{label}']   = m['ic']
            row[f'da_{label}']   = m['dir_acc']
        row['n'] = mask.sum()
        ticker_rows.append(row)

    # Overall row
    overall_row = {'ticker': 'OVERALL'}
    for label, preds in [
        ('leaf', final_preds), ('equal', equal_preds),
        ('timesfm', base_preds_eval[:, 0]),
        ('chronos2', base_preds_eval[:, 1]),
        ('moirai2',  base_preds_eval[:, 2]),
    ]:
        m = eval_metrics(y_true_eval, preds)
        overall_row[f'rmse_{label}'] = m['rmse']
        overall_row[f'mae_{label}']  = m['mae']
        overall_row[f'ic_{label}']   = m['ic']
        overall_row[f'da_{label}']   = m['dir_acc']
    overall_row['n'] = len(y_true_eval)
    ticker_rows.append(overall_row)

    summary_df = pd.DataFrame(ticker_rows)

    print("\n" + "="*80)
    print("LEAF SUMMARY — per ticker")
    print("="*80)
    print(summary_df.round(6).to_string(index=False))

    # ── Model comparison table ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("MODEL COMPARISON (overall)")
    print("="*80)
    comp_rows = []
    for label, name, preds in [
        ('leaf',     'LEAF',           final_preds),
        ('equal',    'Equal-Weight',   equal_preds),
        ('timesfm',  'TimesFM',        base_preds_eval[:, 0]),
        ('chronos2', 'Chronos-2',      base_preds_eval[:, 1]),
        ('moirai2',  'Moirai-2',       base_preds_eval[:, 2]),
    ]:
        m = eval_metrics(y_true_eval, preds)
        comp_rows.append({'model': name, **m})
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.round(6).to_string(index=False))

    # ── Regime analysis ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("REGIME-WISE ANALYSIS")
    print("="*80)
    regime_rows = []
    for regime in ['Bull', 'Bear', 'Sideway']:
        mask = np.array(trend_regimes) == regime
        if mask.sum() == 0:
            continue
        m = eval_metrics(y_true_eval[mask], final_preds[mask])
        avg_w = weights_all[mask].mean(axis=0)
        regime_rows.append({
            'regime': regime, 'n': int(mask.sum()),
            **{k: round(v, 6) for k, v in m.items()},
            'w_timesfm': round(float(avg_w[0]), 3),
            'w_chronos2': round(float(avg_w[1]), 3),
            'w_moirai2': round(float(avg_w[2]), 3),
        })
    print(pd.DataFrame(regime_rows).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        'date':         df_leaf_test['date'].values,
        'ticker':       df_leaf_test['ticker'].values,
        'y_true':       y_true_eval,
        'leaf_pred':    final_preds,
        'equal_pred':   equal_preds,
        'w_timesfm':    weights_all[:, 0],
        'w_chronos2':   weights_all[:, 1],
        'w_moirai2':    weights_all[:, 2],
        'trend_regime': trend_regimes,
        'vol_regime':   vol_regimes,
    })

    results_df.to_csv(OUT_DIR / "leaf_predictions.csv", index=False)
    summary_df.to_csv(OUT_DIR / "leaf_summary.csv", index=False)
    comp_df.to_csv(OUT_DIR / "leaf_model_comparison.csv", index=False)

    print(f"\n{'='*80}")
    print(f"Saved: {OUT_DIR / 'leaf_predictions.csv'}")
    print(f"Saved: {OUT_DIR / 'leaf_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'leaf_model_comparison.csv'}")
    print("="*80)


if __name__ == "__main__":
    main()
