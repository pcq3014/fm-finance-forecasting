"""
LEAF Framework Training Pipeline
================================
3 Phases:
  Phase 1: Walk-forward hyperparameter search (5 folds)
  Phase 2: Final training on full train set
  Phase 3: Test evaluation (1 time only)

Loss: CRPS + L1 regularization
Metrics: CRPS, MAE, RMSE, DA, IC per-regime
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
REGIME_PATH = ROOT_DIR / "data" / "processed" / "v2" / "regime.csv"
SELECTED_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_selected.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEQ_LEN = 60  # macro feature window
EMBEDDING_DIM = 32
CONTEXT_LENGTH = 32
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.1
PATIENCE = 10
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load train, test, regime labels."""
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])

    # Load regime labels
    regime_df = pd.read_csv(REGIME_PATH, parse_dates=["Date"])
    train = train.merge(regime_df[["Date", "regime_hmm"]], on="Date", how="left")
    test = test.merge(regime_df[["Date", "regime_hmm"]], on="Date", how="left")

    # Load selected features
    selected_features_df = pd.read_csv(SELECTED_FEATURES_PATH)
    selected_features = selected_features_df[
        selected_features_df["stage"] == "final"
    ]["feature"].tolist()

    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"  Tickers: {sorted(train['ticker'].unique())}")
    print(f"  Selected features ({len(selected_features)}): {selected_features}")

    return train, test, selected_features


def create_macro_sequences(
    df: pd.DataFrame, selected_features: list, seq_len: int = 60
) -> tuple:
    """
    Create macro feature sequences per ticker.

    Returns:
        X_macro: (n_samples, seq_len, n_features)
        y_targets: (n_samples,) target Close price
        dates: list of date indices
        tickers: list of ticker indices
    """
    tickers = sorted(df["ticker"].unique())
    X_macro_all = []
    y_targets_all = []
    dates_all = []
    tickers_all = []

    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker].sort_values("Date").reset_index(
            drop=True
        )

        # Macro features for this ticker
        X = ticker_data[selected_features].values.astype(np.float32)

        # Target: Close price (next day)
        y = ticker_data["Close"].values.astype(np.float32)

        # Create sequences
        for i in range(seq_len, len(X)):
            X_macro_all.append(X[i - seq_len : i])
            y_targets_all.append(y[i])
            dates_all.append(ticker_data.iloc[i]["Date"])
            tickers_all.append(ticker)

    X_macro = np.array(X_macro_all, dtype=np.float32)
    y_targets = np.array(y_targets_all, dtype=np.float32)

    print(f"  Created {len(X_macro):,} sequences")
    print(f"  Shape: {X_macro.shape}")

    return X_macro, y_targets, dates_all, tickers_all


def crps_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Simple CRPS approximation (assuming point forecast + small std).
    For now, use MAE as proxy.
    """
    return torch.mean(torch.abs(y_true - y_pred))


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device,
) -> float:
    """Train one epoch."""
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch, fm_c_batch, fm_m_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        fm_c_batch = fm_c_batch.to(device)
        fm_m_batch = fm_m_batch.to(device)

        optimizer.zero_grad()

        # Forward
        output = model(X_batch, fm_c_batch, fm_m_batch)
        ensemble_pred = output["ensemble_pred"].mean(dim=-1)  # (batch,)

        # Loss
        loss = crps_loss(y_batch, ensemble_pred)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validate_epoch(
    model: nn.Module,
    val_loader,
    device,
) -> float:
    """Validate one epoch."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch, fm_c_batch, fm_m_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            fm_c_batch = fm_c_batch.to(device)
            fm_m_batch = fm_m_batch.to(device)

            output = model(X_batch, fm_c_batch, fm_m_batch)
            ensemble_pred = output["ensemble_pred"].mean(dim=-1)

            loss = crps_loss(y_batch, ensemble_pred)
            epoch_loss += loss.item()

    return epoch_loss / len(val_loader)


def main():
    print("=" * 70)
    print("LEAF FRAMEWORK — TRAINING PIPELINE")
    print("=" * 70)

    # Load data
    train, test, selected_features = load_data()

    # Create macro sequences
    print("\nCreating train sequences...")
    X_train, y_train, dates_train, tickers_train = create_macro_sequences(
        train, selected_features, seq_len=SEQ_LEN
    )

    print("Creating test sequences...")
    X_test, y_test, dates_test, tickers_test = create_macro_sequences(
        test, selected_features, seq_len=SEQ_LEN
    )

    # TODO: Integrate with baseline FM predictions (Chronos, Moirai)
    # For now, create dummy FM predictions
    print("\nPreparing FM predictions...")
    fm_chronos_train = np.random.randn(len(X_train), 252).astype(np.float32)
    fm_moirai_train = np.random.randn(len(X_train), 252).astype(np.float32)
    fm_chronos_test = np.random.randn(len(X_test), 252).astype(np.float32)
    fm_moirai_test = np.random.randn(len(X_test), 252).astype(np.float32)

    print("\n" + "=" * 70)
    print("PHASE 1: WALK-FORWARD HYPERPARAMETER SEARCH")
    print("=" * 70)

    # 5-fold walk-forward splits (within train period)
    fold_configs = [
        {"name": "Fold1", "train_end": "2016-12-31", "val_start": "2017-01-01", "val_end": "2017-12-31"},
        {"name": "Fold2", "train_end": "2017-12-31", "val_start": "2018-01-01", "val_end": "2018-12-31"},
        {"name": "Fold3", "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31"},
        {"name": "Fold4", "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31"},
        {"name": "Fold5", "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    ]

    fold_results = []

    for fold_config in fold_configs:
        print(f"\n{fold_config['name']}:")
        fold_name = fold_config["name"]

        # Split data by date
        train_mask = np.array([d <= pd.Timestamp(fold_config["train_end"]) for d in dates_train])
        val_mask = np.array(
            [
                pd.Timestamp(fold_config["val_start"]) <= d <= pd.Timestamp(fold_config["val_end"])
                for d in dates_train
            ]
        )

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,}")

        # TODO: Train LEAF on X_tr, validate on X_val
        # For now, just record fold as placeholder
        fold_results.append({
            "fold": fold_name,
            "train_size": len(X_tr),
            "val_size": len(X_val),
        })

    print("\n" + "=" * 70)
    print("PHASE 2: FINAL TRAINING ON FULL TRAIN SET")
    print("=" * 70)
    print("(Using best hyperparameters from Phase 1)")

    # TODO: Train LEAF on full train set with best hyperparameters

    print("\n" + "=" * 70)
    print("PHASE 3: TEST EVALUATION (1 TIME ONLY)")
    print("=" * 70)

    # TODO: Evaluate on test set, save results

    print("\n✓ LEAF training pipeline complete")


if __name__ == "__main__":
    main()
