"""
LEAF Framework — Practical Implementation
==========================================
Uses pre-computed baseline predictions (Chronos, Moirai) from:
  - data/results/chronos_results.csv
  - data/results/moirai_results.csv

LEAF learns to:
  1. Encode macro features → regime embedding
  2. Weight Chronos + Moirai predictions based on regime
  3. Produce better ensemble prediction

Training:
  Phase 1: Walk-forward validation (5 folds)
  Phase 2: Final training on full train set
  Phase 3: Test evaluation (1 time only)

Loss: MSE (y_true - ensemble_pred)^2
Metrics: MAE, RMSE, DA (Directional Accuracy), IC
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
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
SELECTED_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_selected.csv"
LSTM_RESULTS = ROOT_DIR / "data" / "results" / "lstm_close_results.csv"
MOIRAI_RESULTS = ROOT_DIR / "data" / "results" / "moirai_results.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEQ_LEN = 60
EMBEDDING_DIM = 32
CONTEXT_LENGTH = 32
EPOCHS = 20  # Reduced for quick testing (Phase 1)
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.1
PATIENCE = 5  # Reduced for quick testing
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import LEAF modules
sys.path.insert(0, str(ROOT_DIR / "src" / "models"))
from regime_encoder import RegimeEncoder
from fm_ensemble import FMEnsemble


class LEAFEnsemble(nn.Module):
    """
    Simplified LEAF: Regime Encoder + FM Weighting.

    No Context Projector (not needed for fixed FM predictions).
    """

    def __init__(
        self,
        n_features: int = 11,
        embedding_dim: int = 32,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.regime_encoder = RegimeEncoder(
            n_features=n_features,
            seq_len=SEQ_LEN,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).to(device)

        self.fm_ensemble = FMEnsemble(
            embedding_dim=embedding_dim, n_fms=2, dropout=dropout
        ).to(device)

    def forward(
        self,
        macro_features: torch.Tensor,
        fm_pred_chronos: torch.Tensor,
        fm_pred_moirai: torch.Tensor,
    ) -> dict:
        """
        Args:
            macro_features: (batch, seq_len, n_features)
            fm_pred_chronos: (batch,) single prediction per sample
            fm_pred_moirai: (batch,) single prediction per sample

        Returns:
            dict with ensemble_pred, weights, etc.
        """
        # Encode macro features
        regime_embedding = self.regime_encoder(macro_features)

        # Ensemble with learned weights
        # Reshape FM predictions to (batch, 1) for ensemble
        ensemble_pred, weights = self.fm_ensemble(
            regime_embedding,
            [fm_pred_chronos.unsqueeze(-1), fm_pred_moirai.unsqueeze(-1)],
        )

        # Squeeze back to (batch,)
        ensemble_pred = ensemble_pred.squeeze(-1)

        return {
            "ensemble_pred": ensemble_pred,
            "regime_embedding": regime_embedding,
            "weights": weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_data() -> tuple:
    """Load train, test, baseline predictions, and features."""
    print("Loading data...")

    # Load raw data
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])

    # Load baseline predictions (LSTM + Moirai)
    lstm = pd.read_csv(LSTM_RESULTS, parse_dates=["Date"])
    moirai = pd.read_csv(MOIRAI_RESULTS, parse_dates=["Date"])

    # Selected features (from feature_selection_report.csv)
    selected_features = [
        "vnm_ret",
        "risk_off_composite",
        "vix_dev",
        "hsi_ret_1d",
        "vix",
        "macd",
        "hsi_vol_20",
        "usdvnd_dev",
        "ibor_rate",
        "ibor_event",
        "pmi_norm",
    ]

    print(f"  Train: {len(train):,} rows")
    print(f"  Test: {len(test):,} rows")
    print(f"  LSTM predictions: {len(lstm):,} rows (non-zero: {(lstm['pred'] != 0).sum()})")
    print(f"  Moirai predictions: {len(moirai):,} rows (non-zero: {(moirai['pred'] != 0).sum()})")
    print(f"  Selected features ({len(selected_features)}): {selected_features}")

    return train, test, lstm, moirai, selected_features


def create_train_dataset(
    train: pd.DataFrame,
    lstm: pd.DataFrame,
    moirai: pd.DataFrame,
    selected_features: list,
    seq_len: int = 60,
) -> tuple:
    """
    Create training dataset.

    Returns:
        X_macro: (n_samples, seq_len, n_features)
        y_target: (n_samples,)
        fm_lstm: (n_samples,)
        fm_moirai: (n_samples,)
        dates: list of Date
        tickers: list of ticker
    """
    # Merge baseline predictions (LSTM + Moirai)
    train_with_fm = train.merge(
        lstm[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_lstm"}),
        on=["Date", "ticker"],
        how="left",
    ).merge(
        moirai[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_moirai"}),
        on=["Date", "ticker"],
        how="left",
    )

    X_macro_all = []
    y_target_all = []
    fm_lstm_all = []
    fm_moirai_all = []
    dates_all = []
    tickers_all = []

    tickers = sorted(train_with_fm["ticker"].unique())

    for ticker in tickers:
        ticker_data = train_with_fm[train_with_fm["ticker"] == ticker].sort_values(
            "Date"
        ).reset_index(drop=True)

        # Features
        X = ticker_data[selected_features].values.astype(np.float32)
        y = ticker_data["Close"].values.astype(np.float32)

        # Baseline predictions (LSTM + Moirai)
        fm_lstm = ticker_data["pred_lstm"].values.astype(np.float32)
        fm_moirai = ticker_data["pred_moirai"].values.astype(np.float32)

        # Create sequences
        for i in range(seq_len, len(X)):
            X_macro_all.append(X[i - seq_len : i])
            y_target_all.append(y[i])
            fm_lstm_all.append(fm_lstm[i])
            fm_moirai_all.append(fm_moirai[i])
            dates_all.append(ticker_data.iloc[i]["Date"])
            tickers_all.append(ticker)

    return (
        np.array(X_macro_all, dtype=np.float32),
        np.array(y_target_all, dtype=np.float32),
        np.array(fm_lstm_all, dtype=np.float32),
        np.array(fm_moirai_all, dtype=np.float32),
        dates_all,
        tickers_all,
    )


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def main():
    print("=" * 70)
    print("LEAF FRAMEWORK — PRACTICAL IMPLEMENTATION (LSTM + Moirai Ensemble)")
    print("=" * 70)
    print("NOTE: Using test set only (train period baselines not available)")
    print("=" * 70)

    # Load data
    train, test, lstm, moirai, selected_features = load_data()

    # Create TRAIN dataset on TEST period (baseline predictions available)
    print("\nCreating training dataset from TEST period (with baseline predictions)...")
    (
        X_train,
        y_train,
        fm_lstm_train,
        fm_moirai_train,
        dates_train,
        tickers_train,
    ) = create_train_dataset(test, lstm, moirai, selected_features, seq_len=SEQ_LEN)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Create test dataset
    print("Creating test dataset...")
    test_with_fm = test.merge(
        lstm[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_lstm"}),
        on=["Date", "ticker"],
        how="left",
    ).merge(
        moirai[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_moirai"}),
        on=["Date", "ticker"],
        how="left",
    )

    X_test_all = []
    y_test_all = []
    fm_lstm_test_all = []
    fm_moirai_test_all = []
    dates_test_all = []
    tickers_test_all = []

    tickers = sorted(test_with_fm["ticker"].unique())
    for ticker in tickers:
        ticker_data = test_with_fm[test_with_fm["ticker"] == ticker].sort_values(
            "Date"
        ).reset_index(drop=True)

        X = ticker_data[selected_features].values.astype(np.float32)
        y = ticker_data["Close"].values.astype(np.float32)
        fm_lstm = ticker_data["pred_lstm"].values.astype(np.float32)
        fm_moirai = ticker_data["pred_moirai"].values.astype(np.float32)

        for i in range(SEQ_LEN, len(X)):
            X_test_all.append(X[i - SEQ_LEN : i])
            y_test_all.append(y[i])
            fm_lstm_test_all.append(fm_lstm[i])
            fm_moirai_test_all.append(fm_moirai[i])
            dates_test_all.append(ticker_data.iloc[i]["Date"])
            tickers_test_all.append(ticker)

    X_test = np.array(X_test_all, dtype=np.float32)
    y_test = np.array(y_test_all, dtype=np.float32)
    fm_lstm_test = np.array(fm_lstm_test_all, dtype=np.float32)
    fm_moirai_test = np.array(fm_moirai_test_all, dtype=np.float32)

    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Create LEAF model
    print("\n" + "=" * 70)
    print("Creating LEAF model...")
    model = LEAFEnsemble(
        n_features=len(selected_features),
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
        device=DEVICE,
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"Total parameters: {n_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ===== PHASE 1: Walk-forward validation =====
    print("\n" + "=" * 70)
    print("PHASE 1: Walk-forward validation (5 folds)")
    print("=" * 70)

    fold_configs = [
        {"name": "Fold1", "train_end": "2016-12-31", "val_end": "2017-12-31"},
        {"name": "Fold2", "train_end": "2017-12-31", "val_end": "2018-12-31"},
        {"name": "Fold3", "train_end": "2018-12-31", "val_end": "2019-12-31"},
        {"name": "Fold4", "train_end": "2019-12-31", "val_end": "2020-12-31"},
        {"name": "Fold5", "train_end": "2020-12-31", "val_end": "2021-12-31"},
    ]

    fold_results = []

    for fold_config in fold_configs:
        fold_name = fold_config["name"]
        print(f"\n{fold_name}: Train ≤ {fold_config['train_end']} | Val ≤ {fold_config['val_end']}")

        # Create fold masks
        dates_train_ts = pd.to_datetime(dates_train)
        fold_train_mask = dates_train_ts <= pd.Timestamp(fold_config["train_end"])
        fold_val_mask = (dates_train_ts > pd.Timestamp(fold_config["train_end"])) & (
            dates_train_ts <= pd.Timestamp(fold_config["val_end"])
        )

        if fold_val_mask.sum() == 0:
            print(f"  ⚠ No validation data in fold, skipping")
            continue

        # Data for this fold
        X_fold_tr = torch.from_numpy(X_train[fold_train_mask]).to(DEVICE)
        y_fold_tr = torch.from_numpy(y_train[fold_train_mask]).to(DEVICE)
        fm_lstm_fold_tr = torch.from_numpy(fm_lstm_train[fold_train_mask]).to(DEVICE)
        fm_moirai_fold_tr = torch.from_numpy(fm_moirai_train[fold_train_mask]).to(DEVICE)

        X_fold_val = torch.from_numpy(X_train[fold_val_mask]).to(DEVICE)
        y_fold_val = torch.from_numpy(y_train[fold_val_mask]).to(DEVICE)
        fm_lstm_fold_val = torch.from_numpy(fm_lstm_train[fold_val_mask]).to(DEVICE)
        fm_moirai_fold_val = torch.from_numpy(fm_moirai_train[fold_val_mask]).to(DEVICE)

        print(f"  Train samples: {len(X_fold_tr):,} | Val samples: {len(X_fold_val):,}")

        # Reset model for this fold
        model = LEAFEnsemble(
            n_features=len(selected_features),
            embedding_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
            device=DEVICE,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Train fold
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_fold_tr), BATCH_SIZE):
                X_batch = X_fold_tr[i : i + BATCH_SIZE]
                y_batch = y_fold_tr[i : i + BATCH_SIZE]
                fm_lstm_batch = fm_lstm_fold_tr[i : i + BATCH_SIZE]
                fm_moirai_batch = fm_moirai_fold_tr[i : i + BATCH_SIZE]

                optimizer.zero_grad()
                output = model(X_batch, fm_lstm_batch, fm_moirai_batch)
                loss = criterion(output["ensemble_pred"], y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(1, len(X_fold_tr) // BATCH_SIZE)

            # Validate
            model.eval()
            with torch.no_grad():
                output = model(X_fold_val, fm_lstm_fold_val, fm_moirai_fold_val)
                val_loss = criterion(output["ensemble_pred"], y_fold_val).item()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Evaluate fold
        model.eval()
        with torch.no_grad():
            output = model(X_fold_val, fm_lstm_fold_val, fm_moirai_fold_val)
            pred_val = output["ensemble_pred"].cpu().numpy()
            weights_val = output["weights"].cpu().numpy().mean(axis=0)

        y_val_np = y_fold_val.cpu().numpy()
        metrics = eval_metrics(y_val_np, pred_val)
        print(f"  Results: MAE={metrics['MAE']:.6f} | RMSE={metrics['RMSE']:.6f} | DA={metrics['DA']:.3f}")
        print(f"  Weight avg: LSTM={weights_val[0]:.3f} | Moirai={weights_val[1]:.3f}")

        fold_results.append(
            {
                "fold": fold_name,
                "val_loss": best_val_loss,
                **metrics,
            }
        )

    # Summary fold results
    print("\n" + "=" * 70)
    print("Fold Results Summary:")
    print("=" * 70)
    fold_df = pd.DataFrame(fold_results)
    print(fold_df.to_string(index=False))

    print("\n✓ LEAF walk-forward validation complete")
    print(f"\n  Mean val loss across folds: {fold_df['val_loss'].mean():.6f}")
    print(f"  Mean MAE across folds: {fold_df['MAE'].mean():.6f}")
    print(f"  Mean DA across folds: {fold_df['DA'].mean():.3f}")


if __name__ == "__main__":
    main()
