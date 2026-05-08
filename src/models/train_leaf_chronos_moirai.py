"""
LEAF Framework — Moirai + Chronos Ensemble
==========================================
Use Chronos valid predictions (31.5% of test set, from 2022-09 onwards)
Demonstrate regime-aware FM ensemble with 2 comparable foundation models.
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

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
CHRONOS_RESULTS = ROOT_DIR / "data" / "results" / "chronos_results.csv"
MOIRAI_RESULTS = ROOT_DIR / "data" / "results" / "moirai_results.csv"
OUT_DIR = ROOT_DIR / "data" / "results"

# Hyperparameters
SEQ_LEN = 60
EMBEDDING_DIM = 32
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.1
PATIENCE = 5
RANDOM_SEED = 42
VAL_RATIO = 0.2

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT_DIR / "src" / "models"))
from regime_encoder import RegimeEncoder
from fm_ensemble import FMEnsemble


class LEAFEnsemble(nn.Module):
    """Simplified LEAF for Chronos + Moirai weighting."""

    def __init__(self, n_features: int = 11, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.regime_encoder = RegimeEncoder(
            n_features=n_features,
            seq_len=SEQ_LEN,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.fm_ensemble = FMEnsemble(
            embedding_dim=embedding_dim, n_fms=2, dropout=dropout
        )

    def forward(self, macro_features, fm_chronos, fm_moirai):
        regime_embedding = self.regime_encoder(macro_features)
        ensemble_pred, weights = self.fm_ensemble(
            regime_embedding,
            [fm_chronos.unsqueeze(-1), fm_moirai.unsqueeze(-1)],
        )
        return ensemble_pred.squeeze(-1), weights, regime_embedding

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def eval_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def main():
    print("=" * 70)
    print("LEAF FRAMEWORK — Chronos + Moirai (2 Foundation Models)")
    print("=" * 70)

    # Load data
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])
    chronos = pd.read_csv(CHRONOS_RESULTS, parse_dates=["Date"])
    moirai = pd.read_csv(MOIRAI_RESULTS, parse_dates=["Date"])

    selected_features = [
        "vnm_ret", "risk_off_composite", "vix_dev", "hsi_ret_1d", "vix",
        "macd", "hsi_vol_20", "usdvnd_dev", "ibor_rate", "ibor_event", "pmi_norm",
    ]

    # Filter: keep only where Chronos has valid predictions (non-zero)
    chronos_valid = chronos[chronos['pred'] != 0.0].copy()

    print(f"Test period: {test['Date'].min().date()} → {test['Date'].max().date()}")
    print(f"Chronos valid samples: {len(chronos_valid):,} / {len(chronos):,} ({len(chronos_valid)/len(chronos):.1%})")
    print(f"Chronos valid period: {chronos_valid['Date'].min().date()} → {chronos_valid['Date'].max().date()}")
    print(f"Selected features: {selected_features}")

    # Merge predictions (only valid Chronos dates)
    test_filtered = test[test['Date'].isin(chronos_valid['Date'].unique())]

    test_with_fm = test_filtered.merge(
        chronos_valid[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_chronos"}),
        on=["Date", "ticker"],
        how="left",
    ).merge(
        moirai[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_moirai"}),
        on=["Date", "ticker"],
        how="left",
    )

    # Remove remaining NaNs
    test_with_fm = test_with_fm.dropna(subset=["pred_chronos", "pred_moirai"])

    print(f"After merging with both FMs: {len(test_with_fm):,} samples")

    # Create sequences
    X_all = []
    y_all = []
    fm_chronos_all = []
    fm_moirai_all = []
    dates_all = []

    for ticker in sorted(test_with_fm["ticker"].unique()):
        ticker_data = test_with_fm[test_with_fm["ticker"] == ticker].sort_values(
            "Date"
        ).reset_index(drop=True)

        X = ticker_data[selected_features].values.astype(np.float32)
        y = ticker_data["Close"].values.astype(np.float32)
        fm_chronos = ticker_data["pred_chronos"].values.astype(np.float32)
        fm_moirai = ticker_data["pred_moirai"].values.astype(np.float32)

        for i in range(SEQ_LEN, len(X)):
            X_all.append(X[i - SEQ_LEN : i])
            y_all.append(y[i])
            fm_chronos_all.append(fm_chronos[i])
            fm_moirai_all.append(fm_moirai[i])
            dates_all.append(ticker_data.iloc[i]["Date"])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    fm_chronos = np.array(fm_chronos_all, dtype=np.float32)
    fm_moirai = np.array(fm_moirai_all, dtype=np.float32)
    dates = np.array(dates_all)

    print(f"Created {len(X):,} sequences")

    # Train/val split
    n_train = int(len(X) * (1 - VAL_RATIO))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    fm_chronos_train, fm_chronos_val = fm_chronos[:n_train], fm_chronos[n_train:]
    fm_moirai_train, fm_moirai_val = fm_moirai[:n_train], fm_moirai[n_train:]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    # Create model
    model = LEAFEnsemble(n_features=len(selected_features), embedding_dim=EMBEDDING_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\nModel parameters: {model.count_parameters():,}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING LEAF (Chronos + Moirai)")
    print("=" * 70)

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        # Mini-batch training
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = torch.from_numpy(X_train[i : i + BATCH_SIZE]).to(DEVICE)
            y_batch = torch.from_numpy(y_train[i : i + BATCH_SIZE]).to(DEVICE)
            fm_chronos_batch = torch.from_numpy(fm_chronos_train[i : i + BATCH_SIZE]).to(DEVICE)
            fm_moirai_batch = torch.from_numpy(fm_moirai_train[i : i + BATCH_SIZE]).to(DEVICE)

            optimizer.zero_grad()
            pred, weights, _ = model(X_batch, fm_chronos_batch, fm_moirai_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(X_train) // BATCH_SIZE)

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val).to(DEVICE)
            fm_chronos_val_t = torch.from_numpy(fm_chronos_val).to(DEVICE)
            fm_moirai_val_t = torch.from_numpy(fm_moirai_val).to(DEVICE)
            y_val_t = torch.from_numpy(y_val).to(DEVICE)

            pred_val, weights_val, _ = model(X_val_t, fm_chronos_val_t, fm_moirai_val_t)
            val_loss = criterion(pred_val, y_val_t).item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS (Chronos + Moirai Ensemble)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(DEVICE)
        fm_chronos_val_t = torch.from_numpy(fm_chronos_val).to(DEVICE)
        fm_moirai_val_t = torch.from_numpy(fm_moirai_val).to(DEVICE)

        pred_leaf, weights_val, regime_emb = model(X_val_t, fm_chronos_val_t, fm_moirai_val_t)
        pred_leaf_np = pred_leaf.cpu().numpy()
        weights_np = weights_val.cpu().numpy()

    # Ensemble performance
    pred_ensemble_manual = 0.5 * fm_chronos_val + 0.5 * fm_moirai_val
    pred_chronos = fm_chronos_val
    pred_moirai = fm_moirai_val

    m_chronos = eval_metrics(y_val, pred_chronos)
    m_moirai = eval_metrics(y_val, pred_moirai)
    m_ensemble = eval_metrics(y_val, pred_ensemble_manual)
    m_leaf = eval_metrics(y_val, pred_leaf_np)

    print("\nValidation Results:")
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'DA':>10} {'IC':>10}")
    print("-" * 55)
    print(f"{'Chronos':<20} {m_chronos['MAE']:>10.6f} {m_chronos['RMSE']:>10.6f} {m_chronos['DA']:>10.3f} {m_chronos['IC']:>10.4f}")
    print(f"{'Moirai':<20} {m_moirai['MAE']:>10.6f} {m_moirai['RMSE']:>10.6f} {m_moirai['DA']:>10.3f} {m_moirai['IC']:>10.4f}")
    print(f"{'50/50 Ensemble':<20} {m_ensemble['MAE']:>10.6f} {m_ensemble['RMSE']:>10.6f} {m_ensemble['DA']:>10.3f} {m_ensemble['IC']:>10.4f}")
    print(f"{'LEAF (learned)':<20} {m_leaf['MAE']:>10.6f} {m_leaf['RMSE']:>10.6f} {m_leaf['DA']:>10.3f} {m_leaf['IC']:>10.4f}")

    # Improvement over baselines
    imp_vs_chronos = (1 - m_leaf['MAE'] / m_chronos['MAE']) * 100
    imp_vs_moirai = (1 - m_leaf['MAE'] / m_moirai['MAE']) * 100
    imp_vs_ensemble = (1 - m_leaf['MAE'] / m_ensemble['MAE']) * 100

    print(f"\nImprovement vs baselines:")
    print(f"  vs Chronos:  {imp_vs_chronos:+.1f}%")
    print(f"  vs Moirai:   {imp_vs_moirai:+.1f}%")
    print(f"  vs 50/50:    {imp_vs_ensemble:+.1f}%")

    # Weight analysis
    print(f"\nLearned Weights:")
    print(f"  Chronos: {weights_np[:, 0].mean():.3f} (std={weights_np[:, 0].std():.3f}) | min={weights_np[:, 0].min():.3f} max={weights_np[:, 0].max():.3f}")
    print(f"  Moirai:  {weights_np[:, 1].mean():.3f} (std={weights_np[:, 1].std():.3f}) | min={weights_np[:, 1].min():.3f} max={weights_np[:, 1].max():.3f}")

    print("\n✓ LEAF training complete")


if __name__ == "__main__":
    main()
