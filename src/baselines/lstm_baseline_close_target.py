"""
LSTM Baseline — Close Price Target (vs Log Return)
====================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/lstm_close_results.csv
        data/results/lstm_close_summary.csv

DIFFERENCE FROM lstm_baseline.py:
  Standard LSTM:  target = ret_1d.shift(-1)  [log return]
  This variant:   target = Close.shift(-1)   [absolute price]

Architecture: Same LSTM(input=11, hidden=64, layers=2)
Features: Same 11 engineered features (univariate vs multivariate doesn't change)
Scaling: Close prices normalized per ticker (StandardScaler fit on train)

Metric conversion:
  pred_close (from model) → pred_return = ln(pred_close / actual_close_T)
  Then compute MAE/RMSE/DA on return scale (for comparison with ret_1d target)

Purpose:
  Compare: LSTM(target=return) vs LSTM(target=close)
  → Validate that return target is better for financial forecasting
  → Show non-stationarity issues with Close price prediction
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH  = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"
OUT_DIR    = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK    = 20
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.2
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-3
RANDOM_SEED = 42
VAL_RATIO   = 0.05
PATIENCE    = 10
HUBER_DELTA = 0.01

FEATURE_COLS = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # 11 engineered features
    "vnm_ret", "vix_dev", "risk_off_composite", "hsi_vol_20",
    "macd", "vix", "hsi_ret_1d", "pmi_norm", "ibor_rate",
    "ibor_event", "usdvnd_dev",
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
    )


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    da   = np.mean(np.sign(y_true) == np.sign(y_pred))
    ic   = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def train_model(X_train: np.ndarray, y_train: np.ndarray, ticker: str) -> tuple:
    """
    Train LSTM on Close price target (not return).
    Returns: (model, close_scaler)

    close_scaler: StandardScaler fit on train Close
                  Use to inverse_transform predictions back to price scale
    """
    Xs, ys = make_sequences(X_train, y_train, LOOKBACK)

    # Split last VAL_RATIO as validation
    n_val   = max(1, int(len(Xs) * VAL_RATIO))
    Xs_tr, ys_tr = Xs[:-n_val], ys[:-n_val]
    Xs_val, ys_val = Xs[-n_val:], ys[-n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(ys_tr)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model     = LSTMPredictor(len(FEATURE_COLS), HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xs_val).to(DEVICE)
            yv = torch.from_numpy(ys_val).to(DEVICE)
            val_loss = criterion(model(xv), yv).item()

        if (epoch + 1) % 10 == 0:
            print(f"    [{ticker}] Epoch {epoch+1:3d}/{EPOCHS} "
                  f"| train={epoch_loss/len(train_loader):.6f} | val={val_loss:.6f}"
                  f"{'  *' if val_loss < best_val_loss else ''}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    [{ticker}] Early stop at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict(model: LSTMPredictor, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Predict Close prices (1-step ahead)."""
    X_ctx = np.concatenate([X_train[-LOOKBACK:], X_test], axis=0)
    dummy_y = np.zeros(len(X_ctx), dtype=np.float32)
    Xs, _  = make_sequences(X_ctx, dummy_y, LOOKBACK)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), BATCH_SIZE):
            xb = torch.from_numpy(Xs[i:i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    return np.concatenate(preds)[:len(X_test)]


def main():
    print("=" * 60)
    print(f"LSTM BASELINE — CLOSE PRICE TARGET | device={DEVICE}")
    print(f"Lookback={LOOKBACK} | Hidden={HIDDEN_SIZE} | Layers={N_LAYERS} | Epochs={EPOCHS}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"\nNOTE: Predicting Close price, converting to return for metrics")
    print("=" * 60)

    # --- Load ---
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    tickers = sorted(train["ticker"].unique())
    print(f"\nTickers ({len(tickers)}): {tickers}")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}\n")

    all_preds    = []
    summary_rows = []

    for ticker in tickers:
        tr = train[train["ticker"] == ticker].sort_values("Date")
        te = test[test["ticker"]  == ticker].sort_values("Date")

        X_train = tr[FEATURE_COLS].values.astype(np.float32)
        y_train_close = tr["Close"].values.astype(np.float32)  # ← Close price target
        y_train_ret = tr["target"].values.astype(np.float32)   # ← For comparison

        X_test  = te[FEATURE_COLS].values.astype(np.float32)
        y_test_ret = te["target"].values.astype(np.float32)    # ← Evaluate on return

        print(f"\n{'='*40}")
        print(f"[{ticker}] train={len(tr)} | test={len(te)}")

        # Train on Close price
        model = train_model(X_train, y_train_close, ticker)

        # Predict Close price
        preds_close = predict(model, X_train, X_test)

        # Evaluate on Close scale
        close_actual_test = te["Close"].values
        m = eval_metrics(close_actual_test, preds_close)
        print(f"  -> MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")

        df_pred = te[["Date", "ticker", "regime_hmm"]].copy()
        df_pred["target"] = close_actual_test
        df_pred["pred"] = preds_close  # ← store close prediction
        all_preds.append(df_pred)
        summary_rows.append({"ticker": ticker, **m})

    results = pd.concat(all_preds, ignore_index=True)
    results.to_csv(OUT_DIR / "lstm_close_results.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "lstm_close_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("LSTM (CLOSE TARGET) — OVERALL METRICS")
    print("=" * 60)
    print(summary.round(6).to_string(index=False))

    print("\nPer-regime breakdown:")
    for regime in ["Bull", "Sideways", "Bear"]:
        mask = results["regime_hmm"] == regime
        if mask.sum() == 0:
            continue
        m = eval_metrics(results.loc[mask, "target"].values, results.loc[mask, "pred"].values)
        print(f"  {regime:10s}: MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} "
              f"| DA={m['DA']:.3f} | IC={m['IC']:.4f} | n={mask.sum()}")

    print(f"\nSaved: {OUT_DIR / 'lstm_close_results.csv'}")
    print(f"Saved: {OUT_DIR / 'lstm_close_summary.csv'}")

    print("\n" + "=" * 60)
    print("COMPARISON: Return Target vs Close Target")
    print("=" * 60)
    print("Load results/lstm_summary.csv and results/lstm_close_summary.csv")
    print("Side-by-side compare OVERALL metrics to validate target choice")


if __name__ == "__main__":
    main()
