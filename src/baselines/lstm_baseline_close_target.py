"""
LSTM Baseline — Close Price Target (Batch Walk-Forward Validation)
==================================================================
Input:  data/processed/v2/features_train.csv
        data/processed/v2/features_test.csv
Output: data/results/lstm_close_results.csv
        data/results/lstm_close_summary.csv

DIFFERENCE FROM INITIAL LSTM:
  This script uses Batch Walk-Forward Validation:
  - Initial train on the full training set (EPOCHS).
  - Test set is split into batches of size BATCH_DAYS.
  - Predict the current batch using the existing model.
  - Add the batch's actual data to the history.
  - Fine-tune (retrain) the model for RETRAIN_EPOCHS before predicting the next batch.

Purpose:
  Simulate real-world trading where models are continuously updated 
  with new market data to adapt to changing regimes.
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

# --- HYPERPARAMETERS ---
LOOKBACK       = 20
HIDDEN_SIZE    = 64
N_LAYERS       = 2
DROPOUT        = 0.2
EPOCHS         = 100       # Số epoch train gốc
BATCH_SIZE     = 32
LR             = 1e-3
RANDOM_SEED    = 42
VAL_RATIO      = 0.05
PATIENCE       = 10
HUBER_DELTA    = 0.01

# --- BATCH WALK-FORWARD CONFIG ---
BATCH_DAYS     = 50        # Số ngày dự đoán trong một lô
RETRAIN_EPOCHS = 10        # Số epoch để fine-tune sau mỗi lô

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
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
    
    # Tính Directional Accuracy (DA) trên thay đổi của giá thay vì giá trị tuyệt đối
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    if len(y_true_diff) > 0:
        da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
    else:
        da = 0.0
        
    ic   = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def train_model(X_train: np.ndarray, y_train: np.ndarray, ticker: str, epochs: int = EPOCHS, model: nn.Module = None) -> nn.Module:
    """
    Train or fine-tune LSTM on Close price target.
    Nếu model=None, sẽ khởi tạo model mới (cho lần học đầu).
    Nếu model đã có, sẽ tiếp tục học trên model đó (fine-tune).
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

    if model is None:
        model = LSTMPredictor(len(FEATURE_COLS), HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(epochs):
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

        # Chỉ in log chi tiết nếu là quá trình học gốc (epochs == EPOCHS)
        if epochs == EPOCHS and (epoch + 1) % 10 == 0:
            print(f"    [{ticker}] Epoch {epoch+1:3d}/{epochs} | train={epoch_loss/len(train_loader):.6f} | val={val_loss:.6f}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                if epochs == EPOCHS:
                    print(f"    [{ticker}] Early stop at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict(model: LSTMPredictor, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Predict Close prices for a specific batch based on history."""
    if len(X_test) == 0:
        return np.array([])
        
    X_ctx = np.concatenate([X_train[-LOOKBACK:], X_test], axis=0)
    dummy_y = np.zeros(len(X_ctx), dtype=np.float32)
    Xs, _  = make_sequences(X_ctx, dummy_y, LOOKBACK)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), BATCH_SIZE):
            xb = torch.from_numpy(Xs[i:i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    if len(preds) == 0:
        return np.array([])
        
    return np.concatenate(preds)[:len(X_test)]


def main():
    print("=" * 60)
    print(f"LSTM BASELINE — CLOSE TARGET (BATCH WALK-FORWARD)")
    print(f"Device: {DEVICE} | Batch size: {BATCH_DAYS} days | Retrain Epochs: {RETRAIN_EPOCHS}")
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

        # Khởi tạo data gốc từ Train Set
        X_train_base = tr[FEATURE_COLS].values.astype(np.float32)
        y_train_base = tr["Close"].values.astype(np.float32)

        print(f"\n{'='*40}")
        print(f"[{ticker}] Initial Train={len(tr)} | Test={len(te)}")

        # 1. TRAIN BAN ĐẦU (Học trên toàn bộ tập Train)
        model = train_model(X_train_base, y_train_base, ticker, epochs=EPOCHS, model=None)

        # 2. BATCH WALK-FORWARD TRÊN TẬP TEST
        unique_dates = sorted(te["Date"].unique())
        n_batches = (len(unique_dates) + BATCH_DAYS - 1) // BATCH_DAYS
        
        # Biến chứa dữ liệu lịch sử được cộng dồn sau mỗi batch
        X_train_current = X_train_base.copy()
        y_train_current = y_train_base.copy()
        
        preds_all = []
        
        for batch_idx in range(n_batches):
            batch_start_idx = batch_idx * BATCH_DAYS
            batch_end_idx = min(batch_start_idx + BATCH_DAYS, len(unique_dates))
            batch_dates = unique_dates[batch_start_idx:batch_end_idx]
            
            # Trích xuất data của batch này
            mask = te["Date"].isin(batch_dates)
            X_test_batch = te.loc[mask, FEATURE_COLS].values.astype(np.float32)
            y_test_batch = te.loc[mask, "Close"].values.astype(np.float32)
            
            if len(X_test_batch) == 0:
                continue
                
            # A. Dự báo lô hiện tại bằng model ĐANG CÓ (chưa thấy data của lô này)
            batch_preds = predict(model, X_train_current, X_test_batch)
            preds_all.extend(batch_preds)
            
            # B. Nạp data thực tế của lô này vào lịch sử (Mô hình bắt đầu "nhìn thấy" lô này)
            X_train_current = np.concatenate([X_train_current, X_test_batch], axis=0)
            y_train_current = np.concatenate([y_train_current, y_test_batch], axis=0)
            
            # C. Fine-tune model chuẩn bị cho lô tiếp theo (bỏ qua nếu đây là lô cuối)
            if batch_idx < n_batches - 1:
                print(f"    [{ticker}] Fine-tuning model (Batch {batch_idx+1}/{n_batches}, History size: {len(X_train_current)})...")
                model = train_model(X_train_current, y_train_current, ticker, epochs=RETRAIN_EPOCHS, model=model)

        # 3. ĐÁNH GIÁ VÀ LƯU KẾT QUẢ
        preds_close = np.array(preds_all)
        close_actual_test = te["Close"].values

        m = eval_metrics(close_actual_test, preds_close)
        print(f"  -> MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | DA={m['DA']:.3f} | IC={m['IC']:.4f}")

        df_pred = te[["Date", "ticker", "regime_hmm"]].copy()
        df_pred["target"] = close_actual_test
        df_pred["pred"] = preds_close  
        all_preds.append(df_pred)
        summary_rows.append({"ticker": ticker, **m})

    results = pd.concat(all_preds, ignore_index=True)
    results.to_csv(OUT_DIR / "lstm_close_results.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    overall = eval_metrics(results["target"].values, results["pred"].values)
    summary.loc[len(summary)] = {"ticker": "OVERALL", **overall}
    summary.to_csv(OUT_DIR / "lstm_close_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("LSTM (BATCH WALK-FORWARD) — OVERALL METRICS")
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


if __name__ == "__main__":
    main()