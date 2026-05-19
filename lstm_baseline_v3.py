"""
LSTM Baseline - V3 Multivariate (Panel Data)
============================================

Input:  data/processed/panel/panel_train_fixed.csv
        data/processed/panel/panel_val_fixed.csv
        data/processed/panel/panel_test_fixed.csv

Output: data/results/lstm_mv_summary.csv
        data/results/lstm_mv_predictions.csv

Model: LSTM (64 hidden, 2 layers) with 8 features
Multivariate: Uses same 8 transformed features as Chronos-2 & Moirai-2
Target: y_return_1d (log return)
Rolling 1-step prediction with continuous model updates
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_train_fixed.csv"
VAL_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_val_fixed.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "panel" / "panel_test_fixed.csv"
OUT_DIR = ROOT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'y_return_1d'
LOOKBACK = 50
HIDDEN_SIZE = 64
N_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 15
RANDOM_SEED = 42

FEATURES = [
    'vn30_logret',
    'hsi_logret',
    'vn30_volume_log',
    'volume_ratio_20d',
    'trading_value_log',
    'foreign_net_ratio',
    'vnm_ret_lag1',
    'hsi_logret_lag1',
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
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(self.dropout(last_out))
        return out.squeeze(1)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0

    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    dir_acc = np.mean(dir_true == dir_pred)

    return {'rmse': rmse, 'mae': mae, 'ic': ic, 'dir_acc': dir_acc}


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Build (seq, target) pairs from time series"""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def predict_ticker_rolling(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
):
    """Mini-batch LSTM with early stopping on val, rolling test forecast"""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Build sequences
    Xtr_seq, ytr_seq = build_sequences(X_tr_sc, y_train, LOOKBACK)
    Xvl_seq, yvl_seq = build_sequences(
        np.vstack([X_tr_sc[-LOOKBACK:], X_val_sc]),
        np.hstack([y_train[-LOOKBACK:], y_val]),
        LOOKBACK
    )

    if len(Xtr_seq) == 0:
        return np.zeros(len(y_test))

    model = LSTMPredictor(X_train.shape[1], HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr_seq).to(DEVICE)
    ytr_t = torch.tensor(ytr_seq).to(DEVICE)
    Xvl_t = torch.tensor(Xvl_seq).to(DEVICE)
    yvl_t = torch.tensor(yvl_seq).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), BATCH_SIZE):
            batch_idx = idx[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xtr_t[batch_idx]), ytr_t[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xvl_t), yvl_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Rolling forecast on test — context = all prior data (train+val+test seen so far)
    X_all = np.vstack([X_train, X_val, X_test])
    n_prefix = len(X_train) + len(X_val)
    preds = []

    with torch.no_grad():
        for step in range(len(X_test)):
            end = n_prefix + step
            ctx = X_all[max(0, end - LOOKBACK):end]
            ctx_sc = scaler.transform(ctx)
            if len(ctx_sc) < LOOKBACK:
                pad = np.zeros((LOOKBACK - len(ctx_sc), ctx_sc.shape[1]), dtype=np.float32)
                ctx_sc = np.vstack([pad, ctx_sc])
            x_t = torch.tensor(ctx_sc[np.newaxis], dtype=torch.float32).to(DEVICE)
            preds.append(float(model(x_t).cpu().numpy()[0]))

    return np.array(preds, dtype=np.float32)


def main():
    print("\n" + "="*80)
    print("LSTM BASELINE - V3 MULTIVARIATE (PANEL DATA)")
    print("="*80)
    print(f"Target: {TARGET} | Lookback: {LOOKBACK}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES[:3])}...\n")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    tickers = sorted(df_train['ticker'].unique())
    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"Train: {len(df_train):,} rows")
    print(f"Val:   {len(df_val):,} rows")
    print(f"Test:  {len(df_test):,} rows\n")

    all_predictions = []
    summary_rows = []

    print("="*80)
    print("Forecasting per ticker (multivariate - 8 features)...")
    print("="*80 + "\n")

    for ticker in tickers:
        tr  = df_train[df_train['ticker'] == ticker].sort_values('date')
        val = df_val[df_val['ticker'] == ticker].sort_values('date')
        te  = df_test[df_test['ticker'] == ticker].sort_values('date')

        if len(tr) == 0 or len(te) == 0:
            continue

        X_train = tr[FEATURES].values.astype(np.float32)
        X_val   = val[FEATURES].values.astype(np.float32)
        X_test  = te[FEATURES].values.astype(np.float32)

        y_train = tr[TARGET].values.astype(np.float32)
        y_val   = val[TARGET].values.astype(np.float32)
        y_test  = te[TARGET].values.astype(np.float32)

        print(f"[{ticker:6s}] train={len(tr):>5d} | val={len(val):>4d} | test={len(te):>4d} -> ", end="", flush=True)

        try:
            y_pred = predict_ticker_rolling(X_train, y_train, X_val, y_val, X_test, y_test)
            m = eval_metrics(y_test, y_pred)
            print(f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | IC={m['ic']:.4f} | DA={m['dir_acc']:.3f}")

        except Exception as e:
            print(f"Error: {str(e)[:60]}")
            y_pred = np.zeros(len(y_test))
            m = eval_metrics(y_test, y_pred)

        all_predictions.append(pd.DataFrame({
            'date': te['date'].values,
            'ticker': ticker,
            'y_true': y_test,
            'y_pred': y_pred
        }))

        summary_rows.append({
            'ticker': ticker,
            'train_size': len(tr),
            'val_size': len(val),
            'test_size': len(te),
            **m
        })

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        overall_metrics = eval_metrics(predictions_df['y_true'].values, predictions_df['y_pred'].values)
        summary_rows.append({
            'ticker': 'OVERALL',
            'train_size': len(df_train),
            'test_size': len(df_test),
            **overall_metrics
        })

        summary_df = pd.DataFrame(summary_rows)

        print("\n" + "="*80)
        print("LSTM SUMMARY")
        print("="*80)
        print(summary_df.round(6).to_string(index=False))

        predictions_df.to_csv(OUT_DIR / "lstm_mv_predictions.csv", index=False)
        summary_df.to_csv(OUT_DIR / "lstm_mv_summary.csv", index=False)

        print(f"\nSaved: {OUT_DIR / 'lstm_mv_predictions.csv'}")
        print(f"Saved: {OUT_DIR / 'lstm_mv_summary.csv'}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
