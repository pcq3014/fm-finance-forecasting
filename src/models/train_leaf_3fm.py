"""
LEAF Framework — 3 FM Models
=============================
Train LEAF with 3 foundation models (LSTM + Moirai + Chronos).

Thay đổi các FM ở đây:
- FM1 = LSTM (lstm_close_results.csv)
- FM2 = Moirai (moirai_results.csv)  
- FM3 = Chronos (chronos_results.csv)

Hoặc chọn từ: TimesFM, Moirai-2, ...
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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "v2" / "features_test.csv"

# === CHỌN 3 FM MÀ BẠN MUỐN ===
FM1_RESULTS = ROOT_DIR / "data" / "results" / "timesfm_results.csv"
FM2_RESULTS = ROOT_DIR / "data" / "results" / "moirai_results.csv"
FM3_RESULTS = ROOT_DIR / "data" / "results" / "chronos_results.csv"

FM1_NAME = "TimesFM"
FM2_NAME = "Moirai"
FM3_NAME = "Chronos"

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
VAL_RATIO = 0.2  # 80% train, 20% test for LEAF

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT_DIR / "src" / "models"))
from regime_encoder import RegimeEncoder
from fm_ensemble import FMEnsemble


class LEAFEnsemble3FM(nn.Module):
    """LEAF for 3 FM weighting."""

    def __init__(self, n_features: int = 11, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.regime_encoder = RegimeEncoder(
            n_features=n_features,
            seq_len=SEQ_LEN,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        # 3 FMs instead of 2
        self.fm_ensemble = FMEnsemble(
            embedding_dim=embedding_dim, n_fms=3, dropout=dropout
        )

    def forward(self, macro_features, fm1, fm2, fm3):
        regime_embedding = self.regime_encoder(macro_features)
        ensemble_pred, weights = self.fm_ensemble(
            regime_embedding,
            [fm1.unsqueeze(-1), fm2.unsqueeze(-1), fm3.unsqueeze(-1)],
        )
        return ensemble_pred.squeeze(-1), weights, regime_embedding

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def eval_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Tính trên sai phân (sự thay đổi giá)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    if len(y_true_diff) > 0:
        da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        ic = np.corrcoef(y_true_diff, y_pred_diff)[0, 1] if np.std(y_pred_diff) > 0 else 0.0
    else:
        da, ic = 0.0, 0.0
    return {"MAE": mae, "RMSE": rmse, "DA": da, "IC": ic}


def main():
    # 1. Load data
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])
    
    # 3. Load FM predictions
    fm1 = pd.read_csv(FM1_RESULTS, parse_dates=["Date"])
    fm2 = pd.read_csv(FM2_RESULTS, parse_dates=["Date"])
    fm3 = pd.read_csv(FM3_RESULTS, parse_dates=["Date"])
    
    # Find the date range that has FM predictions for all 3 models
    fm_dates_common = set(fm1["Date"].unique()) & set(fm2["Date"].unique()) & set(fm3["Date"].unique())
    fm_min_date = min(fm_dates_common)
    fm_max_date = max(fm_dates_common)
    
    print(f"\n📅 FM predictions available: {fm_min_date.date()} to {fm_max_date.date()}")
    
    # Filter train/test to only dates with FM predictions
    train = train[(train["Date"] >= fm_min_date) & (train["Date"] <= fm_max_date)].copy()
    test = test[(test["Date"] >= fm_min_date) & (test["Date"] <= fm_max_date)].copy()
    
    print(f"   Train data after filtering: {len(train):,} rows")
    print(f"   Test data after filtering: {len(test):,} rows")
    
    # ⚠️ If train is empty, combine both and split for LEAF training
    if len(train) == 0:
        print("\n   ⚠️  Train set has no FM predictions. Combining train+test and splitting for LEAF...")
        combined = pd.concat([train, test], ignore_index=True).sort_values("Date").reset_index(drop=True)
        n_train_leaf = int(len(combined) * (1 - VAL_RATIO))
        train = combined.iloc[:n_train_leaf].copy()
        test = combined.iloc[n_train_leaf:].copy()
        print(f"   New split: Train {len(train):,} rows | Test {len(test):,} rows for LEAF")
    
    # 2. CHUẨN HÓA ĐẶC TRƯNG (Scaling) - fit on train only
    selected_features = [
        "vnm_ret", "risk_off_composite", "vix_dev", "hsi_ret_1d", "vix",
        "macd", "hsi_vol_20", "usdvnd_dev", "ibor_rate", "ibor_event", "pmi_norm",
    ]
    scaler = StandardScaler()
    train[selected_features] = scaler.fit_transform(train[selected_features])
    test[selected_features] = scaler.transform(test[selected_features])

    # Merge dữ liệu train
    train_with_fm = train.merge(
        fm1[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm1"}), on=["Date", "ticker"]
    ).merge(
        fm2[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm2"}), on=["Date", "ticker"]
    ).merge(
        fm3[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm3"}), on=["Date", "ticker"]
    )
    
    print(f"\n✓ Merged train data: {len(train_with_fm):,} rows, tickers: {train_with_fm['ticker'].nunique()}")
    
    # Merge dữ liệu test
    test_with_fm = test.merge(
        fm1[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm1"}), on=["Date", "ticker"]
    ).merge(
        fm2[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm2"}), on=["Date", "ticker"]
    ).merge(
        fm3[["Date", "ticker", "pred"]].rename(columns={"pred": "pred_fm3"}), on=["Date", "ticker"]
    )

    # 4. Tạo Sequence cho TRAIN (Loại bỏ triệt để các giá trị lỗi)
    X_train_all, y_train_all, fm1_train_all, fm2_train_all, fm3_train_all, dates_train_all = [], [], [], [], [], []

    for ticker in sorted(train_with_fm["ticker"].unique()):
        ticker_data = train_with_fm[train_with_fm["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        X = ticker_data[selected_features].values.astype(np.float32)
        y = ticker_data["Close"].values.astype(np.float32)
        p1 = ticker_data["pred_fm1"].values
        p2 = ticker_data["pred_fm2"].values
        p3 = ticker_data["pred_fm3"].values

        for i in range(SEQ_LEN, len(X)):
            # CHỈ LẤY nếu các FM dự báo hợp lệ
            if p1[i] > 1e-5 and p2[i] > 1e-5 and p3[i] > 1e-5:
                X_train_all.append(X[i - SEQ_LEN : i])
                y_train_all.append(y[i])
                fm1_train_all.append(p1[i])
                fm2_train_all.append(p2[i])
                fm3_train_all.append(p3[i])
                dates_train_all.append(ticker_data.iloc[i]["Date"])
    
    # 5. Tạo Sequence cho TEST (Loại bỏ triệt để các giá trị lỗi)
    X_test_all, y_test_all, fm1_test_all, fm2_test_all, fm3_test_all, dates_test_all = [], [], [], [], [], []

    for ticker in sorted(test_with_fm["ticker"].unique()):
        ticker_data = test_with_fm[test_with_fm["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        X = ticker_data[selected_features].values.astype(np.float32)
        y = ticker_data["Close"].values.astype(np.float32)
        p1 = ticker_data["pred_fm1"].values
        p2 = ticker_data["pred_fm2"].values
        p3 = ticker_data["pred_fm3"].values

        for i in range(SEQ_LEN, len(X)):
            # CHỈ LẤY nếu các FM dự báo hợp lệ
            if p1[i] > 1e-5 and p2[i] > 1e-5 and p3[i] > 1e-5:
                X_test_all.append(X[i - SEQ_LEN : i])
                y_test_all.append(y[i])
                fm1_test_all.append(p1[i])
                fm2_test_all.append(p2[i])
                fm3_test_all.append(p3[i])
                dates_test_all.append(ticker_data.iloc[i]["Date"])
    
    # Convert to numpy arrays
    X_train = np.array(X_train_all)
    y_train = np.array(y_train_all)
    fm1_train = np.array(fm1_train_all)
    fm2_train = np.array(fm2_train_all)
    fm3_train = np.array(fm3_train_all)
    
    X_test = np.array(X_test_all)
    y_test = np.array(y_test_all)
    fm1_test = np.array(fm1_test_all)
    fm2_test = np.array(fm2_test_all)
    fm3_test = np.array(fm3_test_all)
    
    print(f"\n🔢 Sequences created:")
    print(f"   Train: {len(X_train):,} sequences")
    print(f"   Test: {len(X_test):,} sequences")
    
    # Check if we have data
    if len(X_train) == 0 or len(X_test) == 0:
        print("\n❌ ERROR: Insufficient sequences created!")
        print(f"   Train tickers merged: {train_with_fm['ticker'].nunique()}")
        print(f"   Max data points needed: {SEQ_LEN + 1} per ticker")
        return

    # Create model
    model = LEAFEnsemble3FM(
        n_features=len(selected_features),
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss()

    print(f"\n✓ Model parameters: {model.count_parameters():,}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    # 5. Training với Best Model Checkpoint
    best_val_loss = float("inf")
    patience_count = 0
    model_path = OUT_DIR / "leaf_best_model.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training batch loop
        n_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(n_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(X_train))
            
            X_batch = torch.from_numpy(X_train[start_idx:end_idx]).float().to(DEVICE)
            y_batch = torch.from_numpy(y_train[start_idx:end_idx]).float().to(DEVICE)
            fm1_batch = torch.from_numpy(fm1_train[start_idx:end_idx]).float().to(DEVICE)
            fm2_batch = torch.from_numpy(fm2_train[start_idx:end_idx]).float().to(DEVICE)
            fm3_batch = torch.from_numpy(fm3_train[start_idx:end_idx]).float().to(DEVICE)
            
            optimizer.zero_grad()
            pred, _, _ = model(X_batch, fm1_batch, fm2_batch, fm3_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * (end_idx - start_idx)
        
        if len(X_train) > 0:
            train_loss /= len(X_train)
        else:
            train_loss = 0.0

        # Validation trên TEST SET
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
            y_test_t = torch.from_numpy(y_test).float().to(DEVICE)
            fm1_test_t = torch.from_numpy(fm1_test).float().to(DEVICE)
            fm2_test_t = torch.from_numpy(fm2_test).float().to(DEVICE)
            fm3_test_t = torch.from_numpy(fm3_test).float().to(DEVICE)
            
            pred_test, _, _ = model(X_test_t, fm1_test_t, fm2_test_t, fm3_test_t)
            val_loss = criterion(pred_test, y_test_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(model_path))
            patience_count = 0
        else:
            patience_count += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | patience={patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 6. Load lại model tốt nhất trước khi đánh giá cuối cùng
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    
    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
        fm1_test_t = torch.from_numpy(fm1_test).float().to(DEVICE)
        fm2_test_t = torch.from_numpy(fm2_test).float().to(DEVICE)
        fm3_test_t = torch.from_numpy(fm3_test).float().to(DEVICE)

        pred_leaf, weights_val, _ = model(X_test_t, fm1_test_t, fm2_test_t, fm3_test_t)
        pred_leaf_np = pred_leaf.cpu().numpy()
        weights_np = weights_val.cpu().numpy()

    # Compare with baselines
    pred_ensemble_manual = (fm1_test + fm2_test + fm3_test) / 3
    pred_fm1 = fm1_test
    pred_fm2 = fm2_test
    pred_fm3 = fm3_test

    m_fm1 = eval_metrics(y_test, pred_fm1)
    m_fm2 = eval_metrics(y_test, pred_fm2)
    m_fm3 = eval_metrics(y_test, pred_fm3)
    m_ensemble = eval_metrics(y_test, pred_ensemble_manual)
    m_leaf = eval_metrics(y_test, pred_leaf_np)

    print("\nTest Results:")
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'DA':>10} {'IC':>10}")
    print("-" * 55)
    print(f"{FM1_NAME:<20} {m_fm1['MAE']:>10.6f} {m_fm1['RMSE']:>10.6f} {m_fm1['DA']:>10.3f} {m_fm1['IC']:>10.4f}")
    print(f"{FM2_NAME:<20} {m_fm2['MAE']:>10.6f} {m_fm2['RMSE']:>10.6f} {m_fm2['DA']:>10.3f} {m_fm2['IC']:>10.4f}")
    print(f"{FM3_NAME:<20} {m_fm3['MAE']:>10.6f} {m_fm3['RMSE']:>10.6f} {m_fm3['DA']:>10.3f} {m_fm3['IC']:>10.4f}")
    print(f"{'Avg (1/3 each)':<20} {m_ensemble['MAE']:>10.6f} {m_ensemble['RMSE']:>10.6f} {m_ensemble['DA']:>10.3f} {m_ensemble['IC']:>10.4f}")
    print(f"{'LEAF (learned)':<20} {m_leaf['MAE']:>10.6f} {m_leaf['RMSE']:>10.6f} {m_leaf['DA']:>10.3f} {m_leaf['IC']:>10.4f}")

    # Weight analysis
    print(f"\nLearned Weights (average):")
    print(f"  {FM1_NAME:<12}: {weights_np[:, 0].mean():.4f} (std={weights_np[:, 0].std():.4f}) | min={weights_np[:, 0].min():.4f} max={weights_np[:, 0].max():.4f}")
    print(f"  {FM2_NAME:<12}: {weights_np[:, 1].mean():.4f} (std={weights_np[:, 1].std():.4f}) | min={weights_np[:, 1].min():.4f} max={weights_np[:, 1].max():.4f}")
    print(f"  {FM3_NAME:<12}: {weights_np[:, 2].mean():.4f} (std={weights_np[:, 2].std():.4f}) | min={weights_np[:, 2].min():.4f} max={weights_np[:, 2].max():.4f}")

    print("\n✓ LEAF training complete!")


if __name__ == "__main__":
    main()
