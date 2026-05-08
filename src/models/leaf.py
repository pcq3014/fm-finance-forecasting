"""
LEAF Framework — Learned Economic Adaptive Forecasting
======================================================
Main module combining:
  1. Regime Encoder (~50K params)
  2. Context Projector (~30K params)
  3. FM Ensemble + Weighting (~20K params)
  Total: ~100K trainable params (FMs frozen)

Training:
  - Phase 1: Walk-forward hyperparameter search (5 folds in train)
  - Phase 2: Final training on full train set
  - Phase 3: Test evaluation (1 time only)

Loss: CRPS + Pinball loss at quantiles [0.1, 0.25, 0.5, 0.75, 0.9]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

from regime_encoder import RegimeEncoder
from context_projector import ContextProjector
from fm_ensemble import FMEnsemble


class LEAFFramework(nn.Module):
    """
    Learned Economic Adaptive Forecasting Framework.
    Ensemble frozen FM with learned macro conditioning.
    """

    def __init__(
        self,
        n_features: int = 11,
        embedding_dim: int = 32,
        context_length: int = 32,
        seq_len: int = 60,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # 3 trainable components
        self.regime_encoder = RegimeEncoder(
            n_features=n_features,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).to(device)

        self.context_projector = ContextProjector(
            embedding_dim=embedding_dim,
            context_length=context_length,
            dropout=dropout,
        ).to(device)

        self.fm_ensemble = FMEnsemble(
            embedding_dim=embedding_dim, n_fms=2, dropout=dropout
        ).to(device)

        # FM models will be loaded separately (frozen)
        self.chronos = None
        self.moirai = None

    def load_fm_models(self):
        """Load frozen foundation models."""
        print("Loading foundation models (Chronos-2, Moirai-2)...")
        try:
            from chronos import Chronos2Pipeline
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.chronos = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2",
                device_map=device,
                torch_dtype=dtype,
            )
            print("✓ Chronos-2 loaded")
        except Exception as e:
            print(f"⚠ Chronos load failed: {e}")

        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

            self.moirai = Moirai2Forecast(
                module=Moirai2Module.from_pretrained(
                    "Salesforce/moirai-2.0-R-small"
                ),
                prediction_length=1,
                context_length=128,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self.moirai_predictor = self.moirai.create_predictor(batch_size=64)
            print("✓ Moirai-2 loaded")
        except Exception as e:
            print(f"⚠ Moirai load failed: {e}")

    def freeze_fm_models(self):
        """Ensure FM models are frozen."""
        if self.chronos is not None:
            for param in self.chronos.parameters():
                param.requires_grad = False
        if self.moirai is not None:
            for param in self.moirai.parameters():
                param.requires_grad = False

    def get_fm_predictions(
        self, close_history: np.ndarray, horizon: int = 1
    ) -> tuple:
        """
        Get predictions from frozen FM models.

        Args:
            close_history: (n_samples, lookback) price series
            horizon: forecast horizon (1 = next day)

        Returns:
            chronos_pred: (n_samples,)
            moirai_pred: (n_samples,)
        """
        # TODO: Implement FM prediction gathering
        # For now, return dummy predictions
        n_samples = close_history.shape[0]
        chronos_pred = torch.zeros(n_samples).to(self.device)
        moirai_pred = torch.zeros(n_samples).to(self.device)
        return chronos_pred, moirai_pred

    def forward(
        self,
        macro_features: torch.Tensor,
        fm_preds_chronos: torch.Tensor,
        fm_preds_moirai: torch.Tensor,
    ) -> dict:
        """
        Forward pass: regime encoding → context projection → ensemble weighting.

        Args:
            macro_features: (batch, seq_len=60, n_features=11)
            fm_preds_chronos: (batch, n_steps) from frozen Chronos
            fm_preds_moirai: (batch, n_steps) from frozen Moirai

        Returns:
            dict with:
              - ensemble_pred: (batch, n_steps) weighted ensemble
              - regime_embedding: (batch, embedding_dim)
              - weights: (batch, 2) FM weights
              - context_tokens: (batch, context_length, embedding_dim)
              - gate: (batch, 1) context influence
        """
        # Encode macro features → regime embedding
        regime_embedding = self.regime_encoder(macro_features)

        # Project regime → context tokens
        context_tokens, gate = self.context_projector(regime_embedding)

        # Ensemble FM predictions with regime-aware weighting
        ensemble_pred, weights = self.fm_ensemble(
            regime_embedding, [fm_preds_chronos, fm_preds_moirai]
        )

        return {
            "ensemble_pred": ensemble_pred,
            "regime_embedding": regime_embedding,
            "weights": weights,
            "context_tokens": context_tokens,
            "gate": gate,
        }

    def count_parameters(self) -> dict:
        """Count trainable parameters."""
        return {
            "regime_encoder": sum(p.numel() for p in self.regime_encoder.parameters()),
            "context_projector": sum(p.numel() for p in self.context_projector.parameters()),
            "fm_ensemble": sum(p.numel() for p in self.fm_ensemble.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create LEAF
    leaf = LEAFFramework(
        n_features=11,
        embedding_dim=32,
        context_length=32,
        seq_len=60,
        dropout=0.1,
        device=device,
    )

    # Count parameters
    params = leaf.count_parameters()
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name:20s}: {count:8,} params")

    # Test forward pass
    print("\nTesting forward pass...")
    macro_features = torch.randn(4, 60, 11).to(device)
    fm_preds_c = torch.randn(4, 252).to(device)
    fm_preds_m = torch.randn(4, 252).to(device)

    output = leaf(macro_features, fm_preds_c, fm_preds_m)
    print(f"  Ensemble pred shape: {output['ensemble_pred'].shape}")
    print(f"  Regime embedding shape: {output['regime_embedding'].shape}")
    print(f"  Weights shape: {output['weights'].shape}")
    print(f"  Weights (sample): {output['weights'][0]}")
