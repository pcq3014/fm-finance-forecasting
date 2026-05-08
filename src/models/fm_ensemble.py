"""
FM Ensemble — Frozen Foundation Models with Regime-Aware Weighting
===================================================================
FMs: Chronos-2, Moirai-2.0 (completely frozen, no gradient updates)

Weighting:
  - Input: regime_embedding (batch, embedding_dim=32)
  - Output: softmax weights (batch, n_fms=2)
  - learns which FM to trust in each regime

Parameters: ~20K (only in weight network, FM frozen)
"""

import torch
import torch.nn as nn
import numpy as np


class FMEnsemble(nn.Module):
    """
    Ensemble frozen FMs with learned regime-aware weighting.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        n_fms: int = 2,  # Chronos, Moirai
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_fms = n_fms
        self.embedding_dim = embedding_dim

        # Dynamic weight network: regime_embedding → softmax weights
        self.weight_network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_fms),
            # No softmax here — will apply in forward for numerical stability
        )

    def forward(
        self,
        regime_embedding: torch.Tensor,
        fm_predictions: list,
    ) -> torch.Tensor:
        """
        Combine FM predictions with regime-aware weights.

        Args:
            regime_embedding: (batch_size, embedding_dim)
            fm_predictions: list of (batch_size, n_steps) tensors from each FM

        Returns:
            ensemble_pred: (batch_size, n_steps) weighted ensemble prediction
        """
        batch_size = regime_embedding.size(0)

        # Get logits for each FM
        logits = self.weight_network(regime_embedding)  # (batch, n_fms)

        # Softmax to get weights (sum to 1)
        weights = torch.softmax(logits, dim=1)  # (batch, n_fms)

        # Stack FM predictions
        # fm_predictions: [pred1 (batch, n_steps), pred2 (batch, n_steps), ...]
        stacked = torch.stack(fm_predictions, dim=1)  # (batch, n_fms, n_steps)

        # Apply weights: (batch, n_fms, 1) × (batch, n_fms, n_steps)
        weights_expanded = weights.unsqueeze(-1)  # (batch, n_fms, 1)
        weighted = stacked * weights_expanded  # (batch, n_fms, n_steps)

        # Sum across FMs
        ensemble_pred = weighted.sum(dim=1)  # (batch, n_steps)

        return ensemble_pred, weights

    def get_weights(self, regime_embedding: torch.Tensor) -> torch.Tensor:
        """Get FM weights for a given regime embedding."""
        logits = self.weight_network(regime_embedding)
        weights = torch.softmax(logits, dim=1)
        return weights


if __name__ == "__main__":
    # Test FM ensemble
    ensemble = FMEnsemble(embedding_dim=32, n_fms=2)

    # Random regime embedding: (batch=4, embedding_dim=32)
    regime_embedding = torch.randn(4, 32)

    # Simulate FM predictions
    # Chronos: (batch=4, n_steps=252)
    # Moirai: (batch=4, n_steps=252)
    fm_pred_chronos = torch.randn(4, 252)
    fm_pred_moirai = torch.randn(4, 252)

    # Ensemble
    ensemble_pred, weights = ensemble(
        regime_embedding, [fm_pred_chronos, fm_pred_moirai]
    )

    print(f"Regime embedding shape: {regime_embedding.shape}")
    print(f"FM predictions shapes: {fm_pred_chronos.shape}, {fm_pred_moirai.shape}")
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights (should sum to 1): {weights.sum(dim=1)}")
    print(f"Sample weights (batch 0): Chronos={weights[0, 0]:.3f}, Moirai={weights[0, 1]:.3f}")

    # Count parameters
    n_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Number of parameters: {n_params:,}")
