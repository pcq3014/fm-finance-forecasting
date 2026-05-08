"""
Regime Encoder — Learned Economic Adaptive Forecasting (LEAF)
============================================================
Input:  Selected macro features (60 ngày gần nhất, shape: (batch, 60, n_features))
Output: Regime embedding (shape: (batch, embedding_dim=32))

Architecture:
  1. MLP layer (n_features → 64 → 32)
  2. Multi-head Attention (4 heads, 60 time steps)
  3. Output projection → 32-dim embedding

Parameters: ~50K
"""

import torch
import torch.nn as nn
import numpy as np


class RegimeEncoder(nn.Module):
    """
    Encode recent macro features into regime embedding.
    Uses MLP + Multi-head Attention to learn economic state representation.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 60,
        embedding_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # MLP projection: n_features → 64 → embedding_dim
        self.mlp_proj = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

        # Multi-head attention over time dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final projection to regime embedding
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_features)

        Returns:
            regime_embedding: (batch_size, embedding_dim)
        """
        batch_size = x.size(0)

        # Project each timestep: (batch, seq_len, n_features) → (batch, seq_len, embedding_dim)
        proj = self.mlp_proj(x)

        # Multi-head attention: (batch, seq_len, embedding_dim)
        attn_out, attn_weights = self.attention(proj, proj, proj)

        # Add residual and normalize
        attn_out = self.layer_norm(proj + attn_out)

        # Aggregate: take mean over time dimension
        # (batch, seq_len, embedding_dim) → (batch, embedding_dim)
        aggregated = attn_out.mean(dim=1)

        # Final projection
        regime_embedding = self.output_proj(aggregated)

        return regime_embedding


if __name__ == "__main__":
    # Test regime encoder
    encoder = RegimeEncoder(n_features=11, seq_len=60, embedding_dim=32)

    # Random input: (batch=4, seq_len=60, n_features=11)
    x = torch.randn(4, 60, 11)
    embedding = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embedding.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters: {n_params:,}")
