"""
Context Projector — Transform regime embedding into FM input context
===================================================================
Input:  regime_embedding (batch, embedding_dim=32)
Output: context_tokens (batch, context_length, embedding_dim)

Strategy:
  1. Expand regime embedding into a sequence of context tokens
  2. Learned gating to control influence on FM input
  3. Prepend to price history before feeding to FM

Parameters: ~30K
"""

import torch
import torch.nn as nn


class ContextProjector(nn.Module):
    """
    Project regime embedding into context sequence for FM.
    Learns to encode economic conditions as soft prompts.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        context_length: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        # Expand regime embedding into context sequence
        self.context_generator = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, context_length * embedding_dim),
        )

        # Learned gating: control how much context affects FM input
        # gate ∈ [0, 1], where 1 = full context influence, 0 = no influence
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, regime_embedding: torch.Tensor) -> tuple:
        """
        Args:
            regime_embedding: (batch_size, embedding_dim)

        Returns:
            context_tokens: (batch_size, context_length, embedding_dim)
            gate: (batch_size, 1) — influence strength
        """
        batch_size = regime_embedding.size(0)

        # Generate context sequence
        context_flat = self.context_generator(regime_embedding)
        context_tokens = context_flat.view(
            batch_size, self.context_length, self.embedding_dim
        )

        # Apply layer norm
        context_tokens = self.layer_norm(context_tokens)

        # Compute gating strength
        gate = self.gate_network(regime_embedding)

        # Apply gate: scale context by gate strength
        context_tokens = context_tokens * gate.unsqueeze(-1)

        return context_tokens, gate


if __name__ == "__main__":
    # Test context projector
    projector = ContextProjector(embedding_dim=32, context_length=32, dropout=0.1)

    # Random regime embedding: (batch=4, embedding_dim=32)
    regime_embedding = torch.randn(4, 32)
    context_tokens, gate = projector(regime_embedding)

    print(f"Regime embedding shape: {regime_embedding.shape}")
    print(f"Context tokens shape: {context_tokens.shape}")
    print(f"Gate shape: {gate.shape}")
    print(f"Gate values (should be [0,1]): {gate.squeeze().detach().numpy()}")

    # Count parameters
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"Number of parameters: {n_params:,}")
