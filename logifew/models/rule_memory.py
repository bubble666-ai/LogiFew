"""Differentiable rule memory bank."""
from __future__ import annotations

import torch
from torch import nn


class RuleMemoryBank(nn.Module):
    """Key-value memory that stores soft symbolic rules."""

    def __init__(self, num_rules: int, key_dim: int, value_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_rules, key_dim) * 0.05)
        self.values = nn.Parameter(torch.randn(num_rules, value_dim) * 0.05)
        self.dropout = nn.Dropout(dropout)
        self.scale = key_dim**-0.5

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Return weighted combination of value memories given query embeddings.

        Args:
            query: Tensor of shape (batch, key_dim).
        """
        attn_scores = torch.matmul(query, self.keys.t()) * self.scale
        weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(weights, self.values)
        return self.dropout(attended)

    def inject_axioms(self, axioms: torch.Tensor, strength: float = 1.0) -> None:
        """Imprint human-defined axioms directly into the memory values."""
        if axioms.ndim != 2 or axioms.size(1) != self.values.size(1):
            raise ValueError("Axioms must have shape (k, value_dim)")
        num = min(axioms.size(0), self.values.size(0))
        with torch.no_grad():
            self.values[:num].lerp_(axioms[:num], strength)
