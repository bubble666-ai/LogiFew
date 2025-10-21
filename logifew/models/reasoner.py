"""Probabilistic backward chaining engine."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli


class ProbabilisticReasoner(nn.Module):
    """Soft reasoning over encoded memories."""

    def __init__(self, hidden_dim: int, temperature: float = 0.67, proof_depth: int = 3) -> None:
        super().__init__()
        self.temperature = temperature
        self.proof_depth = proof_depth
        self.rule_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.verifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, premises: torch.Tensor, memory_values: torch.Tensor) -> dict:
        """Compute entailment probability and soft proof weights."""
        joint = torch.cat([premises, memory_values], dim=-1)
        gate_logits = self.rule_gate(joint)
        relaxed = RelaxedBernoulli(self.temperature, logits=gate_logits)
        samples = relaxed.rsample()
        proof_state = samples
        for _ in range(self.proof_depth - 1):
            proof_state = proof_state * torch.sigmoid(gate_logits)

        logits = self.verifier(proof_state).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "proof_state": proof_state,
        }
