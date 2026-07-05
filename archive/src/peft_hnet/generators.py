"""Generator shells shared across adapter representations within a setup."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BasisStateGenerator(nn.Module):
    """Map condition embeddings to a flat adapter state through shared bases.

    The output layout is representation-specific, while the condition encoder,
    residual MLP, and basis count can remain controlled across a comparison.
    """

    def __init__(self, condition_dim: int, state_dim: int, hidden_dim: int, num_bases: int):
        super().__init__()
        self.input_projection = nn.Linear(condition_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coefficients = nn.Linear(hidden_dim, num_bases)
        self.bases = nn.Parameter(torch.empty(num_bases, state_dim))
        nn.init.normal_(self.bases, std=0.02)

    def forward(self, conditions: Tensor) -> Tensor:
        hidden = self.input_projection(conditions)
        hidden = hidden + self.residual(hidden)
        return self.coefficients(hidden) @ self.bases


def reconstruction_loss(predicted: Tensor, target: Tensor, standardize: bool = True) -> Tensor:
    if standardize:
        mean = target.mean(dim=0, keepdim=True)
        scale = target.std(dim=0, keepdim=True).clamp_min(1e-6)
        predicted = (predicted - mean) / scale
        target = (target - mean) / scale
    return torch.mean((predicted - target) ** 2)

