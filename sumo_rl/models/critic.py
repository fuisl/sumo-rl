"""Centralized twin Q-critics for discrete SAC."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    def __init__(self, latent_dim: int = 64, context_dim: int = 64, num_actions: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z_local: Tensor, context: Tensor) -> Tensor:
        return self.net(torch.cat([z_local, context], dim=-1))


class CentralizedTwinCritic(nn.Module):
    def __init__(self, latent_dim: int = 64, num_actions: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = QNetwork(latent_dim, latent_dim, num_actions, hidden_dim)
        self.q2 = QNetwork(latent_dim, latent_dim, num_actions, hidden_dim)

    @staticmethod
    def _pool(z: Tensor) -> Tensor:
        return z.mean(dim=0)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        context = self._pool(z)
        context = context.unsqueeze(0).expand_as(z)
        return self.q1(z, context), self.q2(z, context)
