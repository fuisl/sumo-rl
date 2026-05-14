"""Shared discrete actor for phase selection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SharedDiscreteActor(nn.Module):
    """Shared-parameter discrete policy head."""

    def __init__(self, latent_dim: int = 64, num_actions: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: Tensor, action_mask: Tensor | None = None) -> Tensor:
        logits = self.net(z)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return logits

    def get_action(self, z: Tensor, action_mask: Tensor | None = None, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        logits = self.forward(z, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def get_log_prob_entropy(
        self, z: Tensor, actions: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        logits = self.forward(z, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def get_action_probs(self, z: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        logits = self.forward(z, action_mask)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs, torch.log(action_probs.clamp(min=1e-8))
