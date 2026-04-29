"""Local-neighbor graph SAC baseline for RESCO traffic simulation."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

from .actor import SharedDiscreteActor
from .critic import CentralizedTwinCritic
from .graph_encoder import GraphEncoder


class LocalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NeighborEncoder(GraphEncoder):
    pass


class FusionMLP(nn.Module):
    def __init__(self, local_dim: int, neighbor_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_dim + neighbor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ELU(),
        )

    def forward(self, z_local: Tensor, z_neighbor: Tensor) -> Tensor:
        return self.net(torch.cat([z_local, z_neighbor], dim=-1))


class LocalNeighborGATDiscreteSAC(nn.Module):
    """Discrete SAC agent with local encoding and graph neighbor aggregation."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        local_encoder_cfg: dict | None = None,
        neighbor_encoder_cfg: dict | None = None,
        fusion_cfg: dict | None = None,
        actor_cfg: dict | None = None,
        critic_cfg: dict | None = None,
        init_alpha: float = 0.2,
        tau: float = 0.005,
    ) -> None:
        super().__init__()

        local_encoder_cfg = local_encoder_cfg or {}
        neighbor_encoder_cfg = neighbor_encoder_cfg or {}
        fusion_cfg = fusion_cfg or {}
        actor_cfg = actor_cfg or {}
        critic_cfg = critic_cfg or {}

        local_out_dim = int(local_encoder_cfg.get("out_dim", 64))
        neighbor_out_dim = int(neighbor_encoder_cfg.get("out_dim", 64))
        fusion_out_dim = int(fusion_cfg.get("out_dim", 64))

        self.local_encoder = LocalEncoder(in_dim=obs_dim, **local_encoder_cfg)
        self.neighbor_encoder = NeighborEncoder(in_dim=obs_dim, **neighbor_encoder_cfg)
        self.fusion = FusionMLP(
            local_dim=local_out_dim,
            neighbor_dim=neighbor_out_dim,
            **fusion_cfg,
        )

        self.actor = SharedDiscreteActor(latent_dim=fusion_out_dim, num_actions=num_actions, **actor_cfg)
        self.critic = CentralizedTwinCritic(latent_dim=fusion_out_dim, num_actions=num_actions, **critic_cfg)
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.log_alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32).log())
        self.target_entropy = 0.98 * math.log(float(max(num_actions, 2)))
        self.tau = float(tau)
        self.num_actions = int(num_actions)

    def encode(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> Tensor:
        del agent_node_indices, agent_node_mask
        z_local = self.local_encoder(obs)
        z_neighbor = self.neighbor_encoder(obs, edge_index, edge_attr)
        return self.fusion(z_local, z_neighbor)

    def select_action(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        z = self.encode(obs, edge_index, edge_attr, agent_node_indices=agent_node_indices, agent_node_mask=agent_node_mask)
        return self.actor.get_action(z, action_mask, deterministic)

    def get_action_probs(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encode(obs, edge_index, edge_attr, agent_node_indices=agent_node_indices, agent_node_mask=agent_node_mask)
        action_probs, log_action_probs = self.actor.get_action_probs(z, action_mask)
        return z, action_probs, log_action_probs

    def critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        z = self.encode(obs, edge_index, edge_attr, agent_node_indices=agent_node_indices, agent_node_mask=agent_node_mask)
        return self.critic(z)

    def target_critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            z = self.encode(obs, edge_index, edge_attr, agent_node_indices=agent_node_indices, agent_node_mask=agent_node_mask)
            return self.target_critic(z)

    @torch.no_grad()
    def soft_update_target(self) -> None:
        for target_param, online_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(online_param.data, alpha=self.tau)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()
