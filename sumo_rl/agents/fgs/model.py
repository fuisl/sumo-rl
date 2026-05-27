"""FGS neural components: FRAP encoder, GAT aggregation, and CTDE critics."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from sumo_rl.agents.frap.model import build_competition_mask, normalize_phase_pairs
from sumo_rl.agents.graph_attention import CoLightGATLayer


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation: str = "relu") -> nn.Sequential:
    act = nn.Tanh if str(activation).lower() == "tanh" else nn.ReLU
    layers = OrderedDict()
    last_dim = int(input_dim)
    for index, hidden_dim in enumerate(hidden_dims):
        layers[f"linear_{index}"] = nn.Linear(last_dim, int(hidden_dim))
        layers[f"activation_{index}"] = act()
        last_dim = int(hidden_dim)
    layers["output"] = nn.Linear(last_dim, int(output_dim))
    return nn.Sequential(layers)


class FRAPEmbeddingEncoder(nn.Module):
    """FRAP phase-competition encoder that returns a local embedding."""

    def __init__(
        self,
        *,
        observation_dim: int,
        num_actions: int,
        phase_pairs: Optional[Iterable[Iterable[int]]] = None,
        demand_shape: int = 2,
        observation_has_phase: bool = True,
        observation_has_min_green: bool = True,
        demand_start: Optional[int] = None,
        demand_layout: str = "split",
        d_out: int = 4,
        p_out: int = 4,
        lane_embed_units: int = 16,
        relation_embed_size: int = 4,
        conv_units: int = 32,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.demand_shape = int(demand_shape)
        self.observation_has_phase = bool(observation_has_phase)
        self.observation_has_min_green = bool(observation_has_min_green)
        self.demand_layout = str(demand_layout)
        self.demand_start = (
            int(demand_start)
            if demand_start is not None
            else (self.num_actions if self.observation_has_phase else 0) + (1 if self.observation_has_min_green else 0)
        )
        demand_width = int(observation_dim) - self.demand_start
        if demand_width <= 0 or demand_width % self.demand_shape != 0:
            raise ValueError(
                "FGS FRAP encoder expects the observation tail to be movement demand features. "
                f"observation_dim={observation_dim}, demand_start={self.demand_start}, demand_shape={self.demand_shape}."
            )
        self.num_movements = demand_width // self.demand_shape
        self.phase_pairs = normalize_phase_pairs(
            phase_pairs,
            num_movements=self.num_movements,
            num_actions=self.num_actions,
        )
        self.phase_embedding = nn.Embedding(2, p_out)
        self.demand_layer = nn.Linear(self.demand_shape, d_out)
        self.lane_embedding = nn.Linear(p_out + d_out, lane_embed_units)
        self.lane_conv = nn.Conv2d(2 * lane_embed_units, conv_units, kernel_size=(1, 1))
        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, conv_units, kernel_size=(1, 1))
        self.hidden_layer = nn.Conv2d(conv_units, conv_units, kernel_size=(1, 1))
        self.output_layer = nn.Linear(conv_units, output_dim)
        self.output_dim = int(output_dim)
        self.register_buffer("competition_mask", build_competition_mask(self.phase_pairs), persistent=False)

    def _current_phase_movements(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.observation_has_phase:
            return torch.zeros((obs.shape[0], self.num_movements), dtype=torch.long, device=obs.device)
        phase_indices = torch.argmax(obs[:, : self.num_actions], dim=-1)
        phase_movements = torch.zeros((obs.shape[0], self.num_movements), dtype=torch.long, device=obs.device)
        for action_index, pair in enumerate(self.phase_pairs):
            selected = phase_indices == action_index
            if torch.any(selected):
                phase_movements[selected, pair[0]] = 1
                phase_movements[selected, pair[1]] = 1
        return phase_movements

    def _movement_demands(self, obs: torch.Tensor) -> torch.Tensor:
        demand = obs[:, self.demand_start : self.demand_start + self.num_movements * self.demand_shape].float()
        if self.demand_layout == "split":
            return demand.reshape(obs.shape[0], self.demand_shape, self.num_movements).transpose(1, 2)
        if self.demand_layout == "interleaved":
            return demand.reshape(obs.shape[0], self.num_movements, self.demand_shape)
        raise ValueError(f"Unsupported FGS FRAP demand_layout: {self.demand_layout!r}.")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        batch_size = obs.shape[0]
        phase_embeds = torch.sigmoid(self.phase_embedding(self._current_phase_movements(obs)))
        demand_embeds = torch.sigmoid(self.demand_layer(self._movement_demands(obs)))
        movement_embeds = F.relu(self.lane_embedding(torch.cat((phase_embeds, demand_embeds), dim=-1)))

        phase_pair_embeds = [movement_embeds[:, pair[0]] + movement_embeds[:, pair[1]] for pair in self.phase_pairs]
        ordered_competitions = []
        for index, phase_embed in enumerate(phase_pair_embeds):
            for other_index, other_phase_embed in enumerate(phase_pair_embeds):
                if index != other_index:
                    ordered_competitions.append(torch.cat((phase_embed, other_phase_embed), dim=-1))
        competitions = torch.stack(ordered_competitions, dim=1)
        competitions = competitions.reshape(batch_size, self.num_actions, self.num_actions - 1, -1).permute(0, 3, 1, 2)
        phase_features = F.relu(self.lane_conv(competitions))

        relation_mask = self.competition_mask.to(device=obs.device).repeat(batch_size, 1, 1)
        relation_features = F.relu(self.relation_embedding(relation_mask)).permute(0, 3, 1, 2)
        relation_features = F.relu(self.relation_conv(relation_features))
        combined = F.relu(self.hidden_layer(phase_features * relation_features))
        pooled = combined.mean(dim=(2, 3))
        return F.relu(self.output_layer(pooled))


class FGSGraphEncoder(nn.Module):
    """Encode all TLS nodes with FRAP/MLP and optional GAT communication."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        num_nodes: int,
        num_actions: int,
        model_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        local_config = dict(model_config.get("local_encoder", {}) or {})
        communication = dict(model_config.get("communication", {}) or {})
        encoder_type = str(local_config.get("type", "frap") or "frap").lower()
        local_output_dim = int(local_config.get("output_dim", 128))
        if encoder_type == "frap":
            frap_config = dict(local_config.get("frap", {}) or {})
            self.local_encoder = FRAPEmbeddingEncoder(
                observation_dim=self.node_feature_dim,
                num_actions=self.num_actions,
                output_dim=local_output_dim,
                phase_pairs=frap_config.get("phase_pairs"),
                demand_shape=int(frap_config.get("demand_shape", 2)),
                observation_has_phase=bool(frap_config.get("observation_has_phase", True)),
                observation_has_min_green=bool(frap_config.get("observation_has_min_green", True)),
                demand_start=frap_config.get("demand_start"),
                demand_layout=str(frap_config.get("demand_layout", "split")),
                d_out=int(frap_config.get("d_out", 4)),
                p_out=int(frap_config.get("p_out", 4)),
                lane_embed_units=int(frap_config.get("lane_embed_units", 16)),
                relation_embed_size=int(frap_config.get("relation_embed_size", 4)),
                conv_units=int(frap_config.get("conv_units", 32)),
            )
        elif encoder_type == "mlp":
            self.local_encoder = _mlp(
                self.node_feature_dim,
                local_config.get("hidden_dims", [128]),
                local_output_dim,
                activation=str(local_config.get("activation", "relu")),
            )
        else:
            raise ValueError("FGS local_encoder.type must be one of: frap, mlp.")

        self.communication_enabled = bool(communication.get("enabled", True))
        self.communication_type = str(communication.get("type", "gat") or "gat").lower()
        self.local_output_dim = local_output_dim
        self.output_dim = local_output_dim
        if self.communication_enabled and self.communication_type == "gat":
            self.gat = CoLightGATLayer(
                input_dim=local_output_dim,
                head_dim=int(communication.get("head_dim", 16)),
                output_dim=int(communication.get("output_dim", local_output_dim)),
                num_heads=int(communication.get("num_heads", 4)),
            )
            self.output_dim = int(communication.get("output_dim", local_output_dim))
        elif self.communication_enabled and self.communication_type != "identity":
            raise ValueError("FGS communication.type must be one of: gat, identity.")
        else:
            self.gat = None

    def _flatten_edges(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = obs["node_features"]
        batch_size, num_nodes = int(node_features.shape[0]), int(node_features.shape[1])
        edge_index = obs["edge_index"].long()
        edge_mask = obs["edge_mask"] > 0
        edges = []
        for batch_index in range(batch_size):
            valid_edges = edge_index[batch_index, :, edge_mask[batch_index]]
            if valid_edges.numel() > 0:
                edges.append(valid_edges + batch_index * num_nodes)
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=node_features.device)
        return torch.cat(edges, dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_features = obs["node_features"].float()
        batch_size, num_nodes, feature_dim = node_features.shape
        if int(num_nodes) != self.num_nodes or int(feature_dim) != self.node_feature_dim:
            raise ValueError(
                "FGS received graph observations with unexpected shape: "
                f"{tuple(node_features.shape)}; expected (*, {self.num_nodes}, {self.node_feature_dim})."
            )
        local = self.local_encoder(node_features.reshape(batch_size * num_nodes, feature_dim))
        if self.gat is not None:
            local = self.gat(local, self._flatten_edges(obs))
        graph_h = local.reshape(batch_size, num_nodes, -1)
        ego_index = obs["ego_index"].long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_h = graph_h[torch.arange(batch_size, device=graph_h.device), ego_index]
        return {"graph": graph_h, "ego": ego_h}


class CentralGraphPolicyCritic(nn.Module):
    """Centralized critic over graph embeddings and all-node policy context."""

    def __init__(
        self,
        *,
        graph_dim: int,
        num_nodes: int,
        num_actions: int,
        hidden_dims: Iterable[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_dim = int(num_nodes) * int(graph_dim) + int(num_nodes) * int(num_actions) + int(graph_dim) + int(num_nodes)
        self.net = _mlp(input_dim, hidden_dims, int(num_actions), activation=activation)
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)

    def forward(self, graph_h: torch.Tensor, all_action_probs: torch.Tensor, ego_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, graph_dim = graph_h.shape
        ego_index = ego_index.long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_h = graph_h[torch.arange(batch_size, device=graph_h.device), ego_index]
        ego_one_hot = F.one_hot(ego_index, num_classes=num_nodes).float()
        features = torch.cat(
            (
                graph_h.reshape(batch_size, num_nodes * graph_dim),
                all_action_probs.reshape(batch_size, num_nodes * self.num_actions),
                ego_h,
                ego_one_hot,
            ),
            dim=-1,
        )
        return self.net(features)
