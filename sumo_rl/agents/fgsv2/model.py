"""FGSv2 neural components: FRAP action tokens, residual GNN context, and CTDE critics."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from sumo_rl.agents.frap.model import normalize_phase_pairs
from sumo_rl.agents.graph_attention import CoLightGATLayer, CoLightGATv2Layer


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = torch.sum(values * mask, dim=dim)
    denominator = torch.sum(mask, dim=dim).clamp_min(1.0)
    return numerator / denominator


class FRAPActionTokenEncoder(nn.Module):
    """FRAP phase-competition block that preserves one token per candidate action."""

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
        adapter_dim: int = 128,
        adapter_hidden_dims: Iterable[int] = (),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if int(num_actions) < 2:
            raise ValueError("FGSv2 FRAP action tokens require at least two actions.")
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
                "FGSv2 FRAP token encoder expects observation tail demand features. "
                f"observation_dim={observation_dim}, demand_start={self.demand_start}, demand_shape={self.demand_shape}."
            )
        self.num_movements = demand_width // self.demand_shape
        self.adapter_dim = int(adapter_dim)
        self.phase_pairs = normalize_phase_pairs(
            phase_pairs,
            num_movements=self.num_movements,
            num_actions=self.num_actions,
        )

        self.phase_embedding = nn.Embedding(2, int(p_out))
        self.demand_layer = nn.Linear(self.demand_shape, int(d_out))
        self.lane_embedding = nn.Linear(int(p_out) + int(d_out), int(lane_embed_units))
        self.lane_conv = nn.Conv2d(2 * int(lane_embed_units), int(conv_units), kernel_size=(1, 1))
        self.relation_embedding = nn.Embedding(2, int(relation_embed_size))
        self.relation_conv = nn.Conv2d(int(relation_embed_size), int(conv_units), kernel_size=(1, 1))
        self.hidden_layer = nn.Conv2d(int(conv_units), int(conv_units), kernel_size=(1, 1))
        self.action_adapter = _mlp(int(conv_units), adapter_hidden_dims, self.adapter_dim, activation=activation)
        self.action_norm = nn.LayerNorm(self.adapter_dim)
        self.summary_norm = nn.LayerNorm(self.adapter_dim)

        phase_pair_mask = torch.zeros((self.num_actions, self.num_movements), dtype=torch.float32)
        for action_index, pair in enumerate(self.phase_pairs):
            phase_pair_mask[action_index, pair[0]] = 1.0
            phase_pair_mask[action_index, pair[1]] = 1.0
        self.register_buffer("default_phase_pair_mask", phase_pair_mask, persistent=False)

    def _movement_demands(self, obs: torch.Tensor) -> torch.Tensor:
        demand = obs[:, self.demand_start : self.demand_start + self.num_movements * self.demand_shape].float()
        if self.demand_layout == "split":
            return demand.reshape(obs.shape[0], self.demand_shape, self.num_movements).transpose(1, 2)
        if self.demand_layout == "interleaved":
            return demand.reshape(obs.shape[0], self.num_movements, self.demand_shape)
        raise ValueError(f"Unsupported FGSv2 demand_layout: {self.demand_layout!r}.")

    def _default_phase_pair_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.default_phase_pair_mask.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _canonical_phase_pair_mask(self, phase_pair_mask: Optional[torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
        batch_size = int(obs.shape[0])
        if phase_pair_mask is None:
            return self._default_phase_pair_mask(batch_size, obs.device, obs.dtype)
        mask = phase_pair_mask.to(device=obs.device, dtype=obs.dtype)[:, : self.num_actions, : self.num_movements]
        if int(mask.shape[-1]) < self.num_movements:
            mask = F.pad(mask, (0, self.num_movements - int(mask.shape[-1])))
        return mask

    def _current_phase_movements(self, obs: torch.Tensor, phase_pair_mask: torch.Tensor) -> torch.Tensor:
        if not self.observation_has_phase:
            return torch.zeros((obs.shape[0], self.num_movements), dtype=torch.long, device=obs.device)
        phase_indices = torch.argmax(obs[:, : self.num_actions], dim=-1).clamp(0, self.num_actions - 1)
        batch_indices = torch.arange(obs.shape[0], device=obs.device)
        return (phase_pair_mask[batch_indices, phase_indices] > 0).long()

    def _phase_pair_embeddings(self, movement_embeds: torch.Tensor, phase_pair_mask: torch.Tensor) -> torch.Tensor:
        counts = torch.sum(phase_pair_mask, dim=-1, keepdim=True)
        scale = torch.clamp(counts, max=2.0)
        weights = torch.where(counts > 0.0, phase_pair_mask * scale / counts.clamp_min(1.0), phase_pair_mask)
        return torch.bmm(weights, movement_embeds)

    def _ordered_competition_mask(self, action_mask: Optional[torch.Tensor], *, batch_size: int, device: torch.device) -> torch.Tensor:
        if action_mask is None:
            valid = torch.ones((batch_size, self.num_actions), dtype=torch.float32, device=device)
        else:
            valid = action_mask.to(device=device, dtype=torch.float32)[:, : self.num_actions]
            if int(valid.shape[-1]) < self.num_actions:
                valid = F.pad(valid, (0, self.num_actions - int(valid.shape[-1])))

        rows = []
        for action_index in range(self.num_actions):
            competitors = []
            for other_index in range(self.num_actions):
                if action_index != other_index:
                    competitors.append(valid[:, action_index] * valid[:, other_index])
            rows.append(torch.stack(competitors, dim=-1))
        return torch.stack(rows, dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        phase_pair_mask: Optional[torch.Tensor] = None,
        competition_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs = obs.float()
        batch_size = int(obs.shape[0])
        pair_mask = self._canonical_phase_pair_mask(phase_pair_mask, obs)
        phase_movements = self._current_phase_movements(obs, pair_mask)
        phase_embeds = torch.sigmoid(self.phase_embedding(phase_movements))
        demand_embeds = torch.sigmoid(self.demand_layer(self._movement_demands(obs)))
        movement_embeds = F.relu(self.lane_embedding(torch.cat((phase_embeds, demand_embeds), dim=-1)))

        phase_pair_embeds = self._phase_pair_embeddings(movement_embeds, pair_mask)
        ordered_competitions = []
        for action_index in range(self.num_actions):
            phase_embed = phase_pair_embeds[:, action_index]
            for other_index in range(self.num_actions):
                if action_index != other_index:
                    other_phase_embed = phase_pair_embeds[:, other_index]
                    ordered_competitions.append(torch.cat((phase_embed, other_phase_embed), dim=-1))
        competitions = torch.stack(ordered_competitions, dim=1)
        competitions = competitions.reshape(batch_size, self.num_actions, self.num_actions - 1, -1).permute(0, 3, 1, 2)
        phase_features = F.relu(self.lane_conv(competitions))

        if competition_mask is None:
            relation_mask = torch.zeros((batch_size, self.num_actions, self.num_actions - 1), dtype=torch.long, device=obs.device)
        else:
            relation_mask = competition_mask.to(device=obs.device, dtype=torch.long)[:, : self.num_actions, : self.num_actions - 1]
            if int(relation_mask.shape[-1]) < self.num_actions - 1:
                relation_mask = F.pad(relation_mask, (0, self.num_actions - 1 - int(relation_mask.shape[-1])))
        relation_features = F.relu(self.relation_embedding(relation_mask.long())).permute(0, 3, 1, 2)
        relation_features = F.relu(self.relation_conv(relation_features))
        competition_features = F.relu(self.hidden_layer(phase_features * relation_features)).permute(0, 2, 3, 1)

        competitor_mask = self._ordered_competition_mask(action_mask, batch_size=batch_size, device=obs.device)
        pooled = _masked_mean(competition_features, competitor_mask, dim=2)
        action_tokens = self.action_norm(F.relu(self.action_adapter(pooled)))
        if action_mask is None:
            valid_actions = torch.ones((batch_size, self.num_actions), dtype=torch.float32, device=obs.device)
        else:
            valid_actions = action_mask.to(device=obs.device, dtype=torch.float32)[:, : self.num_actions]
            if int(valid_actions.shape[-1]) < self.num_actions:
                valid_actions = F.pad(valid_actions, (0, self.num_actions - int(valid_actions.shape[-1])))
        action_tokens = action_tokens * valid_actions.unsqueeze(-1)
        node_summary = self.summary_norm(_masked_mean(action_tokens, valid_actions, dim=1))
        return {"action_tokens": action_tokens, "node_summary": node_summary}


class FGSv2GraphEncoder(nn.Module):
    """Encode graph observations into FRAP action tokens plus residual GNN context."""

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
        adapter_config = dict(model_config.get("adapter", {}) or {})
        frap_config = dict(model_config.get("frap", {}) or {})
        communication = dict(model_config.get("communication", {}) or {})
        self.adapter_dim = int(adapter_config.get("dim", 128))
        self.local_encoder = FRAPActionTokenEncoder(
            observation_dim=self.node_feature_dim,
            num_actions=self.num_actions,
            adapter_dim=self.adapter_dim,
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
            adapter_hidden_dims=adapter_config.get("hidden_dims", []),
            activation=str(adapter_config.get("activation", "relu")),
        )
        self.communication_enabled = bool(communication.get("enabled", True))
        self.communication_type = str(communication.get("type", "gatv2") or "gatv2").lower()
        self.summary_norm = nn.LayerNorm(self.adapter_dim)
        self.context_norm = nn.LayerNorm(self.adapter_dim)
        if self.communication_enabled and self.communication_type in {"gat", "gatv2"}:
            layer_cls = CoLightGATv2Layer if self.communication_type == "gatv2" else CoLightGATLayer
            self.gnn = layer_cls(
                input_dim=self.adapter_dim,
                head_dim=int(communication.get("head_dim", 16)),
                output_dim=self.adapter_dim,
                num_heads=int(communication.get("num_heads", 4)),
            )
        elif self.communication_enabled and self.communication_type != "identity":
            raise ValueError("FGSv2 communication.type must be one of: gat, gatv2, identity.")
        else:
            self.gnn = None
        self.residual_gate = nn.Parameter(torch.tensor(float(communication.get("residual_gate_init", 0.0))))
        self.output_dim = self.adapter_dim

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
                "FGSv2 received graph observations with unexpected shape: "
                f"{tuple(node_features.shape)}; expected (*, {self.num_nodes}, {self.node_feature_dim})."
            )
        flat_node_features = node_features.reshape(batch_size * num_nodes, feature_dim)
        phase_pair_mask = obs.get("phase_pair_mask")
        competition_mask = obs.get("phase_competition_mask")
        node_action_mask = obs.get("node_action_mask")
        encoded = self.local_encoder(
            flat_node_features,
            phase_pair_mask=(
                phase_pair_mask.float().reshape(batch_size * num_nodes, self.num_actions, -1)
                if phase_pair_mask is not None
                else None
            ),
            competition_mask=(
                competition_mask.float().reshape(batch_size * num_nodes, self.num_actions, -1)
                if competition_mask is not None
                else None
            ),
            action_mask=(
                node_action_mask.float().reshape(batch_size * num_nodes, self.num_actions)
                if node_action_mask is not None
                else None
            ),
        )
        action_tokens = encoded["action_tokens"].reshape(batch_size, num_nodes, self.num_actions, self.adapter_dim)
        node_summary = encoded["node_summary"].reshape(batch_size, num_nodes, self.adapter_dim)
        if self.gnn is not None:
            flat_summary = self.summary_norm(node_summary).reshape(batch_size * num_nodes, self.adapter_dim)
            gnn_context = self.gnn(flat_summary, self._flatten_edges(obs)).reshape(batch_size, num_nodes, self.adapter_dim)
            graph_context = self.context_norm(node_summary + self.residual_gate * gnn_context)
        else:
            graph_context = self.context_norm(node_summary)
        ego_index = obs["ego_index"].long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_context = graph_context[torch.arange(batch_size, device=graph_context.device), ego_index]
        ego_tokens = action_tokens[torch.arange(batch_size, device=action_tokens.device), ego_index]
        return {
            "action_tokens": action_tokens,
            "node_summary": node_summary,
            "graph": graph_context,
            "ego": ego_context,
            "ego_action_tokens": ego_tokens,
        }


class ActionConditionedActor(nn.Module):
    """Produce one SAC logit per action from FRAP action tokens and GNN context."""

    def __init__(self, *, token_dim: int, num_actions: int, hidden_dims: Iterable[int] = (128,), activation: str = "relu") -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.net = _mlp(int(token_dim) * 2 + self.num_actions, hidden_dims, 1, activation=activation)

    def forward(self, action_tokens: torch.Tensor, graph_context: torch.Tensor) -> torch.Tensor:
        batch_size, num_actions, token_dim = action_tokens.shape
        action_eye = F.one_hot(torch.arange(num_actions, device=action_tokens.device), num_classes=num_actions).float()
        action_eye = action_eye.unsqueeze(0).expand(batch_size, -1, -1)
        context = graph_context.unsqueeze(1).expand(-1, num_actions, -1)
        features = torch.cat((action_tokens, context, action_eye), dim=-1)
        return self.net(features.reshape(batch_size * num_actions, 2 * token_dim + num_actions)).reshape(batch_size, num_actions)


class CentralGraphActionTokenCritic(nn.Module):
    """Centralized critic over graph context, ego action tokens, and joint-action context."""

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
        self.graph_dim = int(graph_dim)
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        input_dim = (
            self.num_nodes * self.graph_dim
            + self.num_nodes * self.num_actions
            + self.graph_dim
            + self.graph_dim
            + self.num_nodes
        )
        self.net = _mlp(input_dim, hidden_dims, 1, activation=activation)

    def forward(
        self,
        graph_h: torch.Tensor,
        ego_action_tokens: torch.Tensor,
        joint_action_context: torch.Tensor,
        ego_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, graph_dim = graph_h.shape
        ego_index = ego_index.long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_h = graph_h[torch.arange(batch_size, device=graph_h.device), ego_index]
        ego_one_hot = F.one_hot(ego_index, num_classes=num_nodes).float()
        base_context = joint_action_context.float().reshape(batch_size, num_nodes, self.num_actions)

        candidate_contexts = base_context.unsqueeze(1).repeat(1, self.num_actions, 1, 1)
        candidate_actions = F.one_hot(
            torch.arange(self.num_actions, device=graph_h.device),
            num_classes=self.num_actions,
        ).float()
        batch_indices = torch.arange(batch_size, device=graph_h.device).unsqueeze(-1)
        action_indices = torch.arange(self.num_actions, device=graph_h.device).unsqueeze(0)
        candidate_contexts[batch_indices, action_indices, ego_index.unsqueeze(-1), :] = candidate_actions.unsqueeze(0)

        graph_flat = graph_h.reshape(batch_size, num_nodes * graph_dim).unsqueeze(1).expand(-1, self.num_actions, -1)
        action_flat = candidate_contexts.reshape(batch_size, self.num_actions, num_nodes * self.num_actions)
        ego_flat = ego_h.unsqueeze(1).expand(-1, self.num_actions, -1)
        ego_index_flat = ego_one_hot.unsqueeze(1).expand(-1, self.num_actions, -1)
        features = torch.cat((graph_flat, action_flat, ego_action_tokens, ego_flat, ego_index_flat), dim=-1)
        return self.net(features.reshape(batch_size * self.num_actions, -1)).reshape(batch_size, self.num_actions)
