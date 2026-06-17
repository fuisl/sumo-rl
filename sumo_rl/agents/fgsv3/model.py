"""FGSv3 neural components: demand communication, FRAP action tokens, and factored critics."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from sumo_rl.agents.frap.model import normalize_phase_pairs

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
except ImportError as exc:  # pragma: no cover - exercised only without optional extras.
    raise ImportError("FGSv3 graph communication requires torch-geometric. Install the rllib-custom extra.") from exc


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


class FRAPDemandActionTokenEncoder(nn.Module):
    """FRAP encoder that exposes Level-B phase demand plus Level-D action tokens."""

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
        adapter_dim: int = 64,
        adapter_hidden_dims: Iterable[int] = (),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if int(num_actions) < 2:
            raise ValueError("FGSv3 FRAP action tokens require at least two actions.")
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
                "FGSv3 FRAP token encoder expects observation tail demand features. "
                f"observation_dim={observation_dim}, demand_start={self.demand_start}, demand_shape={self.demand_shape}."
            )
        self.num_movements = demand_width // self.demand_shape
        self.adapter_dim = int(adapter_dim)
        self.phase_demand_dim = int(lane_embed_units)
        self.phase_pairs = normalize_phase_pairs(
            phase_pairs,
            num_movements=self.num_movements,
            num_actions=self.num_actions,
        )

        self.phase_embedding = nn.Embedding(2, int(p_out))
        self.demand_layer = nn.Linear(self.demand_shape, int(d_out))
        self.lane_embedding = nn.Linear(int(p_out) + int(d_out), self.phase_demand_dim)
        self.lane_conv = nn.Conv2d(2 * self.phase_demand_dim, int(conv_units), kernel_size=(1, 1))
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
        raise ValueError(f"Unsupported FGSv3 demand_layout: {self.demand_layout!r}.")

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

    def _ordered_competition_mask(
        self,
        action_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
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

    def _valid_actions(
        self,
        action_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if action_mask is None:
            return torch.ones((batch_size, self.num_actions), dtype=torch.float32, device=device)
        valid = action_mask.to(device=device, dtype=torch.float32)[:, : self.num_actions]
        if int(valid.shape[-1]) < self.num_actions:
            valid = F.pad(valid, (0, self.num_actions - int(valid.shape[-1])))
        return valid

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

        phase_demand = self._phase_pair_embeddings(movement_embeds, pair_mask)
        ordered_competitions = []
        for action_index in range(self.num_actions):
            phase_embed = phase_demand[:, action_index]
            for other_index in range(self.num_actions):
                if action_index != other_index:
                    other_phase_embed = phase_demand[:, other_index]
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
        valid_actions = self._valid_actions(action_mask, batch_size=batch_size, device=obs.device)
        action_tokens = action_tokens * valid_actions.unsqueeze(-1)
        node_summary = self.summary_norm(_masked_mean(action_tokens, valid_actions, dim=1))
        return {
            "phase_demand": phase_demand,
            "action_tokens": action_tokens,
            "node_summary": node_summary,
        }


class DemandCommunicationBranch(nn.Module):
    """Build per-node communication features from phase demand and own previous action."""

    def __init__(self, *, phase_demand_dim: int, phase_emb_dim: int, demand_comm_dim: int, num_actions: int) -> None:
        super().__init__()
        self.phase_demand_dim = int(phase_demand_dim)
        self.phase_emb_dim = int(phase_emb_dim)
        self.demand_comm_dim = int(demand_comm_dim)
        self.num_actions = int(num_actions)
        self.padding_idx = self.num_actions
        self.phase_emb = nn.Embedding(self.num_actions + 1, self.phase_emb_dim, padding_idx=self.padding_idx)
        self.proj = nn.Linear(self.phase_demand_dim + self.phase_emb_dim, self.demand_comm_dim)
        self.norm = nn.LayerNorm(self.demand_comm_dim)
        nn.init.uniform_(self.phase_emb.weight, -1.0e-3, 1.0e-3)
        with torch.no_grad():
            self.phase_emb.weight[self.padding_idx].zero_()

    def previous_action_indices(self, prev_joint_action: torch.Tensor) -> torch.Tensor:
        prev = prev_joint_action.float()[..., : self.num_actions]
        has_previous = torch.sum(prev, dim=-1) > 0.0
        indices = torch.argmax(prev, dim=-1).long()
        padding = torch.full_like(indices, self.padding_idx)
        return torch.where(has_previous, indices, padding)

    def forward(
        self,
        phase_demand: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        prev_joint_action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, num_actions, channels = phase_demand.shape
        if int(num_actions) != self.num_actions or int(channels) != self.phase_demand_dim:
            raise ValueError(
                "FGSv3 demand branch received unexpected phase_demand shape: "
                f"{tuple(phase_demand.shape)}; expected (*, *, {self.num_actions}, {self.phase_demand_dim})."
            )
        if action_mask is None:
            valid_actions = torch.ones(
                (batch_size, num_nodes, self.num_actions),
                dtype=phase_demand.dtype,
                device=phase_demand.device,
            )
        else:
            valid_actions = action_mask.to(device=phase_demand.device, dtype=phase_demand.dtype)[..., : self.num_actions]
            if int(valid_actions.shape[-1]) < self.num_actions:
                valid_actions = F.pad(valid_actions, (0, self.num_actions - int(valid_actions.shape[-1])))
        pooled = _masked_mean(phase_demand, valid_actions, dim=2)

        prev = prev_joint_action.to(device=phase_demand.device)
        if int(prev.shape[-1]) < self.num_actions:
            prev = F.pad(prev, (0, self.num_actions - int(prev.shape[-1])))
        prev_indices = self.previous_action_indices(prev)
        phase_intent = self.phase_emb(prev_indices.reshape(batch_size * num_nodes)).reshape(
            batch_size,
            num_nodes,
            self.phase_emb_dim,
        )
        x = torch.cat((pooled, phase_intent), dim=-1)
        return self.norm(F.relu(self.proj(x)))


class WeightedDemandGATv2Layer(MessagePassing):
    """GATv2-style attention with edge-weight log bias for demand communication."""

    def __init__(
        self,
        *,
        input_dim: int,
        head_dim: int = 16,
        output_dim: int = 64,
        num_heads: int = 4,
        negative_slope: float = 0.2,
        edge_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.input_dim = int(input_dim)
        self.head_dim = int(head_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.negative_slope = float(negative_slope)
        self.edge_epsilon = float(edge_epsilon)
        if self.num_heads < 1:
            raise ValueError("FGSv3 weighted GATv2 requires at least one attention head.")

        projected_dim = self.head_dim * self.num_heads
        self.left_projection = nn.Linear(self.input_dim, projected_dim)
        self.right_projection = nn.Linear(self.input_dim, projected_dim)
        self.message_projection = nn.Linear(self.input_dim, projected_dim)
        self.attention = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.output_projection = nn.Linear(self.head_dim, self.output_dim)
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.shape[1],), dtype=x.dtype, device=x.device)
        else:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype).reshape(-1)
        edge_index, edge_weight = add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=int(x.shape[0]),
        )
        aggregated = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
        return F.relu(self.output_projection(aggregated))

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_weight: torch.Tensor,
        index: torch.Tensor,
        ptr=None,
        size_i=None,
    ) -> torch.Tensor:
        left = self.left_projection(x_i).view(-1, self.num_heads, self.head_dim)
        right = self.right_projection(x_j).view(-1, self.num_heads, self.head_dim)
        scores = torch.sum(
            F.leaky_relu(left + right, negative_slope=self.negative_slope) * self.attention.unsqueeze(0),
            dim=-1,
        )
        scores = scores + torch.log(edge_weight.clamp_min(self.edge_epsilon)).unsqueeze(-1)
        alpha = softmax(scores, index=index, ptr=ptr, num_nodes=size_i)
        messages = F.relu(self.message_projection(x_j)).view(-1, self.num_heads, self.head_dim)
        return torch.mean(messages * alpha.unsqueeze(-1), dim=1)


class FGSv3GraphEncoder(nn.Module):
    """Encode graph observations into FRAP action tokens plus demand-GAT context."""

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
        self.token_dim = int(adapter_config.get("dim", 64))
        self.local_encoder = FRAPDemandActionTokenEncoder(
            observation_dim=self.node_feature_dim,
            num_actions=self.num_actions,
            adapter_dim=self.token_dim,
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
        self.output_dim = int(communication.get("demand_comm_dim", communication.get("output_dim", 64)))
        self.demand_branch = DemandCommunicationBranch(
            phase_demand_dim=self.local_encoder.phase_demand_dim,
            phase_emb_dim=int(communication.get("phase_emb_dim", 16)),
            demand_comm_dim=self.output_dim,
            num_actions=self.num_actions,
        )
        self.demand_norm = nn.LayerNorm(self.output_dim)
        self.context_norm = nn.LayerNorm(self.output_dim)
        if self.communication_enabled and self.communication_type == "gatv2":
            self.gnn = WeightedDemandGATv2Layer(
                input_dim=self.output_dim,
                head_dim=int(communication.get("head_dim", 16)),
                output_dim=self.output_dim,
                num_heads=int(communication.get("num_heads", 4)),
                negative_slope=float(communication.get("negative_slope", 0.2)),
                edge_epsilon=float(communication.get("edge_epsilon", 1.0e-6)),
            )
        elif self.communication_enabled and self.communication_type != "identity":
            raise ValueError("FGSv3 communication.type must be one of: gatv2, identity.")
        else:
            self.gnn = None
        self.residual_gate = nn.Parameter(torch.tensor(float(communication.get("residual_gate_init", 0.0))))

    def _flatten_edges(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = obs["node_features"]
        batch_size, num_nodes = int(node_features.shape[0]), int(node_features.shape[1])
        edge_index = obs["edge_index"].to(device=node_features.device, dtype=torch.long)
        edge_mask = obs["edge_mask"].to(device=node_features.device) > 0
        raw_edge_weight = obs.get("edge_weight")
        if raw_edge_weight is None:
            raw_edge_weight = torch.ones_like(edge_mask, dtype=node_features.dtype, device=node_features.device)
        else:
            raw_edge_weight = raw_edge_weight.to(device=node_features.device, dtype=node_features.dtype)

        valid = edge_mask.reshape(-1)
        if not torch.any(valid):
            return (
                torch.empty((2, 0), dtype=torch.long, device=node_features.device),
                torch.empty((0,), dtype=node_features.dtype, device=node_features.device),
            )
        offsets = torch.arange(batch_size, dtype=torch.long, device=node_features.device).view(batch_size, 1, 1) * num_nodes
        flat_edges = (edge_index + offsets).permute(1, 0, 2).reshape(2, -1)
        flat_weights = raw_edge_weight.reshape(-1)
        return flat_edges[:, valid], flat_weights[valid]

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_features = obs["node_features"].float()
        batch_size, num_nodes, feature_dim = node_features.shape
        if int(num_nodes) != self.num_nodes or int(feature_dim) != self.node_feature_dim:
            raise ValueError(
                "FGSv3 received graph observations with unexpected shape: "
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
        action_tokens = encoded["action_tokens"].reshape(batch_size, num_nodes, self.num_actions, self.token_dim)
        phase_demand = encoded["phase_demand"].reshape(
            batch_size,
            num_nodes,
            self.num_actions,
            self.local_encoder.phase_demand_dim,
        )
        if node_action_mask is None:
            node_action_mask = torch.ones((batch_size, num_nodes, self.num_actions), dtype=node_features.dtype, device=node_features.device)
        prev_joint_action = obs.get("prev_joint_action")
        if prev_joint_action is None:
            prev_joint_action = torch.zeros((batch_size, num_nodes, self.num_actions), dtype=node_features.dtype, device=node_features.device)
        demand_context = self.demand_branch(phase_demand, node_action_mask.float(), prev_joint_action.float())
        if self.gnn is not None:
            flat_demand = self.demand_norm(demand_context).reshape(batch_size * num_nodes, self.output_dim)
            edge_index, edge_weight = self._flatten_edges(obs)
            gnn_context = self.gnn(flat_demand, edge_index, edge_weight).reshape(batch_size, num_nodes, self.output_dim)
            graph_context = self.context_norm(demand_context + self.residual_gate * gnn_context)
        else:
            graph_context = self.context_norm(demand_context)
        ego_index = obs["ego_index"].long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_context = graph_context[torch.arange(batch_size, device=graph_context.device), ego_index]
        ego_tokens = action_tokens[torch.arange(batch_size, device=action_tokens.device), ego_index]
        return {
            "action_tokens": action_tokens,
            "phase_demand": phase_demand,
            "graph": graph_context,
            "ego": ego_context,
            "ego_action_tokens": ego_tokens,
        }


class ActionConditionedActor(nn.Module):
    """Produce one SAC logit per action from FRAP action tokens and demand-GAT context."""

    def __init__(
        self,
        *,
        token_dim: int,
        graph_dim: int,
        num_actions: int,
        hidden_dims: Iterable[int] = (128,),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.graph_dim = int(graph_dim)
        self.num_actions = int(num_actions)
        self.net = _mlp(self.token_dim + self.graph_dim + self.num_actions, hidden_dims, 1, activation=activation)

    def forward(self, action_tokens: torch.Tensor, graph_context: torch.Tensor) -> torch.Tensor:
        batch_size, num_actions, _ = action_tokens.shape
        action_eye = F.one_hot(torch.arange(num_actions, device=action_tokens.device), num_classes=num_actions).float()
        action_eye = action_eye.unsqueeze(0).expand(batch_size, -1, -1)
        context = graph_context.unsqueeze(1).expand(-1, num_actions, -1)
        features = torch.cat((action_tokens, context, action_eye), dim=-1)
        return self.net(features.reshape(batch_size * num_actions, self.token_dim + self.graph_dim + self.num_actions)).reshape(
            batch_size,
            num_actions,
        )


class FactoredNeighborhoodActionTokenCritic(nn.Module):
    """Fixed-width critic using normalized weighted sums over direct neighbors."""

    def __init__(
        self,
        *,
        token_dim: int,
        graph_dim: int,
        num_nodes: int,
        num_actions: int,
        hidden_dims: Iterable[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.graph_dim = int(graph_dim)
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        input_dim = self.token_dim + 2 * self.graph_dim + 2 * self.num_actions + self.num_nodes
        self.net = _mlp(input_dim, hidden_dims, 1, activation=activation)

    def neighbor_context(
        self,
        graph_h: torch.Tensor,
        action_context: torch.Tensor,
        ego_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, graph_dim = graph_h.shape
        ego_index = ego_index.long().reshape(batch_size).clamp(0, num_nodes - 1)
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_mask, dtype=graph_h.dtype, device=graph_h.device)
        else:
            edge_weight = edge_weight.to(device=graph_h.device, dtype=graph_h.dtype)
        edge_index = edge_index.to(device=graph_h.device, dtype=torch.long)
        edge_mask = edge_mask.to(device=graph_h.device) > 0
        action_context = action_context.to(device=graph_h.device, dtype=graph_h.dtype).reshape(
            batch_size,
            num_nodes,
            self.num_actions,
        )

        sources = edge_index[:, 0, :].clamp(0, num_nodes - 1)
        targets = edge_index[:, 1, :].clamp(0, num_nodes - 1)
        selected = edge_mask & (targets == ego_index.unsqueeze(-1))
        weights = edge_weight.clamp_min(0.0) * selected.to(dtype=graph_h.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        normalized = weights / weight_sum

        source_h = torch.gather(graph_h, dim=1, index=sources.unsqueeze(-1).expand(-1, -1, graph_dim))
        source_a = torch.gather(action_context, dim=1, index=sources.unsqueeze(-1).expand(-1, -1, self.num_actions))
        h_neighbors = torch.sum(source_h * normalized.unsqueeze(-1), dim=1)
        a_neighbors = torch.sum(source_a * normalized.unsqueeze(-1), dim=1)
        return h_neighbors, a_neighbors

    def forward(
        self,
        graph_h: torch.Tensor,
        ego_action_tokens: torch.Tensor,
        action_context: torch.Tensor,
        ego_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = graph_h.shape
        ego_index = ego_index.long().reshape(batch_size).clamp(0, num_nodes - 1)
        ego_h = graph_h[torch.arange(batch_size, device=graph_h.device), ego_index]
        ego_one_hot = F.one_hot(ego_index, num_classes=num_nodes).float()
        h_neighbors, a_neighbors = self.neighbor_context(
            graph_h,
            action_context,
            ego_index,
            edge_index,
            edge_mask,
            edge_weight,
        )

        candidate_actions = F.one_hot(
            torch.arange(self.num_actions, device=graph_h.device),
            num_classes=self.num_actions,
        ).float()
        ego_h = ego_h.unsqueeze(1).expand(-1, self.num_actions, -1)
        h_neighbors = h_neighbors.unsqueeze(1).expand(-1, self.num_actions, -1)
        a_neighbors = a_neighbors.unsqueeze(1).expand(-1, self.num_actions, -1)
        ego_one_hot = ego_one_hot.unsqueeze(1).expand(-1, self.num_actions, -1)
        candidate_actions = candidate_actions.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat(
            (
                ego_action_tokens,
                ego_h,
                h_neighbors,
                a_neighbors,
                candidate_actions,
                ego_one_hot,
            ),
            dim=-1,
        )
        return self.net(features.reshape(batch_size * self.num_actions, -1)).reshape(batch_size, self.num_actions)
