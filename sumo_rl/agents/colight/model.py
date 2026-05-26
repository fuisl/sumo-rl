"""CoLight graph-attention Q-network."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _as_int_list(values: Iterable[int] | None, default: Sequence[int]) -> list[int]:
    return [int(value) for value in (values if values is not None else default)]


class CoLightGATLayer(nn.Module):
    """Index-free multi-head graph attention layer from CoLight."""

    def __init__(self, input_dim: int, head_dim: int = 16, output_dim: int = 128, num_heads: int = 5) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.head_dim = int(head_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1:
            raise ValueError("CoLight requires at least one attention head.")

        projected_dim = self.head_dim * self.num_heads
        self.target_projection = nn.Linear(self.input_dim, projected_dim)
        self.source_projection = nn.Linear(self.input_dim, projected_dim)
        self.message_projection = nn.Linear(self.input_dim, projected_dim)
        self.output_projection = nn.Linear(self.head_dim, self.output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = int(x.shape[0])
        self_loop_nodes = torch.arange(num_nodes, dtype=torch.long, device=x.device)
        self_loops = torch.stack((self_loop_nodes, self_loop_nodes), dim=0)
        edge_index = torch.cat((edge_index.to(device=x.device), self_loops), dim=1)
        source_index = edge_index[0]
        target_index = edge_index[1]

        target = F.relu(self.target_projection(x[target_index])).view(-1, self.num_heads, self.head_dim)
        source = F.relu(self.source_projection(x[source_index])).view(-1, self.num_heads, self.head_dim)
        scores = torch.sum(target * source, dim=-1)

        scatter_index = target_index.unsqueeze(-1).expand(-1, self.num_heads)
        max_per_target = torch.full(
            (num_nodes, self.num_heads),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        max_per_target.scatter_reduce_(0, scatter_index, scores, reduce="amax", include_self=True)
        centered = scores - max_per_target.index_select(0, target_index)
        exp_scores = torch.exp(centered)
        normalizer = torch.zeros((num_nodes, self.num_heads), dtype=scores.dtype, device=scores.device)
        normalizer.scatter_add_(0, scatter_index, exp_scores)
        alpha = (exp_scores / normalizer.index_select(0, target_index).clamp_min(1e-12)).unsqueeze(-1)

        messages = F.relu(self.message_projection(x[source_index])).view(-1, self.num_heads, self.head_dim)
        messages = torch.mean(messages * alpha, dim=1)
        aggregated = torch.zeros((num_nodes, self.head_dim), dtype=messages.dtype, device=messages.device)
        aggregated.scatter_add_(0, target_index.unsqueeze(-1).expand(-1, self.head_dim), messages)
        return F.relu(self.output_projection(aggregated))


class CoLightQNetwork(nn.Module):
    """Shared graph Q-network for CoLight traffic-signal control."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        num_nodes: int,
        num_actions: int,
        node_embedding_dims: Iterable[int] | None = None,
        num_gat_layers: int = 1,
        num_heads: int | Iterable[int] = 5,
        head_dim: int | Iterable[int] = 16,
        gat_output_dim: int | Iterable[int] = 128,
        output_layers: Iterable[int] | None = None,
        invalid_action_value: float = -1.0e9,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        self.invalid_action_value = float(invalid_action_value)
        if self.num_nodes < 1:
            raise ValueError("CoLight requires at least one graph node.")
        if self.num_actions < 1:
            raise ValueError("CoLight requires at least one action.")

        embedding_dims = _as_int_list(node_embedding_dims, [128, 128])
        embedding = OrderedDict()
        last_dim = self.node_feature_dim
        for index, hidden_dim in enumerate(embedding_dims):
            embedding[f"node_embedding_{index}"] = nn.Linear(last_dim, hidden_dim)
            embedding[f"node_embedding_relu_{index}"] = nn.ReLU()
            last_dim = hidden_dim
        self.node_embedding = nn.Sequential(embedding)

        layer_count = int(num_gat_layers)
        if layer_count < 1:
            raise ValueError("CoLight requires at least one GAT layer.")

        heads = self._expand_layer_values(num_heads, layer_count, "num_heads")
        head_dims = self._expand_layer_values(head_dim, layer_count, "head_dim")
        output_dims = self._expand_layer_values(gat_output_dim, layer_count, "gat_output_dim")
        self.gat_layers = nn.ModuleList()
        for index in range(layer_count):
            layer = CoLightGATLayer(
                input_dim=last_dim,
                head_dim=head_dims[index],
                output_dim=output_dims[index],
                num_heads=heads[index],
            )
            self.gat_layers.append(layer)
            last_dim = output_dims[index]

        q_layers = OrderedDict()
        for index, hidden_dim in enumerate(_as_int_list(output_layers, [])):
            q_layers[f"output_{index}"] = nn.Linear(last_dim, hidden_dim)
            q_layers[f"output_relu_{index}"] = nn.ReLU()
            last_dim = hidden_dim
        q_layers["q_values"] = nn.Linear(last_dim, self.num_actions)
        self.output_layer = nn.Sequential(q_layers)

    @staticmethod
    def _expand_layer_values(value: int | Iterable[int], count: int, name: str) -> list[int]:
        if isinstance(value, int):
            return [int(value)] * count
        values = [int(item) for item in value]
        if len(values) == 1:
            return values * count
        if len(values) != count:
            raise ValueError(f"CoLight {name} must have length 1 or {count}; got {len(values)}.")
        return values

    @classmethod
    def from_model_config(cls, observation_space: Any, action_space: Any, model_config: Dict[str, Any]):
        spaces = observation_space.spaces
        return cls(
            node_feature_dim=int(spaces["node_features"].shape[-1]),
            num_nodes=int(spaces["node_features"].shape[0]),
            num_actions=int(action_space.n),
            node_embedding_dims=model_config.get("node_embedding_dims", [128, 128]),
            num_gat_layers=int(model_config.get("num_gat_layers", model_config.get("n_layers", 1))),
            num_heads=model_config.get("num_heads", 5),
            head_dim=model_config.get("head_dim", 16),
            gat_output_dim=model_config.get("gat_output_dim", 128),
            output_layers=model_config.get("output_layers", []),
            invalid_action_value=float(model_config.get("invalid_action_value", -1.0e9)),
        )

    def _flatten_edges(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = obs["node_features"]
        batch_size, num_nodes = int(node_features.shape[0]), int(node_features.shape[1])
        edge_index = obs["edge_index"].long()
        edge_mask = obs["edge_mask"] > 0

        edges = []
        for batch_index in range(batch_size):
            valid_edges = edge_index[batch_index, :, edge_mask[batch_index]]
            if valid_edges.numel() == 0:
                continue
            edges.append(valid_edges + batch_index * num_nodes)
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=node_features.device)
        return torch.cat(edges, dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = obs["node_features"].float()
        batch_size, num_nodes, feature_dim = node_features.shape
        if int(num_nodes) != self.num_nodes or int(feature_dim) != self.node_feature_dim:
            raise ValueError(
                "CoLight received graph observations with unexpected shape: "
                f"{tuple(node_features.shape)}; expected (*, {self.num_nodes}, {self.node_feature_dim})."
            )

        x = node_features.reshape(batch_size * num_nodes, feature_dim)
        edge_index = self._flatten_edges(obs)
        h = self.node_embedding(x)
        for layer in self.gat_layers:
            h = layer(h, edge_index)

        graph_h = h.reshape(batch_size, num_nodes, -1)
        ego_index = obs["ego_index"].long().reshape(batch_size).clamp(0, num_nodes - 1)
        q_values = self.output_layer(graph_h[torch.arange(batch_size, device=graph_h.device), ego_index])

        action_mask = obs.get("action_mask")
        if action_mask is not None:
            q_values = q_values.masked_fill((action_mask > 0).logical_not(), self.invalid_action_value)
        return q_values
