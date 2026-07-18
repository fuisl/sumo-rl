"""CoLight graph-attention Q-network."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from sumo_rl.agents.graph_attention import CoLightGATLayer


def _as_int_list(values: Iterable[int] | None, default: Sequence[int]) -> list[int]:
    return [int(value) for value in (values if values is not None else default)]


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
    def from_model_config(cls, observation_space: Any, action_space: Any, model_config: dict[str, Any]):
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

    def _flatten_edges(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
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
