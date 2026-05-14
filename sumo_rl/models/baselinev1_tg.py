"""Torch Geometric RLlib model for the baseline-v1 graph policy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BaselineV1TorchGeometricModel(TorchModelV2, nn.Module):
    """Local encoder + neighbor GATv2 encoder + shared actor/value heads.

    The expected observation is produced by
    :class:`sumo_rl.environment.rllib_graph_env.RLLibGraphObservationWrapper`.
    For each agent sample it contains:

    - ``local_obs``: padded observation for the current traffic signal.
    - ``graph_obs``: padded observations for all traffic signals in topology order.
    - ``node_index``: index of the current traffic signal in ``graph_obs``.
    - ``action_mask``: valid padded discrete actions for the current signal.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = dict(model_config.get("custom_model_config", {}))
        base_obs_space = self._base_obs_space(obs_space)
        obs_dim = int(custom_config.get("obs_dim", self._infer_obs_dim(base_obs_space)))
        num_nodes = int(custom_config.get("num_nodes", self._infer_num_nodes(base_obs_space)))

        hidden_dim = int(custom_config.get("hidden_dim", 64))
        latent_dim = int(custom_config.get("latent_dim", 64))
        fusion_dim = int(custom_config.get("fusion_dim", 64))
        actor_hidden = int(custom_config.get("actor_hidden", 128))
        value_hidden = int(custom_config.get("value_hidden", 128))
        heads = int(custom_config.get("heads", 2))
        dropout = float(custom_config.get("dropout", 0.1))
        add_self_loops = bool(custom_config.get("add_self_loops", False))

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GATv2Conv.")

        edge_index = torch.as_tensor(custom_config.get("edge_index", [[], []]), dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_index = edge_index.reshape(2, -1)

        edge_attr = custom_config.get("edge_attr")
        edge_attr_tensor: Optional[Tensor]
        if edge_attr is None:
            edge_attr_tensor = None
        else:
            edge_attr_tensor = torch.as_tensor(edge_attr, dtype=torch.float32).reshape(edge_index.shape[1], -1)
            if bool(custom_config.get("normalize_edge_attr", True)) and edge_attr_tensor.numel() > 0:
                denom = edge_attr_tensor.abs().amax(dim=0).clamp_min(1.0)
                edge_attr_tensor = edge_attr_tensor / denom

        edge_dim = None if edge_attr_tensor is None else int(edge_attr_tensor.shape[-1])

        self.obs_dim = obs_dim
        self.num_nodes = num_nodes
        self.num_outputs = int(num_outputs)
        self.action_dim = self._infer_tuple_action_dim(obs_space)
        self.use_edge_attr = edge_attr_tensor is not None and edge_attr_tensor.numel() > 0
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer(
            "edge_attr",
            torch.empty((0, edge_dim or 0), dtype=torch.float32) if edge_attr_tensor is None else edge_attr_tensor,
            persistent=False,
        )

        self.local_encoder = MLP(obs_dim, hidden_dim, latent_dim, nn.ELU())
        self.neighbor_input = nn.Linear(obs_dim, hidden_dim)
        self.neighbor_gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.neighbor_output = nn.Sequential(nn.ELU(), nn.Linear(hidden_dim, latent_dim))
        self.fusion = MLP(latent_dim + latent_dim, max(hidden_dim, fusion_dim), fusion_dim, nn.ELU())
        head_in_dim = fusion_dim + self.action_dim
        self.policy_head = nn.Sequential(nn.Linear(head_in_dim, actor_hidden), nn.ReLU(), nn.Linear(actor_hidden, num_outputs))
        self.value_head = nn.Sequential(nn.Linear(fusion_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))
        self._last_value: Optional[Tensor] = None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: list,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, list]:
        del seq_lens
        obs, action = self._split_obs(input_dict["obs"])
        z = self._encode(obs)
        head_input = z
        if self.action_dim > 0:
            head_input = torch.cat([z, self._action_to_vector(action, z.shape[0], z.device)], dim=-1)
        logits = self.policy_head(head_input)

        action_mask = obs.get("action_mask")
        if action_mask is not None and logits.shape[-1] == action_mask.shape[-1]:
            mask = action_mask.float().to(logits.device)
            logits = logits.masked_fill(mask <= 0.0, -1.0e9)

        self._last_value = self.value_head(z).squeeze(-1)
        return logits, state

    def value_function(self) -> TensorType:
        if self._last_value is None:
            return torch.zeros(1, device=self.edge_index.device)
        return self._last_value

    def _encode(self, obs: Dict[str, Tensor]) -> Tensor:
        graph_obs = obs["graph_obs"].float()
        local_obs = obs.get("local_obs")
        node_index = obs["node_index"].long().view(-1)

        if graph_obs.dim() == 2:
            graph_obs = graph_obs.unsqueeze(0)
        if local_obs is None:
            local_obs = self._gather_nodes(graph_obs, node_index)
        else:
            local_obs = local_obs.float()
            if local_obs.dim() == 1:
                local_obs = local_obs.unsqueeze(0)

        z_local = self.local_encoder(local_obs)
        z_neighbors = []
        edge_attr = self.edge_attr if self.use_edge_attr else None

        for graph in graph_obs:
            h = F.elu(self.neighbor_input(graph))
            if self.edge_index.numel() > 0:
                h = self.neighbor_gat(h, self.edge_index, edge_attr=edge_attr)
            z_neighbors.append(self.neighbor_output(h))

        neighbor_tensor = torch.stack(z_neighbors, dim=0)
        z_neighbor = self._gather_nodes(neighbor_tensor, node_index)
        return self.fusion(torch.cat([z_local, z_neighbor], dim=-1))

    @staticmethod
    def _gather_nodes(graph_tensor: Tensor, node_index: Tensor) -> Tensor:
        batch_index = torch.arange(graph_tensor.shape[0], device=graph_tensor.device)
        safe_index = node_index.to(graph_tensor.device).clamp(0, graph_tensor.shape[1] - 1)
        return graph_tensor[batch_index, safe_index]

    @staticmethod
    def _action_to_vector(self, action: Any, batch_size: int, device: torch.device) -> Tensor:
        if action is None:
            return torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=device)

        if isinstance(action, (list, tuple)):
            action = action[0]
        action_tensor = action.to(device) if isinstance(action, Tensor) else torch.as_tensor(action, device=device)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.view(1)

        if action_tensor.is_floating_point() and action_tensor.shape[-1] > 1:
            vector = action_tensor.float().view(batch_size, -1)
        else:
            indices = action_tensor.long().view(batch_size, -1)[:, 0].clamp(0, self.action_dim - 1)
            vector = F.one_hot(indices, num_classes=self.action_dim).float()

        if vector.shape[-1] < self.action_dim:
            pad = torch.zeros((batch_size, self.action_dim - vector.shape[-1]), dtype=vector.dtype, device=device)
            vector = torch.cat([vector, pad], dim=-1)
        return vector[:, : self.action_dim]

    @staticmethod
    def _split_obs(obs: Any) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        action = None
        if isinstance(obs, (list, tuple)):
            if len(obs) > 1:
                action = obs[-1]
            obs = obs[0]
        while isinstance(obs, (list, tuple)):
            obs = obs[0]
        if not isinstance(obs, dict):
            raise TypeError("BaselineV1TorchGeometricModel requires a Dict observation.")
        return obs, action

    @staticmethod
    def _base_obs_space(obs_space: Any) -> Any:
        spaces = getattr(obs_space, "spaces", None)
        if isinstance(spaces, tuple) and spaces:
            return spaces[0]
        return obs_space

    @staticmethod
    def _infer_obs_dim(obs_space: Any) -> int:
        spaces = getattr(obs_space, "spaces", {})
        if isinstance(spaces, dict) and "local_obs" in spaces:
            return int(spaces["local_obs"].shape[-1])
        return int(obs_space.shape[-1])

    @staticmethod
    def _infer_num_nodes(obs_space: Any) -> int:
        spaces = getattr(obs_space, "spaces", {})
        if isinstance(spaces, dict) and "graph_obs" in spaces:
            return int(spaces["graph_obs"].shape[0])
        return 1

    @staticmethod
    def _infer_tuple_action_dim(obs_space: Any) -> int:
        spaces = getattr(obs_space, "spaces", None)
        if not isinstance(spaces, tuple) or len(spaces) < 2:
            return 0
        action_space = spaces[-1]
        if hasattr(action_space, "n"):
            return int(action_space.n)
        if hasattr(action_space, "shape") and action_space.shape:
            return int(action_space.shape[-1])
        return 0
