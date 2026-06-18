"""Diffusion-convolutional recurrent Q-network for graph observations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


def _activation_layer(name: str) -> nn.Module:
    activation = str(name or "relu").lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported DCRNN activation: {name!r}.")


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    degree = matrix.sum(axis=1)
    inv_degree = np.zeros_like(degree, dtype=np.float32)
    np.divide(1.0, degree, out=inv_degree, where=degree > 0)
    return inv_degree[:, None] * matrix


def _scaled_laplacian(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.maximum(matrix, matrix.T)
    degree = matrix.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    np.divide(1.0, np.sqrt(degree), out=inv_sqrt, where=degree > 0)
    normalized = np.eye(matrix.shape[0], dtype=np.float32) - inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    try:
        lambda_max = float(np.max(np.real(np.linalg.eigvals(normalized))))
    except np.linalg.LinAlgError:
        lambda_max = 2.0
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        lambda_max = 2.0
    return (2.0 / lambda_max * normalized - np.eye(matrix.shape[0], dtype=np.float32)).astype(np.float32)


def diffusion_supports(adjacency: np.ndarray, filter_type: str) -> list[np.ndarray]:
    """Return dense diffusion supports without optional scipy/torch-geometric deps."""

    filter_type = str(filter_type or "dual_random_walk").lower()
    if filter_type == "laplacian":
        return [_scaled_laplacian(adjacency)]
    if filter_type == "random_walk":
        return [_row_normalize(adjacency).T.astype(np.float32)]
    if filter_type == "dual_random_walk":
        return [_row_normalize(adjacency).astype(np.float32), _row_normalize(adjacency.T).astype(np.float32)]
    return [_scaled_laplacian(adjacency)]


class DiffusionGraphConv(nn.Module):
    """Dense diffusion graph convolution used inside DCGRU gates."""

    def __init__(
        self,
        *,
        supports: Iterable[np.ndarray],
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        max_diffusion_step: int,
        output_dim: int,
        bias_start: float = 0.0,
    ) -> None:
        super().__init__()
        supports = list(supports)
        self.num_nodes = int(num_nodes)
        self.max_diffusion_step = int(max_diffusion_step)
        self.input_size = int(input_dim) + int(hidden_dim)
        self.num_matrices = len(supports) * self.max_diffusion_step + 1
        self.weight = nn.Parameter(torch.empty(self.input_size * self.num_matrices, int(output_dim)))
        self.bias = nn.Parameter(torch.empty(int(output_dim)))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.constant_(self.bias, bias_start)
        for index, support in enumerate(supports):
            self.register_buffer(f"support_{index}", torch.as_tensor(support, dtype=torch.float32))

    @property
    def supports(self) -> list[torch.Tensor]:
        return [value for name, value in self.named_buffers() if name.startswith("support_")]

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]  # (B, num_nodes * input_dim * history)
        inputs = inputs.reshape(batch_size, self.num_nodes, -1)
        state = state.reshape(batch_size, self.num_nodes, -1)
        x0 = torch.cat([inputs, state], dim=-1)
        weight = self.weight.reshape(self.input_size, self.num_matrices, -1)

        projected = torch.einsum("bni,io->bno", x0, weight[:, 0, :])
        weight_index = 1

        for support in self.supports:
            support = support.to(device=x0.device, dtype=x0.dtype)
            x_k = x0
            for _ in range(self.max_diffusion_step):
                x_k = torch.einsum("nm,bmi->bni", support, x_k)
                projected = projected + torch.einsum("bni,io->bno", x_k, weight[:, weight_index, :])
                weight_index += 1

        projected = projected + self.bias
        return projected.reshape(batch_size, -1)


class DCGRUCell(nn.Module):
    """Graph-convolutional GRU cell."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        adjacency: np.ndarray,
        max_diffusion_step: int,
        num_nodes: int,
        filter_type: str = "dual_random_walk",
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        supports = diffusion_supports(adjacency, filter_type)
        self.gate = DiffusionGraphConv(
            supports=supports,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=2 * hidden_dim,
        )
        self.candidate = DiffusionGraphConv(
            supports=supports,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=hidden_dim,
        )

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate(inputs, state)).reshape(-1, self.num_nodes, 2 * self.hidden_dim)
        reset, update = torch.split(gates, self.hidden_dim, dim=-1)
        reset = reset.reshape(-1, self.num_nodes * self.hidden_dim)
        update = update.reshape(-1, self.num_nodes * self.hidden_dim)
        candidate = torch.tanh(self.candidate(inputs, reset * state))
        new_state = update * state + (1.0 - update) * candidate
        return new_state, new_state


class DCRNNEncoder(nn.Module):
    """Stacked DCGRU encoder over graph-history observations."""

    def __init__(
        self,
        *,
        input_dim: int,
        adjacency: np.ndarray,
        max_diffusion_step: int,
        hidden_dim: int,
        num_nodes: int,
        num_rnn_layers: int,
        filter_type: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_rnn_layers = max(1, int(num_rnn_layers))
        cells = [
            DCGRUCell(
                input_dim=int(input_dim),
                hidden_dim=self.hidden_dim,
                adjacency=adjacency,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                filter_type=filter_type,
            )
        ]
        for _ in range(1, self.num_rnn_layers):
            cells.append(
                DCGRUCell(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    adjacency=adjacency,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    filter_type=filter_type,
                )
            )
        self.cells = nn.ModuleList(cells)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = inputs.shape[:2]
        current_inputs = inputs.reshape(seq_len, batch_size, -1)
        for cell in self.cells:
            state = torch.zeros(
                batch_size,
                self.num_nodes * self.hidden_dim,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            outputs = []
            for step in range(seq_len):
                output, state = cell(current_inputs[step], state)
                outputs.append(output)
            current_inputs = torch.stack(outputs, dim=0)
        return current_inputs[-1].reshape(batch_size, self.num_nodes, self.hidden_dim)


class DCRNNBackbone(nn.Module):
    """Shared DCRNN encoder plus per-agent feature fusion."""

    def __init__(
        self,
        *,
        input_dim: int,
        adjacency: np.ndarray,
        num_nodes: int,
        agent_index: int | None,
        hidden_dim: int = 128,
        max_diffusion_step: int = 2,
        num_rnn_layers: int = 1,
        filter_type: str = "dual_random_walk",
        pre_encoder_enabled: bool = False,
        pre_encoder_hidden_dim: int | None = None,
        pre_encoder_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.agent_index = int(agent_index) if agent_index is not None else None
        self.pre_encoder_enabled = bool(pre_encoder_enabled)
        self.pre_encoder_input_dim = self.input_dim
        self.pre_encoder_output_dim = int(pre_encoder_hidden_dim or self.hidden_dim) if self.pre_encoder_enabled else self.input_dim
        self.pre_encoder_activation = str(pre_encoder_activation or "relu")
        self.pre_encoder = None
        if self.pre_encoder_enabled:
            self.pre_encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.pre_encoder_output_dim),
                _activation_layer(self.pre_encoder_activation),
            )
        self.encoder = DCRNNEncoder(
            input_dim=self.pre_encoder_output_dim,
            adjacency=adjacency,
            max_diffusion_step=max_diffusion_step,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            filter_type=filter_type,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim + self.pre_encoder_output_dim

    def _encode_observations(self, obs: torch.Tensor) -> torch.Tensor:
        if self.pre_encoder is None:
            return obs
        batch_size, history_len, num_nodes, _ = obs.shape
        encoded = self.pre_encoder(obs.reshape(batch_size * history_len * num_nodes, self.input_dim))
        return encoded.reshape(batch_size, history_len, num_nodes, self.pre_encoder_output_dim)

    def encode_graph(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.float()
        if obs.ndim != 4:
            raise ValueError(f"DCRNN expects observations with shape [B, H, N, F], got {tuple(obs.shape)}.")
        encoded_obs = self._encode_observations(obs)
        encoded = self.encoder(encoded_obs.transpose(0, 1))
        latest_features = encoded_obs[:, -1]
        return encoded, latest_features

    def select_agent_latent(
        self,
        encoded: torch.Tensor,
        latest_features: torch.Tensor,
        *,
        agent_index: int | None = None,
    ) -> torch.Tensor:
        resolved_agent_index = self.agent_index if agent_index is None else int(agent_index)
        if resolved_agent_index is None:
            raise ValueError("DCRNNBackbone requires an agent_index when selecting an agent latent.")
        agent_hidden = encoded[:, resolved_agent_index, :]
        agent_features = latest_features[:, resolved_agent_index, :]
        return torch.cat([agent_hidden, agent_features], dim=-1)

    def forward_for_agent(self, obs: torch.Tensor, *, agent_index: int) -> torch.Tensor:
        encoded, latest_features = self.encode_graph(obs)
        return self.select_agent_latent(encoded, latest_features, agent_index=agent_index)

    @staticmethod
    def _resolve_pre_encoder_kwargs(
        config: dict[str, Any],
        *,
        hidden_dim: int,
        fallback_enabled: bool = False,
    ) -> dict[str, Any]:
        pre_encoder_config = dict(config.get("pre_encoder", {}) or {})
        enabled = bool(pre_encoder_config.get("enabled", fallback_enabled))
        return {
            "pre_encoder_enabled": enabled,
            "pre_encoder_hidden_dim": int(pre_encoder_config.get("hidden_dim", hidden_dim)) if enabled else None,
            "pre_encoder_activation": str(pre_encoder_config.get("activation", "relu") or "relu"),
        }

    @classmethod
    def from_model_config(cls, observation_space: Any, model_config: dict[str, Any]) -> "DCRNNBackbone":
        history_len, num_nodes, input_dim = observation_space.shape
        del history_len
        adjacency = np.asarray(model_config["adjacency"], dtype=np.float32)
        hidden_dim = int(model_config.get("hid_dim", model_config.get("hidden_dim", 128)))
        return cls(
            input_dim=int(model_config.get("input_dim", input_dim)),
            adjacency=adjacency,
            num_nodes=int(model_config.get("num_nodes", num_nodes)),
            agent_index=int(model_config["agent_index"]),
            hidden_dim=hidden_dim,
            max_diffusion_step=int(model_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(model_config.get("num_rnn_layers", 1)),
            filter_type=str(model_config.get("filter_type", "dual_random_walk")),
            **cls._resolve_pre_encoder_kwargs(model_config, hidden_dim=hidden_dim),
        )

    @classmethod
    def from_shared_ppo_model_config(
        cls,
        observation_space: Any,
        model_config: dict[str, Any],
    ) -> "DCRNNBackbone":
        history_len, num_nodes, input_dim = observation_space.shape
        del history_len
        adjacency = np.asarray(model_config["adjacency"], dtype=np.float32)
        hidden_dim = int(model_config.get("hid_dim", model_config.get("hidden_dim", 128)))
        return cls(
            input_dim=int(model_config.get("input_dim", input_dim)),
            adjacency=adjacency,
            num_nodes=int(model_config.get("num_nodes", num_nodes)),
            agent_index=None,
            hidden_dim=hidden_dim,
            max_diffusion_step=int(model_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(model_config.get("num_rnn_layers", 1)),
            filter_type=str(model_config.get("filter_type", "dual_random_walk")),
            **cls._resolve_pre_encoder_kwargs(model_config, hidden_dim=hidden_dim),
        )

    @classmethod
    def _from_custom_sac_encoder_config(
        cls,
        observation_space: Any,
        model_config: dict[str, Any],
        *,
        branch: str,
    ) -> "DCRNNBackbone":
        custom_sac = dict(model_config.get("custom_sac", {}) or {}) if isinstance(model_config.get("custom_sac"), dict) else {}
        branch_config = dict(custom_sac.get(branch, {}) or {})
        encoder_config = dict(branch_config.get("encoder", {}) or {})
        history_len, num_nodes, input_dim = observation_space.shape
        del history_len
        adjacency = np.asarray(model_config["adjacency"], dtype=np.float32)
        hidden_dim = int(encoder_config.get("hidden_dim", encoder_config.get("hid_dim", 128)))
        return cls(
            input_dim=int(model_config.get("input_dim", input_dim)),
            adjacency=adjacency,
            num_nodes=int(model_config.get("num_nodes", num_nodes)),
            agent_index=int(model_config["agent_index"]),
            hidden_dim=hidden_dim,
            max_diffusion_step=int(encoder_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(encoder_config.get("num_rnn_layers", 1)),
            filter_type=str(encoder_config.get("filter_type", "dual_random_walk")),
            **cls._resolve_pre_encoder_kwargs(
                encoder_config,
                hidden_dim=hidden_dim,
                fallback_enabled=False,
            ),
        )

    @classmethod
    def from_actor_model_config(cls, observation_space: Any, model_config: dict[str, Any]) -> "DCRNNBackbone":
        return cls._from_custom_sac_encoder_config(observation_space, model_config, branch="actor")

    @classmethod
    def from_critic_model_config(cls, observation_space: Any, model_config: dict[str, Any]) -> "DCRNNBackbone":
        return cls._from_custom_sac_encoder_config(observation_space, model_config, branch="critic")

    @classmethod
    def from_shared_sac_model_config(cls, observation_space: Any, model_config: dict[str, Any]) -> "DCRNNBackbone":
        custom_sac = dict(model_config.get("custom_sac", {}) or {}) if isinstance(model_config.get("custom_sac"), dict) else {}
        encoder_config = dict(custom_sac.get("shared_encoder", {}) or {})
        history_len, num_nodes, input_dim = observation_space.shape
        del history_len
        adjacency = np.asarray(model_config["adjacency"], dtype=np.float32)
        hidden_dim = int(encoder_config.get("hidden_dim", encoder_config.get("hid_dim", 128)))
        return cls(
            input_dim=int(model_config.get("input_dim", input_dim)),
            adjacency=adjacency,
            num_nodes=int(model_config.get("num_nodes", num_nodes)),
            agent_index=int(model_config["agent_index"]),
            hidden_dim=hidden_dim,
            max_diffusion_step=int(encoder_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(encoder_config.get("num_rnn_layers", 1)),
            filter_type=str(encoder_config.get("filter_type", "dual_random_walk")),
            **cls._resolve_pre_encoder_kwargs(
                encoder_config,
                hidden_dim=hidden_dim,
                fallback_enabled=False,
            ),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded, latest_features = self.encode_graph(obs)
        return self.select_agent_latent(encoded, latest_features)


class DCRNNQNetwork(nn.Module):
    """DCRNN encoder plus per-agent Q head for discrete traffic-light actions."""

    def __init__(
        self,
        *,
        input_dim: int,
        adjacency: np.ndarray,
        num_nodes: int,
        agent_index: int,
        num_actions: int,
        hidden_dim: int = 128,
        max_diffusion_step: int = 2,
        num_rnn_layers: int = 1,
        filter_type: str = "dual_random_walk",
        head_hidden_dim: int | None = None,
        pre_encoder_enabled: bool = False,
        pre_encoder_hidden_dim: int | None = None,
        pre_encoder_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.backbone = DCRNNBackbone(
            input_dim=input_dim,
            adjacency=adjacency,
            num_nodes=num_nodes,
            agent_index=agent_index,
            hidden_dim=hidden_dim,
            max_diffusion_step=max_diffusion_step,
            num_rnn_layers=num_rnn_layers,
            filter_type=filter_type,
            pre_encoder_enabled=pre_encoder_enabled,
            pre_encoder_hidden_dim=pre_encoder_hidden_dim,
            pre_encoder_activation=pre_encoder_activation,
        )
        head_hidden = int(head_hidden_dim or hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, int(num_actions)),
        )

    @classmethod
    def from_model_config(cls, observation_space: Any, action_space: Any, model_config: dict[str, Any]) -> "DCRNNQNetwork":
        backbone = DCRNNBackbone.from_model_config(observation_space, model_config)
        return cls(
            input_dim=backbone.input_dim,
            adjacency=np.asarray(model_config["adjacency"], dtype=np.float32),
            num_nodes=backbone.num_nodes,
            agent_index=backbone.agent_index,
            num_actions=int(action_space.n),
            hidden_dim=backbone.hidden_dim,
            max_diffusion_step=int(model_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(model_config.get("num_rnn_layers", 1)),
            filter_type=str(model_config.get("filter_type", "dual_random_walk")),
            head_hidden_dim=model_config.get("head_hidden_dim"),
            pre_encoder_enabled=backbone.pre_encoder_enabled,
            pre_encoder_hidden_dim=backbone.pre_encoder_output_dim if backbone.pre_encoder_enabled else None,
            pre_encoder_activation=backbone.pre_encoder_activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))
