"""Diffusion-convolutional recurrent Q-network for graph observations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


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
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.num_nodes, -1)
        state = state.reshape(batch_size, self.num_nodes, -1)
        x = torch.cat([inputs, state], dim=-1)
        x = x.permute(1, 2, 0).reshape(self.num_nodes, -1)

        diffusion_terms = [x]
        for support in self.supports:
            x_k = x
            for _ in range(self.max_diffusion_step):
                x_k = torch.matmul(support.to(device=x.device, dtype=x.dtype), x_k)
                diffusion_terms.append(x_k)

        x = torch.stack(diffusion_terms, dim=0)
        x = x.reshape(self.num_matrices, self.num_nodes, self.input_size, batch_size)
        x = x.permute(3, 1, 2, 0).reshape(batch_size * self.num_nodes, self.input_size * self.num_matrices)
        x = torch.matmul(x, self.weight) + self.bias
        return x.reshape(batch_size, -1)


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
    ) -> None:
        super().__init__()
        self.agent_index = int(agent_index)
        self.encoder = DCRNNEncoder(
            input_dim=input_dim,
            adjacency=adjacency,
            max_diffusion_step=max_diffusion_step,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            filter_type=filter_type,
        )
        head_hidden = int(head_hidden_dim or hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(input_dim), head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, int(num_actions)),
        )

    @classmethod
    def from_model_config(cls, observation_space: Any, action_space: Any, model_config: dict[str, Any]) -> "DCRNNQNetwork":
        history_len, num_nodes, input_dim = observation_space.shape
        del history_len
        adjacency = np.asarray(model_config["adjacency"], dtype=np.float32)
        return cls(
            input_dim=int(model_config.get("input_dim", input_dim)),
            adjacency=adjacency,
            num_nodes=int(model_config.get("num_nodes", num_nodes)),
            agent_index=int(model_config["agent_index"]),
            num_actions=int(action_space.n),
            hidden_dim=int(model_config.get("hid_dim", model_config.get("hidden_dim", 128))),
            max_diffusion_step=int(model_config.get("max_diffusion_step", 2)),
            num_rnn_layers=int(model_config.get("num_rnn_layers", 1)),
            filter_type=str(model_config.get("filter_type", "dual_random_walk")),
            head_hidden_dim=model_config.get("head_hidden_dim"),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if obs.ndim != 4:
            raise ValueError(f"DCRNN expects observations with shape [B, H, N, F], got {tuple(obs.shape)}.")
        encoded = self.encoder(obs.transpose(0, 1))
        latest_features = obs[:, -1]
        agent_hidden = encoded[:, self.agent_index, :]
        agent_features = latest_features[:, self.agent_index, :]
        return self.head(torch.cat([agent_hidden, agent_features], dim=-1))
