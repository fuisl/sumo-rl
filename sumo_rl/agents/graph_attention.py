"""Shared PyTorch Geometric attention layers for traffic-signal graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv, MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
except ImportError as exc:  # pragma: no cover - exercised only without optional extras.
    raise ImportError("Graph attention modules require torch-geometric. Install the rllib-custom extra.") from exc


class CoLightPyGAttentionLayer(MessagePassing):
    """LibSignal-style CoLight multi-head attention implemented with PyG."""

    def __init__(self, input_dim: int, head_dim: int = 16, output_dim: int = 128, num_heads: int = 5) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.input_dim = int(input_dim)
        self.head_dim = int(head_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1:
            raise ValueError("CoLight PyG attention requires at least one attention head.")

        projected_dim = self.head_dim * self.num_heads
        self.target_projection = nn.Linear(self.input_dim, projected_dim)
        self.source_projection = nn.Linear(self.input_dim, projected_dim)
        self.message_projection = nn.Linear(self.input_dim, projected_dim)
        self.output_projection = nn.Linear(self.head_dim, self.output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=int(x.shape[0]))
        aggregated = self.propagate(edge_index=edge_index, x=x)
        return F.relu(self.output_projection(aggregated))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor, ptr=None, size_i=None) -> torch.Tensor:
        target = F.relu(self.target_projection(x_i)).view(-1, self.num_heads, self.head_dim)
        source = F.relu(self.source_projection(x_j)).view(-1, self.num_heads, self.head_dim)
        scores = torch.sum(target * source, dim=-1)
        alpha = softmax(scores, index=index, ptr=ptr, num_nodes=size_i)

        messages = F.relu(self.message_projection(x_j)).view(-1, self.num_heads, self.head_dim)
        return torch.mean(messages * alpha.unsqueeze(-1), dim=1)


class CoLightGATLayer(CoLightPyGAttentionLayer):
    """Backward-compatible public name for the PyG-backed CoLight layer."""


class CoLightGATv2Layer(nn.Module):
    """PyG GATv2Conv adapter with the same public shape as the CoLight GAT layer."""

    def __init__(self, input_dim: int, head_dim: int = 16, output_dim: int = 128, num_heads: int = 5) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.head_dim = int(head_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1:
            raise ValueError("CoLight GATv2 attention requires at least one attention head.")

        self.gatv2 = GATv2Conv(
            in_channels=self.input_dim,
            out_channels=self.head_dim,
            heads=self.num_heads,
            concat=False,
            add_self_loops=True,
        )
        self.output_projection = nn.Linear(self.head_dim, self.output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        aggregated = F.relu(self.gatv2(x, edge_index))
        return F.relu(self.output_projection(aggregated))
