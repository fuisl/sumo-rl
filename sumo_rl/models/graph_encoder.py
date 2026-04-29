"""Graph encoder used by the local-neighbor SAC baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    """Small GAT-style aggregation layer implemented in plain PyTorch."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, edge_dim: int | None = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}.")
        self.heads = int(heads)
        self.head_dim = int(out_dim // heads)
        self.edge_dim = edge_dim
        self.dropout = float(dropout)

        self.src_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.dst_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim is not None else None

        self.attn_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_edge = nn.Parameter(torch.empty(self.heads, self.head_dim)) if edge_dim is not None else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.src_proj.weight)
        nn.init.xavier_uniform_(self.dst_proj.weight)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.attn_edge is not None:
            nn.init.xavier_uniform_(self.attn_edge)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        num_nodes = int(x.shape[0])
        if edge_index.numel() == 0:
            return self.src_proj(x) + self.bias

        src = self.src_proj(x).view(num_nodes, self.heads, self.head_dim)
        dst = self.dst_proj(x).view(num_nodes, self.heads, self.head_dim)
        src_nodes = edge_index[0].long()
        dst_nodes = edge_index[1].long()

        edge_features = None
        if self.edge_proj is not None and edge_attr is not None:
            edge_features = self.edge_proj(edge_attr).view(-1, self.heads, self.head_dim)

        src_edge = src[src_nodes]
        dst_edge = dst[dst_nodes]

        scores = (src_edge * self.attn_src.unsqueeze(0)).sum(dim=-1)
        scores = scores + (dst_edge * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        if edge_features is not None and self.attn_edge is not None:
            scores = scores + (edge_features * self.attn_edge.unsqueeze(0)).sum(dim=-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        messages = src_edge if edge_features is None else src_edge + edge_features
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device, dtype=x.dtype)

        for node_idx in range(num_nodes):
            mask = dst_nodes == node_idx
            if not torch.any(mask):
                continue
            node_scores = scores[mask]
            node_messages = messages[mask]
            weights = torch.softmax(node_scores, dim=0)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            out[node_idx] = (weights.unsqueeze(-1) * node_messages).sum(dim=0)

        return out.reshape(num_nodes, -1) + self.bias


class GraphEncoder(nn.Module):
    """Two-stage graph encoder with a local projection and attention aggregation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 2,
        edge_dim: int | None = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gat = GraphAttentionLayer(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ELU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        h = self.act(self.input_proj(x))
        h = self.act(self.gat(h, edge_index, edge_attr))
        return self.output_proj(h)
