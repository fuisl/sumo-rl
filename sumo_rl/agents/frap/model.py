"""FRAP phase-competition Q-network."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def infer_default_phase_pairs(num_movements: int, num_actions: int) -> list[list[int]]:
    """Infer common FRAP phase-pair layouts from SUMO-RL observation/action sizes."""

    if num_movements == 12 and num_actions == 8:
        return [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]]
    if num_movements == 8 and num_actions == 8:
        return [[0, 4], [1, 5], [0, 1], [4, 5], [2, 6], [3, 7], [6, 7], [2, 3]]
    if num_movements == 8 and num_actions == 4:
        return [[1, 5], [0, 4], [3, 7], [2, 6]]
    if num_movements == 4 and num_actions == 2:
        return [[0, 2], [1, 3]]

    if num_actions <= max(1, num_movements // 2):
        return [[(2 * index) % num_movements, (2 * index + 1) % num_movements] for index in range(num_actions)]
    return [[index % num_movements, (index + 1) % num_movements] for index in range(num_actions)]


def build_competition_mask(phase_pairs: Sequence[Sequence[int]]) -> torch.Tensor:
    """Return the FRAP conflict mask used for pairwise phase competition."""

    mask = []
    for index, pair_a in enumerate(phase_pairs):
        row = []
        for other_index, pair_b in enumerate(phase_pairs):
            if index == other_index:
                continue
            row.append(1 if len(set(pair_a).union(pair_b)) == 3 else 0)
        mask.append(row)
    return torch.tensor(mask, dtype=torch.long)


def normalize_phase_pairs(
    phase_pairs: Optional[Iterable[Iterable[int]]],
    *,
    num_movements: int,
    num_actions: int,
) -> list[list[int]]:
    pairs = [list(map(int, pair)) for pair in phase_pairs or infer_default_phase_pairs(num_movements, num_actions)]
    if len(pairs) != num_actions:
        raise ValueError(f"FRAP requires one phase pair per action; got {len(pairs)} pairs for {num_actions} actions.")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Each FRAP phase pair must contain exactly two movement indices; got {pair!r}.")
        for movement_index in pair:
            if movement_index < 0 or movement_index >= num_movements:
                raise ValueError(
                    f"FRAP phase pair index {movement_index} is outside the movement range [0, {num_movements})."
                )
    return pairs


def infer_num_movements(
    observation_dim: int,
    num_actions: int,
    *,
    demand_shape: int,
    observation_has_phase: bool = True,
    observation_has_min_green: bool = True,
) -> int:
    demand_start = (num_actions if observation_has_phase else 0) + (1 if observation_has_min_green else 0)
    demand_width = observation_dim - demand_start
    if demand_width <= 0 or demand_width % demand_shape != 0:
        raise ValueError(
            "FRAP could not infer movement demand features from the observation space. "
            f"observation_dim={observation_dim}, num_actions={num_actions}, demand_shape={demand_shape}."
        )
    return demand_width // demand_shape


class FRAPQNetwork(nn.Module):
    """Q-network from Learning Phase Competition, adapted to SUMO-RL observations."""

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
        conv_units: int = 20,
    ) -> None:
        super().__init__()
        if num_actions < 2:
            raise ValueError("FRAP phase competition requires at least two actions.")
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
                "FRAP expects the observation tail to be movement demand features. "
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
        self.before_merge = nn.Conv2d(conv_units, 1, kernel_size=(1, 1))
        self.register_buffer("competition_mask", build_competition_mask(self.phase_pairs), persistent=False)

    @classmethod
    def from_model_config(cls, observation_space: Any, action_space: Any, model_config: Dict[str, Any]) -> "FRAPQNetwork":
        observation_dim = int(observation_space.shape[0])
        num_actions = int(action_space.n)
        return cls(
            observation_dim=observation_dim,
            num_actions=num_actions,
            phase_pairs=model_config.get("phase_pairs"),
            demand_shape=int(model_config.get("demand_shape", 2)),
            observation_has_phase=bool(model_config.get("observation_has_phase", True)),
            observation_has_min_green=bool(model_config.get("observation_has_min_green", True)),
            demand_start=model_config.get("demand_start"),
            demand_layout=str(model_config.get("demand_layout", "split")),
            d_out=int(model_config.get("d_out", 4)),
            p_out=int(model_config.get("p_out", 4)),
            lane_embed_units=int(model_config.get("lane_embed_units", 16)),
            relation_embed_size=int(model_config.get("relation_embed_size", 4)),
            conv_units=int(model_config.get("conv_units", 20)),
        )

    def _current_phase_movements(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.observation_has_phase:
            return torch.zeros((obs.shape[0], self.num_movements), dtype=torch.long, device=obs.device)

        phase_one_hot = obs[:, : self.num_actions]
        phase_indices = torch.argmax(phase_one_hot, dim=-1)
        phase_movements = torch.zeros((obs.shape[0], self.num_movements), dtype=torch.long, device=obs.device)
        for action_index, pair in enumerate(self.phase_pairs):
            selected = phase_indices == action_index
            if torch.any(selected):
                phase_movements[selected, pair[0]] = 1
                phase_movements[selected, pair[1]] = 1
        return phase_movements

    def _movement_demands(self, obs: torch.Tensor) -> torch.Tensor:
        demand = obs[:, self.demand_start : self.demand_start + self.num_movements * self.demand_shape]
        demand = demand.float()
        if self.demand_layout == "split":
            return demand.reshape(obs.shape[0], self.demand_shape, self.num_movements).transpose(1, 2)
        if self.demand_layout == "interleaved":
            return demand.reshape(obs.shape[0], self.num_movements, self.demand_shape)
        raise ValueError(f"Unsupported FRAP demand_layout: {self.demand_layout!r}.")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        batch_size = obs.shape[0]
        phase_movements = self._current_phase_movements(obs)
        phase_embeds = torch.sigmoid(self.phase_embedding(phase_movements))
        demand_embeds = torch.sigmoid(self.demand_layer(self._movement_demands(obs)))
        movement_embeds = F.relu(self.lane_embedding(torch.cat((phase_embeds, demand_embeds), dim=-1)))

        phase_pair_embeds = []
        for pair in self.phase_pairs:
            phase_pair_embeds.append(movement_embeds[:, pair[0]] + movement_embeds[:, pair[1]])

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

        combined = phase_features * relation_features
        combined = F.relu(self.hidden_layer(combined))
        combined = self.before_merge(combined)
        combined = combined.reshape(batch_size, self.num_actions, self.num_actions - 1)
        return torch.sum(combined, dim=2)
