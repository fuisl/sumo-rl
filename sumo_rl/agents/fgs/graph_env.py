"""Graph-observation wrapper for FGS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from sumo_rl.agents.frap.model import infer_default_phase_pairs
from sumo_rl.agents.fgs.topology import (
    TLSTopology,
    bidirectional_message_edges,
    extract_tls_topology,
    render_fgs_topology,
)


def _base_sumo_env(env: Any) -> Any:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "traffic_signals") and hasattr(current, "ts_ids"):
            return current
        for attr in ("par_env", "aec_env", "env", "base_env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    return env


class FGSGraphParallelEnv:
    """PettingZoo parallel wrapper exposing full-graph observations per TLS."""

    metadata = {"name": "sumo_rl_fgs_graph_v0", "is_parallelizable": True}

    def __init__(
        self,
        env: Any,
        *,
        net_file: Optional[str] = None,
        topology_source: str = "tls_super_edges",
        render_topology_dir: Optional[Path] = None,
    ) -> None:
        self.env = env
        self.net_file = str(net_file or "")
        self.topology_source = str(topology_source or "tls_super_edges")
        self.possible_agents = [str(agent_id) for agent_id in getattr(env, "possible_agents", getattr(env, "agents", []))]
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self._agent_to_index = {agent_id: index for index, agent_id in enumerate(self.possible_agents)}
        self._latest_local_obs: Dict[str, np.ndarray] = {}
        self._prev_joint_action = np.zeros((max(1, len(self.possible_agents)), 1), dtype=np.float32)
        self._topology: Optional[TLSTopology] = None
        if self.topology_source == "tls_super_edges" and self.net_file:
            self._topology = extract_tls_topology(self.net_file)
            if render_topology_dir is not None:
                render_fgs_topology(self._topology, Path(render_topology_dir))
        self._refresh_spaces()

    def _refresh_spaces(self) -> None:
        self._num_nodes = max(1, len(self.possible_agents))
        self._raw_obs_dims = {
            agent_id: int(self.env.observation_space(agent_id).shape[0])
            for agent_id in self.possible_agents
        }
        self._action_sizes = {
            agent_id: int(self.env.action_space(agent_id).n)
            for agent_id in self.possible_agents
        }
        self._num_actions = max(self._action_sizes.values() or [1])
        self._demand_widths = {}
        for agent_id in self.possible_agents:
            demand_width = self._raw_obs_dims[agent_id] - self._action_sizes[agent_id] - 1
            if demand_width <= 0:
                raise ValueError(
                    "FGS expects default SUMO-RL observations shaped as "
                    "[phase_one_hot, min_green, density, queue]."
                )
            self._demand_widths[agent_id] = demand_width
        self._max_demand_width = max(self._demand_widths.values() or [1])
        if self._max_demand_width % 2 != 0:
            raise ValueError("FGS expects density and queue demand features with equal lane counts.")
        self._num_movements = self._max_demand_width // 2
        self._node_feature_dim = self._num_actions + 1 + self._max_demand_width
        self._prev_joint_action = self._resize_joint_action_context(self._prev_joint_action)
        self._edges, self._edge_weights = self._build_edges()
        self._max_edges = max(1, len(self._edges))
        self._phase_pair_mask = self._build_phase_pair_mask()
        self._phase_competition_mask = self._build_phase_competition_mask(self._phase_pair_mask)

        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "node_features": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self._num_nodes, self._node_feature_dim),
                        dtype=np.float32,
                    ),
                    "edge_index": spaces.Box(
                        low=0,
                        high=max(0, self._num_nodes - 1),
                        shape=(2, self._max_edges),
                        dtype=np.int64,
                    ),
                    "edge_mask": spaces.Box(low=0.0, high=1.0, shape=(self._max_edges,), dtype=np.float32),
                    "edge_weight": spaces.Box(low=0.0, high=np.inf, shape=(self._max_edges,), dtype=np.float32),
                    "ego_index": spaces.Box(low=0, high=max(0, self._num_nodes - 1), shape=(), dtype=np.int64),
                    "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self._num_actions,), dtype=np.float32),
                    "node_action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._num_nodes, self._num_actions),
                        dtype=np.float32,
                    ),
                    "phase_pair_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._num_nodes, self._num_actions, self._num_movements),
                        dtype=np.float32,
                    ),
                    "phase_competition_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._num_nodes, self._num_actions, max(1, self._num_actions - 1)),
                        dtype=np.float32,
                    ),
                    "prev_joint_action": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._num_nodes, self._num_actions),
                        dtype=np.float32,
                    ),
                }
            )
            for agent_id in self.possible_agents
        }
        shared_action_space = spaces.Discrete(self._num_actions)
        self.action_spaces = {agent_id: shared_action_space for agent_id in self.possible_agents}

    def _build_phase_pair_mask(self) -> np.ndarray:
        mask = np.zeros((self._num_nodes, self._num_actions, self._num_movements), dtype=np.float32)
        base_env = _base_sumo_env(self.env)
        traffic_signals = getattr(base_env, "traffic_signals", {})
        for agent_id, node_index in self._agent_to_index.items():
            traffic_signal = traffic_signals.get(agent_id)
            lanes = list(getattr(traffic_signal, "lanes", []) or [])
            phase_lanes = list(getattr(traffic_signal, "phase_lanes", []) or [])
            lane_to_index = {lane: index for index, lane in enumerate(lanes)}
            if phase_lanes and lane_to_index:
                for action_index, active_lanes in enumerate(phase_lanes[: self._action_sizes[agent_id]]):
                    for lane in active_lanes:
                        movement_index = lane_to_index.get(lane)
                        if movement_index is not None and movement_index < self._num_movements:
                            mask[node_index, action_index, movement_index] = 1.0
            else:
                pairs = infer_default_phase_pairs(self._num_movements, self._action_sizes[agent_id])
                for action_index, pair in enumerate(pairs[: self._action_sizes[agent_id]]):
                    for movement_index in pair:
                        if movement_index < self._num_movements:
                            mask[node_index, action_index, movement_index] = 1.0
        return mask

    def _build_phase_competition_mask(self, phase_pair_mask: np.ndarray) -> np.ndarray:
        relation_width = max(1, self._num_actions - 1)
        relation = np.zeros((self._num_nodes, self._num_actions, relation_width), dtype=np.float32)
        if self._num_actions <= 1:
            return relation
        for node_index, agent_id in enumerate(self.possible_agents):
            action_size = self._action_sizes[agent_id]
            for action_index in range(self._num_actions):
                offset = 0
                action_movements = phase_pair_mask[node_index, action_index] > 0
                for other_index in range(self._num_actions):
                    if action_index == other_index:
                        continue
                    if offset >= relation_width:
                        break
                    other_movements = phase_pair_mask[node_index, other_index] > 0
                    if action_index < action_size and other_index < action_size:
                        shared = np.logical_and(action_movements, other_movements)
                        same = np.array_equal(action_movements, other_movements)
                        relation[node_index, action_index, offset] = float(np.any(shared) and not same)
                    offset += 1
        return relation

    def _resize_joint_action_context(self, context: np.ndarray) -> np.ndarray:
        resized = np.zeros((self._num_nodes, self._num_actions), dtype=np.float32)
        if context is None:
            return resized
        context = np.asarray(context, dtype=np.float32)
        rows = min(context.shape[0], resized.shape[0]) if context.ndim == 2 else 0
        cols = min(context.shape[1], resized.shape[1]) if context.ndim == 2 else 0
        if rows > 0 and cols > 0:
            resized[:rows, :cols] = context[:rows, :cols]
        return resized

    def _joint_action_one_hot(self, actions: Dict[str, int]) -> np.ndarray:
        context = np.zeros((self._num_nodes, self._num_actions), dtype=np.float32)
        for agent_id, node_index in self._agent_to_index.items():
            action = int(actions.get(agent_id, 0))
            action = int(np.clip(action, 0, self._action_sizes[agent_id] - 1))
            context[node_index, action] = 1.0
        return context

    def _canonical_local_obs(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        action_size = self._action_sizes[agent_id]
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        canonical = np.zeros(self._node_feature_dim, dtype=np.float32)
        phase_width = min(action_size, raw.shape[0], self._num_actions)
        canonical[:phase_width] = raw[:phase_width]
        min_green_index = action_size
        if raw.shape[0] > min_green_index:
            canonical[self._num_actions] = raw[min_green_index]
        demand = raw[action_size + 1 :]
        demand_width = min(demand.shape[0], self._max_demand_width)
        canonical[self._num_actions + 1 : self._num_actions + 1 + demand_width] = demand[:demand_width]
        return canonical

    def _build_edges(self) -> tuple[list[tuple[int, int]], list[float]]:
        if self._topology is not None:
            edges = bidirectional_message_edges(self._topology, self.possible_agents)
            indexed_edges = []
            weights = []
            for source_id, target_id in edges:
                source_index = self._agent_to_index[source_id]
                target_index = self._agent_to_index[target_id]
                key = (source_id, target_id) if source_id <= target_id else (target_id, source_id)
                indexed_edges.append((source_index, target_index))
                weights.append(float(self._topology.edge_weights.get(key, 1.0)))
            return indexed_edges, weights

        base_env = _base_sumo_env(self.env)
        traffic_signals = getattr(base_env, "traffic_signals", {})
        edges = set()
        for source_id in self.possible_agents:
            source_signal = traffic_signals.get(source_id)
            if source_signal is None:
                continue
            source_out_lanes = set(getattr(source_signal, "out_lanes", []) or [])
            for target_id in self.possible_agents:
                if source_id == target_id:
                    continue
                target_signal = traffic_signals.get(target_id)
                if target_signal is None:
                    continue
                if source_out_lanes.intersection(set(getattr(target_signal, "lanes", []) or [])):
                    edges.add((self._agent_to_index[source_id], self._agent_to_index[target_id]))
                    edges.add((self._agent_to_index[target_id], self._agent_to_index[source_id]))
        indexed = sorted(edges)
        return indexed, [1.0] * len(indexed)

    def _graph_obs(self, agent_id: str) -> Dict[str, np.ndarray]:
        node_features = np.zeros((self._num_nodes, self._node_feature_dim), dtype=np.float32)
        for node_id, node_index in self._agent_to_index.items():
            if node_id in self._latest_local_obs:
                node_features[node_index] = self._canonical_local_obs(node_id, self._latest_local_obs[node_id])

        edge_index = np.zeros((2, self._max_edges), dtype=np.int64)
        edge_mask = np.zeros(self._max_edges, dtype=np.float32)
        edge_weight = np.zeros(self._max_edges, dtype=np.float32)
        for edge_offset, (source, target) in enumerate(self._edges[: self._max_edges]):
            edge_index[:, edge_offset] = [source, target]
            edge_mask[edge_offset] = 1.0
            edge_weight[edge_offset] = self._edge_weights[edge_offset]

        action_mask = np.zeros(self._num_actions, dtype=np.float32)
        action_mask[: self._action_sizes[agent_id]] = 1.0
        node_action_mask = np.zeros((self._num_nodes, self._num_actions), dtype=np.float32)
        for node_id, node_index in self._agent_to_index.items():
            node_action_mask[node_index, : self._action_sizes[node_id]] = 1.0
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "edge_weight": edge_weight,
            "ego_index": np.asarray(self._agent_to_index[agent_id], dtype=np.int64),
            "action_mask": action_mask,
            "node_action_mask": node_action_mask,
            "phase_pair_mask": self._phase_pair_mask.copy(),
            "phase_competition_mask": self._phase_competition_mask.copy(),
            "prev_joint_action": self._prev_joint_action.copy(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        reset_result = self.env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            local_obs, infos = reset_result
        else:
            local_obs, infos = reset_result, {agent_id: {} for agent_id in self.possible_agents}
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        self._latest_local_obs = {str(agent_id): np.asarray(obs, dtype=np.float32) for agent_id, obs in local_obs.items()}
        self._refresh_spaces()
        self._prev_joint_action = np.zeros((self._num_nodes, self._num_actions), dtype=np.float32)
        return {str(agent_id): self._graph_obs(str(agent_id)) for agent_id in local_obs.keys()}, infos

    def step(self, actions):
        clipped_actions = {
            str(agent_id): int(np.clip(int(action), 0, self._action_sizes[str(agent_id)] - 1))
            for agent_id, action in dict(actions or {}).items()
        }
        local_obs, rewards, terminations, truncations, infos = self.env.step(clipped_actions)
        self.agents = list(getattr(self.env, "agents", []))
        self._prev_joint_action = self._joint_action_one_hot(clipped_actions)
        for agent_id, obs in local_obs.items():
            self._latest_local_obs[str(agent_id)] = np.asarray(obs, dtype=np.float32)
        graph_obs = {str(agent_id): self._graph_obs(str(agent_id)) for agent_id in local_obs.keys()}
        return graph_obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[str(agent)]

    def action_space(self, agent):
        return self.action_spaces[str(agent)]

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        save = getattr(self.env, "save_csv", None)
        if callable(save):
            return save(out_csv_name, episode)
        return None
