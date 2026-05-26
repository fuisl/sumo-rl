"""Graph-observation wrapper for CoLight."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from sumo_rl.environment.observations import ObservationFunction


class CoLightObservationFunction(ObservationFunction):
    """Lane-count observation with optional phase encoding for CoLight."""

    include_phase = True
    phase_encoding = "one_hot"
    vehicle_max = 1.0

    def __call__(self) -> np.ndarray:
        features: list[float] = []
        if self.include_phase:
            if self.phase_encoding == "one_hot":
                features.extend(1.0 if self.ts.green_phase == index else 0.0 for index in range(self.ts.num_green_phases))
            elif self.phase_encoding == "index":
                denominator = max(1, self.ts.num_green_phases - 1)
                features.append(float(self.ts.green_phase) / float(denominator))
            else:
                raise ValueError(f"Unsupported CoLight phase_encoding: {self.phase_encoding!r}.")

        scale = max(float(self.vehicle_max), 1e-6)
        for lane_id in self.ts.lanes:
            vehicle_count = len(
                [
                    veh
                    for veh in self.ts.sumo.lane.getLastStepVehicleIDs(lane_id)
                    if not str(veh).startswith("ghost")
                ]
            )
            features.append(float(vehicle_count) / scale)
        return np.asarray(features, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        phase_width = 0
        if self.include_phase:
            phase_width = self.ts.num_green_phases if self.phase_encoding == "one_hot" else 1
        width = phase_width + len(self.ts.lanes)
        return spaces.Box(
            low=np.zeros(width, dtype=np.float32),
            high=np.full(width, np.inf, dtype=np.float32),
            dtype=np.float32,
        )


def make_colight_observation_class(
    *,
    include_phase: bool = True,
    phase_encoding: str = "one_hot",
    vehicle_max: float = 1.0,
):
    class ConfiguredCoLightObservationFunction(CoLightObservationFunction):
        pass

    ConfiguredCoLightObservationFunction.include_phase = bool(include_phase)
    ConfiguredCoLightObservationFunction.phase_encoding = str(phase_encoding)
    ConfiguredCoLightObservationFunction.vehicle_max = float(vehicle_max)
    ConfiguredCoLightObservationFunction.__name__ = "ConfiguredCoLightObservationFunction"
    return ConfiguredCoLightObservationFunction


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


class CoLightGraphParallelEnv:
    """PettingZoo parallel wrapper exposing whole-graph observations per agent."""

    metadata = {"name": "sumo_rl_colight_graph_v0", "is_parallelizable": True}

    def __init__(self, env: Any):
        self.env = env
        self.possible_agents = [str(agent_id) for agent_id in getattr(env, "possible_agents", getattr(env, "agents", []))]
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self._agent_to_index = {agent_id: index for index, agent_id in enumerate(self.possible_agents)}
        self._latest_local_obs: Dict[str, np.ndarray] = {}
        self._refresh_spaces()

    def _refresh_spaces(self) -> None:
        self._num_nodes = max(1, len(self.possible_agents))
        local_dims = []
        action_sizes = []
        for agent_id in self.possible_agents:
            local_dims.append(int(self.env.observation_space(agent_id).shape[0]))
            action_sizes.append(int(self.env.action_space(agent_id).n))
        self._node_feature_dim = max(local_dims or [1])
        self._max_actions = max(action_sizes or [1])
        self._action_sizes = {agent_id: int(self.env.action_space(agent_id).n) for agent_id in self.possible_agents}
        self._edges = self._build_edges()
        self._max_edges = max(1, len(self._edges))

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
                    "ego_index": spaces.Box(low=0, high=max(0, self._num_nodes - 1), shape=(), dtype=np.int64),
                    "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self._max_actions,), dtype=np.float32),
                }
            )
            for agent_id in self.possible_agents
        }
        shared_action_space = spaces.Discrete(self._max_actions)
        self.action_spaces = {agent_id: shared_action_space for agent_id in self.possible_agents}

    def _build_edges(self) -> list[tuple[int, int]]:
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
                    source_index = self._agent_to_index[source_id]
                    target_index = self._agent_to_index[target_id]
                    edges.add((source_index, target_index))
                    edges.add((target_index, source_index))
        return sorted(edges)

    def _pad_local_obs(self, obs: np.ndarray) -> np.ndarray:
        padded = np.zeros(self._node_feature_dim, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        padded[: min(len(obs), self._node_feature_dim)] = obs[: self._node_feature_dim]
        return padded

    def _graph_obs(self, agent_id: str) -> Dict[str, np.ndarray]:
        node_features = np.zeros((self._num_nodes, self._node_feature_dim), dtype=np.float32)
        for node_id, node_index in self._agent_to_index.items():
            if node_id in self._latest_local_obs:
                node_features[node_index] = self._pad_local_obs(self._latest_local_obs[node_id])

        edge_index = np.zeros((2, self._max_edges), dtype=np.int64)
        edge_mask = np.zeros(self._max_edges, dtype=np.float32)
        for edge_offset, (source, target) in enumerate(self._edges[: self._max_edges]):
            edge_index[:, edge_offset] = [source, target]
            edge_mask[edge_offset] = 1.0

        action_mask = np.zeros(self._max_actions, dtype=np.float32)
        action_mask[: self._action_sizes[agent_id]] = 1.0
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "ego_index": np.asarray(self._agent_to_index[agent_id], dtype=np.int64),
            "action_mask": action_mask,
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
        return {agent_id: self._graph_obs(agent_id) for agent_id in local_obs.keys()}, infos

    def step(self, actions):
        clipped_actions = {}
        for agent_id, action in dict(actions or {}).items():
            action_size = self._action_sizes[str(agent_id)]
            clipped_actions[str(agent_id)] = int(np.clip(int(action), 0, action_size - 1))

        local_obs, rewards, terminations, truncations, infos = self.env.step(clipped_actions)
        self.agents = list(getattr(self.env, "agents", []))
        for agent_id, obs in local_obs.items():
            self._latest_local_obs[str(agent_id)] = np.asarray(obs, dtype=np.float32)
        graph_obs = {agent_id: self._graph_obs(str(agent_id)) for agent_id in local_obs.keys()}
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
