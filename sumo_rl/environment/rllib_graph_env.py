"""PettingZoo wrappers for RLlib graph-based traffic signal policies."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from sumo_rl.models.topology import GraphTopology


class RLLibGraphObservationWrapper(ParallelEnv):
    """Expose local and full-graph observations for each PettingZoo agent.

    RLlib models receive one agent sample at a time. This wrapper keeps the
    PettingZoo API intact while adding the full ordered traffic-light observation
    matrix, the requesting agent's node index, and a padded discrete action mask
    to every per-agent observation.
    """

    def __init__(self, env: ParallelEnv, topology: GraphTopology, dtype: Any = np.float32) -> None:
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.dtype = dtype

        possible_agents = list(getattr(env, "possible_agents", getattr(env, "agents", [])))
        topology_ids = list(topology.node_ids)
        self.possible_agents = topology_ids if topology_ids else possible_agents
        self.agents = list(getattr(env, "agents", self.possible_agents))

        self.node_ids = tuple(self.possible_agents)
        self.node_to_index = {agent_id: idx for idx, agent_id in enumerate(self.node_ids)}
        self.num_nodes = len(self.node_ids)

        self._base_observation_spaces = {agent_id: self._get_observation_space(agent_id) for agent_id in self.node_ids}
        self._base_action_spaces = {agent_id: self._get_action_space(agent_id) for agent_id in self.node_ids}

        self.obs_dim = max(int(np.prod(space.shape)) for space in self._base_observation_spaces.values())
        self.max_actions = max(int(space.n) for space in self._base_action_spaces.values())

        self.observation_spaces = {agent_id: self._wrapped_observation_space() for agent_id in self.node_ids}
        self.action_spaces = {agent_id: gym.spaces.Discrete(self.max_actions) for agent_id in self.node_ids}

        self.topology = topology
        self.invalid_action_count = 0
        self._last_raw_obs: Dict[str, np.ndarray] = {
            agent_id: np.zeros(self.obs_dim, dtype=self.dtype) for agent_id in self.node_ids
        }

    def model_config(self, normalize_edge_attr: bool = True) -> Dict[str, Any]:
        edge_attr = self.topology.edge_attr
        return {
            "obs_dim": int(self.obs_dim),
            "num_nodes": int(self.num_nodes),
            "max_actions": int(self.max_actions),
            "node_ids": list(self.node_ids),
            "edge_index": self.topology.edge_index.long().cpu().tolist(),
            "edge_attr": None if edge_attr is None else edge_attr.float().cpu().tolist(),
            "normalize_edge_attr": bool(normalize_edge_attr),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            observations, infos = result
        else:
            observations = result
            infos = {agent_id: {} for agent_id in self.possible_agents}

        self.agents = list(getattr(self.env, "agents", list(observations.keys())))
        self.invalid_action_count = 0
        self._update_last_observations(observations)
        return self._wrap_observations(observations), infos

    def step(self, actions: Mapping[str, Any]):
        safe_actions, invalid_by_agent = self._sanitize_actions(actions)
        result = self.env.step(safe_actions)

        if len(result) == 5:
            observations, rewards, terminations, truncations, infos = result
        else:
            observations, rewards, dones, infos = result
            terminations = dones
            truncations = {agent_id: False for agent_id in dones}

        self.agents = list(getattr(self.env, "agents", list(observations.keys())))
        self._update_last_observations(observations)
        infos = self._annotate_infos(infos, invalid_by_agent)
        return self._wrap_observations(observations), rewards, terminations, truncations, infos

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def state(self) -> np.ndarray:
        return self._graph_observation()

    def _get_observation_space(self, agent: str):
        if hasattr(self.env, "observation_space"):
            return self.env.observation_space(agent)
        return self.env.observation_spaces[agent]

    def _get_action_space(self, agent: str):
        if hasattr(self.env, "action_space"):
            return self.env.action_space(agent)
        return self.env.action_spaces[agent]

    def _wrapped_observation_space(self):
        local_low = np.full((self.obs_dim,), -np.inf, dtype=self.dtype)
        local_high = np.full((self.obs_dim,), np.inf, dtype=self.dtype)
        graph_low = np.full((self.num_nodes, self.obs_dim), -np.inf, dtype=self.dtype)
        graph_high = np.full((self.num_nodes, self.obs_dim), np.inf, dtype=self.dtype)
        mask_low = np.zeros((self.max_actions,), dtype=self.dtype)
        mask_high = np.ones((self.max_actions,), dtype=self.dtype)

        return gym.spaces.Dict(
            {
                "local_obs": gym.spaces.Box(
                    low=local_low,
                    high=local_high,
                    shape=(self.obs_dim,),
                    dtype=self.dtype,
                ),
                "graph_obs": gym.spaces.Box(
                    low=graph_low,
                    high=graph_high,
                    shape=(self.num_nodes, self.obs_dim),
                    dtype=self.dtype,
                ),
                "node_index": gym.spaces.Box(
                    low=0,
                    high=max(self.num_nodes - 1, 0),
                    shape=(1,),
                    dtype=np.int64,
                ),
                "action_mask": gym.spaces.Box(
                    low=mask_low,
                    high=mask_high,
                    shape=(self.max_actions,),
                    dtype=self.dtype,
                ),
            }
        )

    def _sanitize_actions(self, actions: Mapping[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
        safe_actions: Dict[str, int] = {}
        invalid_by_agent: Dict[str, int] = {}
        live_agents = set(getattr(self.env, "agents", self.agents))

        for agent_id, raw_action in actions.items():
            if agent_id not in live_agents:
                continue
            action = int(np.asarray(raw_action).item())
            valid_n = int(self._base_action_spaces[agent_id].n)
            if action < 0 or action >= valid_n:
                self.invalid_action_count += 1
                invalid_by_agent[agent_id] = invalid_by_agent.get(agent_id, 0) + 1
                action = 0
            safe_actions[agent_id] = action

        return safe_actions, invalid_by_agent

    def _annotate_infos(self, infos: Mapping[str, Any], invalid_by_agent: Mapping[str, int]) -> Dict[str, dict]:
        annotated: Dict[str, dict] = {}
        for agent_id in set(infos.keys()).union(invalid_by_agent.keys()):
            info = dict(infos.get(agent_id, {}))
            if agent_id in invalid_by_agent:
                info["baselinev1_invalid_actions"] = int(invalid_by_agent[agent_id])
            info["baselinev1_invalid_actions_total"] = int(self.invalid_action_count)
            annotated[agent_id] = info
        return annotated

    def _update_last_observations(self, observations: Mapping[str, Any]) -> None:
        for agent_id, obs in observations.items():
            if agent_id in self.node_to_index:
                self._last_raw_obs[agent_id] = self._pad_observation(obs)

    def _wrap_observations(self, observations: Mapping[str, Any]) -> Dict[str, dict]:
        graph_obs = self._graph_observation()
        return {agent_id: self._wrap_one(agent_id, graph_obs) for agent_id in observations.keys()}

    def _wrap_one(self, agent_id: str, graph_obs: np.ndarray) -> dict:
        node_index = self.node_to_index[agent_id]
        valid_actions = int(self._base_action_spaces[agent_id].n)
        action_mask = np.zeros(self.max_actions, dtype=self.dtype)
        action_mask[:valid_actions] = 1.0

        return {
            "local_obs": self._last_raw_obs[agent_id].astype(self.dtype, copy=True),
            "graph_obs": graph_obs.astype(self.dtype, copy=True),
            "node_index": np.asarray([node_index], dtype=np.int64),
            "action_mask": action_mask,
        }

    def _graph_observation(self) -> np.ndarray:
        graph_obs = np.zeros((self.num_nodes, self.obs_dim), dtype=self.dtype)
        for node_id, node_idx in self.node_to_index.items():
            graph_obs[node_idx] = self._last_raw_obs[node_id]
        return graph_obs

    def _pad_observation(self, observation: Any) -> np.ndarray:
        raw = np.asarray(observation, dtype=self.dtype).reshape(-1)
        padded = np.zeros(self.obs_dim, dtype=self.dtype)
        padded[: min(raw.size, self.obs_dim)] = raw[: self.obs_dim]
        return padded
