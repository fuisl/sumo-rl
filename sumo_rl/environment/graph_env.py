"""PettingZoo graph-observation wrapper for runner-native GNN algorithms."""

from __future__ import annotations

from collections import deque
from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from sumo_rl.agents.rllib_common import build_sumo_parallel_env
from sumo_rl.models.graph import (
    GraphObservationHistory,
    TrafficSignalGraph,
    build_traffic_signal_graph,
    pack_density_queue_features,
    traffic_signals_from_base_env,
)


def resolve_sumo_base_env(env: Any) -> Any:
    """Find the underlying SumoEnvironment through PettingZoo/wrapper layers."""

    queue = deque([env])
    visited = set()
    while queue:
        current = queue.popleft()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, "traffic_signals") and hasattr(current, "ts_ids"):
            return current
        for attr in ("env", "aec_env", "unwrapped", "base_env", "par_env"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                queue.append(candidate)
    raise RuntimeError("Unable to resolve the underlying SUMO environment for graph observations.")


class GraphParallelEnv:
    """Wrap a parallel PettingZoo env and replace observations with graph histories."""

    metadata = {"name": "sumo_rl_graph_v0", "is_parallelizable": True}

    def __init__(
        self,
        env: Any,
        *,
        history_len: int = 5,
        include_virtual_nodes: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        self.env = env
        self.par_env = env
        self.possible_agents = list(getattr(env, "possible_agents", getattr(env, "agents", [])))
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self.base_env = resolve_sumo_base_env(env)
        self.graph = build_traffic_signal_graph(
            traffic_signals_from_base_env(self.base_env),
            include_virtual_nodes=include_virtual_nodes,
            add_self_loops=add_self_loops,
        )
        self.history = GraphObservationHistory(history_len, self.graph)
        self.observation_spaces = {agent_id: self.history.observation_space for agent_id in self.possible_agents}
        self.action_spaces = {
            agent_id: self._base_action_space(agent_id)
            for agent_id in self.possible_agents
        }

    def _base_action_space(self, agent_id: str):
        action_space = getattr(self.env, "action_space", None)
        if callable(action_space):
            return action_space(agent_id)
        action_spaces = getattr(self.env, "action_spaces", None)
        if isinstance(action_spaces, dict):
            return action_spaces[agent_id]
        raise KeyError(f"Could not resolve action space for graph agent {agent_id!r}.")

    def observation_space(self, agent_id: str):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id: str):
        return self.action_spaces[agent_id]

    def _current_graph_frame(self):
        return pack_density_queue_features(traffic_signals_from_base_env(self.base_env), self.graph)

    def _graph_observations(self, agent_ids):
        graph_obs = self.history.as_array()
        return {agent_id: graph_obs.copy() for agent_id in agent_ids}

    def reset(self, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            observations, infos = result
        else:
            observations, infos = result, {agent_id: {} for agent_id in self.possible_agents}
        self.agents = list(getattr(self.env, "agents", observations.keys()))
        self.history.reset(self._current_graph_frame())
        return self._graph_observations(observations.keys()), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(getattr(self.env, "agents", observations.keys()))
        self.history.append(self._current_graph_frame())
        return self._graph_observations(observations.keys()), rewards, terminations, truncations, infos

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def render(self):
        render = getattr(self.env, "render", None)
        return render() if callable(render) else None

    def save_csv(self, out_csv_name, episode):
        save_csv = getattr(self.env, "save_csv", None)
        if callable(save_csv):
            save_csv(out_csv_name, episode)


def build_graph_parallel_env(cfg: Any, run_dir, seed=None, *, params: dict[str, Any] | None = None) -> GraphParallelEnv:
    params = dict(params or {})
    base_env = build_sumo_parallel_env(cfg, run_dir, seed=seed)
    return GraphParallelEnv(
        base_env,
        history_len=int(params.get("history_len", 5)),
        include_virtual_nodes=bool(params.get("include_virtual_nodes", True)),
        add_self_loops=bool(params.get("add_self_loops", True)),
    )


def build_rllib_graph_parallel_env(cfg: Any, run_dir, seed=None, *, params: dict[str, Any] | None = None):
    return ParallelPettingZooEnv(build_graph_parallel_env(cfg, run_dir, seed=seed, params=params))

