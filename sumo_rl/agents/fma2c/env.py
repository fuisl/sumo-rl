"""Feudal MA2C environment adapter for RLlib.

The adapter keeps the underlying SUMO-RL environment unchanged and exposes a
PettingZoo Parallel API with extra virtual manager agents. It implements the
paper-facing observation and reward transformations used by FMA2C:

* worker local observations are normalized wave/wait lane features;
* manager local observations are north/east/south/west regional boundary waves;
* manager and worker information states contain spatially discounted neighbors;
* worker rewards include manager goal-following via cosine similarity;
* both levels use spatially discounted neighboring rewards.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import math
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium import spaces


MANAGER_PREFIX = "fma2c_manager_"
DEFAULT_MANAGER_ACTION_VECTORS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class NetworkLayout:
    coords: Dict[str, tuple[float, float]]
    edges: list[tuple[str, str]]


def is_manager_agent(agent_id: str) -> bool:
    return str(agent_id).startswith(MANAGER_PREFIX)


def _plain_dict(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return dict(value)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _is_ghost_vehicle(vehicle_id: str) -> bool:
    return isinstance(vehicle_id, str) and vehicle_id.startswith("ghost")


def _space_n(space: Any) -> int:
    value = getattr(space, "n", None)
    if value is None:
        raise TypeError(f"FMA2C expects discrete worker action spaces, got {space!r}.")
    return int(value)


def _get_space(env: Any, agent_id: str, kind: str):
    getter = getattr(env, f"{kind}_space", None)
    if callable(getter):
        return getter(agent_id)
    spaces_dict = getattr(env, f"{kind}_spaces", None)
    if isinstance(spaces_dict, dict):
        return spaces_dict[agent_id]
    raise KeyError(f"Could not resolve {kind} space for {agent_id!r}.")


def _env_children(env: Any) -> list[Any]:
    children = []
    for attr in ("base_env", "par_env", "aec_env", "env", "gym_env", "unwrapped"):
        child = getattr(env, attr, None)
        if child is not None and child is not env:
            children.append(child)
    return children


def _unwrap_sumo_env(env: Any) -> Any:
    queue = [env]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, "traffic_signals") and hasattr(current, "ts_ids"):
            return current
        queue.extend(_env_children(current))
    raise RuntimeError("FMA2C could not find the underlying SumoEnvironment.")


def _parse_network_layout(net_file: Optional[str]) -> NetworkLayout:
    if not net_file:
        return NetworkLayout(coords={}, edges=[])

    path = Path(net_file)
    if not path.exists():
        return NetworkLayout(coords={}, edges=[])

    coords: Dict[str, tuple[float, float]] = {}
    edges: list[tuple[str, str]] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return NetworkLayout(coords={}, edges=[])

    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id:
            continue
        try:
            coords[junction_id] = (float(junction.get("x", "0")), float(junction.get("y", "0")))
        except ValueError:
            continue

    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        from_node = edge.get("from")
        to_node = edge.get("to")
        if from_node and to_node:
            edges.append((from_node, to_node))
    return NetworkLayout(coords=coords, edges=edges)


def _normalize_clip(values: Iterable[float], norm: float, clip: float) -> np.ndarray:
    norm = float(norm) if norm else 1.0
    arr = np.asarray(list(values), dtype=np.float32) / norm
    return np.clip(arr, 0.0, float(clip)).astype(np.float32)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float32).reshape(-1)
    width = min(lhs.size, rhs.size)
    if width == 0:
        return 0.0
    lhs = lhs[:width]
    rhs = rhs[:width]
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


def _one_hot(index: Any, width: int) -> np.ndarray:
    vector = np.zeros(int(width), dtype=np.float32)
    try:
        position = int(index)
    except (TypeError, ValueError):
        return vector
    if 0 <= position < width:
        vector[position] = 1.0
    return vector


def _cardinal_from_points(source: tuple[float, float], target: tuple[float, float]) -> int:
    dx = source[0] - target[0]
    dy = source[1] - target[1]
    if abs(dy) >= abs(dx):
        return 0 if dy >= 0 else 2
    return 1 if dx >= 0 else 3


class FMA2CParallelEnv:
    """PettingZoo Parallel wrapper with FMA2C managers and shaped workers."""

    metadata = {"name": "sumo_rl_fma2c_v0", "is_parallelizable": True}

    def __init__(self, base_env: Any, *, params: Optional[Dict[str, Any]] = None, net_file: Optional[str] = None):
        self.base_env = base_env
        self.env = _unwrap_sumo_env(base_env)
        self.params = dict(params or {})
        self.alpha = float(self.params.get("spatial_discount", self.params.get("alpha", 0.75)))
        self.worker_reward_depth = int(self.params.get("worker_reward_neighbor_depth", 1))
        self.include_manager_neighbor_goals = bool(self.params.get("include_manager_neighbor_goals", True))
        self.use_last_action_fingerprints = str(self.params.get("fingerprint_source", "last_action")) != "none"
        self.wave_norm = float(self.params.get("norm_wave", 5.0))
        self.wait_norm = float(self.params.get("norm_wait", 100.0))
        self.clip_wave = float(self.params.get("clip_wave", 2.0))
        self.clip_wait = float(self.params.get("clip_wait", 2.0))
        self.worker_wait_coeff = float(self.params.get("coef_wait", self.params.get("worker_wait_coeff", 0.2)))
        self.goal_reward_coeff = float(self.params.get("goal_reward_coeff", self.params.get("goal_coeff", 1.0)))
        self.manager_liquidity_coeff = float(self.params.get("manager_liquidity_coeff", 1.0))

        self.worker_ids = [str(agent_id) for agent_id in getattr(base_env, "possible_agents", getattr(base_env, "agents", []))]
        if not self.worker_ids:
            self.worker_ids = [str(agent_id) for agent_id in getattr(self.env, "ts_ids", [])]
        if not self.worker_ids:
            raise RuntimeError("FMA2C requires at least one SUMO-RL traffic-signal worker.")

        self.layout = _parse_network_layout(net_file or getattr(self.env, "_net", None))
        self.worker_action_spaces = {agent_id: _get_space(base_env, agent_id, "action") for agent_id in self.worker_ids}
        self.worker_action_dims = {agent_id: _space_n(space) for agent_id, space in self.worker_action_spaces.items()}
        self.worker_local_dims = {
            agent_id: 2 * len(self.env.traffic_signals[agent_id].lanes)
            for agent_id in self.worker_ids
        }

        self.worker_neighbors = self._build_worker_neighbors()
        self.worker_distances = self._build_worker_distances()
        self.manager_action_vectors = self._build_manager_action_vectors()
        self.manager_action_dim = len(self.manager_action_vectors)
        self.goal_dim = int(self.manager_action_vectors.shape[1])
        self.regions = self._build_regions()
        self.manager_ids = list(self.regions.keys())
        self.worker_to_manager = {
            worker_id: manager_id
            for manager_id, region_workers in self.regions.items()
            for worker_id in region_workers
        }
        self.manager_neighbors = self._build_manager_neighbors()

        self.possible_agents = self.worker_ids + self.manager_ids
        self.agents = self.possible_agents[:]
        self.action_spaces = self._build_action_spaces()
        self.observation_spaces = self._build_observation_spaces()

        self.last_worker_fingerprints = {
            agent_id: np.zeros(self.worker_action_dims[agent_id], dtype=np.float32)
            for agent_id in self.worker_ids
        }
        self.last_manager_fingerprints = {
            manager_id: np.zeros(self.manager_action_dim, dtype=np.float32)
            for manager_id in self.manager_ids
        }
        self.last_manager_goals = {
            manager_id: np.zeros(self.goal_dim, dtype=np.float32)
            for manager_id in self.manager_ids
        }
        self.worker_local_obs = {
            agent_id: np.zeros(self.worker_local_dims[agent_id], dtype=np.float32)
            for agent_id in self.worker_ids
        }
        self.worker_goal_state = {
            agent_id: np.zeros(self.goal_dim, dtype=np.float32)
            for agent_id in self.worker_ids
        }
        self.manager_local_obs = {manager_id: np.zeros(4, dtype=np.float32) for manager_id in self.manager_ids}
        self.raw_manager_boundary_wave = {
            manager_id: np.zeros(4, dtype=np.float32)
            for manager_id in self.manager_ids
        }

    def observation_space(self, agent: str):
        return self.observation_spaces[str(agent)]

    def action_space(self, agent: str):
        return self.action_spaces[str(agent)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        reset_result = self.base_env.reset(seed=seed, options=options)
        base_obs, base_infos = self._normalize_reset_result(reset_result)
        del base_obs
        self.agents = self.possible_agents[:]
        self._reset_fingerprints()
        self._refresh_local_measurements()
        obs = self._build_all_observations()
        infos = self._build_infos(base_infos)
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        actions = dict(actions or {})
        prev_worker_goal_state = {key: value.copy() for key, value in self.worker_goal_state.items()}
        prev_boundary_wave = {key: value.copy() for key, value in self.raw_manager_boundary_wave.items()}

        self._apply_manager_actions(actions)
        worker_actions = {
            agent_id: actions[agent_id]
            for agent_id in self.worker_ids
            if agent_id in actions
        }
        step_result = self.base_env.step(worker_actions)
        base_obs, _base_rewards, base_terms, base_truncs, base_infos = self._normalize_step_result(step_result)
        del base_obs, _base_rewards

        self._refresh_local_measurements()
        worker_intrinsic_rewards = {
            agent_id: self._worker_intrinsic_reward(agent_id)
            for agent_id in self.worker_ids
        }
        worker_goal_rewards = {
            agent_id: self.goal_reward_coeff
            * _cosine_similarity(
                self.worker_goal_state[agent_id] - prev_worker_goal_state[agent_id],
                self.last_manager_goals[self.worker_to_manager[agent_id]],
            )
            for agent_id in self.worker_ids
        }
        worker_augmented_rewards = {
            agent_id: worker_intrinsic_rewards[agent_id] + worker_goal_rewards[agent_id]
            for agent_id in self.worker_ids
        }
        manager_local_rewards = {
            manager_id: self._manager_local_reward(manager_id, prev_boundary_wave[manager_id])
            for manager_id in self.manager_ids
        }
        manager_rewards = self._manager_spatial_rewards(manager_local_rewards)
        rewards = self._worker_spatial_rewards(worker_augmented_rewards, manager_rewards)
        rewards.update(manager_rewards)

        self._update_fingerprints(actions)
        obs = self._build_all_observations()
        terminations, truncations, done_all = self._build_done_dicts(base_terms, base_truncs)
        infos = self._build_infos(base_infos)
        for agent_id in self.worker_ids:
            infos.setdefault(agent_id, {})
            infos[agent_id].update(
                {
                    "fma2c/intrinsic_reward": worker_intrinsic_rewards[agent_id],
                    "fma2c/goal_reward": worker_goal_rewards[agent_id],
                    "fma2c/manager_id": self.worker_to_manager[agent_id],
                }
            )
        for manager_id in self.manager_ids:
            infos.setdefault(manager_id, {})
            infos[manager_id]["fma2c/local_reward"] = manager_local_rewards[manager_id]

        if done_all:
            self.agents = []
        else:
            self.agents = self.possible_agents[:]
        return obs, rewards, terminations, truncations, infos

    def close(self):
        close = getattr(self.base_env, "close", None)
        if callable(close):
            close()

    def render(self):
        render = getattr(self.base_env, "render", None)
        if callable(render):
            return render()
        return None

    def _normalize_reset_result(self, result: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            obs, infos = result
            return dict(obs or {}), dict(infos or {})
        return dict(result or {}), {}

    def _normalize_step_result(self, result: Any):
        if not isinstance(result, tuple):
            raise TypeError(f"Unexpected FMA2C base env step result: {result!r}")
        if len(result) == 5:
            obs, rewards, terminations, truncations, infos = result
            return dict(obs or {}), dict(rewards or {}), dict(terminations or {}), dict(truncations or {}), dict(infos or {})
        if len(result) == 4:
            obs, rewards, dones, infos = result
            dones = dict(dones or {})
            done_all = bool(dones.get("__all__", False))
            terminations = {agent_id: False for agent_id in self.worker_ids}
            truncations = {agent_id: bool(dones.get(agent_id, done_all)) for agent_id in self.worker_ids}
            return dict(obs or {}), dict(rewards or {}), terminations, truncations, dict(infos or {})
        raise TypeError(f"Unexpected FMA2C base env step result length: {len(result)}")

    def _reset_fingerprints(self) -> None:
        for agent_id in self.worker_ids:
            self.last_worker_fingerprints[agent_id] = np.zeros(self.worker_action_dims[agent_id], dtype=np.float32)
        for manager_id in self.manager_ids:
            self.last_manager_fingerprints[manager_id] = np.zeros(self.manager_action_dim, dtype=np.float32)
            self.last_manager_goals[manager_id] = np.zeros(self.goal_dim, dtype=np.float32)

    def _build_manager_action_vectors(self) -> np.ndarray:
        configured = self.params.get("manager_action_vectors")
        vectors = configured if configured is not None else DEFAULT_MANAGER_ACTION_VECTORS
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError("manager_action_vectors must be a non-empty 2D list.")
        return arr

    def _build_regions(self) -> Dict[str, list[str]]:
        explicit = self.params.get("regions")
        if explicit:
            return self._explicit_regions(explicit)
        return self._auto_regions()

    def _explicit_regions(self, explicit: Any) -> Dict[str, list[str]]:
        regions: Dict[str, list[str]] = {}
        if isinstance(explicit, dict):
            items = explicit.items()
        else:
            items = [(f"region_{index}", workers) for index, workers in enumerate(explicit)]

        seen = set()
        for index, (region_name, workers) in enumerate(items):
            manager_id = f"{MANAGER_PREFIX}{index}_{region_name}"
            region_workers = [str(worker) for worker in workers if str(worker) in self.worker_ids]
            if not region_workers:
                continue
            regions[manager_id] = region_workers
            seen.update(region_workers)

        for worker_id in self.worker_ids:
            if worker_id not in seen:
                manager_id = f"{MANAGER_PREFIX}{len(regions)}_singleton"
                regions[manager_id] = [worker_id]
        return regions

    def _auto_regions(self) -> Dict[str, list[str]]:
        if len(self.worker_ids) == 1:
            return {f"{MANAGER_PREFIX}0": [self.worker_ids[0]]}

        manager_grid = self.params.get("manager_grid", [2, 2] if len(self.worker_ids) >= 4 else [1, len(self.worker_ids)])
        rows = max(1, int(manager_grid[0]))
        cols = max(1, int(manager_grid[1]))
        coords = {agent_id: self.layout.coords.get(agent_id) for agent_id in self.worker_ids}
        if all(value is not None for value in coords.values()):
            xs = sorted({coords[agent_id][0] for agent_id in self.worker_ids})
            ys = sorted({coords[agent_id][1] for agent_id in self.worker_ids})
            x_rank = {value: index for index, value in enumerate(xs)}
            y_rank = {value: index for index, value in enumerate(ys)}
            buckets: Dict[tuple[int, int], list[str]] = defaultdict(list)
            for agent_id in self.worker_ids:
                x, y = coords[agent_id]
                col = min(cols - 1, int(x_rank[x] * cols / max(1, len(xs))))
                row = min(rows - 1, int(y_rank[y] * rows / max(1, len(ys))))
                buckets[(row, col)].append(agent_id)
            ordered = [
                sorted(workers)
                for _, workers in sorted(buckets.items(), key=lambda item: item[0])
                if workers
            ]
        else:
            desired = max(1, rows * cols)
            chunk = int(math.ceil(len(self.worker_ids) / desired))
            ordered = [self.worker_ids[index : index + chunk] for index in range(0, len(self.worker_ids), chunk)]

        return {f"{MANAGER_PREFIX}{index}": workers for index, workers in enumerate(ordered)}

    def _build_worker_neighbors(self) -> Dict[str, list[str]]:
        adjacency = {agent_id: set() for agent_id in self.worker_ids}
        worker_set = set(self.worker_ids)
        for source, target in self.layout.edges:
            if source in worker_set and target in worker_set:
                adjacency[source].add(target)
                adjacency[target].add(source)
        if not any(adjacency.values()):
            adjacency = self._coordinate_neighbors()
        return {agent_id: sorted(neighbors) for agent_id, neighbors in adjacency.items()}

    def _coordinate_neighbors(self) -> Dict[str, set[str]]:
        adjacency = {agent_id: set() for agent_id in self.worker_ids}
        coords = {agent_id: self.layout.coords.get(agent_id) for agent_id in self.worker_ids}
        if not all(value is not None for value in coords.values()):
            return adjacency

        xs = sorted({coords[agent_id][0] for agent_id in self.worker_ids})
        ys = sorted({coords[agent_id][1] for agent_id in self.worker_ids})
        dx = min([b - a for a, b in zip(xs, xs[1:])] or [float("inf")])
        dy = min([b - a for a, b in zip(ys, ys[1:])] or [float("inf")])
        tolerance = max(1.0, min(dx, dy) * 0.25 if np.isfinite(min(dx, dy)) else 1.0)
        for left in self.worker_ids:
            lx, ly = coords[left]
            for right in self.worker_ids:
                if left == right:
                    continue
                rx, ry = coords[right]
                same_column_adjacent = abs(lx - rx) <= tolerance and abs(abs(ly - ry) - dy) <= tolerance
                same_row_adjacent = abs(ly - ry) <= tolerance and abs(abs(lx - rx) - dx) <= tolerance
                if same_column_adjacent or same_row_adjacent:
                    adjacency[left].add(right)
        return adjacency

    def _build_worker_distances(self) -> Dict[str, Dict[str, int]]:
        distances: Dict[str, Dict[str, int]] = {}
        for source in self.worker_ids:
            current_distances = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in self.worker_neighbors.get(current, []):
                    if neighbor in current_distances:
                        continue
                    current_distances[neighbor] = current_distances[current] + 1
                    queue.append(neighbor)
            distances[source] = current_distances
        return distances

    def _build_manager_neighbors(self) -> Dict[str, list[str]]:
        manager_edges = {manager_id: set() for manager_id in self.manager_ids}
        for worker_id, neighbors in self.worker_neighbors.items():
            manager_id = self.worker_to_manager.get(worker_id)
            for neighbor_id in neighbors:
                neighbor_manager = self.worker_to_manager.get(neighbor_id)
                if manager_id and neighbor_manager and manager_id != neighbor_manager:
                    manager_edges[manager_id].add(neighbor_manager)
                    manager_edges[neighbor_manager].add(manager_id)
        return {manager_id: sorted(neighbors) for manager_id, neighbors in manager_edges.items()}

    def _build_action_spaces(self) -> Dict[str, spaces.Space]:
        action_spaces = dict(self.worker_action_spaces)
        for manager_id in self.manager_ids:
            action_spaces[manager_id] = spaces.Discrete(self.manager_action_dim)
        return action_spaces

    def _build_observation_spaces(self) -> Dict[str, spaces.Box]:
        observation_spaces: Dict[str, spaces.Box] = {}
        for worker_id in self.worker_ids:
            size = self._worker_observation_size(worker_id)
            observation_spaces[worker_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(size,),
                dtype=np.float32,
            )
        for manager_id in self.manager_ids:
            size = self._manager_observation_size(manager_id)
            observation_spaces[manager_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(size,),
                dtype=np.float32,
            )
        return observation_spaces

    def _worker_observation_size(self, worker_id: str) -> int:
        size = self.worker_local_dims[worker_id]
        for neighbor_id in self._regional_worker_neighbors(worker_id):
            size += self.worker_local_dims[neighbor_id]
            if self.use_last_action_fingerprints:
                size += self.worker_action_dims[neighbor_id]
        for _manager_id in self._worker_manager_goal_ids(worker_id):
            size += self.goal_dim
        return size

    def _manager_observation_size(self, manager_id: str) -> int:
        size = 4
        for _neighbor_id in self.manager_neighbors.get(manager_id, []):
            size += 4
            if self.use_last_action_fingerprints:
                size += self.manager_action_dim
        return size

    def _regional_worker_neighbors(self, worker_id: str) -> list[str]:
        manager_id = self.worker_to_manager[worker_id]
        region_workers = set(self.regions[manager_id])
        neighbors = []
        for neighbor_id in self.worker_neighbors.get(worker_id, []):
            if neighbor_id in region_workers:
                neighbors.append(neighbor_id)
        return sorted(neighbors)

    def _worker_manager_goal_ids(self, worker_id: str) -> list[str]:
        manager_id = self.worker_to_manager[worker_id]
        manager_ids = [manager_id]
        if self.include_manager_neighbor_goals:
            manager_ids.extend(self.manager_neighbors.get(manager_id, []))
        return manager_ids

    def _build_all_observations(self) -> Dict[str, np.ndarray]:
        observations = {agent_id: self._worker_observation(agent_id) for agent_id in self.worker_ids}
        observations.update({manager_id: self._manager_observation(manager_id) for manager_id in self.manager_ids})
        return observations

    def _worker_observation(self, worker_id: str) -> np.ndarray:
        parts = [self.worker_local_obs[worker_id]]
        for neighbor_id in self._regional_worker_neighbors(worker_id):
            parts.append(self.alpha * self.worker_local_obs[neighbor_id])
            if self.use_last_action_fingerprints:
                parts.append(self.last_worker_fingerprints[neighbor_id])
        for manager_id in self._worker_manager_goal_ids(worker_id):
            parts.append(self.last_manager_goals[manager_id])
        return np.concatenate(parts).astype(np.float32)

    def _manager_observation(self, manager_id: str) -> np.ndarray:
        parts = [self.manager_local_obs[manager_id]]
        for neighbor_id in self.manager_neighbors.get(manager_id, []):
            parts.append(self.alpha * self.manager_local_obs[neighbor_id])
            if self.use_last_action_fingerprints:
                parts.append(self.last_manager_fingerprints[neighbor_id])
        return np.concatenate(parts).astype(np.float32)

    def _build_infos(self, base_infos: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        infos = {
            agent_id: dict(base_infos.get(agent_id, {})) if isinstance(base_infos, dict) else {}
            for agent_id in self.worker_ids
        }
        for manager_id in self.manager_ids:
            infos[manager_id] = {"fma2c/region_workers": ",".join(self.regions[manager_id])}
        return infos

    def _build_done_dicts(self, base_terms: Dict[str, Any], base_truncs: Dict[str, Any]):
        done_all = bool(base_terms.get("__all__", False) or base_truncs.get("__all__", False))
        if not done_all:
            done_all = all(bool(base_terms.get(agent_id, False) or base_truncs.get(agent_id, False)) for agent_id in self.worker_ids)

        terminations = {}
        truncations = {}
        for agent_id in self.worker_ids:
            terminations[agent_id] = bool(base_terms.get(agent_id, False))
            truncations[agent_id] = bool(base_truncs.get(agent_id, done_all))
        for manager_id in self.manager_ids:
            terminations[manager_id] = False
            truncations[manager_id] = done_all
        return terminations, truncations, done_all

    def _apply_manager_actions(self, actions: Dict[str, Any]) -> None:
        for manager_id in self.manager_ids:
            if manager_id not in actions:
                continue
            action = int(actions[manager_id])
            if action < 0 or action >= self.manager_action_dim:
                action = 0
            self.last_manager_goals[manager_id] = self.manager_action_vectors[action].astype(np.float32)

    def _update_fingerprints(self, actions: Dict[str, Any]) -> None:
        if not self.use_last_action_fingerprints:
            return
        for worker_id in self.worker_ids:
            if worker_id in actions:
                self.last_worker_fingerprints[worker_id] = _one_hot(actions[worker_id], self.worker_action_dims[worker_id])
        for manager_id in self.manager_ids:
            if manager_id in actions:
                self.last_manager_fingerprints[manager_id] = _one_hot(actions[manager_id], self.manager_action_dim)

    def _refresh_local_measurements(self) -> None:
        for worker_id in self.worker_ids:
            self.worker_local_obs[worker_id] = self._worker_local_observation(worker_id)
            self.worker_goal_state[worker_id] = self._worker_directional_wave(worker_id)
        for manager_id in self.manager_ids:
            raw = self._manager_boundary_wave(manager_id, normalized=False)
            self.raw_manager_boundary_wave[manager_id] = raw
            self.manager_local_obs[manager_id] = _normalize_clip(raw, self.wave_norm, self.clip_wave)

    def _worker_local_observation(self, worker_id: str) -> np.ndarray:
        signal = self.env.traffic_signals[worker_id]
        waves = []
        waits = []
        for lane_id in signal.lanes:
            waves.append(self._lane_wave(lane_id))
            waits.append(self._lane_first_vehicle_wait(lane_id))
        return np.concatenate(
            [
                _normalize_clip(waves, self.wave_norm, self.clip_wave),
                _normalize_clip(waits, self.wait_norm, self.clip_wait),
            ]
        ).astype(np.float32)

    def _worker_directional_wave(self, worker_id: str) -> np.ndarray:
        signal = self.env.traffic_signals[worker_id]
        totals = np.zeros(self.goal_dim, dtype=np.float32)
        for lane_id in signal.lanes:
            direction = self._lane_approach_direction(worker_id, lane_id)
            if direction < self.goal_dim:
                totals[direction] += self._lane_wave(lane_id)
        return _normalize_clip(totals, self.wave_norm, self.clip_wave)

    def _manager_boundary_wave(self, manager_id: str, *, normalized: bool) -> np.ndarray:
        region_workers = set(self.regions[manager_id])
        totals = np.zeros(4, dtype=np.float32)
        for worker_id in self.regions[manager_id]:
            signal = self.env.traffic_signals[worker_id]
            for lane_id in signal.lanes:
                source = self._lane_source_node(lane_id)
                if source in region_workers:
                    continue
                direction = self._lane_approach_direction(worker_id, lane_id)
                totals[direction] += self._lane_wave(lane_id)
        if normalized:
            return _normalize_clip(totals, self.wave_norm, self.clip_wave)
        return totals

    def _lane_source_node(self, lane_id: str) -> Optional[str]:
        try:
            edge_id = self.env.sumo.lane.getEdgeID(lane_id)
            return self.env.sumo.edge.getFromJunction(edge_id)
        except Exception:
            edge_id = lane_id.rsplit("_", 1)[0]
            for source, target in self.layout.edges:
                del target
                if edge_id.startswith(source):
                    return source
        return None

    def _lane_approach_direction(self, worker_id: str, lane_id: str) -> int:
        source = self._lane_source_node(lane_id)
        source_point = self.layout.coords.get(source or "")
        target_point = self.layout.coords.get(worker_id)
        if source_point is None or target_point is None:
            index = self.env.traffic_signals[worker_id].lanes.index(lane_id)
            return int(index % 4)
        return _cardinal_from_points(source_point, target_point)

    def _lane_wave(self, lane_id: str) -> float:
        try:
            vehicles = self.env.sumo.lane.getLastStepVehicleIDs(lane_id)
        except Exception:
            return 0.0
        return float(sum(1 for vehicle_id in vehicles if not _is_ghost_vehicle(vehicle_id)))

    def _lane_first_vehicle_wait(self, lane_id: str) -> float:
        try:
            vehicles = self.env.sumo.lane.getLastStepVehicleIDs(lane_id)
        except Exception:
            return 0.0
        best_position = -1.0
        best_wait = 0.0
        for vehicle_id in vehicles:
            if _is_ghost_vehicle(vehicle_id):
                continue
            try:
                position = float(self.env.sumo.vehicle.getLanePosition(vehicle_id))
                wait = float(self.env.sumo.vehicle.getWaitingTime(vehicle_id))
            except Exception:
                continue
            if position > best_position:
                best_position = position
                best_wait = wait
        return best_wait

    def _worker_intrinsic_reward(self, worker_id: str) -> float:
        signal = self.env.traffic_signals[worker_id]
        wave = 0.0
        wait = 0.0
        for lane_id in signal.lanes:
            wave += self._lane_wave(lane_id)
            wait += self._lane_first_vehicle_wait(lane_id)
        return -wave - self.worker_wait_coeff * wait

    def _worker_liquidity(self, worker_id: str) -> float:
        signal = self.env.traffic_signals[worker_id]
        vehicles = []
        for lane_id in signal.lanes:
            try:
                vehicles.extend(self.env.sumo.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                continue
        liquidity = 0.0
        for vehicle_id in vehicles:
            if _is_ghost_vehicle(vehicle_id):
                continue
            try:
                speed = float(self.env.sumo.vehicle.getSpeed(vehicle_id))
                allowed = max(float(self.env.sumo.vehicle.getAllowedSpeed(vehicle_id)), 1e-6)
            except Exception:
                continue
            liquidity += max(0.0, min(1.0, speed / allowed))
        return liquidity

    def _manager_local_reward(self, manager_id: str, prev_boundary_wave: np.ndarray) -> float:
        boundary_delta = np.maximum(prev_boundary_wave - self.raw_manager_boundary_wave[manager_id], 0.0)
        arrival_proxy = float(np.sum(boundary_delta))
        liquidity = sum(self._worker_liquidity(worker_id) for worker_id in self.regions[manager_id])
        return arrival_proxy + self.manager_liquidity_coeff * float(liquidity)

    def _manager_spatial_rewards(self, manager_local_rewards: Dict[str, float]) -> Dict[str, float]:
        rewards = {}
        for manager_id, reward in manager_local_rewards.items():
            total = float(reward)
            for neighbor_id in self.manager_neighbors.get(manager_id, []):
                total += self.alpha * float(manager_local_rewards.get(neighbor_id, 0.0))
            rewards[manager_id] = total
        return rewards

    def _worker_spatial_rewards(
        self,
        worker_augmented_rewards: Dict[str, float],
        manager_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        for worker_id in self.worker_ids:
            manager_id = self.worker_to_manager[worker_id]
            total = 0.0
            for region_worker in self.regions[manager_id]:
                distance = self.worker_distances.get(worker_id, {}).get(region_worker)
                if distance is None or distance > self.worker_reward_depth:
                    continue
                total += (self.alpha ** distance) * float(worker_augmented_rewards.get(region_worker, 0.0))
            total += float(manager_rewards.get(manager_id, 0.0))
            rewards[worker_id] = total
        return rewards


def wrap_fma2c_env(base_env: Any, *, params: Optional[Dict[str, Any]] = None, net_file: Optional[str] = None) -> FMA2CParallelEnv:
    return FMA2CParallelEnv(base_env, params=_plain_dict(params), net_file=net_file)
