from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    from gymnasium.spaces.utils import flatten, flatdim
except ImportError:  # pragma: no cover - gymnasium is optional for local unit tests
    from types import SimpleNamespace

    class _FallbackSpace:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _FallbackBox(_FallbackSpace):
        def __init__(self, low, high, shape, dtype):
            self.low = np.array(low)
            self.high = np.array(high)
            self.shape = tuple(shape)
            self.dtype = dtype

    class _FallbackDiscrete(_FallbackSpace):
        def __init__(self, n, start=0, dtype=np.int64):
            self.n = int(n)
            self.start = int(start)
            self.dtype = dtype

    class _FallbackMultiDiscrete(_FallbackSpace):
        def __init__(self, nvec, dtype=np.int64):
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.dtype = dtype

    class _FallbackMultiBinary(_FallbackSpace):
        def __init__(self, n, dtype=np.int64):
            self.n = int(n)
            self.dtype = dtype

    def _fallback_flatdim(space):
        if isinstance(space, _FallbackBox):
            return int(np.prod(space.shape))
        if isinstance(space, _FallbackDiscrete):
            return int(space.n)
        if isinstance(space, _FallbackMultiDiscrete):
            return int(np.sum(space.nvec))
        if isinstance(space, _FallbackMultiBinary):
            return int(space.n)
        raise TypeError(f"Unsupported space type: {type(space)!r}")

    def _fallback_flatten(space, value):
        if isinstance(space, _FallbackDiscrete):
            out = np.zeros(space.n, dtype=np.float32)
            index = int(value) - int(getattr(space, "start", 0))
            out[max(0, min(space.n - 1, index))] = 1.0
            return out
        return np.asarray(value, dtype=np.float32).reshape(-1)

    gym = SimpleNamespace(
        Env=object,
        Space=_FallbackSpace,
        spaces=SimpleNamespace(
            Box=_FallbackBox,
            Discrete=_FallbackDiscrete,
            MultiDiscrete=_FallbackMultiDiscrete,
            MultiBinary=_FallbackMultiBinary,
        ),
    )
    Box = _FallbackBox
    flatten = _fallback_flatten
    flatdim = _fallback_flatdim

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _RLLibMultiAgentEnv
except ImportError:  # pragma: no cover - ray is optional during local unit tests
    class _RLLibMultiAgentEnv:  # type: ignore[too-many-ancestors]
        pass


def _repo_root() -> Path:
    import sumo_rl

    return Path(sumo_rl.__file__).resolve().parent.parent


def _as_plain_dict(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _as_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_dict(item) for item in value]
    return value


def scenario_factory_name(cfg: Any) -> str:
    scenario_name = str(getattr(getattr(cfg, "scenario", None), "name", "") or "").strip()
    if scenario_name.startswith("resco_"):
        return scenario_name[len("resco_") :]
    return scenario_name


def _resolve_sumo_path(raw_path: Any) -> Any:
    if not isinstance(raw_path, str):
        return raw_path
    if raw_path.startswith("sumo_rl/"):
        return str(_repo_root() / raw_path)
    return raw_path


def _prepare_env_kwargs(cfg: Any, run_dir: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    kwargs = dict(_as_plain_dict(getattr(getattr(cfg, "env", None), "kwargs", {}) or {}))
    for key, value in list(kwargs.items()):
        if key.endswith("_file") or key in {"net_file", "route_file", "sumo_cfg_file"}:
            kwargs[key] = _resolve_sumo_path(value)

    if not kwargs.get("out_csv_name"):
        kwargs["out_csv_name"] = str(run_dir / "csv" / getattr(getattr(cfg, "experiment", None), "name", "run"))
    if not kwargs.get("tripinfo_output_name"):
        kwargs["tripinfo_output_name"] = str(run_dir / "tripinfo" / getattr(getattr(cfg, "experiment", None), "name", "run"))
    experiment = getattr(cfg, "experiment", None)
    total_timesteps = int(getattr(experiment, "total_timesteps", 0) or 0)
    if total_timesteps > 0 and "num_seconds" not in kwargs:
        kwargs["num_seconds"] = total_timesteps
    if seed is not None:
        kwargs["sumo_seed"] = int(seed)
    if "single_agent" not in kwargs:
        kwargs["single_agent"] = False
    return kwargs


def build_sumo_parallel_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    import sumo_rl

    factory_name = scenario_factory_name(cfg)
    if not factory_name:
        raise ValueError("The scenario name is missing from the configuration.")

    kwargs = _prepare_env_kwargs(cfg, run_dir, seed=seed)
    kwargs.pop("factory", None)
    kwargs.pop("scenario", None)
    kwargs["parallel"] = True

    reward_fn = kwargs.get("reward_fn")
    if not reward_fn:
        kwargs.pop("reward_fn", None)

    factory = getattr(sumo_rl, factory_name, None)
    if factory is None:
        raise ValueError(f"Unsupported RESCO factory: {factory_name}")
    return factory(**kwargs)


def _maybe_pad_pettingzoo_env(env: Any) -> Any:
    try:
        import supersuit as ss
    except ImportError:
        return env

    try:
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    except Exception:
        return env
    return env


def build_multi_agent_env(cfg: Any, run_dir: Path, seed: Optional[int] = None, *, pad_spaces: bool = False):
    """Build a PettingZoo parallel env suitable for RLlib multi-agent training."""

    base_env = build_sumo_parallel_env(cfg, run_dir, seed=seed)
    if pad_spaces:
        return _maybe_pad_pettingzoo_env(base_env)
    return base_env


def _flatten_space_dim(space: gym.Space) -> int:
    return int(flatdim(space))


def _flatten_obs_dict(obs_dict: Dict[str, Any], agent_ids: Iterable[str], spaces: Dict[str, gym.Space]) -> np.ndarray:
    chunks = []
    for agent_id in agent_ids:
        space = spaces[agent_id]
        obs = obs_dict.get(agent_id)
        if obs is None:
            chunks.append(np.zeros(_flatten_space_dim(space), dtype=np.float32))
            continue
        chunks.append(np.asarray(flatten(space, obs), dtype=np.float32).reshape(-1))
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _action_slice_to_space_value(space: gym.Space, raw_slice: np.ndarray) -> Any:
    raw_slice = np.asarray(raw_slice).reshape(-1)
    if isinstance(space, gym.spaces.Discrete):
        if raw_slice.size == 0:
            return int(space.start if hasattr(space, "start") else 0)
        return int(np.argmax(raw_slice))
    if isinstance(space, gym.spaces.MultiDiscrete):
        one_hot_chunks = []
        offset = 0
        for n in space.nvec:
            chunk = raw_slice[offset : offset + int(n)]
            one_hot_chunks.append(int(np.argmax(chunk)) if chunk.size else 0)
            offset += int(n)
        return np.asarray(one_hot_chunks, dtype=space.dtype)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.asarray(np.rint(np.clip(raw_slice, 0.0, 1.0)), dtype=space.dtype)
    if isinstance(space, gym.spaces.Box):
        clipped = np.clip(raw_slice, space.low.reshape(-1), space.high.reshape(-1))
        return clipped.reshape(space.shape).astype(space.dtype)
    return raw_slice


@dataclass
class JointActionBoxEnv(gym.Env):
    """Expose a parallel PettingZoo SUMO env as a single-agent continuous Box env.

    This adapter is used for the SAC compatibility path. It keeps the underlying
    SUMO/PettingZoo environment intact and only converts the joint action and
    observation spaces into RLlib-friendly gymnasium spaces.
    """

    base_env: Any
    reward_reduction: str = "sum"

    def __post_init__(self) -> None:
        self.env = self.base_env
        self.possible_agents = list(getattr(self.base_env, "possible_agents", getattr(self.base_env, "agents", [])))
        if not self.possible_agents:
            raise ValueError("The joint-action adapter requires at least one agent.")
        self.agent_ids = list(self.possible_agents)
        self.observation_spaces = {
            agent_id: self.base_env.observation_space(agent_id)
            if callable(getattr(self.base_env, "observation_space", None))
            else self.base_env.observation_spaces(agent_id)
            for agent_id in self.agent_ids
        }
        self.action_spaces = {
            agent_id: self.base_env.action_space(agent_id)
            if callable(getattr(self.base_env, "action_space", None))
            else self.base_env.action_spaces(agent_id)
            for agent_id in self.agent_ids
        }
        self._obs_space_dim = int(sum(_flatten_space_dim(space) for space in self.observation_spaces.values()))
        self._action_dims = [max(1, _flatten_space_dim(space)) for space in self.action_spaces.values()]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_space_dim,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(int(sum(self._action_dims)),),
            dtype=np.float32,
        )
        self._last_obs_dict: Dict[str, Any] = {}
        self._last_info_dict: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        result = self.base_env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, info_dict = result
        else:
            obs_dict, info_dict = result, {}
        self._last_obs_dict = dict(obs_dict or {})
        self._last_info_dict = dict(info_dict or {})
        return self._flatten_obs(self._last_obs_dict), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        joint_actions: Dict[str, Any] = {}
        offset = 0
        for agent_id, action_space in zip(self.agent_ids, self.action_spaces.values()):
            width = max(1, _flatten_space_dim(action_space))
            action_slice = action[offset : offset + width]
            offset += width
            joint_actions[agent_id] = _action_slice_to_space_value(action_space, action_slice)

        obs_dict, rewards, terminations, truncations, infos = self.base_env.step(joint_actions)
        self._last_obs_dict = dict(obs_dict or {})
        self._last_info_dict = dict(infos or {})
        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or all(bool(terminations.get(agent_id, False)) for agent_id in self.agent_ids)
            or all(bool(truncations.get(agent_id, False)) for agent_id in self.agent_ids)
        )

        reward = 0.0
        if self.reward_reduction == "mean" and rewards:
            reward = float(np.mean([float(value) for value in rewards.values()]))
        else:
            reward = float(sum(float(value) for value in rewards.values()))

        next_obs = self._flatten_obs(self._last_obs_dict)
        info = dict(infos or {})
        info["joint_action_reward"] = reward
        info["joint_action_done"] = done
        return next_obs, reward, done, False, info

    def _flatten_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        return _flatten_obs_dict(obs_dict, self.agent_ids, self.observation_spaces)

    def close(self) -> None:
        if hasattr(self.base_env, "close"):
            self.base_env.close()

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.base_env, item)


class SumoParallelMultiAgentEnv(_RLLibMultiAgentEnv):
    """A light MultiAgentEnv wrapper around the SUMO PettingZoo parallel API."""

    def __init__(self, base_env: Any):
        self.base_env = base_env
        self.env = base_env
        self.possible_agents = list(getattr(base_env, "possible_agents", getattr(base_env, "agents", [])))
        self.agents = list(self.possible_agents)
        self._last_obs_dict: Dict[str, Any] = {}
        self._last_info_dict: Dict[str, Any] = {}
        self._observation_spaces = {
            agent_id: base_env.observation_space(agent_id)
            if callable(getattr(base_env, "observation_space", None))
            else base_env.observation_spaces(agent_id)
            for agent_id in self.agents
        }
        self._action_spaces = {
            agent_id: base_env.action_space(agent_id)
            if callable(getattr(base_env, "action_space", None))
            else base_env.action_spaces(agent_id)
            for agent_id in self.agents
        }

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    def observation_space(self, agent_id: str) -> gym.Space:
        return self._observation_spaces[agent_id]

    def action_space(self, agent_id: str) -> gym.Space:
        return self._action_spaces[agent_id]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        result = self.base_env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, info_dict = result
        else:
            obs_dict, info_dict = result, {}
        self._last_obs_dict = dict(obs_dict or {})
        self._last_info_dict = dict(info_dict or {})
        return self._last_obs_dict, self._last_info_dict

    def step(self, action_dict: Dict[str, Any]):
        obs_dict, rewards, terminations, truncations, infos = self.base_env.step(action_dict)
        self._last_obs_dict = dict(obs_dict or {})
        self._last_info_dict = dict(infos or {})
        terminations = dict(terminations or {})
        truncations = dict(truncations or {})
        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or all(bool(terminations.get(agent_id, False)) for agent_id in self.possible_agents)
            or all(bool(truncations.get(agent_id, False)) for agent_id in self.possible_agents)
        )
        terminations["__all__"] = done
        truncations["__all__"] = done
        return obs_dict, rewards, terminations, truncations, infos

    def close(self) -> None:
        if hasattr(self.base_env, "close"):
            self.base_env.close()

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.base_env, item)


def build_multi_agent_wrapper(
    cfg: Any,
    run_dir: Path,
    seed: Optional[int] = None,
    *,
    pad_spaces: bool = False,
) -> SumoParallelMultiAgentEnv:
    return SumoParallelMultiAgentEnv(build_multi_agent_env(cfg, run_dir, seed=seed, pad_spaces=pad_spaces))


def build_joint_action_env(cfg: Any, run_dir: Path, seed: Optional[int] = None) -> JointActionBoxEnv:
    return JointActionBoxEnv(build_sumo_parallel_env(cfg, run_dir, seed=seed))
