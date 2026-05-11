from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, flatten, flatdim


def _resolve_base_env(env: Any):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if (
            hasattr(current, "finalize_episode_summary")
            or hasattr(current, "metrics")
            or hasattr(current, "last_episode_final_info")
        ):
            return current
        if hasattr(current, "base_env"):
            current = current.base_env
            continue
        if hasattr(current, "par_env"):
            current = current.par_env
            continue
        if hasattr(current, "gym_env"):
            current = current.gym_env
            continue
        if hasattr(current, "unwrapped"):
            unwrapped = current.unwrapped
            if unwrapped is not current:
                current = getattr(unwrapped, "env", unwrapped)
                continue
        if hasattr(current, "unwrapped") and hasattr(current.unwrapped, "env"):
            current = current.unwrapped.env
            continue
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "vec_envs") and current.vec_envs:
            current = current.vec_envs[0]
            continue
        if hasattr(current, "envs") and current.envs:
            current = current.envs[0]
            continue
        if hasattr(current, "env"):
            next_env = current.env
            if next_env is current:
                break
            current = next_env
            continue
        break
    return env


class JointMultiAgentActionWrapper(gym.Env):
    """Adapt a PettingZoo parallel env to a single-agent Box action space.

    The wrapper maps each traffic signal's discrete action space to a slice of a
    continuous vector. Each slice is converted to a discrete action with argmax,
    which keeps the environment compatible with SB3 SAC while preserving the
    underlying discrete traffic-signal control problem.
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        self.base_env = _resolve_base_env(env)
        self.agents: List[str] = list(getattr(env, "possible_agents", None) or getattr(env, "agents", []) or [])
        if not self.agents:
            raise ValueError("The joint action wrapper requires a parallel env with at least one agent.")

        self._obs_spaces = [env.observation_space(agent) for agent in self.agents]
        self._action_dims = [int(env.action_space(agent).n) for agent in self.agents]
        self._obs_dim = int(sum(flatdim(space) for space in self._obs_spaces))
        self._action_dim = int(sum(self._action_dims))

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
        self.last_info: Dict[str, Any] = {}

    def _flatten_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        values = []
        for agent, space in zip(self.agents, self._obs_spaces):
            values.append(np.asarray(flatten(space, observation[agent]), dtype=np.float32).reshape(-1))
        if not values:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(values).astype(np.float32, copy=False)

    def _decode_action(self, action: Any) -> Dict[str, int]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != self._action_dim:
            raise ValueError(f"Expected {self._action_dim} action values, received {action_array.size}.")

        decoded: Dict[str, int] = {}
        offset = 0
        for agent, action_dim in zip(self.agents, self._action_dims):
            slice_end = offset + action_dim
            decoded[agent] = int(np.argmax(action_array[offset:slice_end]))
            offset = slice_end
        return decoded

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.last_info = info if isinstance(info, dict) else {}
        return self._flatten_observation(observation), info

    def step(self, action):
        decoded_action = self._decode_action(action)
        observation, rewards, terminations, truncations, infos = self.env.step(decoded_action)
        self.last_info = infos if isinstance(infos, dict) else {}

        reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
        reward = float(np.mean(reward_values)) if reward_values else 0.0

        terminated = bool(terminations.get("__all__", any(terminations.values()))) if isinstance(terminations, dict) else bool(terminations)
        truncated = bool(truncations.get("__all__", any(truncations.values()))) if isinstance(truncations, dict) else bool(truncations)

        return self._flatten_observation(observation), reward, terminated, truncated, infos

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None


class SeedableVecEnvProxy:
    """Proxy a vec env and provide the seed method SB3 expects."""

    def __init__(self, env):
        self.env = env

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        if seed is None:
            return None
        if hasattr(self.env, "reset"):
            try:
                return self.env.reset(seed=seed)
            except TypeError:
                return None
        return None

    def __getattr__(self, name):
        return getattr(self.env, name)
