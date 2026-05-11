"""Seeded evaluation helpers for Stable-Baselines3 runs."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np


def resolve_eval_seeds(
    base_seed: Optional[int],
    eval_episodes: int,
    eval_seeds: Optional[Iterable[int]] = None,
) -> list[int]:
    """Return a deterministic list of eval seeds."""

    total_episodes = max(0, int(eval_episodes))
    if total_episodes <= 0:
        return []

    seeds: list[int] = []
    if eval_seeds is not None:
        for seed in eval_seeds:
            if seed is None:
                continue
            seeds.append(int(seed))

    if not seeds:
        base = int(base_seed) if base_seed is not None else 0
        return [base + offset for offset in range(total_episodes)]

    if len(seeds) < total_episodes:
        next_seed = seeds[-1] + 1
        seeds.extend(range(next_seed, next_seed + (total_episodes - len(seeds))))

    return seeds[:total_episodes]


def _reset_with_seed(env: Any, seed: Optional[int]) -> Any:
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                try:
                    env.seed(seed)
                except Exception:
                    pass
            result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool]:
    step_result = env.step(action)
    if len(step_result) == 5:
        observation, reward, terminated, truncated, _info = step_result
        done = np.asarray(terminated) | np.asarray(truncated)
    elif len(step_result) == 4:
        observation, reward, done, _info = step_result
    else:
        raise ValueError(f"Unsupported evaluation step return shape: {len(step_result)}")

    if isinstance(reward, dict):
        reward_value = float(sum(float(value) for value in reward.values()))
    else:
        reward_array = np.asarray(reward, dtype=float).reshape(-1)
        reward_value = float(reward_array.sum()) if reward_array.size else 0.0

    if isinstance(done, dict):
        done_value = bool(all(bool(value) for value in done.values()))
    else:
        done_array = np.asarray(done).reshape(-1)
        done_value = bool(done_array.all()) if done_array.size else bool(done)
    return observation, reward_value, done_value


def run_model_episodes_on_seeds(
    model: Any,
    eval_env: Any,
    eval_seeds: Sequence[int],
    *,
    deterministic: bool = True,
    on_episode_end: Optional[Callable[[int, float], None]] = None,
) -> list[float]:
    """Run one evaluation episode per seed and return the episode rewards."""

    seeds = [int(seed) for seed in eval_seeds]
    if not seeds:
        return []

    episode_rewards: list[float] = []
    for seed in seeds:
        observation = _reset_with_seed(eval_env, seed)
        episode_reward = 0.0
        while True:
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, done = _step_env(eval_env, action)
            episode_reward += reward
            if done:
                break
        episode_rewards.append(float(episode_reward))
        if on_episode_end is not None:
            on_episode_end(seed, float(episode_reward))

    return episode_rewards
