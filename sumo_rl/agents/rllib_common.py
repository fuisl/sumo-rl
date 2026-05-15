"""Shared RLlib plumbing for algorithm-specific agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from sumo_rl.experiments.runner import _resolve_num_gpus
from sumo_rl.rllib.envs import build_multi_agent_wrapper, scenario_factory_name


@dataclass
class RllibAlgorithmContext:
    cfg: Any
    run_dir: Path
    algorithm_kind: str
    params: Dict[str, Any]
    policy_mode: str
    env_name: str
    policies: Dict[str, Any]
    active_policies: Dict[str, Any]
    total_timesteps: int


def plain_dict(cfg: Any) -> Any:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if isinstance(cfg, (list, tuple)):
        return list(cfg)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return dict(cfg)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def total_timesteps(cfg: Any) -> int:
    return max(1, int(getattr(getattr(cfg, "experiment", None), "total_timesteps", 1) or 1))


def cap_to_horizon(value: Any, horizon: int) -> int:
    return max(1, min(int(value), int(horizon)))


def policy_mode(params: Dict[str, Any]) -> str:
    return str(params.get("policy_mode", "independent") or "independent").strip().lower()


def policy_id_for_agent(agent_id: str, mode: str) -> str:
    if mode == "shared":
        return "shared_policy"
    return str(agent_id)


def build_policy_mapping(mode: str) -> Callable[..., str]:
    def _mapping_fn(agent_id, *args, **kwargs):
        del args, kwargs
        return policy_id_for_agent(str(agent_id), mode)

    return _mapping_fn


def register_multi_agent_env(cfg: Any, run_dir: Path, algorithm_kind: str, *, pad_spaces: bool = False) -> str:
    from ray.tune.registry import register_env

    env_name = f"sumo_rl_{scenario_factory_name(cfg)}_{algorithm_kind}"

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            experiment = getattr(cfg, "experiment", None)
            base_seed = int(getattr(experiment, "seed", 0) or 0)
            seed = base_seed + int(env_config.get("worker_index", 0) or 0)
        return build_multi_agent_wrapper(cfg, run_dir, seed=seed, pad_spaces=pad_spaces)

    register_env(env_name, _creator)
    return env_name


def build_multi_agent_policies(cfg: Any, run_dir: Path, *, pad_spaces: bool):
    from ray.rllib.policy.policy import PolicySpec

    experiment = getattr(cfg, "experiment", None)
    sample_env = build_multi_agent_wrapper(
        cfg,
        run_dir,
        seed=int(getattr(experiment, "seed", 0) or 0),
        pad_spaces=pad_spaces,
    )
    try:
        policies = {}
        for agent_id in sample_env.possible_agents:
            policies[str(agent_id)] = PolicySpec(
                observation_space=sample_env.observation_space(agent_id),
                action_space=sample_env.action_space(agent_id),
            )
        return policies
    finally:
        sample_env.close()


def build_shared_policy_dict(policies: Dict[str, Any]) -> Dict[str, Any]:
    first_spec = next(iter(policies.values()))
    return {"shared_policy": first_spec}


def build_algorithm_context(cfg: Any, run_dir: Path, algorithm_kind: str) -> RllibAlgorithmContext:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
    mode = policy_mode(params)
    pad_spaces = mode == "shared"
    policies = build_multi_agent_policies(cfg, run_dir, pad_spaces=pad_spaces)
    active_policies = build_shared_policy_dict(policies) if mode == "shared" else policies
    return RllibAlgorithmContext(
        cfg=cfg,
        run_dir=run_dir,
        algorithm_kind=algorithm_kind,
        params=params,
        policy_mode=mode,
        env_name=register_multi_agent_env(cfg, run_dir, algorithm_kind, pad_spaces=pad_spaces),
        policies=policies,
        active_policies=active_policies,
        total_timesteps=total_timesteps(cfg),
    )


def apply_env_runner_settings(config, params: Dict[str, Any]):
    num_env_runners = int(params.get("num_env_runners", 0) or 0)
    num_envs_per_runner = int(params.get("num_envs_per_env_runner", 1) or 1)
    if hasattr(config, "env_runners"):
        config = config.env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_runner,
        )
    if hasattr(config, "learners"):
        learner_kwargs: Dict[str, Any] = {}
        if params.get("num_learners") is not None:
            learner_kwargs["num_learners"] = int(params["num_learners"])
        if params.get("num_cpus_per_learner") is not None:
            learner_kwargs["num_cpus_per_learner"] = float(params["num_cpus_per_learner"])
        if params.get("num_gpus_per_learner", "auto") is not None:
            learner_kwargs["num_gpus_per_learner"] = _resolve_num_gpus(params.get("num_gpus_per_learner", "auto"))
        if params.get("local_gpu_idx") is not None:
            learner_kwargs["local_gpu_idx"] = int(params["local_gpu_idx"])
        if learner_kwargs:
            config = config.learners(**learner_kwargs)
    return config


def apply_training_settings(
    config,
    params: Dict[str, Any],
    *,
    total_timesteps_value: int,
    allowed_keys: tuple[str, ...],
    aliases: Optional[Dict[str, str]] = None,
):
    training_kwargs: Dict[str, Any] = {}
    for key in allowed_keys:
        if key in params and params[key] is not None:
            training_kwargs[key] = params[key]
    for source_key, target_key in (aliases or {}).items():
        if source_key in params and params[source_key] is not None:
            training_kwargs[target_key] = params[source_key]
    if "train_batch_size_per_learner" in training_kwargs:
        training_kwargs["train_batch_size_per_learner"] = cap_to_horizon(
            training_kwargs["train_batch_size_per_learner"],
            total_timesteps_value,
        )
    if "minibatch_size" in training_kwargs:
        training_kwargs["minibatch_size"] = cap_to_horizon(
            training_kwargs["minibatch_size"],
            int(training_kwargs.get("train_batch_size_per_learner", total_timesteps_value)),
        )
    if training_kwargs and hasattr(config, "training"):
        config = config.training(**training_kwargs)
    if hasattr(config, "reporting"):
        config = config.reporting(min_sample_timesteps_per_iteration=total_timesteps_value)
    return config


def apply_multi_agent_settings(config, context: RllibAlgorithmContext):
    return config.multi_agent(
        policies=context.active_policies,
        policy_mapping_fn=build_policy_mapping(context.policy_mode),
        policies_to_train=list(context.active_policies.keys()),
    )


def training_iterations(cfg: Any, params: Dict[str, Any]) -> int:
    value = params.get("train_iterations")
    if value is not None:
        return max(1, int(value))
    return max(1, min(20, total_timesteps(cfg) // 1000 or 1))


def rllib_counter_metrics(result: Dict[str, Any], *, algorithm_kind: str, iteration: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "algorithm/kind": algorithm_kind,
        "train/iteration": iteration,
    }
    for source_key, target_key in (
        ("training_iteration", "train/rllib/training_iteration"),
        ("time_total_s", "train/rllib/time_total_s"),
        ("time_this_iter_s", "train/rllib/time_this_iter_s"),
        ("num_env_steps_sampled_lifetime", "train/env_steps_sampled"),
        ("num_agent_steps_sampled_lifetime", "train/agent_steps_sampled"),
        ("num_episodes_lifetime", "train/episodes_total"),
    ):
        value = result.get(source_key)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            metrics[target_key] = float(value)

    env_runner_metrics = result.get("env_runners")
    if isinstance(env_runner_metrics, dict):
        for source_key, target_key in (
            ("episode_return_mean", "train/episode_return_mean"),
            ("episode_return_min", "train/episode_return_min"),
            ("episode_return_max", "train/episode_return_max"),
            ("episode_len_mean", "train/episode_len_mean"),
            ("num_env_steps_sampled_lifetime", "train/env_steps_sampled"),
            ("num_agent_steps_sampled_lifetime", "train/agent_steps_sampled"),
            ("num_episodes_lifetime", "train/episodes_total"),
        ):
            value = env_runner_metrics.get(source_key)
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                metrics[target_key] = float(value)
    return metrics


def flatten_numeric_metrics(value: Any, *, prefix: str, out: Dict[str, float], max_depth: int = 5) -> None:
    if max_depth < 0:
        return
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            out[prefix] = numeric_value
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        clean_key = str(key).replace("/", "_")
        if (
            key in {"config", "hist_stats", "sampler_results"}
            or "timer" in clean_key
            or "connector" in clean_key
            or "throughput" in clean_key
            or clean_key.startswith("num_trainable_parameters")
            or clean_key.startswith("num_non_trainable_parameters")
        ):
            continue
        flatten_numeric_metrics(item, prefix=f"{prefix}/{clean_key}", out=out, max_depth=max_depth - 1)
