"""Shared RLlib plumbing for algorithm-specific agents."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from sumo_rl.experiments.runner import _prepare_env_kwargs, _resolve_num_gpus


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
    episode_seconds: int
    episode_steps: int


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


def scenario_factory_name(cfg: Any) -> str:
    scenario_name = str(getattr(getattr(cfg, "scenario", None), "name", "") or "").strip()
    if scenario_name.startswith("resco_"):
        return scenario_name[len("resco_") :]
    return scenario_name


def _episode_seconds(cfg: Any) -> int:
    experiment = getattr(cfg, "experiment", None)
    return int(getattr(experiment, "episode_seconds", 0) or 0)


def _env_factory_name(cfg: Any) -> str:
    env_cfg = getattr(cfg, "env", None)
    return str(getattr(env_cfg, "factory", "parallel_env") or "parallel_env")


def _rllib_env_kwargs(cfg: Any, run_dir: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    kwargs = _prepare_env_kwargs(cfg, run_dir)
    seconds = _episode_seconds(cfg)
    if seconds > 0 and "num_seconds" not in kwargs:
        kwargs["num_seconds"] = seconds
    if seed is not None:
        kwargs["sumo_seed"] = int(seed)
    kwargs.setdefault("single_agent", False)
    return kwargs


def build_sumo_parallel_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    """Build the PettingZoo parallel env in the same shape as the RLlib example."""

    import sumo_rl

    kwargs = _rllib_env_kwargs(cfg, run_dir, seed=seed)
    factory = _env_factory_name(cfg)
    if factory == "parallel_env":
        return sumo_rl.parallel_env(**kwargs)

    constructor = getattr(sumo_rl, factory, None)
    if constructor is None:
        raise ValueError(f"Unsupported RLlib env factory: {factory}")
    return constructor(**kwargs)


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


def build_rllib_parallel_env(
    cfg: Any,
    run_dir: Path,
    seed: Optional[int] = None,
    *,
    pad_spaces: bool = False,
):
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    base_env = build_sumo_parallel_env(cfg, run_dir, seed=seed)
    if pad_spaces:
        base_env = _maybe_pad_pettingzoo_env(base_env)
    return ParallelPettingZooEnv(base_env)


def _possible_agents(env: Any) -> list[str]:
    for candidate in (env, getattr(env, "par_env", None), getattr(env, "base_env", None), getattr(env, "env", None)):
        if candidate is None:
            continue
        agents = getattr(candidate, "possible_agents", None) or getattr(candidate, "agents", None)
        if agents:
            return [str(agent_id) for agent_id in agents]
    return []


def _agent_space(env: Any, agent_id: str, kind: str):
    get_space = getattr(env, f"get_{kind}_space", None)
    if callable(get_space):
        return get_space(agent_id)

    attr_name = f"{kind}_space"
    spaces_name = f"{kind}_spaces"
    for candidate in (env, getattr(env, "par_env", None), getattr(env, "base_env", None), getattr(env, "env", None)):
        if candidate is None:
            continue
        space_getter = getattr(candidate, attr_name, None)
        if callable(space_getter):
            return space_getter(agent_id)
        spaces = getattr(candidate, spaces_name, None)
        if isinstance(spaces, dict) and agent_id in spaces:
            return spaces[agent_id]
        space = getattr(candidate, attr_name, None)
        nested_spaces = getattr(space, "spaces", None)
        if isinstance(nested_spaces, dict) and agent_id in nested_spaces:
            return nested_spaces[agent_id]
    raise KeyError(f"Could not resolve {kind} space for RLlib agent {agent_id!r}.")


def _env_children(env: Any) -> list[Any]:
    children = []
    get_sub_environments = getattr(env, "get_sub_environments", None)
    if callable(get_sub_environments):
        try:
            children.extend(get_sub_environments() or [])
        except Exception:
            pass

    for attr in ("base_env", "env", "aec_env", "unwrapped", "gym_env", "par_env", "venv"):
        candidate = getattr(env, attr, None)
        if candidate is not None:
            children.append(candidate)

    for attr in ("envs", "vector_env"):
        candidate = getattr(env, attr, None)
        if isinstance(candidate, (list, tuple)):
            children.extend(item for item in candidate if item is not None)
        elif candidate is not None and candidate is not env:
            children.append(candidate)
    return children


def _resolve_base_env(env: Any) -> Any:
    queue = [env]
    visited = set()
    fallback = env
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        fallback = current
        if (
            hasattr(current, "finalize_episode_summary")
            or hasattr(current, "last_episode_summary")
            or hasattr(current, "metrics")
        ):
            return current
        queue.extend(_env_children(current))
    return fallback


def _completed_episode_summary(env: Any) -> Dict[str, Any]:
    base_env = _resolve_base_env(env)
    cached_summary = getattr(base_env, "last_episode_summary", None)
    if isinstance(cached_summary, dict) and cached_summary and not cached_summary.get("tripinfo/parse_pending"):
        return dict(cached_summary)

    if getattr(base_env, "sumo", None) is not None and getattr(base_env, "tripinfo_output_name", None):
        return {}

    if hasattr(base_env, "finalize_episode_summary"):
        try:
            summary = dict(base_env.finalize_episode_summary() or {})
        except Exception:
            summary = {}
        if summary and not summary.get("tripinfo/parse_pending"):
            return summary

    if isinstance(cached_summary, dict) and not cached_summary.get("tripinfo/parse_pending"):
        return dict(cached_summary)
    return {}


def _completed_episode_summary_history(env: Any) -> list[Dict[str, Any]]:
    base_env = _resolve_base_env(env)
    summaries = []
    for summary in getattr(base_env, "completed_episode_summaries", []) or []:
        if isinstance(summary, dict) and summary and not summary.get("tripinfo/parse_pending"):
            summaries.append(dict(summary))
    latest_summary = _completed_episode_summary(base_env)
    if latest_summary:
        latest_episode = latest_summary.get("episode/index")
        if not any(summary.get("episode/index") == latest_episode for summary in summaries):
            summaries.append(latest_summary)
    return summaries


@lru_cache(maxsize=1)
def training_episode_summary_callbacks_class():
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    def _resolve_callback_env(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        for key in ("base_env", "env", "env_runner"):
            candidate = kwargs.get(key)
            if candidate is not None:
                return candidate
        if args:
            first_arg = args[0]
            for attr in ("base_env", "env", "env_runner", "unwrapped"):
                candidate = getattr(first_arg, attr, None)
                if candidate is not None:
                    return candidate
            return first_arg
        return None

    class TrainingEpisodeSummaryCallbacks(DefaultCallbacks):
        """Capture completed episode summaries during RLlib training."""

        pending_episode_summaries: list[Dict[str, Any]] = []
        seen_episode_summaries: set[tuple[int, float]] = set()

        @classmethod
        def drain_pending_episode_summaries(cls) -> list[Dict[str, Any]]:
            summaries = list(cls.pending_episode_summaries)
            cls.pending_episode_summaries.clear()
            return summaries

        @classmethod
        def reset_episode_summary_tracking(cls) -> None:
            cls.pending_episode_summaries.clear()
            cls.seen_episode_summaries.clear()

        @classmethod
        def collect_completed_episode_summaries(cls, env: Any) -> None:
            base_env = _resolve_base_env(env)
            for summary in _completed_episode_summary_history(base_env):
                episode_index = summary.get("episode/index")
                if not isinstance(episode_index, (int, float, np.integer, np.floating)):
                    continue
                key = (id(base_env), float(episode_index))
                if key in cls.seen_episode_summaries:
                    continue
                cls.pending_episode_summaries.append(summary)
                cls.seen_episode_summaries.add(key)

        def on_episode_start(self, *args, **kwargs) -> None:
            env = _resolve_callback_env(args, kwargs)
            if env is not None:
                self.__class__.collect_completed_episode_summaries(env)

        def on_episode_end(self, *args, **kwargs) -> None:
            episode = kwargs.get("episode")
            if episode is None and args:
                episode = args[0]
            del episode
            env = _resolve_callback_env(args, kwargs)
            if env is not None:
                self.__class__.collect_completed_episode_summaries(env)

    return TrainingEpisodeSummaryCallbacks


def episode_seconds(cfg: Any) -> int:
    experiment = getattr(cfg, "experiment", None)
    value = getattr(experiment, "episode_seconds", 1)
    return max(1, int(value or 1))


def decision_interval_seconds(cfg: Any) -> int:
    env_cfg = getattr(cfg, "env", None)
    kwargs = getattr(env_cfg, "kwargs", None) if env_cfg is not None else None
    value = getattr(kwargs, "delta_time", None) if kwargs is not None else None
    if value is None and isinstance(kwargs, dict):
        value = kwargs.get("delta_time")
    return max(1, int(value or 5))


def episode_steps(cfg: Any) -> int:
    seconds = episode_seconds(cfg)
    delta_time = decision_interval_seconds(cfg)
    return max(1, seconds // delta_time)


def training_episode_target(cfg: Any) -> int:
    return max(1, int(getattr(getattr(cfg, "experiment", None), "episodes", 1) or 1))


def train_log_freq_steps(cfg: Any) -> int:
    logging_cfg = getattr(cfg, "logging", None)
    explicit = getattr(logging_cfg, "train_log_freq_steps", None) if logging_cfg is not None else None
    fallback = getattr(logging_cfg, "log_freq", 1000) if logging_cfg is not None else 1000
    return max(1, int(explicit if explicit is not None else fallback))


def train_log_freq_episodes(cfg: Any) -> int:
    logging_cfg = getattr(cfg, "logging", None)
    explicit = getattr(logging_cfg, "train_log_freq_episodes", None) if logging_cfg is not None else None
    if explicit is None and logging_cfg is not None:
        explicit = getattr(logging_cfg, "train_log_freq_steps", None)
    fallback = getattr(logging_cfg, "log_freq", 1000) if logging_cfg is not None else 1000
    return max(1, int(explicit if explicit is not None else fallback))


def validation_interval_steps(cfg: Any) -> int:
    experiment = getattr(cfg, "experiment", None)
    explicit = getattr(experiment, "validation_interval_steps", None) if experiment is not None else None
    logging_cfg = getattr(cfg, "logging", None)
    fallback = getattr(logging_cfg, "eval_freq", 0) if logging_cfg is not None else 0
    value = explicit if explicit is not None else fallback
    return max(0, int(value or 0))


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
        return build_rllib_parallel_env(cfg, run_dir, seed=seed, pad_spaces=pad_spaces)

    register_env(env_name, _creator)
    return env_name


def build_multi_agent_policies(cfg: Any, run_dir: Path, *, pad_spaces: bool):
    from ray.rllib.policy.policy import PolicySpec

    experiment = getattr(cfg, "experiment", None)
    sample_env = build_sumo_parallel_env(
        cfg,
        run_dir,
        seed=int(getattr(experiment, "seed", 0) or 0),
    )
    if pad_spaces:
        sample_env = _maybe_pad_pettingzoo_env(sample_env)
    try:
        policies = {}
        for agent_id in _possible_agents(sample_env):
            policies[str(agent_id)] = PolicySpec(
                observation_space=_agent_space(sample_env, str(agent_id), "observation"),
                action_space=_agent_space(sample_env, str(agent_id), "action"),
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
        episode_seconds=episode_seconds(cfg),
        episode_steps=episode_steps(cfg),
    )


def apply_env_runner_settings(config, params: Dict[str, Any]):
    num_env_runners = int(params.get("num_env_runners", params.get("num_rollout_workers", 0)) or 0)
    num_envs_per_runner = int(params.get("num_envs_per_env_runner", 1) or 1)
    rollout_fragment_length = params.get("rollout_fragment_length")
    if hasattr(config, "env_runners"):
        runner_kwargs = {
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_runner,
        }
        if rollout_fragment_length is not None:
            runner_kwargs["rollout_fragment_length"] = int(rollout_fragment_length)
        config = config.env_runners(**runner_kwargs)
    elif hasattr(config, "rollouts"):
        rollout_kwargs = {"num_rollout_workers": num_env_runners}
        if rollout_fragment_length is not None:
            rollout_kwargs["rollout_fragment_length"] = int(rollout_fragment_length)
        config = config.rollouts(**rollout_kwargs)
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
    episode_steps_value: int,
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
            episode_steps_value,
        )
    if "minibatch_size" in training_kwargs:
        training_kwargs["minibatch_size"] = cap_to_horizon(
            training_kwargs["minibatch_size"],
            int(training_kwargs.get("train_batch_size_per_learner", episode_steps_value)),
        )
    if training_kwargs and hasattr(config, "training"):
        config = config.training(**training_kwargs)
    if hasattr(config, "reporting"):
        config = config.reporting(min_sample_timesteps_per_iteration=episode_steps_value)
    return config


def apply_standard_evaluation_settings(config, params: Dict[str, Any]):
    """Apply RLlib AlgorithmConfig.evaluation(...) when explicitly configured."""

    evaluation_interval = params.get("evaluation_interval")
    if evaluation_interval in (None, 0, "0", False):
        return config

    evaluation_kwargs: Dict[str, Any] = {
        "evaluation_interval": int(evaluation_interval),
        "evaluation_duration": int(params.get("evaluation_duration", 1)),
        "evaluation_duration_unit": str(params.get("evaluation_duration_unit", "episodes")),
        "evaluation_config": dict(params.get("evaluation_config") or {"explore": False}),
    }
    for key in (
        "evaluation_parallel_to_training",
        "evaluation_force_reset_envs_before_iteration",
        "evaluation_sample_timeout_s",
    ):
        if key in params and params[key] is not None:
            evaluation_kwargs[key] = params[key]

    evaluation = getattr(config, "evaluation", None)
    if callable(evaluation):
        return evaluation(**evaluation_kwargs)
    return config


def apply_multi_agent_settings(config, context: RllibAlgorithmContext):
    return config.multi_agent(
        policies=context.active_policies,
        policy_mapping_fn=build_policy_mapping(context.policy_mode),
        policies_to_train=list(context.active_policies.keys()),
    )


def completed_training_episodes(metrics: Dict[str, Any], cfg: Any) -> int:
    reported_episodes = metrics.get("train/episodes_total")
    if reported_episodes is not None:
        return int(reported_episodes)
    sampled_steps = int(metrics.get("train/env_steps_sampled") or 0)
    return sampled_steps // episode_steps(cfg)


def training_should_stop(metrics: Dict[str, Any], cfg: Any) -> bool:
    target_episodes = training_episode_target(cfg)
    return completed_training_episodes(metrics, cfg) >= target_episodes


def should_log_training_metrics(
    metrics: Dict[str, Any],
    cfg: Any,
    *,
    last_logged_step: int,
    force: bool = False,
) -> bool:
    if force:
        return True
    sampled_steps = int(metrics.get("train/env_steps_sampled") or 0)
    return sampled_steps > 0 and sampled_steps - last_logged_step >= train_log_freq_steps(cfg)


def emit_training_metrics_by_step(
    metrics: Dict[str, Any],
    cfg: Any,
    *,
    last_logged_step: int,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]],
    force: bool = False,
) -> int:
    if emit_metrics is None:
        return last_logged_step

    current_step = int(metrics.get("train/env_step") or metrics.get("train/env_steps_sampled") or 0)
    if current_step <= 0:
        return last_logged_step

    freq = train_log_freq_steps(cfg)
    next_step = last_logged_step + freq
    logged_step = last_logged_step
    while next_step <= current_step:
        row = dict(metrics)
        row["train/env_step"] = float(next_step)
        emit_metrics(row, next_step)
        logged_step = next_step
        next_step += freq

    if force and logged_step != current_step:
        row = dict(metrics)
        row["train/env_step"] = float(current_step)
        emit_metrics(row, current_step)
        logged_step = current_step

    return logged_step


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
    if "train/env_steps_sampled" in metrics:
        metrics["train/env_step"] = metrics["train/env_steps_sampled"]
    if "train/episode_return_mean" in metrics:
        metrics.setdefault("train/reward_mean", metrics["train/episode_return_mean"])
    if "train/episode_return_min" in metrics:
        metrics.setdefault("train/reward_min", metrics["train/episode_return_min"])
    if "train/episode_return_max" in metrics:
        metrics.setdefault("train/reward_max", metrics["train/episode_return_max"])
    evaluation_metrics = result.get("evaluation")
    if isinstance(evaluation_metrics, dict):
        env_runner_eval = evaluation_metrics.get("env_runners")
        if isinstance(env_runner_eval, dict):
            for source_key, target_key in (
                ("episode_return_mean", "validation/rllib/episode_return_mean"),
                ("episode_return_min", "validation/rllib/episode_return_min"),
                ("episode_return_max", "validation/rllib/episode_return_max"),
                ("episode_len_mean", "validation/rllib/episode_len_mean"),
                ("num_env_steps_sampled_lifetime", "validation/rllib/env_steps_sampled"),
                ("num_agent_steps_sampled_lifetime", "validation/rllib/agent_steps_sampled"),
                ("num_episodes_lifetime", "validation/rllib/episodes_total"),
            ):
                value = env_runner_eval.get(source_key)
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    metrics[target_key] = float(value)
    return metrics


def build_training_episode_row(
    metrics: Dict[str, Any],
    episode_summary: Dict[str, Any],
    *,
    algorithm_kind: str,
) -> Dict[str, Any]:
    row = dict(metrics)
    row["algorithm/kind"] = algorithm_kind
    row["train/episode_summary_available"] = 1.0 if episode_summary else 0.0

    episode_index = episode_summary.get("episode/index")
    if isinstance(episode_index, (int, float, np.integer, np.floating)):
        row["train/episode_index"] = float(episode_index)
    else:
        fallback_episode = row.get("train/episodes_total")
        if isinstance(fallback_episode, (int, float, np.integer, np.floating)):
            row["train/episode_index"] = float(fallback_episode)

    reward_mean = row.get("train/reward_mean", row.get("train/episode_return_mean"))
    if isinstance(reward_mean, (int, float, np.integer, np.floating)) and not isinstance(reward_mean, bool):
        row["train/reward_mean"] = float(reward_mean)
        row["train/episode_reward"] = float(reward_mean)

    if isinstance(episode_summary, dict):
        for key, value in episode_summary.items():
            if key.startswith("resco_"):
                row_key = f"train/resco/{key[len('resco_'):]}"
            elif key.startswith("tripinfo/"):
                row_key = f"train/{key}"
            else:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                row[row_key] = float(value)
            else:
                row[row_key] = value
    return row


def emit_training_episode_rows(
    metrics: Dict[str, Any],
    episode_summaries: list[Dict[str, Any]],
    cfg: Any,
    *,
    algorithm_kind: str,
    last_logged_episode: int,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]],
    force: bool = False,
) -> int:
    if emit_metrics is None:
        return int(last_logged_episode)

    rows_by_step: Dict[int, Dict[str, Any]] = {}
    for episode_summary in episode_summaries:
        row = build_training_episode_row(metrics, episode_summary, algorithm_kind=algorithm_kind)
        row_step = int(row.get("train/episode_index") or row.get("train/episodes_total") or 0)
        if row_step > int(last_logged_episode):
            rows_by_step[row_step] = row

    completed_episodes = completed_training_episodes(metrics, cfg)
    for episode_index in range(int(last_logged_episode) + 1, completed_episodes + 1):
        if episode_index not in rows_by_step:
            row = build_training_episode_row(metrics, {}, algorithm_kind=algorithm_kind)
            row["train/episode_index"] = float(episode_index)
            rows_by_step[episode_index] = row

    current_last = int(last_logged_episode)
    for row_step in sorted(rows_by_step):
        if row_step <= current_last:
            continue
        if not should_log_training_episode(row_step, cfg, last_logged_episode=current_last, force=force):
            continue
        emit_metrics(rows_by_step[row_step], row_step)
        current_last = row_step

    if force and current_last == int(last_logged_episode) and not rows_by_step:
        row = build_training_episode_row(metrics, {}, algorithm_kind=algorithm_kind)
        row_step = int(row.get("train/episode_index") or row.get("train/episodes_total") or 0)
        if row_step > 0:
            emit_metrics(row, row_step)
            current_last = row_step

    return current_last


def emit_validation_if_due(
    metrics: Dict[str, Any],
    cfg: Any,
    *,
    last_validation_step: int,
    validate: Optional[Callable[[Dict[str, Any], int], None]],
) -> int:
    if validate is None:
        return int(last_validation_step)

    interval = validation_interval_steps(cfg)
    if interval <= 0:
        return int(last_validation_step)

    current_step = int(metrics.get("train/env_step") or metrics.get("train/env_steps_sampled") or 0)
    if current_step <= 0 or current_step - int(last_validation_step) < interval:
        return int(last_validation_step)

    validate(metrics, current_step)
    return current_step


def should_log_training_episode(episode_index: Any, cfg: Any, *, last_logged_episode: int = 0, force: bool = False) -> bool:
    if force:
        return True
    if not isinstance(episode_index, (int, float, np.integer, np.floating)):
        return False
    current_episode = int(episode_index)
    if current_episode <= 0:
        return False
    return current_episode - int(last_logged_episode) >= train_log_freq_episodes(cfg)


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
