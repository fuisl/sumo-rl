from __future__ import annotations

import csv
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from sumo_rl.agents.sb3 import JointMultiAgentActionWrapper, SB3WandbCallback
from sumo_rl.agents.sb3.evaluation import resolve_eval_seeds, run_model_episodes_on_seeds
from sumo_rl.experiments.metric_utils import (
    add_namespace_aliases as _metric_add_namespace_aliases,
    build_namespaced_metrics as _metric_build_namespaced_metrics,
    keep_namespaced_metrics as _metric_keep_namespaced_metrics,
    namespace_lane_fairness_metrics as _metric_namespace_lane_fairness_metrics,
    reward_formula_text as _metric_reward_formula_text,
)


def _as_plain_dict(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _as_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_dict(item) for item in value]
    return value


def _repo_root() -> Path:
    import sumo_rl

    return Path(sumo_rl.__file__).resolve().parent.parent


def _resolve_sumo_path(raw_path: Any) -> Any:
    if not isinstance(raw_path, str):
        return raw_path
    if raw_path.startswith("sumo_rl/"):
        return str(_repo_root() / raw_path)
    return raw_path


def _prepare_env_kwargs(cfg: DictConfig, run_dir: Path) -> Dict[str, Any]:
    kwargs = dict(_as_plain_dict(cfg.env.kwargs or {}))
    for key, value in list(kwargs.items()):
        if key.endswith("_file") or key in {"net_file", "route_file", "sumo_cfg_file"}:
            kwargs[key] = _resolve_sumo_path(value)

    if not kwargs.get("out_csv_name"):
        kwargs["out_csv_name"] = str(run_dir / "csv" / cfg.experiment.name)
    if not kwargs.get("tripinfo_output_name"):
        kwargs["tripinfo_output_name"] = str(run_dir / "tripinfo" / cfg.experiment.name)

    if "sumo_seed" not in kwargs and cfg.experiment.seed is not None:
        kwargs["sumo_seed"] = int(cfg.experiment.seed)

    return kwargs


def _get_run_seeds(cfg: DictConfig) -> list[int]:
    seeds = _as_plain_dict(getattr(cfg.experiment, "seeds", None))
    if isinstance(seeds, list) and seeds:
        return [int(seed) for seed in seeds]

    total_runs = int(getattr(cfg.experiment, "runs", 1))
    base_seed = int(cfg.experiment.seed) if cfg.experiment.seed is not None else 0
    return [base_seed for _ in range(total_runs)]


def _get_log_every_seconds(cfg: DictConfig) -> int:
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        return 200
    return max(1, int(getattr(logging_cfg, "log_every_seconds", 200)))


def _get_run_dir() -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / f"run_{timestamp}"


def _init_wandb(cfg: DictConfig, run_dir: Path):
    logging_cfg = cfg.logging
    if not logging_cfg.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Hydra logging is enabled but `wandb` is not installed. Install the `experiments` extra first."
        ) from exc

    run_name = logging_cfg.name or cfg.experiment.name
    project = logging_cfg.project or cfg.experiment.project
    group = logging_cfg.group or cfg.experiment.group
    tags = list(logging_cfg.tags or cfg.experiment.tags or [])

    return wandb.init(
        project=project,
        entity=logging_cfg.entity,
        name=run_name,
        group=group,
        tags=tags,
        job_type=logging_cfg.job_type,
        mode=logging_cfg.mode,
        dir=str(run_dir),
        config=_as_plain_dict(cfg),
        reinit="finish_previous",
    )


class _LocalMetricsCsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._fieldnames = ["timestamp", "step"]

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": step,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                row[key] = float(value)
            else:
                row[key] = str(value)

        self._rows.append(row)
        for key in row.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._flush()

    def _flush(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({key: row.get(key, "") for key in self._fieldnames})


def _logging_flag(logging_cfg, name: str, default: bool = False, *aliases: str) -> bool:
    for key in (name, *aliases):
        if logging_cfg is not None and hasattr(logging_cfg, key):
            return bool(getattr(logging_cfg, key))
    return default


def _prepare_row_for_csv(row: Dict[str, Any], logging_cfg) -> Dict[str, Any]:
    prepared = dict(row)
    if _logging_flag(logging_cfg, "log_namespace_aliases_to_csv", False, "add_namespace_aliases_to_csv"):
        prepared = _metric_add_namespace_aliases(prepared)
    return prepared


def _prepare_row_for_wandb(row: Dict[str, Any], logging_cfg, *, include_debug: bool = False, include_final: bool = True) -> Dict[str, Any]:
    if _logging_flag(logging_cfg, "log_namespace_aliases_to_wandb", False):
        prepared = _metric_add_namespace_aliases(dict(row))
    else:
        prepared = _metric_keep_namespaced_metrics(dict(row))
    prepared = {key: value for key, value in prepared.items() if not key.startswith("tripinfo/")}
    if not include_debug:
        prepared = {key: value for key, value in prepared.items() if not key.startswith("debug/")}
    if not include_final:
        prepared = {
            key: value
            for key, value in prepared.items()
            if not (key.startswith("final/") or key.startswith("tripinfo/") or key.startswith("warnings/"))
        }
    return prepared


def _namespace_lane_fairness_metrics(env) -> Dict[str, float]:
    return _metric_namespace_lane_fairness_metrics(_get_base_env(env))


def _reward_formula_text(reward_fn: Any, reward_weights: Any = None) -> str:
    return _metric_reward_formula_text(reward_fn, reward_weights)


def _reward_metadata_from_env(env) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    reward_fn = getattr(base_env, "reward_fn", "diff-waiting-time")
    reward_weights = getattr(base_env, "reward_weights", None)
    reward_name = None

    if isinstance(reward_fn, list):
        reward_name = "composite_reward"
    elif isinstance(reward_fn, dict):
        reward_name = "per_signal_reward_map"
    elif callable(reward_fn):
        reward_name = getattr(reward_fn, "__name__", "custom_reward_fn")
    else:
        reward_name = str(reward_fn)

    return {
        "reward/name": reward_name,
        "reward/formula": _reward_formula_text(reward_fn, reward_weights),
        "reward/source": "sumo_rl.environment.traffic_signal.TrafficSignal.compute_reward",
        "reward/scope": "per-agent environment reward",
    }


def _prefix_metric_keys(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _add_namespace_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    return _metric_add_namespace_aliases(row)


def _build_namespaced_metrics(info: Any, include_agent_metrics_local: bool = False) -> tuple[Dict[str, float], Dict[str, float]]:
    return _metric_build_namespaced_metrics(info, include_agent_metrics_local)


def _log_resco_metrics(
    wandb_run,
    csv_run,
    shared_metrics: Dict[str, Any],
    info: Any,
    step: Optional[int] = None,
    include_agent_metrics_local: bool = False,
) -> None:
    namespaced_metrics, agent_metrics = _build_namespaced_metrics(info, include_agent_metrics_local)
    wandb_metrics = dict(shared_metrics)
    wandb_metrics.update(namespaced_metrics)
    csv_metrics = dict(wandb_metrics)
    csv_metrics.update(agent_metrics)

    if wandb_run is not None:
        wandb_run.log(wandb_metrics, step=step)
    if csv_run is not None:
        csv_run.log(csv_metrics, step=step)


def _log_outputs(wandb_run, csv_run, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if (wandb_run is None and csv_run is None) or not metrics:
        return
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    if csv_run is not None:
        csv_run.log(metrics, step=step)


def _build_env(cfg: DictConfig, run_dir: Path, seed: Optional[int] = None):
    import sumo_rl
    from sumo_rl import SumoEnvironment

    kwargs = _prepare_env_kwargs(cfg, run_dir)
    if seed is not None:
        kwargs["sumo_seed"] = int(seed)
    factory = cfg.env.factory

    if factory == "sumo_env":
        return SumoEnvironment(**kwargs)
    if factory == "env":
        return sumo_rl.env(**kwargs)
    if factory == "parallel_env":
        return sumo_rl.parallel_env(**kwargs)
    if factory == "fixed_time_env":
        kwargs["fixed_ts"] = True
        return sumo_rl.env(**kwargs)
    if factory == "grid4x4":
        return sumo_rl.grid4x4(**kwargs)
    if factory == "arterial4x4":
        return sumo_rl.arterial4x4(**kwargs)
    if factory == "cologne1":
        return sumo_rl.cologne1(**kwargs)
    if factory == "cologne3":
        return sumo_rl.cologne3(**kwargs)
    if factory == "cologne8":
        return sumo_rl.cologne8(**kwargs)
    if factory == "ingolstadt1":
        return sumo_rl.ingolstadt1(**kwargs)
    if factory == "ingolstadt7":
        return sumo_rl.ingolstadt7(**kwargs)
    if factory == "ingolstadt21":
        return sumo_rl.ingolstadt21(**kwargs)

    raise ValueError(f"Unsupported env factory: {factory}")


def _get_final_info(env):
    return _get_completed_episode_final_info(env)


def _get_env_step(env) -> int:
    base_env = _get_base_env(env)
    if hasattr(base_env, "sim_step"):
        return int(base_env.sim_step)
    return 0


def _get_sb3_final_log_step(cfg: DictConfig, model: Any) -> int:
    requested_step = int(cfg.experiment.total_timesteps)
    actual_step = int(getattr(model, "num_timesteps", requested_step) or requested_step)
    return max(requested_step, actual_step)


def _get_sb3_eval_seeds(cfg: DictConfig) -> list[int]:
    eval_episodes = int(getattr(cfg.experiment, "eval_episodes", 0) or 0)
    explicit_eval_seeds = _as_plain_dict(getattr(cfg.experiment, "eval_seeds", None))
    if isinstance(explicit_eval_seeds, list):
        explicit_iterable = explicit_eval_seeds
    elif explicit_eval_seeds is None:
        explicit_iterable = None
    else:
        explicit_iterable = [explicit_eval_seeds]
    base_seed = int(cfg.experiment.seed) if cfg.experiment.seed is not None else None
    return resolve_eval_seeds(base_seed, eval_episodes, explicit_iterable)


def _get_sb3_checkpoint_dir(run_dir: Path, cfg: DictConfig) -> Path:
    return run_dir / "checkpoints" / str(cfg.algorithm.kind)


def _get_base_env(env):
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
    return env


def _aggregate_final_eval_rows(
    rows: list[Dict[str, Any]],
    *,
    algorithm_kind: str,
    eval_mean_reward: float,
    eval_std_reward: float,
    eval_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {
        "algorithm/kind": algorithm_kind,
        "final/eval/mean_reward": float(eval_mean_reward),
        "final/eval/std_reward": float(eval_std_reward),
    }

    if not rows:
        aggregated.update(
            {
                "warnings/no_finished_trips": True,
                "warnings/no_departed_vehicles": True,
                "warnings/no_arrived_vehicles": True,
                "warnings/no_final_summary_metrics": True,
                "warnings/eval_episodes_too_low": bool(eval_episodes is not None and int(eval_episodes) <= 1),
                "warnings/all_zero_traffic_metrics": True,
            }
        )
        return aggregated

    numeric_values: dict[str, list[float]] = defaultdict(list)
    warning_values: dict[str, bool] = {}
    passthrough_keys = {
        "final/reward/name",
        "final/reward/formula",
        "final/reward/scope",
        "debug/base_env_class",
    }

    for row in rows:
        for key, value in row.items():
            if key in {"algorithm/kind", "final/eval/mean_reward", "final/eval/std_reward"}:
                continue
            if key.startswith("warnings/"):
                warning_values[key] = warning_values.get(key, False) or bool(value)
                continue
            if key in passthrough_keys:
                aggregated.setdefault(key, value)
                continue
            if isinstance(value, (bool, int, float, np.integer, np.floating)):
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    numeric_values[key].append(numeric_value)
                elif key not in aggregated:
                    aggregated[key] = float("nan")
            elif key.startswith("debug/"):
                aggregated.setdefault(key, value)

    for key, values in numeric_values.items():
        aggregated[key] = float(np.mean(values)) if values else float("nan")

    aggregated.update(warning_values)
    aggregated["warnings/eval_episodes_too_low"] = bool(eval_episodes is not None and int(eval_episodes) <= 1)
    return aggregated


def _run_sb3_final_evaluation(
    model: Any,
    eval_env: Any,
    *,
    algorithm_kind: str,
    eval_seeds: list[int],
    eval_episodes: Optional[int] = None,
    logging_cfg=None,
) -> tuple[float, float, Dict[str, Any]]:
    seed_rows: list[Dict[str, Any]] = []

    def _capture_seed_row(seed: int, episode_reward: float) -> None:
        summary_env = _get_base_env(eval_env)
        seed_rows.append(
            _build_final_eval_summary_row(
                summary_env,
                algorithm_kind=algorithm_kind,
                eval_mean_reward=float(episode_reward),
                eval_std_reward=0.0,
                eval_episodes=1,
                logging_cfg=logging_cfg,
            )
        )

    episode_rewards = run_model_episodes_on_seeds(model, eval_env, eval_seeds, on_episode_end=_capture_seed_row)
    if episode_rewards:
        eval_mean_reward = float(np.mean(episode_rewards))
        eval_std_reward = float(np.std(episode_rewards))
    else:
        eval_mean_reward = 0.0
        eval_std_reward = 0.0

    final_summary = _aggregate_final_eval_rows(
        seed_rows,
        algorithm_kind=algorithm_kind,
        eval_mean_reward=eval_mean_reward,
        eval_std_reward=eval_std_reward,
        eval_episodes=eval_episodes,
    )
    return eval_mean_reward, eval_std_reward, final_summary


def _episode_index_from_summary(summary: Any) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    value = summary.get("episode/index")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_episode_index(base_env: Any) -> Optional[float]:
    if not hasattr(base_env, "episode"):
        return None
    try:
        return float(getattr(base_env, "episode"))
    except (TypeError, ValueError):
        return None


def _should_prefer_cached_episode(base_env: Any) -> bool:
    cached_summary = getattr(base_env, "last_episode_summary", None)
    cached_episode = _episode_index_from_summary(cached_summary)
    current_episode = _current_episode_index(base_env)
    if cached_episode is None:
        return False
    if current_episode is None:
        return True
    return cached_episode != current_episode


def _get_completed_episode_summary(env) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    cached_summary = getattr(base_env, "last_episode_summary", None)
    if isinstance(cached_summary, dict) and cached_summary:
        if _should_prefer_cached_episode(base_env) or not getattr(base_env, "metrics", None):
            return dict(cached_summary)

    if hasattr(base_env, "finalize_episode_summary"):
        summary = dict(base_env.finalize_episode_summary() or {})
        if summary:
            return summary

    if isinstance(cached_summary, dict):
        return dict(cached_summary)
    return {}


def _get_completed_episode_final_info(env) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    cached_info = getattr(base_env, "last_episode_final_info", None)
    if isinstance(cached_info, dict) and cached_info and _should_prefer_cached_episode(base_env):
        return dict(cached_info)
    if getattr(base_env, "metrics", None):
        return dict(base_env.metrics[-1])
    if isinstance(cached_info, dict):
        return dict(cached_info)
    return {}


def _wrap_sb3_env_if_needed(cfg: DictConfig, env, params: Dict[str, Any], default_num_envs: int):
    num_envs = int(params.pop("num_envs", default_num_envs))
    if cfg.env.factory != "parallel_env" and not hasattr(env, "possible_agents"):
        return env, 1

    import supersuit as ss
    from stable_baselines3.common.vec_env import VecMonitor

    # RESCO scenarios such as cologne3 and ingolstadt7 have heterogeneous agent
    # observation/action spaces, so pad them before converting to a VecEnv.
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_envs,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    def _seed_shim(seed=None):
        if hasattr(env, "venv") and hasattr(env.venv, "seed"):
            try:
                return env.venv.seed(seed)
            except AttributeError:
                return None
        return None

    if not hasattr(env, "seed"):
        env.seed = _seed_shim
    if not hasattr(env, "render_mode"):
        env.render_mode = None
    env = VecMonitor(env)
    env.seed = _seed_shim
    return env, num_envs


def _get_episode_summary(env) -> Dict[str, Any]:
    return _get_completed_episode_summary(env)


def _build_episode_summary_row(
    env,
    run_idx: int,
    episode_idx: int,
    run_seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = _get_episode_summary(env)
    row: Dict[str, Any] = {
        "run/index": run_idx,
        "episode/index": episode_idx,
    }
    if run_seed is not None:
        row["run_seed"] = float(run_seed)
    if extra:
        row.update(extra)
    row.update(summary)
    return row


def _build_resco_summary_row(env, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    summary = _get_completed_episode_summary(base_env)
    row = {
        key: value
        for key, value in summary.items()
        if key.startswith("resco_")
        or key.startswith("tripinfo/")
        or key in {"sim_step", "episode/index", "episode/steps", "episode/sim_time_abs", "episode/elapsed_seconds"}
    }
    namespaced_metrics, agent_metrics = _build_namespaced_metrics(_get_final_info(env), include_agent_metrics_local=False)
    row.update(namespaced_metrics)
    row.update(agent_metrics)
    row.update(_namespace_lane_fairness_metrics(base_env))
    row.update(_reward_metadata_from_env(base_env))
    if extra:
        row.update(extra)
    return row


def _build_final_eval_summary_row(
    env,
    *,
    algorithm_kind: str,
    eval_mean_reward: float,
    eval_std_reward: float,
    eval_episodes: Optional[int] = None,
    logging_cfg=None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    summary = _get_completed_episode_summary(base_env)

    final_row: Dict[str, Any] = {
        "algorithm/kind": algorithm_kind,
        "final/eval/mean_reward": float(eval_mean_reward),
        "final/eval/std_reward": float(eval_std_reward),
        "episode/sim_time_abs": float(summary.get("episode/sim_time_abs", summary.get("episode/steps", 0.0))),
        "episode/elapsed_seconds": float(summary.get("episode/elapsed_seconds", 0.0)),
    }
    reward_metadata = _reward_metadata_from_env(base_env)
    traffic_metrics, _ = _build_namespaced_metrics(_get_final_info(base_env), include_agent_metrics_local=False)

    if _logging_flag(logging_cfg, "log_final_traffic_metrics", True):
        final_row.update(
            {
                "final/resco/avg_delay": float(summary.get("resco_avg_delay", float("nan"))),
                "final/resco/wait": float(summary.get("resco_wait", float("nan"))),
                "final/resco/queue": float(summary.get("resco_queue", float("nan"))),
                "final/resco/trip_time": float(summary.get("resco_trip_time", float("nan"))),
                "tripinfo/finished_count": float(summary.get("tripinfo/finished_count", float("nan"))),
                "tripinfo/unfinished_count": float(summary.get("tripinfo/unfinished_count", float("nan"))),
                "tripinfo/total_count": float(summary.get("tripinfo/total_count", float("nan"))),
                "tripinfo/avg_duration": float(summary.get("tripinfo/avg_duration", float("nan"))),
                "tripinfo/avg_waiting_time": float(summary.get("tripinfo/avg_waiting_time", float("nan"))),
                "tripinfo/avg_time_loss": float(summary.get("tripinfo/avg_time_loss", float("nan"))),
                "final/efficiency/total_arrived": float(traffic_metrics.get("efficiency_total_arrived", float("nan"))),
                "final/efficiency/total_departed": float(traffic_metrics.get("efficiency_total_departed", float("nan"))),
                "final/efficiency/total_running": float(traffic_metrics.get("efficiency_total_running", float("nan"))),
                "final/fairness/jain_waiting_time": float(traffic_metrics.get("fairness_jain_waiting_time", float("nan"))),
                "final/safety/total_teleported": float(traffic_metrics.get("safety_total_teleported", float("nan"))),
                "final/safety/total_emergency_brake": float(traffic_metrics.get("safety_total_emergency_brake", float("nan"))),
            }
        )

    final_row["final/reward/name"] = reward_metadata["reward/name"]
    final_row["final/reward/formula"] = reward_metadata["reward/formula"]
    final_row["final/reward/scope"] = reward_metadata["reward/scope"]

    final_metric_keys = [
        key
        for key in final_row
        if key.startswith("final/resco/")
        or key.startswith("final/efficiency/")
        or key.startswith("final/fairness/")
        or key.startswith("final/safety/")
    ]
    has_final_summary_metrics = any(np.isfinite(float(final_row[key])) for key in final_metric_keys if isinstance(final_row[key], (int, float, np.integer, np.floating)))
    warnings = {
        "warnings/no_finished_trips": bool(float(summary.get("tripinfo/finished_count", 0.0)) == 0.0),
        "warnings/no_departed_vehicles": bool(float(final_row.get("final/efficiency/total_departed", 0.0)) == 0.0),
        "warnings/no_arrived_vehicles": bool(float(final_row.get("final/efficiency/total_arrived", 0.0)) == 0.0),
        "warnings/no_final_summary_metrics": not has_final_summary_metrics,
        "warnings/eval_episodes_too_low": bool(eval_episodes is not None and int(eval_episodes) <= 1),
    }
    all_zero_candidates = [
        final_row.get("final/resco/queue", summary.get("resco_queue", float("nan"))),
        final_row.get("final/resco/wait", summary.get("resco_wait", float("nan"))),
        final_row.get("tripinfo/avg_waiting_time", summary.get("tripinfo/avg_waiting_time", float("nan"))),
    ]
    warnings["warnings/all_zero_traffic_metrics"] = all(
        not np.isfinite(float(value)) or float(value) == 0.0 for value in all_zero_candidates
    )
    final_row.update(warnings)

    if extra:
        final_row.update(extra)
    if _logging_flag(logging_cfg, "debug_metrics", False):
        metrics = getattr(base_env, "metrics", None)
        final_row.update(
            {
                "debug/base_env_class": f"{base_env.__class__.__module__}.{base_env.__class__.__name__}",
                "debug/has_metrics": bool(metrics),
                "debug/metrics_len": float(len(metrics) if isinstance(metrics, list) else 0),
                "debug/has_last_episode_final_info": bool(getattr(base_env, "last_episode_final_info", None)),
                "debug/has_finalize_episode_summary": bool(hasattr(base_env, "finalize_episode_summary")),
                "debug/sim_step": float(getattr(base_env, "sim_step", summary.get("episode/sim_time_abs", 0.0))),
                "debug/num_seconds": float(getattr(base_env, "num_seconds", getattr(base_env, "sim_max_time", 0.0))),
            }
        )
        final_row["debug/final_summary_key_count"] = float(len(final_row) + 1)
    return final_row


def _update_wandb_summary(wandb_run, metrics: Dict[str, Any]) -> None:
    if wandb_run is None or not metrics:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            wandb_run.summary[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            wandb_run.summary[key] = float(value)


def _log_episode_summary(
    wandb_run,
    csv_run,
    row: Dict[str, Any],
    step: Optional[int] = None,
    logging_cfg=None,
    *,
    include_debug_to_wandb: bool = False,
    include_final_to_wandb: bool = True,
) -> None:
    csv_row = _prepare_row_for_csv(row, logging_cfg)
    wandb_row = _prepare_row_for_wandb(
        row,
        logging_cfg,
        include_debug=include_debug_to_wandb,
        include_final=include_final_to_wandb,
    )
    if wandb_run is not None:
        wandb_run.log(wandb_row, step=step)
        _update_wandb_summary(wandb_run, wandb_row)
    if csv_run is not None:
        csv_run.log(csv_row, step=step)


def _log_final_summary_debug(
    episode_summary: Dict[str, Any],
    logging_cfg,
    *,
    algorithm_kind: Optional[str] = None,
    eval_env=None,
    summary_env=None,
) -> None:
    if _logging_flag(logging_cfg, "debug_metrics", False):
        if algorithm_kind is not None:
            print(f"[{algorithm_kind}] eval_env class: {eval_env.__class__.__name__ if eval_env is not None else 'None'}")
            print(f"[{algorithm_kind}] summary_env class: {summary_env.__class__.__name__ if summary_env is not None else 'None'}")
        print(f"Final summary contains {len(episode_summary)} keys")
        print(f"Final summary keys: {sorted(episode_summary.keys())}")


def _aggregate_numeric_rows(
    rows: list[Dict[str, Any]],
    prefix: str = "summary",
    include_agent_metrics: bool = False,
) -> Dict[str, float]:
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key in {"run/index", "episode/index", "seed", "run_seed"}:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not include_agent_metrics and (key.startswith("system_") or key.startswith("efficiency_") or key.startswith("safety_")):
                    continue
                values[key].append(float(value))

    summary = {f"{prefix}/run_count": float(len(rows))}
    for key, series in values.items():
        summary[f"{prefix}/{key}"] = float(np.mean(series))
    return summary


def _aggregate_numeric_row_values(rows: list[Dict[str, Any]], prefix: str = "summary") -> Dict[str, float]:
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key in {"run/index", "episode/index", "seed", "run_seed"}:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                values[key].append(float(value))

    summary = {f"{prefix}/run_count": float(len(rows))}
    for key, series in values.items():
        summary[f"{prefix}/{key}"] = float(np.mean(series))
    return summary


def _build_ql_agents_direct(env, cfg: DictConfig, initial_states: Dict[str, Any]):
    from sumo_rl.agents import QLAgent
    from sumo_rl.exploration import EpsilonGreedy

    params = _as_plain_dict(cfg.algorithm.params or {})
    exploration = EpsilonGreedy(
        initial_epsilon=float(params.get("epsilon", 0.05)),
        min_epsilon=float(params.get("min_epsilon", 0.005)),
        decay=float(params.get("decay", 1.0)),
    )

    return {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=float(params.get("alpha", 0.1)),
            gamma=float(params.get("gamma", 0.99)),
            exploration_strategy=exploration,
        )
        for ts in env.ts_ids
    }


def _build_ql_agents_aec(env, cfg: DictConfig):
    from sumo_rl.agents import QLAgent
    from sumo_rl.exploration import EpsilonGreedy

    params = _as_plain_dict(cfg.algorithm.params or {})
    exploration = EpsilonGreedy(
        initial_epsilon=float(params.get("epsilon", 0.05)),
        min_epsilon=float(params.get("min_epsilon", 0.005)),
        decay=float(params.get("decay", 1.0)),
    )

    return {
        ts: QLAgent(
            starting_state=env.unwrapped.env.encode(env.observe(ts), ts),
            state_space=env.observation_space(ts),
            action_space=env.action_space(ts),
            alpha=float(params.get("alpha", 0.1)),
            gamma=float(params.get("gamma", 0.99)),
            exploration_strategy=exploration,
        )
        for ts in env.agents
    }


def _run_direct_q_learning(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    env = _build_env(cfg, run_dir)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_runs = int(cfg.experiment.runs)
    total_episodes = int(cfg.experiment.episodes)
    fixed_ts = bool(cfg.experiment.fixed_ts)

    try:
        run_metrics: list[Dict[str, Any]] = []
        for run_idx in range(1, total_runs + 1):
            initial_states = env.reset()
            agents = _build_ql_agents_direct(env, cfg, initial_states)
            previous_episode_reward = 0.0
            previous_td_error_mean = 0.0
            previous_td_error_abs_mean = 0.0

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    previous_summary = _build_resco_summary_row(env, extra={"episode/reward": previous_episode_reward})
                    previous_summary["train/episode_reward"] = previous_episode_reward
                    previous_summary["train/td_error_mean"] = previous_td_error_mean
                    previous_summary["train/td_error_abs_mean"] = previous_td_error_abs_mean
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        previous_summary,
                        step=int(previous_summary.get("episode/steps", _get_env_step(env))),
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(previous_summary)
                    initial_states = env.reset()
                    for agent_id in initial_states.keys():
                        agents[agent_id].state = env.encode(initial_states[agent_id], agent_id)

                episode_reward = 0.0
                episode_td_errors: list[float] = []
                done = {"__all__": False}
                info = {}

                if fixed_ts:
                    while not done["__all__"]:
                        _, _, done, info = env.step({})
                else:
                    while not done["__all__"]:
                        actions = {ts: agents[ts].act() for ts in agents.keys()}
                        next_state, reward, done, info = env.step(action=actions)
                        episode_reward += float(sum(reward.values()))

                        for agent_id in agents.keys():
                            td_error = agents[agent_id].learn(
                                next_state=env.encode(next_state[agent_id], agent_id),
                                reward=reward[agent_id],
                            )
                            episode_td_errors.append(float(td_error))

                previous_episode_reward = episode_reward
                previous_td_error_mean = float(np.mean(episode_td_errors)) if episode_td_errors else 0.0
                previous_td_error_abs_mean = float(np.mean(np.abs(episode_td_errors))) if episode_td_errors else 0.0
                env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    env.close()
                    episode_summary = _build_final_eval_summary_row(
                        env,
                        algorithm_kind="q_learning",
                        eval_mean_reward=previous_episode_reward,
                        eval_std_reward=0.0,
                        logging_cfg=cfg.logging,
                        extra={
                            "train/episode_reward": previous_episode_reward,
                            "train/td_error_mean": previous_td_error_mean,
                            "train/td_error_abs_mean": previous_td_error_abs_mean,
                        },
                    )
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=int(episode_summary.get("episode/steps", _get_env_step(env))),
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(episode_summary)

        summary = _aggregate_numeric_rows(run_metrics)
        summary["algorithm/kind"] = "q_learning"
        if wandb_run is not None:
            wandb_run.log(summary, step=len(run_metrics))
            _update_wandb_summary(wandb_run, summary)
        if csv_run is not None:
            csv_run.log(summary, step=len(run_metrics))
    finally:
        env.close()


def _run_aec_q_learning(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    env = _build_env(cfg, run_dir)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_runs = int(cfg.experiment.runs)
    total_episodes = int(cfg.experiment.episodes)
    fixed_ts = bool(cfg.experiment.fixed_ts)

    try:
        run_metrics: list[Dict[str, Any]] = []
        for run_idx in range(1, total_runs + 1):
            env.reset()
            agents = _build_ql_agents_aec(env, cfg)
            previous_episode_reward = 0.0
            previous_td_error_mean = 0.0
            previous_td_error_abs_mean = 0.0

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    previous_summary = _build_resco_summary_row(env, extra={"episode/reward": previous_episode_reward})
                    previous_summary["train/episode_reward"] = previous_episode_reward
                    previous_summary["train/td_error_mean"] = previous_td_error_mean
                    previous_summary["train/td_error_abs_mean"] = previous_td_error_abs_mean
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        previous_summary,
                        step=int(previous_summary.get("episode/steps", _get_env_step(env))),
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(previous_summary)
                    env.reset()
                    for agent_id in env.agents:
                        agents[agent_id].state = env.unwrapped.env.encode(env.observe(agent_id), agent_id)

                episode_reward = 0.0
                episode_td_errors: list[float] = []

                if fixed_ts:
                    while env.agents:
                        env.step(None)
                else:
                    for agent in env.agent_iter():
                        observation, reward, terminated, truncated, _ = env.last()
                        done = terminated or truncated
                        if agents[agent].action is not None:
                            td_error = agents[agent].learn(
                                next_state=env.unwrapped.env.encode(observation, agent),
                                reward=reward,
                            )
                            episode_td_errors.append(float(td_error))

                        action = agents[agent].act() if not done else None
                        env.step(action)
                        episode_reward += float(reward)

                previous_episode_reward = episode_reward
                previous_td_error_mean = float(np.mean(episode_td_errors)) if episode_td_errors else 0.0
                previous_td_error_abs_mean = float(np.mean(np.abs(episode_td_errors))) if episode_td_errors else 0.0
                env.unwrapped.env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    env.close()
                    episode_summary = _build_final_eval_summary_row(
                        env,
                        algorithm_kind="q_learning",
                        eval_mean_reward=previous_episode_reward,
                        eval_std_reward=0.0,
                        logging_cfg=cfg.logging,
                        extra={
                            "train/episode_reward": previous_episode_reward,
                            "train/td_error_mean": previous_td_error_mean,
                            "train/td_error_abs_mean": previous_td_error_abs_mean,
                        },
                    )
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=int(episode_summary.get("episode/steps", _get_env_step(env))),
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(episode_summary)

        summary = _aggregate_numeric_rows(run_metrics)
        summary["algorithm/kind"] = "q_learning"
        if wandb_run is not None:
            wandb_run.log(summary, step=len(run_metrics))
            _update_wandb_summary(wandb_run, summary)
        if csv_run is not None:
            csv_run.log(summary, step=len(run_metrics))
    finally:
        env.close()


def _run_fixed_time(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_episodes = int(cfg.experiment.episodes)
    seeds = _get_run_seeds(cfg)
    run_metrics: list[Dict[str, Any]] = []
    final_step = 0

    for run_idx, seed in enumerate(seeds, start=1):
        env = _build_env(cfg, run_dir, seed=seed)
        try:
            for episode_idx in range(1, total_episodes + 1):
                base_env = _get_base_env(env)
                if hasattr(env, "agent_iter"):
                    env.reset(seed=seed)
                    for agent in env.agent_iter():
                        _obs, _reward, terminated, truncated, _info = env.last()
                        done = bool(terminated or truncated)
                        action = None if done else env.action_space(agent).sample()
                        env.step(action)
                else:
                    reset_result = env.reset(seed=seed)
                    if isinstance(reset_result, tuple):
                        _obs, _info = reset_result
                    done = False
                    while not done:
                        actions = {ts_id: base_env.action_spaces(ts_id).sample() for ts_id in base_env.ts_ids}
                        next_step = env.step(actions)
                        if len(next_step) == 5:
                            _obs, reward, terminated, truncated, info = next_step
                            done = bool(terminated or truncated)
                        else:
                            _obs, reward, dones, info = next_step
                            done = bool(dones["__all__"])
                save_env = base_env if hasattr(base_env, "save_csv") else env
                save_env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    last_step = _get_env_step(base_env)
                    final_step = max(final_step, last_step)
                    save_env.close()
                    episode_summary = _build_resco_summary_row(base_env, extra={"static/policy": "fixed_time"})
                    row_step = run_idx
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=row_step,
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(episode_summary)
        finally:
            env.close()

    wandb_summary = _aggregate_numeric_row_values(run_metrics)
    csv_summary = _aggregate_numeric_row_values(run_metrics)
    wandb_summary["static/policy"] = "fixed_time"
    csv_summary["static/policy"] = "fixed_time"
    if wandb_run is not None:
        wandb_run.log(wandb_summary, step=final_step or None)
        _update_wandb_summary(wandb_run, wandb_summary)
    if csv_run is not None:
        csv_run.log(csv_summary, step=final_step or None)


def _run_static_policy(cfg: DictConfig, run_dir: Path, wandb_run, csv_run, policy_name: str) -> None:
    from sumo_rl.agents.static import GreedyPolicy, MaxPressurePolicy

    policy = MaxPressurePolicy() if policy_name == "max_pressure" else GreedyPolicy()
    seeds = _get_run_seeds(cfg)
    total_episodes = int(cfg.experiment.episodes)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    run_metrics: list[Dict[str, Any]] = []
    final_step = 0

    for run_idx, seed in enumerate(seeds, start=1):
        env = _build_env(cfg, run_dir, seed=seed)
        try:
            for episode_idx in range(1, total_episodes + 1):
                reset_result = env.reset(seed=seed)
                if isinstance(reset_result, tuple):
                    _obs = reset_result[0]

                base_env = _get_base_env(env)
                if hasattr(env, "agent_iter"):
                    done = {"__all__": False}
                    info: Dict[str, Any] = {}

                    while not done["__all__"]:
                        actions = {
                            ts_id: policy.select_action(base_env.traffic_signals[ts_id]) for ts_id in base_env.ts_ids
                        }
                        _, reward, done, info = env.step(action=actions)
                else:
                    done = False
                    while not done:
                        actions = {
                            ts_id: policy.select_action(base_env.traffic_signals[ts_id]) for ts_id in base_env.ts_ids
                        }
                        next_step = env.step(actions)
                        if len(next_step) == 5:
                            _obs, reward, terminated, truncated, info = next_step
                            if isinstance(terminated, dict):
                                terminated_all = bool(terminated.get("__all__", False))
                                truncated_all = bool(truncated.get("__all__", False))
                                done = terminated_all or truncated_all
                            else:
                                done = bool(terminated or truncated)
                        else:
                            _obs, reward, dones, info = next_step
                            done = bool(dones["__all__"])
                save_env = base_env if hasattr(base_env, "save_csv") else env
                save_env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    last_step = _get_env_step(base_env)
                    final_step = max(final_step, last_step)
                    save_env.close()
                    episode_summary = _build_resco_summary_row(base_env, extra={"static/policy": policy_name})
                    row_step = run_idx
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=row_step,
                        logging_cfg=cfg.logging,
                    )
                    run_metrics.append(episode_summary)
        finally:
            env.close()

    wandb_summary = _aggregate_numeric_row_values(run_metrics)
    csv_summary = _aggregate_numeric_row_values(run_metrics)
    wandb_summary["static/policy"] = policy_name
    csv_summary["static/policy"] = policy_name
    if wandb_run is not None:
        wandb_run.log(wandb_summary, step=final_step or None)
        _update_wandb_summary(wandb_run, wandb_summary)
    if csv_run is not None:
        csv_run.log(csv_summary, step=final_step or None)


def _run_sb3_dqn(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from stable_baselines3 import DQN

    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    eval_seeds = _get_sb3_eval_seeds(cfg)
    log_freq = int(getattr(cfg.logging, "log_freq", 1000))
    eval_freq = int(getattr(cfg.logging, "eval_freq", log_freq))
    env = _build_env(cfg, run_dir)

    try:
        env, _ = _wrap_sb3_env_if_needed(cfg, env, params, default_num_envs=1)
        eval_env = _build_env(cfg, run_dir)
        eval_params = _as_plain_dict(cfg.algorithm.params or {})
        eval_params.pop("num_envs", None)
        eval_env, _ = _wrap_sb3_env_if_needed(cfg, eval_env, eval_params, default_num_envs=1)
        callback = SB3WandbCallback(
            wandb_run,
            csv_run,
            logging_cfg=cfg.logging,
            log_freq=log_freq,
            eval_env=eval_env,
            eval_episodes=eval_episodes,
            eval_freq=eval_freq,
            eval_seeds=eval_seeds,
            checkpoint_dir=_get_sb3_checkpoint_dir(run_dir, cfg),
            checkpoint_freq=int(getattr(cfg.logging, "checkpoint_freq", 0)),
            save_checkpoints=_logging_flag(cfg.logging, "save_checkpoints", False),
            save_final_model=_logging_flag(cfg.logging, "save_final_model", True),
        ).build()
        model = DQN(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            seed=int(cfg.experiment.seed) if cfg.experiment.seed is not None else None,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps), callback=callback)
        final_log_step = _get_sb3_final_log_step(cfg, model)

        try:
            mean_reward, std_reward, episode_summary = _run_sb3_final_evaluation(
                model,
                eval_env,
                algorithm_kind="dqn_sb3",
                eval_seeds=eval_seeds,
                eval_episodes=eval_episodes,
                logging_cfg=cfg.logging,
            )
            _log_final_summary_debug(
                episode_summary,
                cfg.logging,
                algorithm_kind="dqn_sb3",
                eval_env=eval_env,
                summary_env=_get_base_env(eval_env),
            )
            _log_episode_summary(
                wandb_run,
                csv_run,
                episode_summary,
                step=final_log_step,
                logging_cfg=cfg.logging,
                include_debug_to_wandb=_logging_flag(cfg.logging, "debug_metrics", False),
                include_final_to_wandb=_logging_flag(cfg.logging, "log_final_traffic_metrics", True),
            )
        finally:
            eval_env.close()
    finally:
        env.close()


def _run_sb3_ppo(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from stable_baselines3 import PPO

    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    eval_seeds = _get_sb3_eval_seeds(cfg)
    log_freq = int(getattr(cfg.logging, "log_freq", 1000))
    eval_freq = int(getattr(cfg.logging, "eval_freq", log_freq))
    env = _build_env(cfg, run_dir)

    try:
        env, num_envs = _wrap_sb3_env_if_needed(cfg, env, params, default_num_envs=2)
        eval_env = _build_env(cfg, run_dir)
        eval_params = _as_plain_dict(cfg.algorithm.params or {})
        eval_params.pop("num_envs", None)
        eval_env, _ = _wrap_sb3_env_if_needed(cfg, eval_env, eval_params, default_num_envs=1)
        callback = SB3WandbCallback(
            wandb_run,
            csv_run,
            logging_cfg=cfg.logging,
            log_freq=log_freq,
            eval_env=eval_env,
            eval_episodes=eval_episodes,
            eval_freq=eval_freq,
            eval_seeds=eval_seeds,
            checkpoint_dir=_get_sb3_checkpoint_dir(run_dir, cfg),
            checkpoint_freq=int(getattr(cfg.logging, "checkpoint_freq", 0)),
            save_checkpoints=_logging_flag(cfg.logging, "save_checkpoints", False),
            save_final_model=_logging_flag(cfg.logging, "save_final_model", True),
        ).build()

        model = PPO(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps), callback=callback)
        final_log_step = _get_sb3_final_log_step(cfg, model)

        try:
            mean_reward, std_reward, episode_summary = _run_sb3_final_evaluation(
                model,
                eval_env,
                algorithm_kind="ppo_sb3",
                eval_seeds=eval_seeds,
                eval_episodes=eval_episodes,
                logging_cfg=cfg.logging,
            )
            _log_final_summary_debug(
                episode_summary,
                cfg.logging,
                algorithm_kind="ppo_sb3",
                eval_env=eval_env,
                summary_env=_get_base_env(eval_env),
            )
            _log_episode_summary(
                wandb_run,
                csv_run,
                episode_summary,
                step=final_log_step,
                logging_cfg=cfg.logging,
                include_debug_to_wandb=_logging_flag(cfg.logging, "debug_metrics", False),
                include_final_to_wandb=_logging_flag(cfg.logging, "log_final_traffic_metrics", True),
            )
        finally:
            eval_env.close()
    finally:
        env.close()


def _run_sb3_sac(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from stable_baselines3 import SAC

    if cfg.env.factory != "parallel_env":
        raise ValueError("The SAC runner requires `env.factory=parallel_env` so it can use the joint-action wrapper.")

    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    eval_seeds = _get_sb3_eval_seeds(cfg)
    log_freq = int(getattr(cfg.logging, "log_freq", 1000))
    eval_freq = int(getattr(cfg.logging, "eval_freq", log_freq))
    env = _build_env(cfg, run_dir)

    try:
        env = JointMultiAgentActionWrapper(env)
        eval_env = JointMultiAgentActionWrapper(_build_env(cfg, run_dir))
        callback = SB3WandbCallback(
            wandb_run,
            csv_run,
            logging_cfg=cfg.logging,
            log_freq=log_freq,
            eval_env=eval_env,
            eval_episodes=eval_episodes,
            eval_freq=eval_freq,
            eval_seeds=eval_seeds,
            checkpoint_dir=_get_sb3_checkpoint_dir(run_dir, cfg),
            checkpoint_freq=int(getattr(cfg.logging, "checkpoint_freq", 0)),
            save_checkpoints=_logging_flag(cfg.logging, "save_checkpoints", False),
            save_final_model=_logging_flag(cfg.logging, "save_final_model", True),
        ).build()

        model = SAC(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            seed=int(cfg.experiment.seed) if cfg.experiment.seed is not None else None,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps), callback=callback)
        final_log_step = _get_sb3_final_log_step(cfg, model)

        try:
            mean_reward, std_reward, episode_summary = _run_sb3_final_evaluation(
                model,
                eval_env,
                algorithm_kind="sac_sb3",
                eval_seeds=eval_seeds,
                eval_episodes=eval_episodes,
                logging_cfg=cfg.logging,
            )
            _log_final_summary_debug(
                episode_summary,
                cfg.logging,
                algorithm_kind="sac_sb3",
                eval_env=eval_env,
                summary_env=_get_base_env(eval_env),
            )
            _log_episode_summary(
                wandb_run,
                csv_run,
                episode_summary,
                step=final_log_step,
                logging_cfg=cfg.logging,
                include_debug_to_wandb=_logging_flag(cfg.logging, "debug_metrics", False),
                include_final_to_wandb=_logging_flag(cfg.logging, "log_final_traffic_metrics", True),
            )
        finally:
            eval_env.close()
    finally:
        env.close()


def _resolve_libsignal_root(cfg: DictConfig) -> Path:
    params = _as_plain_dict(cfg.algorithm.params or {})
    raw_root = params.get("libsignal_root") if isinstance(params, dict) else None
    if raw_root:
        return Path(str(raw_root)).expanduser().resolve()

    env_root = os.environ.get("LIBSIGNAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    raise FileNotFoundError(
        "Phase 5 requires a LibSignal checkout. Set `algorithm.params.libsignal_root` or `LIBSIGNAL_ROOT` to the external repository."
    )


def _libsignal_spec_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    params = _as_plain_dict(cfg.algorithm.params or {})
    spec = {
        "agent": str(params.get("libsignal_agent", "dqn")),
        "network": str(params.get("libsignal_network", "sumo4x4")),
        "task": str(params.get("libsignal_task", "tsc")),
        "world": str(params.get("libsignal_world", "sumo")),
        "dataset": str(params.get("libsignal_dataset", "onfly")),
        "interface": str(params.get("libsignal_interface", "libsumo")),
        "delay_type": str(params.get("libsignal_delay_type", "real")),
        "thread_num": int(params.get("libsignal_thread_num", 1)),
        "ngpu": str(params.get("libsignal_ngpu", "-1")),
        "episodes": int(params.get("libsignal_episodes", max(1, int(cfg.experiment.episodes)))),
        "steps": int(params.get("libsignal_steps", 3600)),
        "test_steps": int(params.get("libsignal_test_steps", 3600)),
        "learning_start": int(params.get("libsignal_learning_start", 1000)),
        "buffer_size": int(params.get("libsignal_buffer_size", 5000)),
        "update_model_rate": int(params.get("libsignal_update_model_rate", 1)),
        "update_target_rate": int(params.get("libsignal_update_target_rate", 10)),
        "save_rate": int(params.get("libsignal_save_rate", 1)),
    }
    return spec


def _run_libsignal_phase5(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from sumo_rl.integrations.libsignal import (
        LibSignalSpec,
        build_phase5_trace_row,
        load_phase5_summary,
        resolve_libsignal_root,
        run_phase5_libsignal_seed,
    )

    params = _as_plain_dict(cfg.algorithm.params or {})
    libsignal_root = resolve_libsignal_root(params.get("libsignal_root"))
    spec_dict = _libsignal_spec_from_cfg(cfg)
    spec = LibSignalSpec(**spec_dict)
    seeds = _get_run_seeds(cfg)
    run_metrics: list[Dict[str, Any]] = []

    repo_root = Path(__file__).resolve().parents[2]
    phase5_dir = run_dir / "phase5" / "libsignal"
    phase5_dir.mkdir(parents=True, exist_ok=True)

    for run_idx, seed in enumerate(seeds, start=1):
        seed_prefix = f"{cfg.experiment.name}_seed{seed}"
        output_root = phase5_dir / seed_prefix / "external"
        summary_path = phase5_dir / seed_prefix / "summary.json"
        output_root.mkdir(parents=True, exist_ok=True)

        summary = run_phase5_libsignal_seed(
            repo_root=repo_root,
            libsignal_root=libsignal_root,
            output_root=output_root,
            summary_path=summary_path,
            spec=spec,
            run_seed=seed,
            run_prefix=seed_prefix,
        )
        if not isinstance(summary, dict):
            summary = load_phase5_summary(summary_path)

        row = build_phase5_trace_row(summary, run_idx=run_idx, run_seed=seed, experiment_name=cfg.experiment.name)

        external_log_file = summary.get("external_log_file")
        if isinstance(external_log_file, str) and external_log_file:
            raw_log_copy = phase5_dir / seed_prefix / Path(external_log_file).name
            try:
                raw_log_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(external_log_file, raw_log_copy)
                row["phase5/raw_log_copy"] = str(raw_log_copy)
            except Exception:
                row["phase5/raw_log_copy"] = str(external_log_file)

        _log_episode_summary(
            wandb_run,
            csv_run,
            row,
            step=int(row.get("episode/steps", 0)),
            logging_cfg=cfg.logging,
        )
        run_metrics.append(row)

    summary_row = _aggregate_numeric_row_values(run_metrics)
    summary_row["algorithm/kind"] = "libsignal_phase5"
    summary_row["phase5/backend"] = "libsignal"
    summary_row["phase5/model_group"] = "idqn/mplight/ippo"
    if wandb_run is not None:
        wandb_run.log(summary_row, step=len(run_metrics))
        _update_wandb_summary(wandb_run, summary_row)
    if csv_run is not None:
        csv_run.log(summary_row, step=len(run_metrics))


def run(cfg: DictConfig) -> None:
    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(cfg.experiment.seed) if cfg.experiment.seed is not None else 0)
    random.seed(int(cfg.experiment.seed) if cfg.experiment.seed is not None else 0)

    wandb_run = _init_wandb(cfg, run_dir)
    csv_run = _LocalMetricsCsvLogger(run_dir / "logs" / "metrics.csv")
    try:
        algorithm_kind = cfg.algorithm.kind
        if algorithm_kind == "q_learning":
            if cfg.env.factory == "env":
                _run_aec_q_learning(cfg, run_dir, wandb_run, csv_run)
            else:
                _run_direct_q_learning(cfg, run_dir, wandb_run, csv_run)
        elif algorithm_kind == "fixed_time":
            _run_fixed_time(cfg, run_dir, wandb_run, csv_run)
        elif algorithm_kind == "static_max_pressure":
            _run_static_policy(cfg, run_dir, wandb_run, csv_run, "max_pressure")
        elif algorithm_kind == "static_greedy":
            _run_static_policy(cfg, run_dir, wandb_run, csv_run, "greedy")
        elif algorithm_kind == "dqn_sb3":
            _run_sb3_dqn(cfg, run_dir, wandb_run, csv_run)
        elif algorithm_kind == "ppo_sb3":
            _run_sb3_ppo(cfg, run_dir, wandb_run, csv_run)
        elif algorithm_kind == "sac_sb3":
            _run_sb3_sac(cfg, run_dir, wandb_run, csv_run)
        elif algorithm_kind == "libsignal_phase5":
            _run_libsignal_phase5(cfg, run_dir, wandb_run, csv_run)
        else:
            raise ValueError(f"Unsupported algorithm kind: {algorithm_kind}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
