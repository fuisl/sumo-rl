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


def _numeric_metrics(data: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            nested_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[nested_prefix] = float(value)
            elif isinstance(value, dict):
                metrics.update(_numeric_metrics(value, nested_prefix))
    return metrics


def _split_resco_metrics(info: Any, include_agent_metrics: bool = False) -> tuple[Dict[str, float], Dict[str, float]]:
    flat_metrics = _numeric_metrics(info)
    system_metrics: Dict[str, float] = {}
    agent_metrics: Dict[str, float] = {}

    for key, value in flat_metrics.items():
        if key == "step":
            continue
        if key.startswith("system_"):
            system_metrics[f"resco_{key}"] = value
        elif include_agent_metrics:
            agent_metrics[f"resco_{key}"] = value

    return system_metrics, agent_metrics


def _log_resco_metrics(
    wandb_run,
    csv_run,
    shared_metrics: Dict[str, Any],
    info: Any,
    step: Optional[int] = None,
    include_agent_metrics_local: bool = False,
) -> None:
    system_metrics, agent_metrics = _split_resco_metrics(info, include_agent_metrics_local)
    wandb_metrics = dict(shared_metrics)
    wandb_metrics.update(system_metrics)
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
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "env") and getattr(env.unwrapped.env, "metrics", None):
        return env.unwrapped.env.metrics[-1]
    if getattr(env, "metrics", None):
        return env.metrics[-1]
    return {}


def _get_env_step(env) -> int:
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "env") and hasattr(env.unwrapped.env, "sim_step"):
        return int(env.unwrapped.env.sim_step)
    if hasattr(env, "sim_step"):
        return int(env.sim_step)
    return 0


def _get_base_env(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "finalize_episode_summary"):
            return current
        if hasattr(current, "unwrapped") and hasattr(current.unwrapped, "env"):
            current = current.unwrapped.env
            continue
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "envs") and current.envs:
            current = current.envs[0]
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
            break
    return env


def _wrap_sb3_env_if_needed(cfg: DictConfig, env, params: Dict[str, Any], default_num_envs: int):
    if cfg.env.factory != "parallel_env" and not hasattr(env, "possible_agents"):
        return env, 1

    import supersuit as ss
    from stable_baselines3.common.vec_env import VecMonitor

    num_envs = int(params.pop("num_envs", default_num_envs))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_envs,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    if not hasattr(env, "render_mode"):
        env.render_mode = None
    env = VecMonitor(env)
    return env, num_envs


def _get_episode_summary(env) -> Dict[str, Any]:
    base_env = _get_base_env(env)
    summary = dict(getattr(base_env, "last_episode_summary", {}) or {})
    if not summary and hasattr(base_env, "finalize_episode_summary"):
        summary = dict(base_env.finalize_episode_summary() or {})
    return summary


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
    summary = _get_episode_summary(env)
    allowed_keys = {
        "sim_step",
        "system_mean_pressure",
        "system_mean_average_speed",
        "system_mean_speed",
        "system_total_stopped",
        "system_total_departed",
        "system_total_arrived",
        "system_total_emergency_brake",
        "system_total_queued",
        "system_mean_queued",
        "system_max_queue",
    }
    row = {
        key: value
        for key, value in summary.items()
        if key.startswith("resco_") or key in allowed_keys
    }
    if extra:
        row.update(extra)
    return row


def _log_episode_summary(wandb_run, csv_run, row: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb_run is not None:
        wandb_run.log(row, step=step)
    if csv_run is not None:
        csv_run.log(row, step=step)


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
                if not include_agent_metrics and key.startswith("resco_") and not key.startswith("resco_system_"):
                    continue
                if not include_agent_metrics and key not in {"episode/reward"} and not key.startswith("resco_system_"):
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

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    initial_states = env.reset()
                    previous_summary = _build_episode_summary_row(env, run_idx, episode_idx - 1)
                    previous_summary["episode/reward"] = previous_episode_reward
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        previous_summary,
                        step=int(previous_summary.get("episode/steps", _get_env_step(env))),
                    )
                    run_metrics.append(previous_summary)
                    for agent_id in initial_states.keys():
                        agents[agent_id].state = env.encode(initial_states[agent_id], agent_id)

                episode_reward = 0.0
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
                            agents[agent_id].learn(
                                next_state=env.encode(next_state[agent_id], agent_id),
                                reward=reward[agent_id],
                            )

                previous_episode_reward = episode_reward
                env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    env.close()
                    episode_summary = _build_episode_summary_row(
                        env,
                        run_idx,
                        episode_idx,
                        extra={"episode/reward": previous_episode_reward},
                    )
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=int(episode_summary.get("episode/steps", _get_env_step(env))),
                    )
                    run_metrics.append(episode_summary)

        summary = _aggregate_numeric_rows(run_metrics)
        summary["algorithm/kind"] = "q_learning"
        if wandb_run is not None:
            wandb_run.log(summary, step=len(run_metrics))
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

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    env.reset()
                    previous_summary = _build_episode_summary_row(env, run_idx, episode_idx - 1)
                    previous_summary["episode/reward"] = previous_episode_reward
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        previous_summary,
                        step=int(previous_summary.get("episode/steps", _get_env_step(env))),
                    )
                    run_metrics.append(previous_summary)
                    for agent_id in env.agents:
                        agents[agent_id].state = env.unwrapped.env.encode(env.observe(agent_id), agent_id)

                episode_reward = 0.0

                if fixed_ts:
                    while env.agents:
                        env.step(None)
                else:
                    for agent in env.agent_iter():
                        observation, reward, terminated, truncated, _ = env.last()
                        done = terminated or truncated
                        if agents[agent].action is not None:
                            agents[agent].learn(
                                next_state=env.unwrapped.env.encode(observation, agent),
                                reward=reward,
                            )

                        action = agents[agent].act() if not done else None
                        env.step(action)
                        episode_reward += float(reward)

                previous_episode_reward = episode_reward
                env.unwrapped.env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                if episode_idx == total_episodes:
                    env.close()
                    episode_summary = _build_episode_summary_row(
                        env,
                        run_idx,
                        episode_idx,
                        extra={"episode/reward": previous_episode_reward},
                    )
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=int(episode_summary.get("episode/steps", _get_env_step(env))),
                    )
                    run_metrics.append(episode_summary)

        summary = _aggregate_numeric_rows(run_metrics)
        summary["algorithm/kind"] = "q_learning"
        if wandb_run is not None:
            wandb_run.log(summary, step=len(run_metrics))
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
                    episode_summary = _build_resco_summary_row(
                        base_env,
                        extra={"static/policy": "fixed_time"},
                    )
                    row_step = run_idx
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=row_step,
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
                    episode_summary = _build_resco_summary_row(
                        base_env,
                        extra={"static/policy": policy_name},
                    )
                    row_step = run_idx
                    _log_episode_summary(
                        wandb_run,
                        csv_run,
                        episode_summary,
                        step=row_step,
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
    if csv_run is not None:
        csv_run.log(csv_summary, step=final_step or None)


def _run_sb3_dqn(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy

    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    env = _build_env(cfg, run_dir)

    try:
        env, _ = _wrap_sb3_env_if_needed(cfg, env, params, default_num_envs=1)
        model = DQN(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            seed=int(cfg.experiment.seed) if cfg.experiment.seed is not None else None,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps))

        eval_env = None
        try:
            eval_env = _build_env(cfg, run_dir)
            eval_params = _as_plain_dict(cfg.algorithm.params or {})
            eval_env, _ = _wrap_sb3_env_if_needed(cfg, eval_env, eval_params, default_num_envs=1)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
            summary_env = _get_base_env(eval_env)
            eval_env.close()
            episode_summary = _build_episode_summary_row(
                summary_env,
                1,
                1,
                extra={
                    "episode/reward": float(mean_reward),
                    "algorithm/kind": "dqn_sb3",
                    "eval/std_reward": float(std_reward),
                },
            )
            _log_episode_summary(
                wandb_run,
                csv_run,
                episode_summary,
                step=int(episode_summary.get("episode/steps", _get_env_step(eval_env))),
            )
        finally:
            if eval_env is not None:
                eval_env.close()
    finally:
        env.close()


def _run_sb3_ppo(cfg: DictConfig, run_dir: Path, wandb_run, csv_run) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    env = _build_env(cfg, run_dir)

    try:
        env, num_envs = _wrap_sb3_env_if_needed(cfg, env, params, default_num_envs=2)

        model = PPO(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps))

        eval_env = None
        try:
            eval_env = _build_env(cfg, run_dir)
            eval_params = _as_plain_dict(cfg.algorithm.params or {})
            eval_env, _ = _wrap_sb3_env_if_needed(cfg, eval_env, eval_params, default_num_envs=num_envs)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
            summary_env = _get_base_env(eval_env)
            eval_env.close()
            episode_summary = _build_episode_summary_row(
                summary_env,
                1,
                1,
                extra={
                    "episode/reward": float(mean_reward),
                    "algorithm/kind": "ppo_sb3",
                    "eval/std_reward": float(std_reward),
                },
            )
            _log_episode_summary(
                wandb_run,
                csv_run,
                episode_summary,
                step=int(episode_summary.get("episode/steps", _get_env_step(eval_env))),
            )
        finally:
            if eval_env is not None:
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
        )
        run_metrics.append(row)

    summary_row = _aggregate_numeric_row_values(run_metrics)
    summary_row["algorithm/kind"] = "libsignal_phase5"
    summary_row["phase5/backend"] = "libsignal"
    summary_row["phase5/model_group"] = "idqn/mplight/ippo"
    if wandb_run is not None:
        wandb_run.log(summary_row, step=len(run_metrics))
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
        elif algorithm_kind == "libsignal_phase5":
            _run_libsignal_phase5(cfg, run_dir, wandb_run, csv_run)
        else:
            raise ValueError(f"Unsupported algorithm kind: {algorithm_kind}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
