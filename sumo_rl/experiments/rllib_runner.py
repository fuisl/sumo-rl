from __future__ import annotations

from collections import deque
import colorsys
from dataclasses import dataclass
import importlib
import json
import os
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

for _env_var in _CPU_THREAD_ENV_VARS:
    os.environ.setdefault(_env_var, "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

# Ray computes its default storage path while importing modules, so point the
# process home directory at the workspace before any Ray import can happen.
os.environ["HOME"] = str(Path.cwd())
os.environ["USERPROFILE"] = str(Path.cwd())

from sumo_rl.experiments.runner import (
    _LocalMetricsCsvLogger,
    _aggregate_final_eval_rows,
    _build_final_eval_summary_row,
    _get_completed_episode_summary,
    _get_run_dir,
    _init_wandb,
    _log_outputs,
    _resolve_num_gpus,
    _update_wandb_summary,
)
from sumo_rl.agents.fgsv2 import fgsv2 as fgsv2_agent
from sumo_rl.experiments.metric_utils import map_system_metrics_to_namespaces
from sumo_rl.agents.rllib_common import (
    build_rllib_parallel_env,
    build_policy_mapping as _build_policy_mapping,
    decision_interval_seconds,
    _possible_agents,
    plain_dict as _plain_dict,
    policy_id_for_agent as _policy_id_for_agent,
    policy_mode as _policy_mode,
    scenario_factory_name,
    validation_interval_episodes,
)
SUPPORTED_RLLIB_ALGORITHMS = {
    "ppo",
    "ppo_dcrnn_mlp",
    "dqn",
    "frap",
    "colight",
    "fgs",
    "fgs_ppo",
    "dqn_dcrnn",
    "dqn_dcrnn_mlp",
    "dcrnn",
    "sac_builtin",
    fgsv2_agent.KIND,
    "sac_mlp",
    "sac_dcrnn_actor",
    "sac_dcrnn_actor_mlp",
    "sac_dcrnn_full",
    "sac_dcrnn_full_mlp",
    "sac_dcrnn_shared_mlp",
    "sac_custom",
}


def normalize_algorithm_kind(algorithm_kind: str) -> str:
    kind = str(algorithm_kind or "").strip()
    if kind == "dcrnn":
        return "dqn_dcrnn"
    if kind == "sac_custom":
        return "sac_mlp"
    return kind


@dataclass
class TripinfoDistributionArtifact:
    wait_values: list[float]
    delay_values: list[float]
    finished_count: int
    unfinished_count: int
    total_count: int


@dataclass
class ValidationSeedArtifacts:
    seed: int
    episode_summary: Dict[str, Any]
    action_traces: Dict[str, list[int]]
    action_space_sizes: Dict[str, int]
    phase_queue_traces: Dict[str, list[Dict[str, Any]]]
    tripinfo: TripinfoDistributionArtifact


def _eval_seeds(cfg: DictConfig) -> list[int]:
    explicit = _plain_dict(getattr(cfg.experiment, "eval_seeds", None))
    eval_episodes = int(getattr(cfg.experiment, "eval_episodes", 0) or 0)
    if isinstance(explicit, list) and explicit:
        seeds = [int(seed) for seed in explicit]
        if eval_episodes > 0:
            return seeds[:eval_episodes]
        return seeds
    base_seed = int(cfg.experiment.seed) if getattr(cfg.experiment, "seed", None) is not None else 0
    count = max(1, eval_episodes or 1)
    return [base_seed + index for index in range(count)]


def _rllib_run_name(cfg: DictConfig, algorithm_kind: str) -> str:
    logging_name = str(getattr(getattr(cfg, "logging", None), "name", "") or "").strip()
    if logging_name:
        return logging_name
    experiment_name = str(getattr(getattr(cfg, "experiment", None), "name", "") or "").strip()
    if experiment_name and experiment_name != "rllib":
        return experiment_name
    scenario_name = scenario_factory_name(cfg) or str(getattr(getattr(cfg, "scenario", None), "name", "scenario"))
    timestamp = datetime.now().strftime("%H%M%S")
    return f"{scenario_name}__{algorithm_kind}__{timestamp}"


def _algorithm_module(algorithm_kind: str):
    algorithm_kind = normalize_algorithm_kind(algorithm_kind)
    if algorithm_kind in {"ppo", "ppo_dcrnn_mlp"}:
        return importlib.import_module("sumo_rl.agents.ppo.ppo")
    if algorithm_kind == "dqn":
        return importlib.import_module("sumo_rl.agents.dqn.dqn")
    if algorithm_kind in {"dqn_dcrnn", "dqn_dcrnn_mlp"}:
        return importlib.import_module("sumo_rl.agents.dcrnn.dcrnn")
    if algorithm_kind == "frap":
        return importlib.import_module("sumo_rl.agents.frap.frap")
    if algorithm_kind == "colight":
        return importlib.import_module("sumo_rl.agents.colight.colight")
    if algorithm_kind in {"fgs", "fgs_ppo"}:
        return importlib.import_module("sumo_rl.agents.fgs.fgs")
    if algorithm_kind == fgsv2_agent.KIND:
        return fgsv2_agent
    if algorithm_kind in {
        "sac_builtin",
        "sac_mlp",
        "sac_dcrnn_actor",
        "sac_dcrnn_actor_mlp",
        "sac_dcrnn_full",
        "sac_dcrnn_full_mlp",
        "sac_dcrnn_shared_mlp",
    }:
        return importlib.import_module("sumo_rl.agents.sac.sac")
    raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")


def _build_algorithm_config(cfg: DictConfig, run_dir: Path, algorithm_kind: str):
    algorithm_kind = normalize_algorithm_kind(algorithm_kind)
    module = _algorithm_module(algorithm_kind)
    if algorithm_kind in {
        "ppo_dcrnn_mlp",
        "dqn_dcrnn_mlp",
        "fgs_ppo",
        "sac_builtin",
        "sac_mlp",
        "sac_dcrnn_actor",
        "sac_dcrnn_actor_mlp",
        "sac_dcrnn_full",
        "sac_dcrnn_full_mlp",
        "sac_dcrnn_shared_mlp",
    }:
        return module.build_config(cfg, run_dir, algorithm_kind=algorithm_kind)
    return module.build_config(cfg, run_dir)


def _train_algorithm(algo, cfg: DictConfig, algorithm_kind: str, emit_metrics, validate=None) -> None:
    algorithm_kind = normalize_algorithm_kind(algorithm_kind)
    module = _algorithm_module(algorithm_kind)
    if algorithm_kind in {
        "ppo_dcrnn_mlp",
        "dqn_dcrnn_mlp",
        "fgs_ppo",
        "sac_builtin",
        "sac_mlp",
        "sac_dcrnn_actor",
        "sac_dcrnn_actor_mlp",
        "sac_dcrnn_full",
        "sac_dcrnn_full_mlp",
        "sac_dcrnn_shared_mlp",
    }:
        module.train(algo, cfg, algorithm_kind=algorithm_kind, emit_metrics=emit_metrics, validate=validate)
    else:
        module.train(algo, cfg, emit_metrics=emit_metrics, validate=validate)


def _compute_single_action(algo, obs, *, policy_id: Optional[str] = None):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        try:
            module = get_module(policy_id) if policy_id is not None else get_module()
        except Exception:
            module = None
        if module is not None and hasattr(module, "forward_inference"):
            import torch
            from ray.rllib.core.columns import Columns

            try:
                module_device = next(module.parameters()).device
            except (AttributeError, StopIteration):
                module_device = torch.device("cpu")
            if isinstance(obs, dict):
                obs_batch = {
                    key: torch.as_tensor(np.asarray(value), device=module_device).unsqueeze(0)
                    for key, value in obs.items()
                }
            else:
                obs_batch = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=module_device).unsqueeze(0)
            with torch.no_grad():
                output = module.forward_inference({Columns.OBS: obs_batch})
                if Columns.ACTIONS not in output and Columns.ACTION_DIST_INPUTS in output:
                    action_dist = module.get_inference_action_dist_cls().from_logits(
                        output[Columns.ACTION_DIST_INPUTS]
                    )
                    output[Columns.ACTIONS] = action_dist.to_deterministic().sample()
            action = output[Columns.ACTIONS]
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            return np.asarray(action).reshape(-1)[0].item()

    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        try:
            if policy_id is None:
                action = compute_single_action(obs, explore=False)
            else:
                action = compute_single_action(obs, policy_id=policy_id, explore=False)
            return action[0] if isinstance(action, tuple) else action
        except AttributeError:
            # Some RLlib API-stack combinations route through env runners that
            # do not expose `get_policy` for `compute_single_action()`.
            pass

    get_policy = getattr(algo, "get_policy", None)
    if callable(get_policy):
        try:
            policy = get_policy(policy_id) if policy_id else get_policy()
        except Exception:
            policy = None
        if policy is not None and hasattr(policy, "compute_single_action"):
            action = policy.compute_single_action(obs, explore=False)
            return action[0] if isinstance(action, tuple) else action
    raise AttributeError("Algorithm does not expose a usable validation inference interface.")


def _build_eval_env(cfg: DictConfig, run_dir: Path, seed: int, *, algorithm_kind: str, policy_mode: str):
    algorithm_kind = normalize_algorithm_kind(algorithm_kind)
    module = _algorithm_module(algorithm_kind)
    if algorithm_kind in {
        "dqn_dcrnn",
        "dqn_dcrnn_mlp",
        "ppo_dcrnn_mlp",
        "sac_dcrnn_actor",
        "sac_dcrnn_actor_mlp",
        "sac_dcrnn_full",
        "sac_dcrnn_full_mlp",
        "sac_dcrnn_shared_mlp",
    }:
        return module.build_graph_eval_env(cfg, run_dir, seed=seed)
    build_eval_env = getattr(module, "build_eval_env", None)
    if callable(build_eval_env):
        return build_eval_env(cfg, run_dir, seed=seed)
    return build_rllib_parallel_env(
        cfg,
        run_dir,
        seed=seed,
        pad_spaces=(policy_mode == "shared"),
    )


def _run_multi_agent_episode(algo, env, seed: int, *, policy_mode: str) -> float:
    total_reward, _, _, _ = _run_multi_agent_episode_trace(algo, env, seed, policy_mode=policy_mode)
    return total_reward


def _action_space_size(env, agent_id: str) -> int:
    action_space_fn = getattr(env, "action_space", None)
    if callable(action_space_fn):
        try:
            space = action_space_fn(agent_id)
        except TypeError:
            space = action_space_fn()
        if hasattr(space, "n"):
            return max(0, int(space.n))
    action_spaces = getattr(env, "action_spaces", None)
    if isinstance(action_spaces, dict):
        space = action_spaces.get(agent_id)
        if hasattr(space, "n"):
            return max(0, int(space.n))
    return 0


def _to_discrete_action(action: Any) -> int:
    if isinstance(action, (np.integer, int)) and not isinstance(action, bool):
        return int(action)
    array = np.asarray(action)
    return int(array.reshape(-1)[0])


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


def _resolve_sumo_base_env(env: Any) -> Any:
    queue = [env]
    visited = set()
    fallback = env
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        fallback = current
        if hasattr(current, "traffic_signals") and hasattr(current, "sim_step"):
            return current
        queue.extend(_env_children(current))
    return fallback


def _collect_phase_queue_snapshot(env: Any, agent_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    base_env = _resolve_sumo_base_env(env)
    traffic_signals = getattr(base_env, "traffic_signals", {})
    snapshot: Dict[str, Dict[str, Any]] = {}
    for agent_id in agent_ids:
        traffic_signal = traffic_signals.get(agent_id)
        if traffic_signal is None:
            continue
        phase_queues = list(getattr(traffic_signal, "get_phase_queued_counts", lambda: [])() or [])
        snapshot[agent_id] = {
            "active_phase": int(getattr(traffic_signal, "green_phase", 0) or 0),
            "phase_queues": [int(value) for value in phase_queues],
        }
    return snapshot


def _validation_tripinfo_output_path(env: Any) -> Optional[Path]:
    base_env = _resolve_sumo_base_env(env)
    build_path = getattr(base_env, "_build_tripinfo_output_path", None)
    if callable(build_path):
        try:
            path = build_path()
        except Exception:
            path = None
        if path is not None:
            return Path(path)
    tripinfo_output_name = getattr(base_env, "tripinfo_output_name", None)
    label = getattr(base_env, "label", None)
    episode = getattr(base_env, "episode", None)
    if tripinfo_output_name and label is not None and episode is not None:
        return Path(f"{tripinfo_output_name}_conn{label}_ep{episode}.xml")
    return None


def _parse_tripinfo_distribution_file(tripinfo_path: Path) -> TripinfoDistributionArtifact:
    wait_values: list[float] = []
    delay_values: list[float] = []
    finished_count = 0
    unfinished_count = 0
    if not tripinfo_path.exists():
        return TripinfoDistributionArtifact(wait_values=[], delay_values=[], finished_count=0, unfinished_count=0, total_count=0)
    try:
        tree = ET.parse(tripinfo_path)
    except (ET.ParseError, OSError):
        return TripinfoDistributionArtifact(wait_values=[], delay_values=[], finished_count=0, unfinished_count=0, total_count=0)

    def _is_truthy_xml_value(value: Optional[str]) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    for vehicle in tree.getroot().findall(".//tripinfo"):
        vehicle_id = str(vehicle.attrib.get("id", "") or "")
        if vehicle_id.startswith("ghost"):
            continue
        is_unfinished = _is_truthy_xml_value(vehicle.attrib.get("vaporized")) or _is_truthy_xml_value(
            vehicle.attrib.get("unfinished")
        )
        if is_unfinished:
            unfinished_count += 1
            continue
        finished_count += 1
        time_loss = float(vehicle.attrib.get("timeLoss", 0.0))
        depart_delay = float(vehicle.attrib.get("departDelay", 0.0))
        delay_values.append(time_loss + depart_delay)
        wait_values.append(float(vehicle.attrib.get("waitingTime", 0.0)))
    return TripinfoDistributionArtifact(
        wait_values=wait_values,
        delay_values=delay_values,
        finished_count=finished_count,
        unfinished_count=unfinished_count,
        total_count=finished_count + unfinished_count,
    )


def _extract_validation_seed_artifacts(
    *,
    seed: int,
    tripinfo_path: Optional[Path],
    episode_summary: Dict[str, Any],
    action_traces: Dict[str, list[int]],
    action_space_sizes: Dict[str, int],
    phase_queue_traces: Dict[str, list[Dict[str, Any]]],
    remove_tripinfo_after_parse: bool,
) -> ValidationSeedArtifacts:
    tripinfo = (
        _parse_tripinfo_distribution_file(tripinfo_path)
        if tripinfo_path is not None
        else TripinfoDistributionArtifact(wait_values=[], delay_values=[], finished_count=0, unfinished_count=0, total_count=0)
    )
    if remove_tripinfo_after_parse and tripinfo_path is not None:
        try:
            tripinfo_path.unlink(missing_ok=True)
        except OSError:
            pass
    return ValidationSeedArtifacts(
        seed=seed,
        episode_summary=dict(episode_summary),
        action_traces={str(agent_id): list(trace) for agent_id, trace in action_traces.items()},
        action_space_sizes={str(agent_id): int(size) for agent_id, size in action_space_sizes.items()},
        phase_queue_traces={str(agent_id): list(rows) for agent_id, rows in phase_queue_traces.items()},
        tripinfo=tripinfo,
    )


def _run_multi_agent_episode_trace(
    algo,
    env,
    seed: int,
    *,
    policy_mode: str,
) -> tuple[float, Dict[str, list[int]], Dict[str, int], Dict[str, list[Dict[str, Any]]]]:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    possible_agents = _possible_agents(env)
    agent_ids = [str(agent_id) for agent_id in possible_agents if not str(agent_id).startswith("__")]
    action_traces = {agent_id: [] for agent_id in agent_ids}
    action_space_sizes = {agent_id: _action_space_size(env, agent_id) for agent_id in agent_ids}
    phase_queue_traces = {agent_id: [] for agent_id in agent_ids}
    while not done:
        actions = {}
        for agent_id, agent_obs in obs.items():
            if agent_id.startswith("__"):
                continue
            actions[agent_id] = _compute_single_action(
                algo,
                agent_obs,
                policy_id=_policy_id_for_agent(str(agent_id), policy_mode),
            )
        for agent_id, action in actions.items():
            action_traces[str(agent_id)].append(_to_discrete_action(action))
        obs, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += float(sum(float(value) for value in rewards.values()))
        phase_queue_snapshot = _collect_phase_queue_snapshot(env, agent_ids)
        for agent_id, agent_snapshot in phase_queue_snapshot.items():
            phase_queue_traces[agent_id].append(
                {
                    "step": float(len(phase_queue_traces[agent_id]) + 1),
                    "active_phase": int(agent_snapshot["active_phase"]),
                    "phase_queues": [int(value) for value in agent_snapshot["phase_queues"]],
                }
            )
        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or all(bool(terminations.get(agent_id, False)) for agent_id in possible_agents)
            or all(bool(truncations.get(agent_id, False)) for agent_id in possible_agents)
        )
    for agent_id, trace in action_traces.items():
        if trace:
            action_space_sizes[agent_id] = max(int(action_space_sizes.get(agent_id, 0)), max(trace) + 1)
    for agent_id, rows in phase_queue_traces.items():
        max_phase_count = max((len(item.get("phase_queues", [])) for item in rows), default=0)
        if max_phase_count > 0:
            action_space_sizes[agent_id] = max(int(action_space_sizes.get(agent_id, 0)), max_phase_count)
    return total_reward, action_traces, action_space_sizes, phase_queue_traces


def _copy_numeric_metric(row: Dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        row[key] = float(value)


def _summary_value(summary: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def _validation_seed_metrics(episode_summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    reward_key_map = {
        "validation/reward_mean": ("reward/mean",),
        "validation/reward_max": ("reward/max",),
        "validation/reward_std": ("reward/std",),
    }
    resco_key_map = {
        "validation/resco_delay_mean": ("resco_delay_mean", "resco_avg_delay"),
        "validation/resco_delay_max": ("resco_delay_max",),
        "validation/resco_delay_std": ("resco_delay_std", "resco_avg_delay_std"),
        "validation/resco_wait_mean": ("resco_wait_mean", "resco_wait"),
        "validation/resco_wait_max": ("resco_wait_max",),
        "validation/resco_wait_std": ("resco_wait_std",),
        "validation/resco_trip_time_mean": ("resco_trip_time_mean", "resco_trip_time"),
        "validation/resco_queue_mean": ("resco_queue_mean", "resco_queue"),
        "validation/resco_queue_max": ("resco_queue_max", "resco_max_queue"),
        "validation/resco_tripinfo_count": ("resco_tripinfo_count",),
    }
    for row_key, summary_keys in reward_key_map.items():
        _copy_numeric_metric(row, row_key, _summary_value(episode_summary, *summary_keys))
    for row_key, summary_keys in resco_key_map.items():
        _copy_numeric_metric(row, row_key, _summary_value(episode_summary, *summary_keys))

    namespaced_metrics = map_system_metrics_to_namespaces(
        {key: value for key, value in episode_summary.items() if key.startswith("system_")}
    )
    for source_key in ("efficiency_total_arrived", "efficiency_total_departed"):
        _copy_numeric_metric(row, f"validation/{source_key}", namespaced_metrics.get(source_key))
    for source_key in (
        "safety_total_teleported",
        "safety_total_emergency_brake",
        "safety_total_collisions",
    ):
        _copy_numeric_metric(row, f"validation/{source_key}", namespaced_metrics.get(source_key))

    return row


def _strip_final_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("final/")}


def _validation_action_window_steps(cfg: Any) -> int:
    return max(1, int(round(60.0 / float(decision_interval_seconds(cfg)))))


def _validation_action_plot_max_agents(logging_cfg: Any) -> Optional[int]:
    value = getattr(logging_cfg, "validation_action_plot_max_agents", None)
    if value is None:
        return None
    return max(1, int(value))


def _action_distribution_rows(actions: list[int], *, num_actions: int, window_size: int) -> list[Dict[str, float]]:
    if num_actions <= 0:
        return []
    counts = [0 for _ in range(num_actions)]
    window: deque[int] = deque()
    rows: list[Dict[str, float]] = []
    for step_index, action in enumerate(actions, start=1):
        if action < 0 or action >= num_actions:
            continue
        window.append(action)
        counts[action] += 1
        while len(window) > window_size:
            dropped = window.popleft()
            counts[dropped] -= 1
        denom = float(len(window))
        row: Dict[str, float] = {"step": float(step_index)}
        for action_index in range(num_actions):
            row[f"action_{action_index}"] = float(counts[action_index]) / denom if denom > 0 else 0.0
        rows.append(row)
    return rows


def _average_action_distribution_rows(seed_rows: list[list[Dict[str, float]]], *, num_actions: int) -> list[Dict[str, float]]:
    if not seed_rows or num_actions <= 0:
        return []
    max_length = max(len(rows) for rows in seed_rows)
    aggregated: list[Dict[str, float]] = []
    for row_index in range(max_length):
        active_rows = [rows[row_index] for rows in seed_rows if row_index < len(rows)]
        if not active_rows:
            continue
        row: Dict[str, float] = {"step": float(row_index + 1)}
        for action_index in range(num_actions):
            key = f"action_{action_index}"
            row[key] = float(np.mean([float(item.get(key, 0.0)) for item in active_rows]))
        aggregated.append(row)
    return aggregated


def _aggregate_validation_tripinfo_distributions(
    seed_artifacts: list[ValidationSeedArtifacts],
) -> Dict[str, Any]:
    total_seeds = len(seed_artifacts)
    seeds_with_completed = sum(1 for artifact in seed_artifacts if artifact.tripinfo.finished_count > 0)
    total_completed = sum(int(artifact.tripinfo.finished_count) for artifact in seed_artifacts)
    total_unfinished = sum(int(artifact.tripinfo.unfinished_count) for artifact in seed_artifacts)
    total_trips = sum(int(artifact.tripinfo.total_count) for artifact in seed_artifacts)
    wait_series = [list(artifact.tripinfo.wait_values) for artifact in seed_artifacts]
    delay_series = [list(artifact.tripinfo.delay_values) for artifact in seed_artifacts]
    pooled_wait_values = [float(value) for artifact in seed_artifacts for value in artifact.tripinfo.wait_values]
    pooled_delay_values = [float(value) for artifact in seed_artifacts for value in artifact.tripinfo.delay_values]
    return {
        "waiting_time": wait_series,
        "delay": delay_series,
        "pooled_waiting_time": pooled_wait_values,
        "pooled_delay": pooled_delay_values,
        "total_seeds": int(total_seeds),
        "seeds_with_completed_trips": int(seeds_with_completed),
        "seeds_without_completed_trips": int(max(0, total_seeds - seeds_with_completed)),
        "total_completed_trips": int(total_completed),
        "total_unfinished_trips": int(total_unfinished),
        "total_trips": int(total_trips),
    }


def _build_validation_action_plot_rows(
    action_traces_by_seed: list[Dict[str, list[int]]],
    action_space_sizes_by_seed: list[Dict[str, int]],
    *,
    window_size: int,
    max_agents: Optional[int] = None,
) -> Dict[str, list[Dict[str, float]]]:
    agent_ids = sorted(
        {
            str(agent_id)
            for seed_traces in action_traces_by_seed
            for agent_id, trace in seed_traces.items()
            if trace
        }
    )
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    payload: Dict[str, list[Dict[str, float]]] = {}
    for agent_id in agent_ids:
        num_actions = max(
            [int(seed_sizes.get(agent_id, 0)) for seed_sizes in action_space_sizes_by_seed] + [0]
        )
        if num_actions <= 0:
            continue
        per_seed_rows = []
        for seed_traces in action_traces_by_seed:
            trace = list(seed_traces.get(agent_id, []))
            if not trace:
                continue
            rows = _action_distribution_rows(trace, num_actions=num_actions, window_size=window_size)
            if rows:
                per_seed_rows.append(rows)
        averaged_rows = _average_action_distribution_rows(per_seed_rows, num_actions=num_actions)
        if averaged_rows:
            payload[agent_id] = averaged_rows
    return payload


def _build_validation_action_timeline_rows(
    action_traces_by_seed: list[Dict[str, list[int]]],
    action_space_sizes_by_seed: list[Dict[str, int]],
    *,
    max_agents: Optional[int] = None,
) -> Dict[str, list[int]]:
    agent_ids = sorted(
        {
            str(agent_id)
            for seed_traces in action_traces_by_seed
            for agent_id, trace in seed_traces.items()
            if trace
        }
    )
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    payload: Dict[str, list[int]] = {}
    for agent_id in agent_ids:
        num_actions = max(
            [int(seed_sizes.get(agent_id, 0)) for seed_sizes in action_space_sizes_by_seed] + [0]
        )
        if num_actions <= 0:
            continue
        max_length = max((len(seed_traces.get(agent_id, [])) for seed_traces in action_traces_by_seed), default=0)
        if max_length <= 0:
            continue
        aggregated: list[int] = []
        for row_index in range(max_length):
            active_values = [
                int(seed_traces[agent_id][row_index])
                for seed_traces in action_traces_by_seed
                if row_index < len(seed_traces.get(agent_id, []))
            ]
            if not active_values:
                continue
            bincount = np.bincount(np.asarray(active_values, dtype=int), minlength=num_actions)
            aggregated.append(int(np.argmax(bincount)))
        if aggregated:
            payload[agent_id] = aggregated
    return payload


def _build_validation_phase_queue_rows(
    phase_queue_traces_by_seed: list[Dict[str, list[Dict[str, Any]]]],
    *,
    max_agents: Optional[int] = None,
) -> Dict[str, list[Dict[str, float]]]:
    agent_ids = sorted(
        {
            str(agent_id)
            for seed_rows in phase_queue_traces_by_seed
            for agent_id, rows in seed_rows.items()
            if rows
        }
    )
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    payload: Dict[str, list[Dict[str, float]]] = {}
    for agent_id in agent_ids:
        max_length = max((len(seed_rows.get(agent_id, [])) for seed_rows in phase_queue_traces_by_seed), default=0)
        if max_length <= 0:
            continue
        max_phase_count = max(
            (
                len(row.get("phase_queues", []))
                for seed_rows in phase_queue_traces_by_seed
                for row in seed_rows.get(agent_id, [])
            ),
            default=0,
        )
        if max_phase_count <= 0:
            continue
        rows: list[Dict[str, float]] = []
        for row_index in range(max_length):
            active_rows = [
                seed_rows.get(agent_id, [])[row_index]
                for seed_rows in phase_queue_traces_by_seed
                if row_index < len(seed_rows.get(agent_id, []))
            ]
            if not active_rows:
                continue
            row: Dict[str, float] = {"step": float(row_index + 1)}
            active_phases = [int(item.get("active_phase", 0)) for item in active_rows]
            bincount = np.bincount(np.asarray(active_phases, dtype=int), minlength=max_phase_count)
            row["active_phase"] = float(int(np.argmax(bincount)))
            for phase_index in range(max_phase_count):
                values = []
                for item in active_rows:
                    phase_queues = item.get("phase_queues", [])
                    if phase_index < len(phase_queues):
                        values.append(float(phase_queues[phase_index]))
                row[f"phase_{phase_index}"] = float(np.mean(values)) if values else 0.0
            rows.append(row)
        if rows:
            payload[agent_id] = rows
    return payload


def _action_color(action_index: int) -> tuple[int, int, int]:
    base_palette = [
        (49, 130, 206),
        (255, 140, 66),
        (48, 181, 90),
        (235, 87, 87),
        (161, 98, 247),
        (0, 172, 193),
        (245, 194, 66),
        (236, 72, 153),
        (34, 197, 166),
        (120, 113, 108),
    ]
    if action_index < len(base_palette):
        return base_palette[action_index]
    hue = (action_index * 0.61803398875) % 1.0
    saturation = 0.65
    value = 0.9
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(red * 255), int(green * 255), int(blue * 255))


def _render_validation_action_plot_image(
    agent_id: str,
    rows: list[Dict[str, float]],
    *,
    width: int = 1040,
    height: int = 560,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    card_left = 18
    card_top = 18
    card_right = width - 18
    card_bottom = height - 18
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_bottom),
        radius=18,
        fill=(255, 255, 255),
        outline=(225, 232, 240),
        width=2,
    )

    left = card_left + 86
    right = card_right - 28
    top = card_top + 78
    bottom = card_bottom - 86
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)

    action_keys = [key for key in rows[0].keys() if key.startswith("action_")] if rows else []
    num_actions = len(action_keys)
    steps = [float(row["step"]) for row in rows] if rows else [1.0]
    min_step = steps[0]
    max_step = steps[-1]
    step_span = max(max_step - min_step, 1.0)

    title = f"validation {agent_id}"
    draw.text((card_left + 22, card_top + 18), title, fill=(28, 37, 54), font=font)
    subtitle = f"{num_actions} phase{'s' if num_actions != 1 else ''}  |  stacked action share"
    draw.text((card_left + 22, card_top + 38), subtitle, fill=(96, 109, 128), font=font)

    y_ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    for tick_value in y_ticks:
        y = top + (1.0 - tick_value) * plot_height
        is_baseline = abs(tick_value) <= 1e-9
        draw.line((left, y, right, y), fill=(214, 223, 233) if is_baseline else (234, 239, 244), width=1)
        label = f"{tick_value:.1f}"
        draw.text((card_left + 18, y - 6), label, fill=(94, 105, 122), font=font)

    x_tick_values = [min_step, min_step + step_span * 0.25, min_step + step_span * 0.5, min_step + step_span * 0.75, max_step]
    for tick_value in x_tick_values:
        x = left + ((tick_value - min_step) / step_span) * plot_width
        draw.line((x, top, x, bottom), fill=(244, 247, 250), width=1)

    draw.line((left, top, left, bottom), fill=(122, 134, 153), width=1)
    draw.line((left, bottom, right, bottom), fill=(122, 134, 153), width=1)
    draw.text((width // 2 - 12, card_bottom - 34), "step", fill=(94, 105, 122), font=font)
    draw.text((card_left + 18, top - 20), "share", fill=(94, 105, 122), font=font)

    if rows and num_actions > 0:
        x_coords = [
            left + ((float(row["step"]) - min_step) / step_span) * plot_width
            for row in rows
        ]
        cumulative = [0.0 for _ in rows]
        for action_index, action_key in enumerate(action_keys):
            upper = [min(1.0, max(0.0, cumulative[i] + float(rows[i].get(action_key, 0.0)))) for i in range(len(rows))]
            lower_points = [
                (
                    x_coords[i],
                    top + (1.0 - cumulative[i]) * plot_height,
                )
                for i in range(len(rows))
            ]
            upper_points = [
                (
                    x_coords[i],
                    top + (1.0 - upper[i]) * plot_height,
                )
                for i in range(len(rows))
            ]
            polygon = upper_points + list(reversed(lower_points))
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=_action_color(action_index))
            cumulative = upper

    legend_x = left + 12
    legend_y = top + 12
    legend_items_per_row = max(1, min(4, num_actions))
    legend_row_height = 22
    legend_col_width = 118
    for action_index, action_key in enumerate(action_keys):
        row_index = action_index // legend_items_per_row
        col_index = action_index % legend_items_per_row
        item_x = legend_x + col_index * legend_col_width
        item_y = legend_y + row_index * legend_row_height
        color = _action_color(action_index)
        draw.rounded_rectangle((item_x, item_y, item_x + 12, item_y + 12), radius=3, fill=color, outline=color)
        draw.text((item_x + 18, item_y), action_key.replace("_", " "), fill=(70, 79, 94), font=font)

    if rows:
        for tick_value in x_tick_values:
            x = left + ((tick_value - min_step) / step_span) * plot_width
            draw.line((x, bottom, x, bottom + 5), fill=(122, 134, 153), width=1)
            label = f"{int(round(tick_value))}"
            label_x = int(x - (len(label) * 3))
            draw.text((label_x, bottom + 10), label, fill=(94, 105, 122), font=font)

    return image


def _count_prefixed_series_keys(rows: list[Dict[str, float]], prefix: str) -> int:
    if not rows:
        return 0
    return sum(1 for key in rows[0].keys() if key.startswith(prefix))


def _format_env_time_label(seconds_value: float) -> str:
    total_seconds = max(0, int(round(seconds_value)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _render_validation_action_timeline_image(
    agent_id: str,
    actions: list[int],
    *,
    decision_seconds: int,
    num_actions: Optional[int] = None,
    width: int = 1040,
    height: int = 420,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    inferred_num_actions = max(actions) + 1 if actions else 0
    num_actions = max(int(num_actions or 0), inferred_num_actions)

    card_left = 18
    card_top = 18
    card_right = width - 18
    card_bottom = height - 18
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_bottom),
        radius=18,
        fill=(255, 255, 255),
        outline=(225, 232, 240),
        width=2,
    )

    left = card_left + 86
    right = card_right - 28
    top = card_top + 78
    bottom = card_bottom - 58
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)

    draw.text((card_left + 22, card_top + 18), f"timeline {agent_id}", fill=(28, 37, 54), font=font)
    draw.text((card_left + 22, card_top + 38), "active phase between adjacent decisions", fill=(96, 109, 128), font=font)

    if not actions or num_actions <= 0:
        return image

    row_gap = 8
    total_gap = row_gap * max(0, num_actions - 1)
    row_height = max(16.0, (plot_height - total_gap) / float(num_actions))
    total_episode_seconds = len(actions) * decision_seconds

    for phase_index in range(num_actions):
        row_top = top + phase_index * (row_height + row_gap)
        row_bottom = row_top + row_height
        draw.rounded_rectangle(
            (left, row_top, right, row_bottom),
            radius=6,
            fill=(246, 248, 251),
            outline=(234, 239, 244),
        )
        draw.text((card_left + 18, row_top + max(0, (row_height - 10) / 2.0)), f"phase {phase_index}", fill=(94, 105, 122), font=font)

    x_tick_steps = [0, len(actions) * 0.25, len(actions) * 0.5, len(actions) * 0.75, len(actions)]
    for tick_step in x_tick_steps:
        x = left + (float(tick_step) / max(1.0, float(len(actions)))) * plot_width
        draw.line((x, top - 4, x, bottom + 4), fill=(241, 245, 249), width=1)
        label = _format_env_time_label(float(tick_step) * float(decision_seconds))
        label_x = int(x - (len(label) * 3))
        draw.text((label_x, bottom + 10), label, fill=(94, 105, 122), font=font)

    previous_action = None
    run_start = 0
    for step_index, action in enumerate(actions + [None]):
        if step_index == 0:
            previous_action = action
            continue
        if action == previous_action:
            continue
        if previous_action is not None and 0 <= int(previous_action) < num_actions:
            x0 = left + (float(run_start) / max(1.0, float(len(actions)))) * plot_width
            x1 = left + (float(step_index) / max(1.0, float(len(actions)))) * plot_width
            row_top = top + int(previous_action) * (row_height + row_gap)
            row_bottom = row_top + row_height
            draw.rounded_rectangle(
                (x0, row_top, max(x0 + 1, x1), row_bottom),
                radius=6,
                fill=_action_color(int(previous_action)),
                outline=_action_color(int(previous_action)),
            )
        run_start = step_index
        previous_action = action

    draw.text((width // 2 - 26, card_bottom - 28), "env time (mm:ss)", fill=(94, 105, 122), font=font)
    return image


def _render_validation_phase_queue_image(
    agent_id: str,
    rows: list[Dict[str, float]],
    *,
    decision_seconds: int,
    width: int = 1040,
    height: int = 520,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    card_left = 18
    card_top = 18
    card_right = width - 18
    card_bottom = height - 18
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_bottom),
        radius=18,
        fill=(255, 255, 255),
        outline=(225, 232, 240),
        width=2,
    )

    left = card_left + 86
    right = card_right - 28
    top = card_top + 78
    bottom = card_bottom - 70
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)

    draw.text((card_left + 22, card_top + 18), f"phase queue {agent_id}", fill=(28, 37, 54), font=font)
    draw.text(
        (card_left + 22, card_top + 38),
        "queued vehicles per phase lane-set; background shows active phase",
        fill=(96, 109, 128),
        font=font,
    )

    if not rows:
        return image

    phase_keys = [key for key in rows[0].keys() if key.startswith("phase_")]
    max_queue = max([float(row.get(key, 0.0)) for row in rows for key in phase_keys] + [1.0])
    y_max = max(1.0, float(np.ceil(max_queue)))
    steps = [float(row["step"]) for row in rows]
    min_step = steps[0]
    max_step = steps[-1]
    step_span = max(max_step - min_step, 1.0)

    active_phase_runs = []
    run_start = 0
    previous_phase = int(rows[0].get("active_phase", 0))
    for row_index in range(1, len(rows) + 1):
        current_phase = None if row_index == len(rows) else int(rows[row_index].get("active_phase", 0))
        if current_phase == previous_phase:
            continue
        active_phase_runs.append((run_start, row_index, previous_phase))
        run_start = row_index
        previous_phase = current_phase if current_phase is not None else previous_phase

    for run_start, run_end, phase_index in active_phase_runs:
        x0 = left + (float(run_start) / max(1.0, float(len(rows)))) * plot_width
        x1 = left + (float(run_end) / max(1.0, float(len(rows)))) * plot_width
        red, green, blue = _action_color(phase_index)
        fill_color = (
            int(0.85 * 255 + 0.15 * red),
            int(0.85 * 255 + 0.15 * green),
            int(0.85 * 255 + 0.15 * blue),
        )
        draw.rectangle((x0, top, max(x0 + 1, x1), bottom), fill=fill_color)

    y_tick_values = [0.0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    for tick_value in y_tick_values:
        y = top + (1.0 - (tick_value / y_max)) * plot_height
        is_baseline = abs(tick_value) <= 1e-9
        draw.line((left, y, right, y), fill=(214, 223, 233) if is_baseline else (234, 239, 244), width=1)
        label = f"{int(round(tick_value))}"
        draw.text((card_left + 18, y - 6), label, fill=(94, 105, 122), font=font)

    x_tick_values = [min_step, min_step + step_span * 0.25, min_step + step_span * 0.5, min_step + step_span * 0.75, max_step]
    for tick_value in x_tick_values:
        x = left + ((tick_value - min_step) / step_span) * plot_width
        draw.line((x, top, x, bottom), fill=(244, 247, 250), width=1)

    draw.line((left, top, left, bottom), fill=(122, 134, 153), width=1)
    draw.line((left, bottom, right, bottom), fill=(122, 134, 153), width=1)
    draw.text((width // 2 - 26, card_bottom - 28), "env time (mm:ss)", fill=(94, 105, 122), font=font)
    draw.text((card_left + 18, top - 20), "queued", fill=(94, 105, 122), font=font)

    x_coords = [left + ((float(row["step"]) - min_step) / step_span) * plot_width for row in rows]
    for phase_index, phase_key in enumerate(phase_keys):
        points = []
        for row_index, row in enumerate(rows):
            value = float(row.get(phase_key, 0.0))
            y = top + (1.0 - (value / y_max)) * plot_height
            points.append((x_coords[row_index], y))
        if len(points) >= 2:
            draw.line(points, fill=_action_color(phase_index), width=3)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=_action_color(phase_index), outline=_action_color(phase_index))

    legend_x = left + 12
    legend_y = top + 12
    legend_items_per_row = max(1, min(4, len(phase_keys)))
    legend_row_height = 22
    legend_col_width = 118
    for phase_index, phase_key in enumerate(phase_keys):
        row_index = phase_index // legend_items_per_row
        col_index = phase_index % legend_items_per_row
        item_x = legend_x + col_index * legend_col_width
        item_y = legend_y + row_index * legend_row_height
        color = _action_color(phase_index)
        draw.rounded_rectangle((item_x, item_y, item_x + 12, item_y + 12), radius=3, fill=color, outline=color)
        draw.text((item_x + 18, item_y), phase_key.replace("_", " "), fill=(70, 79, 94), font=font)

    for tick_value in x_tick_values:
        x = left + ((tick_value - min_step) / step_span) * plot_width
        draw.line((x, bottom, x, bottom + 5), fill=(122, 134, 153), width=1)
        label = _format_env_time_label(float(max(0.0, tick_value - 1.0)) * float(decision_seconds))
        label_x = int(x - (len(label) * 3))
        draw.text((label_x, bottom + 10), label, fill=(94, 105, 122), font=font)

    return image


def _render_validation_tripinfo_distribution_image(
    metric_name: str,
    seed_series: list[list[float]],
    pooled_values: list[float],
    *,
    total_seeds: int,
    seeds_with_completed_trips: int,
    total_completed_trips: int,
    total_unfinished_trips: int,
    width: int = 1040,
    height: int = 460,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    card_left = 18
    card_top = 18
    card_right = width - 18
    card_bottom = height - 18
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_bottom),
        radius=18,
        fill=(255, 255, 255),
        outline=(225, 232, 240),
        width=2,
    )

    left = card_left + 86
    right = card_right - 28
    top = card_top + 78
    bottom = card_bottom - 70
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)

    title = f"validation {metric_name} distribution"
    subtitle = (
        f"pooled completed-trip distribution; {seeds_with_completed_trips}/{total_seeds} seeds with completed trips; "
        f"{total_completed_trips} completed, {total_unfinished_trips} unfinished"
    )
    draw.text((card_left + 22, card_top + 18), title, fill=(28, 37, 54), font=font)
    draw.text((card_left + 22, card_top + 38), subtitle, fill=(96, 109, 128), font=font)

    finite_series = []
    for values in seed_series:
        array = np.asarray(values, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size > 0:
            finite_series.append(array)
    pooled_array = np.asarray(pooled_values, dtype=float).reshape(-1)
    pooled_array = pooled_array[np.isfinite(pooled_array)]
    if pooled_array.size <= 0:
        draw.text((card_left + 22, card_top + 68), "no completed vehicle tripinfo values available", fill=(148, 163, 184), font=font)
        return image

    percentiles = np.linspace(0.0, 1.0, 21)
    percentile_labels = percentiles * 100.0
    seed_curves = [np.quantile(values, percentiles) for values in finite_series]
    pooled_curve = np.quantile(pooled_array, percentiles)
    curve_values = list(seed_curves)
    curve_values.append(np.asarray(pooled_curve, dtype=float))
    y_max = max(1.0, float(np.max(np.asarray(curve_values, dtype=float))))

    accent = (49, 130, 206) if metric_name == "waiting time" else (255, 140, 66)
    light_accent = tuple(int(0.72 * 255 + 0.28 * channel) for channel in accent)

    y_tick_values = np.linspace(0.0, y_max, 5)
    for tick_value in y_tick_values:
        y = top + (1.0 - (float(tick_value) / y_max)) * plot_height
        is_baseline = abs(float(tick_value)) <= 1e-9
        draw.line((left, y, right, y), fill=(214, 223, 233) if is_baseline else (234, 239, 244), width=1)
        draw.text((card_left + 18, y - 6), f"{int(round(float(tick_value)))}", fill=(94, 105, 122), font=font)

    x_tick_values = [0.0, 25.0, 50.0, 75.0, 100.0]
    for tick_value in x_tick_values:
        x = left + (float(tick_value) / 100.0) * plot_width
        draw.line((x, top, x, bottom), fill=(244, 247, 250), width=1)
        label = f"{int(round(float(tick_value)))}"
        label_x = int(x - (len(label) * 3))
        draw.text((label_x, bottom + 10), label, fill=(94, 105, 122), font=font)

    draw.line((left, top, left, bottom), fill=(122, 134, 153), width=1)
    draw.line((left, bottom, right, bottom), fill=(122, 134, 153), width=1)
    draw.text((width // 2 - 28, card_bottom - 28), "vehicle percentile", fill=(94, 105, 122), font=font)
    draw.text((card_left + 18, top - 20), "seconds", fill=(94, 105, 122), font=font)

    x_coords = [left + (float(label) / 100.0) * plot_width for label in percentile_labels]
    for curve in seed_curves:
        points = [
            (x_coords[index], top + (1.0 - (float(curve[index]) / y_max)) * plot_height)
            for index in range(len(curve))
        ]
        draw.line(points, fill=light_accent, width=2)

    pooled_points = [
        (x_coords[index], top + (1.0 - (float(pooled_curve[index]) / y_max)) * plot_height)
        for index in range(len(pooled_curve))
    ]
    draw.line(pooled_points, fill=accent, width=4)

    legend_x = left + 12
    legend_y = top + 12
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 12, legend_y + 12), radius=3, fill=light_accent, outline=light_accent)
    draw.text((legend_x + 18, legend_y), "per-seed curve", fill=(70, 79, 94), font=font)
    draw.rounded_rectangle((legend_x + 140, legend_y, legend_x + 152, legend_y + 12), radius=3, fill=accent, outline=accent)
    draw.text((legend_x + 158, legend_y), "pooled curve", fill=(70, 79, 94), font=font)

    return image


def _log_validation_action_plot_images(
    wandb_run,
    plot_rows_by_agent: Dict[str, list[Dict[str, float]]],
    timeline_actions_by_agent: Dict[str, list[int]],
    phase_queue_rows_by_agent: Dict[str, list[Dict[str, float]]],
    *,
    pass_index: int,
    env_step: int,
    episode_index: int,
    decision_seconds: int,
) -> None:
    if wandb_run is None or (not plot_rows_by_agent and not timeline_actions_by_agent and not phase_queue_rows_by_agent):
        return
    import wandb

    agent_ids = sorted(set(plot_rows_by_agent.keys()) | set(timeline_actions_by_agent.keys()) | set(phase_queue_rows_by_agent.keys()))
    for agent_id in agent_ids:
        rows = plot_rows_by_agent.get(agent_id, [])
        timeline_actions = timeline_actions_by_agent.get(agent_id, [])
        phase_queue_rows = phase_queue_rows_by_agent.get(agent_id, [])
        num_actions = max(
            _count_prefixed_series_keys(rows, "action_"),
            _count_prefixed_series_keys(phase_queue_rows, "phase_"),
        )
        payload = {
            "validation/rollout_index": float(episode_index),
            "validation/episode_index": float(episode_index),
            "validation/pass_index": float(pass_index),
            "validation/env_step": float(env_step),
        }
        if not rows:
            pass
        else:
            payload[f"validation/actions_share/{agent_id}"] = wandb.Image(
                _render_validation_action_plot_image(agent_id, rows),
                caption=f"validation pass {pass_index} at env step {env_step}",
            )
        if timeline_actions:
            payload[f"validation/actions_timeline/{agent_id}"] = wandb.Image(
                _render_validation_action_timeline_image(
                    agent_id,
                    timeline_actions,
                    decision_seconds=decision_seconds,
                    num_actions=num_actions,
                ),
                caption=f"validation pass {pass_index} at env step {env_step}",
            )
        if phase_queue_rows:
            payload[f"validation/phase_queue/{agent_id}"] = wandb.Image(
                _render_validation_phase_queue_image(
                    agent_id,
                    phase_queue_rows,
                    decision_seconds=decision_seconds,
                ),
                caption=f"validation pass {pass_index} at env step {env_step}",
            )
        if len(payload) > 3:
            wandb_run.log(payload)


def _log_validation_tripinfo_distribution_images(
    wandb_run,
    tripinfo_distributions: Dict[str, Any],
    *,
    pass_index: int,
    env_step: int,
    episode_index: int,
) -> None:
    if wandb_run is None:
        return
    wait_series = list(tripinfo_distributions.get("waiting_time", []))
    delay_series = list(tripinfo_distributions.get("delay", []))
    pooled_wait_values = list(tripinfo_distributions.get("pooled_waiting_time", []))
    pooled_delay_values = list(tripinfo_distributions.get("pooled_delay", []))
    total_seeds = int(tripinfo_distributions.get("total_seeds", 0) or 0)
    seeds_with_completed_trips = int(tripinfo_distributions.get("seeds_with_completed_trips", 0) or 0)
    total_completed_trips = int(tripinfo_distributions.get("total_completed_trips", 0) or 0)
    total_unfinished_trips = int(tripinfo_distributions.get("total_unfinished_trips", 0) or 0)
    if total_completed_trips <= 0:
        return
    import wandb

    payload = {
        "validation/rollout_index": float(episode_index),
        "validation/episode_index": float(episode_index),
        "validation/pass_index": float(pass_index),
        "validation/env_step": float(env_step),
    }
    if pooled_wait_values:
        payload["validation/tripinfo_wait_distribution"] = wandb.Image(
            _render_validation_tripinfo_distribution_image(
                "waiting time",
                wait_series,
                pooled_wait_values,
                total_seeds=total_seeds,
                seeds_with_completed_trips=seeds_with_completed_trips,
                total_completed_trips=total_completed_trips,
                total_unfinished_trips=total_unfinished_trips,
            ),
            caption=(
                f"validation pass {pass_index} at env step {env_step} | "
                f"{seeds_with_completed_trips}/{total_seeds} seeds with completed trips | "
                f"{total_completed_trips} completed | {total_unfinished_trips} unfinished"
            ),
        )
    if pooled_delay_values:
        payload["validation/tripinfo_delay_distribution"] = wandb.Image(
            _render_validation_tripinfo_distribution_image(
                "delay",
                delay_series,
                pooled_delay_values,
                total_seeds=total_seeds,
                seeds_with_completed_trips=seeds_with_completed_trips,
                total_completed_trips=total_completed_trips,
                total_unfinished_trips=total_unfinished_trips,
            ),
            caption=(
                f"validation pass {pass_index} at env step {env_step} | "
                f"{seeds_with_completed_trips}/{total_seeds} seeds with completed trips | "
                f"{total_completed_trips} completed | {total_unfinished_trips} unfinished"
            ),
        )
    if len(payload) > 3:
        wandb_run.log(payload)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _best_validation_metric_name(logging_cfg: Any) -> str:
    value = getattr(logging_cfg, "best_validation_metric", "validation/resco_delay_mean")
    return str(value or "validation/resco_delay_mean")


def _best_validation_checkpoint_count(logging_cfg: Any) -> int:
    value = getattr(logging_cfg, "best_validation_checkpoint_count", 3)
    return max(1, int(value or 3))


def _should_save_best_validation_checkpoints(logging_cfg: Any) -> bool:
    return bool(getattr(logging_cfg, "save_best_validation_checkpoints", False))


def _best_validation_directory(run_dir: Path, algorithm_kind: str) -> Path:
    return run_dir / "checkpoints" / algorithm_kind / "best_validation"


def _best_metric_value(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    metric_value = metrics.get(metric_name)
    if not isinstance(metric_value, (int, float, np.integer, np.floating)) or isinstance(metric_value, bool):
        return None
    metric_value = float(metric_value)
    if not np.isfinite(metric_value):
        return None
    return metric_value


def _consider_best_metrics_row(
    current_best: Optional[Dict[str, Any]],
    candidate_metrics: Dict[str, Any],
    *,
    metric_name: str,
) -> Optional[Dict[str, Any]]:
    candidate_value = _best_metric_value(candidate_metrics, metric_name)
    if candidate_value is None:
        return current_best
    if current_best is None:
        return dict(candidate_metrics)

    current_value = _best_metric_value(current_best, metric_name)
    if current_value is None or candidate_value < current_value:
        return dict(candidate_metrics)
    return current_best


def _prefix_best_metrics(metrics: Dict[str, Any], *, source_prefix: str, target_prefix: str) -> Dict[str, Any]:
    prefix = f"{source_prefix}/"
    target = f"{target_prefix}/"
    prefixed = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        if key.startswith(prefix):
            prefixed[f"{target}{key[len(prefix):]}"] = value
    return prefixed


def _clear_wandb_summary_prefix(wandb_run, prefix: str) -> None:
    if wandb_run is None:
        return
    summary = getattr(wandb_run, "summary", None)
    if summary is None:
        return
    key_prefix = f"{prefix}/"
    keys = getattr(summary, "keys", None)
    if not callable(keys):
        return
    for key in list(keys()):
        if not isinstance(key, str) or not key.startswith(key_prefix):
            continue
        try:
            del summary[key]
        except Exception:
            try:
                summary.pop(key, None)
            except Exception:
                pass


def _update_wandb_best_summary(wandb_run, best_rows: Dict[str, Optional[Dict[str, Any]]]) -> None:
    if wandb_run is None:
        return
    payloads = {}
    for target_prefix, source_prefix in (("best_train", "train"), ("best_validation", "validation")):
        metrics = best_rows.get(target_prefix)
        _clear_wandb_summary_prefix(wandb_run, target_prefix)
        if metrics:
            payloads.update(_prefix_best_metrics(metrics, source_prefix=source_prefix, target_prefix=target_prefix))
    _update_wandb_summary(wandb_run, payloads)


def _save_checkpoint(algo, checkpoint_dir: Path) -> Optional[Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if hasattr(algo, "save_to_path"):
        checkpoint = algo.save_to_path(str(checkpoint_dir))
    elif hasattr(algo, "save"):
        checkpoint = algo.save(str(checkpoint_dir))
    if checkpoint is None:
        return None
    return Path(checkpoint)


def _restore_checkpoint(algo, checkpoint_path: Path | str) -> Any:
    checkpoint_path = Path(checkpoint_path)
    if hasattr(algo, "restore_from_path"):
        return algo.restore_from_path(str(checkpoint_path))
    if hasattr(algo, "restore"):
        return algo.restore(str(checkpoint_path))
    raise AttributeError("Algorithm does not support checkpoint restore.")


def _sync_env_runner_weights_for_evaluation(algo) -> bool:
    env_runner = getattr(algo, "env_runner", None)
    learner_group = getattr(algo, "learner_group", None)
    if env_runner is None or learner_group is None:
        return False

    get_weights = getattr(learner_group, "get_weights", None)
    set_weights = getattr(env_runner, "set_weights", None)
    if not callable(get_weights) or not callable(set_weights):
        return False

    # RLlib's inference-only state sync can leave custom SAC+DCRNN env-runner
    # weights stale relative to the learner. Refresh the local env-runner from
    # learner weights right before manual/validation rollouts.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            set_weights(get_weights())
    except Exception:
        return False
    return True


def _remove_checkpoint_path(path_value: Any, *, root_dir: Path) -> None:
    path = Path(str(path_value)).resolve()
    root = root_dir.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Refusing to delete checkpoint outside best-validation directory: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _write_best_validation_metadata(metadata_path: Path, state: Dict[str, Any]) -> None:
    retained = []
    for rank, entry in enumerate(state["retained"], start=1):
        item = dict(entry)
        item["rank"] = rank
        retained.append(_json_safe_value(item))

    payload = {
        "metric_name": state["metric_name"],
        "lower_is_better": True,
        "max_retained": state["count"],
        "validation_passes_seen": state["validation_pass_index"],
        "retained": retained,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _init_best_validation_checkpoint_state(run_dir: Path, algorithm_kind: str, logging_cfg: Any) -> Dict[str, Any]:
    base_dir = _best_validation_directory(run_dir, algorithm_kind)
    return {
        "enabled": _should_save_best_validation_checkpoints(logging_cfg),
        "count": _best_validation_checkpoint_count(logging_cfg),
        "metric_name": _best_validation_metric_name(logging_cfg),
        "base_dir": base_dir,
        "metadata_path": base_dir / "metadata.json",
        "retained": [],
        "validation_pass_index": 0,
    }


def _consider_best_validation_checkpoint(
    state: Dict[str, Any],
    algo,
    *,
    validation_metrics: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    evaluation_seed_rows: list[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not state.get("enabled", False):
        return None

    state["validation_pass_index"] = int(state.get("validation_pass_index", 0)) + 1
    metric_name = str(state["metric_name"])
    metric_value = _best_metric_value(validation_metrics, metric_name)
    if metric_value is None:
        return None

    retained = list(state["retained"])
    retained.sort(key=lambda item: (float(item["metric_value"]), int(item["validation_pass_index"])))
    if any(metric_value == float(item["metric_value"]) for item in retained):
        return None
    if len(retained) >= int(state["count"]):
        worst = retained[-1]
        if not metric_value < float(worst["metric_value"]):
            return None

    env_step_value = validation_metrics.get("validation/env_step")
    if isinstance(env_step_value, (int, float, np.integer, np.floating)) and not isinstance(env_step_value, bool):
        env_step = int(float(env_step_value))
    else:
        env_step = 0
    candidate_dir = state["base_dir"] / (
        f"validation_pass_{state['validation_pass_index']:04d}"
        f"__step_{env_step:07d}"
        f"__delay_{metric_value:.6f}"
    )
    checkpoint_path = _save_checkpoint(algo, candidate_dir)
    if checkpoint_path is None:
        return None

    entry = {
        "rank": 0,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_pass_index": int(state["validation_pass_index"]),
        "validation_env_step": float(env_step),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "validation_metrics": _json_safe_value(dict(validation_metrics)),
        "evaluation_summary": _json_safe_value(_strip_final_metrics(dict(evaluation_summary))),
        "evaluation_seed_rows": _json_safe_value([_strip_final_metrics(dict(row)) for row in evaluation_seed_rows]),
    }
    retained.append(entry)
    retained.sort(key=lambda item: (float(item["metric_value"]), int(item["validation_pass_index"])))
    keep = retained[: int(state["count"])]
    removed = [item for item in retained[int(state["count"]) :] if item not in keep]
    for removed_entry in removed:
        _remove_checkpoint_path(removed_entry["checkpoint_path"], root_dir=state["base_dir"])

    state["retained"] = keep
    _write_best_validation_metadata(state["metadata_path"], state)
    return entry


def _evaluate_with_details(
    cfg: DictConfig,
    run_dir: Path,
    algo,
    algorithm_kind: str,
    logging_cfg,
    *,
    include_validation_metrics: bool = False,
) -> tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    Dict[str, list[Dict[str, float]]],
    Dict[str, list[int]],
    Dict[str, list[Dict[str, float]]],
    Dict[str, Any],
]:
    _sync_env_runner_weights_for_evaluation(algo)
    seed_rows = []
    seed_action_traces = []
    seed_action_space_sizes = []
    seed_phase_queue_traces = []
    seed_artifacts: list[ValidationSeedArtifacts] = []
    eval_seeds = _eval_seeds(cfg)
    policy_mode = _policy_mode(_plain_dict(getattr(cfg.algorithm, "params", {}) or {}))
    for seed_index, seed in enumerate(eval_seeds):
        eval_episode = seed_index + 1
        eval_env = _build_eval_env(
            cfg,
            run_dir,
            seed,
            algorithm_kind=algorithm_kind,
            policy_mode=policy_mode,
        )
        tripinfo_path = _validation_tripinfo_output_path(eval_env)
        base_env = _resolve_sumo_base_env(eval_env)
        has_keep_tripinfo_output = hasattr(base_env, "keep_tripinfo_output")
        original_keep_tripinfo_output = getattr(base_env, "keep_tripinfo_output", None)
        if has_keep_tripinfo_output:
            base_env.keep_tripinfo_output = True
        try:
            episode_reward, action_traces, action_space_sizes, phase_queue_traces = _run_multi_agent_episode_trace(
                algo,
                eval_env,
                seed,
                policy_mode=policy_mode,
            )
        finally:
            # SUMO writes tripinfo XML on close; build summaries only after the
            # file has been flushed so RESCO trip metrics do not become NaN.
            try:
                eval_env.close()
            finally:
                if has_keep_tripinfo_output:
                    base_env.keep_tripinfo_output = original_keep_tripinfo_output
        episode_summary = _get_completed_episode_summary(eval_env)
        seed_artifacts.append(
            _extract_validation_seed_artifacts(
                seed=seed,
                tripinfo_path=tripinfo_path,
                episode_summary=episode_summary,
                action_traces=action_traces,
                action_space_sizes=action_space_sizes,
                phase_queue_traces=phase_queue_traces,
                remove_tripinfo_after_parse=not bool(getattr(logging_cfg, "save_tripinfo_output", False)),
            )
        )

        seed_row = _build_final_eval_summary_row(
            eval_env,
            algorithm_kind=algorithm_kind,
            eval_mean_reward=float(episode_reward),
            eval_std_reward=0.0,
            eval_episodes=1,
            logging_cfg=logging_cfg,
            extra={
                "eval/seed": float(seed),
                "eval/seed_index": float(seed_index),
                "eval/episode": float(eval_episode),
            },
        )
        if include_validation_metrics:
            seed_row.update(_validation_seed_metrics(episode_summary))
        seed_rows.append(seed_row)
        seed_action_traces.append(action_traces)
        seed_action_space_sizes.append(action_space_sizes)
        seed_phase_queue_traces.append(phase_queue_traces)

    eval_mean_reward = float(np.mean([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    eval_std_reward = float(np.std([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    summary = _aggregate_final_eval_rows(
        seed_rows,
        algorithm_kind=algorithm_kind,
        eval_mean_reward=eval_mean_reward,
        eval_std_reward=eval_std_reward,
        eval_episodes=len(eval_seeds),
    )
    summary["eval/episode"] = float(len(eval_seeds))
    action_plot_rows_by_agent: Dict[str, list[Dict[str, float]]] = {}
    action_timeline_by_agent: Dict[str, list[int]] = {}
    phase_queue_rows_by_agent: Dict[str, list[Dict[str, float]]] = {}
    tripinfo_distributions = _aggregate_validation_tripinfo_distributions(seed_artifacts)
    if include_validation_metrics:
        action_plot_rows_by_agent = _build_validation_action_plot_rows(
            seed_action_traces,
            seed_action_space_sizes,
            window_size=_validation_action_window_steps(cfg),
            max_agents=_validation_action_plot_max_agents(logging_cfg),
        )
        action_timeline_by_agent = _build_validation_action_timeline_rows(
            seed_action_traces,
            seed_action_space_sizes,
            max_agents=_validation_action_plot_max_agents(logging_cfg),
        )
        phase_queue_rows_by_agent = _build_validation_phase_queue_rows(
            seed_phase_queue_traces,
            max_agents=_validation_action_plot_max_agents(logging_cfg),
        )
    return summary, seed_rows, action_plot_rows_by_agent, action_timeline_by_agent, phase_queue_rows_by_agent, tripinfo_distributions


def _evaluate(
    cfg: DictConfig,
    run_dir: Path,
    algo,
    algorithm_kind: str,
    logging_cfg,
    *,
    include_validation_metrics: bool = False,
) -> Dict[str, Any]:
    summary, _, _, _, _, _ = _evaluate_with_details(
        cfg,
        run_dir,
        algo,
        algorithm_kind,
        logging_cfg,
        include_validation_metrics=include_validation_metrics,
    )
    return summary


def _validation_summary_row(summary: Dict[str, Any], *, step: int, episode_index: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "validation/env_step": float(step),
        "validation/rollout_index": float(episode_index),
        "validation/episode_index": float(episode_index),
    }
    for key, value in summary.items():
        if key == "algorithm/kind":
            row[key] = value
        elif key in {"validation/env_step", "validation/rollout_index", "validation/episode_index"}:
            continue
        elif key.startswith("validation/"):
            row[key] = value
        elif key.startswith("warnings/"):
            row[f"validation/{key}"] = value
        elif key in {"episode/sim_time_abs", "episode/elapsed_seconds", "eval/episode"}:
            row[f"validation/{key}"] = value
    return row


def _summary_step_from_metrics(metrics: Dict[str, Any]) -> int:
    candidate = metrics.get("train/env_step")
    if isinstance(candidate, (int, float, np.integer, np.floating)) and not isinstance(candidate, bool):
        return int(float(candidate))
    candidate = metrics.get("train/env_steps_sampled")
    if isinstance(candidate, (int, float, np.integer, np.floating)) and not isinstance(candidate, bool):
        return int(float(candidate))
    return 0


def _summary_episode_index_from_metrics(metrics: Dict[str, Any]) -> int:
    candidate = metrics.get("train/rollout_index", metrics.get("train/episode_index", metrics.get("train/episodes_total")))
    if isinstance(candidate, (int, float, np.integer, np.floating)) and not isinstance(candidate, bool):
        return int(float(candidate))
    return 0


def _optional_positive_int(value: Any, *, setting_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{setting_name} must be a positive integer, null, or auto.")
    return parsed


def _rllib_runtime_params(cfg: DictConfig) -> Dict[str, Any]:
    raw_resources = getattr(cfg, "resources", {}) or {}
    if isinstance(raw_resources, DictConfig) or hasattr(raw_resources, "items"):
        resources = _plain_dict(raw_resources)
    elif hasattr(raw_resources, "__dict__"):
        resources = vars(raw_resources)
    else:
        resources = _plain_dict(raw_resources)
    params = _plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
    merged = dict(resources)
    merged.update(params)
    return merged


def _cpu_thread_env(num_threads: Optional[int]) -> Dict[str, str]:
    if num_threads is None:
        return {}
    env_vars = {env_var: str(num_threads) for env_var in _CPU_THREAD_ENV_VARS}
    env_vars["OMP_DYNAMIC"] = "FALSE"
    return env_vars


def _cuda_visible_devices_env(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    raw_value = str(value).strip()
    if raw_value.lower() in {"", "none", "null"}:
        return {}
    return {"CUDA_VISIBLE_DEVICES": raw_value}


def _ray_address(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw_value = str(value).strip()
    if raw_value.lower() in {"", "none", "null"}:
        return None
    return raw_value


def _is_auto_ray_address(address: Optional[str]) -> bool:
    if address is None:
        return False
    return str(address).strip().lower() == "auto"


def _is_existing_ray_address(address: Optional[str]) -> bool:
    if address is None:
        return False
    return str(address).strip().lower() not in {"", "local", "none", "null"}


def _format_ray_resource_value(resources: Dict[str, Any], key: str) -> str:
    value = resources.get(key, 0.0)
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _ray_resource_summary(ray_module: Any) -> Dict[str, Dict[str, Any]]:
    cluster_resources = {}
    available_resources = {}
    try:
        cluster_resources = dict(ray_module.cluster_resources())
    except Exception:
        pass
    try:
        available_resources = dict(ray_module.available_resources())
    except Exception:
        pass
    return {
        "cluster": cluster_resources,
        "available": available_resources,
    }


def _print_ray_resource_summary(ray_module: Any) -> None:
    summary = _ray_resource_summary(ray_module)
    cluster = summary["cluster"]
    available = summary["available"]
    print(
        "RLlib Ray resources: "
        f"cluster_cpu={_format_ray_resource_value(cluster, 'CPU')}, "
        f"cluster_gpu={_format_ray_resource_value(cluster, 'GPU')}, "
        f"available_cpu={_format_ray_resource_value(available, 'CPU')}, "
        f"available_gpu={_format_ray_resource_value(available, 'GPU')}."
    )


def _apply_cpu_thread_limit(num_threads: Optional[int]) -> Dict[str, str]:
    env_vars = _cpu_thread_env(num_threads)
    for env_var, value in env_vars.items():
        os.environ[env_var] = value
    try:
        import torch

        if num_threads is not None:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
    except (ImportError, RuntimeError):
        pass
    return env_vars


def train_rllib(cfg: DictConfig) -> Dict[str, Any]:
    requested_algorithm_kind = str(getattr(cfg.algorithm, "kind", "") or "").strip()
    if requested_algorithm_kind not in SUPPORTED_RLLIB_ALGORITHMS:
        raise ValueError(f"Unsupported RLlib algorithm kind: {requested_algorithm_kind}")
    algorithm_kind = normalize_algorithm_kind(requested_algorithm_kind)
    if algorithm_kind not in SUPPORTED_RLLIB_ALGORITHMS:
        raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")

    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg = cfg.logging
    run_name = _rllib_run_name(cfg, algorithm_kind)
    wandb_run = _init_wandb(cfg, run_dir, run_name=run_name, include_final_metrics=False)
    csv_run = _LocalMetricsCsvLogger(run_dir / "csv" / f"{cfg.experiment.name}.csv")

    import ray

    params = _plain_dict(getattr(cfg.algorithm, "params", {}) or {})
    runtime_params = _rllib_runtime_params(cfg)
    ray_num_cpus = _optional_positive_int(runtime_params.get("ray_num_cpus", 2), setting_name="ray_num_cpus")
    native_num_threads = _optional_positive_int(
        runtime_params.get("native_num_threads", 1),
        setting_name="native_num_threads",
    )
    ray_address = _ray_address(runtime_params.get("ray_address", os.environ.get("RAY_ADDRESS")))
    cpu_thread_env = _apply_cpu_thread_limit(native_num_threads)
    cuda_env = _cuda_visible_devices_env(runtime_params.get("cuda_visible_devices"))
    for env_var, value in cuda_env.items():
        os.environ[env_var] = value
    ray_num_gpus = _resolve_num_gpus(params.get("ray_num_gpus", params.get("num_gpus_per_learner", "auto")))
    runtime_env_vars = dict(cpu_thread_env)
    runtime_env_vars.update(cuda_env)
    ray_init_kwargs: Dict[str, Any] = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }
    ray_init_kwargs["address"] = ray_address if ray_address is not None else "local"
    if not _is_existing_ray_address(ray_address):
        ray_init_kwargs["include_dashboard"] = False
        ray_init_kwargs["num_gpus"] = ray_num_gpus
        if ray_num_cpus is not None:
            ray_init_kwargs["num_cpus"] = ray_num_cpus
    if runtime_env_vars:
        ray_init_kwargs["runtime_env"] = {"env_vars": runtime_env_vars}
    connected_to_existing_cluster = _is_existing_ray_address(ray_address)
    try:
        ray.init(**ray_init_kwargs)
    except ConnectionError:
        if not _is_auto_ray_address(ray_address):
            raise
        fallback_ray_init_kwargs: Dict[str, Any] = {
            "address": "local",
            "ignore_reinit_error": True,
            "log_to_driver": False,
            "include_dashboard": False,
            "num_gpus": ray_num_gpus,
        }
        if ray_num_cpus is not None:
            fallback_ray_init_kwargs["num_cpus"] = ray_num_cpus
        if runtime_env_vars:
            fallback_ray_init_kwargs["runtime_env"] = {"env_vars": runtime_env_vars}
        print("RLlib Ray connection: address=auto but no running Ray cluster was found; starting a local Ray instance.")
        ray.init(**fallback_ray_init_kwargs)
        ray_address = None
        connected_to_existing_cluster = False
    print(
        "RLlib CPU allocation: "
        f"ray_num_cpus={ray_num_cpus if ray_num_cpus is not None else 'auto'}, "
        f"native_num_threads={native_num_threads if native_num_threads is not None else 'preserve-env'}."
    )
    if connected_to_existing_cluster:
        print(
            "RLlib Ray connection: "
            f"address={ray_address}; using existing cluster resources "
            "instead of local ray_num_cpus/ray_num_gpus startup values."
        )
    else:
        print(f"RLlib Ray connection: address={ray_address or 'local'}; local Ray instance is managed by this process.")
    _print_ray_resource_summary(ray)
    if cuda_env:
        print(f"RLlib CUDA_VISIBLE_DEVICES={cuda_env['CUDA_VISIBLE_DEVICES']}.")
    algo = None
    final_summary: Dict[str, Any] = {}
    best_validation_state = _init_best_validation_checkpoint_state(run_dir, algorithm_kind, logging_cfg)
    latest_training_state: Dict[str, int] = {"env_step": 0, "episode_index": 0}
    validation_pass_state: Dict[str, int] = {"index": 0}
    last_validation_state: Dict[str, Any] = {"env_step": None, "episode_index": None, "row": None}
    best_summary_rows: Dict[str, Optional[Dict[str, Any]]] = {"best_train": None, "best_validation": None}
    try:
        config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else config.build()

        def _emit_training_metrics(metrics: Dict[str, Any], step: int) -> None:
            latest_training_state["env_step"] = max(
                int(latest_training_state.get("env_step", 0)),
                int(_summary_step_from_metrics(metrics) or 0),
            )
            latest_training_state["episode_index"] = max(
                int(latest_training_state.get("episode_index", 0)),
                int(_summary_episode_index_from_metrics(metrics) or 0),
            )
            best_summary_rows["best_train"] = _consider_best_metrics_row(
                best_summary_rows.get("best_train"),
                metrics,
                metric_name="train/resco_delay_mean",
            )
            _log_outputs(wandb_run, csv_run, metrics, step=step)

        def _validate_and_log(step: int) -> Dict[str, Any]:
            validation_pass_state["index"] = int(validation_pass_state.get("index", 0)) + 1
            pass_index = int(validation_pass_state["index"])
            (
                evaluation_summary,
                evaluation_seed_rows,
                action_plot_rows_by_agent,
                action_timeline_by_agent,
                phase_queue_rows_by_agent,
                tripinfo_distributions,
            ) = _evaluate_with_details(
                cfg,
                run_dir,
                algo,
                algorithm_kind,
                logging_cfg,
                include_validation_metrics=True,
            )
            episode_index = int(latest_training_state.get("episode_index", 0))
            validation_row = _validation_summary_row(evaluation_summary, step=step, episode_index=episode_index)
            validation_row["validation/pass_index"] = float(pass_index)
            best_summary_rows["best_validation"] = _consider_best_metrics_row(
                best_summary_rows.get("best_validation"),
                validation_row,
                metric_name="validation/resco_delay_mean",
            )
            _log_outputs(wandb_run, csv_run, validation_row, step=step)
            _log_validation_action_plot_images(
                wandb_run,
                action_plot_rows_by_agent,
                action_timeline_by_agent,
                phase_queue_rows_by_agent,
                pass_index=pass_index,
                env_step=step,
                episode_index=episode_index,
                decision_seconds=decision_interval_seconds(cfg),
            )
            _log_validation_tripinfo_distribution_images(
                wandb_run,
                tripinfo_distributions,
                pass_index=pass_index,
                env_step=step,
                episode_index=episode_index,
            )
            _consider_best_validation_checkpoint(
                best_validation_state,
                algo,
                validation_metrics=validation_row,
                evaluation_summary=validation_row,
                evaluation_seed_rows=evaluation_seed_rows,
            )
            last_validation_state["env_step"] = int(step)
            last_validation_state["episode_index"] = int(episode_index)
            last_validation_state["row"] = dict(validation_row)
            return validation_row

        _train_algorithm(
            algo,
            cfg,
            algorithm_kind,
            emit_metrics=_emit_training_metrics,
            validate=lambda metrics, step: _validate_and_log(step),
        )

        final_validation_step = int(latest_training_state.get("env_step", 0))
        final_episode_index = int(latest_training_state.get("episode_index", 0))
        if (
            last_validation_state.get("row") is not None
            and int(last_validation_state.get("env_step") or -1) == final_validation_step
            and int(last_validation_state.get("episode_index") or -1) == final_episode_index
        ):
            final_summary = dict(last_validation_state["row"])
        else:
            final_summary = _validate_and_log(final_validation_step)
        _update_wandb_summary(wandb_run, final_summary)

        if bool(getattr(logging_cfg, "save_final_model", True)):
            checkpoint_dir = run_dir / "checkpoints" / algorithm_kind
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = None
            if hasattr(algo, "save_to_path"):
                checkpoint = algo.save_to_path(str(checkpoint_dir))
            elif hasattr(algo, "save"):
                checkpoint = algo.save(str(checkpoint_dir))
            if checkpoint is not None:
                print(f"[{algorithm_kind}] saved checkpoint to {checkpoint}")
        return final_summary
    finally:
        _update_wandb_best_summary(wandb_run, best_summary_rows)
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
