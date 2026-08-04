from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pformat
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from sumo_rl.experiments.validation_metrics import aggregate_numeric_rows, enrich_seed_row, format_metric_value
from sumo_rl.util.tripinfo import collect_tripinfo_metrics
from sumo_rl.util.tripinfo import is_ghost_vehicle as _is_ghost_vehicle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis validation for RLlib checkpoints and static baselines.")
    parser.add_argument("--controller", choices=("rllib", "fixed_time", "static_max_pressure"), required=True)
    parser.add_argument("--run-dir", help="Hydra run directory for RLlib evaluation.")
    parser.add_argument("--checkpoint-path", help="Explicit checkpoint path for RLlib evaluation.")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory containing one or more checkpoints.")
    parser.add_argument("--checkpoint-selector", choices=("best", "latest"), default="best")
    parser.add_argument("--scenario", help="Scenario override for static controllers.")
    parser.add_argument("--config-name", help="Static-controller config name override.")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra overrides for static controllers.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Validation seeds. One seed equals one run.")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Concurrent seed processes. Defaults to len(seeds).")
    parser.add_argument("--metrics-profile", choices=("thesis-default", "full"), default="thesis-default")
    parser.add_argument("--show-terminal-table", choices=("compact", "compact+fairness", "full"), default="compact")
    parser.add_argument("--disable-completed-view", action="store_true")
    parser.add_argument("--record-seed", type=int, default=None, help="Optional seed to record as an MP4.")
    parser.add_argument("--record-output-dir", default=None, help="Optional output directory for video files.")
    parser.add_argument("--output-dir", default=None, help="Validation artifact output directory.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--diagnostic-junction", default=None, help="Traffic signal id to trace at each RLlib decision step, e.g. gnej143.")
    parser.add_argument(
        "--diagnostic-max-steps",
        type=int,
        default=None,
        help="Optional maximum number of diagnostic rows to keep per seed.",
    )
    parser.add_argument(
        "--diagnostic-demand-ablation",
        choices=("none", "zero"),
        default="zero",
        help="For diagnostic-junction, also recompute the policy action after masking density/queue-like demand features.",
    )
    parser.add_argument(
        "--max-decision-steps",
        type=int,
        default=None,
        help="Optional early stop after this many validation decision steps. Useful for diagnostics.",
    )
    parser.add_argument(
        "--progress-log-steps",
        type=int,
        default=50,
        help="Print validation progress every N decision steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--ray-num-gpus",
        type=float,
        default=None,
        help="Optional override for GPUs exposed to local Ray during RLlib validation. Defaults to the saved config.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Optional override for CPUs exposed to local Ray during RLlib validation. Defaults to the saved config.",
    )
    parser.add_argument(
        "--native-num-threads",
        type=int,
        default=None,
        help="Optional override for native Torch/NumPy/BLAS thread caps during RLlib validation. Defaults to the saved config.",
    )
    return parser.parse_args()


def _resolve_static_cfg(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir

    config_name = args.config_name or ("fixed_time" if args.controller == "fixed_time" else "static_max_pressure")
    overrides = list(args.override or [])
    if args.scenario:
        overrides.append(f"scenario={args.scenario}")
    with initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        return compose(config_name=config_name, overrides=overrides)


def _resolve_eval_cfg(args: argparse.Namespace):
    if args.controller == "rllib":
        if not args.run_dir:
            raise ValueError("--run-dir is required for RLlib validation.")
        return OmegaConf.load(Path(args.run_dir).resolve() / ".hydra" / "config.yaml")
    return _resolve_static_cfg(args)


def _default_seeds(cfg, args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    explicit = getattr(getattr(cfg, "experiment", None), "eval_seeds", None)
    if explicit:
        return [int(seed) for seed in list(explicit)]
    experiment_seed = int(getattr(getattr(cfg, "experiment", None), "seed", 0) or 0)
    return [experiment_seed]


def _resolve_checkpoint_path(cfg, args: argparse.Namespace) -> Path:
    if args.controller != "rllib":
        raise ValueError("Checkpoint resolution only applies to RLlib controller.")
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        return checkpoint_path

    run_dir = Path(args.run_dir).resolve()
    algorithm_kind = str(cfg.algorithm.kind)
    if args.checkpoint_selector == "best":
        metadata_path = run_dir / "checkpoints" / algorithm_kind / "best_validation" / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            retained = list(metadata.get("retained", []))
            if retained:
                raw_path = str(retained[0].get("checkpoint_path", ""))
                candidate = Path(raw_path).expanduser()
                if candidate.exists():
                    return candidate.resolve()
                fallback = metadata_path.parent / candidate.name
                if fallback.exists():
                    return fallback.resolve()
        raise FileNotFoundError(
            "Best-validation checkpoint metadata was not found or empty. "
            "Pass --checkpoint-path or use --checkpoint-selector latest."
        )

    checkpoint_root = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else run_dir / "checkpoints" / algorithm_kind
    candidates = sorted(
        [path for path in checkpoint_root.rglob("*") if path.is_dir() and path.name.startswith("checkpoint")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under {checkpoint_root}")
    return candidates[0].resolve()


def _validation_output_dir(cfg, args: argparse.Namespace, checkpoint_path: Optional[Path]) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scenario_name = str(getattr(getattr(cfg, "scenario", None), "name", "scenario") or "scenario")
    if args.controller == "rllib":
        run_dir = Path(args.run_dir).resolve()
        checkpoint_label = checkpoint_path.name if checkpoint_path is not None else "checkpoint"
        return run_dir / "validation_runs" / f"{scenario_name}__{checkpoint_label}__{timestamp}"
    return ROOT / "outputs" / "validation_runs" / f"{args.controller}__{scenario_name}__{timestamp}"


def _prepare_cfg_for_seed(cfg, *, seed: int, seed_dir: Path, args: argparse.Namespace):
    prepared = deepcopy(cfg)
    OmegaConf.set_struct(prepared, False)
    if getattr(prepared, "logging", None) is None:
        prepared.logging = OmegaConf.create({})
    prepared.logging.save_tripinfo_output = True
    if getattr(prepared, "env", None) is None:
        prepared.env = OmegaConf.create({})
    if getattr(prepared.env, "kwargs", None) is None:
        prepared.env.kwargs = OmegaConf.create({})
    prepared.env.kwargs.out_csv_name = str(seed_dir / "csv" / "validation")
    prepared.env.kwargs.tripinfo_output_name = str(seed_dir / "tripinfo" / "tripinfo")
    prepared.env.kwargs.keep_tripinfo_output = True
    prepared.env.kwargs.statistic_output_name = str(seed_dir / "statistics" / "statistics")
    prepared.env.kwargs.keep_statistic_output = True
    prepared.env.kwargs.use_gui = bool(args.use_gui or args.record_seed == seed)
    if getattr(prepared, "resources", None) is None:
        prepared.resources = OmegaConf.create({})
    if args.ray_num_cpus is not None:
        prepared.resources.ray_num_cpus = int(args.ray_num_cpus)
    if args.native_num_threads is not None:
        prepared.resources.native_num_threads = int(args.native_num_threads)
    prepared.experiment.eval_seeds = [int(seed)]
    prepared.experiment.eval_episodes = 1
    prepared.experiment.seed = int(seed)
    return prepared


def _compact_columns(show_terminal_table: str) -> list[str]:
    columns = [
        "method/controller",
        "scenario",
        "seed",
        "avg_observed_delay_all",
        "avg_observed_system_time_all",
        "completion_ratio",
        "avg_queue_length",
        "avg_waiting_time_completed",
        "running_vehicle_count",
        "undeparted_vehicle_count",
        "jam_teleports",
        "avg_speed_live",
        "status",
    ]
    if show_terminal_table == "compact+fairness":
        columns.extend(["fairness/max_waiting_time_completed", "fairness/gini_waiting_time"])
    if show_terminal_table == "full":
        columns.extend(
            [
                "throughput",
                "max_queue",
                "resource/validation_wall_clock_seconds",
                "resource/p95_action_selection_latency_ms",
                "resource/deadline_ratio_p95",
            ]
        )
    return columns


def _render_table(rows: list[Dict[str, Any]], *, show_terminal_table: str) -> str:
    columns = _compact_columns(show_terminal_table)
    widths = {column: max(len(column), *(len(format_metric_value(row.get(column, ""))) for row in rows)) for column in columns}
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(format_metric_value(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _save_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_tripinfo_series(path: str | Path | None) -> Dict[str, list[float]]:
    from xml.etree import ElementTree as ET

    if not path:
        return {"wait_values": [], "delay_values": []}
    tripinfo_path = Path(path)
    if not tripinfo_path.exists():
        return {"wait_values": [], "delay_values": []}
    try:
        root = ET.parse(tripinfo_path).getroot()
    except (ET.ParseError, OSError):
        return {"wait_values": [], "delay_values": []}
    metrics = collect_tripinfo_metrics(root.findall(".//tripinfo"))
    return {"wait_values": list(metrics.wait_values), "delay_values": list(metrics.delay_values)}


def _run_recording_step(record_state: Dict[str, Any], env: Any, width: int, height: int) -> None:
    from experiments.record_rollout import _capture_frame, _open_video_writer

    if not record_state.get("enabled"):
        return
    screenshot_path = record_state["screenshot_path"]
    frame = _capture_frame(env, screenshot_path=screenshot_path, width=width, height=height)
    writer = record_state.get("writer")
    if writer is None:
        frame_height, frame_width = frame.shape[:2]
        writer = _open_video_writer(Path(record_state["video_path"]), fps=record_state["fps"], width=frame_width, height=frame_height)
        record_state["writer"] = writer
    writer.write(frame)
    record_state["frames_written"] = int(record_state.get("frames_written", 0)) + 1


def _run_rllib_seed_episode(algo, eval_env, *, seed: int, algorithm_kind: str, policy_mode: str, record_state: Dict[str, Any]):
    from sumo_rl.experiments.rllib_runner import (
        _collect_phase_queue_snapshot,
        _compute_single_action,
        _policy_id_for_agent,
        _possible_agents,
    )

    obs, _ = eval_env.reset(seed=seed)
    diagnostic_junction = str(record_state.get("diagnostic_junction") or "")
    diagnostic_phase_lane_override = _apply_diagnostic_phase_lane_override(eval_env, diagnostic_junction)
    diagnostic_max_steps = record_state.get("diagnostic_max_steps")
    max_decision_steps = record_state.get("max_decision_steps")
    progress_log_steps = int(record_state.get("progress_log_steps") or 0)
    diagnostic_rows: list[Dict[str, Any]] = []
    action_traces: Dict[str, list[int]] = {}
    action_space_sizes: Dict[str, int] = {}
    phase_queue_traces: Dict[str, list[Dict[str, Any]]] = {}
    action_latency_seconds: list[float] = []
    total_reward = 0.0
    agent_ids = [str(agent_id) for agent_id in _possible_agents(eval_env) if not str(agent_id).startswith("__")]
    for agent_id in agent_ids:
        action_traces[agent_id] = []
        phase_queue_traces[agent_id] = []
    if diagnostic_junction and diagnostic_junction not in agent_ids:
        raise KeyError(
            f"Diagnostic junction {diagnostic_junction!r} is not an active agent. "
            f"Available agents: {', '.join(agent_ids)}"
        )
    done = False
    decision_steps = 0
    while not done:
        decision_steps += 1
        if progress_log_steps > 0 and (decision_steps == 1 or decision_steps % progress_log_steps == 0):
            print(f"[seed {seed}] validation decision step {decision_steps}", flush=True)
        action_start = time.perf_counter()
        actions = {}
        for agent_id, agent_obs in obs.items():
            if agent_id.startswith("__"):
                continue
            actions[agent_id] = _compute_single_action(
                algo,
                agent_obs,
                policy_id=_policy_id_for_agent(str(agent_id), policy_mode),
                algorithm_kind=algorithm_kind,
            )
        if diagnostic_junction and diagnostic_junction in actions:
            if diagnostic_max_steps is None or len(diagnostic_rows) < int(diagnostic_max_steps):
                ablated_action = None
                ablation_status = "disabled"
                if str(record_state.get("diagnostic_demand_ablation") or "none") != "none":
                    ablated_obs = _ablate_demand_observation(eval_env, diagnostic_junction, obs[diagnostic_junction])
                    if ablated_obs is None:
                        ablation_status = "unsupported_observation"
                    else:
                        ablated_action = _compute_single_action(
                            algo,
                            ablated_obs,
                            policy_id=_policy_id_for_agent(str(diagnostic_junction), policy_mode),
                            algorithm_kind=algorithm_kind,
                        )
                        ablation_status = "ok"
                diagnostic_rows.append(
                    _build_junction_diagnostic_row(
                        eval_env,
                        junction_id=diagnostic_junction,
                        decision_step=decision_steps,
                        seed=seed,
                        chosen_action=actions[diagnostic_junction],
                        ablated_action=ablated_action,
                        ablation_status=ablation_status,
                        phase_lane_override=diagnostic_phase_lane_override,
                    )
                )
        action_latency_seconds.append(time.perf_counter() - action_start)
        for agent_id, action in actions.items():
            action_value = int(np.asarray(action).reshape(-1)[0])
            action_traces[agent_id].append(action_value)
            action_space_sizes[agent_id] = max(int(action_space_sizes.get(agent_id, 0)), action_value + 1)
        obs, rewards, terminations, truncations, _ = eval_env.step(actions)
        total_reward += float(sum(float(value) for value in rewards.values()))
        if diagnostic_rows and diagnostic_rows[-1].get("junction_id") == diagnostic_junction:
            row_decision_step = int(float(diagnostic_rows[-1].get("decision_step", -1)))
            if row_decision_step == decision_steps:
                reward_value = rewards.get(diagnostic_junction)
                diagnostic_rows[-1]["realized_reward"] = "" if reward_value is None else float(reward_value)
        snapshot = _collect_phase_queue_snapshot(eval_env, agent_ids)
        for agent_id, item in snapshot.items():
            phase_queue_traces[agent_id].append(
                {
                    "step": float(len(phase_queue_traces[agent_id]) + 1),
                    "active_phase": int(item["active_phase"]),
                    "phase_queues": [int(value) for value in item["phase_queues"]],
                }
            )
        if record_state.get("enabled"):
            _run_recording_step(record_state, eval_env, record_state["width"], record_state["height"])
        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or all(bool(terminations.get(agent_id, False)) for agent_id in agent_ids)
            or all(bool(truncations.get(agent_id, False)) for agent_id in agent_ids)
        )
        if max_decision_steps is not None and decision_steps >= int(max_decision_steps):
            print(f"[seed {seed}] stopping early after {decision_steps} validation decision steps", flush=True)
            done = True
    return total_reward, action_traces, action_space_sizes, phase_queue_traces, action_latency_seconds, decision_steps, diagnostic_rows


def _argmax(values: list[float]) -> int | None:
    finite_items = [(index, float(value)) for index, value in enumerate(values) if np.isfinite(float(value))]
    if not finite_items:
        return None
    return max(finite_items, key=lambda item: item[1])[0]


def _argmin(values: list[float]) -> int | None:
    finite_items = [(index, float(value)) for index, value in enumerate(values) if np.isfinite(float(value))]
    if not finite_items:
        return None
    return min(finite_items, key=lambda item: item[1])[0]


def _phase_total_waiting_times(traffic_signal) -> list[float]:
    totals = []
    for phase_lanes in getattr(traffic_signal, "phase_lanes", []) or []:
        total_wait = 0.0
        for veh in traffic_signal._get_unique_phase_vehicle_ids(phase_lanes):
            total_wait += float(traffic_signal.sumo.vehicle.getWaitingTime(veh))
        totals.append(total_wait)
    return totals


def _apply_diagnostic_phase_lane_override(eval_env, junction_id: str) -> str:
    """Apply narrow diagnostic-only phase lane corrections for known RESCO issues."""
    if junction_id != "gneJ143":
        return ""

    base_env = _resolve_validation_sumo_base_env(eval_env)
    traffic_signals = getattr(base_env, "traffic_signals", {})
    traffic_signal = traffic_signals.get(junction_id)
    if traffic_signal is None:
        return ""

    phase_lanes = getattr(traffic_signal, "phase_lanes", None)
    if not phase_lanes or len(phase_lanes) <= 2:
        return ""

    replacement_lanes = [
        "10425609#0_1",
        "10425609#0_2",
        "10425609#0_3",
    ]
    known_lanes = _sumo_lane_ids(traffic_signal)
    if known_lanes:
        replacement_lanes = [lane for lane in replacement_lanes if lane in known_lanes]

    if not replacement_lanes:
        return ""

    updated_phase_lanes = [list(lanes) for lanes in phase_lanes]
    updated_phase_lanes[2] = list(dict.fromkeys(replacement_lanes))
    traffic_signal.phase_lanes = updated_phase_lanes
    traffic_signal._phase_stats_cache_step = None
    traffic_signal._phase_stats_cache = None
    return "ingolstadt7_gneJ143_phase_2_10425609_upstream"


def _gnej143_target_lanes(traffic_signal) -> list[str]:
    candidate_lanes = [
        "10425609#0_1",
        "10425609#0_2",
        "10425609#0_3",
        "10425609#1_1",
        "10425609#1_2",
        "10425609#1_3",
    ]
    known_lanes = _sumo_lane_ids(traffic_signal)
    if not known_lanes:
        return candidate_lanes
    return [lane for lane in candidate_lanes if lane in known_lanes]


def _add_gnej143_target_road_counts(row: Dict[str, Any], traffic_signal) -> None:
    """Add raw lane counts matching the visual diagnostic notebook for gneJ143."""
    lane_domain = getattr(getattr(traffic_signal, "sumo", None), "lane", None)
    vehicle_domain = getattr(getattr(traffic_signal, "sumo", None), "vehicle", None)
    if lane_domain is None or vehicle_domain is None:
        return

    lane_ids = _gnej143_target_lanes(traffic_signal)
    row["diagnostic_target_lanes"] = json.dumps(lane_ids)
    row["diagnostic_target_edges"] = json.dumps(["10425609#0", "10425609#1"])

    edge_vehicle_counts = {"10425609#0": 0, "10425609#1": 0}
    edge_queue_counts = {"10425609#0": 0, "10425609#1": 0}
    edge_raw_vehicle_counts = {"10425609#0": 0, "10425609#1": 0}
    edge_raw_queue_counts = {"10425609#0": 0, "10425609#1": 0}
    for lane in lane_ids:
        raw_vehicle_ids = [str(vehicle_id) for vehicle_id in lane_domain.getLastStepVehicleIDs(lane)]
        vehicle_ids = [vehicle_id for vehicle_id in raw_vehicle_ids if not _is_ghost_vehicle(vehicle_id)]
        raw_stopped_vehicle_ids = [
            vehicle_id for vehicle_id in raw_vehicle_ids if float(vehicle_domain.getSpeed(vehicle_id)) < 0.1
        ]
        stopped_vehicle_ids = [vehicle_id for vehicle_id in raw_stopped_vehicle_ids if not _is_ghost_vehicle(vehicle_id)]

        row[f"target_lane/{lane}/vehicle_count"] = len(vehicle_ids)
        row[f"target_lane/{lane}/queue_count"] = len(stopped_vehicle_ids)
        row[f"target_lane/{lane}/raw_vehicle_count"] = len(raw_vehicle_ids)
        row[f"target_lane/{lane}/raw_queue_count"] = len(raw_stopped_vehicle_ids)
        row[f"target_lane/{lane}/vehicle_ids"] = json.dumps(vehicle_ids)
        row[f"target_lane/{lane}/stopped_vehicle_ids"] = json.dumps(stopped_vehicle_ids)

        edge = lane.rsplit("_", 1)[0]
        if edge not in edge_vehicle_counts:
            continue
        edge_vehicle_counts[edge] += len(vehicle_ids)
        edge_queue_counts[edge] += len(stopped_vehicle_ids)
        edge_raw_vehicle_counts[edge] += len(raw_vehicle_ids)
        edge_raw_queue_counts[edge] += len(raw_stopped_vehicle_ids)

    for edge in ("10425609#0", "10425609#1"):
        row[f"target_edge/{edge}/vehicle_count"] = edge_vehicle_counts[edge]
        row[f"target_edge/{edge}/queue_count"] = edge_queue_counts[edge]
        row[f"target_edge/{edge}/raw_vehicle_count"] = edge_raw_vehicle_counts[edge]
        row[f"target_edge/{edge}/raw_queue_count"] = edge_raw_queue_counts[edge]


def _sumo_lane_ids(traffic_signal) -> set[str]:
    lane_domain = getattr(getattr(traffic_signal, "sumo", None), "lane", None)
    get_id_list = getattr(lane_domain, "getIDList", None)
    if not callable(get_id_list):
        return set()
    return {str(lane_id) for lane_id in get_id_list()}


def _ablate_default_vector_demand(obs: Any, traffic_signal) -> Any | None:
    array = np.asarray(obs)
    if array.ndim != 1:
        return None
    num_phases = int(getattr(traffic_signal, "num_green_phases", 0) or 0)
    lane_count = len(getattr(traffic_signal, "lanes", []) or [])
    demand_start = num_phases + 1
    demand_width = 2 * lane_count
    if num_phases <= 0 or demand_width <= 0 or array.shape[0] < demand_start + demand_width:
        return None
    ablated = np.array(array, copy=True)
    ablated[demand_start : demand_start + demand_width] = 0.0
    return ablated


def _ablate_demand_observation(eval_env, junction_id: str, agent_obs: Any) -> Any | None:
    from sumo_rl.experiments.rllib_runner import _resolve_sumo_base_env

    base_env = _resolve_sumo_base_env(eval_env)
    traffic_signal = base_env.traffic_signals[junction_id]
    vector_ablation = _ablate_default_vector_demand(agent_obs, traffic_signal)
    if vector_ablation is not None:
        return vector_ablation
    graph_history_ablation = _ablate_graph_history_demand(eval_env, junction_id, agent_obs)
    if graph_history_ablation is not None:
        return graph_history_ablation
    if not isinstance(agent_obs, dict) or "node_features" not in agent_obs:
        return None

    node_features = np.asarray(agent_obs["node_features"])
    if node_features.ndim != 2:
        return None
    agent_to_index = getattr(eval_env, "_agent_to_index", None)
    if not isinstance(agent_to_index, dict):
        agent_to_index = getattr(getattr(eval_env, "env", None), "_agent_to_index", None)
    node_index = agent_to_index.get(junction_id) if isinstance(agent_to_index, dict) else None
    if node_index is None or int(node_index) >= node_features.shape[0]:
        ego_index = agent_obs.get("ego_index")
        try:
            node_index = int(np.asarray(ego_index).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None

    ablated = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in agent_obs.items()}
    node_features_copy = np.array(node_features, copy=True)
    local_ablation = _ablate_default_vector_demand(node_features_copy[int(node_index)], traffic_signal)
    if local_ablation is None:
        return None
    node_features_copy[int(node_index)] = local_ablation
    ablated["node_features"] = node_features_copy
    return ablated


def _ablate_graph_history_demand(eval_env, junction_id: str, agent_obs: Any) -> Any | None:
    array = np.asarray(agent_obs)
    if array.ndim != 3:
        return None
    graph = _find_env_attr(eval_env, "graph")
    ts_index = getattr(graph, "ts_index", None)
    node_index = ts_index.get(junction_id) if isinstance(ts_index, dict) else None
    if node_index is None or int(node_index) >= array.shape[1]:
        return None
    density_offset = getattr(graph, "density_offset", None)
    queue_offset = getattr(graph, "queue_offset", None)
    max_lanes = getattr(graph, "max_lanes", None)
    if density_offset is None or queue_offset is None or max_lanes is None:
        return None
    ablated = np.array(array, copy=True)
    node_index = int(node_index)
    max_lanes = int(max_lanes)
    ablated[:, node_index, int(density_offset) : int(density_offset) + max_lanes] = 0.0
    ablated[:, node_index, int(queue_offset) : int(queue_offset) + max_lanes] = 0.0
    return ablated


def _find_env_attr(env: Any, attr_name: str) -> Any:
    queue = [env]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        value = getattr(current, attr_name, None)
        if value is not None:
            return value
        for child_attr in ("base_env", "env", "aec_env", "unwrapped", "gym_env", "par_env", "venv"):
            candidate = getattr(current, child_attr, None)
            if candidate is not None and candidate is not current:
                queue.append(candidate)
    return None


def _resolve_validation_sumo_base_env(env: Any) -> Any:
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
        for child_attr in ("base_env", "env", "aec_env", "unwrapped", "gym_env", "par_env", "venv"):
            candidate = getattr(current, child_attr, None)
            if candidate is not None and candidate is not current:
                queue.append(candidate)
    return fallback


def _build_junction_diagnostic_row(
    eval_env,
    *,
    junction_id: str,
    decision_step: int,
    seed: int,
    chosen_action: Any,
    ablated_action: Any = None,
    ablation_status: str = "disabled",
    phase_lane_override: str = "",
) -> Dict[str, Any]:
    base_env = _resolve_validation_sumo_base_env(eval_env)
    traffic_signal = base_env.traffic_signals[junction_id]
    chosen_phase = int(np.asarray(chosen_action).reshape(-1)[0])
    ablated_phase = ""
    if ablated_action is not None:
        ablated_phase = int(np.asarray(ablated_action).reshape(-1)[0])
    phase_vehicle_counts = [len(traffic_signal._get_unique_phase_vehicle_ids(lanes)) for lanes in traffic_signal.phase_lanes]
    phase_queue_counts = [int(value) for value in traffic_signal.get_phase_queued_counts()]
    phase_average_speeds = [float(value) for value in traffic_signal.get_phase_average_speeds()]
    get_windowed_stats = getattr(traffic_signal, "get_windowed_phase_speed_wait_stats", None)
    if callable(get_windowed_stats):
        window_average_speeds, window_max_waiting_times = get_windowed_stats()
    else:
        window_average_speeds = list(phase_average_speeds)
        window_max_waiting_times = [float(value) for value in traffic_signal.get_phase_max_waiting_times()]
    get_windowed_vehicle_counts = getattr(traffic_signal, "get_windowed_phase_vehicle_counts", None)
    if callable(get_windowed_vehicle_counts):
        window_vehicle_counts = get_windowed_vehicle_counts()
    else:
        window_vehicle_counts = list(phase_vehicle_counts)
    window_average_speeds = [float(value) for value in window_average_speeds]
    window_max_waiting_times = [float(value) for value in window_max_waiting_times]
    window_vehicle_counts = [float(value) for value in window_vehicle_counts]
    phase_total_waiting_times = _phase_total_waiting_times(traffic_signal)

    argmax_count = _argmax([float(value) for value in phase_vehicle_counts])
    argmax_queue = _argmax([float(value) for value in phase_queue_counts])
    argmin_current_speed = _argmin(phase_average_speeds)
    argmin_window_speed = _argmin(window_average_speeds)
    argmax_total_wait = _argmax(phase_total_waiting_times)
    argmax_window_max_wait = _argmax(window_max_waiting_times)

    row: Dict[str, Any] = {
        "seed": int(seed),
        "sim_step": float(getattr(base_env, "sim_step", float("nan"))),
        "decision_step": int(decision_step),
        "junction_id": junction_id,
        "phase_lane_override": phase_lane_override,
        "active_phase_before": int(getattr(traffic_signal, "green_phase", -1)),
        "chosen_phase": chosen_phase,
        "demand_ablation_status": ablation_status,
        "demand_ablated_phase": ablated_phase,
        "demand_ablation_changed_action": "" if ablated_phase == "" else chosen_phase != ablated_phase,
        "argmax_vehicle_count": "" if argmax_count is None else int(argmax_count),
        "argmax_queue_count": "" if argmax_queue is None else int(argmax_queue),
        "argmin_current_avg_speed": "" if argmin_current_speed is None else int(argmin_current_speed),
        "argmin_window_avg_speed": "" if argmin_window_speed is None else int(argmin_window_speed),
        "argmax_total_wait": "" if argmax_total_wait is None else int(argmax_total_wait),
        "argmax_window_max_wait": "" if argmax_window_max_wait is None else int(argmax_window_max_wait),
        "chosen_is_argmax_vehicle_count": chosen_phase == argmax_count,
        "chosen_is_argmax_queue_count": chosen_phase == argmax_queue,
        "chosen_is_argmin_current_avg_speed": chosen_phase == argmin_current_speed,
        "chosen_is_argmin_window_avg_speed": chosen_phase == argmin_window_speed,
        "chosen_is_argmax_total_wait": chosen_phase == argmax_total_wait,
        "chosen_is_argmax_window_max_wait": chosen_phase == argmax_window_max_wait,
        "phase_vehicle_counts": json.dumps(phase_vehicle_counts),
        "phase_window_vehicle_counts": json.dumps(window_vehicle_counts),
        "phase_queue_counts": json.dumps(phase_queue_counts),
        "phase_current_avg_speeds": json.dumps(phase_average_speeds),
        "phase_window_avg_speeds": json.dumps(window_average_speeds),
        "phase_total_waiting_times": json.dumps(phase_total_waiting_times),
        "phase_window_max_waiting_times": json.dumps(window_max_waiting_times),
        "reward_name": str(getattr(traffic_signal, "reward_fn", "")),
        "reward_nsw_window_seconds": int(getattr(traffic_signal, "reward_nsw_window_seconds", 0)),
        "reward_nash_epsilon": float(getattr(traffic_signal, "reward_nash_epsilon", float("nan"))),
        "realized_reward": "",
    }
    for phase_index in range(max(len(phase_vehicle_counts), len(window_average_speeds))):
        row[f"phase_{phase_index}/vehicle_count"] = phase_vehicle_counts[phase_index] if phase_index < len(phase_vehicle_counts) else ""
        row[f"phase_{phase_index}/window_vehicle_count"] = (
            window_vehicle_counts[phase_index] if phase_index < len(window_vehicle_counts) else ""
        )
        row[f"phase_{phase_index}/queue_count"] = phase_queue_counts[phase_index] if phase_index < len(phase_queue_counts) else ""
        row[f"phase_{phase_index}/current_avg_speed"] = phase_average_speeds[phase_index] if phase_index < len(phase_average_speeds) else ""
        row[f"phase_{phase_index}/window_avg_speed"] = window_average_speeds[phase_index] if phase_index < len(window_average_speeds) else ""
        row[f"phase_{phase_index}/total_wait"] = phase_total_waiting_times[phase_index] if phase_index < len(phase_total_waiting_times) else ""
        row[f"phase_{phase_index}/window_max_wait"] = window_max_waiting_times[phase_index] if phase_index < len(window_max_waiting_times) else ""
    if junction_id == "gneJ143":
        _add_gnej143_target_road_counts(row, traffic_signal)
    return row


def _run_static_seed_episode(env, *, policy, record_state: Dict[str, Any]):
    from sumo_rl.experiments.runner import _fixed_time_step_action, _get_base_env

    base_env = _get_base_env(env)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    agent_ids = [str(agent_id) for agent_id in getattr(base_env, "ts_ids", [])]
    action_traces = {agent_id: [] for agent_id in agent_ids}
    phase_queue_traces = {agent_id: [] for agent_id in agent_ids}
    action_space_sizes = {agent_id: len(getattr(base_env.traffic_signals[agent_id], "green_phases", []) or []) for agent_id in agent_ids}
    action_latency_seconds: list[float] = []
    total_reward = 0.0
    done = False
    decision_steps = 0
    from sumo_rl.experiments.rllib_runner import _collect_phase_queue_snapshot

    while not done:
        decision_steps += 1
        action_start = time.perf_counter()
        if policy is None:
            actions = _fixed_time_step_action(env, base_env, agent_ids)
        else:
            actions = {agent_id: policy.select_action(base_env.traffic_signals[agent_id]) for agent_id in agent_ids}
        action_latency_seconds.append(time.perf_counter() - action_start)
        next_step = env.step(actions)
        for agent_id, action in (actions or {}).items():
            action_traces[agent_id].append(int(action))
        total_reward += float(next_step[1]) if len(next_step) == 5 and not isinstance(next_step[1], dict) else float(sum(float(v) for v in (next_step[1] or {}).values()))
        snapshot = _collect_phase_queue_snapshot(env, agent_ids)
        for agent_id, item in snapshot.items():
            phase_queue_traces[agent_id].append(
                {
                    "step": float(len(phase_queue_traces[agent_id]) + 1),
                    "active_phase": int(item["active_phase"]),
                    "phase_queues": [int(value) for value in item["phase_queues"]],
                }
            )
        if record_state.get("enabled"):
            _run_recording_step(record_state, env, record_state["width"], record_state["height"])
        if len(next_step) == 5:
            _, rewards, terminated, truncated, _ = next_step
            if isinstance(terminated, dict):
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            else:
                done = bool(terminated or truncated)
        else:
            _, _, dones, _ = next_step
            done = bool(dones.get("__all__", False))
    return base_env, total_reward, action_traces, action_space_sizes, phase_queue_traces, action_latency_seconds, decision_steps


def _run_seed_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(payload["seed"])
    seed_dir = Path(payload["seed_dir"])
    seed_dir.mkdir(parents=True, exist_ok=True)
    record_enabled = bool(payload.get("record_enabled", False))
    record_output_dir = Path(payload["record_output_dir"]).resolve()
    record_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_ctx = tempfile.TemporaryDirectory(prefix=f"validation_seed_{seed}_")
    record_state = {
        "enabled": record_enabled,
        "writer": None,
        "fps": int(payload["fps"]),
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "video_path": str((record_output_dir / f"seed_{seed}.mp4").resolve()),
        "screenshot_path": Path(temp_dir_ctx.name) / "frame.png",
        "frames_written": 0,
        "diagnostic_junction": payload.get("diagnostic_junction") or "",
        "diagnostic_max_steps": payload.get("diagnostic_max_steps"),
        "diagnostic_demand_ablation": payload.get("diagnostic_demand_ablation") or "none",
        "max_decision_steps": payload.get("max_decision_steps"),
        "progress_log_steps": payload.get("progress_log_steps"),
    }
    try:
        if payload["controller"] == "rllib":
            return _run_rllib_seed_worker(payload, seed_dir, record_state)
        return _run_static_seed_worker(payload, seed_dir, record_state)
    except Exception:
        return {
            "seed": seed,
            "status": "error",
            "error": traceback.format_exc(),
            "row": {"seed": float(seed), "status": "error"},
            "artifacts": {},
        }
    finally:
        writer = record_state.get("writer")
        if writer is not None:
            writer.release()
        temp_dir_ctx.cleanup()


def _run_rllib_seed_worker(payload: Dict[str, Any], seed_dir: Path, record_state: Dict[str, Any]) -> Dict[str, Any]:
    import ray

    from sumo_rl.agents.rllib_common import plain_dict as _plain_dict, policy_mode as _policy_mode
    from sumo_rl.experiments.runner import _build_final_eval_summary_row
    from sumo_rl.experiments.rllib_runner import (
        _build_algorithm_config,
        _build_eval_env,
        _init_rllib_ray_runtime,
        _resolve_sumo_base_env,
        _restore_checkpoint,
        _sync_env_runner_weights_for_evaluation,
        decision_interval_seconds,
    )

    overall_start = time.perf_counter()
    cfg = _prepare_cfg_for_seed(OmegaConf.load(Path(payload["run_dir"]) / ".hydra" / "config.yaml"), seed=payload["seed"], seed_dir=seed_dir, args=argparse.Namespace(**payload["args"]))
    algorithm_kind = str(cfg.algorithm.kind)
    checkpoint_path = Path(payload["checkpoint_path"]).resolve()
    ray_init_start = time.perf_counter()
    ray_runtime = _init_rllib_ray_runtime(cfg, ray, ray_num_gpus_override=payload.get("ray_num_gpus"))
    ray_init_seconds = time.perf_counter() - ray_init_start
    algo = None
    eval_env = None
    try:
        algo_config = _build_algorithm_config(cfg, Path(payload["run_dir"]).resolve(), algorithm_kind)
        build_algo = getattr(algo_config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else algo_config.build()
        restore_start = time.perf_counter()
        _restore_checkpoint(algo, checkpoint_path)
        checkpoint_restore_seconds = time.perf_counter() - restore_start
        _sync_env_runner_weights_for_evaluation(algo)
        params = _plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
        mode = _policy_mode(params)
        env_build_start = time.perf_counter()
        eval_env = _build_eval_env(cfg, Path(payload["run_dir"]).resolve(), payload["seed"], algorithm_kind=algorithm_kind, policy_mode=mode)
        env_build_seconds = time.perf_counter() - env_build_start
        evaluation_start = time.perf_counter()
        (
            episode_reward,
            action_traces,
            action_space_sizes,
            phase_queue_traces,
            action_latency_seconds,
            decision_steps,
            diagnostic_rows,
        ) = _run_rllib_seed_episode(
            algo,
            eval_env,
            seed=payload["seed"],
            algorithm_kind=algorithm_kind,
            policy_mode=mode,
            record_state=record_state,
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        eval_env.close()
        base_env = _resolve_sumo_base_env(eval_env)
        tripinfo_path = base_env._build_tripinfo_output_path()
        statistic_output_path = base_env._build_statistic_output_path()
        try:
            base_env.finalize_episode_summary(parse_tripinfo=True)
        except Exception:
            pass
        seed_row = _build_final_eval_summary_row(
            eval_env,
            algorithm_kind=algorithm_kind,
            eval_mean_reward=float(episode_reward),
            eval_std_reward=0.0,
            eval_episodes=1,
            logging_cfg=cfg.logging,
            extra={
                "eval/seed": float(payload["seed"]),
                "eval/seed_index": 0.0,
                "eval/episode": 1.0,
            },
        )
        resource_timings = {
            "validation_wall_clock_seconds": time.perf_counter() - overall_start,
            "ray_init_seconds": ray_init_seconds,
            "ray_num_cpus": ray_runtime.get("ray_num_cpus"),
            "ray_num_gpus": ray_runtime.get("ray_num_gpus"),
            "checkpoint_restore_seconds": checkpoint_restore_seconds,
            "env_build_seconds": env_build_seconds,
            "evaluation_seconds": evaluation_seconds,
            "video_record_seconds": evaluation_seconds if record_state.get("enabled") else 0.0,
            "decision_steps": decision_steps,
            "early_stop_decision_steps": payload.get("max_decision_steps"),
            "action_latency_seconds": action_latency_seconds,
            "control_interval_seconds": decision_interval_seconds(cfg),
        }
        row = enrich_seed_row(
            seed_row=seed_row,
            base_env=base_env,
            statistic_output_path=statistic_output_path,
            phase_queue_traces=phase_queue_traces,
            resource_timings=resource_timings,
            controller="rllib",
            scenario=str(cfg.scenario.name),
            seed=payload["seed"],
            checkpoint_label=checkpoint_path.name,
            disable_completed_view=bool(payload.get("disable_completed_view", False)),
        )
        row["status"] = "ok"
        row["tripinfo_path"] = str(tripinfo_path) if tripinfo_path is not None else ""
        row["statistic_output_path"] = str(statistic_output_path) if statistic_output_path is not None else ""
        if record_state.get("enabled"):
            row["record/video_path"] = record_state["video_path"]
        diagnostic_path = ""
        if diagnostic_rows:
            diagnostic_dir = seed_dir / "diagnostics"
            diagnostic_path = str((diagnostic_dir / f"{payload.get('diagnostic_junction')}_decisions.csv").resolve())
            _save_csv(Path(diagnostic_path), diagnostic_rows)
            _save_json(diagnostic_dir / f"{payload.get('diagnostic_junction')}_decisions.json", diagnostic_rows)
            row["diagnostic/decision_trace_path"] = diagnostic_path
        return {
            "seed": payload["seed"],
            "status": "ok",
            "row": row,
            "artifacts": {
                "action_traces": action_traces,
                "action_space_sizes": action_space_sizes,
                "phase_queue_traces": phase_queue_traces,
                "tripinfo_series": _collect_tripinfo_series(tripinfo_path),
                "junction_diagnostics": diagnostic_rows,
                "junction_diagnostics_path": diagnostic_path,
            },
        }
    finally:
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()


def _run_static_seed_worker(payload: Dict[str, Any], seed_dir: Path, record_state: Dict[str, Any]) -> Dict[str, Any]:
    from sumo_rl.agents.static import MaxPressurePolicy
    from sumo_rl.experiments.runner import _build_env, _build_final_eval_summary_row
    from sumo_rl.agents.rllib_common import decision_interval_seconds

    overall_start = time.perf_counter()
    args_ns = argparse.Namespace(**payload["args"])
    cfg = _prepare_cfg_for_seed(_resolve_static_cfg(args_ns), seed=payload["seed"], seed_dir=seed_dir, args=args_ns)
    policy = None if payload["controller"] == "fixed_time" else MaxPressurePolicy()
    env_build_start = time.perf_counter()
    env = _build_env(cfg, seed_dir, seed=payload["seed"])
    env_build_seconds = time.perf_counter() - env_build_start
    from sumo_rl.experiments.runner import _get_base_env

    evaluation_start = time.perf_counter()
    base_env, episode_reward, action_traces, action_space_sizes, phase_queue_traces, action_latency_seconds, decision_steps = _run_static_seed_episode(
        env,
        policy=policy,
        record_state=record_state,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    env.close()
    tripinfo_path = base_env._build_tripinfo_output_path()
    statistic_output_path = base_env._build_statistic_output_path()
    try:
        base_env.finalize_episode_summary(parse_tripinfo=True)
    except Exception:
        pass
    seed_row = _build_final_eval_summary_row(
        env,
        algorithm_kind=payload["controller"],
        eval_mean_reward=float(episode_reward),
        eval_std_reward=0.0,
        eval_episodes=1,
        logging_cfg=cfg.logging,
        extra={"eval/seed": float(payload["seed"]), "eval/seed_index": 0.0, "eval/episode": 1.0},
    )
    resource_timings = {
        "validation_wall_clock_seconds": time.perf_counter() - overall_start,
        "ray_init_seconds": 0.0,
        "checkpoint_restore_seconds": 0.0,
        "env_build_seconds": env_build_seconds,
        "evaluation_seconds": evaluation_seconds,
        "video_record_seconds": evaluation_seconds if record_state.get("enabled") else 0.0,
        "decision_steps": decision_steps,
        "action_latency_seconds": action_latency_seconds,
        "control_interval_seconds": decision_interval_seconds(cfg),
    }
    row = enrich_seed_row(
        seed_row=seed_row,
        base_env=base_env,
        statistic_output_path=statistic_output_path,
        phase_queue_traces=phase_queue_traces,
        resource_timings=resource_timings,
        controller=payload["controller"],
        scenario=str(cfg.scenario.name),
        seed=payload["seed"],
        disable_completed_view=bool(payload.get("disable_completed_view", False)),
    )
    row["status"] = "ok"
    row["tripinfo_path"] = str(tripinfo_path) if tripinfo_path is not None else ""
    row["statistic_output_path"] = str(statistic_output_path) if statistic_output_path is not None else ""
    if record_state.get("enabled"):
        row["record/video_path"] = record_state["video_path"]
    return {
        "seed": payload["seed"],
        "status": "ok",
        "row": row,
        "artifacts": {
            "action_traces": action_traces,
            "action_space_sizes": action_space_sizes,
            "phase_queue_traces": phase_queue_traces,
            "tripinfo_series": _collect_tripinfo_series(tripinfo_path),
        },
    }


def _render_metric_bar_panel(title: str, rows: list[Dict[str, Any]], metric_keys: list[str], output_path: Path) -> None:
    width, height = 1100, 420
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), title, fill=(28, 37, 54), font=font)
    if not rows or not metric_keys:
        image.save(output_path)
        return
    left, top, right, bottom = 70, 80, width - 30, height - 50
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    labels = [f"seed {int(row.get('seed', idx + 1))}" for idx, row in enumerate(rows)]
    values = [max((_finite(row.get(metric, float("nan"))) for metric in metric_keys), default=float("nan")) for row in rows]
    finite_values = [value for value in values if np.isfinite(value)]
    y_max = max(1.0, max(finite_values) if finite_values else 1.0)
    colors = [(49, 130, 206), (255, 140, 66), (48, 181, 90), (235, 87, 87)]
    for tick in np.linspace(0.0, y_max, 5):
        y = top + (1.0 - (tick / y_max)) * plot_height
        draw.line((left, y, right, y), fill=(232, 238, 244), width=1)
        draw.text((18, y - 6), format_metric_value(tick), fill=(94, 105, 122), font=font)
    group_width = plot_width / max(1, len(rows))
    bar_width = max(8, int(group_width / max(2, len(metric_keys) + 1)))
    for row_index, row in enumerate(rows):
        base_x = left + row_index * group_width + 10
        for metric_index, metric in enumerate(metric_keys):
            value = _finite(row.get(metric))
            if not np.isfinite(value):
                continue
            bar_left = int(base_x + metric_index * (bar_width + 6))
            bar_right = int(bar_left + bar_width)
            bar_top = int(top + (1.0 - (value / y_max)) * plot_height)
            draw.rectangle((bar_left, bar_top, bar_right, bottom), fill=colors[metric_index % len(colors)])
        draw.text((int(base_x), bottom + 8), labels[row_index], fill=(70, 79, 94), font=font)
    legend_y = 40
    for metric_index, metric in enumerate(metric_keys):
        x = 24 + metric_index * 240
        color = colors[metric_index % len(colors)]
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=color)
        draw.text((x + 18, legend_y), metric, fill=(70, 79, 94), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _save_validation_plots(output_dir: Path, cfg, successful: list[Dict[str, Any]], results: list[Dict[str, Any]]) -> None:
    from sumo_rl.experiments.rllib_runner import (
        _build_validation_action_plot_rows,
        _build_validation_action_timeline_rows,
        _build_validation_phase_queue_rows,
        _render_validation_action_plot_image,
        _render_validation_action_timeline_image,
        _render_validation_phase_queue_image,
        _render_validation_tripinfo_distribution_image,
        decision_interval_seconds,
    )

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    action_traces = [item["artifacts"]["action_traces"] for item in results if item.get("status") == "ok"]
    action_sizes = [item["artifacts"]["action_space_sizes"] for item in results if item.get("status") == "ok"]
    phase_queues = [item["artifacts"]["phase_queue_traces"] for item in results if item.get("status") == "ok"]
    trip_wait_series = [item["artifacts"]["tripinfo_series"]["wait_values"] for item in results if item.get("status") == "ok"]
    trip_delay_series = [item["artifacts"]["tripinfo_series"]["delay_values"] for item in results if item.get("status") == "ok"]

    action_plot_rows = _build_validation_action_plot_rows(action_traces, action_sizes, window_size=20)
    action_timelines = _build_validation_action_timeline_rows(action_traces, action_sizes)
    phase_queue_rows = _build_validation_phase_queue_rows(phase_queues)
    for agent_id, rows in action_plot_rows.items():
        _render_validation_action_plot_image(agent_id, rows).save(plot_dir / f"actions_share_{agent_id}.png")
    for agent_id, timeline in action_timelines.items():
        _render_validation_action_timeline_image(agent_id, timeline, decision_seconds=decision_interval_seconds(cfg)).save(
            plot_dir / f"actions_timeline_{agent_id}.png"
        )
    for agent_id, rows in phase_queue_rows.items():
        _render_validation_phase_queue_image(agent_id, rows, decision_seconds=decision_interval_seconds(cfg)).save(
            plot_dir / f"phase_queue_{agent_id}.png"
        )
    pooled_wait = [value for series in trip_wait_series for value in series]
    pooled_delay = [value for series in trip_delay_series for value in series]
    _render_validation_tripinfo_distribution_image(
        "waiting time",
        trip_wait_series,
        pooled_wait,
        total_seeds=len(trip_wait_series),
        seeds_with_completed_trips=sum(1 for series in trip_wait_series if series),
        total_completed_trips=len(pooled_wait),
        total_unfinished_trips=int(sum(_finite(row.get("tripinfo/unfinished_count")) for row in successful if np.isfinite(_finite(row.get("tripinfo/unfinished_count"))))),
    ).save(plot_dir / "tripinfo_wait_distribution.png")
    _render_validation_tripinfo_distribution_image(
        "delay",
        trip_delay_series,
        pooled_delay,
        total_seeds=len(trip_delay_series),
        seeds_with_completed_trips=sum(1 for series in trip_delay_series if series),
        total_completed_trips=len(pooled_delay),
        total_unfinished_trips=int(sum(_finite(row.get("tripinfo/unfinished_count")) for row in successful if np.isfinite(_finite(row.get("tripinfo/unfinished_count"))))),
    ).save(plot_dir / "tripinfo_delay_distribution.png")
    _render_metric_bar_panel(
        "Headline all-demand metrics",
        successful,
        ["avg_observed_delay_all", "avg_observed_system_time_all", "completion_ratio"],
        plot_dir / "headline_all_demand.png",
    )
    _render_metric_bar_panel(
        "Completed vs all-demand comparison",
        successful,
        ["avg_delay_completed", "avg_observed_delay_all", "avg_observed_system_time_all"],
        plot_dir / "completed_vs_all_demand.png",
    )
    _render_metric_bar_panel(
        "Throughput and completion",
        successful,
        ["arrived_vehicle_count", "running_vehicle_count", "undeparted_vehicle_count", "completion_ratio"],
        plot_dir / "throughput_completion.png",
    )
    _render_metric_bar_panel(
        "Fairness metrics",
        successful,
        ["fairness/max_waiting_time_completed", "fairness/std_waiting_time_completed", "fairness/gini_waiting_time"],
        plot_dir / "fairness.png",
    )
    _render_metric_bar_panel(
        "Resource metrics",
        successful,
        ["resource/validation_wall_clock_seconds", "resource/evaluation_seconds", "resource/p95_action_selection_latency_ms", "resource/deadline_ratio_p95"],
        plot_dir / "resource.png",
    )


def main() -> None:
    args = _parse_args()
    cfg = _resolve_eval_cfg(args)
    checkpoint_path = _resolve_checkpoint_path(cfg, args) if args.controller == "rllib" else None
    output_dir = _validation_output_dir(cfg, args, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _default_seeds(cfg, args)
    workers = int(args.parallel_workers or len(seeds) or 1)
    payloads = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{int(seed)}"
        payloads.append(
            {
                "controller": args.controller,
                "seed": int(seed),
                "seed_dir": str(seed_dir),
                "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else "",
                "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else "",
                "record_enabled": args.record_seed is not None and int(args.record_seed) == int(seed),
                "record_output_dir": str((Path(args.record_output_dir).expanduser().resolve() if args.record_output_dir else (output_dir / "videos")).resolve()),
                "fps": int(args.fps),
                "width": int(args.width),
                "height": int(args.height),
                "disable_completed_view": bool(args.disable_completed_view),
                "diagnostic_junction": str(args.diagnostic_junction or ""),
                "diagnostic_max_steps": args.diagnostic_max_steps,
                "diagnostic_demand_ablation": str(args.diagnostic_demand_ablation or "none"),
                "max_decision_steps": args.max_decision_steps,
                "progress_log_steps": args.progress_log_steps,
                "ray_num_gpus": None if args.ray_num_gpus is None else float(args.ray_num_gpus),
                "args": vars(args),
            }
        )

    print(f"Validation output dir: {output_dir}")
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Seeds: {seeds}")
    print(f"Parallel workers: {workers}")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_seed_worker, payload): payload["seed"] for payload in payloads}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            if result.get("status") == "ok":
                print(f"[seed {result['seed']}] completed")
            else:
                print(f"[seed {result['seed']}] failed")

    results.sort(key=lambda item: int(item.get("seed", 0)))
    rows = [dict(item.get("row", {})) for item in results]
    successful = [row for row in rows if str(row.get("status", "")) == "ok"]
    summary_row = aggregate_numeric_rows(successful)
    summary_row["method/controller"] = args.controller
    summary_row["scenario"] = str(getattr(getattr(cfg, "scenario", None), "name", "") or "")
    summary_row["seed"] = float("nan")
    summary_row["status"] = "ok" if len(successful) == len(rows) else ("partial" if successful else "error")
    rows_with_summary = rows + [summary_row]

    table_text = _render_table(rows_with_summary, show_terminal_table=args.show_terminal_table)
    print("\n" + table_text)
    (output_dir / "terminal_summary.txt").write_text(table_text + "\n", encoding="utf-8")
    _save_json(output_dir / "seed_rows.json", rows)
    _save_json(output_dir / "summary.json", summary_row)
    _save_json(output_dir / "raw_results.json", results)
    _save_csv(output_dir / "seed_rows.csv", rows)
    _save_csv(output_dir / "summary.csv", [summary_row])
    diagnostic_rows = [
        row
        for result in results
        if result.get("status") == "ok"
        for row in result.get("artifacts", {}).get("junction_diagnostics", [])
    ]
    if diagnostic_rows:
        diagnostic_dir = output_dir / "diagnostics"
        diagnostic_label = str(args.diagnostic_junction or "junction")
        _save_json(diagnostic_dir / f"{diagnostic_label}_decisions_all_seeds.json", diagnostic_rows)
        _save_csv(diagnostic_dir / f"{diagnostic_label}_decisions_all_seeds.csv", diagnostic_rows)
    if successful:
        _save_validation_plots(output_dir, cfg, successful, results)
    if any(result.get("status") != "ok" for result in results):
        errors = {f"seed_{item['seed']}": item.get("error", "") for item in results if item.get("status") != "ok"}
        _save_json(output_dir / "errors.json", errors)


if __name__ == "__main__":
    main()
