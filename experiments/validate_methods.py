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
    action_traces: Dict[str, list[int]] = {}
    action_space_sizes: Dict[str, int] = {}
    phase_queue_traces: Dict[str, list[Dict[str, Any]]] = {}
    action_latency_seconds: list[float] = []
    total_reward = 0.0
    agent_ids = [str(agent_id) for agent_id in _possible_agents(eval_env) if not str(agent_id).startswith("__")]
    for agent_id in agent_ids:
        action_traces[agent_id] = []
        phase_queue_traces[agent_id] = []
    done = False
    decision_steps = 0
    while not done:
        decision_steps += 1
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
        action_latency_seconds.append(time.perf_counter() - action_start)
        for agent_id, action in actions.items():
            action_value = int(np.asarray(action).reshape(-1)[0])
            action_traces[agent_id].append(action_value)
            action_space_sizes[agent_id] = max(int(action_space_sizes.get(agent_id, 0)), action_value + 1)
        obs, rewards, terminations, truncations, _ = eval_env.step(actions)
        total_reward += float(sum(float(value) for value in rewards.values()))
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
    return total_reward, action_traces, action_space_sizes, phase_queue_traces, action_latency_seconds, decision_steps


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
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_gpus=0)
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
        episode_reward, action_traces, action_space_sizes, phase_queue_traces, action_latency_seconds, decision_steps = _run_rllib_seed_episode(
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
            "checkpoint_restore_seconds": checkpoint_restore_seconds,
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
    if successful:
        _save_validation_plots(output_dir, cfg, successful, results)
    if any(result.get("status") != "ok" for result in results):
        errors = {f"seed_{item['seed']}": item.get("error", "") for item in results if item.get("status") != "ok"}
        _save_json(output_dir / "errors.json", errors)


if __name__ == "__main__":
    main()
