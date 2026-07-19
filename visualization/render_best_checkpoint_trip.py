from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.render_fgs_topology import Point, extract_node_stage


DEFAULT_RUN_DIR = ROOT / "outputs/resco_ingolstadt1__fgs_mlp_gatv2_ppo/2026-06-21_01-39-09"
DEFAULT_OUTPUT_DIR = ROOT / "visualization/outputs/resco_ingolstadt1__fgs_mlp_gatv2_ppo"
DEFAULT_FIXED_TIME_CONFIG = "presets/resco_ingolstadt21/fixed_time"
DEFAULT_FIXED_TIME_OUTPUT_DIR = ROOT / "visualization/outputs/resco_ingolstadt21__fixed_time"
FONT_DIR = ROOT / "visualization/assets/fonts"

STYLE = {
    "ink": "#141413",
    "paper": "#faf9f5",
    "card": "#e8e6dc",
    "card_alt": "#f1efe7",
    "mid_gray": "#b0aea5",
    "road_wash": "#e8e6dc",
    "road_ink": "#8c8980",
    "orange": "#d97757",
    "blue": "#6a9bcc",
    "green": "#788c5d",
    "tan": "#c9a35b",
}


@dataclass(frozen=True)
class BestCheckpoint:
    rank: int
    checkpoint_path: Path
    metric_name: str
    metric_value: float
    validation_pass_index: int
    validation_env_step: float
    entry: dict[str, Any]


def select_best_checkpoint(run_dir: str | Path, *, best_index: int = 0) -> BestCheckpoint:
    metadata_path = _best_validation_metadata_path(Path(run_dir))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    retained = list(metadata.get("retained") or [])
    if not retained:
        raise ValueError(f"No retained best-validation checkpoints in {metadata_path}")
    retained.sort(key=lambda item: (int(item.get("rank", 10**9)), float(item.get("metric_value", math.inf))))
    if best_index < 0 or best_index >= len(retained):
        raise IndexError(f"best_index {best_index} is outside retained checkpoint range 0..{len(retained) - 1}")

    entry = retained[best_index]
    return BestCheckpoint(
        rank=int(entry.get("rank", best_index + 1)),
        checkpoint_path=Path(str(entry["checkpoint_path"])),
        metric_name=str(entry.get("metric_name") or metadata.get("metric_name") or ""),
        metric_value=float(entry.get("metric_value", math.nan)),
        validation_pass_index=int(entry.get("validation_pass_index", 0) or 0),
        validation_env_step=float(entry.get("validation_env_step", 0.0) or 0.0),
        entry=dict(entry),
    )


def _best_validation_metadata_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("checkpoints/*/best_validation/metadata.json"))
    if not candidates:
        legacy = run_dir / "checkpoints" / "fgs_ppo" / "best_validation" / "metadata.json"
        raise FileNotFoundError(f"No best-validation metadata found under {run_dir / 'checkpoints'} or {legacy}")
    if len(candidates) == 1:
        return candidates[0]
    def best_metric(path: Path) -> tuple[float, str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            retained = payload.get("retained") or []
            metric = min(float(item.get("metric_value", math.inf)) for item in retained) if retained else math.inf
        except Exception:
            metric = math.inf
        return metric, str(path)

    return sorted(candidates, key=best_metric)[0]


def render_trip_animation(
    trace: dict[str, Any],
    output_path: str | Path,
    *,
    width: int = 1200,
    aspect_ratio: float | None = None,
    fps: int = 12,
    frame_count: int = 160,
    max_render_vehicles: int = 1200,
    show_overlay: bool = True,
    show_legend: bool = True,
) -> Path:
    frames = list(trace.get("frames") or [])
    if not frames:
        raise ValueError("Trace does not contain any frames to render.")

    network = dict(trace.get("network") or {})
    road_polylines = _coerce_roads(network.get("road_polylines") or [])
    tls_positions = _coerce_positions(network.get("tls_positions") or {})
    selected_frames = _sample_frames(frames, frame_count)
    points = [point for road in road_polylines for point in road]
    points.extend(tls_positions.values())
    for frame in selected_frames:
        points.extend((float(vehicle["x"]), float(vehicle["y"])) for vehicle in frame.get("vehicles", []))
    project, height = _projector(points, width, aspect_ratio=aspect_ratio)

    max_pressure = max(
        (abs(float(value)) for frame in selected_frames for value in dict(frame.get("pressures") or {}).values()),
        default=1.0,
    )
    max_pressure = max(max_pressure, 1.0)
    max_speed = max(
        (float(vehicle.get("speed", 0.0)) for frame in selected_frames for vehicle in frame.get("vehicles", [])),
        default=1.0,
    )
    max_speed = max(max_speed, 1.0)

    font = _load_font(
        15 if width >= 800 else 13,
        [
            FONT_DIR / "LibertinusSans-Regular.otf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ],
    )
    large_font = _load_font(
        21 if width >= 800 else 17,
        [
            FONT_DIR / "LibertinusSans-Bold.otf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ],
    )
    images = []
    for index, frame in enumerate(selected_frames):
        image = Image.new("RGB", (width, height), STYLE["paper"])
        draw = ImageDraw.Draw(image)
        _draw_paper_background(draw, width, height)
        _draw_roads(draw, road_polylines, project)
        _draw_pressure_bars(
            draw,
            tls_positions,
            dict(frame.get("pressures") or {}),
            project,
            max_pressure,
            font,
            show_labels=show_overlay or show_legend,
        )
        _draw_vehicles(
            draw,
            list(frame.get("vehicles") or []),
            project,
            max_speed=max_speed,
            max_render_vehicles=max_render_vehicles,
        )
        if show_overlay:
            _draw_overlay(
                draw,
                width=width,
                frame_index=index,
                frame_count=len(selected_frames),
                frame=frame,
                trace=trace,
                font=font,
                large_font=large_font,
            )
        if show_legend:
            _draw_legend(draw, width=width, font=font)
        images.append(image)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000 / max(1, fps))))
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        disposal=2,
        loop=0,
        optimize=False,
    )
    return output


def run_best_checkpoint_trip_visualization(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    best_index: int = 0,
    width: int = 1200,
    aspect_ratio: float | None = None,
    fps: int = 12,
    frame_count: int = 160,
    max_render_vehicles: int = 1200,
    show_overlay: bool = True,
    show_legend: bool = True,
) -> dict[str, Path]:
    run_path = Path(run_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_dir = out_dir / "tripinfo"
    trace_path = out_dir / "trip_trace.json"
    gif_path = out_dir / "trip_animation.gif"
    metadata_path = out_dir / "trip_animation_metadata.json"

    checkpoint = select_best_checkpoint(run_path, best_index=best_index)
    trace, metadata = _restore_evaluate_and_trace(
        run_path,
        checkpoint,
        tripinfo_output_prefix=tripinfo_dir / "best_checkpoint_eval",
    )
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    render_trip_animation(
        trace,
        gif_path,
        width=width,
        aspect_ratio=aspect_ratio,
        fps=fps,
        frame_count=frame_count,
        max_render_vehicles=max_render_vehicles,
        show_overlay=show_overlay,
        show_legend=show_legend,
    )
    metadata.update(
        {
            "animation_path": str(gif_path),
            "trace_path": str(trace_path),
            "render": {
                "width": int(width),
                "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else None,
                "fps": int(fps),
                "frame_count": int(min(frame_count, len(trace.get("frames", [])))),
                "max_render_vehicles": int(max_render_vehicles),
                "show_overlay": bool(show_overlay),
                "show_legend": bool(show_legend),
            },
        }
    )
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "animation": gif_path,
        "trace": trace_path,
        "metadata": metadata_path,
        "tripinfo": Path(str(metadata.get("tripinfo_path", ""))),
    }


def run_fixed_time_trip_visualization(
    output_dir: str | Path = DEFAULT_FIXED_TIME_OUTPUT_DIR,
    *,
    config_name: str = DEFAULT_FIXED_TIME_CONFIG,
    config_file: str | Path | None = None,
    seed: int | None = None,
    width: int = 1200,
    aspect_ratio: float | None = None,
    fps: int = 12,
    frame_count: int = 160,
    max_render_vehicles: int = 1200,
    show_overlay: bool = True,
    show_legend: bool = True,
) -> dict[str, Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_dir = out_dir / "tripinfo"
    trace_path = out_dir / "trip_trace.json"
    gif_path = out_dir / "trip_animation.gif"
    metadata_path = out_dir / "trip_animation_metadata.json"

    cfg = _load_fixed_time_config(config_name=config_name, config_file=config_file)
    trace, metadata = _evaluate_fixed_time_and_trace(
        cfg,
        out_dir,
        seed=seed,
        tripinfo_output_prefix=tripinfo_dir / "fixed_time_eval",
    )
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    render_trip_animation(
        trace,
        gif_path,
        width=width,
        aspect_ratio=aspect_ratio,
        fps=fps,
        frame_count=frame_count,
        max_render_vehicles=max_render_vehicles,
        show_overlay=show_overlay,
        show_legend=show_legend,
    )
    metadata.update(
        {
            "animation_path": str(gif_path),
            "trace_path": str(trace_path),
            "render": {
                "width": int(width),
                "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else None,
                "fps": int(fps),
                "frame_count": int(min(frame_count, len(trace.get("frames", [])))),
                "max_render_vehicles": int(max_render_vehicles),
                "show_overlay": bool(show_overlay),
                "show_legend": bool(show_legend),
            },
        }
    )
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "animation": gif_path,
        "trace": trace_path,
        "metadata": metadata_path,
        "tripinfo": Path(str(metadata.get("tripinfo_path", ""))),
    }


def _load_fixed_time_config(*, config_name: str, config_file: str | Path | None):
    from omegaconf import OmegaConf

    if config_file is not None:
        cfg = OmegaConf.load(Path(config_file))
    else:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            cfg = compose(config_name=config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.experiment.fixed_ts = True
    cfg.env.kwargs.fixed_ts = True
    cfg.env.kwargs.use_gui = False
    return cfg


def _first_configured_seed(cfg: Any) -> int:
    from omegaconf import OmegaConf

    eval_seeds = getattr(getattr(cfg, "experiment", None), "eval_seeds", None)
    if OmegaConf.is_config(eval_seeds):
        eval_seeds = OmegaConf.to_container(eval_seeds, resolve=True)
    if isinstance(eval_seeds, list) and eval_seeds:
        return int(eval_seeds[0])
    seeds = getattr(getattr(cfg, "experiment", None), "seeds", None)
    if OmegaConf.is_config(seeds):
        seeds = OmegaConf.to_container(seeds, resolve=True)
    if isinstance(seeds, list) and seeds:
        return int(seeds[0])
    configured = getattr(getattr(cfg, "experiment", None), "seed", None)
    return int(configured if configured is not None else 42)


def _evaluate_fixed_time_and_trace(
    cfg: Any,
    run_dir: Path,
    *,
    seed: int | None,
    tripinfo_output_prefix: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sumo_rl.experiments.runner import _build_env, _get_completed_episode_summary

    chosen_seed = _first_configured_seed(cfg) if seed is None else int(seed)
    cfg.experiment.seed = chosen_seed
    cfg.env.kwargs.tripinfo_output_name = str(tripinfo_output_prefix)
    cfg.env.kwargs.keep_tripinfo_output = True
    cfg.env.kwargs.fixed_ts = True
    cfg.env.kwargs.use_gui = False

    env = None
    total_reward = 0.0
    try:
        env = _build_env(cfg, run_dir, seed=chosen_seed)
        base_env = _resolve_sumo_base_env(env)
        base_env.tripinfo_output_name = str(tripinfo_output_prefix)
        base_env.keep_tripinfo_output = True
        base_env.fixed_ts = True

        env.reset(seed=chosen_seed)
        tripinfo_path = _tripinfo_path(base_env)
        trace_frames = [_capture_frame(base_env)]
        done = False
        while not done:
            step_result = env.step(None)
            if len(step_result) == 5:
                _obs, rewards, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                _obs, rewards, dones, _info = step_result
                done = bool(dones.get("__all__", False) if isinstance(dones, dict) else dones)
            if isinstance(rewards, dict):
                total_reward += float(sum(float(value) for value in rewards.values()))
            elif rewards is not None:
                total_reward += float(rewards)
            trace_frames.append(_capture_frame(base_env))
        env.close()
        env = None
        episode_summary = _get_completed_episode_summary(base_env)
    finally:
        if env is not None:
            env.close()

    net_file = Path(str(cfg.env.kwargs.net_file))
    if not net_file.is_absolute():
        net_file = ROOT / net_file
    extraction = extract_node_stage(net_file)
    tls_positions = _tls_positions_from_extraction(extraction)
    trace = {
        "metadata": {
            "experiment": str(cfg.experiment.name),
            "scenario": str(cfg.scenario.name),
            "seed": chosen_seed,
            "algorithm_kind": "fixed_time",
            "total_reward": total_reward,
            "metric_name": "resco_delay_mean",
            "metric_value": float(episode_summary.get("resco_delay_mean", math.nan)),
        },
        "network": {
            "net_file": str(net_file),
            "route_file": str(cfg.env.kwargs.route_file),
            "road_polylines": [[list(point) for point in road] for road in extraction.road_polylines],
            "tls_positions": tls_positions,
        },
        "frames": trace_frames,
    }
    metadata = {
        "run_dir": str(run_dir),
        "algorithm_kind": "fixed_time",
        "seed": chosen_seed,
        "net_file": str(net_file),
        "route_file": str(cfg.env.kwargs.route_file),
        "tripinfo_path": str(tripinfo_path) if tripinfo_path is not None else None,
        "trace_frames": len(trace_frames),
        "max_live_vehicles": max((len(frame["vehicles"]) for frame in trace_frames), default=0),
        "tls_count": len(tls_positions),
        "episode_summary": episode_summary,
    }
    return trace, metadata


def _restore_evaluate_and_trace(
    run_dir: Path,
    checkpoint: BestCheckpoint,
    *,
    tripinfo_output_prefix: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")

    from omegaconf import OmegaConf

    from sumo_rl.experiments.rllib_runner import (
        _build_algorithm_config,
        _build_eval_env,
        _compute_single_action,
        _eval_seeds,
        _policy_id_for_agent,
        _policy_mode,
        _plain_dict,
        _possible_agents,
        _restore_checkpoint,
        _sync_env_runner_weights_for_evaluation,
    )
    from sumo_rl.experiments.runner import _get_completed_episode_summary

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.env.kwargs.tripinfo_output_name = str(tripinfo_output_prefix)
    cfg.env.kwargs.keep_tripinfo_output = True
    cfg.env.kwargs.use_gui = False
    cfg.algorithm.params.num_env_runners = 0
    cfg.algorithm.params.num_rollout_workers = 0
    cfg.algorithm.params.num_gpus_per_env_runner = 0
    cfg.algorithm.params.num_gpus_per_learner = 0
    cfg.algorithm.params.ray_num_gpus = 0
    if "model_config" in cfg.algorithm.params and "topology" in cfg.algorithm.params.model_config:
        cfg.algorithm.params.model_config.topology.render = False

    algorithm_kind = str(cfg.algorithm.kind)
    policy_mode = _policy_mode(_plain_dict(getattr(cfg.algorithm, "params", {}) or {}))
    seed = int(_eval_seeds(cfg)[0])

    import ray

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_gpus=0)
    algo = None
    eval_env = None
    try:
        algo_config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(algo_config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else algo_config.build()
        _restore_checkpoint(algo, checkpoint.checkpoint_path)
        _sync_env_runner_weights_for_evaluation(algo)

        eval_env = _build_eval_env(cfg, run_dir, seed, algorithm_kind=algorithm_kind, policy_mode=policy_mode)
        base_env = _resolve_sumo_base_env(eval_env)
        base_env.tripinfo_output_name = str(tripinfo_output_prefix)
        base_env.keep_tripinfo_output = True

        obs, _ = eval_env.reset(seed=seed)
        tripinfo_path = _tripinfo_path(base_env)
        trace_frames = [_capture_frame(base_env)]
        total_reward = 0.0
        possible_agents = _possible_agents(eval_env)
        done = False
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                if str(agent_id).startswith("__"):
                    continue
                actions[agent_id] = _compute_single_action(
                    algo,
                    agent_obs,
                    policy_id=_policy_id_for_agent(str(agent_id), policy_mode),
                )
            obs, rewards, terminations, truncations, _ = eval_env.step(actions)
            total_reward += float(sum(float(value) for value in rewards.values()))
            trace_frames.append(_capture_frame(base_env))
            done = bool(
                terminations.get("__all__", False)
                or truncations.get("__all__", False)
                or all(bool(terminations.get(agent_id, False)) for agent_id in possible_agents)
                or all(bool(truncations.get(agent_id, False)) for agent_id in possible_agents)
            )

        try:
            eval_env.close()
        finally:
            eval_env = None
        episode_summary = _get_completed_episode_summary(base_env)
    finally:
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()

    net_file = Path(str(cfg.env.kwargs.net_file))
    if not net_file.is_absolute():
        net_file = ROOT / net_file
    extraction = extract_node_stage(net_file)
    tls_positions = _tls_positions_from_extraction(extraction)
    trace = {
        "metadata": {
            "experiment": str(cfg.experiment.name),
            "scenario": str(cfg.scenario.name),
            "seed": seed,
            "algorithm_kind": algorithm_kind,
            "total_reward": total_reward,
            "checkpoint_rank": checkpoint.rank,
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "metric_name": checkpoint.metric_name,
            "metric_value": checkpoint.metric_value,
        },
        "network": {
            "net_file": str(net_file),
            "route_file": str(cfg.env.kwargs.route_file),
            "road_polylines": [[list(point) for point in road] for road in extraction.road_polylines],
            "tls_positions": tls_positions,
        },
        "frames": trace_frames,
    }
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint_rank": checkpoint.rank,
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "metric_name": checkpoint.metric_name,
        "metric_value": checkpoint.metric_value,
        "validation_pass_index": checkpoint.validation_pass_index,
        "validation_env_step": checkpoint.validation_env_step,
        "seed": seed,
        "net_file": str(net_file),
        "route_file": str(cfg.env.kwargs.route_file),
        "tripinfo_path": str(tripinfo_path) if tripinfo_path is not None else None,
        "trace_frames": len(trace_frames),
        "max_live_vehicles": max((len(frame["vehicles"]) for frame in trace_frames), default=0),
        "tls_count": len(tls_positions),
        "episode_summary": episode_summary,
    }
    return trace, metadata


def _resolve_sumo_base_env(env: Any) -> Any:
    current = env
    visited = set()
    for _ in range(20):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if hasattr(current, "traffic_signals") and hasattr(current, "ts_ids") and hasattr(current, "sumo"):
            return current
        for attr in ("par_env", "aec_env", "env", "base_env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    raise RuntimeError("Unable to resolve underlying SumoEnvironment for live trace capture.")


def _tripinfo_path(base_env: Any) -> Optional[Path]:
    build_path = getattr(base_env, "_build_tripinfo_output_path", None)
    if callable(build_path):
        path = build_path()
        return Path(path) if path is not None else None
    output_name = getattr(base_env, "tripinfo_output_name", None)
    label = getattr(base_env, "label", None)
    episode = getattr(base_env, "episode", None)
    if output_name and label is not None and episode is not None:
        return Path(f"{output_name}_conn{label}_ep{episode}.xml")
    return None


def _capture_frame(base_env: Any) -> dict[str, Any]:
    sumo = base_env.sumo
    vehicle_ids = sorted(str(vehicle_id) for vehicle_id in sumo.vehicle.getIDList())
    vehicles = []
    for vehicle_id in vehicle_ids:
        try:
            x, y = sumo.vehicle.getPosition(vehicle_id)
            vehicles.append(
                {
                    "id": vehicle_id,
                    "x": float(x),
                    "y": float(y),
                    "speed": float(sumo.vehicle.getSpeed(vehicle_id)),
                    "edge": str(sumo.vehicle.getRoadID(vehicle_id)),
                    "lane": str(sumo.vehicle.getLaneID(vehicle_id)),
                }
            )
        except Exception:
            continue
    pressures = {}
    for tls_id in getattr(base_env, "ts_ids", []):
        signal = base_env.traffic_signals.get(tls_id)
        if signal is None:
            continue
        try:
            pressures[str(tls_id)] = float(signal.get_pressure())
        except Exception:
            pressures[str(tls_id)] = 0.0
    return {
        "time": float(base_env.sim_step),
        "vehicles": vehicles,
        "pressures": pressures,
    }


def _tls_positions_from_extraction(extraction: Any) -> dict[str, list[float]]:
    positions = {}
    for junction in extraction.junctions:
        for tls_id in junction.tls_program_ids:
            positions[str(tls_id)] = [float(junction.position[0]), float(junction.position[1])]
    return dict(sorted(positions.items()))


def _sample_frames(frames: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    if frame_count <= 0 or len(frames) <= frame_count:
        return frames
    if frame_count == 1:
        return [frames[-1]]
    indexes = [round(index * (len(frames) - 1) / (frame_count - 1)) for index in range(frame_count)]
    return [frames[int(index)] for index in indexes]


def _coerce_roads(raw_roads: list[Any]) -> list[list[Point]]:
    roads = []
    for raw_road in raw_roads:
        road = []
        for raw_point in raw_road:
            if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
                road.append((float(raw_point[0]), float(raw_point[1])))
        if len(road) >= 2:
            roads.append(road)
    return roads


def _coerce_positions(raw_positions: dict[str, Any]) -> dict[str, Point]:
    positions = {}
    for key, raw_point in raw_positions.items():
        if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
            positions[str(key)] = (float(raw_point[0]), float(raw_point[1]))
    return positions


def _projector(points: list[Point], width: int, *, aspect_ratio: float | None = None) -> tuple[Callable[[Point], Point], int]:
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    if aspect_ratio is not None and aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive when provided.")
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    if aspect_ratio is None:
        height = max(520, min(1600, int(width * (max_y - min_y) / max(max_x - min_x, 1e-6))))
    else:
        height = max(1, int(round(width / aspect_ratio)))
    padding = 64
    available_width = max(1.0, width - 2 * padding)
    available_height = max(1.0, height - 2 * padding)
    scale = min(available_width / (max_x - min_x), available_height / (max_y - min_y))
    scaled_width = (max_x - min_x) * scale
    scaled_height = (max_y - min_y) * scale
    offset_x = (width - scaled_width) / 2.0
    offset_y = (height - scaled_height) / 2.0

    def project(point: Point) -> Point:
        x = offset_x + (point[0] - min_x) * scale
        y = height - (offset_y + (point[1] - min_y) * scale)
        return x, y

    return project, height


def _load_font(size: int, candidates: list[Path]) -> ImageFont.ImageFont:
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_paper_background(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    draw.rectangle((0, 0, width - 1, height - 1), outline=STYLE["card"])
    for index in range(max(width, height) // 18):
        x = (index * 89 + 37) % max(width, 1)
        y = (index * 53 + 19) % max(height, 1)
        if index % 3 == 0:
            draw.line((x, y, min(width - 1, x + 9), y), fill=STYLE["card_alt"], width=1)
        elif index % 3 == 1:
            draw.point((x, y), fill=STYLE["card_alt"])


def _draw_roads(draw: ImageDraw.ImageDraw, roads: list[list[Point]], project: Callable[[Point], Point]) -> None:
    for road_index, road in enumerate(roads):
        points = [_int_point(project(point)) for point in road]
        if len(points) >= 2:
            draw.line(_sketch_points(points, amount=1.15, salt=road_index), fill=STYLE["road_wash"], width=4)
            draw.line(_sketch_points(points, amount=0.65, salt=road_index + 97), fill=STYLE["road_ink"], width=1)


def _draw_pressure_bars(
    draw: ImageDraw.ImageDraw,
    tls_positions: dict[str, Point],
    pressures: dict[str, Any],
    project: Callable[[Point], Point],
    max_pressure: float,
    font: ImageFont.ImageFont,
    *,
    show_labels: bool,
) -> None:
    for tls_id, point in sorted(tls_positions.items()):
        x, y = _int_point(project(point))
        pressure = float(pressures.get(tls_id, 0.0) or 0.0)
        magnitude = min(1.0, abs(pressure) / max(max_pressure, 1e-6))
        bar_height = int(round(12 + 38 * magnitude))
        color = STYLE["orange"] if pressure < 0 else STYLE["blue"]
        _draw_organic_dot(draw, x, y, 7, STYLE["green"], outline=STYLE["paper"], width=2, salt=len(tls_id))
        _draw_organic_dot(draw, x, y, 3, STYLE["paper"], outline=None, width=0, salt=len(tls_id) + 13)
        draw.rounded_rectangle((x + 9, y - bar_height, x + 17, y), radius=3, fill=STYLE["card_alt"], outline=STYLE["mid_gray"])
        fill_top = y - int(round(bar_height * magnitude))
        if fill_top < y:
            draw.rounded_rectangle((x + 10, fill_top, x + 16, y - 1), radius=2, fill=color)
        if show_labels and magnitude > 0.72:
            draw.text((x + 21, y - bar_height - 3), tls_id, fill=STYLE["ink"], font=font, stroke_width=2, stroke_fill=STYLE["paper"])


def _draw_vehicles(
    draw: ImageDraw.ImageDraw,
    vehicles: list[dict[str, Any]],
    project: Callable[[Point], Point],
    *,
    max_speed: float,
    max_render_vehicles: int,
) -> None:
    for vehicle in sorted(vehicles, key=lambda item: str(item.get("id", "")))[:max_render_vehicles]:
        x, y = _int_point(project((float(vehicle["x"]), float(vehicle["y"]))))
        speed = float(vehicle.get("speed", 0.0) or 0.0)
        if speed < 0.1:
            fill = STYLE["orange"]
            radius = 6
        elif speed < 5.0:
            fill = STYLE["tan"]
            radius = 5
        else:
            fill = STYLE["blue"]
            radius = 4
        _draw_organic_dot(draw, x, y, radius, fill, outline=STYLE["paper"], width=1, salt=len(str(vehicle.get("id", ""))))


def _draw_overlay(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    frame_index: int,
    frame_count: int,
    frame: dict[str, Any],
    trace: dict[str, Any],
    font: ImageFont.ImageFont,
    large_font: ImageFont.ImageFont,
) -> None:
    metadata = dict(trace.get("metadata") or {})
    title = str(metadata.get("experiment", "best-checkpoint trip"))
    metric = metadata.get("metric_value")
    metric_text = f"delay {float(metric):.3f}" if isinstance(metric, (int, float)) else "best checkpoint"
    checkpoint_rank = metadata.get("checkpoint_rank")
    if checkpoint_rank is not None:
        run_text = f"rank {checkpoint_rank} checkpoint | {metric_text}"
    else:
        run_text = f"{metadata.get('algorithm_kind', 'control')} | {metric_text}"
    time_value = float(frame.get("time", 0.0) or 0.0)
    frame_vehicles = list(frame.get("vehicles") or [])
    vehicles = len(frame_vehicles)
    stopped = sum(1 for vehicle in frame_vehicles if float(vehicle.get("speed", 0.0) or 0.0) < 0.1)
    slow = sum(1 for vehicle in frame_vehicles if 0.1 <= float(vehicle.get("speed", 0.0) or 0.0) < 5.0)
    moving = max(0, vehicles - slow - stopped)
    progress = 0.0 if frame_count <= 1 else frame_index / (frame_count - 1)
    box_right = min(width - 18, 590)
    box_bottom = 148
    draw.rounded_rectangle((18, 18, box_right, box_bottom), radius=8, fill=STYLE["card"], outline=STYLE["ink"])
    _draw_organic_dot(draw, 39, 43, 8, STYLE["orange"], outline=STYLE["paper"], width=1, salt=3)
    title_max_width = max(80, box_right - 66)
    draw.text((54, 31), _fit_text(draw, title, large_font, title_max_width), fill=STYLE["ink"], font=large_font)
    draw.text((34, 64), _fit_text(draw, run_text, font, box_right - 68), fill=STYLE["ink"], font=font)
    stats = f"t = {time_value:.0f}s | flow: {moving} | slow: {slow} | queue: {stopped}"
    draw.text((34, 87), _fit_text(draw, stats, font, box_right - 68), fill=STYLE["ink"], font=font)
    track_left = 34
    track_right = box_right - 34
    track_top = 121
    draw.rounded_rectangle((track_left, track_top, track_right, track_top + 10), radius=5, fill=STYLE["card_alt"], outline=STYLE["mid_gray"])
    if progress > 0.0:
        fill_right = track_left + int((track_right - track_left) * progress)
        draw.rounded_rectangle((track_left, track_top, fill_right, track_top + 10), radius=5, fill=STYLE["blue"])


def _draw_legend(draw: ImageDraw.ImageDraw, *, width: int, font: ImageFont.ImageFont) -> None:
    legend_width = 292
    x = max(18, width - legend_width - 18)
    y = 18 if width >= 900 else 158
    row_y = y + 42
    row_gap = 25
    rows = [
        ("line", STYLE["road_ink"], "sketched road network"),
        ("dot", STYLE["blue"], "moving flow >= 5 m/s"),
        ("dot", STYLE["tan"], "slow approach 0.1-5 m/s"),
        ("dot", STYLE["orange"], "queued/stopped < 0.1 m/s"),
        ("dot", STYLE["green"], "traffic signal"),
        ("bar", STYLE["blue"], "positive pressure"),
        ("bar", STYLE["orange"], "negative pressure"),
    ]
    card_height = 58 + row_gap * (len(rows) - 1) + 22
    draw.rounded_rectangle((x, y, x + legend_width, y + card_height), radius=8, fill=STYLE["paper"], outline=STYLE["ink"])
    draw.rectangle((x + 14, y + 14, x + 31, y + 31), fill=STYLE["green"])
    draw.text((x + 42, y + 13), "Legend", fill=STYLE["ink"], font=font)
    for index, (kind, color, label) in enumerate(rows):
        item_y = row_y + row_gap * index
        if kind == "line":
            _legend_line(draw, x + 16, item_y, color, label, font, line=True)
        elif kind == "dot":
            _legend_dot(draw, x + 16, item_y, color, label, font)
        else:
            _legend_bar(draw, x + 16, item_y, color, label, font)


def _legend_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: str,
    label: str,
    font: ImageFont.ImageFont,
    *,
    line: bool = False,
) -> None:
    if line:
        draw.line(_sketch_points([(x, y + 7), (x + 22, y + 7)], amount=0.65, salt=y), fill=color, width=1)
    draw.text((x + 32, y), label, fill=STYLE["ink"], font=font)


def _legend_dot(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str, font: ImageFont.ImageFont) -> None:
    _draw_organic_dot(draw, x + 10, y + 7, 6, color, outline=STYLE["paper"], width=1, salt=y)
    draw.text((x + 32, y), label, fill=STYLE["ink"], font=font)


def _legend_bar(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle((x + 7, y, x + 15, y + 17), radius=3, fill=STYLE["card_alt"], outline=STYLE["ink"])
    draw.rounded_rectangle((x + 9, y + 5, x + 13, y + 16), radius=2, fill=color)
    draw.text((x + 32, y), label, fill=STYLE["ink"], font=font)


def _sketch_points(points: list[tuple[int, int]], *, amount: float, salt: int) -> list[tuple[int, int]]:
    sketched = []
    for index, (x, y) in enumerate(points):
        dx = _jitter(x, y, index + salt, amount)
        dy = _jitter(y, x, index + salt + 1009, amount)
        sketched.append((int(round(x + dx)), int(round(y + dy))))
    return sketched


def _draw_organic_dot(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    fill: str,
    *,
    outline: str | None,
    width: int,
    salt: int,
) -> None:
    sides = 12
    points = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        wobble = 1.0 + 0.12 * _jitter(x + radius, y - radius, index + salt, 1.0)
        px = x + math.cos(angle) * radius * wobble
        py = y + math.sin(angle) * radius * wobble
        points.append((int(round(px)), int(round(py))))
    draw.polygon(points, fill=fill)
    if outline is not None and width > 0:
        draw.line(points + [points[0]], fill=outline, width=width)


def _jitter(x: float, y: float, salt: int, amount: float) -> float:
    value = math.sin(x * 12.9898 + y * 78.233 + salt * 37.719) * 43758.5453
    return (value - math.floor(value) - 0.5) * 2.0 * amount


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    available = max(0, max_width - _text_width(draw, suffix, font))
    clipped = ""
    for char in text:
        if _text_width(draw, clipped + char, font) > available:
            break
        clipped += char
    return clipped.rstrip() + suffix


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    try:
        left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left)
    except AttributeError:
        return int(draw.textsize(text, font=font)[0])


def _int_point(point: Point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a best-checkpoint SUMO trip animation with pressure bars.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Hydra run directory to restore.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for GIF and JSON.")
    parser.add_argument("--best-index", type=int, default=0, help="Retained best-validation checkpoint index, sorted by rank.")
    parser.add_argument("--width", type=int, default=1200, help="Animation width in pixels.")
    parser.add_argument("--aspect-ratio", type=float, default=None, help="Optional output width/height ratio, e.g. 1.65.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--frame-count", type=int, default=160, help="Maximum rendered GIF frame count.")
    parser.add_argument("--max-render-vehicles", type=int, default=1200, help="Maximum vehicle dots drawn per frame.")
    parser.add_argument("--hide-overlay", action="store_true", help="Do not draw the top-left run/status overlay.")
    parser.add_argument("--hide-legend", action="store_true", help="Do not draw the legend card.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_best_checkpoint_trip_visualization(
        args.run_dir,
        args.output_dir,
        best_index=args.best_index,
        width=args.width,
        aspect_ratio=args.aspect_ratio,
        fps=args.fps,
        frame_count=args.frame_count,
        max_render_vehicles=args.max_render_vehicles,
        show_overlay=not args.hide_overlay,
        show_legend=not args.hide_legend,
    )
    print(f"Wrote trip animation GIF: {paths['animation']}")
    print(f"Wrote live trace JSON: {paths['trace']}")
    print(f"Wrote metadata JSON: {paths['metadata']}")
    print(f"Retained tripinfo XML: {paths['tripinfo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
