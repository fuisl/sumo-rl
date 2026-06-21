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
    fps: int = 12,
    frame_count: int = 160,
    max_render_vehicles: int = 1200,
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
    project, height = _projector(points, width)

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

    font = ImageFont.load_default()
    large_font = ImageFont.load_default()
    images = []
    for index, frame in enumerate(selected_frames):
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)
        _draw_roads(draw, road_polylines, project)
        _draw_pressure_bars(draw, tls_positions, dict(frame.get("pressures") or {}), project, max_pressure, font)
        _draw_vehicles(
            draw,
            list(frame.get("vehicles") or []),
            project,
            max_speed=max_speed,
            max_render_vehicles=max_render_vehicles,
        )
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
    fps: int = 12,
    frame_count: int = 160,
    max_render_vehicles: int = 1200,
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
        fps=fps,
        frame_count=frame_count,
        max_render_vehicles=max_render_vehicles,
    )
    metadata.update(
        {
            "animation_path": str(gif_path),
            "trace_path": str(trace_path),
            "render": {
                "width": int(width),
                "fps": int(fps),
                "frame_count": int(min(frame_count, len(trace.get("frames", [])))),
                "max_render_vehicles": int(max_render_vehicles),
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


def _projector(points: list[Point], width: int) -> tuple[Callable[[Point], Point], int]:
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    height = max(520, min(1600, int(width * (max_y - min_y) / max(max_x - min_x, 1e-6))))
    padding = 64
    scale = min((width - 2 * padding) / (max_x - min_x), (height - 2 * padding) / (max_y - min_y))

    def project(point: Point) -> Point:
        x = padding + (point[0] - min_x) * scale
        y = height - (padding + (point[1] - min_y) * scale)
        return x, y

    return project, height


def _draw_roads(draw: ImageDraw.ImageDraw, roads: list[list[Point]], project: Callable[[Point], Point]) -> None:
    for road in roads:
        points = [_int_point(project(point)) for point in road]
        if len(points) >= 2:
            draw.line(points, fill="#64748b", width=1)


def _draw_pressure_bars(
    draw: ImageDraw.ImageDraw,
    tls_positions: dict[str, Point],
    pressures: dict[str, Any],
    project: Callable[[Point], Point],
    max_pressure: float,
    font: ImageFont.ImageFont,
) -> None:
    for tls_id, point in sorted(tls_positions.items()):
        x, y = _int_point(project(point))
        pressure = float(pressures.get(tls_id, 0.0) or 0.0)
        magnitude = min(1.0, abs(pressure) / max(max_pressure, 1e-6))
        bar_height = int(round(10 + 34 * magnitude))
        color = "#dc2626" if pressure < 0 else "#0891b2"
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#0f766e", outline="#ffffff", width=2)
        draw.rounded_rectangle((x + 7, y - bar_height, x + 13, y), radius=2, fill="#e2e8f0")
        draw.rounded_rectangle((x + 7, y - bar_height, x + 13, y), radius=2, outline="#ffffff")
        fill_top = y - int(round(bar_height * magnitude))
        draw.rounded_rectangle((x + 7, fill_top, x + 13, y), radius=2, fill=color)
        if magnitude > 0.72:
            draw.text((x + 16, y - bar_height - 2), tls_id, fill="#0f172a", font=font, stroke_width=2, stroke_fill="#ffffff")


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
            fill = "#ef4444"
            radius = 6
        elif speed < 5.0:
            fill = "#f59e0b"
            radius = 5
        else:
            fill = "#2563eb"
            radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#ffffff")


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
    time_value = float(frame.get("time", 0.0) or 0.0)
    frame_vehicles = list(frame.get("vehicles") or [])
    vehicles = len(frame_vehicles)
    stopped = sum(1 for vehicle in frame_vehicles if float(vehicle.get("speed", 0.0) or 0.0) < 0.1)
    progress = 0.0 if frame_count <= 1 else frame_index / (frame_count - 1)
    box_right = min(width - 18, 520)
    draw.rounded_rectangle((18, 18, box_right, 128), radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((34, 34), title, fill="#0f172a", font=large_font)
    draw.text((34, 56), f"rank {metadata.get('checkpoint_rank', 1)} checkpoint | {metric_text}", fill="#334155", font=font)
    draw.text((34, 76), f"t = {time_value:.0f}s | live: {vehicles} | stopped: {stopped}", fill="#334155", font=font)
    draw.rounded_rectangle((34, 102, box_right - 34, 110), radius=4, fill="#e2e8f0")
    draw.rounded_rectangle((34, 102, 34 + int((box_right - 68) * progress), 110), radius=4, fill="#2563eb")


def _draw_legend(draw: ImageDraw.ImageDraw, *, width: int, font: ImageFont.ImageFont) -> None:
    legend_width = 256
    x = max(18, width - legend_width - 18)
    y = 18 if width >= 820 else 142
    row_y = y + 36
    draw.rounded_rectangle((x, y, x + legend_width, y + 172), radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((x + 16, y + 14), "Legend", fill="#0f172a", font=font)
    _legend_line(draw, x + 16, row_y, "#64748b", "1px road network", font, line=True)
    _legend_dot(draw, x + 16, row_y + 24, "#2563eb", "moving vehicle >= 5 m/s", font)
    _legend_dot(draw, x + 16, row_y + 48, "#f59e0b", "slow vehicle 0.1-5 m/s", font)
    _legend_dot(draw, x + 16, row_y + 72, "#ef4444", "stopped vehicle < 0.1 m/s", font)
    _legend_dot(draw, x + 16, row_y + 96, "#0f766e", "traffic signal", font)
    _legend_bar(draw, x + 16, row_y + 120, "#0891b2", "positive pressure", font)
    _legend_bar(draw, x + 16, row_y + 144, "#dc2626", "negative pressure", font)


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
        draw.line((x, y + 6, x + 20, y + 6), fill=color, width=1)
    draw.text((x + 30, y), label, fill="#0f172a", font=font)


def _legend_dot(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str, font: ImageFont.ImageFont) -> None:
    draw.ellipse((x + 4, y + 1, x + 16, y + 13), fill=color, outline="#ffffff")
    draw.text((x + 30, y), label, fill="#0f172a", font=font)


def _legend_bar(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle((x + 7, y, x + 13, y + 14), radius=2, fill=color)
    draw.text((x + 30, y), label, fill="#0f172a", font=font)


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
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--frame-count", type=int, default=160, help="Maximum rendered GIF frame count.")
    parser.add_argument("--max-render-vehicles", type=int, default=1200, help="Maximum vehicle dots drawn per frame.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_best_checkpoint_trip_visualization(
        args.run_dir,
        args.output_dir,
        best_index=args.best_index,
        width=args.width,
        fps=args.fps,
        frame_count=args.frame_count,
        max_render_vehicles=args.max_render_vehicles,
    )
    print(f"Wrote trip animation GIF: {paths['animation']}")
    print(f"Wrote live trace JSON: {paths['trace']}")
    print(f"Wrote metadata JSON: {paths['metadata']}")
    print(f"Retained tripinfo XML: {paths['tripinfo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
