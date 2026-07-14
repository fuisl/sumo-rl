from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import math

import numpy as np

from sumo_rl.util.statistics_output import StatisticOutputParseResult, parse_statistic_output


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _ratio(numerator: Any, denominator: Any) -> float:
    num = _finite_float(numerator)
    den = _finite_float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return num / den


def jain_index(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    numerator = float(np.sum(array)) ** 2
    denominator = float(array.size) * float(np.sum(array**2))
    if denominator <= 0.0:
        return 1.0
    return numerator / denominator


def gini_coefficient(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if np.allclose(array, 0.0):
        return 0.0
    sorted_values = np.sort(np.clip(array, a_min=0.0, a_max=None))
    n = sorted_values.size
    cumulative = np.cumsum(sorted_values, dtype=float)
    denominator = float(cumulative[-1])
    if denominator <= 0.0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cumulative) / denominator) / n)


def load_statistic_output(path: str | Path | None) -> Dict[str, float]:
    stats = parse_statistic_output(path)
    return {
        "sumo_clock_duration": stats.clock_duration,
        "sumo_traci_duration": stats.traci_duration,
        "sumo_real_time_factor": stats.real_time_factor,
        "sumo_vehicle_updates_per_second": stats.vehicle_updates_per_second,
        "sumo_person_updates_per_second": stats.person_updates_per_second,
        "sumo_clock_begin": stats.clock_begin,
        "sumo_clock_end": stats.clock_end,
        "sumo_begin": stats.begin,
        "sumo_end": stats.end,
        "sumo_duration": stats.duration,
        "loaded_vehicle_count": stats.vehicles_loaded,
        "inserted_vehicle_count": stats.vehicles_inserted,
        "running_vehicle_count": stats.vehicles_running,
        "undeparted_vehicle_count": stats.vehicles_waiting,
        "teleports_total": stats.teleports_total,
        "teleports_jam": stats.teleports_jam,
        "teleports_yield": stats.teleports_yield,
        "teleports_wrong_lane": stats.teleports_wrong_lane,
        "collisions_total": stats.collisions,
        "emergency_stops_total": stats.emergency_stops,
        "avg_speed_completed": stats.trip_speed,
        "completed_trip_count_statistics": stats.trip_count,
        "avg_route_length_completed": stats.trip_route_length,
        "avg_trip_time_statistics": stats.trip_duration,
        "avg_waiting_time_statistics": stats.trip_waiting_time,
        "avg_time_loss_statistics": stats.trip_time_loss,
        "avg_depart_delay_statistics": stats.trip_depart_delay,
        "avg_depart_delay_waiting_statistics": stats.trip_depart_delay_waiting,
        "total_travel_time_statistics": stats.trip_total_travel_time,
        "total_depart_delay_statistics": stats.trip_total_depart_delay,
    }


def build_completed_trip_metrics(seed_row: Dict[str, Any], *, disable_completed_view: bool = False) -> Dict[str, float]:
    if disable_completed_view:
        return {}
    return {
        "completed_trip_count": _finite_float(seed_row.get("tripinfo/finished_count")),
        "avg_delay_completed": _finite_float(seed_row.get("final/resco/avg_delay")),
        "std_delay_completed": _finite_float(seed_row.get("final/resco/avg_delay_std", seed_row.get("validation/resco_delay_std"))),
        "max_delay_completed": _finite_float(seed_row.get("validation/resco_delay_max")),
        "avg_trip_time_completed": _finite_float(seed_row.get("final/resco/trip_time")),
        "std_trip_time_completed": _finite_float(seed_row.get("tripinfo/std_duration")),
        "avg_waiting_time_completed": _finite_float(seed_row.get("final/resco/wait")),
        "std_waiting_time_completed": _finite_float(seed_row.get("final/resco/wait_std", seed_row.get("validation/resco_wait_std"))),
        "max_waiting_time_completed": _finite_float(seed_row.get("validation/resco_wait_max")),
        "avg_time_loss_completed": _finite_float(seed_row.get("tripinfo/avg_time_loss")),
        "std_time_loss_completed": _finite_float(seed_row.get("tripinfo/std_time_loss")),
    }


def build_all_demand_metrics(seed_row: Dict[str, Any], stats: StatisticOutputParseResult | Dict[str, float]) -> Dict[str, float]:
    if isinstance(stats, dict):
        loaded = _finite_float(stats.get("loaded_vehicle_count"))
        inserted = _finite_float(stats.get("inserted_vehicle_count"))
        running = _finite_float(stats.get("running_vehicle_count"))
        undeparted = _finite_float(stats.get("undeparted_vehicle_count"))
        avg_duration = _finite_float(stats.get("avg_trip_time_statistics"))
        avg_time_loss = _finite_float(stats.get("avg_time_loss_statistics"))
        avg_depart_delay = _finite_float(stats.get("avg_depart_delay_statistics"))
        avg_depart_delay_waiting = _finite_float(stats.get("avg_depart_delay_waiting_statistics"))
        total_travel_time = _finite_float(stats.get("total_travel_time_statistics"))
        total_depart_delay = _finite_float(stats.get("total_depart_delay_statistics"))
    else:
        loaded = _finite_float(stats.vehicles_loaded)
        inserted = _finite_float(stats.vehicles_inserted)
        running = _finite_float(stats.vehicles_running)
        undeparted = _finite_float(stats.vehicles_waiting)
        avg_duration = _finite_float(stats.trip_duration)
        avg_time_loss = _finite_float(stats.trip_time_loss)
        avg_depart_delay = _finite_float(stats.trip_depart_delay)
        avg_depart_delay_waiting = _finite_float(stats.trip_depart_delay_waiting)
        total_travel_time = _finite_float(stats.trip_total_travel_time)
        total_depart_delay = _finite_float(stats.trip_total_depart_delay)

    arrived = _finite_float(seed_row.get("final/efficiency/total_arrived"))
    if not np.isfinite(arrived):
        arrived = _finite_float(seed_row.get("validation/efficiency_total_arrived"))
    if not np.isfinite(inserted):
        inserted = _finite_float(seed_row.get("final/efficiency/total_departed"))
    if not np.isfinite(running):
        running = _finite_float(seed_row.get("final/efficiency/total_running"))
    if not np.isfinite(undeparted):
        undeparted = _finite_float(seed_row.get("tripinfo/undeparted_count"))

    if not np.isfinite(total_travel_time):
        total_travel_time = inserted * avg_duration if np.isfinite(inserted) and np.isfinite(avg_duration) else float("nan")
    if not np.isfinite(total_depart_delay):
        total_depart_delay = inserted * avg_depart_delay if np.isfinite(inserted) and np.isfinite(avg_depart_delay) else float("nan")

    waiting_delay_total = undeparted * avg_depart_delay_waiting if np.isfinite(undeparted) and np.isfinite(avg_depart_delay_waiting) else float("nan")
    observed_total_travel_time_and_delay = (
        total_travel_time + total_depart_delay + waiting_delay_total
        if np.isfinite(total_travel_time) and np.isfinite(total_depart_delay) and np.isfinite(waiting_delay_total)
        else float("nan")
    )
    observed_total_delay_all = (
        (inserted * (avg_time_loss + avg_depart_delay)) + waiting_delay_total
        if np.isfinite(inserted) and np.isfinite(avg_time_loss) and np.isfinite(avg_depart_delay) and np.isfinite(waiting_delay_total)
        else float("nan")
    )
    total_depart_delay_all = (
        total_depart_delay + waiting_delay_total
        if np.isfinite(total_depart_delay) and np.isfinite(waiting_delay_total)
        else float("nan")
    )

    preferred_denominator = loaded if np.isfinite(loaded) and loaded > 0 else inserted
    return {
        "loaded_vehicle_count": loaded,
        "inserted_vehicle_count": inserted,
        "arrived_vehicle_count": arrived,
        "running_vehicle_count": running,
        "undeparted_vehicle_count": undeparted,
        "throughput": arrived,
        "completion_ratio": _ratio(arrived, preferred_denominator),
        "completion_ratio_per_loaded": _ratio(arrived, loaded),
        "completion_ratio_per_inserted": _ratio(arrived, inserted),
        "observed_total_travel_time_and_delay": observed_total_travel_time_and_delay,
        "avg_observed_system_time_all": _ratio(observed_total_travel_time_and_delay, preferred_denominator),
        "avg_observed_system_time_all_per_loaded": _ratio(observed_total_travel_time_and_delay, loaded),
        "avg_observed_system_time_all_per_inserted": _ratio(observed_total_travel_time_and_delay, inserted),
        "observed_total_delay_all": observed_total_delay_all,
        "avg_observed_delay_all": _ratio(observed_total_delay_all, preferred_denominator),
        "avg_observed_delay_all_per_loaded": _ratio(observed_total_delay_all, loaded),
        "avg_observed_delay_all_per_inserted": _ratio(observed_total_delay_all, inserted),
        "avg_depart_delay_all": _ratio(total_depart_delay_all, preferred_denominator),
        "avg_depart_delay_all_per_loaded": _ratio(total_depart_delay_all, loaded),
        "avg_depart_delay_all_per_inserted": _ratio(total_depart_delay_all, inserted),
        "total_depart_delay_all": total_depart_delay_all,
    }


def build_live_congestion_metrics(base_env: Any, seed_row: Dict[str, Any]) -> Dict[str, float]:
    metrics = list(getattr(base_env, "metrics", []) or [])
    delta_time = float(getattr(base_env, "delta_time", 1.0) or 1.0)
    episode_summary = dict(getattr(base_env, "last_episode_summary", {}) or {})
    final_info = dict(getattr(base_env, "last_episode_final_info", {}) or {})
    stopped_series = [
        _finite_float(row.get("system_total_stopped"))
        for row in metrics
        if isinstance(row, dict) and np.isfinite(_finite_float(row.get("system_total_stopped")))
    ]
    return {
        "avg_queue_length": _finite_float(episode_summary.get("resco_queue", seed_row.get("final/resco/queue"))),
        "max_queue": _finite_float(episode_summary.get("resco_max_queue", seed_row.get("validation/resco_queue_max"))),
        "total_queued_vehicles": _finite_float(final_info.get("system_total_queued")),
        "total_stopped_vehicles": _finite_float(final_info.get("system_total_stopped")),
        "stopped_vehicle_seconds": float(np.sum(stopped_series) * delta_time) if stopped_series else float("nan"),
        "avg_waiting_time_live": _finite_float(final_info.get("system_mean_waiting_time")),
        "total_waiting_time_live": _finite_float(final_info.get("system_total_waiting_time")),
        "avg_speed_live": _finite_float(final_info.get("system_mean_speed")),
        "avg_pressure_live": _finite_float(final_info.get("system_mean_pressure")),
        "occupancy_mean": float("nan"),
        "jam_teleports": _finite_float(seed_row.get("teleports_jam", seed_row.get("final/safety/total_teleported"))),
    }


def _flatten_lane_values(mapping: Dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for row in mapping.values():
        values.extend(float(value) for value in row if np.isfinite(float(value)))
    return values


def build_fairness_metrics(base_env: Any, phase_queue_traces: Dict[str, list[Dict[str, Any]]]) -> Dict[str, float]:
    lane_waiting = getattr(base_env, "last_episode_lane_waiting_times", {}) or {}
    lane_queues = getattr(base_env, "last_episode_lane_queue_levels", {}) or {}
    lane_wait_values = _flatten_lane_values(lane_waiting)
    lane_queue_values = _flatten_lane_values(lane_queues)

    phase_wait_values: list[float] = []
    for signal_id, waits in lane_waiting.items():
        traffic_signal = getattr(base_env, "traffic_signals", {}).get(signal_id)
        if traffic_signal is None:
            continue
        lane_map = {lane: float(value) for lane, value in zip(getattr(traffic_signal, "lanes", []), waits)}
        for phase_lanes in getattr(traffic_signal, "phase_lanes", []) or []:
            phase_wait_values.append(float(sum(lane_map.get(lane, 0.0) for lane in phase_lanes)))

    final_phase_queues: list[float] = []
    per_agent_queue_imbalance: list[float] = []
    for rows in phase_queue_traces.values():
        if not rows:
            continue
        phase_queues = [float(value) for value in rows[-1].get("phase_queues", [])]
        if not phase_queues:
            continue
        final_phase_queues.extend(phase_queues)
        per_agent_queue_imbalance.append(float(max(phase_queues) - min(phase_queues)))

    return {
        "fairness/max_waiting_time_completed": _finite_float(getattr(base_env, "last_episode_summary", {}).get("tripinfo/max_waiting_time")),
        "fairness/std_waiting_time_completed": _finite_float(getattr(base_env, "last_episode_summary", {}).get("tripinfo/std_waiting_time")),
        "fairness/max_queue": _finite_float(getattr(base_env, "last_episode_summary", {}).get("resco_max_queue")),
        "fairness/queue_imbalance": float(np.mean(per_agent_queue_imbalance)) if per_agent_queue_imbalance else float("nan"),
        "fairness/phase_waiting_time_std": float(np.std(phase_wait_values)) if phase_wait_values else float("nan"),
        "fairness/phase_queue_std": float(np.std(final_phase_queues)) if final_phase_queues else float("nan"),
        "fairness/lane_waiting_time_std": float(np.std(lane_wait_values)) if lane_wait_values else float("nan"),
        "fairness/lane_queue_std": float(np.std(lane_queue_values)) if lane_queue_values else float("nan"),
        "fairness/jain_waiting_time": jain_index(lane_wait_values),
        "fairness/gini_waiting_time": gini_coefficient(lane_wait_values),
    }


def build_resource_metrics(
    *,
    validation_wall_clock_seconds: float,
    ray_init_seconds: float,
    checkpoint_restore_seconds: float,
    env_build_seconds: float,
    evaluation_seconds: float,
    video_record_seconds: float,
    decision_steps: int,
    action_latency_seconds: list[float],
    control_interval_seconds: float,
    statistic_metrics: Dict[str, float],
) -> Dict[str, float]:
    latencies_ms = np.asarray([float(value) * 1000.0 for value in action_latency_seconds if np.isfinite(float(value))], dtype=float)
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms.size else float("nan")
    return {
        "resource/validation_wall_clock_seconds": float(validation_wall_clock_seconds),
        "resource/ray_init_seconds": float(ray_init_seconds),
        "resource/checkpoint_restore_seconds": float(checkpoint_restore_seconds),
        "resource/env_build_seconds": float(env_build_seconds),
        "resource/evaluation_seconds": float(evaluation_seconds),
        "resource/video_record_seconds": float(video_record_seconds),
        "resource/sumo_clock_duration": _finite_float(statistic_metrics.get("sumo_clock_duration")),
        "resource/sumo_traci_duration": _finite_float(statistic_metrics.get("sumo_traci_duration")),
        "resource/sumo_real_time_factor": _finite_float(statistic_metrics.get("sumo_real_time_factor")),
        "resource/sumo_vehicle_updates_per_second": _finite_float(statistic_metrics.get("sumo_vehicle_updates_per_second")),
        "resource/decision_steps": float(decision_steps),
        "resource/mean_action_selection_latency_ms": float(np.mean(latencies_ms)) if latencies_ms.size else float("nan"),
        "resource/p95_action_selection_latency_ms": p95_latency_ms,
        "resource/max_action_selection_latency_ms": float(np.max(latencies_ms)) if latencies_ms.size else float("nan"),
        "resource/deadline_ratio_p95": _ratio(p95_latency_ms / 1000.0, control_interval_seconds),
    }


def enrich_seed_row(
    *,
    seed_row: Dict[str, Any],
    base_env: Any,
    statistic_output_path: str | Path | None,
    phase_queue_traces: Dict[str, list[Dict[str, Any]]],
    resource_timings: Dict[str, Any],
    controller: str,
    scenario: str,
    seed: int,
    checkpoint_label: str | None = None,
    disable_completed_view: bool = False,
) -> Dict[str, Any]:
    statistic_metrics = load_statistic_output(statistic_output_path)
    stats_object = parse_statistic_output(statistic_output_path)
    enriched = dict(seed_row)
    enriched["method/controller"] = controller
    enriched["scenario"] = scenario
    enriched["seed"] = float(seed)
    if checkpoint_label:
        enriched["checkpoint/label"] = checkpoint_label
    enriched.update(build_completed_trip_metrics(seed_row, disable_completed_view=disable_completed_view))
    enriched.update(build_all_demand_metrics(seed_row, stats_object))
    enriched.update(build_live_congestion_metrics(base_env, seed_row))
    enriched.update(statistic_metrics)
    enriched.update(build_fairness_metrics(base_env, phase_queue_traces))
    enriched.update(
        build_resource_metrics(
            validation_wall_clock_seconds=float(resource_timings.get("validation_wall_clock_seconds", float("nan"))),
            ray_init_seconds=float(resource_timings.get("ray_init_seconds", float("nan"))),
            checkpoint_restore_seconds=float(resource_timings.get("checkpoint_restore_seconds", float("nan"))),
            env_build_seconds=float(resource_timings.get("env_build_seconds", float("nan"))),
            evaluation_seconds=float(resource_timings.get("evaluation_seconds", float("nan"))),
            video_record_seconds=float(resource_timings.get("video_record_seconds", float("nan"))),
            decision_steps=int(resource_timings.get("decision_steps", 0) or 0),
            action_latency_seconds=list(resource_timings.get("action_latency_seconds", []) or []),
            control_interval_seconds=float(resource_timings.get("control_interval_seconds", 1.0) or 1.0),
            statistic_metrics=statistic_metrics,
        )
    )
    return enriched


def aggregate_numeric_rows(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    if not rows:
        return aggregate
    aggregate["method/controller"] = rows[0].get("method/controller", "")
    aggregate["scenario"] = rows[0].get("scenario", "")
    aggregate["seed"] = float("nan")
    aggregate["status"] = "ok" if all(str(row.get("status", "ok")) == "ok" for row in rows) else "partial"
    numeric_values: Dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                numeric_values.setdefault(key, []).append(float(value))
    for key, values in numeric_values.items():
        aggregate[key] = float(np.mean(values))
        if len(values) > 1:
            aggregate[f"{key}__std"] = float(np.std(values))
    return aggregate


def format_metric_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    numeric = _finite_float(value)
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1000:
        return f"{numeric:.1f}"
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"
