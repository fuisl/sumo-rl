from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def numeric_metrics(data: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            nested_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[nested_prefix] = float(value)
            elif isinstance(value, dict):
                metrics.update(numeric_metrics(value, nested_prefix))
    return metrics


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 1.0

    series = np.asarray([float(value) for value in values], dtype=float)
    denominator = float(np.sum(series**2))
    if denominator <= 0.0:
        return 1.0
    numerator = float(np.sum(series) ** 2)
    return float(numerator / (series.size * denominator))


def extract_agent_waiting_times(info: Any) -> Dict[str, float]:
    flat_metrics = numeric_metrics(info)
    waiting_times: Dict[str, float] = {}
    for key, value in flat_metrics.items():
        if key.endswith("_accumulated_waiting_time") and not key.startswith("agents_total_"):
            agent_id = key[: -len("_accumulated_waiting_time")]
            waiting_times[agent_id] = float(value)
    return waiting_times


def extract_lane_waiting_times(env: Any) -> Dict[str, list[float]]:
    lane_waiting_times: Dict[str, list[float]] = {}
    traffic_signals = getattr(env, "traffic_signals", {})
    current_snapshot = getattr(env, "last_lane_waiting_times", {}) or {}
    episode_snapshot = getattr(env, "last_episode_lane_waiting_times", {}) or {}

    for agent_id, signal in traffic_signals.items():
        values = current_snapshot.get(agent_id) or episode_snapshot.get(agent_id)
        if not values and getattr(env, "sumo", None) is not None:
            values = signal.get_accumulated_waiting_time_per_lane()
        if values:
            lane_waiting_times[agent_id] = [float(value) for value in values]

    if lane_waiting_times:
        return lane_waiting_times

    for agent_id, values in episode_snapshot.items():
        if values:
            lane_waiting_times[agent_id] = [float(value) for value in values]
    return lane_waiting_times


def namespace_system_metrics(info: Any) -> Dict[str, float]:
    flat_metrics = numeric_metrics(info)
    namespaced: Dict[str, float] = {}
    mapping = {
        "system_total_running": "efficiency_total_running",
        "system_total_backlogged": "efficiency_total_backlogged",
        "system_total_stopped": "efficiency_total_stopped",
        "system_total_queued": "efficiency_total_queued",
        "system_mean_queued": "efficiency_mean_queued",
        "system_mean_pressure": "efficiency_mean_pressure",
        "system_mean_average_speed": "efficiency_mean_average_speed",
        "system_max_queue": "efficiency_max_queue",
        "system_total_arrived": "efficiency_total_arrived",
        "system_total_departed": "efficiency_total_departed",
        "system_total_teleported": "safety_total_teleported",
        "system_total_emergency_brake": "safety_total_emergency_brake",
        "system_total_collisions": "safety_total_collisions",
        "system_total_waiting_time": "efficiency_total_waiting_time",
        "system_mean_waiting_time": "efficiency_mean_waiting_time",
        "system_mean_speed": "efficiency_mean_speed",
    }

    for source_key, target_key in mapping.items():
        if source_key in flat_metrics:
            namespaced[target_key] = float(flat_metrics[source_key])

    return namespaced


def namespace_fairness_metrics(info: Any, include_agent_metrics_local: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    waiting_times = extract_agent_waiting_times(info)
    fairness_metrics: Dict[str, float] = {}
    agent_metrics: Dict[str, float] = {}

    waiting_values = list(waiting_times.values())
    if waiting_values:
        fairness_metrics["fairness_jain_waiting_time"] = jain_fairness(waiting_values)
        fairness_metrics["fairness_waiting_time_mean"] = float(np.mean(waiting_values))
        fairness_metrics["fairness_waiting_time_std"] = float(np.std(waiting_values))

    if include_agent_metrics_local:
        for agent_id, value in waiting_times.items():
            agent_metrics[f"fairness_waiting_time_{agent_id}"] = float(value)

    return fairness_metrics, agent_metrics


def namespace_lane_fairness_metrics(env: Any) -> Dict[str, float]:
    lane_waiting_times = extract_lane_waiting_times(env)
    metrics: Dict[str, float] = {}
    jain_values: list[float] = []
    mean_values: list[float] = []
    std_values: list[float] = []

    for agent_id, values in lane_waiting_times.items():
        lane_prefix = f"fairness_lane/{agent_id}"
        jain_value = jain_fairness(values)
        mean_value = float(np.mean(values)) if values else 0.0
        std_value = float(np.std(values)) if values else 0.0
        metrics[f"{lane_prefix}/jain_waiting_time"] = jain_value
        metrics[f"{lane_prefix}/waiting_time_mean"] = mean_value
        metrics[f"{lane_prefix}/waiting_time_std"] = std_value
        jain_values.append(jain_value)
        mean_values.append(mean_value)
        std_values.append(std_value)

    metrics["fairness_lane/jain_waiting_time_mean"] = float(np.mean(jain_values)) if jain_values else 1.0
    metrics["fairness_lane/waiting_time_mean"] = float(np.mean(mean_values)) if mean_values else 0.0
    metrics["fairness_lane/waiting_time_std"] = float(np.mean(std_values)) if std_values else 0.0
    return metrics


def build_namespaced_metrics(info: Any, include_agent_metrics_local: bool = False) -> tuple[Dict[str, float], Dict[str, float]]:
    system_metrics = namespace_system_metrics(info)
    fairness_metrics, agent_metrics = namespace_fairness_metrics(info, include_agent_metrics_local)
    metrics = dict(system_metrics)
    metrics.update(fairness_metrics)
    return metrics, agent_metrics


def reward_formula_text(reward_fn: Any, reward_weights: Any = None) -> str:
    if isinstance(reward_fn, list):
        if reward_weights is not None:
            return "weighted_sum(reward_fn_i) across the configured reward functions"
        return "vector_reward(reward_fn_i) across the configured reward functions"

    reward_name = str(reward_fn)
    if reward_name == "diff-waiting-time":
        return "last_waiting_time - current_waiting_time, where current_waiting_time = sum(accumulated_waiting_time_per_lane) / 100"
    if reward_name == "average-speed":
        return "average vehicle speed for the signal"
    if reward_name == "queue":
        return "- total queued vehicles for the signal"
    if reward_name == "pressure":
        return "vehicle_count(outgoing_lanes) - vehicle_count(incoming_lanes)"
    if reward_name == "co2":
        return "- total CO2 emissions for the signal"
    if callable(reward_fn):
        return f"custom callable reward function: {getattr(reward_fn, '__name__', 'anonymous_reward_fn')}"
    return f"custom or unresolved reward function: {reward_name}"


def add_namespace_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {}
    for key, value in row.items():
        if "/" not in key:
            continue
        if key.startswith("debug/"):
            continue
        aliases[key.replace("/", "_")] = value
    row.update(aliases)
    return row


def keep_namespaced_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if "/" in key}
