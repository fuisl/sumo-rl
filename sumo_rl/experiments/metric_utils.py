from __future__ import annotations

from typing import Any, Dict

import numpy as np


SYSTEM_METRIC_NAMESPACE_MAP = {
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


def flatten_numeric_metric_values(data: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            nested_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[nested_prefix] = float(value)
            elif isinstance(value, dict):
                metrics.update(flatten_numeric_metric_values(value, nested_prefix))
    return metrics


def map_system_metrics_to_namespaces(info: Any) -> Dict[str, float]:
    """Map raw `system_*` info fields into the public efficiency/safety namespaces."""

    flat_metrics = flatten_numeric_metric_values(info)
    namespaced: Dict[str, float] = {}

    for source_key, target_key in SYSTEM_METRIC_NAMESPACE_MAP.items():
        if source_key in flat_metrics:
            namespaced[target_key] = float(flat_metrics[source_key])

    return namespaced


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
