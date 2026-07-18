from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


def _safe_float(value, *, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class StatisticOutputParseResult:
    clock_begin: float = float("nan")
    clock_end: float = float("nan")
    clock_duration: float = float("nan")
    traci_duration: float = float("nan")
    real_time_factor: float = float("nan")
    vehicle_updates_per_second: float = float("nan")
    person_updates_per_second: float = float("nan")
    begin: float = float("nan")
    end: float = float("nan")
    duration: float = float("nan")
    vehicles_loaded: float = float("nan")
    vehicles_inserted: float = float("nan")
    vehicles_running: float = float("nan")
    vehicles_waiting: float = float("nan")
    teleports_total: float = float("nan")
    teleports_jam: float = float("nan")
    teleports_yield: float = float("nan")
    teleports_wrong_lane: float = float("nan")
    collisions: float = float("nan")
    emergency_stops: float = float("nan")
    trip_count: float = float("nan")
    trip_route_length: float = float("nan")
    trip_speed: float = float("nan")
    trip_duration: float = float("nan")
    trip_waiting_time: float = float("nan")
    trip_time_loss: float = float("nan")
    trip_depart_delay: float = float("nan")
    trip_depart_delay_waiting: float = float("nan")
    trip_total_travel_time: float = float("nan")
    trip_total_depart_delay: float = float("nan")


def _find_first(root: ET.Element, *tags: str) -> ET.Element | None:
    for tag in tags:
        element = root.find(f".//{tag}")
        if element is not None:
            return element
    return None


def parse_statistic_output(path: str | Path | None) -> StatisticOutputParseResult:
    if not path:
        return StatisticOutputParseResult()
    statistic_path = Path(path)
    if not statistic_path.exists():
        return StatisticOutputParseResult()

    try:
        root = ET.parse(statistic_path).getroot()
    except (ET.ParseError, OSError):
        return StatisticOutputParseResult()

    performance = _find_first(root, "performance")
    timing = _find_first(root, "timing")
    vehicles = _find_first(root, "vehicles")
    teleports = _find_first(root, "teleports")
    safety = _find_first(root, "safety")
    trip_stats = _find_first(root, "vehicleTripStatistics")

    result = StatisticOutputParseResult()
    if performance is not None:
        result.clock_begin = _safe_float(performance.attrib.get("clockBegin"))
        result.clock_end = _safe_float(performance.attrib.get("clockEnd"))
        result.clock_duration = _safe_float(performance.attrib.get("clockDuration"))
        result.traci_duration = _safe_float(performance.attrib.get("traciDuration"))
        result.real_time_factor = _safe_float(performance.attrib.get("realTimeFactor"))
        result.vehicle_updates_per_second = _safe_float(performance.attrib.get("vehicleUpdatesPerSecond"))
        result.person_updates_per_second = _safe_float(performance.attrib.get("personUpdatesPerSecond"))
    if timing is not None:
        result.begin = _safe_float(timing.attrib.get("begin"))
        result.end = _safe_float(timing.attrib.get("end"))
        result.duration = _safe_float(timing.attrib.get("duration"))
    if vehicles is not None:
        result.vehicles_loaded = _safe_float(vehicles.attrib.get("loaded"))
        result.vehicles_inserted = _safe_float(vehicles.attrib.get("inserted"))
        result.vehicles_running = _safe_float(vehicles.attrib.get("running"))
        result.vehicles_waiting = _safe_float(vehicles.attrib.get("waiting"))
    if teleports is not None:
        result.teleports_total = _safe_float(teleports.attrib.get("total"))
        result.teleports_jam = _safe_float(teleports.attrib.get("jam"))
        result.teleports_yield = _safe_float(teleports.attrib.get("yield"))
        result.teleports_wrong_lane = _safe_float(teleports.attrib.get("wrongLane"))
    if safety is not None:
        result.collisions = _safe_float(safety.attrib.get("collisions"))
        result.emergency_stops = _safe_float(safety.attrib.get("emergencyStops"))
    if trip_stats is not None:
        result.trip_count = _safe_float(trip_stats.attrib.get("count"))
        result.trip_route_length = _safe_float(trip_stats.attrib.get("routeLength"))
        result.trip_speed = _safe_float(trip_stats.attrib.get("speed"))
        result.trip_duration = _safe_float(trip_stats.attrib.get("duration"))
        result.trip_waiting_time = _safe_float(trip_stats.attrib.get("waitingTime"))
        result.trip_time_loss = _safe_float(trip_stats.attrib.get("timeLoss"))
        result.trip_depart_delay = _safe_float(trip_stats.attrib.get("departDelay"))
        result.trip_depart_delay_waiting = _safe_float(trip_stats.attrib.get("departDelayWaiting"))
        result.trip_total_travel_time = _safe_float(trip_stats.attrib.get("totalTravelTime"))
        result.trip_total_depart_delay = _safe_float(trip_stats.attrib.get("totalDepartDelay"))
    return result
