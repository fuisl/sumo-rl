from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def is_ghost_vehicle(vehicle_id: str) -> bool:
    return isinstance(vehicle_id, str) and vehicle_id.startswith("ghost")


def is_truthy_xml_value(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: str | None, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_tripinfo_vehicle(attrs: Mapping[str, str]) -> str:
    depart = _safe_float(attrs.get("depart"), default=-1.0)
    arrival = _safe_float(attrs.get("arrival"), default=-1.0)
    vaporized_reason = str(attrs.get("vaporized", "") or "").strip().lower()

    if is_truthy_xml_value(attrs.get("unfinished")):
        return "undeparted" if depart < 0.0 else "running_unfinished"
    if depart < 0.0:
        return "undeparted"
    if vaporized_reason:
        return "running_unfinished"
    if arrival < 0.0:
        return "running_unfinished"
    return "finished"


@dataclass
class TripinfoParseResult:
    delay_values: list[float] = field(default_factory=list)
    duration_values: list[float] = field(default_factory=list)
    wait_values: list[float] = field(default_factory=list)
    time_loss_values: list[float] = field(default_factory=list)
    finished_count: int = 0
    running_unfinished_count: int = 0
    undeparted_count: int = 0

    @property
    def unfinished_count(self) -> int:
        return int(self.running_unfinished_count + self.undeparted_count)

    @property
    def total_count(self) -> int:
        return int(self.finished_count + self.unfinished_count)


def collect_tripinfo_metrics(vehicles) -> TripinfoParseResult:
    result = TripinfoParseResult()
    for vehicle in vehicles:
        vehicle_id = str(vehicle.attrib.get("id", "") or "")
        if is_ghost_vehicle(vehicle_id):
            continue

        status = classify_tripinfo_vehicle(vehicle.attrib)
        if status == "running_unfinished":
            result.running_unfinished_count += 1
            continue
        if status == "undeparted":
            result.undeparted_count += 1
            continue

        result.finished_count += 1
        time_loss = _safe_float(vehicle.attrib.get("timeLoss"), default=0.0)
        depart_delay = _safe_float(vehicle.attrib.get("departDelay"), default=0.0)
        result.delay_values.append(time_loss + depart_delay)
        result.duration_values.append(_safe_float(vehicle.attrib.get("duration"), default=0.0))
        result.wait_values.append(_safe_float(vehicle.attrib.get("waitingTime"), default=0.0))
        result.time_loss_values.append(time_loss)
    return result
