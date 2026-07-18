"""Static traffic-signal control policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class StaticPolicy(Protocol):
    """Protocol for deterministic or heuristic traffic-signal policies."""

    def select_action(self, traffic_signal) -> int:
        """Return the selected action index for the provided traffic signal."""

        raise NotImplementedError


@dataclass
class _PhaseScorer:
    def _phase_lane_sets(self, traffic_signal, phase_state: str) -> tuple[set[str], set[str]]:
        links = traffic_signal.sumo.trafficlight.getControlledLinks(traffic_signal.id)
        incoming_lanes: set[str] = set()
        outgoing_lanes: set[str] = set()
        for link_index, link in enumerate(links):
            if link_index >= len(phase_state):
                break
            if phase_state[link_index].lower() not in {"g", "s"}:
                continue
            for signal_link in link or ():
                if not isinstance(signal_link, tuple | list) or len(signal_link) < 2:
                    continue
                incoming_lane = signal_link[0]
                outgoing_lane = signal_link[1]
                if isinstance(incoming_lane, str) and incoming_lane:
                    incoming_lanes.add(incoming_lane)
                if isinstance(outgoing_lane, str) and outgoing_lane:
                    outgoing_lanes.add(outgoing_lane)
        return incoming_lanes, outgoing_lanes

    def score(self, traffic_signal, phase_state: str) -> float:
        incoming_lanes, outgoing_lanes = self._phase_lane_sets(traffic_signal, phase_state)
        incoming_queued = sum(float(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)) for lane in incoming_lanes)
        outgoing_queued = sum(float(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)) for lane in outgoing_lanes)
        return incoming_queued - outgoing_queued


class MaxPressurePolicy:
    """Choose the green phase with the highest pressure score."""

    def __init__(self):
        """Initialize the reusable pressure scorer."""

        self._scorer = _PhaseScorer()

    def select_action(self, traffic_signal) -> int:
        """Return the phase index with the greatest incoming-outgoing queue gap."""

        best_action = 0
        best_score = float("-inf")
        for phase_index, phase in enumerate(getattr(traffic_signal, "green_phases", [])):
            score = self._scorer.score(traffic_signal, phase.state)
            if score > best_score:
                best_score = score
                best_action = phase_index
        return best_action
