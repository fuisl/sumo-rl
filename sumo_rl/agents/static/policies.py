"""Static traffic-signal control policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class StaticPolicy(Protocol):
    def select_action(self, traffic_signal) -> int:
        raise NotImplementedError


@dataclass
class _PhaseScorer:
    mode: str

    def score(self, traffic_signal, phase_state: str) -> float:
        links = traffic_signal.sumo.trafficlight.getControlledLinks(traffic_signal.id)
        score = 0.0
        for link_index, link in enumerate(links):
            if link_index >= len(phase_state):
                break
            if phase_state[link_index].lower() not in {"g", "s"}:
                continue

            incoming_lane = link[0][0]
            outgoing_lane = link[0][1]
            incoming_queued = traffic_signal.sumo.lane.getLastStepHaltingNumber(incoming_lane)

            if self.mode == "max_pressure":
                outgoing_queued = traffic_signal.sumo.lane.getLastStepHaltingNumber(outgoing_lane)
                score += float(incoming_queued - outgoing_queued)
            else:
                score += float(incoming_queued)

        return score


class _BaseStaticPolicy:
    mode = "greedy"

    def __init__(self):
        self._scorer = _PhaseScorer(self.mode)

    def select_action(self, traffic_signal) -> int:
        best_action = 0
        best_score = float("-inf")
        for phase_index, phase in enumerate(getattr(traffic_signal, "green_phases", [])):
            score = self._scorer.score(traffic_signal, phase.state)
            if score > best_score:
                best_score = score
                best_action = phase_index
        return best_action


class GreedyPolicy(_BaseStaticPolicy):
    mode = "greedy"


class MaxPressurePolicy(_BaseStaticPolicy):
    mode = "max_pressure"
