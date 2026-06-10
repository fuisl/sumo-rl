from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.environment.traffic_signal import TrafficSignal


def test_diff_waiting_time_with_unchosen_phase_penalty_reward_applies_max_penalty() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.last_ts_waiting_time = 10.0
    signal.reward_penalty_lambda = 0.1
    signal.green_phase = 0
    signal.lanes = ["lane_a", "lane_b", "lane_c"]
    signal.phase_lanes = [["lane_a"], ["lane_b"], ["lane_c"]]
    signal.get_accumulated_waiting_time_per_lane = lambda: [100.0, 50.0, 30.0]
    signal.get_phase_queued_counts = lambda: [2, 5, 2]

    reward = signal._diff_waiting_time_with_unchosen_phase_penalty_reward()

    assert reward == pytest.approx(6.7)
    assert signal.last_ts_waiting_time == pytest.approx(1.8)


def test_diff_waiting_time_with_unchosen_phase_penalty_reward_ignores_zero_queue_phases() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.last_ts_waiting_time = 5.0
    signal.reward_penalty_lambda = 0.1
    signal.green_phase = 1
    signal.lanes = ["lane_a", "lane_b", "lane_c"]
    signal.phase_lanes = [["lane_a"], ["lane_b"], ["lane_c"]]
    signal.get_accumulated_waiting_time_per_lane = lambda: [20.0, 30.0, 40.0]
    signal.get_phase_queued_counts = lambda: [0, 3, 0]

    reward = signal._diff_waiting_time_with_unchosen_phase_penalty_reward()

    assert reward == pytest.approx(4.1)
    assert signal.last_ts_waiting_time == pytest.approx(0.9)
