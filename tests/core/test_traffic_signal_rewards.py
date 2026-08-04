# ruff: noqa: E402

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.environment.traffic_signal import TrafficSignal

pytestmark = pytest.mark.core_fast


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


def test_nash_average_speed_reward_uses_phase_geometric_mean() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_windowed_phase_speed_wait_stats = lambda: ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])

    reward = signal._nash_average_speed_reward()

    assert reward == pytest.approx(0.6)


def test_weighted_nash_average_speed_reward_emphasizes_high_wait_phase() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_windowed_phase_speed_wait_stats = lambda: ([0.9, 0.2], [2.0, 8.0])

    reward = signal._weighted_nash_average_speed_reward()

    expected = math.exp(0.2 * math.log(1.0) + 0.8 * math.log(0.3))
    assert reward == pytest.approx(expected)


def test_weighted_nash_average_speed_reward_uses_uniform_weights_when_waits_are_zero() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_windowed_phase_speed_wait_stats = lambda: ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])

    reward = signal._weighted_nash_average_speed_reward()

    assert reward == pytest.approx(1.1)


def test_nash_average_speed_reward_uses_configured_epsilon() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.01
    signal.get_windowed_phase_speed_wait_stats = lambda: ([0.5, 0.5], [0.0, 0.0])

    reward = signal._nash_average_speed_reward()

    assert reward == pytest.approx(0.51)


def test_nsw_cycle_length_derivation_uses_signal_program_durations() -> None:
    short_cycle = [SimpleNamespace(duration=10), SimpleNamespace(duration=3), SimpleNamespace(duration=20)]
    long_cycle = [SimpleNamespace(duration=25), SimpleNamespace(duration=5), SimpleNamespace(duration=30)]

    assert TrafficSignal._derive_fixed_cycle_length_seconds(short_cycle) == pytest.approx(33.0)
    assert TrafficSignal._derive_fixed_cycle_length_seconds(long_cycle) == pytest.approx(60.0)
    assert TrafficSignal._compute_nsw_window_seconds(33.0, 0.5) == 17
    assert TrafficSignal._compute_nsw_window_seconds(33.0, 1.0) == 33
    assert TrafficSignal._compute_nsw_window_seconds(33.0, 1.5) == 50
    assert TrafficSignal._compute_nsw_window_seconds(0.1, 0.5) == 1


def test_windowed_nsw_stats_trim_to_configured_window_and_warm_up() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal._nsw_window_samples = deque(maxlen=2)
    stats_by_step = iter(
        [
            [{"average_speed": 0.2, "max_waiting_time": 1.0}],
            [{"average_speed": 0.4, "max_waiting_time": 3.0}],
            [{"average_speed": 1.0, "max_waiting_time": 2.0}],
        ]
    )
    signal._get_phase_speed_wait_stats = lambda: next(stats_by_step)

    signal.record_nsw_window_sample()
    speeds, waits = signal.get_windowed_phase_speed_wait_stats()
    assert speeds == pytest.approx([0.2])
    assert waits == pytest.approx([1.0])

    signal.record_nsw_window_sample()
    signal.record_nsw_window_sample()

    speeds, waits = signal.get_windowed_phase_speed_wait_stats()
    assert speeds == pytest.approx([0.7])
    assert waits == pytest.approx([3.0])


def test_windowed_nsw_vehicle_counts_trim_to_configured_window_and_warm_up() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal._nsw_window_samples = deque(maxlen=2)
    stats_by_step = iter(
        [
            [{"average_speed": 0.2, "max_waiting_time": 1.0, "vehicle_count": 2}],
            [{"average_speed": 0.4, "max_waiting_time": 3.0, "vehicle_count": 4}],
            [{"average_speed": 1.0, "max_waiting_time": 2.0, "vehicle_count": 8}],
        ]
    )
    signal._get_phase_speed_wait_stats = lambda: next(stats_by_step)

    signal.record_nsw_window_sample()
    assert signal.get_windowed_phase_vehicle_counts() == pytest.approx([2.0])

    signal.record_nsw_window_sample()
    signal.record_nsw_window_sample()

    assert signal.get_windowed_phase_vehicle_counts() == pytest.approx([6.0])


def test_windowed_nash_average_speed_reward_uses_window_mean_phase_speeds() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal._nsw_window_samples = deque(
        [
            {"average_speeds": [0.2, 0.8], "max_waiting_times": [1.0, 4.0]},
            {"average_speeds": [0.6, 0.4], "max_waiting_times": [3.0, 2.0]},
        ],
        maxlen=5,
    )

    reward = signal._nash_average_speed_reward()

    expected = math.exp(0.5 * (math.log(0.4 + 0.1) + math.log(0.6 + 0.1)))
    assert reward == pytest.approx(expected)


def test_windowed_weighted_nash_average_speed_reward_uses_window_max_waits() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal._nsw_window_samples = deque(
        [
            {"average_speeds": [0.8, 0.2], "max_waiting_times": [2.0, 1.0]},
            {"average_speeds": [0.4, 0.6], "max_waiting_times": [4.0, 8.0]},
        ],
        maxlen=5,
    )

    reward = signal._weighted_nash_average_speed_reward()

    expected = math.exp((4.0 / 12.0) * math.log(0.6 + 0.1) + (8.0 / 12.0) * math.log(0.4 + 0.1))
    assert reward == pytest.approx(expected)


def test_windowed_vehicle_weighted_nash_average_speed_reward_uses_window_mean_vehicle_counts() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal._nsw_window_samples = deque(
        [
            {"average_speeds": [0.8, 0.2], "max_waiting_times": [2.0, 1.0], "vehicle_counts": [2, 8]},
            {"average_speeds": [0.4, 0.6], "max_waiting_times": [4.0, 8.0], "vehicle_counts": [6, 2]},
        ],
        maxlen=5,
    )

    reward = signal._vehicle_weighted_nash_average_speed_reward()

    expected = math.exp((4.0 / 9.0) * math.log(0.6 + 0.1) + (5.0 / 9.0) * math.log(0.4 + 0.1))
    assert reward == pytest.approx(expected)


def test_vehicle_weighted_nash_average_speed_reward_uses_uniform_weights_when_counts_are_zero() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_windowed_phase_speed_wait_stats = lambda: ([0.3, 0.7], [0.0, 0.0])
    signal.get_windowed_phase_vehicle_counts = lambda: [0.0, 0.0]

    reward = signal._vehicle_weighted_nash_average_speed_reward()

    expected = math.exp(0.5 * math.log(0.4) + 0.5 * math.log(0.8))
    assert reward == pytest.approx(expected)


def test_phase_average_speeds_and_max_waits_handle_empty_phases() -> None:
    class DummyLaneAPI:
        @staticmethod
        def getLastStepVehicleIDs(lane):
            if lane == "lane_a":
                return ["veh_1", "veh_2"]
            return []

        @staticmethod
        def getLength(_lane):
            return 100.0

        @staticmethod
        def getLastStepLength(_lane):
            return 5.0

    class DummyVehicleAPI:
        @staticmethod
        def getSpeed(veh):
            return {"veh_1": 6.0, "veh_2": 3.0}[veh]

        @staticmethod
        def getAllowedSpeed(veh):
            return {"veh_1": 12.0, "veh_2": 6.0}[veh]

        @staticmethod
        def getWaitingTime(veh):
            return {"veh_1": 4.0, "veh_2": 9.0}[veh]

    class DummySumo:
        lane = DummyLaneAPI()
        vehicle = DummyVehicleAPI()

    signal = TrafficSignal.__new__(TrafficSignal)
    signal.sumo = DummySumo()
    signal.phase_lanes = [["lane_a"], ["lane_b"]]

    assert signal.get_phase_average_speeds() == pytest.approx([0.5, 1.0])
    assert signal.get_phase_max_waiting_times() == pytest.approx([9.0, 0.0])


def test_phase_speed_wait_stats_cache_is_reused_within_same_step() -> None:
    lane_call_count = 0

    class DummyLaneAPI:
        @staticmethod
        def getLastStepVehicleIDs(lane):
            nonlocal lane_call_count
            lane_call_count += 1
            if lane == "lane_a":
                return ["veh_1", "veh_2"]
            return ["veh_3"]

        @staticmethod
        def getLength(_lane):
            return 100.0

        @staticmethod
        def getLastStepLength(_lane):
            return 5.0

    class DummyVehicleAPI:
        @staticmethod
        def getSpeed(veh):
            return {"veh_1": 6.0, "veh_2": 3.0, "veh_3": 2.0}[veh]

        @staticmethod
        def getAllowedSpeed(veh):
            return {"veh_1": 12.0, "veh_2": 6.0, "veh_3": 4.0}[veh]

        @staticmethod
        def getWaitingTime(veh):
            return {"veh_1": 4.0, "veh_2": 9.0, "veh_3": 7.0}[veh]

    class DummySumo:
        lane = DummyLaneAPI()
        vehicle = DummyVehicleAPI()

    class DummyEnv:
        sim_step = 10

    signal = TrafficSignal.__new__(TrafficSignal)
    signal.sumo = DummySumo()
    signal.env = DummyEnv()
    signal.phase_lanes = [["lane_a"], ["lane_b"]]
    signal._phase_stats_cache_step = None
    signal._phase_stats_cache = None

    assert signal.get_phase_average_speeds() == pytest.approx([0.5, 0.5])
    assert signal.get_phase_max_waiting_times() == pytest.approx([9.0, 7.0])
    assert lane_call_count == 2

    signal.env.sim_step = 11
    assert signal.get_phase_average_speeds() == pytest.approx([0.5, 0.5])
    assert lane_call_count == 4
