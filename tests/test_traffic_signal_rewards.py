from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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
    signal.get_phase_average_speeds = lambda: [0.5, 0.5, 0.5]

    reward = signal._nash_average_speed_reward()

    assert reward == pytest.approx(0.6)


def test_weighted_nash_average_speed_reward_emphasizes_high_wait_phase() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_phase_average_speeds = lambda: [0.9, 0.2]
    signal.get_phase_max_waiting_times = lambda: [2.0, 8.0]

    reward = signal._weighted_nash_average_speed_reward()

    expected = math.exp(0.2 * math.log(1.0) + 0.8 * math.log(0.3))
    assert reward == pytest.approx(expected)


def test_weighted_nash_average_speed_reward_uses_uniform_weights_when_waits_are_zero() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.1
    signal.get_phase_average_speeds = lambda: [1.0, 1.0, 1.0]
    signal.get_phase_max_waiting_times = lambda: [0.0, 0.0, 0.0]

    reward = signal._weighted_nash_average_speed_reward()

    assert reward == pytest.approx(1.1)


def test_nash_average_speed_reward_uses_configured_epsilon() -> None:
    signal = TrafficSignal.__new__(TrafficSignal)
    signal.reward_nash_epsilon = 0.01
    signal.get_phase_average_speeds = lambda: [0.5, 0.5]

    reward = signal._nash_average_speed_reward()

    assert reward == pytest.approx(0.51)


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
