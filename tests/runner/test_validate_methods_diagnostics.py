# ruff: noqa: E402

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import validate_methods


class DummyLaneDomain:
    def __init__(self, lane_ids, lane_vehicle_ids=None):
        self._lane_ids = list(lane_ids)
        self._lane_vehicle_ids = lane_vehicle_ids or {}

    def getIDList(self):
        return list(self._lane_ids)

    def getLastStepVehicleIDs(self, lane_id):
        return list(self._lane_vehicle_ids.get(lane_id, []))


class DummyVehicleDomain:
    def __init__(self, speeds=None, waiting_times=None):
        self._speeds = speeds or {}
        self._waiting_times = waiting_times or {}

    def getSpeed(self, vehicle_id):
        return self._speeds.get(vehicle_id, 0.0)

    def getWaitingTime(self, vehicle_id):
        return self._waiting_times.get(vehicle_id, 0.0)

    def getAllowedSpeed(self, vehicle_id):
        return 10.0


def test_gnej143_diagnostic_override_replaces_phase_2_with_upstream_lanes():
    traffic_signal = SimpleNamespace(
        phase_lanes=[
            ["phase_0_lane"],
            ["phase_1_lane"],
            ["201956821#1.68_1"],
        ],
        lanes_length={
            "201956821#1.68_1": 24.32,
            "201956821#1.68_2": 24.32,
            "201956821#1.68_3": 24.32,
        },
        _phase_stats_cache_step=123,
        _phase_stats_cache=[{"average_speed": 1.0, "max_waiting_time": 0.0}],
        sumo=SimpleNamespace(
            lane=DummyLaneDomain(
                [
                    "201956821#0_1",
                    "201956821#0_2",
                    "201956821#0_3",
                    "10425609#0_1",
                    "10425609#0_2",
                    "10425609#0_3",
                    "201956821#1.68_1",
                ]
            )
        ),
    )
    env = SimpleNamespace(sim_step=0, traffic_signals={"gneJ143": traffic_signal})

    label = validate_methods._apply_diagnostic_phase_lane_override(env, "gneJ143")

    assert label == "ingolstadt7_gneJ143_phase_2_10425609_upstream"
    assert traffic_signal.phase_lanes == [
        ["phase_0_lane"],
        ["phase_1_lane"],
        [
            "10425609#0_1",
            "10425609#0_2",
            "10425609#0_3",
        ],
    ]
    assert traffic_signal._phase_stats_cache_step is None
    assert traffic_signal._phase_stats_cache is None


def test_gnej143_diagnostic_override_uses_only_known_upstream_lanes():
    traffic_signal = SimpleNamespace(
        phase_lanes=[
            ["phase_0_lane"],
            ["phase_1_lane"],
            ["201956821#1.68_1"],
        ],
        _phase_stats_cache_step=None,
        _phase_stats_cache=None,
        sumo=SimpleNamespace(lane=DummyLaneDomain(["10425609#0_1", "10425609#0_2"])),
    )
    env = SimpleNamespace(sim_step=0, traffic_signals={"gneJ143": traffic_signal})

    label = validate_methods._apply_diagnostic_phase_lane_override(env, "gneJ143")

    assert label == "ingolstadt7_gneJ143_phase_2_10425609_upstream"
    assert traffic_signal.phase_lanes[2] == ["10425609#0_1", "10425609#0_2"]


def test_diagnostic_override_ignores_other_junctions():
    traffic_signal = SimpleNamespace(phase_lanes=[["lane_a"], ["lane_b"], ["lane_c"]])
    env = SimpleNamespace(sim_step=0, traffic_signals={"other": traffic_signal})

    label = validate_methods._apply_diagnostic_phase_lane_override(env, "other")

    assert label == ""
    assert traffic_signal.phase_lanes == [["lane_a"], ["lane_b"], ["lane_c"]]


def test_gnej143_diagnostic_row_logs_target_lane_and_edge_counts():
    lane_vehicle_ids = {
        "10425609#0_1": ["veh_a", "veh_b"],
        "10425609#0_2": ["veh_c"],
        "10425609#0_3": ["ghost_veh", "veh_d"],
        "10425609#1_1": ["veh_e"],
        "10425609#1_2": [],
        "10425609#1_3": ["veh_f"],
    }
    speeds = {
        "veh_a": 0.0,
        "veh_b": 1.0,
        "veh_c": 0.0,
        "ghost_veh": 0.0,
        "veh_d": 0.0,
        "veh_e": 0.0,
        "veh_f": 4.0,
    }
    traffic_signal = SimpleNamespace(
        green_phase=0,
        phase_lanes=[
            ["phase_0_lane"],
            ["phase_1_lane"],
            ["10425609#0_1", "10425609#0_2", "10425609#0_3"],
        ],
        reward_fn="diff-waiting-time",
        reward_nsw_window_seconds=0,
        reward_nash_epsilon=0.0,
        sumo=SimpleNamespace(
            lane=DummyLaneDomain(lane_vehicle_ids.keys(), lane_vehicle_ids),
            vehicle=DummyVehicleDomain(speeds=speeds),
        ),
    )
    traffic_signal._get_unique_phase_vehicle_ids = lambda lanes: [
        vehicle_id
        for lane in lanes
        for vehicle_id in lane_vehicle_ids.get(lane, [])
        if not str(vehicle_id).startswith("ghost")
    ]
    traffic_signal.get_phase_queued_counts = lambda: [0, 0, 3]
    traffic_signal.get_phase_average_speeds = lambda: [1.0, 1.0, 0.2]
    traffic_signal.get_phase_max_waiting_times = lambda: [0.0, 0.0, 0.0]
    env = SimpleNamespace(sim_step=57640, traffic_signals={"gneJ143": traffic_signal})

    row = validate_methods._build_junction_diagnostic_row(
        env,
        junction_id="gneJ143",
        decision_step=9,
        seed=0,
        chosen_action=0,
        phase_lane_override="ingolstadt7_gneJ143_phase_2_10425609_upstream",
    )

    assert row["phase_2/queue_count"] == 3
    assert row["target_lane/10425609#0_1/queue_count"] == 1
    assert row["target_lane/10425609#0_2/queue_count"] == 1
    assert row["target_lane/10425609#0_3/queue_count"] == 1
    assert row["target_lane/10425609#0_3/raw_queue_count"] == 2
    assert row["target_edge/10425609#0/queue_count"] == 3
    assert row["target_edge/10425609#0/raw_queue_count"] == 4
    assert row["target_edge/10425609#1/queue_count"] == 1
    assert json.loads(row["diagnostic_target_lanes"]) == [
        "10425609#0_1",
        "10425609#0_2",
        "10425609#0_3",
        "10425609#1_1",
        "10425609#1_2",
        "10425609#1_3",
    ]
