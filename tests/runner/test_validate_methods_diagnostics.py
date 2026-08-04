# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import validate_methods


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
    )
    env = SimpleNamespace(sim_step=0, traffic_signals={"gneJ143": traffic_signal})

    label = validate_methods._apply_diagnostic_phase_lane_override(env, "gneJ143")

    assert label == "ingolstadt7_gneJ143_phase_2_201956821_upstream"
    assert traffic_signal.phase_lanes == [
        ["phase_0_lane"],
        ["phase_1_lane"],
        [
            "201956821#0_1",
            "201956821#0_2",
            "201956821#0_3",
        ],
    ]
    assert traffic_signal._phase_stats_cache_step is None
    assert traffic_signal._phase_stats_cache is None


def test_diagnostic_override_ignores_other_junctions():
    traffic_signal = SimpleNamespace(phase_lanes=[["lane_a"], ["lane_b"], ["lane_c"]])
    env = SimpleNamespace(sim_step=0, traffic_signals={"other": traffic_signal})

    label = validate_methods._apply_diagnostic_phase_lane_override(env, "other")

    assert label == ""
    assert traffic_signal.phase_lanes == [["lane_a"], ["lane_b"], ["lane_c"]]
