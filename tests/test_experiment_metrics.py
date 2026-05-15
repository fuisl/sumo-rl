import importlib.util
import sys
from pathlib import Path
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODULE_PATH = ROOT / "sumo_rl" / "experiments" / "metric_utils.py"
_SPEC = importlib.util.spec_from_file_location("metric_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_RUNNER_MODULE_PATH = ROOT / "sumo_rl" / "experiments" / "runner.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("runner", _RUNNER_MODULE_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER_MODULE = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER_MODULE)

_build_namespaced_metrics = _MODULE.build_namespaced_metrics
_build_resco_summary_row = _RUNNER_MODULE._build_resco_summary_row
_build_final_eval_summary_row = _RUNNER_MODULE._build_final_eval_summary_row


def test_namespaced_metrics_split_efficiency_and_safety() -> None:
    info = {
        "step": 12,
        "system_mean_speed": 8.5,
        "system_total_emergency_brake": 3,
        "system_total_teleported": 1,
        "system_total_collisions": 2,
    }

    metrics = _build_namespaced_metrics(info)

    assert metrics["efficiency_mean_speed"] == 8.5
    assert metrics["safety_total_emergency_brake"] == 3.0
    assert metrics["safety_total_teleported"] == 1.0
    assert metrics["safety_total_collisions"] == 2.0


def test_resco_summary_row_uses_standard_static_metric_names() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.sumo = None
            self.reward_fn = "diff-waiting-time"
            self.reward_weights = None
            self.last_episode_summary = {
                "episode/index": 3.0,
                "episode/steps": 3600.0,
                "sim_step": 3600.0,
                "resco_avg_delay": 12.0,
                "resco_trip_time": 34.0,
                "resco_wait": 7.0,
                "resco_queue": 2.5,
                "resco_max_queue": 9.0,
            }
            self.last_episode_final_info = {
                "system_mean_speed": 8.5,
                "system_total_emergency_brake": 3.0,
                "system_total_teleported": 1.0,
                "system_total_collisions": 2.0,
            }
            self.last_lane_waiting_times = {"agent_a": [], "agent_b": []}
            self.last_episode_lane_waiting_times = {
                "agent_a": [1.0, 3.0],
                "agent_b": [2.0, 4.0],
            }
            self.traffic_signals = {"agent_a": object(), "agent_b": object()}

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

    row = _build_resco_summary_row(DummyBaseEnv(), extra={"algorithm/kind": "fixed_time", "static/policy": "fixed_time"})

    assert row["algorithm/kind"] == "fixed_time"
    assert row["static/policy"] == "fixed_time"
    assert row["resco_avg_delay"] == 12.0
    assert row["efficiency_mean_speed"] == 8.5
    assert row["safety_total_emergency_brake"] == 3.0
    assert row["safety_total_collisions"] == 2.0
    assert row["reward/formula"] == (
        "last_waiting_time - current_waiting_time, where current_waiting_time = "
        "sum(accumulated_waiting_time_per_lane) / 100"
    )


def test_final_eval_summary_row_uses_standard_final_metric_names() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = [
                {
                    "step": 3600.0,
                    "system_total_running": 10.0,
                    "system_total_backlogged": 2.0,
                    "system_mean_speed": 8.5,
                    "system_mean_waiting_time": 4.0,
                    "system_total_departed": 6.0,
                    "system_total_arrived": 8.0,
                    "system_total_teleported": 1.0,
                    "system_total_emergency_brake": 2.0,
                    "system_total_collisions": 1.0,
                    "agent_a_accumulated_waiting_time": 5.0,
                    "agent_b_accumulated_waiting_time": 10.0,
                }
            ]
            self.sumo = None
            self.reward_fn = "diff-waiting-time"
            self.reward_weights = None
            self.last_episode_summary = {
                "episode/index": 3.0,
                "episode/sim_time_abs": 3600.0,
                "episode/elapsed_seconds": 600.0,
                "resco_avg_delay": 12.0,
                "resco_trip_time": 34.0,
                "resco_wait": 7.0,
                "resco_queue": 2.5,
                "resco_max_queue": 9.0,
                "tripinfo/finished_count": 4.0,
                "tripinfo/unfinished_count": 1.0,
                "tripinfo/total_count": 5.0,
                "tripinfo/avg_duration": 34.0,
                "tripinfo/avg_waiting_time": 7.0,
                "tripinfo/avg_time_loss": 9.0,
            }
            self.last_episode_final_info = self.metrics[-1]
            self.last_episode_lane_waiting_times = {"agent_a": [1.0, 3.0], "agent_b": [2.0, 4.0]}
            self.traffic_signals = {"agent_a": object(), "agent_b": object()}
            self.num_seconds = 3600
            self.sim_max_time = 3600
            self.begin_time = 0

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

        def _build_tripinfo_output_path(self):
            return Path("dummy-tripinfo.xml")

    row = _build_final_eval_summary_row(
        DummyBaseEnv(),
        algorithm_kind="static_max_pressure",
        eval_mean_reward=1.5,
        eval_std_reward=0.25,
        logging_cfg=types.SimpleNamespace(log_final_traffic_metrics=True, debug_metrics=True),
    )

    assert row["algorithm/kind"] == "static_max_pressure"
    assert row["final/eval/mean_reward"] == 1.5
    assert row["final/resco/avg_delay"] == 12.0
    assert row["final/efficiency/total_arrived"] == 8.0
    assert row["final/efficiency/total_departed"] == 6.0
    assert row["final/efficiency/total_running"] == 10.0
    assert row["final/safety/total_teleported"] == 1.0
    assert row["final/safety/total_emergency_brake"] == 2.0
    assert row["final/safety/total_collisions"] == 1.0
    assert "final/fairness/jain_waiting_time" not in row
    assert row["tripinfo/finished_count"] == 4.0
    assert row["warnings/no_finished_trips"] is False
    assert row["warnings/no_final_summary_metrics"] is False
    assert row["debug/has_metrics"] is True
    assert row["debug/num_seconds"] == 3600.0
    assert "eval/mean_reward" not in row
