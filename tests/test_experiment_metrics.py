import importlib.util
import sys
from pathlib import Path
import types

import numpy as np
from sumo_rl.environment.env import SumoEnvironment


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


def _summary_env(tmp_path, tripinfo_xml: str):
    tripinfo_prefix = tmp_path / "tripinfo" / "resco"
    tripinfo_path = Path(f"{tripinfo_prefix}_conn0_ep1.xml")
    tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
    tripinfo_path.write_text(tripinfo_xml)

    env = SumoEnvironment.__new__(SumoEnvironment)
    env.episode = 1
    env.begin_time = 0
    env.sim_max_time = 100
    env.sumo = None
    env.metrics = [
        {"step": 5.0, "system_mean_queued": 2.0, "system_max_queue": 4.0},
        {"step": 10.0, "system_mean_queued": 4.0, "system_max_queue": 7.0},
    ]
    env.ts_ids = ["tls_1", "tls_2"]
    env.tripinfo_output_name = str(tripinfo_prefix)
    env.keep_tripinfo_output = False
    env.label = "0"
    env.last_episode_summary = {}
    env.last_episode_final_info = {}
    env.last_episode_lane_waiting_times = {}
    env.last_lane_waiting_times = {"tls_1": [1.0], "tls_2": [2.0]}
    env.episode_agent_reward_totals = {"tls_1": 3.0, "tls_2": 5.0}
    env.completed_episode_summaries = []
    return env, tripinfo_path


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


def test_resco_tripinfo_metrics_match_benchmark_formulas_and_delete_xml(tmp_path) -> None:
    env, tripinfo_path = _summary_env(
        tmp_path,
        """
        <tripinfos>
            <tripinfo id="veh_1" duration="30" waitingTime="5" timeLoss="8" departDelay="2" />
            <tripinfo id="veh_2" duration="50" waitingTime="7" timeLoss="10" departDelay="4" />
            <tripinfo id="ghost_1" duration="999" waitingTime="999" timeLoss="999" departDelay="999" />
            <tripinfo id="veh_3" duration="20" waitingTime="3" timeLoss="4" departDelay="1" vaporized="true" />
        </tripinfos>
        """,
    )

    summary = env.finalize_episode_summary(parse_tripinfo=True)

    assert summary["tripinfo/finished_count"] == 2.0
    assert summary["tripinfo/unfinished_count"] == 1.0
    assert summary["tripinfo/avg_delay"] == 12.0
    assert summary["resco_avg_delay"] == 12.0
    assert summary["resco_delay_mean"] == 12.0
    assert summary["resco_delay_max"] == 14.0
    assert summary["resco_delay_std"] == 2.0
    assert summary["resco_trip_time"] == 40.0
    assert summary["resco_trip_time_mean"] == 40.0
    assert summary["resco_wait"] == 6.0
    assert summary["resco_wait_mean"] == 6.0
    assert summary["resco_wait_max"] == 7.0
    assert summary["resco_queue"] == 3.0
    assert summary["resco_queue_mean"] == 3.0
    assert summary["resco_max_queue"] == 7.0
    assert summary["resco_queue_max"] == 7.0
    assert summary["reward/mean"] == 4.0
    assert summary["reward/max"] == 5.0
    assert summary["reward/std"] == 1.0
    assert summary["reward/agent/tls_1"] == 3.0
    assert summary["tripinfo/parse_success"] == 1.0
    assert summary["tripinfo/parse_pending"] == 0.0
    assert env.completed_episode_summaries[-1]["resco_trip_time"] == 40.0
    assert not tripinfo_path.exists()


def test_empty_tripinfo_xml_is_a_successful_parse_with_no_finished_trips(tmp_path) -> None:
    env, tripinfo_path = _summary_env(tmp_path, "<tripinfos></tripinfos>")

    summary = env.finalize_episode_summary(parse_tripinfo=True)

    assert summary["tripinfo/parse_success"] == 1.0
    assert summary["tripinfo/finished_count"] == 0.0
    assert summary["tripinfo/total_count"] == 0.0
    assert np.isnan(summary["resco_trip_time"])
    assert not tripinfo_path.exists()


def test_pending_tripinfo_summary_is_replaced_after_sumo_close(tmp_path) -> None:
    env, _ = _summary_env(
        tmp_path,
        """
        <tripinfos>
            <tripinfo id="veh_1" duration="30" waitingTime="5" timeLoss="8" departDelay="2" />
        </tripinfos>
        """,
    )

    pending = env.finalize_episode_summary(parse_tripinfo=False)
    parsed = env.finalize_episode_summary(parse_tripinfo=True)

    assert pending["tripinfo/parse_pending"] == 1.0
    assert parsed["tripinfo/parse_pending"] == 0.0
    assert parsed["resco_avg_delay"] == 10.0
    assert len(env.completed_episode_summaries) == 1
    assert env.completed_episode_summaries[0]["resco_avg_delay"] == 10.0


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
                "resco_avg_delay_std": 1.25,
                "resco_trip_time": 34.0,
                "resco_wait": 7.0,
                "resco_wait_std": 0.5,
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
    assert row["resco_avg_delay_std"] == 1.25
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
                "resco_avg_delay_std": 1.25,
                "resco_trip_time": 34.0,
                "resco_wait": 7.0,
                "resco_wait_std": 0.5,
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
    assert row["eval/mean_reward"] == 1.5
    assert row["eval/std_reward"] == 0.25
    assert row["final/resco/avg_delay"] == 12.0
    assert row["final/resco/avg_delay_std"] == 1.25
    assert row["final/resco/wait_std"] == 0.5
    assert row["eval/resco/avg_delay"] == 12.0
    assert row["eval/resco/avg_delay_std"] == 1.25
    assert row["eval/safety/total_emergency_brake"] == 2.0
    assert row["eval/safety/total_collisions"] == 1.0
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
