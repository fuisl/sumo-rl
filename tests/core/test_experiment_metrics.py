# ruff: noqa: E402

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.environment.env import SumoEnvironment

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

_map_system_metrics_to_namespaces = _MODULE.map_system_metrics_to_namespaces
_build_episode_benchmark_summary_row = _RUNNER_MODULE._build_episode_benchmark_summary_row
_build_final_eval_summary_row = _RUNNER_MODULE._build_final_eval_summary_row
_get_validation_run_seeds = _RUNNER_MODULE._get_validation_run_seeds
_emit_baseline_reference_validation_rows = _RUNNER_MODULE._emit_baseline_reference_validation_rows
_static_validation_summary_row = _RUNNER_MODULE._static_validation_summary_row
_run_static_validation_episode_trace = _RUNNER_MODULE._run_static_validation_episode_trace


pytestmark = pytest.mark.core_fast


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

    metrics = _map_system_metrics_to_namespaces(info)

    assert metrics["efficiency_mean_speed"] == 8.5
    assert metrics["safety_total_emergency_brake"] == 3.0
    assert metrics["safety_total_teleported"] == 1.0
    assert metrics["safety_total_collisions"] == 2.0


def test_resco_tripinfo_metrics_include_dispatched_unfinished_vehicles_and_delete_xml(tmp_path) -> None:
    env, tripinfo_path = _summary_env(
        tmp_path,
        """
        <tripinfos>
            <tripinfo
                id="veh_1" depart="0" arrival="30" duration="30" waitingTime="5" timeLoss="8" departDelay="2" vaporized=""
            />
            <tripinfo
                id="veh_2" depart="5" arrival="55" duration="50" waitingTime="7" timeLoss="10" departDelay="4" vaporized=""
            />
            <tripinfo id="ghost_1" duration="999" waitingTime="999" timeLoss="999" departDelay="999" />
            <tripinfo
                id="veh_3" depart="10" arrival="-1" duration="20" waitingTime="3" timeLoss="4" departDelay="1" vaporized=""
            />
            <tripinfo id="veh_4" depart="-1" arrival="-1" duration="0" waitingTime="0" timeLoss="0" departDelay="0" />
        </tripinfos>
        """,
    )

    summary = env.finalize_episode_summary(parse_tripinfo=True)

    assert summary["tripinfo/finished_count"] == 2.0
    assert summary["tripinfo/running_unfinished_count"] == 1.0
    assert summary["tripinfo/undeparted_count"] == 1.0
    assert summary["tripinfo/unfinished_count"] == 2.0
    assert summary["tripinfo/total_count"] == 4.0
    assert abs(summary["tripinfo/avg_delay"] - (29.0 / 3.0)) <= 1e-9
    assert abs(summary["resco_avg_delay"] - (29.0 / 3.0)) <= 1e-9
    assert abs(summary["resco_delay_mean"] - (29.0 / 3.0)) <= 1e-9
    assert summary["resco_delay_max"] == 14.0
    assert abs(summary["resco_delay_std"] - np.std([10.0, 14.0, 5.0])) <= 1e-9
    assert abs(summary["resco_trip_time"] - (100.0 / 3.0)) <= 1e-9
    assert abs(summary["resco_trip_time_mean"] - (100.0 / 3.0)) <= 1e-9
    assert summary["resco_wait"] == 5.0
    assert summary["resco_wait_mean"] == 5.0
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
    assert abs(env.completed_episode_summaries[-1]["resco_trip_time"] - (100.0 / 3.0)) <= 1e-9
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
            <tripinfo
                id="veh_1" depart="0" arrival="30" duration="30" waitingTime="5" timeLoss="8" departDelay="2" vaporized=""
            />
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
            self.reward_penalty_lambda = 0.1
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

    row = _build_episode_benchmark_summary_row(
        DummyBaseEnv(), extra={"algorithm/kind": "fixed_time", "static/policy": "fixed_time"}
    )

    assert row["algorithm/kind"] == "fixed_time"
    assert row["static/policy"] == "fixed_time"
    assert row["resco_avg_delay"] == 12.0
    assert row["resco_avg_delay_std"] == 1.25
    assert row["efficiency_mean_speed"] == 8.5
    assert row["safety_total_emergency_brake"] == 3.0
    assert row["safety_total_collisions"] == 2.0
    assert row["reward/formula"] == (
        "last_waiting_time - current_waiting_time, where current_waiting_time = sum(accumulated_waiting_time_per_lane) / 100"
    )


def test_resco_summary_row_includes_unchosen_phase_penalty_formula() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.sumo = None
            self.reward_fn = "diff-waiting-time-with-unchosen-phase-penalty"
            self.reward_weights = None
            self.reward_penalty_lambda = 0.25
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
            self.last_episode_final_info = {}
            self.last_lane_waiting_times = {"agent_a": []}
            self.last_episode_lane_waiting_times = {"agent_a": [1.0, 3.0]}
            self.traffic_signals = {"agent_a": object()}

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

    row = _build_episode_benchmark_summary_row(DummyBaseEnv(), extra={"algorithm/kind": "fixed_time"})

    assert row["reward/formula"] == (
        "last_waiting_time - current_waiting_time - "
        "0.25 * max(cumulative_waiting_time_per_unchosen_phase / queue_length_per_unchosen_phase), "
        "where current_waiting_time = sum(accumulated_waiting_time_per_lane) / 100 and phases with zero queue contribute 0"
    )


def test_resco_summary_row_includes_weighted_nash_average_speed_formula() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.sumo = None
            self.reward_fn = "weighted-nash-average-speed"
            self.reward_weights = None
            self.reward_penalty_lambda = None
            self.reward_nash_epsilon = 0.05
            self.reward_nsw_window_cycle_multiplier = 1.5
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
            self.last_episode_final_info = {}
            self.last_lane_waiting_times = {"agent_a": []}
            self.last_episode_lane_waiting_times = {"agent_a": [1.0, 3.0]}
            self.traffic_signals = {"agent_a": object()}

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

    row = _build_episode_benchmark_summary_row(DummyBaseEnv(), extra={"algorithm/kind": "fixed_time"})

    assert row["reward/formula"] == (
        "exp(sum(window_phase_weight * log(window_mean_phase_average_speed + 0.05))) across green phases, "
        "where the rolling window is 1.5 * fixed_time_cycle_length for each signal, "
        "window_phase_weight = window_max_phase_waiting_time / sum(window_max_phase_waiting_time), "
        "empty phases use average_speed = 1.0 and max_waiting_time = 0, "
        "and zero total max waiting falls back to uniform phase weights"
    )


def test_resco_summary_row_includes_windowed_nash_average_speed_formula() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.sumo = None
            self.reward_fn = "nash-average-speed"
            self.reward_weights = None
            self.reward_penalty_lambda = None
            self.reward_nash_epsilon = 0.05
            self.reward_nsw_window_cycle_multiplier = 0.5
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
            self.last_episode_final_info = {}
            self.last_lane_waiting_times = {"agent_a": []}
            self.last_episode_lane_waiting_times = {"agent_a": [1.0, 3.0]}
            self.traffic_signals = {"agent_a": object()}

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

    row = _build_episode_benchmark_summary_row(DummyBaseEnv(), extra={"algorithm/kind": "fixed_time"})

    assert row["reward/formula"] == (
        "geometric_mean(window_mean_phase_average_speed + 0.05) across green phases, "
        "where the rolling window is 0.5 * fixed_time_cycle_length for each signal "
        "and empty phases use average_speed = 1.0"
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
            self.reward_penalty_lambda = 0.1
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
                "tripinfo/running_unfinished_count": 1.0,
                "tripinfo/undeparted_count": 2.0,
                "tripinfo/unfinished_count": 3.0,
                "tripinfo/total_count": 7.0,
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
    assert row["final/resco/avg_delay_std"] == 1.25
    assert row["final/resco/wait_std"] == 0.5
    assert row["final/efficiency/total_arrived"] == 8.0
    assert row["final/efficiency/total_departed"] == 6.0
    assert row["final/efficiency/total_running"] == 10.0
    assert row["final/safety/total_teleported"] == 1.0
    assert row["final/safety/total_emergency_brake"] == 2.0
    assert row["final/safety/total_collisions"] == 1.0
    assert "eval/mean_reward" not in row
    assert "eval/resco/avg_delay" not in row
    assert "eval/safety/total_collisions" not in row
    assert "final/fairness/jain_waiting_time" not in row
    assert row["tripinfo/finished_count"] == 4.0
    assert row["tripinfo/running_unfinished_count"] == 1.0
    assert row["tripinfo/undeparted_count"] == 2.0
    assert row["warnings/no_finished_trips"] is False
    assert row["warnings/no_final_summary_metrics"] is False
    assert row["debug/has_metrics"] is True
    assert row["debug/num_seconds"] == 3600.0


def test_validation_run_seeds_prefers_eval_seeds_and_respects_eval_episode_count() -> None:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            seed=42,
            runs=5,
            seeds=[10, 11, 12, 13, 14],
            eval_seeds=[1, 2, 3, 4, 5],
            eval_episodes=3,
        )
    )

    assert _get_validation_run_seeds(cfg) == [1, 2, 3]


def test_emit_baseline_reference_validation_rows_replays_constant_validation_points() -> None:
    calls = []
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episode_seconds=3600, validation_interval_episodes=5),
        env=SimpleNamespace(kwargs=SimpleNamespace(delta_time=5)),
        logging=SimpleNamespace(
            baseline_line_max_episode_index=15,
            baseline_line_episode_stride=None,
        ),
    )

    original_log_outputs = _RUNNER_MODULE._log_outputs
    try:
        _RUNNER_MODULE._log_outputs = lambda wandb_run, csv_run, metrics, step=None: calls.append((metrics, step))
        _emit_baseline_reference_validation_rows(
            cfg,
            wandb_run=None,
            csv_run=None,
            validation_row={
                "algorithm/kind": "fixed_time",
                "validation/resco_delay_mean": 12.0,
                "validation/pass_index": 1.0,
                "validation/episode_index": 5.0,
                "validation/env_step": 3600.0,
            },
        )
    finally:
        _RUNNER_MODULE._log_outputs = original_log_outputs

    assert [metrics["validation/rollout_index"] for metrics, _step in calls] == [5.0, 10.0, 15.0]
    assert [metrics["validation/episode_index"] for metrics, _step in calls] == [5.0, 10.0, 15.0]
    assert [metrics["validation/env_step"] for metrics, _step in calls] == [3600.0, 7200.0, 10800.0]
    assert all(metrics["validation/resco_delay_mean"] == 12.0 for metrics, _step in calls)
    assert all(metrics["validation/reference_line"] is True for metrics, _step in calls)


def test_static_validation_summary_row_keeps_only_validation_and_warning_metrics() -> None:
    row = _static_validation_summary_row(
        {
            "algorithm/kind": "fixed_time",
            "validation/resco_delay_mean": 7.5,
            "warnings/no_finished_trips": False,
            "eval/episode": 5.0,
            "final/eval/mean_reward": 9.0,
            "final/resco/avg_delay": 7.5,
        },
        step=3600,
        episode_index=5,
        policy_name="fixed_time",
        pass_index=1,
    )

    assert row["algorithm/kind"] == "fixed_time"
    assert row["static/policy"] == "fixed_time"
    assert row["validation/env_step"] == 3600.0
    assert row["validation/rollout_index"] == 5.0
    assert row["validation/episode_index"] == 5.0
    assert row["validation/pass_index"] == 1.0
    assert row["validation/resco_delay_mean"] == 7.5
    assert row["validation/warnings/no_finished_trips"] is False
    assert "eval/episode" not in row
    assert "validation/eval/episode" not in row
    assert all(not key.startswith("final/") for key in row)


def test_run_static_validation_episode_trace_uses_none_actions_for_fixed_time() -> None:
    class DummySignal:
        def __init__(self, green_phase: int) -> None:
            self.green_phase = green_phase

    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.ts_ids = ["tls_1", "tls_2"]
            self.keep_tripinfo_output = False
            self.episode_agent_reward_totals = {"tls_1": 2.0, "tls_2": 3.0}
            self.traffic_signals = {
                "tls_1": DummySignal(1),
                "tls_2": DummySignal(0),
            }

        def action_spaces(self, agent_id):
            del agent_id
            return SimpleNamespace(n=2)

    class DummyEnv:
        def __init__(self) -> None:
            self.base_env = DummyBaseEnv()
            self.actions = []
            self.steps = 0

        def reset(self):
            return {}, {}

        def step(self, action):
            self.actions.append(action)
            self.steps += 1
            done = self.steps >= 2
            self.base_env.traffic_signals["tls_1"].green_phase = self.steps % 2
            self.base_env.traffic_signals["tls_2"].green_phase = 0
            return {}, {}, False, done, {}

        def close(self):
            return None

    helper = SimpleNamespace(
        _validation_tripinfo_output_path=lambda env: Path("dummy-tripinfo.xml"),
        _collect_phase_queue_snapshot=lambda env, agent_ids: {
            agent_id: {"active_phase": 0, "phase_queues": [1, 2]} for agent_id in agent_ids
        },
    )

    env = DummyEnv()
    original_helpers = _RUNNER_MODULE._rllib_validation_helpers
    original_summary = _RUNNER_MODULE._get_completed_episode_summary
    try:
        _RUNNER_MODULE._rllib_validation_helpers = lambda: helper
        _RUNNER_MODULE._get_completed_episode_summary = lambda env: {"reward/mean": 2.5}
        (
            base_env,
            reward_total,
            episode_summary,
            action_traces,
            action_space_sizes,
            phase_queue_traces,
            tripinfo_path,
        ) = _run_static_validation_episode_trace(env, policy=None)
    finally:
        _RUNNER_MODULE._rllib_validation_helpers = original_helpers
        _RUNNER_MODULE._get_completed_episode_summary = original_summary

    assert base_env is env.base_env
    assert reward_total == 5.0
    assert episode_summary["reward/mean"] == 2.5
    assert env.actions == [None, None]
    assert action_traces["tls_1"] == [1, 0]
    assert action_space_sizes["tls_1"] == 2
    assert len(phase_queue_traces["tls_1"]) == 2
    assert tripinfo_path == Path("dummy-tripinfo.xml")


def test_run_static_validation_episode_trace_uses_mapping_actions_for_parallel_fixed_time_envs() -> None:
    class DummySignal:
        def __init__(self, green_phase: int) -> None:
            self.green_phase = green_phase

    class DummyBaseEnv:
        def __init__(self) -> None:
            self.metrics = []
            self.ts_ids = ["tls_1"]
            self.keep_tripinfo_output = False
            self.episode_agent_reward_totals = {"tls_1": 4.0}
            self.traffic_signals = {"tls_1": DummySignal(0)}

        def action_spaces(self, agent_id):
            del agent_id
            return SimpleNamespace(n=2)

    class DummyParallelEnv:
        def __init__(self) -> None:
            self.base_env = DummyBaseEnv()
            self.possible_agents = ["tls_1"]
            self.agents = ["tls_1"]
            self.actions = []
            self.steps = 0

        def reset(self):
            return {}, {}

        def step(self, action):
            self.actions.append(action)
            self.steps += 1
            done = self.steps >= 2
            self.base_env.traffic_signals["tls_1"].green_phase = self.steps % 2
            return {}, {}, False, done, {}

        def close(self):
            return None

    helper = SimpleNamespace(
        _validation_tripinfo_output_path=lambda env: Path("dummy-tripinfo.xml"),
        _collect_phase_queue_snapshot=lambda env, agent_ids: {
            agent_id: {"active_phase": 0, "phase_queues": [1, 2]} for agent_id in agent_ids
        },
    )

    env = DummyParallelEnv()
    original_helpers = _RUNNER_MODULE._rllib_validation_helpers
    original_summary = _RUNNER_MODULE._get_completed_episode_summary
    try:
        _RUNNER_MODULE._rllib_validation_helpers = lambda: helper
        _RUNNER_MODULE._get_completed_episode_summary = lambda env: {"reward/mean": 4.0}
        _run_static_validation_episode_trace(env, policy=None)
    finally:
        _RUNNER_MODULE._rllib_validation_helpers = original_helpers
        _RUNNER_MODULE._get_completed_episode_summary = original_summary

    assert env.actions == [{"tls_1": 0}, {"tls_1": 1}]


def test_max_pressure_policy_skips_malformed_controlled_links() -> None:
    from sumo_rl.agents.static.policies import MaxPressurePolicy

    class DummyLaneAPI:
        def getLastStepHaltingNumber(self, lane_id):
            return {"in_ok": 4, "out_ok": 1}.get(lane_id, 0)

    class DummyTrafficLightAPI:
        def getControlledLinks(self, traffic_light_id):
            del traffic_light_id
            return [
                (),
                ((None,),),
                (("in_ok", "out_ok", None),),
            ]

    traffic_signal = SimpleNamespace(
        id="tls_1",
        green_phases=[SimpleNamespace(state="GGG")],
        sumo=SimpleNamespace(
            trafficlight=DummyTrafficLightAPI(),
            lane=DummyLaneAPI(),
        ),
    )

    assert MaxPressurePolicy().select_action(traffic_signal) == 0


def test_max_pressure_policy_scores_all_unique_lanes_in_phase() -> None:
    from sumo_rl.agents.static.policies import MaxPressurePolicy

    class DummyLaneAPI:
        def getLastStepHaltingNumber(self, lane_id):
            values = {
                "in_a": 5,
                "in_b": 4,
                "out_a": 1,
                "out_b": 2,
                "out_c": 6,
            }
            return values.get(lane_id, 0)

    class DummyTrafficLightAPI:
        def getControlledLinks(self, traffic_light_id):
            del traffic_light_id
            return [
                (("in_a", "out_a", None), ("in_a", "out_b", None)),
                (("in_b", "out_c", None),),
            ]

    traffic_signal = SimpleNamespace(
        id="tls_1",
        green_phases=[
            SimpleNamespace(state="Gr"),
            SimpleNamespace(state="rG"),
        ],
        sumo=SimpleNamespace(
            trafficlight=DummyTrafficLightAPI(),
            lane=DummyLaneAPI(),
        ),
    )

    assert MaxPressurePolicy().select_action(traffic_signal) == 0


def test_run_validation_only_static_baseline_logs_only_repeated_validation_rows_and_graphs_once() -> None:
    logged_rows = []
    action_plot_calls = []
    tripinfo_calls = []
    run_calls = {"count": 0}

    class DummyEnv:
        pass

    helper = SimpleNamespace(
        _extract_validation_seed_artifacts=lambda **kwargs: {"seed": kwargs["seed"]},
        _aggregate_validation_tripinfo_distributions=lambda artifacts: {
            "waiting_time": [],
            "delay": [],
            "pooled_waiting_time": [],
            "pooled_delay": [],
            "total_seeds": len(artifacts),
            "seeds_with_completed_trips": 0,
            "seeds_without_completed_trips": len(artifacts),
            "total_completed_trips": 0,
            "total_unfinished_trips": 0,
            "total_trips": 0,
        },
        _build_validation_action_plot_rows=lambda *args, **kwargs: {"tls_1": [{"step": 1.0, "action_0": 1.0}]},
        _build_validation_action_timeline_rows=lambda *args, **kwargs: {"tls_1": [0]},
        _build_validation_phase_queue_rows=lambda *args, **kwargs: {
            "tls_1": [{"step": 1.0, "active_phase": 0.0, "phase_0": 1.0}]
        },
        _validation_action_window_steps=lambda cfg: 12,
        _validation_action_plot_max_agents=lambda logging_cfg: None,
        _log_validation_action_plot_images=lambda *args, **kwargs: action_plot_calls.append((args, kwargs)),
        _log_validation_tripinfo_distribution_images=lambda *args, **kwargs: tripinfo_calls.append((args, kwargs)),
        decision_interval_seconds=lambda cfg: 5,
    )

    def fake_run_static_trace(env, *, policy=None):
        del env, policy
        run_calls["count"] += 1
        index = run_calls["count"]
        base_env = SimpleNamespace(
            ts_ids=["tls_1"],
            traffic_signals={"tls_1": object()},
            save_csv=lambda prefix, idx: None,
        )
        episode_summary = {
            "reward/mean": float(2 * index),
            "reward/max": float(2 * index + 1),
            "reward/std": 1.0,
            "resco_delay_mean": float(10 + index),
            "resco_delay_max": float(12 + index),
            "resco_delay_std": 0.5,
            "resco_wait_mean": float(5 + index),
            "resco_wait_max": float(6 + index),
            "resco_wait_std": 0.25,
            "resco_queue_mean": float(index),
            "resco_queue_max": float(index + 2),
            "resco_trip_time_mean": float(20 + index),
            "resco_tripinfo_count": float(7 + index),
            "system_total_arrived": float(10 + index),
            "system_total_departed": float(11 + index),
            "system_total_teleported": float(index - 1),
            "system_total_emergency_brake": float(index),
            "system_total_collisions": 0.0,
        }
        return (
            base_env,
            float(100 + index),
            episode_summary,
            {"tls_1": [0, 1]},
            {"tls_1": 2},
            {"tls_1": [{"step": 1.0, "active_phase": 0, "phase_queues": [1, 2]}]},
            Path(f"tripinfo_{index}.xml"),
        )

    def fake_build_final_eval_summary_row(env, **kwargs):
        del env
        return {
            "algorithm/kind": kwargs["algorithm_kind"],
            "final/eval/mean_reward": kwargs["eval_mean_reward"],
            "final/eval/std_reward": kwargs["eval_std_reward"],
            "warnings/no_finished_trips": False,
            "warnings/no_departed_vehicles": False,
            "warnings/no_arrived_vehicles": False,
            "warnings/no_final_summary_metrics": False,
            "warnings/eval_episodes_too_low": False,
            "warnings/all_zero_traffic_metrics": False,
        }

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episode_seconds=100),
        env=SimpleNamespace(kwargs=SimpleNamespace(delta_time=5)),
        logging=SimpleNamespace(
            baseline_line_max_episode_index=15,
            baseline_line_episode_stride=5,
            save_tripinfo_output=False,
        ),
    )

    original_helpers = _RUNNER_MODULE._rllib_validation_helpers
    original_build_env = _RUNNER_MODULE._build_env
    original_run_trace = _RUNNER_MODULE._run_static_validation_episode_trace
    original_build_final = _RUNNER_MODULE._build_final_eval_summary_row
    original_get_seeds = _RUNNER_MODULE._get_validation_run_seeds
    original_log_outputs = _RUNNER_MODULE._log_outputs
    original_prepare_env_kwargs = _RUNNER_MODULE._prepare_env_kwargs
    try:
        _RUNNER_MODULE._rllib_validation_helpers = lambda: helper
        _RUNNER_MODULE._build_env = lambda cfg, run_dir, seed=None: DummyEnv()
        _RUNNER_MODULE._run_static_validation_episode_trace = fake_run_static_trace
        _RUNNER_MODULE._build_final_eval_summary_row = fake_build_final_eval_summary_row
        _RUNNER_MODULE._get_validation_run_seeds = lambda cfg: [1, 2]
        _RUNNER_MODULE._log_outputs = lambda wandb_run, csv_run, metrics, step=None: logged_rows.append((metrics, step))
        _RUNNER_MODULE._prepare_env_kwargs = lambda cfg, run_dir: {"out_csv_name": str(Path("dummy"))}
        _RUNNER_MODULE._run_validation_only_static_baseline(
            cfg,
            Path("."),
            wandb_run=None,
            csv_run=None,
            algorithm_kind="fixed_time",
            policy_name="fixed_time",
            policy=None,
        )
    finally:
        _RUNNER_MODULE._rllib_validation_helpers = original_helpers
        _RUNNER_MODULE._build_env = original_build_env
        _RUNNER_MODULE._run_static_validation_episode_trace = original_run_trace
        _RUNNER_MODULE._build_final_eval_summary_row = original_build_final
        _RUNNER_MODULE._get_validation_run_seeds = original_get_seeds
        _RUNNER_MODULE._log_outputs = original_log_outputs
        _RUNNER_MODULE._prepare_env_kwargs = original_prepare_env_kwargs

    assert [row["validation/rollout_index"] for row, _step in logged_rows] == [5.0, 10.0, 15.0]
    assert [row["validation/episode_index"] for row, _step in logged_rows] == [5.0, 10.0, 15.0]
    assert [row["validation/env_step"] for row, _step in logged_rows] == [100.0, 200.0, 300.0]
    assert all(row["validation/pass_index"] == 1.0 for row, _step in logged_rows)
    assert all(row["static/policy"] == "fixed_time" for row, _step in logged_rows)
    assert all("validation/resco_delay_mean" in row for row, _step in logged_rows)
    assert all("validation/warnings/no_finished_trips" in row for row, _step in logged_rows)
    assert all("final/eval/mean_reward" not in row for row, _step in logged_rows)
    assert all("eval/episode" not in row for row, _step in logged_rows)
    assert all("validation/eval/episode" not in row for row, _step in logged_rows)
    assert len(action_plot_calls) == 1
    assert len(tripinfo_calls) == 1


def test_static_launchers_use_neutral_top_level_configs() -> None:
    fixed_time_launcher = (ROOT / "experiments" / "fixed_time.py").read_text(encoding="utf-8")
    static_launcher = (ROOT / "experiments" / "static_max_pressure.py").read_text(encoding="utf-8")

    assert 'config_name="fixed_time"' in fixed_time_launcher
    assert 'config_name="static_max_pressure"' in static_launcher
    assert 'config_name="presets/resco_grid4x4/fixed_time"' not in fixed_time_launcher
    assert 'config_name="presets/resco_grid4x4/static_max_pressure"' not in static_launcher


def test_static_max_pressure_presets_derive_run_name_from_selected_scenario() -> None:
    preset_paths = [
        ROOT / "configs" / "presets" / "resco_grid4x4" / "static_max_pressure.yaml",
        ROOT / "configs" / "presets" / "resco_cologne1" / "static_max_pressure.yaml",
        ROOT / "configs" / "presets" / "resco_cologne8" / "static_max_pressure.yaml",
        ROOT / "configs" / "presets" / "resco_ingolstadt1" / "static_max_pressure.yaml",
        ROOT / "configs" / "presets" / "resco_ingolstadt21" / "static_max_pressure.yaml",
    ]

    for preset_path in preset_paths:
        preset_text = preset_path.read_text(encoding="utf-8")
        assert "name: ${scenario.name}__static_max_pressure" in preset_text


def test_non_resco_scenario_aliases_point_to_canonical_resco_configs() -> None:
    alias_map = {
        "cologne1": "resco_cologne1",
        "cologne3": "resco_cologne3",
        "cologne8": "resco_cologne8",
        "ingolstadt1": "resco_ingolstadt1",
        "ingolstadt7": "resco_ingolstadt7",
        "ingolstadt21": "resco_ingolstadt21",
    }

    for alias_name, canonical_name in alias_map.items():
        alias_text = (ROOT / "configs" / "scenario" / f"{alias_name}.yaml").read_text(encoding="utf-8")
        assert f"- {canonical_name}" in alias_text


def test_resco_canonical_scenarios_use_raw_sumo_env() -> None:
    canonical_paths = [
        ROOT / "configs" / "scenario" / "resco_cologne1.yaml",
        ROOT / "configs" / "scenario" / "resco_cologne3.yaml",
        ROOT / "configs" / "scenario" / "resco_cologne8.yaml",
        ROOT / "configs" / "scenario" / "resco_ingolstadt1.yaml",
        ROOT / "configs" / "scenario" / "resco_ingolstadt7.yaml",
        ROOT / "configs" / "scenario" / "resco_ingolstadt21.yaml",
    ]

    for scenario_path in canonical_paths:
        scenario_text = scenario_path.read_text(encoding="utf-8")
        assert "factory: sumo_env" in scenario_text


def test_non_resco_preset_aliases_point_to_canonical_resco_presets() -> None:
    alias_map = {
        ROOT / "configs" / "presets" / "cologne8" / "fixed_time.yaml": "/presets/resco_cologne8/fixed_time",
        ROOT / "configs" / "presets" / "cologne8" / "static_max_pressure.yaml": "/presets/resco_cologne8/static_max_pressure",
        ROOT / "configs" / "presets" / "ingolstadt21" / "fixed_time.yaml": "/presets/resco_ingolstadt21/fixed_time",
        ROOT
        / "configs"
        / "presets"
        / "ingolstadt21"
        / "static_max_pressure.yaml": "/presets/resco_ingolstadt21/static_max_pressure",
    }

    for preset_path, alias_target in alias_map.items():
        preset_text = preset_path.read_text(encoding="utf-8")
        assert alias_target in preset_text
