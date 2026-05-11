import importlib.util
import sys
import types
import tempfile
from pathlib import Path

import pytest

from sumo_rl.agents.ql_agent import QLAgent


_MODULE_PATH = Path(__file__).resolve().parents[1] / "sumo_rl" / "experiments" / "metric_utils.py"
_SPEC = importlib.util.spec_from_file_location("metric_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_SB3_STUB = types.ModuleType("sumo_rl.agents.sb3")
_SB3_STUB.JointMultiAgentActionWrapper = object
_SB3_STUB.SB3WandbCallback = object
_SB3_STUB.__path__ = []
sys.modules.setdefault("sumo_rl.agents.sb3", _SB3_STUB)

_RUNNER_MODULE_PATH = Path(__file__).resolve().parents[1] / "sumo_rl" / "experiments" / "runner.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("runner", _RUNNER_MODULE_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER_MODULE = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER_MODULE)

_CALLBACKS_MODULE_PATH = Path(__file__).resolve().parents[1] / "sumo_rl" / "agents" / "sb3" / "callbacks.py"
_CALLBACKS_SPEC = importlib.util.spec_from_file_location("callbacks", _CALLBACKS_MODULE_PATH)
assert _CALLBACKS_SPEC is not None and _CALLBACKS_SPEC.loader is not None
_CALLBACKS_MODULE = importlib.util.module_from_spec(_CALLBACKS_SPEC)
_CALLBACKS_SPEC.loader.exec_module(_CALLBACKS_MODULE)

_build_namespaced_metrics = _MODULE.build_namespaced_metrics
_jain_fairness = _MODULE.jain_fairness
_build_resco_summary_row = _RUNNER_MODULE._build_resco_summary_row
_build_final_eval_summary_row = _RUNNER_MODULE._build_final_eval_summary_row
_get_sb3_final_log_step = _RUNNER_MODULE._get_sb3_final_log_step
safe_scalar = _CALLBACKS_MODULE.safe_scalar


def test_jain_fairness_prefers_equal_waiting_times() -> None:
    assert _jain_fairness([5.0, 5.0]) == 1.0
    assert _jain_fairness([1.0, 3.0]) < 1.0


def test_namespaced_metrics_split_efficiency_fairness_and_safety() -> None:
    info = {
        "step": 12,
        "system_mean_speed": 8.5,
        "system_total_emergency_brake": 3,
        "system_total_teleported": 1,
        "agent_a_accumulated_waiting_time": 5.0,
        "agent_b_accumulated_waiting_time": 10.0,
    }

    metrics, agent_metrics = _build_namespaced_metrics(info, include_agent_metrics_local=True)

    assert metrics["efficiency_mean_speed"] == 8.5
    assert metrics["safety_total_emergency_brake"] == 3.0
    assert metrics["safety_total_teleported"] == 1.0
    assert metrics["fairness_jain_waiting_time"] == 0.9
    assert metrics["fairness_waiting_time_mean"] == 7.5
    assert agent_metrics["fairness_waiting_time_agent_a"] == 5.0
    assert agent_metrics["fairness_waiting_time_agent_b"] == 10.0


def test_q_learning_learn_returns_td_error() -> None:
    class DummyActionSpace:
        n = 2

    agent = QLAgent(starting_state="s0", state_space=None, action_space=DummyActionSpace(), alpha=0.5, gamma=0.9)
    agent.action = 1
    td_error = agent.learn("s1", reward=2.0)

    assert isinstance(td_error, float)


def test_sb3_summary_row_uses_cached_episode_metrics_after_auto_reset() -> None:
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
                "agent_a_accumulated_waiting_time": 5.0,
                "agent_b_accumulated_waiting_time": 10.0,
            }
            self.last_lane_waiting_times = {"agent_a": [], "agent_b": []}
            self.last_episode_lane_waiting_times = {
                "agent_a": [1.0, 3.0],
                "agent_b": [2.0, 4.0],
            }
            self.traffic_signals = {"agent_a": object(), "agent_b": object()}

        def finalize_episode_summary(self):
            return dict(self.last_episode_summary)

    row = _build_resco_summary_row(DummyBaseEnv(), extra={"algorithm/kind": "ppo_sb3"})

    assert row["resco_avg_delay"] == 12.0
    assert row["efficiency_mean_speed"] == 8.5
    assert row["safety_total_emergency_brake"] == 3.0
    assert row["fairness_jain_waiting_time"] == 0.9
    assert row["fairness_lane/agent_a/waiting_time_mean"] == 2.0
    assert row["fairness_lane/agent_b/waiting_time_mean"] == 3.0
    assert row["fairness_lane/jain_waiting_time_mean"] == pytest.approx(0.85)
    assert row["reward/formula"] == (
        "last_waiting_time - current_waiting_time, where current_waiting_time = "
        "sum(accumulated_waiting_time_per_lane) / 100"
    )


def test_final_eval_summary_row_separates_final_and_eval_metrics() -> None:
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
        algorithm_kind="ppo_sb3",
        eval_mean_reward=1.5,
        eval_std_reward=0.25,
        logging_cfg=types.SimpleNamespace(log_final_traffic_metrics=True, debug_metrics=True),
    )

    assert row["algorithm/kind"] == "ppo_sb3"
    assert row["final/eval/mean_reward"] == 1.5
    assert row["final/resco/avg_delay"] == 12.0
    assert row["final/efficiency/total_arrived"] == 8.0
    assert row["final/efficiency/total_departed"] == 6.0
    assert row["final/efficiency/total_running"] == 10.0
    assert row["final/fairness/jain_waiting_time"] == 0.9
    assert row["final/safety/total_teleported"] == 1.0
    assert row["final/safety/total_emergency_brake"] == 2.0
    assert row["tripinfo/finished_count"] == 4.0
    assert row["warnings/no_finished_trips"] is False
    assert row["warnings/no_final_summary_metrics"] is False
    assert row["debug/has_metrics"] is True
    assert row["debug/num_seconds"] == 3600.0
    assert "eval/mean_reward" not in row


def test_final_eval_summary_uses_cached_completed_episode_after_auto_reset() -> None:
    class DummyBaseEnv:
        def __init__(self) -> None:
            self.episode = 4
            self.metrics = [
                {
                    "step": 25200.0,
                    "system_total_running": 0.0,
                    "system_total_departed": 0.0,
                    "system_total_arrived": 0.0,
                    "system_total_teleported": 0.0,
                    "system_total_emergency_brake": 0.0,
                }
            ]
            self.sumo = None
            self.reward_fn = {"junction_a": "diff-waiting-time"}
            self.reward_weights = None
            self.last_episode_summary = {
                "episode/index": 3.0,
                "episode/sim_time_abs": 25800.0,
                "episode/elapsed_seconds": 600.0,
                "resco_avg_delay": 12.0,
                "resco_trip_time": 34.0,
                "resco_wait": 7.0,
                "resco_queue": 2.5,
                "tripinfo/finished_count": 4.0,
                "tripinfo/unfinished_count": 1.0,
                "tripinfo/total_count": 5.0,
                "tripinfo/avg_duration": 34.0,
                "tripinfo/avg_waiting_time": 7.0,
                "tripinfo/avg_time_loss": 9.0,
            }
            self.last_episode_final_info = {
                "step": 25800.0,
                "system_total_running": 10.0,
                "system_total_departed": 6.0,
                "system_total_arrived": 8.0,
                "system_total_teleported": 1.0,
                "system_total_emergency_brake": 2.0,
                "agent_a_accumulated_waiting_time": 5.0,
                "agent_b_accumulated_waiting_time": 10.0,
            }
            self.num_seconds = 3600
            self.begin_time = 25200

        def finalize_episode_summary(self):
            raise AssertionError("The completed episode cache should be used after auto-reset.")

    row = _build_final_eval_summary_row(
        DummyBaseEnv(),
        algorithm_kind="sac_sb3",
        eval_mean_reward=-10.0,
        eval_std_reward=1.0,
        eval_episodes=1,
        logging_cfg=types.SimpleNamespace(log_final_traffic_metrics=True, debug_metrics=False),
    )

    assert row["episode/sim_time_abs"] == 25800.0
    assert row["episode/elapsed_seconds"] == 600.0
    assert row["final/resco/avg_delay"] == 12.0
    assert row["final/efficiency/total_departed"] == 6.0
    assert row["final/efficiency/total_arrived"] == 8.0
    assert row["final/safety/total_teleported"] == 1.0
    assert row["warnings/no_departed_vehicles"] is False
    assert row["warnings/no_arrived_vehicles"] is False
    assert row["warnings/eval_episodes_too_low"] is True


def test_safe_scalar_accepts_scalar_values_and_skips_non_scalars() -> None:
    assert safe_scalar(3) == 3.0
    assert safe_scalar(2.5) == 2.5
    assert safe_scalar("nope") is None
    assert safe_scalar({"a": 1}) is None
    assert safe_scalar([1, 2]) is None


def test_sb3_callback_saves_periodic_and_final_checkpoints(monkeypatch) -> None:
    stable_baselines3 = types.ModuleType("stable_baselines3")
    stable_baselines3_common = types.ModuleType("stable_baselines3.common")
    stable_baselines3_common_callbacks = types.ModuleType("stable_baselines3.common.callbacks")
    stable_baselines3_common_evaluation = types.ModuleType("stable_baselines3.common.evaluation")

    class BaseCallback:
        pass

    def evaluate_policy(*args, **kwargs):
        return 0.0, 0.0

    stable_baselines3_common_callbacks.BaseCallback = BaseCallback
    stable_baselines3_common_evaluation.evaluate_policy = evaluate_policy
    sb3_wrappers = types.ModuleType("sumo_rl.agents.sb3.wrappers")
    sb3_wrappers._resolve_base_env = lambda env: env
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", stable_baselines3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", stable_baselines3_common_callbacks)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.evaluation", stable_baselines3_common_evaluation)
    monkeypatch.setitem(sys.modules, "sumo_rl.agents.sb3.wrappers", sb3_wrappers)

    class DummyLogger:
        def __init__(self) -> None:
            self.name_to_value = {"train/loss": 1.23}

    class DummyModel:
        def __init__(self) -> None:
            self.logger = DummyLogger()
            self.num_timesteps = 1
            self.saved_paths = []

        def save(self, path: str) -> None:
            self.saved_paths.append(path)

    class DummyLogSink:
        def __init__(self) -> None:
            self.rows = []

        def log(self, metrics, step=None) -> None:
            self.rows.append((dict(metrics), step))

    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        callback = _CALLBACKS_MODULE.SB3WandbCallback(
            DummyLogSink(),
            DummyLogSink(),
            logging_cfg=types.SimpleNamespace(
                log_sb3_internal_metrics=True,
                log_sac_diagnostics=False,
                log_traffic_metrics_during_training=False,
                save_checkpoints=True,
                save_final_model=True,
            ),
            log_freq=1,
            eval_env=None,
            eval_episodes=0,
            checkpoint_dir=Path(tmp_dir) / "checkpoints",
            checkpoint_freq=1,
            save_checkpoints=True,
            save_final_model=True,
        ).build()

        dummy_model = DummyModel()
        callback.model = dummy_model
        callback.num_timesteps = 1
        callback.locals = {}

        callback._on_step()
        callback._on_training_end()

        saved_paths = [Path(path) for path in dummy_model.saved_paths]
        assert any("checkpoint_step_000000001" in path.name for path in saved_paths)
        assert any("final_model" in path.name for path in saved_paths)


def test_sb3_final_log_step_uses_actual_timesteps_when_rollout_overshoots() -> None:
    cfg = types.SimpleNamespace(experiment=types.SimpleNamespace(total_timesteps=200))
    model = types.SimpleNamespace(num_timesteps=512)

    assert _get_sb3_final_log_step(cfg, model) == 512
