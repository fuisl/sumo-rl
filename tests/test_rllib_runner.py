import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner
from sumo_rl.experiments.rllib_runner import _build_policy_mapping, _policy_id_for_agent
from sumo_rl.agents.rllib_common import (
    episode_steps,
    build_training_episode_row,
    completed_training_episodes,
    emit_training_metrics_by_step,
    should_log_training_episode,
    should_log_training_metrics,
    train_log_freq_steps,
    training_episode_target,
    training_should_stop,
)


def test_policy_id_for_agent_shared_mode_uses_shared_policy_name():
    assert _policy_id_for_agent("tls_1", "shared") == "shared_policy"


def test_policy_id_for_agent_independent_mode_uses_agent_id():
    assert _policy_id_for_agent("tls_1", "independent") == "tls_1"


def test_build_policy_mapping_shared_mode_maps_all_agents_to_one_policy():
    mapping_fn = _build_policy_mapping("shared")
    assert mapping_fn("tls_1") == "shared_policy"
    assert mapping_fn("tls_2") == "shared_policy"


def test_build_policy_mapping_independent_mode_keeps_agent_identity():
    mapping_fn = _build_policy_mapping("independent")
    assert mapping_fn("tls_1") == "tls_1"
    assert mapping_fn("tls_2") == "tls_2"


def test_evaluate_closes_env_before_building_final_summary(monkeypatch, tmp_path):
    class DummyEvalEnv:
        possible_agents = ["tls_1"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    eval_env = DummyEvalEnv()

    def fake_build_multi_agent_wrapper(*args, **kwargs):
        del args, kwargs
        return eval_env

    def fake_run_episode(*args, **kwargs):
        del args, kwargs
        return 12.5

    def fake_build_summary(env, **kwargs):
        assert env.closed is True
        return {
            "algorithm/kind": kwargs["algorithm_kind"],
            "final/eval/mean_reward": kwargs["eval_mean_reward"],
            "final/eval/std_reward": kwargs["eval_std_reward"],
            "final/resco/avg_delay": 3.0,
            "final/resco/avg_delay_std": 0.8,
            "final/resco/wait_std": 0.4,
        }

    monkeypatch.setattr(rllib_runner, "build_multi_agent_wrapper", fake_build_multi_agent_wrapper)
    monkeypatch.setattr(rllib_runner, "_run_multi_agent_episode", fake_run_episode)
    monkeypatch.setattr(rllib_runner, "_build_final_eval_summary_row", fake_build_summary)

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(seed=7, eval_episodes=1, eval_seeds=None),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )

    summary = rllib_runner._evaluate(
        cfg,
        tmp_path,
        algo=object(),
        algorithm_kind="ppo_rllib",
        logging_cfg=SimpleNamespace(log_final_traffic_metrics=True),
    )

    assert eval_env.closed is True
    assert summary["algorithm/kind"] == "ppo_rllib"
    assert summary["final/eval/mean_reward"] == 12.5
    assert summary["final/resco/avg_delay"] == 3.0
    assert summary["final/resco/avg_delay_std"] == 0.8
    assert summary["final/resco/wait_std"] == 0.4


def test_training_episode_row_uses_episode_cadence_and_resco_metrics():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_episodes=2, train_log_freq_steps=1, log_freq=1000))
    metrics = {
        "algorithm/kind": "ppo_rllib",
        "train/episode_return_mean": 4.5,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
    }
    episode_summary = {
        "episode/index": 2.0,
        "resco_avg_delay": 12.0,
        "resco_wait": 7.0,
        "resco_queue": 3.0,
        "resco_trip_time": 33.0,
        "resco_max_queue": 9.0,
        "resco_avg_delay_std": 1.5,
        "resco_wait_std": 0.5,
    }

    assert should_log_training_episode(1, cfg, last_logged_episode=0) is False
    assert should_log_training_episode(2, cfg, last_logged_episode=0) is True

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo_rllib")

    assert row["train/episode_index"] == 2.0
    assert row["train/reward_mean"] == 4.5
    assert row["train/episode_reward"] == 4.5
    assert row["train/resco/avg_delay"] == 12.0
    assert row["train/resco/wait"] == 7.0
    assert row["train/resco/queue"] == 3.0
    assert row["train/resco/trip_time"] == 33.0
    assert row["train/resco/max_queue"] == 9.0
    assert row["train/resco/avg_delay_std"] == 1.5
    assert row["train/resco/wait_std"] == 0.5


def test_rllib_training_budget_uses_experiment_episodes():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, episode_seconds=100))

    assert training_episode_target(cfg) == 3
    assert episode_steps(cfg) == 20
    assert training_should_stop({"train/episodes_total": 2.0, "train/env_steps_sampled": 40.0}, cfg) is False
    assert training_should_stop({"train/episodes_total": 3.0, "train/env_steps_sampled": 60.0}, cfg) is True


def test_rllib_training_budget_falls_back_to_completed_horizons_not_iterations():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, episode_seconds=100))

    assert completed_training_episodes({"train/env_steps_sampled": 40.0}, cfg) == 2
    assert training_should_stop({"train/env_steps_sampled": 40.0}, cfg) is False
    assert training_should_stop({"train/env_steps_sampled": 60.0}, cfg) is True


def test_rllib_training_log_frequency_uses_sampled_steps():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_steps=25, log_freq=1000))

    assert train_log_freq_steps(cfg) == 25
    assert should_log_training_metrics({"train/env_steps_sampled": 20.0}, cfg, last_logged_step=0) is False
    assert should_log_training_metrics({"train/env_steps_sampled": 25.0}, cfg, last_logged_step=0) is True


def test_rllib_training_metrics_can_emit_every_sampled_step():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_steps=1, log_freq=1000))
    emitted = []

    last_step = emit_training_metrics_by_step(
        {"train/env_steps_sampled": 3.0, "train/iteration": 1},
        cfg,
        last_logged_step=0,
        emit_metrics=lambda row, step: emitted.append((step, row["train/env_step"])),
    )

    assert last_step == 3
    assert emitted == [(1, 1.0), (2, 2.0), (3, 3.0)]
