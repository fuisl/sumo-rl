import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner
from sumo_rl.experiments.rllib_runner import _build_policy_mapping, _policy_id_for_agent
from sumo_rl.agents.dqn.dqn import build_replay_buffer_config
from sumo_rl.agents.rllib_common import (
    apply_standard_evaluation_settings,
    episode_steps,
    build_training_episode_row,
    completed_training_episodes,
    emit_validation_if_due,
    emit_training_episode_rows,
    emit_training_metrics_by_step,
    should_log_training_episode,
    should_log_training_metrics,
    train_log_freq_steps,
    training_episode_target,
    training_should_stop,
    validation_interval_steps,
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


def test_dqn_uses_multi_agent_episode_replay_buffer_by_default():
    replay_config = build_replay_buffer_config({})

    assert replay_config["type"] == "MultiAgentPrioritizedEpisodeReplayBuffer"
    assert replay_config["capacity"] == 50000
    assert replay_config["alpha"] == 0.6
    assert replay_config["beta"] == 0.4


def test_dqn_replay_buffer_config_is_customizable():
    replay_config = build_replay_buffer_config(
        {
            "replay_buffer_type": "MultiAgentEpisodeReplayBuffer",
            "replay_buffer_capacity": 123,
        }
    )

    assert replay_config == {"type": "MultiAgentEpisodeReplayBuffer", "capacity": 123}


def test_evaluate_closes_env_before_building_final_summary(monkeypatch, tmp_path):
    class DummyEvalEnv:
        possible_agents = ["tls_1"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    eval_env = DummyEvalEnv()

    def fake_build_rllib_parallel_env(*args, **kwargs):
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

    monkeypatch.setattr(rllib_runner, "build_rllib_parallel_env", fake_build_rllib_parallel_env)
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
        algorithm_kind="ppo",
        logging_cfg=SimpleNamespace(log_final_traffic_metrics=True),
    )

    assert eval_env.closed is True
    assert summary["algorithm/kind"] == "ppo"
    assert summary["final/eval/mean_reward"] == 12.5
    assert summary["final/resco/avg_delay"] == 3.0
    assert summary["final/resco/avg_delay_std"] == 0.8
    assert summary["final/resco/wait_std"] == 0.4


def test_training_episode_row_uses_episode_cadence_and_resco_metrics():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_episodes=2, train_log_freq_steps=1, log_freq=1000))
    metrics = {
        "algorithm/kind": "ppo",
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
        "tripinfo/finished_count": 4.0,
        "tripinfo/parse_success": 1.0,
    }

    assert should_log_training_episode(1, cfg, last_logged_episode=0) is False
    assert should_log_training_episode(2, cfg, last_logged_episode=0) is True

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo")

    assert row["train/episode_index"] == 2.0
    assert row["train/episode_summary_available"] == 1.0
    assert row["train/reward_mean"] == 4.5
    assert row["train/episode_reward"] == 4.5
    assert row["train/resco/avg_delay"] == 12.0
    assert row["train/resco/wait"] == 7.0
    assert row["train/resco/queue"] == 3.0
    assert row["train/resco/trip_time"] == 33.0
    assert row["train/resco/max_queue"] == 9.0
    assert row["train/resco/avg_delay_std"] == 1.5
    assert row["train/resco/wait_std"] == 0.5
    assert row["train/tripinfo/finished_count"] == 4.0
    assert row["train/tripinfo/parse_success"] == 1.0


def test_rllib_training_episode_emission_logs_every_summary_episode():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000))
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [
            {"episode/index": 1.0, "resco_wait": 5.0},
            {"episode/index": 2.0, "resco_wait": 6.0},
        ],
        cfg,
        algorithm_kind="ppo",
        last_logged_episode=0,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 2
    assert [step for step, _ in emitted] == [1, 2]
    assert emitted[0][1]["train/resco/wait"] == 5.0
    assert emitted[1][1]["train/resco/wait"] == 6.0
    assert emitted[0][1]["train/episode_summary_available"] == 1.0


def test_rllib_training_episode_emission_falls_back_to_completed_episode_counters():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episodes=3, episode_seconds=100),
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000),
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/env_steps_sampled": 60.0,
        "train/iteration": 7,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [],
        cfg,
        algorithm_kind="ppo",
        last_logged_episode=0,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 3
    assert [step for step, _ in emitted] == [1, 2, 3]
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert all(row["train/episode_summary_available"] == 0.0 for _, row in emitted)
    assert all(row["train/episode_reward"] == 4.5 for _, row in emitted)


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


def test_validation_interval_prefers_experiment_override_over_logging_eval_freq():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(validation_interval_steps=25),
        logging=SimpleNamespace(eval_freq=5000),
    )

    assert validation_interval_steps(cfg) == 25


def test_validation_interval_falls_back_to_logging_eval_freq():
    cfg = SimpleNamespace(experiment=SimpleNamespace(), logging=SimpleNamespace(eval_freq=5000))

    assert validation_interval_steps(cfg) == 5000


def test_rllib_training_loop_emits_validation_when_due():
    cfg = SimpleNamespace(experiment=SimpleNamespace(), logging=SimpleNamespace(eval_freq=10))
    emitted = []

    last_step = emit_validation_if_due(
        {"train/env_step": 9.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append((step, metrics["train/env_step"])),
    )
    assert last_step == 0
    assert emitted == []

    last_step = emit_validation_if_due(
        {"train/env_step": 10.0},
        cfg,
        last_validation_step=last_step,
        validate=lambda metrics, step: emitted.append((step, metrics["train/env_step"])),
    )

    assert last_step == 10
    assert emitted == [(10, 10.0)]


def test_validation_summary_row_maps_final_metrics_to_validation_namespace():
    row = rllib_runner._validation_summary_row(
        {
            "algorithm/kind": "ppo",
            "final/eval/mean_reward": 12.0,
            "final/resco/avg_delay": 4.0,
            "final/efficiency/throughput": 8.0,
            "final/safety/collisions": 0.0,
            "tripinfo/parse_success": 1.0,
            "warnings/missing_tripinfo": 0.0,
            "eval/episode": 2.0,
        },
        step=100,
    )

    assert row["algorithm/kind"] == "ppo"
    assert row["validation/env_step"] == 100.0
    assert row["validation/eval/mean_reward"] == 12.0
    assert row["validation/resco/avg_delay"] == 4.0
    assert row["validation/efficiency/throughput"] == 8.0
    assert row["validation/safety/collisions"] == 0.0
    assert row["validation/tripinfo/parse_success"] == 1.0
    assert row["validation/warnings/missing_tripinfo"] == 0.0
    assert row["validation/eval/episode"] == 2.0


def test_standard_evaluation_settings_use_rllib_algorithm_config_api():
    class DummyConfig:
        def __init__(self):
            self.kwargs = None

        def evaluation(self, **kwargs):
            self.kwargs = kwargs
            return self

    config = DummyConfig()
    returned = apply_standard_evaluation_settings(
        config,
        {
            "evaluation_interval": 3,
            "evaluation_duration": 2,
            "evaluation_duration_unit": "episodes",
            "evaluation_config": {"explore": False},
            "evaluation_parallel_to_training": True,
        },
    )

    assert returned is config
    assert config.kwargs == {
        "evaluation_interval": 3,
        "evaluation_duration": 2,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {"explore": False},
        "evaluation_parallel_to_training": True,
    }
