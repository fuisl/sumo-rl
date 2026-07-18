# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.dqn.dqn import build_replay_buffer_config
from sumo_rl.agents.rllib_common import (
    apply_standard_evaluation_settings,
    build_training_episode_row,
    completed_training_episodes,
    emit_training_metrics_by_step,
    emit_validation_if_due,
    episode_steps,
    should_log_training_metrics,
    trace_mode,
    train_log_freq_steps,
    training_episode_target,
    training_should_stop,
    validation_interval_episodes,
    validation_interval_steps,
)
from sumo_rl.experiments import rllib_runner


def test_rllib_run_name_uses_logging_name_when_set():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name="wandb-title"),
        experiment=SimpleNamespace(name="experiment-title"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo") == "wandb-title"


def test_rllib_run_name_uses_explicit_experiment_name():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name=None),
        experiment=SimpleNamespace(name="experiment-title"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo") == "experiment-title"


def test_rllib_run_name_keeps_generated_name_for_default_experiment_name():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name=None),
        experiment=SimpleNamespace(name="rllib"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo").startswith("grid4x4__ppo__")


def test_rllib_runtime_params_reads_hydra_resources_before_algorithm_params():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "resources": {
                "ray_address": "auto",
                "ray_num_cpus": 7,
                "native_num_threads": 1,
            },
            "algorithm": {
                "params": {
                    "ray_num_cpus": 3,
                    "num_env_runners": 1,
                },
            },
        }
    )

    params = rllib_runner._rllib_runtime_params(cfg)

    assert params["ray_address"] == "auto"
    assert params["ray_num_cpus"] == 3
    assert params["native_num_threads"] == 1
    assert params["num_env_runners"] == 1


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


def test_trace_mode_defaults_to_training():
    cfg = SimpleNamespace(logging=SimpleNamespace())

    assert trace_mode(cfg) == "training"


def test_debug_trace_mode_moves_internal_metrics_under_debug_namespace():
    cfg = SimpleNamespace(logging=SimpleNamespace(trace_mode="debug"))
    metrics = {
        "train/env_step": 25.0,
        "train/episodes_total": 2.0,
        "train/env_steps_sampled": 25.0,
        "train/episode_return_mean": 4.5,
        "train/episode_return_min": 3.0,
        "train/episode_return_max": 6.0,
        "train/episode_len_mean": 12.0,
        "train/rllib/training_iteration": 3.0,
        "train/rllib/time_total_s": 15.0,
        "train/ppo/learners/default_policy/loss": 1.25,
        "train/ppo/entropy_mean": 0.33,
    }
    episode_summary = {
        "episode/index": 2.0,
        "reward/agent/tls_1": 2.0,
        "system_total_arrived": 8.0,
        "system_total_running": 5.0,
    }

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo", cfg=cfg)

    assert row["train/efficiency_total_arrived"] == 8.0
    assert row["debug/env_episode_index"] == 2.0
    assert row["debug/reward/tls_1"] == 2.0
    assert row["debug/efficiency_total_running"] == 5.0
    assert row["debug/episode_return_mean"] == 4.5
    assert row["debug/episode_return_min"] == 3.0
    assert row["debug/episode_return_max"] == 6.0
    assert row["debug/episode_len_mean"] == 12.0
    assert row["debug/rllib/training_iteration"] == 3.0
    assert row["debug/rllib/time_total_s"] == 15.0
    assert row["debug/ppo/learners/default_policy/loss"] == 1.25
    assert row["debug/ppo/entropy_mean"] == 0.33
    assert "train/episode_return_mean" not in row


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


def test_validation_interval_episodes_is_explicit_episode_cadence():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(validation_interval_episodes=5, validation_interval_steps=25),
        logging=SimpleNamespace(eval_freq=5000),
    )

    assert validation_interval_episodes(cfg) == 5


def test_validation_interval_falls_back_to_logging_eval_freq():
    cfg = SimpleNamespace(experiment=SimpleNamespace(), logging=SimpleNamespace(eval_freq=5000))

    assert validation_interval_steps(cfg) == 5000


def test_rllib_training_loop_emits_step_validation_when_due():
    cfg = SimpleNamespace(experiment=SimpleNamespace(validation_interval_episodes=None), logging=SimpleNamespace(eval_freq=10))
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


def test_rllib_training_loop_prefers_episode_validation_cadence():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            validation_interval_episodes=5,
            episodes=20,
            episode_seconds=100,
        ),
        logging=SimpleNamespace(eval_freq=10),
    )
    emitted = []

    last_progress = emit_validation_if_due(
        {"train/env_step": 80.0, "train/episodes_total": 4.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append((step, metrics["train/episodes_total"])),
    )
    assert last_progress == 0
    assert emitted == []

    last_progress = emit_validation_if_due(
        {"train/env_step": 100.0, "train/episodes_total": 5.0},
        cfg,
        last_validation_step=last_progress,
        validate=lambda metrics, step: emitted.append((step, metrics["train/episodes_total"])),
    )

    assert last_progress == 5
    assert emitted == [(100, 5.0)]


def test_episode_validation_cadence_uses_derived_env_step_when_dqn_result_has_only_episode_count():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            validation_interval_episodes=5,
            episodes=20,
            episode_seconds=100,
        ),
        logging=SimpleNamespace(eval_freq=5000),
    )
    emitted = []

    last_progress = emit_validation_if_due(
        {"train/episodes_total": 5.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append(step),
    )

    assert last_progress == 5
    assert emitted == [100]


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
