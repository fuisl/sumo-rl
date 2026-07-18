# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.ppo.ppo import extract_training_metrics as extract_ppo_training_metrics
from sumo_rl.agents.rllib_common import (
    _completed_episode_summary_history,
    build_training_episode_row,
    completed_training_episodes,
    emit_training_episode_rows,
    should_log_training_episode,
)
from sumo_rl.agents.sac.sac import extract_training_metrics as extract_sac_training_metrics
from sumo_rl.experiments.runner import _log_outputs


def test_training_episode_row_uses_episode_cadence_and_resco_metrics():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(train_log_freq_episodes=2, train_log_freq_steps=1, log_freq=1000, trace_mode="training")
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/env_step": 40.0,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
    }
    episode_summary = {
        "episode/index": 2.0,
        "reward/mean": 4.5,
        "reward/max": 6.0,
        "reward/std": 1.5,
        "reward/agent/tls_1": 3.0,
        "reward/agent/tls_2": 6.0,
        "resco_delay_mean": 12.0,
        "resco_delay_max": 14.0,
        "resco_delay_std": 1.5,
        "resco_wait_mean": 7.0,
        "resco_wait_max": 9.0,
        "resco_wait_std": 0.5,
        "resco_queue_mean": 3.0,
        "resco_queue_max": 9.0,
        "resco_trip_time_mean": 33.0,
        "resco_tripinfo_count": 4.0,
        "system_total_arrived": 11.0,
        "system_total_departed": 12.0,
        "system_total_teleported": 1.0,
        "system_total_running": 8.0,
        "system_mean_queued": 2.0,
    }

    assert should_log_training_episode(1, cfg, last_logged_episode=0) is False
    assert should_log_training_episode(2, cfg, last_logged_episode=0) is True

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo", cfg=cfg)

    assert row["train/rollout_index"] == 2.0
    assert row["train/episode_index"] == 2.0
    assert row["train/env_step"] == 40.0
    assert row["train/reward_mean"] == 4.5
    assert row["train/reward_max"] == 6.0
    assert row["train/reward_std"] == 1.5
    assert row["train/resco_delay_mean"] == 12.0
    assert row["train/resco_delay_max"] == 14.0
    assert row["train/resco_delay_std"] == 1.5
    assert row["train/resco_wait_mean"] == 7.0
    assert row["train/resco_wait_max"] == 9.0
    assert row["train/resco_wait_std"] == 0.5
    assert row["train/resco_queue_mean"] == 3.0
    assert row["train/resco_queue_max"] == 9.0
    assert row["train/resco_trip_time_mean"] == 33.0
    assert row["train/resco_tripinfo_count"] == 4.0
    assert row["train/efficiency_total_arrived"] == 11.0
    assert row["train/efficiency_total_departed"] == 12.0
    assert row["train/safety_total_teleported"] == 1.0
    assert "train/efficiency_total_running" not in row
    assert "train/efficiency_mean_queued" not in row
    assert row["debug/efficiency_total_running"] == 8.0
    assert row["debug/reward/tls_1"] == 3.0
    assert row["debug/reward/tls_2"] == 6.0
    assert "debug/episode_return_mean" not in row


def test_rllib_training_episode_emission_logs_every_summary_episode():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="training")
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
        "train/rllib/rollout_jump": 2.0,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [
            {"episode/index": 1.0, "resco_wait_mean": 5.0},
            {"episode/index": 2.0, "resco_wait_mean": 6.0},
        ],
        cfg,
        algorithm_kind="ppo",
        last_logged_episode=0,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 2
    assert [step for step, _ in emitted] == [1, 2]
    assert [row["train/rollout_index"] for _, row in emitted] == [1.0, 2.0]
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0]
    assert all(row["train/rllib/rollout_jump"] == 2.0 for _, row in emitted)
    assert emitted[0][1]["train/resco_wait_mean"] == 5.0
    assert emitted[1][1]["train/resco_wait_mean"] == 6.0


def test_rllib_training_episode_emission_clamps_env_local_index_to_rllib_rollout_count():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episodes=20, episode_seconds=100),
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="debug"),
    )
    metrics = {
        "algorithm/kind": "dqn",
        "train/episodes_total": 13.0,
        "train/env_step": 130.0,
        "train/dqn/replay/num_added": 999.0,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [{"episode/index": 30.0, "resco_wait_mean": 9.0}],
        cfg,
        algorithm_kind="dqn",
        last_logged_episode=12,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 13
    assert [step for step, _ in emitted] == [13]
    assert emitted[0][1]["train/rollout_index"] == 13.0
    assert emitted[0][1]["train/episode_index"] == 13.0
    assert emitted[0][1]["debug/env_episode_index"] == 30.0
    assert emitted[0][1]["train/resco_wait_mean"] == 9.0


def test_reset_only_episode_summaries_are_not_logged_as_zero_metrics():
    class DummyEnv:
        completed_episode_summaries = [
            {
                "episode/index": 1.0,
                "episode/elapsed_seconds": 0.0,
                "resco_wait_mean": 0.0,
                "resco_queue_mean": 0.0,
                "tripinfo/parse_pending": 0.0,
            },
            {
                "episode/index": 2.0,
                "episode/elapsed_seconds": 3600.0,
                "resco_wait_mean": 6.0,
                "resco_queue_mean": 2.5,
                "tripinfo/parse_pending": 0.0,
            },
        ]
        last_episode_summary = {}
        sumo = None

    summaries = _completed_episode_summary_history(DummyEnv())

    assert [summary["episode/index"] for summary in summaries] == [2.0]


def test_rllib_training_episode_emission_falls_back_to_completed_episode_counters():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episodes=3, episode_seconds=100),
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="training"),
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
    assert [row["train/rollout_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert all(row["train/env_step"] == 60.0 for _, row in emitted)
    assert all("train/reward_mean" not in row for _, row in emitted)


def test_completed_training_episodes_ignores_off_policy_replay_activity_for_rollout_count():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episode_seconds=100))
    metrics = {
        "train/episodes_total": 2.0,
        "train/env_steps_sampled": 500.0,
        "train/dqn/replay/num_added": 5000.0,
        "train/dqn/replay/num_sampled": 9000.0,
    }

    assert completed_training_episodes(metrics, cfg) == 2


def test_ppo_extract_training_metrics_adds_entropy_mean():
    metrics = extract_ppo_training_metrics(
        {
            "env_runners": {"num_episodes_lifetime": 1.0},
            "learners": {"default_policy": {"curr_entropy": 0.42, "loss": 1.0}},
        },
        iteration=1,
    )

    assert metrics["train/ppo/entropy_mean"] == 0.42


def test_sac_extract_training_metrics_adds_entropy_mean():
    metrics = extract_sac_training_metrics(
        {
            "env_runners": {"num_episodes_lifetime": 1.0},
            "learners": {"default_policy": {"entropy_mean": 0.18, "critic_loss": 2.0}},
        },
        iteration=1,
        algorithm_kind="sac_builtin",
    )

    assert metrics["train/sac/entropy_mean"] == 0.18


def test_log_outputs_lets_wandb_custom_step_axes_control_train_and_validation_steps():
    class DummyWandbRun:
        def __init__(self):
            self.calls = []

        def log(self, metrics, step=None):
            self.calls.append((metrics, step))

    wandb_run = DummyWandbRun()

    _log_outputs(wandb_run, None, {"train/env_step": 40320.0, "train/episode_index": 62.0}, step=62)
    _log_outputs(
        wandb_run,
        None,
        {"validation/env_step": 45360.0, "validation/episode_index": 70.0, "validation/reward_mean": 1.0},
        step=45360,
    )

    assert wandb_run.calls == [
        ({"train/env_step": 40320.0, "train/episode_index": 62.0}, None),
        ({"validation/env_step": 45360.0, "validation/episode_index": 70.0, "validation/reward_mean": 1.0}, None),
    ]
