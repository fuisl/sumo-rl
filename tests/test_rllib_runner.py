import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments.rllib_runner import _build_policy_mapping, _policy_id_for_agent
from sumo_rl.agents.rllib_common import (
    completed_training_episodes,
    emit_training_metrics_by_step,
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


def test_rllib_training_budget_uses_experiment_episodes():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, total_timesteps=20))

    assert training_episode_target(cfg) == 3
    assert training_should_stop({"train/episodes_total": 2.0, "train/env_steps_sampled": 40.0}, cfg) is False
    assert training_should_stop({"train/episodes_total": 3.0, "train/env_steps_sampled": 60.0}, cfg) is True


def test_rllib_training_budget_falls_back_to_completed_horizons_not_iterations():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, total_timesteps=20))

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
