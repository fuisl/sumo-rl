from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

import sumo_rl
from sumo_rl.agents.sac.sac import build_config, build_replay_buffer_config
from tests._support.envs import DummyDiscreteParallelEnv as _DummyDiscreteParallelEnv


def test_sac_uses_multi_agent_episode_replay_buffer_by_default():
    replay_config = build_replay_buffer_config({})

    assert replay_config["type"] == "MultiAgentPrioritizedEpisodeReplayBuffer"
    assert replay_config["capacity"] == int(1e6)
    assert replay_config["alpha"] == 0.6
    assert replay_config["beta"] == 0.4


def test_sac_replay_buffer_config_is_customizable():
    replay_config = build_replay_buffer_config(
        {
            "replay_buffer_type": "MultiAgentEpisodeReplayBuffer",
            "replay_buffer_capacity": 1234,
        }
    )

    assert replay_config == {
        "type": "MultiAgentEpisodeReplayBuffer",
        "capacity": 1234,
    }


def test_custom_sac_build_config_installs_project_owned_multi_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDiscreteParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_mlp_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
                "model_config": {
                    "architecture_tag": "custom_test",
                    "communication": {"enabled": True, "type": "message_passing"},
                },
            }
        ),
    )

    config = build_config(cfg, tmp_path, algorithm_kind="sac_mlp")

    assert config.rl_module_spec.multi_rl_module_class.__name__ == "CustomSACMultiRLModule"
    assert set(config.rl_module_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    for spec in config.rl_module_spec.rl_module_specs.values():
        assert spec.module_class.__name__ == "CustomSACTorchRLModule"
        assert spec.model_config["architecture_tag"] == "custom_test"


def test_builtin_sac_build_config_uses_default_module_spec(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDiscreteParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_builtin_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
            }
        ),
    )

    config = build_config(cfg, tmp_path, algorithm_kind="sac_builtin")

    assert config.rl_module_spec.module_class.__name__ == "DefaultSACTorchRLModule"
