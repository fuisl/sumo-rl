from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl
from sumo_rl.agents.rllib_common import build_algorithm_context
from sumo_rl.agents.sac.custom_sac import build_custom_sac_module_spec, normalize_custom_sac_model_config
from sumo_rl.agents.sac.sac import build_config, build_replay_buffer_config


class _DummyDiscreteParallelEnv:
    possible_agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return Discrete(3)

    def close(self):
        pass


class _FakeGraphTrafficSignal:
    def __init__(self, ts_id, lanes, out_lanes, density, queue):
        self.id = ts_id
        self.lanes = list(lanes)
        self.out_lanes = list(out_lanes)
        self._density = list(density)
        self._queue = list(queue)

    def get_lanes_density(self):
        return self._density

    def get_lanes_queue(self):
        return self._queue


class _DummyGraphParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        signals = [
            _FakeGraphTrafficSignal("tls_0", ["in_0"], ["lane_0_1"], [0.25], [0.5]),
            _FakeGraphTrafficSignal("tls_1", ["lane_0_1", "in_1"], ["out_1"], [0.75, 0.1], [0.2, 0.3]),
        ]
        self.ts_ids = [signal.id for signal in signals]
        self.traffic_signals = {signal.id: signal for signal in signals}

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return Discrete(3)

    def close(self):
        pass


def test_sac_algorithm_context_uses_discrete_action_spaces(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDiscreteParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_discrete_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )

    context = build_algorithm_context(cfg, tmp_path, "sac_builtin")

    assert context.policy_mode == "independent"
    assert set(context.active_policies.keys()) == {"tls_0", "tls_1"}
    for policy_spec in context.active_policies.values():
        assert isinstance(policy_spec.action_space, Discrete)
        assert policy_spec.action_space.n == 3


def test_custom_sac_module_spec_keeps_discrete_action_space():
    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)

    spec = build_custom_sac_module_spec(obs_space, action_space)

    assert spec.observation_space == obs_space
    assert spec.action_space == action_space
    assert spec.model_config["architecture_tag"] == "sac_mlp"
    assert spec.model_config["twin_q"] is True
    assert spec.model_config["custom_sac"]["critic"]["twin_q"] is True


def test_custom_sac_default_architecture_matches_builtin_sac_rlmodule_defaults():
    from ray.rllib.algorithms.sac.sac_catalog import SACCatalog

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)
    builtin_catalog = SACCatalog(obs_space, action_space, {"twin_q": True})
    custom_config = normalize_custom_sac_model_config({})

    assert custom_config["fcnet_hiddens"] == builtin_catalog._model_config_dict["fcnet_hiddens"]
    assert custom_config["fcnet_activation"] == builtin_catalog._model_config_dict["fcnet_activation"]
    assert custom_config["head_fcnet_hiddens"] == builtin_catalog._model_config_dict["head_fcnet_hiddens"]
    assert custom_config["head_fcnet_activation"] == builtin_catalog._model_config_dict["head_fcnet_activation"]
    assert custom_config["critic_fcnet_hiddens"] == builtin_catalog._model_config_dict["fcnet_hiddens"]
    assert custom_config["critic_fcnet_activation"] == builtin_catalog._model_config_dict["fcnet_activation"]
    assert custom_config["critic_head_fcnet_hiddens"] == builtin_catalog._model_config_dict["head_fcnet_hiddens"]
    assert custom_config["critic_head_fcnet_activation"] == builtin_catalog._model_config_dict["head_fcnet_activation"]


def test_custom_sac_module_spec_builds_and_exposes_actor_critic_hooks():
    torch = pytest.importorskip("torch")
    from ray.rllib.core.columns import Columns

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)
    spec = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "actor_small",
            "actor": {"encoder": {"hidden_dims": [32], "activation": "relu"}},
            "critic": {"encoder": {"hidden_dims": [64], "activation": "relu"}},
        },
    )

    module = spec.build()
    output = module.forward_inference({Columns.OBS: torch.zeros(2, 4)})

    assert module._architecture_tag == "actor_small"
    assert module._communication_enabled is False
    assert module.catalog.__class__.__name__ == "CustomSACCatalog"
    assert module.catalog.latent_dims == (32,)
    assert module.catalog.qf_latent_dims == [64]
    assert output[Columns.ACTION_DIST_INPUTS].shape == (2, 4)
    assert spec.model_config["fcnet_hiddens"] == [32]
    assert spec.model_config["critic_fcnet_hiddens"] == [64]


def test_custom_sac_model_config_accepts_message_passing_placeholder():
    model_config = normalize_custom_sac_model_config(
        {
            "architecture_tag": "sac_dcrnn_actor",
            "actor": {"encoder": {"hidden_dims": [128]}},
            "communication": {
                "enabled": True,
                "type": "gat",
                "apply_to": ["actor"],
                "scope": "multi_agent",
            },
        }
    )

    assert model_config["architecture_tag"] == "sac_dcrnn_actor"
    assert model_config["custom_sac"]["communication"]["enabled"] is True
    assert model_config["custom_sac"]["communication"]["type"] == "gat"
    assert model_config["custom_sac"]["communication"]["apply_to"] == ["actor"]
    assert model_config["fcnet_hiddens"] == [128]


def test_custom_sac_model_config_accepts_dcrnn_actor_encoder():
    model_config = normalize_custom_sac_model_config(
        {
            "architecture_tag": "sac_dcrnn_actor",
            "actor": {
                "encoder": {
                    "type": "dcrnn",
                    "hidden_dim": 32,
                    "max_diffusion_step": 1,
                    "num_rnn_layers": 2,
                },
                "head": {"hidden_dims": [16], "activation": "relu"},
            },
        }
    )

    assert model_config["architecture_tag"] == "sac_dcrnn_actor"
    assert model_config["custom_sac"]["actor"]["encoder"]["type"] == "dcrnn"
    assert model_config["custom_sac"]["actor"]["encoder"]["hidden_dim"] == 32
    assert model_config["fcnet_hiddens"] == []


def test_custom_sac_model_config_rejects_dcrnn_critic_encoder():
    with pytest.raises(ValueError, match="critic.encoder.type=mlp"):
        normalize_custom_sac_model_config(
            {
                "critic": {
                    "encoder": {
                        "type": "dcrnn",
                    }
                }
            }
        )


def test_custom_sac_forward_train_exposes_actor_twin_critic_outputs():
    torch = pytest.importorskip("torch")
    from ray.rllib.algorithms.sac.sac_learner import (
        ACTION_LOG_PROBS,
        ACTION_PROBS,
        QF_PREDS,
        QF_TARGET_NEXT,
        QF_TWIN_PREDS,
    )
    from ray.rllib.core.columns import Columns

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(3)
    module = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "actor": {"encoder": {"hidden_dims": [16]}},
            "critic": {
                "encoder": {"hidden_dims": [32]},
                "head": {"hidden_dims": [8]},
            },
            "communication": {
                "enabled": True,
                "type": "gat",
                "apply_to": ["actor", "critic"],
            },
        },
    ).build()
    module.make_target_networks()

    output = module.forward_train(
        {
            Columns.OBS: torch.zeros(5, 4),
            Columns.NEXT_OBS: torch.ones(5, 4),
        }
    )

    assert output[ACTION_PROBS].shape == (5, 3)
    assert output[ACTION_LOG_PROBS].shape == (5, 3)
    assert output[QF_PREDS].shape == (5, 3)
    assert output[QF_TWIN_PREDS].shape == (5, 3)
    assert output[QF_TARGET_NEXT].shape == (5, 3)


def test_custom_sac_dcrnn_actor_forward_train_exposes_actor_twin_critic_outputs():
    torch = pytest.importorskip("torch")
    from ray.rllib.algorithms.sac.sac_learner import (
        ACTION_LOG_PROBS,
        ACTION_PROBS,
        QF_PREDS,
        QF_TARGET_NEXT,
        QF_TWIN_PREDS,
    )
    from ray.rllib.core.columns import Columns

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)
    action_space = Discrete(3)
    module = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "sac_dcrnn_actor",
            "agent_index": 1,
            "num_nodes": 4,
            "input_dim": 4,
            "adjacency": np.eye(4, dtype=np.float32).tolist(),
            "actor": {
                "encoder": {
                    "type": "dcrnn",
                    "hidden_dim": 16,
                    "max_diffusion_step": 1,
                },
                "head": {"hidden_dims": [8], "activation": "relu"},
            },
            "critic": {
                "encoder": {"hidden_dims": [32]},
                "head": {"hidden_dims": [8]},
            },
        },
    ).build()
    module.make_target_networks()

    inference_output = module.forward_inference({Columns.OBS: torch.zeros(2, 5, 4, 4)})
    output = module.forward_train(
        {
            Columns.OBS: torch.zeros(5, 5, 4, 4),
            Columns.NEXT_OBS: torch.ones(5, 5, 4, 4),
        }
    )

    assert inference_output[Columns.ACTION_DIST_INPUTS].shape == (2, 3)
    assert output[ACTION_PROBS].shape == (5, 3)
    assert output[ACTION_LOG_PROBS].shape == (5, 3)
    assert output[QF_PREDS].shape == (5, 3)
    assert output[QF_TWIN_PREDS].shape == (5, 3)
    assert output[QF_TARGET_NEXT].shape == (5, 3)


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


def test_sac_dcrnn_actor_algorithm_kind_is_supported_by_rllib_runner():
    pytest.importorskip("ray")
    from sumo_rl.experiments import rllib_runner

    assert "sac_dcrnn_actor" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_sac_dcrnn_actor_build_config_installs_graph_multi_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name="sac_dcrnn_actor_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "history_len": 5,
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
                "model_config": {
                    "architecture_tag": "sac_dcrnn_actor",
                    "actor": {
                        "encoder": {
                            "type": "dcrnn",
                            "hidden_dim": 16,
                            "max_diffusion_step": 1,
                        }
                    },
                },
            }
        ),
    )

    config = build_config(cfg, tmp_path, algorithm_kind="sac_dcrnn_actor")

    assert config.rl_module_spec.multi_rl_module_class.__name__ == "CustomSACMultiRLModule"
    assert set(config.rl_module_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    for spec in config.rl_module_spec.rl_module_specs.values():
        assert spec.module_class.__name__ == "CustomSACTorchRLModule"
        assert spec.model_config["architecture_tag"] == "sac_dcrnn_actor"
        assert spec.model_config["custom_sac"]["actor"]["encoder"]["type"] == "dcrnn"
        assert "agent_index" in spec.model_config
        assert "adjacency" in spec.model_config


def test_sac_dcrnn_actor_rejects_shared_policy_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name="sac_dcrnn_actor_shared_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "shared",
                "history_len": 5,
                "model_config": {
                    "actor": {
                        "encoder": {
                            "type": "dcrnn",
                        }
                    },
                },
            }
        ),
    )

    with pytest.raises(ValueError, match="sac_dcrnn_actor currently supports"):
        build_config(cfg, tmp_path, algorithm_kind="sac_dcrnn_actor")


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


def test_sac_custom_alias_normalizes_to_sac_mlp(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDiscreteParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_custom_alias_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
            }
        ),
    )

    config = build_config(cfg, tmp_path, algorithm_kind="sac_custom")

    assert config.rl_module_spec.multi_rl_module_class.__name__ == "CustomSACMultiRLModule"
