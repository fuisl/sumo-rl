# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from gymnasium.spaces import Dict as DictSpace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

import sumo_rl
from sumo_rl.agents.fgsv2 import fgsv2
from sumo_rl.agents.fgsv2.model import CentralGraphActionTokenCritic, FGSv2GraphEncoder
from sumo_rl.agents.fgsv2.rllib_module import build_fgsv2_sac_module_spec, normalize_fgsv2_model_config
from sumo_rl.experiments import rllib_runner


def _phase_pair_mask(batch_size=3, num_nodes=2, num_actions=4, num_movements=4):
    mask = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    return mask[:num_nodes, :num_actions, :num_movements].unsqueeze(0).repeat(batch_size, 1, 1, 1)


def _graph_obs_space(num_nodes=2, node_dim=13, max_edges=2, num_actions=4, num_movements=4):
    return DictSpace(
        {
            "node_features": Box(-np.inf, np.inf, shape=(num_nodes, node_dim), dtype=np.float32),
            "edge_index": Box(0, max(0, num_nodes - 1), shape=(2, max_edges), dtype=np.int64),
            "edge_mask": Box(0.0, 1.0, shape=(max_edges,), dtype=np.float32),
            "edge_weight": Box(0.0, np.inf, shape=(max_edges,), dtype=np.float32),
            "ego_index": Box(0, max(0, num_nodes - 1), shape=(), dtype=np.int64),
            "action_mask": Box(0.0, 1.0, shape=(num_actions,), dtype=np.float32),
            "node_action_mask": Box(0.0, 1.0, shape=(num_nodes, num_actions), dtype=np.float32),
            "phase_pair_mask": Box(0.0, 1.0, shape=(num_nodes, num_actions, num_movements), dtype=np.float32),
            "phase_competition_mask": Box(0.0, 1.0, shape=(num_nodes, num_actions, num_actions - 1), dtype=np.float32),
            "prev_joint_action": Box(0.0, 1.0, shape=(num_nodes, num_actions), dtype=np.float32),
        }
    )


def _graph_obs(batch_size=3, num_nodes=2, node_dim=13, max_edges=2, num_actions=4):
    obs = torch.zeros((batch_size, num_nodes, node_dim), dtype=torch.float32)
    obs[:, 0, 0] = 1.0
    obs[:, 1, 1] = 1.0
    obs[:, :, 5:] = 0.25
    node_action_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    node_action_mask = node_action_mask[:num_nodes, :num_actions].unsqueeze(0).repeat(batch_size, 1, 1)
    ego_index = torch.tensor([0, 1, 0], dtype=torch.long)[:batch_size]
    action_mask = torch.stack([node_action_mask[i, ego_index[i]] for i in range(batch_size)], dim=0)
    prev_joint_action = torch.zeros((batch_size, num_nodes, num_actions), dtype=torch.float32)
    prev_joint_action[:, 0, 0] = 1.0
    prev_joint_action[:, 1, 1] = 1.0
    return {
        "node_features": obs,
        "edge_index": torch.tensor([[[0, 1], [1, 0]]] * batch_size, dtype=torch.long),
        "edge_mask": torch.ones((batch_size, max_edges), dtype=torch.float32),
        "edge_weight": torch.ones((batch_size, max_edges), dtype=torch.float32),
        "ego_index": ego_index,
        "action_mask": action_mask,
        "node_action_mask": node_action_mask,
        "phase_pair_mask": _phase_pair_mask(batch_size=batch_size, num_nodes=num_nodes, num_actions=num_actions),
        "phase_competition_mask": torch.zeros((batch_size, num_nodes, num_actions, num_actions - 1), dtype=torch.float32),
        "prev_joint_action": prev_joint_action,
    }


def test_rllib_runner_supports_fgsv2_algorithm_kind():
    assert "fgsv2" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_fgsv2_model_config_normalization_accepts_defaults_and_rejects_invalid_values():
    config = normalize_fgsv2_model_config({})

    assert config["architecture_tag"] == "fgsv2_frap_tokens_gnn_sac"
    assert config["communication"]["type"] == "gatv2"
    assert config["critic"]["type"] == "central_graph_joint_action"
    with pytest.raises(ValueError, match="communication.type"):
        normalize_fgsv2_model_config({"communication": {"type": "bad"}})
    with pytest.raises(ValueError, match="critic.type"):
        normalize_fgsv2_model_config({"critic": {"type": "central_graph_policy_context"}})


def test_fgsv2_graph_encoder_returns_frap_action_tokens_for_heterogeneous_masks():
    encoder = FGSv2GraphEncoder(
        node_feature_dim=13,
        num_nodes=2,
        num_actions=4,
        model_config={
            "frap": {"demand_shape": 2, "conv_units": 8},
            "adapter": {"dim": 16},
            "communication": {"enabled": True, "type": "gatv2", "num_heads": 1, "head_dim": 4},
        },
    )

    output = encoder(_graph_obs(batch_size=2))

    assert output["action_tokens"].shape == (2, 2, 4, 16)
    assert output["graph"].shape == (2, 2, 16)
    assert output["ego_action_tokens"].shape == (2, 4, 16)
    assert torch.isfinite(output["action_tokens"]).all()
    assert output["action_tokens"][0, 1, 2:].abs().sum().item() == pytest.approx(0.0)


def test_fgsv2_actor_logits_mask_invalid_padded_actions():
    from ray.rllib.core.columns import Columns

    module = build_fgsv2_sac_module_spec(
        _graph_obs_space(),
        Discrete(4),
        model_config={
            "frap": {"demand_shape": 2, "conv_units": 8},
            "adapter": {"dim": 16},
            "communication": {"enabled": True, "type": "gatv2", "num_heads": 1, "head_dim": 4},
            "actor": {"hidden_dims": [16]},
            "critic": {"type": "central_graph_joint_action", "hidden_dims": [32]},
        },
    ).build()

    logits = module.forward_inference({Columns.OBS: _graph_obs(batch_size=2)})[Columns.ACTION_DIST_INPUTS]

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits[0]).all()
    assert logits[1, 2:].tolist() == [-1.0e9, -1.0e9]


def test_fgsv2_critic_depends_on_joint_action_context():
    critic = CentralGraphActionTokenCritic(graph_dim=8, num_nodes=2, num_actions=3, hidden_dims=[16])
    graph_h = torch.randn(2, 2, 8)
    ego_tokens = torch.randn(2, 3, 8)
    ego_index = torch.zeros(2, dtype=torch.long)
    context_a = torch.nn.functional.one_hot(torch.tensor([[0, 0], [0, 0]]), num_classes=3).float()
    context_b = torch.nn.functional.one_hot(torch.tensor([[0, 1], [0, 2]]), num_classes=3).float()

    q_a = critic(graph_h, ego_tokens, context_a, ego_index)
    q_b = critic(graph_h, ego_tokens, context_b, ego_index)

    assert q_a.shape == (2, 3)
    assert torch.isfinite(q_a).all()
    assert not torch.allclose(q_a, q_b)


def test_fgsv2_module_inference_and_train_outputs_discrete_sac_tensors():
    from ray.rllib.algorithms.sac.sac_learner import ACTION_PROBS, QF_PREDS, QF_TARGET_NEXT, QF_TWIN_PREDS
    from ray.rllib.core.columns import Columns

    module = build_fgsv2_sac_module_spec(
        _graph_obs_space(),
        Discrete(4),
        model_config={
            "frap": {"demand_shape": 2, "conv_units": 8},
            "adapter": {"dim": 16},
            "communication": {"enabled": True, "type": "gatv2", "num_heads": 1, "head_dim": 4},
            "actor": {"hidden_dims": [16]},
            "critic": {"type": "central_graph_joint_action", "hidden_dims": [32]},
        },
    ).build()
    module.make_target_networks()

    obs = _graph_obs(batch_size=2)
    inference = module.forward_inference({Columns.OBS: obs})
    train_out = module.forward_train({Columns.OBS: obs, Columns.NEXT_OBS: obs})

    assert inference[Columns.ACTION_DIST_INPUTS].shape == (2, 4)
    assert train_out[ACTION_PROBS].shape == (2, 4)
    assert train_out[QF_PREDS].shape == (2, 4)
    assert train_out[QF_TWIN_PREDS].shape == (2, 4)
    assert train_out[QF_TARGET_NEXT].shape == (2, 4)


class _DummySignal:
    def __init__(self, lanes, out_lanes, phase_lanes=None):
        self.lanes = lanes
        self.out_lanes = out_lanes
        self.phase_lanes = phase_lanes if phase_lanes is not None else [lanes]


class _DummyBaseEnv:
    ts_ids = ["tls_0", "tls_1"]

    def __init__(self):
        self.traffic_signals = {
            "tls_0": _DummySignal(
                ["a", "b", "c", "d"],
                ["e"],
                phase_lanes=[["a", "c"], ["b", "d"], ["a", "b"], ["c", "d"]],
            ),
            "tls_1": _DummySignal(
                ["b", "c"],
                ["d"],
                phase_lanes=[["b"], ["c"]],
            ),
        }


class _DummyParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self):
        self.env = _DummyBaseEnv()

    def observation_space(self, agent_id):
        if agent_id == "tls_1":
            return Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        return Box(0.0, 1.0, shape=(13,), dtype=np.float32)

    def action_space(self, agent_id):
        if agent_id == "tls_1":
            return Discrete(2)
        return Discrete(4)

    def reset(self, seed=None, options=None):
        del seed, options
        obs = np.zeros(13, dtype=np.float32)
        obs[0] = 1.0
        small_obs = np.array([0.0, 1.0, 1.0, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        return {"tls_0": obs, "tls_1": small_obs}, {"tls_0": {}, "tls_1": {}}

    def step(self, actions):
        del actions
        local_obs, infos = self.reset()
        rewards = {"tls_0": 1.0, "tls_1": 2.0}
        terminations = {"tls_0": False, "tls_1": False, "__all__": False}
        truncations = {"tls_0": False, "tls_1": False, "__all__": False}
        return local_obs, rewards, terminations, truncations, infos

    def close(self):
        pass


def test_fgsv2_build_config_registers_shared_custom_rl_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyParallelEnv())
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="fgsv2_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "shared",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
                "model_config": {
                    "frap": {"demand_shape": 2, "conv_units": 8},
                    "adapter": {"dim": 16},
                    "communication": {"enabled": False, "type": "identity"},
                    "actor": {"hidden_dims": [16]},
                    "critic": {"type": "central_graph_joint_action", "hidden_dims": [32]},
                    "topology": {"source": "direct_lane", "render": False},
                },
            }
        ),
    )

    config = fgsv2.build_config(cfg, tmp_path)

    assert set(config.rl_module_spec.rl_module_specs.keys()) == {"shared_policy"}
    spec = config.rl_module_spec.rl_module_specs["shared_policy"]
    assert spec.module_class.__name__ == "FGSv2SACTorchRLModule"
    assert spec.model_config["architecture_tag"] == "fgsv2_frap_tokens_gnn_sac"
    assert config.learner_class.__name__ == "FGSv2SACTorchLearner"
