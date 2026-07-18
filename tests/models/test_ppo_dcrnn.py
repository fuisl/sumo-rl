# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl


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
        self.graph = SimpleNamespace(
            model_config=lambda agent_id: {
                "agent_id": str(agent_id),
                "agent_index": 0 if agent_id == "tls_0" else 1,
                "num_nodes": 4,
                "input_dim": 4,
                "adjacency": np.eye(4, dtype=np.float32).tolist(),
                "ts_ids": ["tls_0", "tls_1"],
            }
        )

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)

    def action_space(self, agent_id):
        return Discrete(2 if agent_id == "tls_0" else 3)

    def close(self):
        pass


def test_rllib_runner_supports_ppo_dcrnn_mlp_algorithm_kind():
    pytest.importorskip("ray")
    from sumo_rl.experiments import rllib_runner

    assert "ppo_dcrnn_mlp" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS
    assert "ppo_dcrnn_shared_mlp" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def _ppo_graph_cfg(algorithm_kind: str):
    return SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name=f"{algorithm_kind}_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "history_len": 5,
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
                "model_config": {
                    "hid_dim": 16,
                    "max_diffusion_step": 1,
                    "num_rnn_layers": 1,
                },
            }
        ),
    )


def test_ppo_dcrnn_mlp_build_config_registers_graph_rl_modules(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.ppo import ppo

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))
    cfg = _ppo_graph_cfg("ppo_dcrnn_mlp")

    config = ppo.build_config(cfg, tmp_path, algorithm_kind="ppo_dcrnn_mlp")
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    spec = multi_spec.rl_module_specs["tls_0"]
    assert spec.model_config["architecture_tag"] == "ppo_dcrnn_mlp"
    assert spec.model_config["feature_layout"] == "phase_min_green_density_queue"
    assert spec.model_config["pre_encoder"]["enabled"] is True


def test_ppo_dcrnn_shared_mlp_build_config_registers_custom_multi_module_and_learner(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.ppo import ppo
    from sumo_rl.agents.ppo.learner import PPOSharedEncoderTorchLearner

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))
    cfg = _ppo_graph_cfg("ppo_dcrnn_shared_mlp")

    config = ppo.build_config(cfg, tmp_path, algorithm_kind="ppo_dcrnn_shared_mlp")
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert config.learner_class is PPOSharedEncoderTorchLearner
    assert multi_spec.multi_rl_module_class.__name__ == "PPODCRNNSharedMultiRLModule"
    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    assert multi_spec.rl_module_specs["tls_0"].model_config["architecture_tag"] == "ppo_dcrnn_shared_mlp"


def test_ppo_dcrnn_mlp_shared_backbone_receives_policy_and_value_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    action_space = Discrete(3)
    module = build_ppo_dcrnn_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "ppo_dcrnn_mlp",
            "agent_index": 1,
            "num_nodes": 4,
            "input_dim": 8,
            "adjacency": np.eye(4, dtype=np.float32).tolist(),
            "hid_dim": 16,
            "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
        },
    ).build()

    batch = {Columns.OBS: torch.zeros(3, 5, 4, 8)}
    outputs = module.forward_train(batch)
    loss = outputs[Columns.ACTION_DIST_INPUTS].sum() + outputs[Columns.VF_PREDS].sum()
    loss.backward()

    backbone_grads = [param.grad for param in module.backbone.parameters() if param.requires_grad]
    assert backbone_grads
    assert all(grad is not None for grad in backbone_grads)


def test_ppo_dcrnn_mlp_compute_values_reuses_train_embeddings():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    action_space = Discrete(3)
    module = build_ppo_dcrnn_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "ppo_dcrnn_mlp",
            "agent_index": 1,
            "num_nodes": 4,
            "input_dim": 8,
            "adjacency": np.eye(4, dtype=np.float32).tolist(),
            "hid_dim": 16,
            "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
        },
    ).build()

    batch = {Columns.OBS: torch.zeros(3, 5, 4, 8)}
    outputs = module.forward_train(batch)

    original_backbone = module.backbone

    def _fail_on_backbone(obs):
        del obs
        raise AssertionError("compute_values should reuse train embeddings instead of recomputing the backbone")

    object.__setattr__(module, "backbone", _fail_on_backbone)
    try:
        values = module.compute_values(batch, embeddings=outputs[Columns.EMBEDDINGS])
    finally:
        object.__setattr__(module, "backbone", original_backbone)

    assert values.shape == (3,)


def test_ppo_dcrnn_shared_mlp_parent_owns_shared_backbone_and_children_keep_separate_heads():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(
        {
            "tls_0": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(2),
                model_config={**shared_config, "agent_index": 0},
            ),
            "tls_1": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(3),
                model_config={**shared_config, "agent_index": 1},
            ),
        },
        model_config=shared_config,
    )
    module = multi_spec.build()
    module_0 = module["tls_0"]
    module_1 = module["tls_1"]

    assert module_0._require_shared_backbone() is module.shared_backbone
    assert module_1._require_shared_backbone() is module.shared_backbone
    assert module_0.policy_head[-1].out_features == 2
    assert module_1.policy_head[-1].out_features == 3

    shared_param_ids = {id(param) for param in module.shared_backbone.parameters()}
    child_0_param_ids = {id(param) for param in module_0.parameters()}
    child_1_param_ids = {id(param) for param in module_1.parameters()}

    assert shared_param_ids.isdisjoint(child_0_param_ids)
    assert shared_param_ids.isdisjoint(child_1_param_ids)

    shared_obs = torch.zeros(2, 5, 4, 8)
    outputs = module.forward_train(
        {
            "tls_0": {Columns.OBS: shared_obs},
            "tls_1": {Columns.OBS: shared_obs},
        }
    )
    loss = (
        outputs["tls_0"][Columns.ACTION_DIST_INPUTS].sum()
        + outputs["tls_0"][Columns.VF_PREDS].sum()
        + outputs["tls_1"][Columns.ACTION_DIST_INPUTS].sum()
        + outputs["tls_1"][Columns.VF_PREDS].sum()
    )
    loss.backward()

    shared_grads = [param.grad for param in module.shared_backbone.parameters() if param.requires_grad]
    assert shared_grads
    assert all(grad is not None for grad in shared_grads)


def test_ppo_dcrnn_shared_mlp_reuses_one_graph_encode_for_shared_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(
        {
            "tls_0": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(2),
                model_config={**shared_config, "agent_index": 0},
            ),
            "tls_1": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(3),
                model_config={**shared_config, "agent_index": 1},
            ),
        },
        model_config=shared_config,
    )
    module = multi_spec.build()
    encode_calls = 0
    original_encode_graph = module.shared_backbone.encode_graph

    def _counting_encode_graph(obs):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode_graph(obs)

    monkeypatch.setattr(module.shared_backbone, "encode_graph", _counting_encode_graph)
    shared_obs = torch.zeros(2, 5, 4, 8)
    outputs = module.forward_train(
        {
            "tls_0": {Columns.OBS: shared_obs},
            "tls_1": {Columns.OBS: shared_obs},
        }
    )

    assert encode_calls == 1
    assert outputs["tls_0"][Columns.EMBEDDINGS].shape == (2, 32)
    assert outputs["tls_1"][Columns.EMBEDDINGS].shape == (2, 32)


def test_ppo_dcrnn_shared_mlp_refreshes_cache_for_new_observation_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(
        {
            "tls_0": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(2),
                model_config={**shared_config, "agent_index": 0},
            ),
            "tls_1": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(3),
                model_config={**shared_config, "agent_index": 1},
            ),
        },
        model_config=shared_config,
    )
    module = multi_spec.build()
    encode_calls = 0
    original_encode_graph = module.shared_backbone.encode_graph

    def _counting_encode_graph(obs):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode_graph(obs)

    monkeypatch.setattr(module.shared_backbone, "encode_graph", _counting_encode_graph)
    first_obs = torch.zeros(2, 5, 4, 8)
    second_obs = torch.ones(2, 5, 4, 8)

    module.forward_train(
        {
            "tls_0": {Columns.OBS: first_obs},
            "tls_1": {Columns.OBS: first_obs},
        }
    )
    module.forward_train(
        {
            "tls_0": {Columns.OBS: second_obs},
            "tls_1": {Columns.OBS: second_obs},
        }
    )

    assert encode_calls == 2


def test_ppo_dcrnn_shared_mlp_shared_forward_matches_direct_per_agent_outputs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(
        {
            "tls_0": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(2),
                model_config={**shared_config, "agent_index": 0},
            ),
            "tls_1": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(3),
                model_config={**shared_config, "agent_index": 1},
            ),
        },
        model_config=shared_config,
    )
    module = multi_spec.build()
    shared_obs = torch.randn(2, 5, 4, 8)

    shared_outputs = module.forward_train(
        {
            "tls_0": {Columns.OBS: shared_obs},
            "tls_1": {Columns.OBS: shared_obs},
        }
    )
    direct_outputs_0 = module["tls_0"].forward_train({Columns.OBS: shared_obs})
    direct_outputs_1 = module["tls_1"].forward_train({Columns.OBS: shared_obs})

    assert torch.allclose(shared_outputs["tls_0"][Columns.EMBEDDINGS], direct_outputs_0[Columns.EMBEDDINGS])
    assert torch.allclose(shared_outputs["tls_0"][Columns.ACTION_DIST_INPUTS], direct_outputs_0[Columns.ACTION_DIST_INPUTS])
    assert torch.allclose(shared_outputs["tls_0"][Columns.VF_PREDS], direct_outputs_0[Columns.VF_PREDS])
    assert torch.allclose(shared_outputs["tls_1"][Columns.EMBEDDINGS], direct_outputs_1[Columns.EMBEDDINGS])
    assert torch.allclose(shared_outputs["tls_1"][Columns.ACTION_DIST_INPUTS], direct_outputs_1[Columns.ACTION_DIST_INPUTS])
    assert torch.allclose(shared_outputs["tls_1"][Columns.VF_PREDS], direct_outputs_1[Columns.VF_PREDS])


def test_ppo_dcrnn_shared_mlp_shared_forward_matches_direct_per_agent_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core.columns import Columns

    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(
        {
            "tls_0": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(2),
                model_config={**shared_config, "agent_index": 0},
            ),
            "tls_1": build_ppo_dcrnn_shared_module_spec(
                obs_space,
                Discrete(3),
                model_config={**shared_config, "agent_index": 1},
            ),
        },
        model_config=shared_config,
    )
    shared_module = multi_spec.build()
    direct_module = multi_spec.build()
    direct_module.set_state(shared_module.get_state())
    shared_obs = torch.randn(2, 5, 4, 8)

    shared_outputs = shared_module.forward_train(
        {
            "tls_0": {Columns.OBS: shared_obs},
            "tls_1": {Columns.OBS: shared_obs},
        }
    )
    shared_loss = (
        shared_outputs["tls_0"][Columns.ACTION_DIST_INPUTS].sum()
        + shared_outputs["tls_0"][Columns.VF_PREDS].sum()
        + shared_outputs["tls_1"][Columns.ACTION_DIST_INPUTS].sum()
        + shared_outputs["tls_1"][Columns.VF_PREDS].sum()
    )
    shared_loss.backward()

    direct_outputs_0 = direct_module["tls_0"].forward_train({Columns.OBS: shared_obs})
    direct_outputs_1 = direct_module["tls_1"].forward_train({Columns.OBS: shared_obs})
    direct_loss = (
        direct_outputs_0[Columns.ACTION_DIST_INPUTS].sum()
        + direct_outputs_0[Columns.VF_PREDS].sum()
        + direct_outputs_1[Columns.ACTION_DIST_INPUTS].sum()
        + direct_outputs_1[Columns.VF_PREDS].sum()
    )
    direct_loss.backward()

    for shared_param, direct_param in zip(
        shared_module.shared_backbone.parameters(),
        direct_module.shared_backbone.parameters(),
    ):
        assert torch.allclose(shared_param.grad, direct_param.grad)


def test_ppo_dcrnn_shared_mlp_state_round_trip_restores_shared_backbone_and_heads():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.ppo.rllib_module import build_ppo_dcrnn_shared_module_spec, build_ppo_dcrnn_shared_multi_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    shared_config = {
        "architecture_tag": "ppo_dcrnn_shared_mlp",
        "num_nodes": 4,
        "input_dim": 8,
        "adjacency": np.eye(4, dtype=np.float32).tolist(),
        "hid_dim": 16,
        "pre_encoder": {"enabled": True, "hidden_dim": 16, "activation": "relu"},
    }
    module_specs = {
        "tls_0": build_ppo_dcrnn_shared_module_spec(
            obs_space,
            Discrete(2),
            model_config={**shared_config, "agent_index": 0},
        ),
        "tls_1": build_ppo_dcrnn_shared_module_spec(
            obs_space,
            Discrete(3),
            model_config={**shared_config, "agent_index": 1},
        ),
    }
    multi_spec = build_ppo_dcrnn_shared_multi_module_spec(module_specs, model_config=shared_config)
    source_module = multi_spec.build()
    with torch.no_grad():
        for index, param in enumerate(source_module.shared_backbone.parameters()):
            param.fill_(index + 1)
        for index, param in enumerate(source_module["tls_0"].policy_head.parameters()):
            param.fill_(index + 11)
        for index, param in enumerate(source_module["tls_1"].value_head.parameters()):
            param.fill_(index + 21)

    restored_module = multi_spec.build()
    restored_module.set_state(source_module.get_state())

    for source_param, restored_param in zip(
        source_module.shared_backbone.parameters(),
        restored_module.shared_backbone.parameters(),
    ):
        assert torch.equal(source_param, restored_param)
    for source_param, restored_param in zip(
        source_module["tls_0"].policy_head.parameters(),
        restored_module["tls_0"].policy_head.parameters(),
    ):
        assert torch.equal(source_param, restored_param)
    for source_param, restored_param in zip(
        source_module["tls_1"].value_head.parameters(),
        restored_module["tls_1"].value_head.parameters(),
    ):
        assert torch.equal(source_param, restored_param)


def test_ppo_dcrnn_shared_mlp_learner_registers_one_optimizer_with_deduped_shared_params(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from ray.rllib.core import ALL_MODULES

    from sumo_rl.agents.ppo import ppo

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))
    cfg = _ppo_graph_cfg("ppo_dcrnn_shared_mlp")
    config = ppo.build_config(cfg, tmp_path, algorithm_kind="ppo_dcrnn_shared_mlp")
    learner = config.build_learner()

    assert list(learner._module_optimizers.keys()) == [ALL_MODULES]
    optimizer_names = learner._module_optimizers[ALL_MODULES]
    assert len(optimizer_names) == 1
    optimizer = learner._named_optimizers[optimizer_names[0]]

    optimizer_param_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(optimizer_param_ids) == len(set(optimizer_param_ids))

    shared_param_ids = {id(param) for param in learner.module.shared_backbone.parameters()}
    head_param_ids = {id(param) for module in learner.module.values() for param in module.parameters()}
    optimizer_param_id_set = set(optimizer_param_ids)

    assert shared_param_ids.issubset(optimizer_param_id_set)
    assert head_param_ids.issubset(optimizer_param_id_set)
