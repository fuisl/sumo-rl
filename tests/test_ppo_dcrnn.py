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


def test_ppo_dcrnn_mlp_build_config_registers_graph_rl_modules(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.ppo import ppo

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyGraphParallelEnv(**kwargs))
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name="ppo_dcrnn_mlp_test", seed=7, episode_seconds=60),
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

    config = ppo.build_config(cfg, tmp_path, algorithm_kind="ppo_dcrnn_mlp")
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    spec = multi_spec.rl_module_specs["tls_0"]
    assert spec.model_config["architecture_tag"] == "ppo_dcrnn_mlp"
    assert spec.model_config["feature_layout"] == "phase_min_green_density_queue"
    assert spec.model_config["pre_encoder"]["enabled"] is True


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
