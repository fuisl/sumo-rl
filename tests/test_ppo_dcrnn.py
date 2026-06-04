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


class _DummyGraphParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
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
    assert spec.model_config["pre_encoder"]["enabled"] is True
