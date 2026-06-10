from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from gymnasium.spaces import Box, Discrete
import sumo_rl

from sumo_rl.agents.frap import frap
from sumo_rl.agents.frap.model import FRAPQNetwork, build_competition_mask, infer_default_phase_pairs
from sumo_rl.agents.frap.rllib_module import build_frap_dqn_module_spec
from sumo_rl.experiments import rllib_runner


class _DummyFRAPParallelEnv:
    possible_agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(25,), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return Discrete(8)

    def close(self):
        pass


def test_frap_competition_mask_matches_reference_rule():
    mask = build_competition_mask([[0, 4], [1, 5], [0, 1], [4, 5]])

    assert mask.tolist() == [
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 0],
    ]


def test_frap_q_network_outputs_one_q_value_per_phase_pair():
    obs = torch.zeros((3, 8 + 1 + 2 * 8), dtype=torch.float32)
    obs[:, 0] = 1.0
    obs[:, 9:] = 0.25
    model = FRAPQNetwork(
        observation_dim=25,
        num_actions=8,
        phase_pairs=infer_default_phase_pairs(num_movements=8, num_actions=8),
        demand_shape=2,
    )

    q_values = model(obs)

    assert q_values.shape == (3, 8)
    assert torch.isfinite(q_values).all()


def test_frap_demand_extractor_uses_sumo_rl_split_density_queue_layout():
    obs = torch.zeros((1, 8 + 1 + 2 * 8), dtype=torch.float32)
    obs[:, 9:17] = torch.arange(8, dtype=torch.float32)
    obs[:, 17:25] = torch.arange(10, 18, dtype=torch.float32)
    model = FRAPQNetwork(observation_dim=25, num_actions=8, demand_shape=2)

    demands = model._movement_demands(obs)

    assert demands[0, 0].tolist() == [0.0, 10.0]
    assert demands[0, 7].tolist() == [7.0, 17.0]


def test_frap_module_spec_uses_discrete_action_space():
    obs_space = Box(low=0.0, high=1.0, shape=(25,), dtype=float)
    action_space = Discrete(8)

    spec = build_frap_dqn_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "frap_phase_competition",
            "demand_shape": 2,
            "epsilon": 0.0,
            "double_q": True,
            "num_atoms": 1,
        },
    )

    assert spec.observation_space == obs_space
    assert spec.action_space == action_space
    assert spec.model_config["architecture_tag"] == "frap_phase_competition"


def test_rllib_runner_supports_frap_algorithm_kind():
    assert "frap" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_frap_build_config_registers_custom_rl_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyFRAPParallelEnv(**kwargs))
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="frap_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "model_config": {"demand_shape": 2},
            }
        ),
    )

    config = frap.build_config(cfg, tmp_path)
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    for spec in multi_spec.rl_module_specs.values():
        assert spec.model_config["architecture_tag"] == "frap_phase_competition"
        assert spec.model_config["epsilon"] == [(0, 0.1), (100000, 0.01)]
