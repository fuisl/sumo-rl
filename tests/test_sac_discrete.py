from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gymnasium.spaces import Box, Discrete


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl
from sumo_rl.agents.rllib_common import build_algorithm_context
from sumo_rl.agents.sac.custom_sac import build_custom_sac_module_spec
from sumo_rl.agents.sac.sac import build_replay_buffer_config


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
    assert spec.model_config["architecture_tag"] == "custom_sac_mlp"


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
