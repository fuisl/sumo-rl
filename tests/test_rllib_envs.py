import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.rllib import envs as rllib_envs
from sumo_rl.rllib.envs import JointActionBoxEnv, SumoParallelMultiAgentEnv, scenario_factory_name

gym = rllib_envs.gym


class _DummyParallelEnv:
    def __init__(self) -> None:
        self.possible_agents = ["a1", "a2"]
        self.agents = list(self.possible_agents)
        self._last_actions = None
        self._obs = {
            "a1": np.array([1.0, 2.0], dtype=np.float32),
            "a2": np.array([3.0, 4.0], dtype=np.float32),
        }

    def observation_space(self, agent_id):
        del agent_id
        return gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        del seed, options
        return dict(self._obs), {"reset": True}

    def step(self, action_dict):
        self._last_actions = dict(action_dict)
        obs = {
            "a1": np.array([5.0, 6.0], dtype=np.float32),
            "a2": np.array([7.0, 8.0], dtype=np.float32),
        }
        rewards = {"a1": 1.0, "a2": 2.0}
        terminations = {"a1": True, "a2": True, "__all__": True}
        truncations = {"a1": False, "a2": False, "__all__": False}
        infos = {"a1": {}, "a2": {}}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        return None


def test_joint_action_box_env_decodes_continuous_slices_to_discrete_actions():
    base_env = _DummyParallelEnv()
    env = JointActionBoxEnv(base_env)

    obs, info = env.reset()
    assert info == {}
    assert obs.shape == (4,)

    next_obs, reward, terminated, truncated, step_info = env.step(np.array([0.1, 0.9, 0.0, 0.8, 0.1, 0.1]))

    assert terminated is True
    assert truncated is False
    assert reward == 3.0
    assert step_info["joint_action_reward"] == 3.0
    assert base_env._last_actions == {"a1": 1, "a2": 0}
    assert next_obs.shape == (4,)


def test_parallel_multi_agent_wrapper_keeps_spaces_and_passthroughs_actions():
    base_env = _DummyParallelEnv()
    env = SumoParallelMultiAgentEnv(base_env)

    obs, info = env.reset(seed=123)
    assert info == {"reset": True}
    assert set(obs.keys()) == {"a1", "a2"}
    assert env.action_space("a1").n == 3
    assert env.observation_space("a2").shape == (2,)

    _, rewards, terminations, truncations, _ = env.step({"a1": 2, "a2": 1})

    assert rewards == {"a1": 1.0, "a2": 2.0}
    assert terminations["__all__"] is True
    assert truncations["__all__"] is False
    assert base_env._last_actions == {"a1": 2, "a2": 1}


def test_scenario_factory_name_strips_resco_prefix():
    cfg = SimpleNamespace(scenario=SimpleNamespace(name="resco_grid4x4"))
    assert scenario_factory_name(cfg) == "grid4x4"


def test_scenario_factory_name_accepts_unprefixed_resco_names():
    cfg = SimpleNamespace(scenario=SimpleNamespace(name="ingolstadt21"))
    assert scenario_factory_name(cfg) == "ingolstadt21"
