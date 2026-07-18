# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gymnasium.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl
from sumo_rl.agents import rllib_common


def _cfg(**env_kwargs):
    return SimpleNamespace(
        experiment=SimpleNamespace(name="rllib_test", seed=7, episode_seconds=3600),
        env=SimpleNamespace(factory="parallel_env", kwargs=env_kwargs),
        logging=SimpleNamespace(save_tripinfo_output=False),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )


def test_scenario_factory_name_strips_resco_prefix():
    cfg = SimpleNamespace(scenario=SimpleNamespace(name="resco_grid4x4"))
    assert rllib_common.scenario_factory_name(cfg) == "grid4x4"


def test_scenario_factory_name_accepts_unprefixed_resco_names():
    cfg = SimpleNamespace(scenario=SimpleNamespace(name="ingolstadt21"))
    assert rllib_common.scenario_factory_name(cfg) == "ingolstadt21"


def test_build_sumo_parallel_env_calls_parallel_env_with_configured_kwargs(monkeypatch, tmp_path):
    calls = []
    expected_env = object()

    def fake_parallel_env(**kwargs):
        calls.append(kwargs)
        return expected_env

    monkeypatch.setattr(sumo_rl, "parallel_env", fake_parallel_env)

    env = rllib_common.build_sumo_parallel_env(
        _cfg(
            net_file="sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
            route_file="sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
            out_csv_name="outputs/4x4grid/ppo",
            use_gui=False,
            delta_time=5,
        ),
        tmp_path,
        seed=11,
    )

    assert env is expected_env
    assert len(calls) == 1
    kwargs = calls[0]
    assert Path(kwargs["net_file"]).as_posix().endswith("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml")
    assert Path(kwargs["route_file"]).as_posix().endswith("sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml")
    assert kwargs["out_csv_name"] == "outputs/4x4grid/ppo"
    assert kwargs["use_gui"] is False
    assert kwargs["num_seconds"] == 3600
    assert kwargs["sumo_seed"] == 11
    assert kwargs["single_agent"] is False
    assert "use_libsumo" not in kwargs


def test_build_multi_agent_policies_uses_post_reset_spaces(monkeypatch, tmp_path):
    class DummyParallelEnv:
        possible_agents = ["tls_0", "tls_1"]

        def __init__(self):
            self.reset_called = False
            self.closed = False

        def reset(self, seed=None):
            del seed
            self.reset_called = True
            return {}, {}

        def observation_space(self, agent_id):
            if agent_id == "tls_0":
                size = 16 if self.reset_called else 21
            else:
                size = 12
            return Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)

        def action_space(self, agent_id):
            size = 4 if agent_id == "tls_0" else 3
            return Discrete(size)

        def close(self):
            self.closed = True

    env = DummyParallelEnv()
    monkeypatch.setattr(rllib_common, "build_sumo_parallel_env", lambda cfg, run_dir, seed: env)

    policies = rllib_common.build_multi_agent_policies(_cfg(), tmp_path, pad_spaces=False)

    assert env.reset_called is True
    assert env.closed is True
    assert policies["tls_0"].observation_space.shape == (16,)
    assert policies["tls_0"].action_space.n == 4
    assert policies["tls_1"].observation_space.shape == (12,)
    assert policies["tls_1"].action_space.n == 3
