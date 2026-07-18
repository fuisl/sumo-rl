from types import SimpleNamespace

import sumo_rl
from sumo_rl.agents.rllib_common import build_sumo_parallel_env


def test_build_sumo_parallel_env_maps_raw_sumo_factories_to_parallel_env(monkeypatch, tmp_path):
    calls = []

    class DummyParallelEnv:
        def close(self):
            return None

    def fake_parallel_env(**kwargs):
        calls.append(kwargs)
        return DummyParallelEnv()

    monkeypatch.setattr(sumo_rl, "parallel_env", fake_parallel_env)

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_ingolstadt1"),
        experiment=SimpleNamespace(name="rllib_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="sumo_env", kwargs={"net_file": "net.xml", "route_file": "route.xml"}),
    )

    env = build_sumo_parallel_env(cfg, tmp_path, seed=11)

    assert isinstance(env, DummyParallelEnv)
    assert len(calls) == 1
    assert calls[0]["single_agent"] is False
    assert calls[0]["sumo_seed"] == 11
