import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
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
    assert kwargs["net_file"].endswith("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml")
    assert kwargs["route_file"].endswith("sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml")
    assert kwargs["out_csv_name"] == "outputs/4x4grid/ppo"
    assert kwargs["use_gui"] is False
    assert kwargs["num_seconds"] == 3600
    assert kwargs["sumo_seed"] == 11
    assert kwargs["single_agent"] is False
