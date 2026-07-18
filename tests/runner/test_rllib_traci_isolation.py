# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl
from sumo_rl.agents.colight import colight
from sumo_rl.agents.fgs import fgs
from sumo_rl.agents.fgsv2 import fgsv2


def _cfg():
    return SimpleNamespace(
        experiment=SimpleNamespace(name="rllib_test", seed=7, episode_seconds=3600),
        env=SimpleNamespace(
            factory="parallel_env",
            kwargs={
                "net_file": "sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
                "route_file": "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
                "use_gui": False,
            },
        ),
        logging=SimpleNamespace(save_tripinfo_output=False),
        scenario=SimpleNamespace(name="resco_grid4x4"),
        algorithm=SimpleNamespace(params={}),
    )


def test_colight_builder_does_not_override_libsumo_mode(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(colight, "CoLightGraphParallelEnv", lambda env: env)

    colight.build_colight_parallel_env(_cfg(), tmp_path, {}, seed=5)

    assert "use_libsumo" not in calls[0]


def test_fgs_builder_does_not_override_libsumo_mode(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(fgs, "FGSGraphParallelEnv", lambda env, **kwargs: env)

    fgs.build_fgs_parallel_env(_cfg(), tmp_path, {}, seed=5)

    assert "use_libsumo" not in calls[0]


def test_fgsv2_builder_does_not_override_libsumo_mode(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(fgsv2, "FGSGraphParallelEnv", lambda env, **kwargs: env)

    fgsv2.build_fgsv2_parallel_env(_cfg(), tmp_path, {}, seed=5)

    assert "use_libsumo" not in calls[0]
