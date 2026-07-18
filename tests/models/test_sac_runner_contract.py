from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

import sumo_rl
from sumo_rl.agents.sac.sac import build_config
from tests._support.envs import DummyDiscreteParallelEnv as _DummyDiscreteParallelEnv


def test_sac_builtin_algorithm_kind_is_supported_by_rllib_runner():
    from sumo_rl.experiments import rllib_runner

    assert "sac_builtin" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_sac_mlp_algorithm_kind_is_supported_by_rllib_runner():
    from sumo_rl.experiments import rllib_runner

    assert "sac_mlp" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_sac_custom_alias_normalizes_to_sac_mlp(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDiscreteParallelEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_custom_alias_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
            }
        ),
    )

    config = build_config(cfg, tmp_path, algorithm_kind="sac_custom")

    assert config.rl_module_spec.multi_rl_module_class.__name__ == "CustomSACMultiRLModule"
