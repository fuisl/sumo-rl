# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace

import pytest

gymnasium = pytest.importorskip("gymnasium")
Discrete = gymnasium.spaces.Discrete

import sumo_rl
from sumo_rl.agents.rllib_common import build_algorithm_context
from tests._support.envs import (
    DummyDiscreteParallelEnv as _DummyDiscreteParallelEnv,
)
from tests._support.envs import (
    DummyHeterogeneousSharedEnv as _DummyHeterogeneousSharedEnv,
)


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


def test_shared_policy_context_merges_heterogeneous_box_and_discrete_spaces(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl.agents.rllib_common, "_maybe_pad_pettingzoo_env", lambda env: env)
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyHeterogeneousSharedEnv(**kwargs))

    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="sac_shared_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(params={"policy_mode": "shared"}),
    )

    context = build_algorithm_context(cfg, tmp_path, "sac_mlp")

    shared_spec = context.active_policies["shared_policy"]
    assert shared_spec.observation_space.shape == (16,)
    assert shared_spec.action_space.n == 5
