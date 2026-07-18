# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("ray")

from sumo_rl.experiments import rllib_runner


def test_build_eval_env_defaults_to_traci_when_training_uses_libsumo(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda cfg, run_dir, seed, pad_spaces, use_libsumo=None: calls.append(
            {"seed": seed, "pad_spaces": pad_spaces, "use_libsumo": use_libsumo}
        )
        or object(),
    )

    cfg = SimpleNamespace(
        env=SimpleNamespace(kwargs={"use_libsumo": True}),
        logging=SimpleNamespace(eval_use_libsumo=False),
    )

    rllib_runner._build_eval_env(cfg, tmp_path, 11, algorithm_kind="dqn", policy_mode="independent")

    assert calls == [{"seed": 11, "pad_spaces": False, "use_libsumo": False}]


def test_training_uses_libsumo_reads_hydra_config():
    cfg = SimpleNamespace(env=SimpleNamespace(kwargs={}))

    assert rllib_runner._training_uses_libsumo(cfg) is False

    cfg.env.kwargs["use_libsumo"] = True
    assert rllib_runner._training_uses_libsumo(cfg) is True


def test_validate_manual_evaluation_backend_config_allows_explicit_split_backends():
    cfg = SimpleNamespace(
        env=SimpleNamespace(kwargs={"use_libsumo": True}),
        logging=SimpleNamespace(eval_use_libsumo=False),
        algorithm=SimpleNamespace(params={}),
    )

    rllib_runner._validate_manual_evaluation_backend_config(cfg)


def test_build_eval_env_can_explicitly_use_libsumo(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda cfg, run_dir, seed, pad_spaces, use_libsumo=None: calls.append(
            {"seed": seed, "pad_spaces": pad_spaces, "use_libsumo": use_libsumo}
        )
        or object(),
    )

    cfg = SimpleNamespace(
        env=SimpleNamespace(kwargs={"use_libsumo": True}),
        logging=SimpleNamespace(eval_use_libsumo=True),
    )

    rllib_runner._build_eval_env(cfg, tmp_path, 13, algorithm_kind="dqn", policy_mode="shared")

    assert calls == [{"seed": 13, "pad_spaces": True, "use_libsumo": True}]


def test_validate_manual_evaluation_backend_config_rejects_rllib_native_eval_conflict():
    cfg = SimpleNamespace(
        env=SimpleNamespace(kwargs={"use_libsumo": True}),
        logging=SimpleNamespace(eval_use_libsumo=False),
        algorithm=SimpleNamespace(params={"evaluation_interval": 3}),
    )

    try:
        rllib_runner._validate_manual_evaluation_backend_config(cfg)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "evaluation_interval" in str(exc)


def test_build_eval_env_does_not_mutate_tripinfo_retention(monkeypatch, tmp_path):
    class DummyEvalEnv:
        def __init__(self):
            self.keep_tripinfo_output = False
            self.traffic_signals = {}
            self.sim_step = 0

    eval_env = DummyEvalEnv()

    monkeypatch.setattr(rllib_runner, "build_rllib_parallel_env", lambda *args, **kwargs: eval_env)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(params={"policy_mode": "independent"}))
    built_env = rllib_runner._build_eval_env(cfg, tmp_path, seed=7, algorithm_kind="ppo", policy_mode="independent")

    assert built_env is eval_env
    assert eval_env.keep_tripinfo_output is False


def test_build_eval_env_uses_graph_eval_env_for_ppo_dcrnn_mlp(monkeypatch, tmp_path):
    graph_eval_env = object()

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(build_graph_eval_env=lambda *args, **kwargs: graph_eval_env),
    )
    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("flat eval env should not be used")),
    )

    cfg = SimpleNamespace(algorithm=SimpleNamespace(params={"policy_mode": "independent"}))
    built_env = rllib_runner._build_eval_env(
        cfg,
        tmp_path,
        seed=7,
        algorithm_kind="ppo_dcrnn_mlp",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env


def test_build_eval_env_uses_graph_eval_env_for_dqn_dcrnn_mlp(monkeypatch, tmp_path):
    graph_eval_env = object()

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(build_graph_eval_env=lambda *args, **kwargs: graph_eval_env),
    )
    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("flat eval env should not be used")),
    )

    cfg = SimpleNamespace(algorithm=SimpleNamespace(params={"policy_mode": "independent"}))
    built_env = rllib_runner._build_eval_env(
        cfg,
        tmp_path,
        seed=7,
        algorithm_kind="dqn_dcrnn_mlp",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env


def test_build_eval_env_uses_flat_env_for_sac_builtin(monkeypatch, tmp_path):
    calls = []
    flat_eval_env = object()

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(
            build_graph_eval_env=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("graph eval env should not be used")
            )
        ),
    )
    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda *args, **kwargs: calls.append(kwargs) or flat_eval_env,
    )

    cfg = SimpleNamespace(algorithm=SimpleNamespace(params={"policy_mode": "independent"}))
    built_env = rllib_runner._build_eval_env(
        cfg,
        tmp_path,
        seed=7,
        algorithm_kind="sac_builtin",
        policy_mode="independent",
    )

    assert built_env is flat_eval_env
    assert calls == [{"seed": 7, "pad_spaces": False, "use_libsumo": False}]


def test_build_eval_env_uses_flat_env_for_sac_mlp(monkeypatch, tmp_path):
    calls = []
    flat_eval_env = object()

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(
            build_graph_eval_env=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("graph eval env should not be used")
            )
        ),
    )
    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda *args, **kwargs: calls.append(kwargs) or flat_eval_env,
    )

    cfg = SimpleNamespace(algorithm=SimpleNamespace(params={"policy_mode": "independent"}))
    built_env = rllib_runner._build_eval_env(
        cfg,
        tmp_path,
        seed=9,
        algorithm_kind="sac_mlp",
        policy_mode="independent",
    )

    assert built_env is flat_eval_env
    assert calls == [{"seed": 9, "pad_spaces": False, "use_libsumo": False}]
