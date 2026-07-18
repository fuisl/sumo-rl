# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments.runner import _init_wandb


def test_init_wandb_binds_debug_metrics_to_train_episode_index(monkeypatch, tmp_path):
    class DummyRun:
        def __init__(self):
            self.metric_calls = []

        def define_metric(self, *args, **kwargs):
            self.metric_calls.append((args, kwargs))

    run = DummyRun()

    class DummyWandb:
        @staticmethod
        def init(**kwargs):
            return run

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=True,
            env_file="",
            name=None,
            project=None,
            entity=None,
            group=None,
            tags=[],
            job_type="train",
            mode="disabled",
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[]),
    )

    result = _init_wandb(cfg, tmp_path)

    assert result is run
    assert (("train/*",), {"step_metric": "train/episode_index"}) in run.metric_calls
    assert (("debug/*",), {"step_metric": "train/episode_index"}) in run.metric_calls
    assert (("train/rollout_index",), {}) in run.metric_calls


def test_init_wandb_uses_experiment_name_as_run_name(monkeypatch, tmp_path):
    init_calls = []

    class DummyRun:
        def __init__(self):
            self.name = None

        def define_metric(self, *args, **kwargs):
            del args, kwargs

    run = DummyRun()

    class DummyWandb:
        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return run

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=True,
            env_file="",
            name=None,
            project=None,
            entity=None,
            group=None,
            tags=[],
            job_type="train",
            mode="disabled",
        ),
        experiment=SimpleNamespace(name="fgs_mlp_gat_sac_seed7", project="proj", group=None, tags=[]),
    )

    result = _init_wandb(cfg, tmp_path)

    assert result is run
    assert init_calls[0]["name"] == "fgs_mlp_gat_sac_seed7"
    assert run.name == "fgs_mlp_gat_sac_seed7"


def test_init_wandb_can_skip_final_metric_definitions(monkeypatch, tmp_path):
    class DummyRun:
        def __init__(self):
            self.metric_calls = []

        def define_metric(self, *args, **kwargs):
            self.metric_calls.append((args, kwargs))

    run = DummyRun()

    class DummyWandb:
        @staticmethod
        def init(**kwargs):
            return run

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=True,
            env_file="",
            name=None,
            project=None,
            entity=None,
            group=None,
            tags=[],
            job_type="train",
            mode="disabled",
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[]),
    )

    result = _init_wandb(cfg, tmp_path, include_final_metrics=False)

    assert result is run
    assert (("validation/*",), {"step_metric": "validation/episode_index"}) in run.metric_calls
    assert (("validation/rollout_index",), {}) in run.metric_calls
    assert all(args != ("final/*",) for args, kwargs in run.metric_calls)
    assert all(args != ("eval/episode",) for args, kwargs in run.metric_calls)
