# ruff: noqa: E402

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner


def test_train_rllib_existing_ray_address_does_not_pass_local_startup_resources(monkeypatch, tmp_path):
    ray_init_calls = []

    class DummyRay:
        @staticmethod
        def init(**kwargs):
            ray_init_calls.append(kwargs)
            return None

        @staticmethod
        def shutdown():
            return None

        @staticmethod
        def cluster_resources():
            return {"CPU": 8.0, "GPU": 1.0}

        @staticmethod
        def available_resources():
            return {"CPU": 6.0, "GPU": 1.0}

    class DummyAlgo:
        def stop(self):
            return None

    class DummyConfig:
        def build(self):
            return DummyAlgo()

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind, validate
        emit_metrics({"train/env_step": 1.0, "train/episode_index": 1.0}, 1)

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, include_validation_metrics
        return (
            {"algorithm/kind": "ppo", "validation/resco_delay_mean": 1.0},
            [],
            {},
            {},
            {},
            {
                "waiting_time": [],
                "delay": [],
                "pooled_waiting_time": [],
                "pooled_delay": [],
                "total_seeds": 0,
                "seeds_with_completed_trips": 0,
                "seeds_without_completed_trips": 0,
                "total_completed_trips": 0,
                "total_unfinished_trips": 0,
                "total_trips": 0,
            },
        )

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=False,
            save_final_model=False,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(ray_address="auto", ray_num_cpus=7, cuda_visible_devices="1"),
        algorithm=SimpleNamespace(
            kind="ppo",
            params={"ray_num_gpus": 1, "num_gpus_per_learner": 1},
        ),
    )

    rllib_runner.train_rllib(cfg)

    assert ray_init_calls
    ray_init_kwargs = ray_init_calls[0]
    assert ray_init_kwargs["address"] == "auto"
    assert "num_cpus" not in ray_init_kwargs
    assert "num_gpus" not in ray_init_kwargs
    assert "include_dashboard" not in ray_init_kwargs
    assert ray_init_kwargs["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_train_rllib_local_ray_address_forces_explicit_local_startup(monkeypatch, tmp_path):
    ray_init_calls = []
    address_file = tmp_path / "ray_current_cluster"

    class DummyRay:
        class _private:
            class utils:
                @staticmethod
                def get_ray_address_file(_temp_dir=None):
                    return str(address_file)

        @staticmethod
        def init(**kwargs):
            ray_init_calls.append(kwargs)
            return None

        @staticmethod
        def shutdown():
            return None

        @staticmethod
        def cluster_resources():
            return {"CPU": 8.0, "GPU": 1.0}

        @staticmethod
        def available_resources():
            return {"CPU": 6.0, "GPU": 1.0}

    class DummyAlgo:
        def stop(self):
            return None

    class DummyConfig:
        def build(self):
            return DummyAlgo()

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind, validate
        emit_metrics({"train/env_step": 1.0, "train/episode_index": 1.0}, 1)

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, include_validation_metrics
        return (
            {"algorithm/kind": "ppo", "validation/resco_delay_mean": 1.0},
            [],
            {},
            {},
            {},
            {
                "waiting_time": [],
                "delay": [],
                "pooled_waiting_time": [],
                "pooled_delay": [],
                "total_seeds": 0,
                "seeds_with_completed_trips": 0,
                "seeds_without_completed_trips": 0,
                "total_completed_trips": 0,
                "total_unfinished_trips": 0,
                "total_trips": 0,
            },
        )

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setenv("RAY_ADDRESS", "192.168.20.123:6379")
    address_file.write_text("192.168.20.123:6379", encoding="utf-8")
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=False,
            save_final_model=False,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(ray_address=None, ray_num_cpus=7, cuda_visible_devices="1"),
        algorithm=SimpleNamespace(
            kind="ppo",
            params={"ray_num_gpus": 1, "num_gpus_per_learner": 1},
        ),
    )

    rllib_runner.train_rllib(cfg)

    assert len(ray_init_calls) == 1
    init_kwargs = ray_init_calls[0]
    assert init_kwargs["address"] == "local"
    assert init_kwargs["num_cpus"] == 7
    assert init_kwargs["num_gpus"] == 1
    assert init_kwargs["include_dashboard"] is False
    assert init_kwargs["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "RAY_ADDRESS" not in os.environ
    assert not address_file.exists()


def test_train_rllib_auto_ray_address_falls_back_to_explicit_local_startup(monkeypatch, tmp_path):
    ray_init_calls = []
    ray_shutdown_calls = []
    address_file = tmp_path / "ray_current_cluster"

    class DummyRay:
        class _private:
            class utils:
                @staticmethod
                def get_ray_address_file(_temp_dir=None):
                    return str(address_file)

        @staticmethod
        def init(**kwargs):
            ray_init_calls.append(kwargs)
            if len(ray_init_calls) == 1:
                raise ConnectionError("no running Ray cluster")
            return None

        @staticmethod
        def shutdown():
            ray_shutdown_calls.append(True)
            return None

        @staticmethod
        def cluster_resources():
            return {"CPU": 8.0, "GPU": 1.0}

        @staticmethod
        def available_resources():
            return {"CPU": 6.0, "GPU": 1.0}

    class DummyAlgo:
        def stop(self):
            return None

    class DummyConfig:
        def build(self):
            return DummyAlgo()

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind, validate
        emit_metrics({"train/env_step": 1.0, "train/episode_index": 1.0}, 1)

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, include_validation_metrics
        return (
            {"algorithm/kind": "ppo", "validation/resco_delay_mean": 1.0},
            [],
            {},
            {},
            {},
            {
                "waiting_time": [],
                "delay": [],
                "pooled_waiting_time": [],
                "pooled_delay": [],
                "total_seeds": 0,
                "seeds_with_completed_trips": 0,
                "seeds_without_completed_trips": 0,
                "total_completed_trips": 0,
                "total_unfinished_trips": 0,
                "total_trips": 0,
            },
        )

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setenv("RAY_ADDRESS", "192.168.20.123:6379")
    address_file.write_text("192.168.20.123:6379", encoding="utf-8")
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=False,
            save_final_model=False,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(ray_address="auto", ray_num_cpus=7, cuda_visible_devices="1"),
        algorithm=SimpleNamespace(
            kind="ppo",
            params={"ray_num_gpus": 1, "num_gpus_per_learner": 1},
        ),
    )

    rllib_runner.train_rllib(cfg)

    assert len(ray_init_calls) == 2
    assert len(ray_shutdown_calls) == 1
    assert ray_init_calls[0]["address"] == "auto"
    assert "num_cpus" not in ray_init_calls[0]
    assert "num_gpus" not in ray_init_calls[0]

    fallback_kwargs = ray_init_calls[1]
    assert fallback_kwargs["address"] == "local"
    assert fallback_kwargs["num_cpus"] == 7
    assert fallback_kwargs["num_gpus"] == 1
    assert fallback_kwargs["include_dashboard"] is False
    assert fallback_kwargs["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "RAY_ADDRESS" not in os.environ
    assert not address_file.exists()
