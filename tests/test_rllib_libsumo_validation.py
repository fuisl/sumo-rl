import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner
from sumo_rl.agents.sac import sac as sac_agent


def test_train_rllib_keeps_periodic_validation_under_libsumo_with_traci_envs(monkeypatch, tmp_path):
    validate_holder = {"callback": "unset"}

    class DummyRay:
        @staticmethod
        def init(**kwargs):
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

        def build_algo(self):
            return DummyAlgo()

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind, emit_metrics
        validate_holder["callback"] = validate

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, include_validation_metrics
        return {"algorithm/kind": "ppo", "validation/resco_delay_mean": 1.0}, [], {}, {}, {}, {
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
        }

    monkeypatch.setenv("LIBSUMO_AS_TRACI", "1")
    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(rllib_runner, "_update_wandb_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(rllib_runner, "_update_wandb_best_summary", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=False,
            save_final_model=False,
        ),
        experiment=SimpleNamespace(
            name="demo",
            project="proj",
            group=None,
            tags=[],
            seed=1,
            eval_episodes=1,
            validation_interval_episodes=5,
        ),
        resources=SimpleNamespace(ray_address=None, ray_num_cpus=7, cuda_visible_devices="1"),
        algorithm=SimpleNamespace(
            kind="ppo",
            params={"ray_num_gpus": 0, "num_gpus_per_learner": 0},
        ),
    )

    rllib_runner.train_rllib(cfg)

    assert callable(validate_holder["callback"])


def test_sac_graph_eval_env_uses_isolated_traci_when_libsumo_env_var_is_set(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setenv("LIBSUMO_AS_TRACI", "1")
    monkeypatch.setattr(
        sac_agent,
        "graph_params",
        lambda params: {"copied_params": dict(params)},
    )

    def fake_build_rllib_graph_parallel_env(cfg, run_dir, seed=None, *, params=None, use_libsumo=None):
        del cfg, run_dir
        calls.append(
            {
                "seed": seed,
                "params": params,
                "use_libsumo": use_libsumo,
            }
        )
        return object()

    import sumo_rl.environment.graph_env as graph_env_mod

    monkeypatch.setattr(graph_env_mod, "build_rllib_graph_parallel_env", fake_build_rllib_graph_parallel_env)

    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "history_len": 3,
            }
        )
    )

    sac_agent.build_graph_eval_env(cfg, tmp_path, seed=11)

    assert calls == [{"seed": 11, "params": {"copied_params": {"policy_mode": "independent", "history_len": 3}}, "use_libsumo": False}]
