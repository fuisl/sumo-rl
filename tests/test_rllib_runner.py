import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner
from sumo_rl.experiments.runner import _init_wandb, _log_outputs
from sumo_rl.experiments.rllib_runner import _build_policy_mapping, _policy_id_for_agent
from sumo_rl.agents.dqn.dqn import build_replay_buffer_config
from sumo_rl.agents.ppo.ppo import extract_training_metrics as extract_ppo_training_metrics
from sumo_rl.agents.sac.sac import extract_training_metrics as extract_sac_training_metrics
from sumo_rl.agents.rllib_common import (
    _completed_episode_summary_history,
    apply_standard_evaluation_settings,
    build_training_episode_row,
    completed_training_episodes,
    emit_validation_if_due,
    emit_training_episode_rows,
    emit_training_metrics_by_step,
    episode_steps,
    should_log_training_episode,
    should_log_training_metrics,
    trace_mode,
    train_log_freq_steps,
    training_episode_target,
    training_should_stop,
    validation_interval_episodes,
    validation_interval_steps,
)


def test_policy_id_for_agent_shared_mode_uses_shared_policy_name():
    assert _policy_id_for_agent("tls_1", "shared") == "shared_policy"


def test_policy_id_for_agent_independent_mode_uses_agent_id():
    assert _policy_id_for_agent("tls_1", "independent") == "tls_1"


def test_build_policy_mapping_shared_mode_maps_all_agents_to_one_policy():
    mapping_fn = _build_policy_mapping("shared")
    assert mapping_fn("tls_1") == "shared_policy"
    assert mapping_fn("tls_2") == "shared_policy"


def test_build_policy_mapping_independent_mode_keeps_agent_identity():
    mapping_fn = _build_policy_mapping("independent")
    assert mapping_fn("tls_1") == "tls_1"
    assert mapping_fn("tls_2") == "tls_2"


def test_rllib_run_name_uses_logging_name_when_set():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name="wandb-title"),
        experiment=SimpleNamespace(name="experiment-title"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo") == "wandb-title"


def test_rllib_run_name_uses_explicit_experiment_name():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name=None),
        experiment=SimpleNamespace(name="experiment-title"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo") == "experiment-title"


def test_rllib_run_name_keeps_generated_name_for_default_experiment_name():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(name=None),
        experiment=SimpleNamespace(name="rllib"),
        scenario=SimpleNamespace(name="resco_grid4x4"),
    )

    assert rllib_runner._rllib_run_name(cfg, "ppo").startswith("grid4x4__ppo__")


def test_rllib_runtime_params_reads_hydra_resources_before_algorithm_params():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "resources": {
                "ray_address": "auto",
                "ray_num_cpus": 7,
                "native_num_threads": 1,
            },
            "algorithm": {
                "params": {
                    "ray_num_cpus": 3,
                    "num_env_runners": 1,
                },
            },
        }
    )

    params = rllib_runner._rllib_runtime_params(cfg)

    assert params["ray_address"] == "auto"
    assert params["ray_num_cpus"] == 3
    assert params["native_num_threads"] == 1
    assert params["num_env_runners"] == 1


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


def test_validation_image_loggers_respect_disabled_toggles(monkeypatch):
    class DummyWandbRun:
        def __init__(self):
            self.calls = []

        def log(self, payload):
            self.calls.append(payload)

    class DummyWandb:
        @staticmethod
        def Image(value, caption=None):
            return {"value": value, "caption": caption}

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)

    wandb_run = DummyWandbRun()
    logging_cfg = SimpleNamespace(
        validation_log_action_shares=False,
        validation_log_action_timelines=False,
        validation_log_phase_queues=False,
        validation_log_tripinfo_distributions=False,
    )

    rllib_runner._log_validation_action_plot_images(
        wandb_run,
        {"tls_0": [{"action_0": 1.0}]},
        {"tls_0": [0, 1]},
        {"tls_0": [{"phase_0": 2.0}]},
        pass_index=1,
        env_step=10,
        episode_index=2,
        decision_seconds=5,
        logging_cfg=logging_cfg,
    )
    rllib_runner._log_validation_tripinfo_distribution_images(
        wandb_run,
        {
            "waiting_time": [[1.0]],
            "delay": [[2.0]],
            "pooled_waiting_time": [1.0],
            "pooled_delay": [2.0],
            "total_seeds": 1,
            "seeds_with_completed_trips": 1,
            "total_completed_trips": 1,
            "total_unfinished_trips": 0,
        },
        pass_index=1,
        env_step=10,
        episode_index=2,
        logging_cfg=logging_cfg,
    )

    assert wandb_run.calls == []


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


def test_dqn_uses_multi_agent_episode_replay_buffer_by_default():
    replay_config = build_replay_buffer_config({})

    assert replay_config["type"] == "MultiAgentPrioritizedEpisodeReplayBuffer"
    assert replay_config["capacity"] == 50000
    assert replay_config["alpha"] == 0.6
    assert replay_config["beta"] == 0.4


def test_dqn_replay_buffer_config_is_customizable():
    replay_config = build_replay_buffer_config(
        {
            "replay_buffer_type": "MultiAgentEpisodeReplayBuffer",
            "replay_buffer_capacity": 123,
        }
    )

    assert replay_config == {"type": "MultiAgentEpisodeReplayBuffer", "capacity": 123}


def test_evaluate_closes_env_before_building_final_summary(monkeypatch, tmp_path):
    class DummyEvalEnv:
        possible_agents = ["tls_1"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    eval_env = DummyEvalEnv()

    def fake_build_rllib_parallel_env(*args, **kwargs):
        del args, kwargs
        return eval_env

    def fake_run_episode_trace(*args, **kwargs):
        del args, kwargs
        return 12.5, {"tls_1": [0, 1, 0]}, {"tls_1": 2}, {"tls_1": []}

    def fake_build_summary(env, **kwargs):
        assert env.closed is True
        return {
            "algorithm/kind": kwargs["algorithm_kind"],
            "final/eval/mean_reward": kwargs["eval_mean_reward"],
            "final/eval/std_reward": kwargs["eval_std_reward"],
            "final/resco/avg_delay": 3.0,
            "final/resco/avg_delay_std": 0.8,
            "final/resco/wait_std": 0.4,
        }

    monkeypatch.setattr(rllib_runner, "build_rllib_parallel_env", fake_build_rllib_parallel_env)
    monkeypatch.setattr(rllib_runner, "_run_multi_agent_episode_trace", fake_run_episode_trace)
    monkeypatch.setattr(rllib_runner, "_build_final_eval_summary_row", fake_build_summary)

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(seed=7, eval_episodes=1, eval_seeds=None),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )

    summary = rllib_runner._evaluate(
        cfg,
        tmp_path,
        algo=object(),
        algorithm_kind="ppo",
        logging_cfg=SimpleNamespace(log_final_traffic_metrics=True),
    )

    assert eval_env.closed is True
    assert summary["algorithm/kind"] == "ppo"
    assert summary["final/eval/mean_reward"] == 12.5
    assert summary["final/resco/avg_delay"] == 3.0
    assert summary["final/resco/avg_delay_std"] == 0.8
    assert summary["final/resco/wait_std"] == 0.4


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


def test_build_eval_env_uses_graph_eval_env_for_sac_dcrnn_actor(monkeypatch, tmp_path):
    graph_eval_env = object()
    calls = []

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(build_graph_eval_env=lambda *args, **kwargs: calls.append(kwargs) or graph_eval_env),
    )
    monkeypatch.setattr(
        rllib_runner,
        "build_rllib_parallel_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("flat eval env should not be used")),
    )

    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
        logging=SimpleNamespace(eval_use_libsumo=False),
    )
    built_env = rllib_runner._build_eval_env(
        cfg,
        tmp_path,
        seed=7,
        algorithm_kind="sac_dcrnn_actor",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env
    assert calls == [{"seed": 7, "use_libsumo": False}]


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


def test_build_eval_env_uses_graph_eval_env_for_sac_dcrnn_full(monkeypatch, tmp_path):
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
        algorithm_kind="sac_dcrnn_full",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env


def test_build_eval_env_uses_graph_eval_env_for_sac_dcrnn_shared_mlp(monkeypatch, tmp_path):
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
        algorithm_kind="sac_dcrnn_shared_mlp",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env


def test_build_eval_env_uses_flat_env_for_sac_builtin(monkeypatch, tmp_path):
    calls = []
    flat_eval_env = object()

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(build_graph_eval_env=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph eval env should not be used"))),
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


def test_build_eval_env_uses_traci_isolation_for_graph_sac_eval(monkeypatch, tmp_path):
    graph_eval_env = object()
    calls = []

    monkeypatch.setattr(
        rllib_runner,
        "_algorithm_module",
        lambda algorithm_kind: SimpleNamespace(
            build_graph_eval_env=lambda *args, **kwargs: calls.append(kwargs) or graph_eval_env
        ),
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
        algorithm_kind="sac_dcrnn_actor",
        policy_mode="independent",
    )

    assert built_env is graph_eval_env
    assert calls == [{"seed": 7}]


def test_sync_env_runner_weights_for_evaluation_uses_learner_weights():
    calls = []

    class DummyLearnerGroup:
        def get_weights(self):
            return {"module": {"weight": 123}}

    class DummyEnvRunner:
        def set_weights(self, weights):
            calls.append(weights)

    algo = SimpleNamespace(
        learner_group=DummyLearnerGroup(),
        env_runner=DummyEnvRunner(),
    )

    synced = rllib_runner._sync_env_runner_weights_for_evaluation(algo)

    assert synced is True
    assert calls == [{"module": {"weight": 123}}]


def test_sync_env_runner_weights_for_evaluation_returns_false_without_sync_api():
    algo = SimpleNamespace(
        learner_group=object(),
        env_runner=object(),
    )

    synced = rllib_runner._sync_env_runner_weights_for_evaluation(algo)

    assert synced is False


def test_compute_single_action_prefers_module_forward_over_algo_compute_single_action():
    class DummyColumns:
        ACTIONS = "actions"
        OBS = "obs"

    class DummyTensor:
        def __init__(self, values):
            self._values = values

        def unsqueeze(self, dim):
            del dim
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class DummyTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            del dtype, device
            return DummyTensor(values)

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def no_grad():
            class _NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _NoGrad()

    class DummyRayColumnsModule:
        Columns = DummyColumns

    class DummyModule:
        def __init__(self):
            self.calls = []

        def forward_inference(self, batch):
            self.calls.append(batch)
            return {DummyColumns.ACTIONS: DummyTensor([2])}

    class DummyAlgo:
        def __init__(self):
            self.calls = []
            self.module = DummyModule()

        def compute_single_action(self, obs, policy_id=None, explore=None):
            self.calls.append(
                {
                    "obs": obs,
                    "policy_id": policy_id,
                    "explore": explore,
                }
            )
            return 3

        def get_module(self, policy_id=None):
            self.calls.append({"get_module_policy_id": policy_id})
            return self.module

    algo = DummyAlgo()

    original_torch = sys.modules.get("torch")
    original_ray = sys.modules.get("ray")
    original_ray_rllib = sys.modules.get("ray.rllib")
    original_ray_rllib_core = sys.modules.get("ray.rllib.core")
    original_ray_rllib_core_columns = sys.modules.get("ray.rllib.core.columns")
    try:
        sys.modules["torch"] = DummyTorch
        sys.modules["ray"] = SimpleNamespace()
        sys.modules["ray.rllib"] = SimpleNamespace()
        sys.modules["ray.rllib.core"] = SimpleNamespace()
        sys.modules["ray.rllib.core.columns"] = DummyRayColumnsModule

        action = rllib_runner._compute_single_action(algo, {"graph": [1, 2, 3]}, policy_id="tls_1")
    finally:
        for name, original in (
            ("torch", original_torch),
            ("ray", original_ray),
            ("ray.rllib", original_ray_rllib),
            ("ray.rllib.core", original_ray_rllib_core),
            ("ray.rllib.core.columns", original_ray_rllib_core_columns),
        ):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    assert action == 2
    assert algo.calls == [{"get_module_policy_id": "tls_1"}]


def test_compute_single_action_uses_module_forward_when_rllib_compute_single_action_hits_env_runner_gap():
    class DummyColumns:
        ACTIONS = "actions"
        OBS = "obs"

    class DummyTensor:
        def __init__(self, values):
            self._values = values

        def unsqueeze(self, dim):
            del dim
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class DummyTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            del dtype, device
            return DummyTensor(values)

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def no_grad():
            class _NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _NoGrad()

    class DummyRayColumnsModule:
        Columns = DummyColumns

    class DummyModule:
        def forward_inference(self, batch):
            del batch
            return {DummyColumns.ACTIONS: DummyTensor([2])}

    class DummyAlgo:
        def compute_single_action(self, obs, policy_id=None, explore=None):
            del obs, policy_id, explore
            raise AttributeError("'MultiAgentEnvRunner' object has no attribute 'get_policy'")

        def get_module(self, policy_id=None):
            del policy_id
            return DummyModule()

    original_torch = sys.modules.get("torch")
    original_ray = sys.modules.get("ray")
    original_ray_rllib = sys.modules.get("ray.rllib")
    original_ray_rllib_core = sys.modules.get("ray.rllib.core")
    original_ray_rllib_core_columns = sys.modules.get("ray.rllib.core.columns")
    try:
        sys.modules["torch"] = DummyTorch
        sys.modules["ray"] = SimpleNamespace()
        sys.modules["ray.rllib"] = SimpleNamespace()
        sys.modules["ray.rllib.core"] = SimpleNamespace()
        sys.modules["ray.rllib.core.columns"] = DummyRayColumnsModule

        action = rllib_runner._compute_single_action(DummyAlgo(), {"graph": [1, 2, 3]}, policy_id="tls_1")
    finally:
        for name, original in (
            ("torch", original_torch),
            ("ray", original_ray),
            ("ray.rllib", original_ray_rllib),
            ("ray.rllib.core", original_ray_rllib_core),
            ("ray.rllib.core.columns", original_ray_rllib_core_columns),
        ):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    assert action == 2


def test_compute_single_action_falls_back_to_algo_compute_single_action_without_module():
    class DummyAlgo:
        def __init__(self):
            self.calls = []

        def get_module(self, policy_id=None):
            del policy_id
            raise RuntimeError("module is unavailable")

        def compute_single_action(self, obs, policy_id=None, explore=None):
            self.calls.append(
                {
                    "obs": obs,
                    "policy_id": policy_id,
                    "explore": explore,
                }
            )
            return 3

    algo = DummyAlgo()

    action = rllib_runner._compute_single_action(algo, {"graph": [1, 2, 3]}, policy_id="tls_1")

    assert action == 3
    assert algo.calls == [
        {
            "obs": {"graph": [1, 2, 3]},
            "policy_id": "tls_1",
            "explore": False,
        }
    ]


def test_evaluate_validation_metrics_use_episode_summary_and_average_across_eval_seeds(monkeypatch, tmp_path):
    class DummyEvalEnv:
        possible_agents = ["tls_1"]

        def __init__(self, seed):
            self.seed = seed
            self.closed = False

        def close(self):
            self.closed = True

    episode_summaries = {
        7: {
            "reward/mean": 4.0,
            "reward/max": 5.0,
            "reward/std": 1.0,
            "resco_delay_mean": 10.0,
            "resco_delay_max": 12.0,
            "resco_delay_std": 0.5,
            "resco_wait_mean": 6.0,
            "resco_wait_max": 7.0,
            "resco_wait_std": 0.25,
            "resco_queue_mean": 2.0,
            "resco_queue_max": 4.0,
            "resco_trip_time_mean": 30.0,
            "resco_tripinfo_count": 8.0,
            "system_total_arrived": 11.0,
            "system_total_departed": 12.0,
            "system_total_teleported": 1.0,
            "system_total_emergency_brake": 2.0,
            "system_total_collisions": 0.0,
        },
        8: {
            "reward/mean": 8.0,
            "reward/max": 9.0,
            "reward/std": 3.0,
            "resco_delay_mean": 14.0,
            "resco_delay_max": 16.0,
            "resco_delay_std": 1.5,
            "resco_wait_mean": 10.0,
            "resco_wait_max": 12.0,
            "resco_wait_std": 0.75,
            "resco_queue_mean": 6.0,
            "resco_queue_max": 8.0,
            "resco_trip_time_mean": 40.0,
            "resco_tripinfo_count": 10.0,
            "system_total_arrived": 21.0,
            "system_total_departed": 22.0,
            "system_total_teleported": 3.0,
            "system_total_emergency_brake": 4.0,
            "system_total_collisions": 2.0,
        },
    }

    def fake_build_rllib_parallel_env(cfg, run_dir, seed, pad_spaces):
        del cfg, run_dir, pad_spaces
        return DummyEvalEnv(seed)

    def fake_run_episode_trace(*args, **kwargs):
        del args, kwargs
        return 999.0, {"tls_1": [0, 1, 1]}, {"tls_1": 2}, {"tls_1": []}

    def fake_completed_episode_summary(env):
        return dict(episode_summaries[env.seed])

    def fake_build_summary(env, **kwargs):
        assert env.closed is True
        return {
            "algorithm/kind": kwargs["algorithm_kind"],
            "final/eval/mean_reward": kwargs["eval_mean_reward"],
            "final/eval/std_reward": kwargs["eval_std_reward"],
            "final/resco/avg_delay": float(env.seed),
            "tripinfo/avg_duration": 123.0,
            "warnings/no_finished_trips": False,
        }

    monkeypatch.setattr(rllib_runner, "build_rllib_parallel_env", fake_build_rllib_parallel_env)
    monkeypatch.setattr(rllib_runner, "_run_multi_agent_episode_trace", fake_run_episode_trace)
    monkeypatch.setattr(rllib_runner, "_get_completed_episode_summary", fake_completed_episode_summary)
    monkeypatch.setattr(rllib_runner, "_build_final_eval_summary_row", fake_build_summary)

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(seed=7, eval_episodes=2, eval_seeds=None),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )

    summary = rllib_runner._evaluate(
        cfg,
        tmp_path,
        algo=object(),
        algorithm_kind="ppo",
        logging_cfg=SimpleNamespace(log_final_traffic_metrics=True),
        include_validation_metrics=True,
    )

    assert summary["validation/reward_mean"] == 6.0
    assert summary["validation/reward_max"] == 7.0
    assert summary["validation/reward_std"] == 2.0
    assert summary["validation/resco_delay_mean"] == 12.0
    assert summary["validation/resco_delay_max"] == 14.0
    assert summary["validation/resco_delay_std"] == 1.0
    assert summary["validation/resco_wait_mean"] == 8.0
    assert summary["validation/resco_wait_max"] == 9.5
    assert summary["validation/resco_wait_std"] == 0.5
    assert summary["validation/resco_queue_mean"] == 4.0
    assert summary["validation/resco_queue_max"] == 6.0
    assert summary["validation/resco_trip_time_mean"] == 35.0
    assert summary["validation/resco_tripinfo_count"] == 9.0
    assert summary["validation/efficiency_total_arrived"] == 16.0
    assert summary["validation/efficiency_total_departed"] == 17.0
    assert summary["validation/safety_total_teleported"] == 2.0
    assert summary["validation/safety_total_emergency_brake"] == 3.0
    assert summary["validation/safety_total_collisions"] == 1.0
    assert "validation/tripinfo/avg_duration" not in summary
    assert "validation/eval/mean_reward" not in summary
    assert summary["final/eval/mean_reward"] == 999.0


def test_action_distribution_rows_sum_to_one_and_respect_sliding_window():
    rows = rllib_runner._action_distribution_rows([0, 1, 1, 0], num_actions=2, window_size=2)

    assert [row["step"] for row in rows] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0]["action_0"] == 1.0
    assert rows[0]["action_1"] == 0.0
    assert rows[1]["action_0"] == 0.5
    assert rows[1]["action_1"] == 0.5
    assert rows[2]["action_0"] == 0.0
    assert rows[2]["action_1"] == 1.0
    assert rows[3]["action_0"] == 0.5
    assert rows[3]["action_1"] == 0.5
    assert all(abs((row["action_0"] + row["action_1"]) - 1.0) <= 1e-9 for row in rows)


def test_action_distribution_rows_handle_short_episode_window():
    rows = rllib_runner._action_distribution_rows([2, 2], num_actions=3, window_size=50)

    assert len(rows) == 2
    assert rows[0]["action_2"] == 1.0
    assert rows[1]["action_2"] == 1.0
    assert all(abs(sum(row[f"action_{index}"] for index in range(3)) - 1.0) <= 1e-9 for row in rows)


def test_run_multi_agent_episode_trace_upgrades_action_count_from_phase_queue_snapshot(monkeypatch):
    class DummyEnv:
        possible_agents = ["tls_1"]

        def __init__(self):
            self.step_count = 0

        def reset(self, seed=None):
            self.step_count = 0
            return {"tls_1": {"obs": 1}}, {}

        def step(self, actions):
            del actions
            self.step_count += 1
            done = self.step_count >= 1
            return (
                {"tls_1": {"obs": 1}},
                {"tls_1": 0.0},
                {"tls_1": done, "__all__": done},
                {"tls_1": False, "__all__": False},
                {},
            )

    monkeypatch.setattr(rllib_runner, "_compute_single_action", lambda *args, **kwargs: 0)
    monkeypatch.setattr(rllib_runner, "_action_space_size", lambda env, agent_id: 3)
    monkeypatch.setattr(
        rllib_runner,
        "_collect_phase_queue_snapshot",
        lambda env, agent_ids: {
            "tls_1": {"active_phase": 0, "phase_queues": [5, 2, 0, 0]},
        },
    )

    _, action_traces, action_space_sizes, phase_queue_traces = rllib_runner._run_multi_agent_episode_trace(
        object(),
        DummyEnv(),
        seed=7,
        policy_mode="independent",
    )

    assert action_traces == {"tls_1": [0]}
    assert action_space_sizes == {"tls_1": 4}
    assert phase_queue_traces == {
        "tls_1": [
            {"step": 1.0, "active_phase": 0, "phase_queues": [5, 2, 0, 0]},
        ]
    }


def test_average_action_distribution_rows_aligns_steps_across_seeds():
    averaged = rllib_runner._average_action_distribution_rows(
        [
            [
                {"step": 1.0, "action_0": 1.0, "action_1": 0.0},
                {"step": 2.0, "action_0": 0.5, "action_1": 0.5},
            ],
            [
                {"step": 1.0, "action_0": 0.0, "action_1": 1.0},
            ],
        ],
        num_actions=2,
    )

    assert averaged == [
        {"step": 1.0, "action_0": 0.5, "action_1": 0.5},
        {"step": 2.0, "action_0": 0.5, "action_1": 0.5},
    ]


def test_build_validation_action_plot_rows_averages_per_seed_traces_and_caps_agents():
    rows_by_agent = rllib_runner._build_validation_action_plot_rows(
        [
            {"tls_a": [0, 1, 1], "tls_b": [1, 1, 1]},
            {"tls_a": [1, 1, 0], "tls_b": [0, 0, 0]},
        ],
        [
            {"tls_a": 2, "tls_b": 2},
            {"tls_a": 2, "tls_b": 2},
        ],
        window_size=2,
        max_agents=1,
    )

    assert list(rows_by_agent.keys()) == ["tls_a"]
    tls_rows = rows_by_agent["tls_a"]
    assert [row["step"] for row in tls_rows] == [1.0, 2.0, 3.0]
    assert all(abs(sum(row[f"action_{index}"] for index in range(2)) - 1.0) <= 1e-9 for row in tls_rows)


def test_validation_action_window_steps_uses_one_minute_of_env_time():
    cfg = SimpleNamespace(env=SimpleNamespace(kwargs=SimpleNamespace(delta_time=5)))
    assert rllib_runner._validation_action_window_steps(cfg) == 12

    cfg = SimpleNamespace(env=SimpleNamespace(kwargs=SimpleNamespace(delta_time=10)))
    assert rllib_runner._validation_action_window_steps(cfg) == 6


def test_build_validation_action_timeline_rows_uses_majority_vote_per_step_and_caps_agents():
    timeline_by_agent = rllib_runner._build_validation_action_timeline_rows(
        [
            {"tls_a": [0, 1, 1], "tls_b": [1, 1, 1]},
            {"tls_a": [1, 1, 0], "tls_b": [0, 0, 0]},
            {"tls_a": [1, 0, 0], "tls_b": [0, 1, 0]},
        ],
        [
            {"tls_a": 2, "tls_b": 2},
            {"tls_a": 2, "tls_b": 2},
            {"tls_a": 2, "tls_b": 2},
        ],
        max_agents=1,
    )

    assert timeline_by_agent == {"tls_a": [1, 1, 0]}


def test_build_validation_phase_queue_rows_averages_counts_and_keeps_active_phase():
    rows_by_agent = rllib_runner._build_validation_phase_queue_rows(
        [
            {
                "tls_a": [
                    {"step": 1.0, "active_phase": 0, "phase_queues": [4, 1]},
                    {"step": 2.0, "active_phase": 1, "phase_queues": [2, 3]},
                ]
            },
            {
                "tls_a": [
                    {"step": 1.0, "active_phase": 0, "phase_queues": [6, 3]},
                    {"step": 2.0, "active_phase": 1, "phase_queues": [4, 5]},
                ]
            },
        ]
    )

    assert rows_by_agent == {
        "tls_a": [
            {"step": 1.0, "active_phase": 0.0, "phase_0": 5.0, "phase_1": 2.0},
            {"step": 2.0, "active_phase": 1.0, "phase_0": 3.0, "phase_1": 4.0},
        ]
    }


def test_build_validation_phase_queue_rows_keeps_zero_only_phase_columns():
    rows_by_agent = rllib_runner._build_validation_phase_queue_rows(
        [
            {
                "tls_a": [
                    {"step": 1.0, "active_phase": 0, "phase_queues": [4, 1, 0]},
                ]
            },
            {
                "tls_a": [
                    {"step": 1.0, "active_phase": 0, "phase_queues": [6, 3, 0]},
                ]
            },
        ]
    )

    assert rows_by_agent == {
        "tls_a": [
            {"step": 1.0, "active_phase": 0.0, "phase_0": 5.0, "phase_1": 2.0, "phase_2": 0.0},
        ]
    }


def test_render_validation_action_plot_image_returns_chart_image():
    image = rllib_runner._render_validation_action_plot_image(
        "tls_1",
        [
            {"step": 1.0, "action_0": 1.0, "action_1": 0.0},
            {"step": 2.0, "action_0": 0.5, "action_1": 0.5},
            {"step": 3.0, "action_0": 0.0, "action_1": 1.0},
        ],
    )

    assert image.size == (1040, 560)


def test_render_validation_action_timeline_image_returns_chart_image():
    image = rllib_runner._render_validation_action_timeline_image(
        "tls_1",
        [0, 0, 1, 2, 2, 1],
        decision_seconds=5,
        num_actions=3,
    )

    assert image.size == (1040, 420)


def test_render_validation_phase_queue_image_returns_chart_image():
    image = rllib_runner._render_validation_phase_queue_image(
        "tls_1",
        [
            {"step": 1.0, "active_phase": 0.0, "phase_0": 4.0, "phase_1": 1.0},
            {"step": 2.0, "active_phase": 0.0, "phase_0": 5.0, "phase_1": 2.0},
            {"step": 3.0, "active_phase": 1.0, "phase_0": 2.0, "phase_1": 6.0},
        ],
        decision_seconds=5,
    )

    assert image.size == (1040, 520)


def test_log_validation_action_plot_images_emits_one_image_per_agent(monkeypatch):
    class DummyImage:
        def __init__(self, image, caption=None):
            self.image = image
            self.caption = caption

    class DummyWandb:
        Image = DummyImage

    class DummyRun:
        def __init__(self):
            self.calls = []

        def log(self, payload):
            self.calls.append(payload)

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)
    run = DummyRun()

    rllib_runner._log_validation_action_plot_images(
        run,
        {
            "tls_1": [
                {"step": 1.0, "action_0": 1.0, "action_1": 0.0},
                {"step": 2.0, "action_0": 0.5, "action_1": 0.5},
            ],
            "tls_2": [{"step": 1.0, "action_0": 0.0, "action_1": 1.0}],
        },
        {
            "tls_1": [0, 1, 1, 0],
            "tls_2": [1, 1, 0],
        },
        {
            "tls_1": [
                {"step": 1.0, "active_phase": 0.0, "phase_0": 4.0, "phase_1": 2.0},
            ],
            "tls_2": [
                {"step": 1.0, "active_phase": 1.0, "phase_0": 1.0, "phase_1": 3.0},
            ],
        },
        pass_index=3,
        env_step=120,
        episode_index=18,
        decision_seconds=5,
    )

    assert len(run.calls) == 2
    assert run.calls[0]["validation/rollout_index"] == 18.0
    assert run.calls[0]["validation/episode_index"] == 18.0
    assert run.calls[0]["validation/pass_index"] == 3.0
    assert run.calls[0]["validation/env_step"] == 120.0
    assert isinstance(run.calls[0]["validation/actions_share/tls_1"], DummyImage)
    assert run.calls[0]["validation/actions_share/tls_1"].image.size == (1040, 560)
    assert isinstance(run.calls[0]["validation/actions_timeline/tls_1"], DummyImage)
    assert run.calls[0]["validation/actions_timeline/tls_1"].image.size == (1040, 420)
    assert isinstance(run.calls[0]["validation/phase_queue/tls_1"], DummyImage)
    assert run.calls[0]["validation/phase_queue/tls_1"].image.size == (1040, 520)
    assert "validation pass 3" in run.calls[0]["validation/actions_share/tls_1"].caption
    assert isinstance(run.calls[1]["validation/actions_share/tls_2"], DummyImage)
    assert isinstance(run.calls[1]["validation/actions_timeline/tls_2"], DummyImage)
    assert isinstance(run.calls[1]["validation/phase_queue/tls_2"], DummyImage)


def test_log_validation_action_plot_images_passes_full_phase_count_to_timeline_renderer(monkeypatch):
    captured = {}

    class DummyImage:
        def __init__(self, image, caption=None):
            self.image = image
            self.caption = caption

    class DummyWandb:
        Image = DummyImage

    class DummyRun:
        def __init__(self):
            self.calls = []

        def log(self, payload):
            self.calls.append(payload)

    def fake_render_timeline(agent_id, actions, *, decision_seconds, num_actions=None, width=1040, height=420):
        captured["agent_id"] = agent_id
        captured["actions"] = list(actions)
        captured["decision_seconds"] = decision_seconds
        captured["num_actions"] = num_actions
        captured["width"] = width
        captured["height"] = height
        return object()

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)
    monkeypatch.setattr(rllib_runner, "_render_validation_action_plot_image", lambda *args, **kwargs: object())
    monkeypatch.setattr(rllib_runner, "_render_validation_action_timeline_image", fake_render_timeline)
    monkeypatch.setattr(rllib_runner, "_render_validation_phase_queue_image", lambda *args, **kwargs: object())
    run = DummyRun()

    rllib_runner._log_validation_action_plot_images(
        run,
        {
            "tls_1": [
                {"step": 1.0, "action_0": 1.0, "action_1": 0.0, "action_2": 0.0},
                {"step": 2.0, "action_0": 0.5, "action_1": 0.5, "action_2": 0.0},
            ],
        },
        {
            "tls_1": [0, 1, 1, 0],
        },
        {
            "tls_1": [
                {"step": 1.0, "active_phase": 0.0, "phase_0": 4.0, "phase_1": 2.0, "phase_2": 0.0},
            ],
        },
        pass_index=3,
        env_step=120,
        episode_index=18,
        decision_seconds=5,
    )

    assert len(run.calls) == 1
    assert captured == {
        "agent_id": "tls_1",
        "actions": [0, 1, 1, 0],
        "decision_seconds": 5,
        "num_actions": 3,
        "width": 1040,
        "height": 420,
    }


def test_extract_validation_seed_artifacts_parses_tripinfo_and_removes_temp_file(tmp_path):
    tripinfo_path = tmp_path / "tripinfo.xml"
    tripinfo_path.write_text(
        """
<routes>
  <tripinfo id="veh_1" depart="0" arrival="20" duration="20" waitingTime="4" timeLoss="3" departDelay="1" vaporized="" />
  <tripinfo id="veh_2" depart="5" arrival="-1" duration="15" waitingTime="9" timeLoss="12" departDelay="0" vaporized="" />
  <tripinfo id="veh_3" depart="-1" arrival="-1" duration="0" waitingTime="0" timeLoss="0" departDelay="0" />
</routes>
""".strip(),
        encoding="utf-8",
    )

    artifact = rllib_runner._extract_validation_seed_artifacts(
        seed=7,
        tripinfo_path=tripinfo_path,
        episode_summary={"reward/mean": 1.0},
        action_traces={"tls_1": [0, 1]},
        action_space_sizes={"tls_1": 2},
        phase_queue_traces={"tls_1": [{"step": 1.0, "active_phase": 0, "phase_queues": [2, 1]}]},
        remove_tripinfo_after_parse=True,
    )

    assert artifact.tripinfo.wait_values == [4.0]
    assert artifact.tripinfo.delay_values == [4.0]
    assert artifact.tripinfo.finished_count == 1
    assert artifact.tripinfo.unfinished_count == 2
    assert artifact.tripinfo.total_count == 3
    assert tripinfo_path.exists() is False


def test_extract_validation_seed_artifacts_keeps_tripinfo_when_requested(tmp_path):
    tripinfo_path = tmp_path / "tripinfo.xml"
    tripinfo_path.write_text(
        """
<routes>
  <tripinfo id="veh_1" depart="0" arrival="20" duration="20" waitingTime="4" timeLoss="3" departDelay="1" vaporized="" />
</routes>
""".strip(),
        encoding="utf-8",
    )

    artifact = rllib_runner._extract_validation_seed_artifacts(
        seed=7,
        tripinfo_path=tripinfo_path,
        episode_summary={"reward/mean": 1.0},
        action_traces={},
        action_space_sizes={},
        phase_queue_traces={},
        remove_tripinfo_after_parse=False,
    )

    assert artifact.tripinfo.finished_count == 1
    assert tripinfo_path.exists() is True


def test_aggregate_validation_tripinfo_distributions_keeps_empty_seeds_and_pools_completed_values():
    aggregated = rllib_runner._aggregate_validation_tripinfo_distributions(
        [
            rllib_runner.ValidationSeedArtifacts(
                seed=1,
                episode_summary={},
                action_traces={},
                action_space_sizes={},
                phase_queue_traces={},
                tripinfo=rllib_runner.TripinfoDistributionArtifact(
                    wait_values=[2.0, 6.0],
                    delay_values=[4.0, 8.0],
                    finished_count=2,
                    unfinished_count=1,
                    total_count=3,
                ),
            ),
            rllib_runner.ValidationSeedArtifacts(
                seed=2,
                episode_summary={},
                action_traces={},
                action_space_sizes={},
                phase_queue_traces={},
                tripinfo=rllib_runner.TripinfoDistributionArtifact(
                    wait_values=[],
                    delay_values=[],
                    finished_count=0,
                    unfinished_count=2,
                    total_count=2,
                ),
            ),
        ]
    )

    assert aggregated["waiting_time"] == [[2.0, 6.0], []]
    assert aggregated["delay"] == [[4.0, 8.0], []]
    assert aggregated["pooled_waiting_time"] == [2.0, 6.0]
    assert aggregated["pooled_delay"] == [4.0, 8.0]
    assert aggregated["total_seeds"] == 2
    assert aggregated["seeds_with_completed_trips"] == 1
    assert aggregated["seeds_without_completed_trips"] == 1
    assert aggregated["total_completed_trips"] == 2
    assert aggregated["total_unfinished_trips"] == 3
    assert aggregated["total_trips"] == 5


def test_log_validation_tripinfo_distribution_images_emits_network_level_media(monkeypatch):
    class DummyImage:
        def __init__(self, image, caption=None):
            self.image = image
            self.caption = caption

    class DummyWandb:
        Image = DummyImage

    class DummyRun:
        def __init__(self):
            self.calls = []

        def log(self, payload):
            self.calls.append(payload)

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)
    run = DummyRun()

    rllib_runner._log_validation_tripinfo_distribution_images(
        run,
        {
            "waiting_time": [[1.0, 2.0, 4.0], []],
            "delay": [[3.0, 6.0], []],
            "pooled_waiting_time": [1.0, 2.0, 4.0],
            "pooled_delay": [3.0, 6.0],
            "total_seeds": 2,
            "seeds_with_completed_trips": 1,
            "seeds_without_completed_trips": 1,
            "total_completed_trips": 3,
            "total_unfinished_trips": 2,
            "total_trips": 5,
        },
        pass_index=4,
        env_step=240,
        episode_index=21,
    )

    assert len(run.calls) == 1
    assert run.calls[0]["validation/rollout_index"] == 21.0
    assert run.calls[0]["validation/episode_index"] == 21.0
    assert run.calls[0]["validation/pass_index"] == 4.0
    assert run.calls[0]["validation/env_step"] == 240.0
    assert isinstance(run.calls[0]["validation/tripinfo_wait_distribution"], DummyImage)
    assert run.calls[0]["validation/tripinfo_wait_distribution"].image.size == (1040, 460)
    assert isinstance(run.calls[0]["validation/tripinfo_delay_distribution"], DummyImage)
    assert "1/2 seeds with completed trips" in run.calls[0]["validation/tripinfo_wait_distribution"].caption


def test_log_validation_tripinfo_distribution_images_skips_when_no_completed_trips(monkeypatch):
    class DummyWandb:
        Image = object

    class DummyRun:
        def __init__(self):
            self.calls = []

        def log(self, payload):
            self.calls.append(payload)

    monkeypatch.setitem(sys.modules, "wandb", DummyWandb)
    run = DummyRun()

    rllib_runner._log_validation_tripinfo_distribution_images(
        run,
        {
            "waiting_time": [[], []],
            "delay": [[], []],
            "pooled_waiting_time": [],
            "pooled_delay": [],
            "total_seeds": 2,
            "seeds_with_completed_trips": 0,
            "seeds_without_completed_trips": 2,
            "total_completed_trips": 0,
            "total_unfinished_trips": 4,
            "total_trips": 4,
        },
        pass_index=4,
        env_step=240,
        episode_index=21,
    )

    assert run.calls == []


def test_evaluate_with_details_returns_validation_action_plot_rows(monkeypatch, tmp_path):
    class DummyEvalEnv:
        possible_agents = ["tls_1"]

        def __init__(self, seed):
            self.seed = seed
            self.closed = False

        def close(self):
            self.closed = True

    def fake_build_rllib_parallel_env(cfg, run_dir, seed, pad_spaces):
        del cfg, run_dir, pad_spaces
        return DummyEvalEnv(seed)

    action_traces_by_seed = {
        7: (
            10.0,
            {"tls_1": [0, 1, 1]},
            {"tls_1": 2},
            {"tls_1": [{"step": 1.0, "active_phase": 0, "phase_queues": [3, 1]}]},
        ),
        8: (
            12.0,
            {"tls_1": [1, 1, 0]},
            {"tls_1": 2},
            {"tls_1": [{"step": 1.0, "active_phase": 0, "phase_queues": [5, 2]}]},
        ),
    }

    def fake_run_episode_trace(algo, env, seed, *, policy_mode):
        del algo, policy_mode
        return action_traces_by_seed[seed]

    def fake_completed_episode_summary(env):
        return {
            "reward/mean": float(env.seed),
            "reward/max": float(env.seed),
            "reward/std": 0.0,
        }

    def fake_build_summary(env, **kwargs):
        return {
            "algorithm/kind": kwargs["algorithm_kind"],
            "final/eval/mean_reward": kwargs["eval_mean_reward"],
            "final/eval/std_reward": kwargs["eval_std_reward"],
        }

    monkeypatch.setattr(rllib_runner, "build_rllib_parallel_env", fake_build_rllib_parallel_env)
    monkeypatch.setattr(rllib_runner, "_run_multi_agent_episode_trace", fake_run_episode_trace)
    monkeypatch.setattr(rllib_runner, "_get_completed_episode_summary", fake_completed_episode_summary)
    monkeypatch.setattr(rllib_runner, "_build_final_eval_summary_row", fake_build_summary)

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(seed=7, eval_episodes=2, eval_seeds=None),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )
    logging_cfg = SimpleNamespace(
        log_final_traffic_metrics=True,
        validation_action_plot_max_agents=None,
    )

    summary, seed_rows, plot_rows, timeline_rows, phase_queue_rows, tripinfo_distributions = rllib_runner._evaluate_with_details(
        cfg,
        tmp_path,
        algo=object(),
        algorithm_kind="ppo",
        logging_cfg=logging_cfg,
        include_validation_metrics=True,
    )

    assert summary["validation/reward_mean"] == 7.5
    assert len(seed_rows) == 2
    assert list(plot_rows.keys()) == ["tls_1"]
    assert [row["step"] for row in plot_rows["tls_1"]] == [1.0, 2.0, 3.0]
    assert all(abs((row["action_0"] + row["action_1"]) - 1.0) <= 1e-9 for row in plot_rows["tls_1"])
    assert timeline_rows == {"tls_1": [0, 1, 0]}
    assert phase_queue_rows == {"tls_1": [{"step": 1.0, "active_phase": 0.0, "phase_0": 4.0, "phase_1": 1.5}]}
    assert tripinfo_distributions == {
        "waiting_time": [[], []],
        "delay": [[], []],
        "pooled_waiting_time": [],
        "pooled_delay": [],
        "total_seeds": 2,
        "seeds_with_completed_trips": 0,
        "seeds_without_completed_trips": 2,
        "total_completed_trips": 0,
        "total_unfinished_trips": 0,
        "total_trips": 0,
    }


def test_best_validation_checkpoint_retention_writes_full_metadata_and_keeps_top_three(tmp_path):
    class FakeAlgo:
        def __init__(self):
            self.saved_metric = None

        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "checkpoint.json").write_text(
                json.dumps({"metric": self.saved_metric}),
                encoding="utf-8",
            )
            return str(checkpoint_dir)

    algo = FakeAlgo()
    logging_cfg = SimpleNamespace(
        save_best_validation_checkpoints=True,
        best_validation_checkpoint_count=3,
        best_validation_metric="validation/resco_delay_mean",
    )
    state = rllib_runner._init_best_validation_checkpoint_state(tmp_path, "ppo", logging_cfg)

    candidates = [
        (12.0, 10.0),
        (9.0, 20.0),
        (11.0, 30.0),
        (8.0, 40.0),
        (8.0, 50.0),
    ]
    for metric_value, env_step in candidates:
        algo.saved_metric = metric_value
        rllib_runner._consider_best_validation_checkpoint(
            state,
            algo,
            validation_metrics={
                "validation/resco_delay_mean": metric_value,
                "validation/env_step": env_step,
            },
            evaluation_summary={
                "validation/resco_delay_mean": metric_value,
                "validation/eval/episode": 2.0,
                "final/eval/mean_reward": metric_value,
            },
            evaluation_seed_rows=[
                {
                    "eval/seed": 1.0,
                    "validation/resco_delay_mean": metric_value + 0.5,
                    "final/eval/mean_reward": metric_value + 1.0,
                },
                {
                    "eval/seed": 2.0,
                    "validation/resco_delay_mean": metric_value - 0.5,
                    "final/eval/mean_reward": metric_value - 1.0,
                },
            ],
        )

    retained_metrics = [entry["metric_value"] for entry in state["retained"]]
    assert retained_metrics == [8.0, 9.0, 11.0]
    metadata = json.loads(state["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["metric_name"] == "validation/resco_delay_mean"
    assert [item["metric_value"] for item in metadata["retained"]] == [8.0, 9.0, 11.0]
    assert metadata["retained"][0]["validation_metrics"]["validation/resco_delay_mean"] == 8.0
    assert metadata["retained"][0]["evaluation_summary"]["validation/resco_delay_mean"] == 8.0
    assert len(metadata["retained"][0]["evaluation_seed_rows"]) == 2
    assert all(
        not key.startswith("final/")
        for key in metadata["retained"][0]["evaluation_summary"].keys()
    )
    assert all(
        not key.startswith("final/")
        for row in metadata["retained"][0]["evaluation_seed_rows"]
        for key in row.keys()
    )
    assert metadata["retained"][0]["rank"] == 1
    assert metadata["retained"][1]["rank"] == 2
    assert metadata["retained"][2]["rank"] == 3
    assert not (state["base_dir"] / "validation_pass_0001__step_0000010__delay_12.000000").exists()
    assert (state["base_dir"] / "validation_pass_0004__step_0000040__delay_8.000000").exists()
    assert not (state["base_dir"] / "validation_pass_0005__step_0000050__delay_8.000000").exists()


def test_best_validation_checkpoint_skips_missing_or_non_finite_metric(tmp_path):
    class FakeAlgo:
        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return str(checkpoint_dir)

    logging_cfg = SimpleNamespace(
        save_best_validation_checkpoints=True,
        best_validation_checkpoint_count=3,
        best_validation_metric="validation/resco_delay_mean",
    )
    state = rllib_runner._init_best_validation_checkpoint_state(tmp_path, "ppo", logging_cfg)
    algo = FakeAlgo()

    assert (
        rllib_runner._consider_best_validation_checkpoint(
            state,
            algo,
            validation_metrics={"validation/env_step": 10.0},
            evaluation_summary={},
            evaluation_seed_rows=[],
        )
        is None
    )
    assert (
        rllib_runner._consider_best_validation_checkpoint(
            state,
            algo,
            validation_metrics={
                "validation/resco_delay_mean": float("nan"),
                "validation/env_step": 20.0,
            },
            evaluation_summary={},
            evaluation_seed_rows=[],
        )
        is None
    )
    assert state["retained"] == []
    assert not state["metadata_path"].exists()


def test_consider_best_metrics_row_prefers_lower_delay_and_keeps_full_row():
    current = None
    current = rllib_runner._consider_best_metrics_row(
        current,
        {"train/resco_delay_mean": 12.0, "train/reward_mean": 3.0},
        metric_name="train/resco_delay_mean",
    )
    current = rllib_runner._consider_best_metrics_row(
        current,
        {"train/resco_delay_mean": 9.0, "train/reward_mean": 7.5, "train/custom_metric": 11.0},
        metric_name="train/resco_delay_mean",
    )
    current = rllib_runner._consider_best_metrics_row(
        current,
        {"train/resco_delay_mean": 9.0, "train/reward_mean": 99.0},
        metric_name="train/resco_delay_mean",
    )

    assert current == {
        "train/resco_delay_mean": 9.0,
        "train/reward_mean": 7.5,
        "train/custom_metric": 11.0,
    }


def test_consider_best_metrics_row_skips_missing_or_non_finite_metric():
    current = rllib_runner._consider_best_metrics_row(
        None,
        {"validation/resco_delay_mean": 8.5, "validation/reward_mean": 2.0},
        metric_name="validation/resco_delay_mean",
    )

    unchanged = rllib_runner._consider_best_metrics_row(
        current,
        {"validation/reward_mean": 4.0},
        metric_name="validation/resco_delay_mean",
    )
    unchanged = rllib_runner._consider_best_metrics_row(
        unchanged,
        {"validation/resco_delay_mean": float("nan"), "validation/reward_mean": 6.0},
        metric_name="validation/resco_delay_mean",
    )

    assert unchanged == current


def test_update_wandb_best_summary_prefixes_train_and_validation_metrics_and_cleans_stale_keys():
    class DummyRun:
        def __init__(self):
            self.summary = {
                "best_train/obsolete": 99.0,
                "best_validation/obsolete": 101.0,
                "validation/resco_delay_mean": 7.0,
            }

    run = DummyRun()

    rllib_runner._update_wandb_best_summary(
        run,
        {
            "best_train": {
                "train/resco_delay_mean": 5.0,
                "train/reward_mean": 9.0,
                "debug/ignored": 1.0,
            },
            "best_validation": {
                "validation/resco_delay_mean": 4.0,
                "validation/reward_mean": 8.0,
                "train/ignored": 2.0,
            },
        },
    )

    assert run.summary == {
        "validation/resco_delay_mean": 7.0,
        "best_train/resco_delay_mean": 5.0,
        "best_train/reward_mean": 9.0,
        "best_validation/resco_delay_mean": 4.0,
        "best_validation/reward_mean": 8.0,
    }


def test_restore_checkpoint_loads_saved_weights_and_reproduces_metric(tmp_path):
    class FakeAlgo:
        def __init__(self, metric_value=0.0):
            self.metric_value = metric_value

        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "checkpoint.json").write_text(
                json.dumps({"metric_value": self.metric_value}),
                encoding="utf-8",
            )
            return str(checkpoint_dir)

        def restore_from_path(self, path):
            payload = json.loads((Path(path) / "checkpoint.json").read_text(encoding="utf-8"))
            self.metric_value = float(payload["metric_value"])

    logging_cfg = SimpleNamespace(
        save_best_validation_checkpoints=True,
        best_validation_checkpoint_count=3,
        best_validation_metric="validation/resco_delay_mean",
    )
    state = rllib_runner._init_best_validation_checkpoint_state(tmp_path, "ppo", logging_cfg)
    saved_algo = FakeAlgo(metric_value=7.25)
    entry = rllib_runner._consider_best_validation_checkpoint(
        state,
        saved_algo,
        validation_metrics={
            "validation/resco_delay_mean": 7.25,
            "validation/env_step": 100.0,
        },
        evaluation_summary={"validation/resco_delay_mean": 7.25},
        evaluation_seed_rows=[{"eval/seed": 1.0, "validation/resco_delay_mean": 7.25}],
    )

    restored_algo = FakeAlgo(metric_value=0.0)
    rllib_runner._restore_checkpoint(restored_algo, entry["checkpoint_path"])

    assert abs(restored_algo.metric_value - entry["metric_value"]) <= 1e-9


def test_training_episode_row_uses_episode_cadence_and_resco_metrics():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(train_log_freq_episodes=2, train_log_freq_steps=1, log_freq=1000, trace_mode="training")
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/env_step": 40.0,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
    }
    episode_summary = {
        "episode/index": 2.0,
        "reward/mean": 4.5,
        "reward/max": 6.0,
        "reward/std": 1.5,
        "reward/agent/tls_1": 3.0,
        "reward/agent/tls_2": 6.0,
        "resco_delay_mean": 12.0,
        "resco_delay_max": 14.0,
        "resco_delay_std": 1.5,
        "resco_wait_mean": 7.0,
        "resco_wait_max": 9.0,
        "resco_wait_std": 0.5,
        "resco_queue_mean": 3.0,
        "resco_queue_max": 9.0,
        "resco_trip_time_mean": 33.0,
        "resco_tripinfo_count": 4.0,
        "system_total_arrived": 11.0,
        "system_total_departed": 12.0,
        "system_total_teleported": 1.0,
        "system_total_running": 8.0,
        "system_mean_queued": 2.0,
    }

    assert should_log_training_episode(1, cfg, last_logged_episode=0) is False
    assert should_log_training_episode(2, cfg, last_logged_episode=0) is True

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo", cfg=cfg)

    assert row["train/rollout_index"] == 2.0
    assert row["train/episode_index"] == 2.0
    assert row["train/env_step"] == 40.0
    assert row["train/reward_mean"] == 4.5
    assert row["train/reward_max"] == 6.0
    assert row["train/reward_std"] == 1.5
    assert row["train/resco_delay_mean"] == 12.0
    assert row["train/resco_delay_max"] == 14.0
    assert row["train/resco_delay_std"] == 1.5
    assert row["train/resco_wait_mean"] == 7.0
    assert row["train/resco_wait_max"] == 9.0
    assert row["train/resco_wait_std"] == 0.5
    assert row["train/resco_queue_mean"] == 3.0
    assert row["train/resco_queue_max"] == 9.0
    assert row["train/resco_trip_time_mean"] == 33.0
    assert row["train/resco_tripinfo_count"] == 4.0
    assert row["train/efficiency_total_arrived"] == 11.0
    assert row["train/efficiency_total_departed"] == 12.0
    assert row["train/safety_total_teleported"] == 1.0
    assert "train/efficiency_total_running" not in row
    assert "train/efficiency_mean_queued" not in row
    assert row["debug/efficiency_total_running"] == 8.0
    assert row["debug/reward/tls_1"] == 3.0
    assert row["debug/reward/tls_2"] == 6.0
    assert "debug/episode_return_mean" not in row


def test_rllib_training_episode_emission_logs_every_summary_episode():
    cfg = SimpleNamespace(
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="training")
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/episodes_total": 2.0,
        "train/iteration": 7,
        "train/rllib/rollout_jump": 2.0,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [
            {"episode/index": 1.0, "resco_wait_mean": 5.0},
            {"episode/index": 2.0, "resco_wait_mean": 6.0},
        ],
        cfg,
        algorithm_kind="ppo",
        last_logged_episode=0,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 2
    assert [step for step, _ in emitted] == [1, 2]
    assert [row["train/rollout_index"] for _, row in emitted] == [1.0, 2.0]
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0]
    assert all(row["train/rllib/rollout_jump"] == 2.0 for _, row in emitted)
    assert emitted[0][1]["train/resco_wait_mean"] == 5.0
    assert emitted[1][1]["train/resco_wait_mean"] == 6.0


def test_rllib_training_episode_emission_clamps_env_local_index_to_rllib_rollout_count():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episodes=20, episode_seconds=100),
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="debug"),
    )
    metrics = {
        "algorithm/kind": "dqn",
        "train/episodes_total": 13.0,
        "train/env_step": 130.0,
        "train/dqn/replay/num_added": 999.0,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [{"episode/index": 30.0, "resco_wait_mean": 9.0}],
        cfg,
        algorithm_kind="dqn",
        last_logged_episode=12,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 13
    assert [step for step, _ in emitted] == [13]
    assert emitted[0][1]["train/rollout_index"] == 13.0
    assert emitted[0][1]["train/episode_index"] == 13.0
    assert emitted[0][1]["debug/env_episode_index"] == 30.0
    assert emitted[0][1]["train/resco_wait_mean"] == 9.0


def test_reset_only_episode_summaries_are_not_logged_as_zero_metrics():
    class DummyEnv:
        completed_episode_summaries = [
            {
                "episode/index": 1.0,
                "episode/elapsed_seconds": 0.0,
                "resco_wait_mean": 0.0,
                "resco_queue_mean": 0.0,
                "tripinfo/parse_pending": 0.0,
            },
            {
                "episode/index": 2.0,
                "episode/elapsed_seconds": 3600.0,
                "resco_wait_mean": 6.0,
                "resco_queue_mean": 2.5,
                "tripinfo/parse_pending": 0.0,
            },
        ]
        last_episode_summary = {}
        sumo = None

    summaries = _completed_episode_summary_history(DummyEnv())

    assert [summary["episode/index"] for summary in summaries] == [2.0]


def test_rllib_training_episode_emission_falls_back_to_completed_episode_counters():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(episodes=3, episode_seconds=100),
        logging=SimpleNamespace(train_log_freq_episodes=1, train_log_freq_steps=1, log_freq=1000, trace_mode="training"),
    )
    metrics = {
        "algorithm/kind": "ppo",
        "train/episode_return_mean": 4.5,
        "train/env_steps_sampled": 60.0,
        "train/iteration": 7,
    }
    emitted = []

    last_logged = emit_training_episode_rows(
        metrics,
        [],
        cfg,
        algorithm_kind="ppo",
        last_logged_episode=0,
        emit_metrics=lambda row, step: emitted.append((step, row)),
    )

    assert last_logged == 3
    assert [step for step, _ in emitted] == [1, 2, 3]
    assert [row["train/rollout_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert all(row["train/env_step"] == 60.0 for _, row in emitted)
    assert all("train/reward_mean" not in row for _, row in emitted)


def test_completed_training_episodes_ignores_off_policy_replay_activity_for_rollout_count():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episode_seconds=100))
    metrics = {
        "train/episodes_total": 2.0,
        "train/env_steps_sampled": 500.0,
        "train/dqn/replay/num_added": 5000.0,
        "train/dqn/replay/num_sampled": 9000.0,
    }

    assert completed_training_episodes(metrics, cfg) == 2


def test_trace_mode_defaults_to_training():
    cfg = SimpleNamespace(logging=SimpleNamespace())

    assert trace_mode(cfg) == "training"


def test_debug_trace_mode_moves_internal_metrics_under_debug_namespace():
    cfg = SimpleNamespace(logging=SimpleNamespace(trace_mode="debug"))
    metrics = {
        "train/env_step": 25.0,
        "train/episodes_total": 2.0,
        "train/env_steps_sampled": 25.0,
        "train/episode_return_mean": 4.5,
        "train/episode_return_min": 3.0,
        "train/episode_return_max": 6.0,
        "train/episode_len_mean": 12.0,
        "train/rllib/training_iteration": 3.0,
        "train/rllib/time_total_s": 15.0,
        "train/ppo/learners/default_policy/loss": 1.25,
        "train/ppo/entropy_mean": 0.33,
    }
    episode_summary = {
        "episode/index": 2.0,
        "reward/agent/tls_1": 2.0,
        "system_total_arrived": 8.0,
        "system_total_running": 5.0,
    }

    row = build_training_episode_row(metrics, episode_summary, algorithm_kind="ppo", cfg=cfg)

    assert row["train/efficiency_total_arrived"] == 8.0
    assert row["debug/env_episode_index"] == 2.0
    assert row["debug/reward/tls_1"] == 2.0
    assert row["debug/efficiency_total_running"] == 5.0
    assert row["debug/episode_return_mean"] == 4.5
    assert row["debug/episode_return_min"] == 3.0
    assert row["debug/episode_return_max"] == 6.0
    assert row["debug/episode_len_mean"] == 12.0
    assert row["debug/rllib/training_iteration"] == 3.0
    assert row["debug/rllib/time_total_s"] == 15.0
    assert row["debug/ppo/learners/default_policy/loss"] == 1.25
    assert row["debug/ppo/entropy_mean"] == 0.33
    assert "train/episode_return_mean" not in row


def test_ppo_extract_training_metrics_adds_entropy_mean():
    metrics = extract_ppo_training_metrics(
        {
            "env_runners": {"num_episodes_lifetime": 1.0},
            "learners": {"default_policy": {"curr_entropy": 0.42, "loss": 1.0}},
        },
        iteration=1,
    )

    assert metrics["train/ppo/entropy_mean"] == 0.42


def test_sac_extract_training_metrics_adds_entropy_mean():
    metrics = extract_sac_training_metrics(
        {
            "env_runners": {"num_episodes_lifetime": 1.0},
            "learners": {"default_policy": {"entropy_mean": 0.18, "critic_loss": 2.0}},
        },
        iteration=1,
        algorithm_kind="sac_builtin",
    )

    assert metrics["train/sac/entropy_mean"] == 0.18


def test_rllib_training_budget_uses_experiment_episodes():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, episode_seconds=100))

    assert training_episode_target(cfg) == 3
    assert episode_steps(cfg) == 20
    assert training_should_stop({"train/episodes_total": 2.0, "train/env_steps_sampled": 40.0}, cfg) is False
    assert training_should_stop({"train/episodes_total": 3.0, "train/env_steps_sampled": 60.0}, cfg) is True


def test_rllib_training_budget_falls_back_to_completed_horizons_not_iterations():
    cfg = SimpleNamespace(experiment=SimpleNamespace(episodes=3, episode_seconds=100))

    assert completed_training_episodes({"train/env_steps_sampled": 40.0}, cfg) == 2
    assert training_should_stop({"train/env_steps_sampled": 40.0}, cfg) is False
    assert training_should_stop({"train/env_steps_sampled": 60.0}, cfg) is True


def test_rllib_training_log_frequency_uses_sampled_steps():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_steps=25, log_freq=1000))

    assert train_log_freq_steps(cfg) == 25
    assert should_log_training_metrics({"train/env_steps_sampled": 20.0}, cfg, last_logged_step=0) is False
    assert should_log_training_metrics({"train/env_steps_sampled": 25.0}, cfg, last_logged_step=0) is True


def test_rllib_training_metrics_can_emit_every_sampled_step():
    cfg = SimpleNamespace(logging=SimpleNamespace(train_log_freq_steps=1, log_freq=1000))
    emitted = []

    last_step = emit_training_metrics_by_step(
        {"train/env_steps_sampled": 3.0, "train/iteration": 1},
        cfg,
        last_logged_step=0,
        emit_metrics=lambda row, step: emitted.append((step, row["train/env_step"])),
    )

    assert last_step == 3
    assert emitted == [(1, 1.0), (2, 2.0), (3, 3.0)]


def test_validation_interval_prefers_experiment_override_over_logging_eval_freq():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(validation_interval_steps=25),
        logging=SimpleNamespace(eval_freq=5000),
    )

    assert validation_interval_steps(cfg) == 25


def test_validation_interval_episodes_is_explicit_episode_cadence():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(validation_interval_episodes=5, validation_interval_steps=25),
        logging=SimpleNamespace(eval_freq=5000),
    )

    assert validation_interval_episodes(cfg) == 5


def test_validation_interval_falls_back_to_logging_eval_freq():
    cfg = SimpleNamespace(experiment=SimpleNamespace(), logging=SimpleNamespace(eval_freq=5000))

    assert validation_interval_steps(cfg) == 5000


def test_rllib_training_loop_emits_step_validation_when_due():
    cfg = SimpleNamespace(experiment=SimpleNamespace(validation_interval_episodes=None), logging=SimpleNamespace(eval_freq=10))
    emitted = []

    last_step = emit_validation_if_due(
        {"train/env_step": 9.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append((step, metrics["train/env_step"])),
    )
    assert last_step == 0
    assert emitted == []

    last_step = emit_validation_if_due(
        {"train/env_step": 10.0},
        cfg,
        last_validation_step=last_step,
        validate=lambda metrics, step: emitted.append((step, metrics["train/env_step"])),
    )

    assert last_step == 10
    assert emitted == [(10, 10.0)]


def test_rllib_training_loop_prefers_episode_validation_cadence():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            validation_interval_episodes=5,
            episodes=20,
            episode_seconds=100,
        ),
        logging=SimpleNamespace(eval_freq=10),
    )
    emitted = []

    last_progress = emit_validation_if_due(
        {"train/env_step": 80.0, "train/episodes_total": 4.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append((step, metrics["train/episodes_total"])),
    )
    assert last_progress == 0
    assert emitted == []

    last_progress = emit_validation_if_due(
        {"train/env_step": 100.0, "train/episodes_total": 5.0},
        cfg,
        last_validation_step=last_progress,
        validate=lambda metrics, step: emitted.append((step, metrics["train/episodes_total"])),
    )

    assert last_progress == 5
    assert emitted == [(100, 5.0)]


def test_episode_validation_cadence_uses_derived_env_step_when_dqn_result_has_only_episode_count():
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            validation_interval_episodes=5,
            episodes=20,
            episode_seconds=100,
        ),
        logging=SimpleNamespace(eval_freq=5000),
    )
    emitted = []

    last_progress = emit_validation_if_due(
        {"train/episodes_total": 5.0},
        cfg,
        last_validation_step=0,
        validate=lambda metrics, step: emitted.append(step),
    )

    assert last_progress == 5
    assert emitted == [100]


def test_validation_summary_row_maps_final_metrics_to_validation_namespace():
    row = rllib_runner._validation_summary_row(
        {
            "algorithm/kind": "ppo",
            "validation/reward_mean": 12.0,
            "validation/resco_delay_mean": 4.0,
            "validation/env_step": 3600.0,
            "validation/rollout_index": 99.0,
            "validation/episode_index": 99.0,
            "validation/efficiency_total_arrived": 8.0,
            "validation/safety_total_collisions": 0.0,
            "warnings/missing_tripinfo": 0.0,
            "eval/episode": 2.0,
            "episode/sim_time_abs": 3600.0,
        },
        step=100,
        episode_index=5,
    )

    assert row["algorithm/kind"] == "ppo"
    assert row["validation/env_step"] == 100.0
    assert row["validation/rollout_index"] == 5.0
    assert row["validation/episode_index"] == 5.0
    assert row["validation/reward_mean"] == 12.0
    assert row["validation/resco_delay_mean"] == 4.0
    assert row["validation/efficiency_total_arrived"] == 8.0
    assert row["validation/safety_total_collisions"] == 0.0
    assert row["validation/warnings/missing_tripinfo"] == 0.0
    assert row["validation/eval/episode"] == 2.0
    assert row["validation/episode/sim_time_abs"] == 3600.0


def test_train_rllib_validation_saves_best_checkpoints_and_final_model(monkeypatch, tmp_path):
    ray_init_calls = []

    class DummyRay:
        @staticmethod
        def init(**kwargs):
            ray_init_calls.append(kwargs)
            return None

        @staticmethod
        def shutdown():
            return None

    class DummyAlgo:
        def __init__(self):
            self.saved_paths = []

        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "marker.txt").write_text("ok", encoding="utf-8")
            self.saved_paths.append(str(checkpoint_dir))
            return str(checkpoint_dir)

        def stop(self):
            return None

    class DummyConfig:
        def build(self):
            return algo

    class DummyWandbRun:
        def __init__(self):
            self.summary = {}
            self.finished = False

        def finish(self):
            self.finished = True

    algo = DummyAlgo()
    wandb_run = DummyWandbRun()
    logged_rows = []
    action_plot_logs = []
    validation_summaries = [
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 12.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 12.0}],
            {"tls_1": [{"step": 1.0, "action_0": 1.0, "action_1": 0.0}]},
            {"tls_1": [0, 1, 0]},
            {"tls_1": [{"step": 1.0, "active_phase": 0.0, "phase_0": 3.0, "phase_1": 1.0}]},
        ),
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 9.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 9.0}],
            {"tls_1": [{"step": 1.0, "action_0": 0.5, "action_1": 0.5}]},
            {"tls_1": [1, 1, 0]},
            {"tls_1": [{"step": 1.0, "active_phase": 1.0, "phase_0": 2.0, "phase_1": 4.0}]},
        ),
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 11.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 11.0}],
            {"tls_1": [{"step": 1.0, "action_0": 0.25, "action_1": 0.75}]},
            {"tls_1": [1, 0, 1]},
            {"tls_1": [{"step": 1.0, "active_phase": 1.0, "phase_0": 1.0, "phase_1": 5.0}]},
        ),
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 13.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 13.0}],
            {"tls_1": [{"step": 1.0, "action_0": 0.75, "action_1": 0.25}]},
            {"tls_1": [0, 0, 1]},
            {"tls_1": [{"step": 1.0, "active_phase": 0.0, "phase_0": 4.0, "phase_1": 2.0}]},
        ),
    ]

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind
        emit_metrics({"train/env_step": 40.0, "train/episode_index": 4.0, "train/resco_delay_mean": 10.0}, 4)
        validate({}, 10)
        validate({}, 20)
        validate({}, 30)
        validate({}, 40)

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg
        if include_validation_metrics:
            summary, seed_rows, plot_rows, timeline_rows, phase_queue_rows = validation_summaries.pop(0)
            return dict(summary), list(seed_rows), dict(plot_rows), dict(timeline_rows), dict(phase_queue_rows), {
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
        return {"algorithm/kind": "ppo", "final/resco/avg_delay": 7.0, "eval/episode": 1.0}, [], {}, {}, {}, {
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

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_init_wandb", lambda *args, **kwargs: wandb_run)
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: logged_rows.append((args, kwargs)))
    monkeypatch.setattr(
        rllib_runner,
        "_log_validation_action_plot_images",
        lambda wandb_run, plot_rows_by_agent, action_timeline_by_agent, phase_queue_rows_by_agent, *, pass_index, env_step, episode_index, decision_seconds: action_plot_logs.append(
            {
                "wandb_run": wandb_run,
                "plot_rows_by_agent": plot_rows_by_agent,
                "action_timeline_by_agent": action_timeline_by_agent,
                "phase_queue_rows_by_agent": phase_queue_rows_by_agent,
                "pass_index": pass_index,
                "env_step": env_step,
                "episode_index": episode_index,
                "decision_seconds": decision_seconds,
            }
        ),
    )
    monkeypatch.setattr(rllib_runner, "_log_validation_tripinfo_distribution_images", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=True,
            best_validation_checkpoint_count=3,
            best_validation_metric="validation/resco_delay_mean",
            save_final_model=True,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(cuda_visible_devices="1"),
        algorithm=SimpleNamespace(kind="ppo", params={}),
    )

    result = rllib_runner.train_rllib(cfg)

    metadata_path = tmp_path / "checkpoints" / "ppo" / "best_validation" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert [item["metric_value"] for item in metadata["retained"]] == [9.0, 11.0, 12.0]
    assert len(metadata["retained"][0]["evaluation_seed_rows"]) == 1
    assert len(algo.saved_paths) == 4
    assert any(Path(path).name == "ppo" for path in algo.saved_paths)
    assert [entry["pass_index"] for entry in action_plot_logs] == [1, 2, 3, 4]
    assert all(entry["episode_index"] == 4 for entry in action_plot_logs)
    assert all(entry["plot_rows_by_agent"]["tls_1"][0]["step"] == 1.0 for entry in action_plot_logs)
    assert all("tls_1" in entry["action_timeline_by_agent"] for entry in action_plot_logs)
    assert all("tls_1" in entry["phase_queue_rows_by_agent"] for entry in action_plot_logs)
    assert all(entry["decision_seconds"] == 5 for entry in action_plot_logs)
    assert ray_init_calls
    assert ray_init_calls[0]["num_cpus"] == 2
    assert ray_init_calls[0]["runtime_env"]["env_vars"]["OMP_NUM_THREADS"] == "1"
    assert ray_init_calls[0]["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1"
    validation_rows = [args[2] for args, kwargs in logged_rows if isinstance(args[2], dict) and "validation/env_step" in args[2]]
    assert [row["validation/pass_index"] for row in validation_rows] == [1.0, 2.0, 3.0, 4.0]
    assert all(row["validation/rollout_index"] == 4.0 for row in validation_rows)
    assert all(row["validation/episode_index"] == 4.0 for row in validation_rows)
    assert result["validation/resco_delay_mean"] == 13.0
    assert result["validation/env_step"] == 40.0
    assert result["validation/rollout_index"] == 4.0
    assert result["validation/episode_index"] == 4.0
    assert result["validation/pass_index"] == 4.0
    assert wandb_run.summary["validation/resco_delay_mean"] == 13.0
    assert wandb_run.summary["best_train/resco_delay_mean"] == 10.0
    assert wandb_run.summary["best_train/episode_index"] == 4.0
    assert wandb_run.summary["best_train/env_step"] == 40.0
    assert wandb_run.summary["best_validation/resco_delay_mean"] == 9.0
    assert wandb_run.summary["best_validation/pass_index"] == 2.0
    assert wandb_run.finished is True


def test_summary_episode_index_prefers_rollout_index():
    metrics = {
        "train/rollout_index": 13.0,
        "train/episode_index": 30.0,
        "train/episodes_total": 30.0,
    }

    assert rllib_runner._summary_episode_index_from_metrics(metrics) == 13


def test_train_rllib_writes_best_summary_on_interrupt(monkeypatch, tmp_path):
    ray_shutdown_calls = []

    class DummyRay:
        @staticmethod
        def init(**kwargs):
            return None

        @staticmethod
        def shutdown():
            ray_shutdown_calls.append(True)

    class DummyAlgo:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    class DummyConfig:
        def build(self):
            return algo

    class DummyWandbRun:
        def __init__(self):
            self.summary = {
                "best_train/obsolete": 123.0,
                "best_validation/resco_delay_mean": 5.0,
            }
            self.finished = False

        def finish(self):
            self.finished = True

    algo = DummyAlgo()
    wandb_run = DummyWandbRun()

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind, validate
        emit_metrics({"train/env_step": 10.0, "train/episode_index": 1.0, "train/resco_delay_mean": 15.0}, 1)
        emit_metrics({"train/env_step": 20.0, "train/episode_index": 2.0, "train/resco_delay_mean": 9.0}, 2)
        raise KeyboardInterrupt()

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_init_wandb", lambda *args, **kwargs: wandb_run)
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=True,
            save_best_validation_checkpoints=False,
            best_validation_checkpoint_count=3,
            best_validation_metric="validation/resco_delay_mean",
            save_final_model=False,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(cuda_visible_devices="1"),
        algorithm=SimpleNamespace(kind="ppo", params={}),
    )

    try:
        rllib_runner.train_rllib(cfg)
        assert False, "Expected KeyboardInterrupt"
    except KeyboardInterrupt:
        pass

    assert wandb_run.summary["best_train/resco_delay_mean"] == 9.0
    assert wandb_run.summary["best_train/env_step"] == 20.0
    assert "best_train/obsolete" not in wandb_run.summary
    assert "best_validation/resco_delay_mean" not in wandb_run.summary
    assert wandb_run.finished is True
    assert algo.stop_calls == 1
    assert ray_shutdown_calls == [True]


def test_standard_evaluation_settings_use_rllib_algorithm_config_api():
    class DummyConfig:
        def __init__(self):
            self.kwargs = None

        def evaluation(self, **kwargs):
            self.kwargs = kwargs
            return self

    config = DummyConfig()
    returned = apply_standard_evaluation_settings(
        config,
        {
            "evaluation_interval": 3,
            "evaluation_duration": 2,
            "evaluation_duration_unit": "episodes",
            "evaluation_config": {"explore": False},
            "evaluation_parallel_to_training": True,
        },
    )

    assert returned is config
    assert config.kwargs == {
        "evaluation_interval": 3,
        "evaluation_duration": 2,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {"explore": False},
        "evaluation_parallel_to_training": True,
    }


def test_log_outputs_lets_wandb_custom_step_axes_control_train_and_validation_steps():
    class DummyWandbRun:
        def __init__(self):
            self.calls = []

        def log(self, metrics, step=None):
            self.calls.append((metrics, step))

    wandb_run = DummyWandbRun()

    _log_outputs(wandb_run, None, {"train/env_step": 40320.0, "train/episode_index": 62.0}, step=62)
    _log_outputs(
        wandb_run,
        None,
        {"validation/env_step": 45360.0, "validation/episode_index": 70.0, "validation/reward_mean": 1.0},
        step=45360,
    )

    assert wandb_run.calls == [
        ({"train/env_step": 40320.0, "train/episode_index": 62.0}, None),
        ({"validation/env_step": 45360.0, "validation/episode_index": 70.0, "validation/reward_mean": 1.0}, None),
    ]


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
