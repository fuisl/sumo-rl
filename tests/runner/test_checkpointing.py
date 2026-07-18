# ruff: noqa: E402

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner


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
    assert all(not key.startswith("final/") for key in metadata["retained"][0]["evaluation_summary"].keys())
    assert all(not key.startswith("final/") for row in metadata["retained"][0]["evaluation_seed_rows"] for key in row.keys())
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


def test_periodic_checkpoint_state_saves_crossed_milestones_without_duplicates(tmp_path):
    class FakeAlgo:
        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "checkpoint.json").write_text("ok", encoding="utf-8")
            return str(checkpoint_dir)

    logging_cfg = SimpleNamespace(
        save_periodic_checkpoints=True,
        checkpoint_every_episodes=50,
    )
    state = rllib_runner._init_periodic_checkpoint_state(tmp_path, "ppo", logging_cfg, resumed_run=False)
    algo = FakeAlgo()

    assert rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=40, env_step=400) == []

    saved = rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=101, env_step=1010)

    assert [entry["milestone_episode"] for entry in saved] == [50, 100]
    assert (state["base_dir"] / "episode_00050__observed_00101__step_0001010").exists()
    assert (state["base_dir"] / "episode_00100__observed_00101__step_0001010").exists()
    assert rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=120, env_step=1200) == []
    metadata = json.loads(state["metadata_path"].read_text(encoding="utf-8"))
    assert [entry["milestone_episode"] for entry in metadata["saved"]] == [50, 100]


def test_periodic_checkpoint_state_bootstraps_resumed_run_before_next_save(tmp_path):
    class FakeAlgo:
        def save_to_path(self, path):
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "checkpoint.json").write_text("ok", encoding="utf-8")
            return str(checkpoint_dir)

    logging_cfg = SimpleNamespace(
        save_periodic_checkpoints=True,
        checkpoint_every_episodes=50,
    )
    state = rllib_runner._init_periodic_checkpoint_state(tmp_path, "ppo", logging_cfg, resumed_run=True)
    algo = FakeAlgo()

    assert rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=120, env_step=1200) == []
    assert state["last_saved_multiple"] == 2
    assert rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=149, env_step=1490) == []

    saved = rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=150, env_step=1500)

    assert [entry["milestone_episode"] for entry in saved] == [150]
    assert (state["base_dir"] / "episode_00150__observed_00150__step_0001500").exists()
    assert rllib_runner._maybe_save_periodic_checkpoint(state, algo, completed_episode=151, env_step=1510) == []


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
            return (
                dict(summary),
                list(seed_rows),
                dict(plot_rows),
                dict(timeline_rows),
                dict(phase_queue_rows),
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
        return (
            {"algorithm/kind": "ppo", "final/resco/avg_delay": 7.0, "eval/episode": 1.0},
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
    monkeypatch.setattr(rllib_runner, "_init_wandb", lambda *args, **kwargs: wandb_run)
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: logged_rows.append((args, kwargs)))
    monkeypatch.setattr(
        rllib_runner,
        "_log_validation_action_plot_images",
        lambda wandb_run,
        plot_rows_by_agent,
        action_timeline_by_agent,
        phase_queue_rows_by_agent,
        *,
        pass_index,
        env_step,
        episode_index,
        decision_seconds: action_plot_logs.append(
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
    validation_rows = [
        args[2] for args, kwargs in logged_rows if isinstance(args[2], dict) and "validation/env_step" in args[2]
    ]
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


def test_train_rllib_restores_resume_checkpoint_before_training(monkeypatch, tmp_path):
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
            self.restored_path = None
            self.stop_calls = 0

        def restore_from_path(self, path):
            self.restored_path = path

        def stop(self):
            self.stop_calls += 1

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
    resume_checkpoint = tmp_path / "resume-checkpoint"
    resume_checkpoint.mkdir(parents=True, exist_ok=True)

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del cfg, algorithm_kind, validate
        assert algo_obj is algo
        emit_metrics(
            {
                "algorithm/kind": "ppo",
                "train/env_step": 10.0,
                "train/episode_index": 1.0,
                "train/rollout_index": 1.0,
                "train/resco_delay_mean": 9.0,
            },
            10,
        )

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_init_wandb", lambda *args, **kwargs: wandb_run)
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(rllib_runner, "_log_validation_action_plot_images", lambda *args, **kwargs: None)
    monkeypatch.setattr(rllib_runner, "_log_validation_tripinfo_distribution_images", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rllib_runner,
        "_evaluate_with_details",
        lambda *args, **kwargs: (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 8.0,
                "validation/eval/episode": 1.0,
                "eval/episode": 1.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 8.0}],
            {},
            {},
            {},
            {},
        ),
    )

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            resume_from_checkpoint=str(resume_checkpoint),
            save_periodic_checkpoints=True,
            checkpoint_every_episodes=50,
            save_best_validation_checkpoints=False,
            best_validation_checkpoint_count=3,
            best_validation_metric="validation/resco_delay_mean",
            save_final_model=False,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        resources=SimpleNamespace(cuda_visible_devices="1"),
        algorithm=SimpleNamespace(kind="ppo", params={}),
    )

    result = rllib_runner.train_rllib(cfg)

    assert algo.restored_path == str(resume_checkpoint.resolve())
    assert result["validation/resco_delay_mean"] == 8.0
    assert algo.stop_calls == 1
    assert wandb_run.finished is True
    assert ray_init_calls


def test_train_rllib_rejects_missing_resume_checkpoint_path(tmp_path):
    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            resume_from_checkpoint=str(tmp_path / "missing-checkpoint"),
        ),
        experiment=SimpleNamespace(name="demo"),
        algorithm=SimpleNamespace(kind="ppo", params={}),
    )

    try:
        rllib_runner.train_rllib(cfg)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "Resume checkpoint path does not exist" in str(exc)


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
