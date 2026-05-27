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
        return 12.5, {"tls_1": [0, 1, 0]}, {"tls_1": 2}

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
        return 999.0, {"tls_1": [0, 1, 1]}, {"tls_1": 2}

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
        pass_index=3,
        env_step=120,
        decision_seconds=5,
    )

    assert len(run.calls) == 2
    assert run.calls[0]["validation/pass_index"] == 3.0
    assert run.calls[0]["validation/env_step"] == 120.0
    assert isinstance(run.calls[0]["validation/actions_share/tls_1"], DummyImage)
    assert run.calls[0]["validation/actions_share/tls_1"].image.size == (1040, 560)
    assert isinstance(run.calls[0]["validation/actions_timeline/tls_1"], DummyImage)
    assert run.calls[0]["validation/actions_timeline/tls_1"].image.size == (1040, 420)
    assert "validation pass 3" in run.calls[0]["validation/actions_share/tls_1"].caption
    assert isinstance(run.calls[1]["validation/actions_share/tls_2"], DummyImage)
    assert isinstance(run.calls[1]["validation/actions_timeline/tls_2"], DummyImage)


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
        7: (10.0, {"tls_1": [0, 1, 1]}, {"tls_1": 2}),
        8: (12.0, {"tls_1": [1, 1, 0]}, {"tls_1": 2}),
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
        log_validation_action_plots=True,
        validation_action_plot_max_agents=None,
    )

    summary, seed_rows, plot_rows, timeline_rows = rllib_runner._evaluate_with_details(
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
    assert emitted[0][1]["train/resco_wait_mean"] == 5.0
    assert emitted[1][1]["train/resco_wait_mean"] == 6.0


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
    assert [row["train/episode_index"] for _, row in emitted] == [1.0, 2.0, 3.0]
    assert all(row["train/env_step"] == 60.0 for _, row in emitted)
    assert all("train/reward_mean" not in row for _, row in emitted)


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
            "validation/efficiency_total_arrived": 8.0,
            "validation/safety_total_collisions": 0.0,
            "warnings/missing_tripinfo": 0.0,
            "eval/episode": 2.0,
            "episode/sim_time_abs": 3600.0,
        },
        step=100,
    )

    assert row["algorithm/kind"] == "ppo"
    assert row["validation/env_step"] == 100.0
    assert row["validation/reward_mean"] == 12.0
    assert row["validation/resco_delay_mean"] == 4.0
    assert row["validation/efficiency_total_arrived"] == 8.0
    assert row["validation/safety_total_collisions"] == 0.0
    assert row["validation/warnings/missing_tripinfo"] == 0.0
    assert row["validation/eval/episode"] == 2.0
    assert row["validation/episode/sim_time_abs"] == 3600.0


def test_train_rllib_validation_saves_best_checkpoints_and_final_model(monkeypatch, tmp_path):
    class DummyRay:
        @staticmethod
        def init(**kwargs):
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

    algo = DummyAlgo()
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
        ),
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 8.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 8.0}],
            {"tls_1": [{"step": 1.0, "action_0": 0.75, "action_1": 0.25}]},
            {"tls_1": [0, 0, 1]},
        ),
        (
            {
                "algorithm/kind": "ppo",
                "validation/resco_delay_mean": 7.0,
                "validation/eval/episode": 2.0,
                "eval/episode": 2.0,
            },
            [{"eval/seed": 1.0, "validation/resco_delay_mean": 7.0}],
            {"tls_1": [{"step": 1.0, "action_0": 0.0, "action_1": 1.0}]},
            {"tls_1": [1, 1, 1]},
        ),
    ]

    def fake_train_algorithm(algo_obj, cfg, algorithm_kind, emit_metrics, validate=None):
        del algo_obj, cfg, algorithm_kind
        emit_metrics({"train/env_step": 40.0}, 4)
        validate({}, 10)
        validate({}, 20)
        validate({}, 30)
        validate({}, 40)

    def fake_evaluate_with_details(cfg, run_dir, algo_obj, algorithm_kind, logging_cfg, *, include_validation_metrics=False):
        del cfg, run_dir, algo_obj, algorithm_kind, logging_cfg
        if include_validation_metrics:
            summary, seed_rows, plot_rows, timeline_rows = validation_summaries.pop(0)
            return dict(summary), list(seed_rows), dict(plot_rows), dict(timeline_rows)
        return {"algorithm/kind": "ppo", "final/resco/avg_delay": 7.0, "eval/episode": 1.0}, [], {}, {}

    monkeypatch.setitem(sys.modules, "ray", DummyRay)
    monkeypatch.setattr(rllib_runner, "_get_run_dir", lambda: tmp_path)
    monkeypatch.setattr(rllib_runner, "_build_algorithm_config", lambda cfg, run_dir, algorithm_kind: DummyConfig())
    monkeypatch.setattr(rllib_runner, "_train_algorithm", fake_train_algorithm)
    monkeypatch.setattr(rllib_runner, "_evaluate_with_details", fake_evaluate_with_details)
    monkeypatch.setattr(rllib_runner, "_log_outputs", lambda *args, **kwargs: logged_rows.append((args, kwargs)))
    monkeypatch.setattr(
        rllib_runner,
        "_log_validation_action_plot_images",
        lambda wandb_run, plot_rows_by_agent, action_timeline_by_agent, *, pass_index, env_step, decision_seconds: action_plot_logs.append(
            {
                "wandb_run": wandb_run,
                "plot_rows_by_agent": plot_rows_by_agent,
                "action_timeline_by_agent": action_timeline_by_agent,
                "pass_index": pass_index,
                "env_step": env_step,
                "decision_seconds": decision_seconds,
            }
        ),
    )
    monkeypatch.setattr(rllib_runner, "_update_wandb_summary", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        logging=SimpleNamespace(
            enabled=False,
            save_best_validation_checkpoints=True,
            best_validation_checkpoint_count=3,
            best_validation_metric="validation/resco_delay_mean",
            log_validation_action_plots=True,
            save_final_model=True,
        ),
        experiment=SimpleNamespace(name="demo", project="proj", group=None, tags=[], seed=1, eval_episodes=1),
        algorithm=SimpleNamespace(kind="ppo", params={}),
    )

    result = rllib_runner.train_rllib(cfg)

    metadata_path = tmp_path / "checkpoints" / "ppo" / "best_validation" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert [item["metric_value"] for item in metadata["retained"]] == [7.0, 8.0, 9.0]
    assert len(metadata["retained"][0]["evaluation_seed_rows"]) == 1
    assert len(algo.saved_paths) == 6
    assert any(Path(path).name == "ppo" for path in algo.saved_paths)
    assert [entry["pass_index"] for entry in action_plot_logs] == [1, 2, 3, 4, 5]
    assert all(entry["plot_rows_by_agent"]["tls_1"][0]["step"] == 1.0 for entry in action_plot_logs)
    assert all("tls_1" in entry["action_timeline_by_agent"] for entry in action_plot_logs)
    assert all(entry["decision_seconds"] == 5 for entry in action_plot_logs)
    validation_rows = [args[2] for args, kwargs in logged_rows if isinstance(args[2], dict) and "validation/env_step" in args[2]]
    assert [row["validation/pass_index"] for row in validation_rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["validation/resco_delay_mean"] == 7.0
    assert result["validation/env_step"] == 40.0
    assert result["validation/pass_index"] == 5.0


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
    _log_outputs(wandb_run, None, {"validation/env_step": 45360.0, "validation/reward_mean": 1.0}, step=45360)

    assert wandb_run.calls == [
        ({"train/env_step": 40320.0, "train/episode_index": 62.0}, None),
        ({"validation/env_step": 45360.0, "validation/reward_mean": 1.0}, None),
    ]


def test_init_wandb_binds_debug_metrics_to_train_env_step(monkeypatch, tmp_path):
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
    assert (("train/*",), {"step_metric": "train/env_step"}) in run.metric_calls
    assert (("debug/*",), {"step_metric": "train/env_step"}) in run.metric_calls


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
    assert (("validation/*",), {"step_metric": "validation/env_step"}) in run.metric_calls
    assert all(args != ("final/*",) for args, kwargs in run.metric_calls)
    assert all(args != ("eval/episode",) for args, kwargs in run.metric_calls)
