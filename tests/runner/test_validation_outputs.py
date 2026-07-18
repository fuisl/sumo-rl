# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments import rllib_runner


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
        algorithm_kind="ppo",
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

    summary, seed_rows, plot_rows, timeline_rows, phase_queue_rows, tripinfo_distributions = (
        rllib_runner._evaluate_with_details(
            cfg,
            tmp_path,
            algo=object(),
            algorithm_kind="ppo",
            logging_cfg=logging_cfg,
            include_validation_metrics=True,
        )
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
