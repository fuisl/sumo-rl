"""DQN-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    build_algorithm_context,
    completed_training_episodes,
    emit_training_episode_rows,
    emit_validation_if_due,
    flatten_numeric_metrics,
    plain_dict,
    extract_rllib_result_metrics,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)


KIND = "dqn"


def build_replay_buffer_config(params: Dict[str, Any]) -> Dict[str, Any]:
    explicit = params.get("replay_buffer_config")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)

    buffer_type = str(params.get("replay_buffer_type", "MultiAgentPrioritizedEpisodeReplayBuffer"))
    config: Dict[str, Any] = {
        "type": buffer_type,
        "capacity": int(params.get("replay_buffer_capacity", 50000)),
    }
    if "Prioritized" in buffer_type:
        config["alpha"] = float(params.get("replay_buffer_alpha", 0.6))
        config["beta"] = float(params.get("replay_buffer_beta", 0.4))
    return config


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.dqn import DQNConfig

    context = build_algorithm_context(cfg, run_dir, KIND)
    callbacks_class = training_episode_summary_callbacks_class()
    params = dict(context.params)
    params["replay_buffer_config"] = build_replay_buffer_config(params)
    if "num_steps_sampled_before_learning_starts" in params:
        params["num_steps_sampled_before_learning_starts"] = max(
            int(params["num_steps_sampled_before_learning_starts"]),
            context.episode_steps + 1,
        )

    config = DQNConfig().framework("torch").environment(env=context.env_name, disable_env_checking=True)
    config = apply_env_runner_settings(config, params)
    config = apply_training_settings(
        config,
        params,
        episode_steps_value=context.episode_steps,
        allowed_keys=(
            "lr",
            "gamma",
            "grad_clip",
            "train_batch_size_per_learner",
            "n_step",
            "training_intensity",
            "num_steps_sampled_before_learning_starts",
            "target_network_update_freq",
            "replay_buffer_config",
        ),
    )
    config = apply_multi_agent_settings(config, context)
    config = apply_standard_evaluation_settings(config, params)
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = extract_rllib_result_metrics(result, algorithm_kind=KIND, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/dqn/learners", out=metrics)
    replay_metrics = result.get("replay_buffer") or result.get("replay_buffers")
    if isinstance(replay_metrics, dict):
        flatten_numeric_metrics(replay_metrics, prefix="train/dqn/replay", out=metrics)
    return metrics


def train(
    algo,
    cfg: Any,
    *,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]] = None,
    validate: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> None:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    del params
    callbacks_class = training_episode_summary_callbacks_class()
    callbacks_class.reset_episode_summary_tracking()
    iteration = 0
    last_logged_step = 0
    last_validation_progress = 0
    while True:
        iteration += 1
        result = algo.train()
        metrics = extract_training_metrics(result, iteration)
        episode_summaries = callbacks_class.drain_pending_episode_summaries()
        is_final = training_should_stop(metrics, cfg)
        last_logged_step = emit_training_episode_rows(
            metrics,
            episode_summaries,
            cfg,
            algorithm_kind=KIND,
            last_logged_episode=last_logged_step,
            emit_metrics=emit_metrics,
            force=is_final,
        )
        last_validation_progress = emit_validation_if_due(
            metrics,
            cfg,
            last_validation_step=last_validation_progress,
            validate=validate,
        )
        completed_episodes = completed_training_episodes(metrics, cfg)
        print(
            f"[{KIND}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}"
        )
        if is_final:
            break
