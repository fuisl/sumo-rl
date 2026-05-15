"""PPO-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_training_settings,
    build_algorithm_context,
    build_training_episode_row,
    completed_training_episodes,
    flatten_numeric_metrics,
    plain_dict,
    rllib_counter_metrics,
    should_log_training_episode,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)


KIND = "ppo_rllib"


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.ppo import PPOConfig

    context = build_algorithm_context(cfg, run_dir, KIND)
    callbacks_class = training_episode_summary_callbacks_class()
    config = PPOConfig().framework("torch").environment(context.env_name)
    config = apply_env_runner_settings(config, context.params)
    config = apply_training_settings(
        config,
        context.params,
        episode_steps_value=context.episode_steps,
        allowed_keys=(
            "lr",
            "gamma",
            "lambda_",
            "clip_param",
            "entropy_coeff",
            "grad_clip",
            "train_batch_size_per_learner",
            "num_epochs",
            "minibatch_size",
        ),
        aliases={
            "num_sgd_iter": "num_epochs",
            "sgd_minibatch_size": "minibatch_size",
        },
    )
    config = apply_multi_agent_settings(config, context)
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = rllib_counter_metrics(result, algorithm_kind=KIND, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/ppo/learners", out=metrics)
    return metrics


def train(
    algo,
    cfg: Any,
    *,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> None:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    del params
    callbacks_class = training_episode_summary_callbacks_class()
    callbacks_class.drain_pending_episode_summaries()
    iteration = 0
    last_logged_step = 0
    while True:
        iteration += 1
        result = algo.train()
        metrics = extract_training_metrics(result, iteration)
        episode_summaries = callbacks_class.drain_pending_episode_summaries()
        is_final = training_should_stop(metrics, cfg)
        if emit_metrics is not None:
            for episode_summary in episode_summaries:
                episode_index = episode_summary.get("episode/index")
                if not should_log_training_episode(
                    episode_index,
                    cfg,
                    last_logged_episode=last_logged_step,
                    force=is_final,
                ):
                    continue
                row = build_training_episode_row(metrics, episode_summary, algorithm_kind=KIND)
                row_step = int(row.get("train/episode_index") or row.get("train/episodes_total") or 0)
                emit_metrics(row, row_step)
                if row_step > 0:
                    last_logged_step = row_step
            if is_final and not episode_summaries:
                row = build_training_episode_row(metrics, {}, algorithm_kind=KIND)
                row_step = int(row.get("train/episode_index") or row.get("train/episodes_total") or 0)
                emit_metrics(row, row_step)
                if row_step > 0:
                    last_logged_step = row_step
        completed_episodes = completed_training_episodes(metrics, cfg)
        print(
            f"[{KIND}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}"
        )
        if is_final:
            break
