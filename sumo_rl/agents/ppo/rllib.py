"""PPO-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_training_settings,
    build_algorithm_context,
    completed_training_episodes,
    emit_training_metrics_by_step,
    flatten_numeric_metrics,
    plain_dict,
    rllib_counter_metrics,
    training_episode_target,
    training_should_stop,
)


KIND = "ppo_rllib"


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.ppo import PPOConfig

    context = build_algorithm_context(cfg, run_dir, KIND)
    config = PPOConfig().framework("torch").environment(context.env_name)
    config = apply_env_runner_settings(config, context.params)
    config = apply_training_settings(
        config,
        context.params,
        total_timesteps_value=context.total_timesteps,
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
    return apply_multi_agent_settings(config, context)


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
    iteration = 0
    last_logged_step = 0
    while True:
        iteration += 1
        result = algo.train()
        metrics = extract_training_metrics(result, iteration)
        is_final = training_should_stop(metrics, cfg)
        last_logged_step = emit_training_metrics_by_step(
            metrics,
            cfg,
            last_logged_step=last_logged_step,
            emit_metrics=emit_metrics,
            force=is_final,
        )
        completed_episodes = completed_training_episodes(metrics, cfg)
        print(
            f"[{KIND}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}"
        )
        if is_final:
            break
