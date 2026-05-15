"""DQN-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_training_settings,
    build_algorithm_context,
    flatten_numeric_metrics,
    plain_dict,
    rllib_counter_metrics,
    training_iterations,
)


KIND = "dqn_rllib"


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.dqn import DQNConfig

    context = build_algorithm_context(cfg, run_dir, KIND)
    params = dict(context.params)
    if "num_steps_sampled_before_learning_starts" in params:
        params["num_steps_sampled_before_learning_starts"] = max(
            int(params["num_steps_sampled_before_learning_starts"]),
            context.total_timesteps + 1,
        )

    config = DQNConfig().framework("torch").environment(context.env_name)
    config = apply_env_runner_settings(config, params)
    config = apply_training_settings(
        config,
        params,
        total_timesteps_value=context.total_timesteps,
        allowed_keys=(
            "lr",
            "gamma",
            "grad_clip",
            "train_batch_size_per_learner",
            "n_step",
            "training_intensity",
            "num_steps_sampled_before_learning_starts",
            "target_network_update_freq",
        ),
    )
    return apply_multi_agent_settings(config, context)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = rllib_counter_metrics(result, algorithm_kind=KIND, iteration=iteration)
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
) -> None:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    for iteration in range(training_iterations(cfg, params)):
        result = algo.train()
        metrics = extract_training_metrics(result, iteration + 1)
        step = int(metrics.get("train/env_steps_sampled") or iteration + 1)
        if emit_metrics is not None:
            emit_metrics(metrics, step)
        print(f"[{KIND}] iteration={iteration + 1} result_keys={sorted(result.keys())[:8]}")
