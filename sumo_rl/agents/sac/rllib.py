"""SAC-specific RLlib config, training loop, and training metrics."""

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
from sumo_rl.rllib.custom_sac import build_custom_sac_module_spec


BUILTIN_KIND = "sac_rllib_builtin"
CUSTOM_KIND = "sac_rllib_custom"
KINDS = {BUILTIN_KIND, CUSTOM_KIND}


def build_config(cfg: Any, run_dir: Path, *, algorithm_kind: str):
    from ray.rllib.algorithms.sac import SACConfig

    context = build_algorithm_context(cfg, run_dir, algorithm_kind)
    config = SACConfig().framework("torch").environment(context.env_name)
    config = apply_env_runner_settings(config, context.params)
    config = apply_training_settings(
        config,
        context.params,
        total_timesteps_value=context.total_timesteps,
        allowed_keys=(
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "tau",
            "initial_alpha",
            "target_entropy",
            "gamma",
            "grad_clip",
            "train_batch_size_per_learner",
            "n_step",
            "training_intensity",
            "num_steps_sampled_before_learning_starts",
            "target_network_update_freq",
            "twin_q",
            "clip_actions",
        ),
    )
    config = apply_multi_agent_settings(config, context)

    if algorithm_kind == CUSTOM_KIND:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

        rl_module_specs = {
            policy_id: build_custom_sac_module_spec(
                policy_spec.observation_space,
                policy_spec.action_space,
                model_config=context.params.get("model_config"),
            )
            for policy_id, policy_spec in context.active_policies.items()
        }
        config = config.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs))

    return config


def extract_training_metrics(result: Dict[str, Any], iteration: int, *, algorithm_kind: str) -> Dict[str, Any]:
    metrics = rllib_counter_metrics(result, algorithm_kind=algorithm_kind, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/sac/learners", out=metrics)
    replay_metrics = result.get("replay_buffer") or result.get("replay_buffers")
    if isinstance(replay_metrics, dict):
        flatten_numeric_metrics(replay_metrics, prefix="train/sac/replay", out=metrics)
    return metrics


def train(
    algo,
    cfg: Any,
    *,
    algorithm_kind: str,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> None:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    for iteration in range(training_iterations(cfg, params)):
        result = algo.train()
        metrics = extract_training_metrics(result, iteration + 1, algorithm_kind=algorithm_kind)
        step = int(metrics.get("train/env_steps_sampled") or iteration + 1)
        if emit_metrics is not None:
            emit_metrics(metrics, step)
        print(f"[{algorithm_kind}] iteration={iteration + 1} result_keys={sorted(result.keys())[:8]}")
