"""SAC-specific RLlib config, training loop, and training metrics.

This path uses RLlib SAC directly on the SUMO multi-agent discrete action spaces.
There is no joint continuous action adapter in the current implementation.
"""

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
    rllib_counter_metrics,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)
from sumo_rl.agents.sac.custom_sac import (
    build_custom_sac_module_spec,
    build_custom_sac_multi_module_spec,
    normalize_custom_sac_model_config,
)


BUILTIN_KIND = "sac_builtin"
CUSTOM_KIND = "sac_custom"
KINDS = {BUILTIN_KIND, CUSTOM_KIND}


def build_replay_buffer_config(params: Dict[str, Any]) -> Dict[str, Any]:
    explicit = params.get("replay_buffer_config")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)

    buffer_type = str(params.get("replay_buffer_type", "MultiAgentPrioritizedEpisodeReplayBuffer"))
    config: Dict[str, Any] = {
        "type": buffer_type,
        "capacity": int(params.get("replay_buffer_capacity", int(1e6))),
    }
    if "Prioritized" in buffer_type:
        config["alpha"] = float(params.get("replay_buffer_alpha", 0.6))
        config["beta"] = float(params.get("replay_buffer_beta", 0.4))
    return config


def build_config(cfg: Any, run_dir: Path, *, algorithm_kind: str):
    from ray.rllib.algorithms.sac import SACConfig

    context = build_algorithm_context(cfg, run_dir, algorithm_kind)
    callbacks_class = training_episode_summary_callbacks_class()
    params = dict(context.params)
    params["replay_buffer_config"] = build_replay_buffer_config(params)
    custom_model_config = params.get("model_config")
    if algorithm_kind == CUSTOM_KIND:
        normalized_custom_model_config = normalize_custom_sac_model_config(custom_model_config)
        custom_model_config = normalized_custom_model_config
        params.setdefault("twin_q", bool(normalized_custom_model_config.get("twin_q", True)))
    if "num_steps_sampled_before_learning_starts" in params:
        params["num_steps_sampled_before_learning_starts"] = max(
            int(params["num_steps_sampled_before_learning_starts"]),
            context.episode_steps + 1,
        )

    config = SACConfig().framework("torch").environment(env=context.env_name, disable_env_checking=True)
    config = apply_env_runner_settings(config, params)
    config = apply_training_settings(
        config,
        params,
        episode_steps_value=context.episode_steps,
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
            "replay_buffer_config",
        ),
    )
    config = apply_multi_agent_settings(config, context)
    config = apply_standard_evaluation_settings(config, params)

    if algorithm_kind == CUSTOM_KIND:
        rl_module_specs = {
            policy_id: build_custom_sac_module_spec(
                policy_spec.observation_space,
                policy_spec.action_space,
                model_config=custom_model_config,
            )
            for policy_id, policy_spec in context.active_policies.items()
        }
        config = config.rl_module(
            rl_module_spec=build_custom_sac_multi_module_spec(
                rl_module_specs,
                model_config=custom_model_config,
            )
        )

    return config.callbacks(callbacks_class)


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
        metrics = extract_training_metrics(result, iteration, algorithm_kind=algorithm_kind)
        episode_summaries = callbacks_class.drain_pending_episode_summaries()
        is_final = training_should_stop(metrics, cfg)
        last_logged_step = emit_training_episode_rows(
            metrics,
            episode_summaries,
            cfg,
            algorithm_kind=algorithm_kind,
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
            f"[{algorithm_kind}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}"
        )
        if is_final:
            break
