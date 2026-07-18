"""PPO-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sumo_rl.agents.dcrnn.dcrnn import build_graph_algorithm_context, graph_params
from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    build_algorithm_context,
    completed_training_episodes,
    emit_training_episode_rows,
    emit_validation_if_due,
    extract_entropy_mean,
    extract_rllib_result_metrics,
    flatten_numeric_metrics,
    plain_dict,
    training_episode_jump,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)

KIND = "ppo"
MLP_DCRNN_KIND = "ppo_dcrnn_mlp"
SHARED_MLP_DCRNN_KIND = "ppo_dcrnn_shared_mlp"


def _format_gib(num_bytes: float) -> str:
    return f"{num_bytes / float(1024**3):.2f} GiB"


def _warn_if_ppo_graph_memory_is_large(context) -> None:
    if not context.active_policies:
        return

    first_policy = next(iter(context.active_policies.values()))
    obs_space = getattr(first_policy, "observation_space", None)
    obs_shape = tuple(getattr(obs_space, "shape", ()) or ())
    if len(obs_shape) != 3:
        return

    history_len, num_nodes, feature_dim = (int(dim) for dim in obs_shape)
    dtype_name = str(getattr(obs_space, "dtype", "float32"))
    num_policies = max(1, len(context.active_policies))
    per_sample_obs_bytes = history_len * num_nodes * feature_dim * 4
    rollout_obs_bytes = per_sample_obs_bytes * max(1, int(context.episode_steps)) * num_policies
    train_batch_size = max(1, int(context.params.get("train_batch_size_per_learner", context.episode_steps)))
    minibatch_size = max(
        1, int(context.params.get("minibatch_size", context.params.get("sgd_minibatch_size", train_batch_size)))
    )
    effective_train_batch_size = min(train_batch_size, max(1, int(context.episode_steps)))
    minibatch_obs_bytes = per_sample_obs_bytes * minibatch_size

    if max(rollout_obs_bytes, minibatch_obs_bytes) < 256 * 1024 * 1024:
        return

    print(
        f"[{MLP_DCRNN_KIND}] Large PPO graph footprint detected: "
        f"{num_policies} policies x [{history_len}, {num_nodes}, {feature_dim}] {dtype_name} observations. "
        f"One full on-policy rollout across all policies stores about {_format_gib(rollout_obs_bytes)} of observations; "
        f"one learner minibatch holds about {_format_gib(minibatch_obs_bytes)} of observations. "
        f"Effective train_batch_size_per_learner is capped to {effective_train_batch_size} by the episode horizon. "
        f"Shared-backbone PPO reuses one graph encode across agent heads, so remaining memory pressure is mostly "
        f"rollout duplication plus learner minibatch size. If memory spikes, lower sgd_minibatch_size or history_len "
        f"before changing the backbone."
    )


def _ppo_dcrnn_model_config(params: dict[str, Any], graph_model_config: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config.setdefault("architecture_tag", MLP_DCRNN_KIND)
    model_config.setdefault("hid_dim", 128)
    model_config.setdefault("max_diffusion_step", 2)
    model_config.setdefault("num_rnn_layers", 1)
    model_config.setdefault("filter_type", "dual_random_walk")
    pre_encoder = dict(model_config.get("pre_encoder", {}) or {})
    pre_encoder.setdefault("enabled", True)
    pre_encoder.setdefault("hidden_dim", int(model_config.get("hid_dim", model_config.get("hidden_dim", 128))))
    pre_encoder.setdefault("activation", "relu")
    model_config["pre_encoder"] = pre_encoder
    model_config.update(graph_model_config)
    return model_config


def _ppo_dcrnn_shared_model_config(params: dict[str, Any], graph_model_config: dict[str, Any]) -> dict[str, Any]:
    model_config = _ppo_dcrnn_model_config(params, graph_model_config)
    model_config["architecture_tag"] = SHARED_MLP_DCRNN_KIND
    return model_config


def build_graph_eval_env(
    cfg: Any,
    run_dir: Path,
    seed: int | None = None,
    *,
    use_libsumo: bool | None = None,
):
    from sumo_rl.environment.graph_env import build_rllib_graph_parallel_env

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    return build_rllib_graph_parallel_env(cfg, run_dir, seed=seed, params=graph_params(params), use_libsumo=use_libsumo)


def build_config(cfg: Any, run_dir: Path, *, algorithm_kind: str = KIND):
    from ray.rllib.algorithms.ppo import PPOConfig

    algorithm_kind = str(algorithm_kind or KIND).strip()
    if algorithm_kind in {MLP_DCRNN_KIND, SHARED_MLP_DCRNN_KIND}:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

        from sumo_rl.agents.ppo.rllib_module import (
            build_ppo_dcrnn_module_spec,
            build_ppo_dcrnn_shared_module_spec,
            build_ppo_dcrnn_shared_multi_module_spec,
        )

        context, model_configs = build_graph_algorithm_context(
            cfg,
            run_dir,
            algorithm_kind=algorithm_kind,
            model_config_builder=(
                _ppo_dcrnn_shared_model_config if algorithm_kind == SHARED_MLP_DCRNN_KIND else _ppo_dcrnn_model_config
            ),
        )
        _warn_if_ppo_graph_memory_is_large(context)
    else:
        context = build_algorithm_context(cfg, run_dir, KIND)
        model_configs = None
    callbacks_class = training_episode_summary_callbacks_class()
    config = PPOConfig().framework("torch").environment(env=context.env_name, disable_env_checking=True)
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
    config = apply_standard_evaluation_settings(config, context.params)
    if algorithm_kind in {MLP_DCRNN_KIND, SHARED_MLP_DCRNN_KIND}:
        rl_module_spec_builder = (
            build_ppo_dcrnn_shared_module_spec if algorithm_kind == SHARED_MLP_DCRNN_KIND else build_ppo_dcrnn_module_spec
        )
        rl_module_specs = {
            policy_id: rl_module_spec_builder(
                policy_spec.observation_space,
                policy_spec.action_space,
                model_config=model_configs[policy_id],
            )
            for policy_id, policy_spec in context.active_policies.items()
        }
        if algorithm_kind == SHARED_MLP_DCRNN_KIND:
            from sumo_rl.agents.ppo.learner import PPOSharedEncoderTorchLearner

            config = config.rl_module(
                rl_module_spec=build_ppo_dcrnn_shared_multi_module_spec(
                    rl_module_specs,
                    model_config=next(iter(model_configs.values())),
                )
            )
            learners = getattr(config, "learners", None)
            if callable(learners):
                config = config.learners(learner_class=PPOSharedEncoderTorchLearner)
            else:
                config = config.training(learner_class=PPOSharedEncoderTorchLearner)
        else:
            config = config.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs))
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: dict[str, Any], iteration: int, *, algorithm_kind: str = KIND) -> dict[str, Any]:
    metrics = extract_rllib_result_metrics(result, algorithm_kind=algorithm_kind, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/ppo/learners", out=metrics)
        entropy_mean = extract_entropy_mean(learner_metrics)
        if entropy_mean is not None:
            metrics["train/ppo/entropy_mean"] = float(entropy_mean)
    return metrics


def train(
    algo,
    cfg: Any,
    *,
    algorithm_kind: str = KIND,
    emit_metrics: Callable[[dict[str, Any], int], None] | None = None,
    validate: Callable[[dict[str, Any], int], None] | None = None,
) -> None:
    algorithm_kind = str(algorithm_kind or KIND).strip()
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    del params
    callbacks_class = training_episode_summary_callbacks_class()
    callbacks_class.reset_episode_summary_tracking()
    iteration = 0
    last_logged_step = 0
    last_completed_episode = 0
    observed_completed_episodes = 0
    last_validation_progress = 0
    while True:
        iteration += 1
        result = algo.train()
        metrics = extract_training_metrics(result, iteration, algorithm_kind=algorithm_kind)
        progress_jump = training_episode_jump(metrics, cfg, last_completed_episode=last_completed_episode)
        metrics["train/rllib/rollout_jump"] = float(progress_jump)
        metrics["debug/rllib/rollout_jump"] = float(progress_jump)
        episode_summaries = callbacks_class.drain_pending_episode_summaries()
        observed_completed_episodes += len(episode_summaries)
        metrics["train/observed_completed_episodes_jump"] = float(len(episode_summaries))
        metrics["train/observed_completed_episodes_total"] = float(observed_completed_episodes)
        metrics["debug/env_completed_episodes_jump"] = float(len(episode_summaries))
        metrics["debug/env_completed_episodes_total"] = float(observed_completed_episodes)
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
        last_completed_episode = completed_episodes
        if progress_jump > 1:
            print(
                f"[{KIND}] RLlib episode jump detected: +{progress_jump} "
                f"(from {completed_episodes - progress_jump} to {completed_episodes}) "
                f"at iteration={iteration}"
            )
        print(
            f"[{algorithm_kind}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}"
        )
        if is_final:
            break
