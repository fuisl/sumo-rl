"""SAC-specific RLlib config, training loop, and training metrics.

This path uses RLlib SAC directly on the SUMO multi-agent discrete action spaces.
There is no joint continuous action adapter in the current implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
    flatten_numeric_metrics,
    plain_dict,
    extract_rllib_result_metrics,
    training_episode_jump,
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
CUSTOM_KIND = "sac_mlp"
DCRNN_ACTOR_KIND = "sac_dcrnn_actor"
DCRNN_FULL_KIND = "sac_dcrnn_full"
DCRNN_ACTOR_MLP_KIND = "sac_dcrnn_actor_mlp"
DCRNN_FULL_MLP_KIND = "sac_dcrnn_full_mlp"
DCRNN_SHARED_MLP_KIND = "sac_dcrnn_shared_mlp"
CUSTOM_ALIASES = {"sac_custom"}
GRAPH_KINDS = {
    DCRNN_ACTOR_KIND,
    DCRNN_FULL_KIND,
    DCRNN_ACTOR_MLP_KIND,
    DCRNN_FULL_MLP_KIND,
    DCRNN_SHARED_MLP_KIND,
}
KINDS = {BUILTIN_KIND, CUSTOM_KIND, *GRAPH_KINDS}
ALL_KINDS = {BUILTIN_KIND, CUSTOM_KIND, *GRAPH_KINDS, *CUSTOM_ALIASES}


def normalize_kind(algorithm_kind: str) -> str:
    kind = str(algorithm_kind or "").strip()
    if kind in CUSTOM_ALIASES:
        return CUSTOM_KIND
    return kind


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


def _with_dcrnn_encoder_defaults(
    model_config: Dict[str, Any],
    *,
    branch: str,
    architecture_tag: str,
) -> Dict[str, Any]:
    model_config = dict(model_config)
    model_config.setdefault("architecture_tag", architecture_tag)
    branch_config = dict(model_config.get(branch) or {})
    encoder_config = dict(branch_config.get("encoder") or {})
    encoder_config.setdefault("type", "dcrnn")
    encoder_config.setdefault("hidden_dim", 128)
    encoder_config.setdefault("max_diffusion_step", 2)
    encoder_config.setdefault("num_rnn_layers", 1)
    encoder_config.setdefault("filter_type", "dual_random_walk")
    branch_config["encoder"] = encoder_config
    model_config[branch] = branch_config
    return model_config


def _with_pre_encoder_defaults(
    model_config: Dict[str, Any],
    *,
    branch: str,
) -> Dict[str, Any]:
    model_config = dict(model_config)
    branch_config = dict(model_config.get(branch) or {})
    encoder_config = dict(branch_config.get("encoder") or {})
    hidden_dim = int(encoder_config.get("hidden_dim", encoder_config.get("hid_dim", 128)))
    pre_encoder = dict(encoder_config.get("pre_encoder") or {})
    pre_encoder.setdefault("enabled", True)
    pre_encoder.setdefault("hidden_dim", hidden_dim)
    pre_encoder.setdefault("activation", "relu")
    encoder_config["pre_encoder"] = pre_encoder
    branch_config["encoder"] = encoder_config
    model_config[branch] = branch_config
    return model_config


def _with_shared_dcrnn_encoder_defaults(
    model_config: Dict[str, Any],
    *,
    architecture_tag: str,
) -> Dict[str, Any]:
    model_config = dict(model_config)
    model_config.setdefault("architecture_tag", architecture_tag)
    model_config["encoder_layout"] = "shared"
    shared_encoder = dict(model_config.get("shared_encoder") or {})
    shared_encoder.setdefault("type", "dcrnn")
    shared_encoder.setdefault("hidden_dim", 128)
    shared_encoder.setdefault("max_diffusion_step", 2)
    shared_encoder.setdefault("num_rnn_layers", 1)
    shared_encoder.setdefault("filter_type", "dual_random_walk")
    model_config["shared_encoder"] = shared_encoder
    return model_config


def _with_shared_pre_encoder_defaults(model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(model_config)
    shared_encoder = dict(model_config.get("shared_encoder") or {})
    hidden_dim = int(shared_encoder.get("hidden_dim", shared_encoder.get("hid_dim", 128)))
    pre_encoder = dict(shared_encoder.get("pre_encoder") or {})
    pre_encoder.setdefault("enabled", True)
    pre_encoder.setdefault("hidden_dim", hidden_dim)
    pre_encoder.setdefault("activation", "relu")
    shared_encoder["pre_encoder"] = pre_encoder
    model_config["shared_encoder"] = shared_encoder
    return model_config


def _dcrnn_actor_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config = _with_dcrnn_encoder_defaults(
        model_config,
        branch="actor",
        architecture_tag=DCRNN_ACTOR_KIND,
    )
    model_config.update(graph_model_config)
    return model_config


def _dcrnn_full_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config = _with_dcrnn_encoder_defaults(
        model_config,
        branch="actor",
        architecture_tag=DCRNN_FULL_KIND,
    )
    model_config = _with_dcrnn_encoder_defaults(
        model_config,
        branch="critic",
        architecture_tag=DCRNN_FULL_KIND,
    )
    model_config.update(graph_model_config)
    return model_config


def _dcrnn_actor_mlp_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = _dcrnn_actor_model_config(params, graph_model_config)
    model_config["architecture_tag"] = DCRNN_ACTOR_MLP_KIND
    return _with_pre_encoder_defaults(model_config, branch="actor")


def _dcrnn_full_mlp_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = _dcrnn_full_model_config(params, graph_model_config)
    model_config["architecture_tag"] = DCRNN_FULL_MLP_KIND
    model_config = _with_pre_encoder_defaults(model_config, branch="actor")
    model_config = _with_pre_encoder_defaults(model_config, branch="critic")
    return model_config


def _dcrnn_shared_mlp_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config = _with_shared_dcrnn_encoder_defaults(
        model_config,
        architecture_tag=DCRNN_SHARED_MLP_KIND,
    )
    model_config = _with_shared_pre_encoder_defaults(model_config)
    model_config.update(graph_model_config)
    return model_config


def build_graph_eval_env(
    cfg: Any,
    run_dir: Path,
    seed: Optional[int] = None,
    *,
    use_libsumo: Optional[bool] = None,
):
    from sumo_rl.environment.graph_env import build_rllib_graph_parallel_env

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    return build_rllib_graph_parallel_env(cfg, run_dir, seed=seed, params=graph_params(params), use_libsumo=use_libsumo)


def build_config(cfg: Any, run_dir: Path, *, algorithm_kind: str):
    from ray.rllib.algorithms.sac import SACConfig

    algorithm_kind = normalize_kind(algorithm_kind)
    if algorithm_kind in GRAPH_KINDS:
        if algorithm_kind == DCRNN_ACTOR_KIND:
            model_config_builder = _dcrnn_actor_model_config
        elif algorithm_kind == DCRNN_ACTOR_MLP_KIND:
            model_config_builder = _dcrnn_actor_mlp_model_config
        elif algorithm_kind == DCRNN_FULL_KIND:
            model_config_builder = _dcrnn_full_model_config
        elif algorithm_kind == DCRNN_SHARED_MLP_KIND:
            model_config_builder = _dcrnn_shared_mlp_model_config
        else:
            model_config_builder = _dcrnn_full_mlp_model_config
        context, policy_model_configs = build_graph_algorithm_context(
            cfg,
            run_dir,
            algorithm_kind=algorithm_kind,
            model_config_builder=model_config_builder,
        )
    else:
        context = build_algorithm_context(cfg, run_dir, algorithm_kind)
        policy_model_configs = None
    callbacks_class = training_episode_summary_callbacks_class()
    params = dict(context.params)
    params["replay_buffer_config"] = build_replay_buffer_config(params)
    custom_model_config = (
        (
            _dcrnn_actor_model_config(params, {})
            if algorithm_kind == DCRNN_ACTOR_KIND
            else _dcrnn_actor_mlp_model_config(params, {})
            if algorithm_kind == DCRNN_ACTOR_MLP_KIND
            else _dcrnn_full_model_config(params, {})
            if algorithm_kind == DCRNN_FULL_KIND
            else _dcrnn_shared_mlp_model_config(params, {})
            if algorithm_kind == DCRNN_SHARED_MLP_KIND
            else _dcrnn_full_mlp_model_config(params, {})
        )
        if algorithm_kind in GRAPH_KINDS
        else params.get("model_config")
    )
    if algorithm_kind in {CUSTOM_KIND, *GRAPH_KINDS}:
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

    if algorithm_kind in {CUSTOM_KIND, *GRAPH_KINDS}:
        rl_module_specs = {
            policy_id: build_custom_sac_module_spec(
                policy_spec.observation_space,
                policy_spec.action_space,
                model_config=(
                    policy_model_configs[policy_id]
                    if policy_model_configs is not None
                    else custom_model_config
                ),
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
    algorithm_kind = normalize_kind(algorithm_kind)
    metrics = extract_rllib_result_metrics(result, algorithm_kind=algorithm_kind, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/sac/learners", out=metrics)
        entropy_mean = extract_entropy_mean(learner_metrics)
        if entropy_mean is not None:
            metrics["train/sac/entropy_mean"] = float(entropy_mean)
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
    algorithm_kind = normalize_kind(algorithm_kind)
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
                f"[{algorithm_kind}] RLlib episode jump detected: +{progress_jump} "
                f"(from {completed_episodes - progress_jump} to {completed_episodes}) "
                f"at iteration={iteration}"
            )
        print(
            f"[{algorithm_kind}] episode={min(completed_episodes, training_episode_target(cfg))}/"
            f"{training_episode_target(cfg)} iteration={iteration} "
            f"result_keys={sorted(result.keys())[:8]}",
            flush=True,
        )
        if is_final:
            break
