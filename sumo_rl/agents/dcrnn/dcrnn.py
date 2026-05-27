"""DCRNN-specific RLlib config, training loop, and training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.dqn.dqn import build_replay_buffer_config
from sumo_rl.agents.rllib_common import (
    RllibAlgorithmContext,
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    completed_training_episodes,
    episode_seconds,
    episode_steps,
    emit_training_episode_rows,
    emit_validation_if_due,
    extract_rllib_result_metrics,
    flatten_numeric_metrics,
    plain_dict,
    policy_mode,
    scenario_factory_name,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)


KIND = "dcrnn"


def _graph_params(params: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    return {
        "history_len": int(params.get("history_len", model_config.get("history_len", 5))),
        "include_virtual_nodes": bool(params.get("include_virtual_nodes", model_config.get("include_virtual_nodes", True))),
        "add_self_loops": bool(params.get("add_self_loops", model_config.get("add_self_loops", True))),
    }


def _dcrnn_model_config(params: Dict[str, Any], graph_model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config.setdefault("architecture_tag", "dcrnn_dqn")
    model_config.setdefault("hid_dim", 128)
    model_config.setdefault("max_diffusion_step", 2)
    model_config.setdefault("num_rnn_layers", 1)
    model_config.setdefault("filter_type", "dual_random_walk")
    for key in ("double_q", "dueling", "epsilon", "num_atoms", "v_min", "v_max"):
        if key in params and params[key] is not None:
            model_config[key] = params[key]
    model_config.update(graph_model_config)
    return model_config


def build_graph_eval_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    from sumo_rl.environment.graph_env import build_rllib_graph_parallel_env

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    return build_rllib_graph_parallel_env(cfg, run_dir, seed=seed, params=_graph_params(params))


def _register_graph_env(cfg: Any, run_dir: Path, params: Dict[str, Any]) -> str:
    from ray.tune.registry import register_env
    from sumo_rl.environment.graph_env import build_rllib_graph_parallel_env

    env_name = f"sumo_rl_graph_{scenario_factory_name(cfg)}_{KIND}"
    graph_params = _graph_params(params)

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            experiment = getattr(cfg, "experiment", None)
            base_seed = int(getattr(experiment, "seed", 0) or 0)
            seed = base_seed + int(env_config.get("worker_index", 0) or 0)
        return build_rllib_graph_parallel_env(cfg, run_dir, seed=seed, params=graph_params)

    register_env(env_name, _creator)
    return env_name


def _build_graph_context(cfg: Any, run_dir: Path) -> tuple[RllibAlgorithmContext, Dict[str, Dict[str, Any]]]:
    from ray.rllib.policy.policy import PolicySpec
    from sumo_rl.environment.graph_env import build_graph_parallel_env

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    mode = policy_mode(params)
    if mode != "independent":
        raise ValueError("dcrnn currently supports algorithm.params.policy_mode=independent only.")

    experiment = getattr(cfg, "experiment", None)
    sample_env = build_graph_parallel_env(
        cfg,
        run_dir,
        seed=int(getattr(experiment, "seed", 0) or 0),
        params=_graph_params(params),
    )
    try:
        policies = {}
        model_configs = {}
        for agent_id in sample_env.possible_agents:
            policies[str(agent_id)] = PolicySpec(
                observation_space=sample_env.observation_space(agent_id),
                action_space=sample_env.action_space(agent_id),
            )
            model_configs[str(agent_id)] = _dcrnn_model_config(
                params,
                sample_env.graph.model_config(agent_id),
            )
    finally:
        sample_env.close()

    context = RllibAlgorithmContext(
        cfg=cfg,
        run_dir=run_dir,
        algorithm_kind=KIND,
        params=params,
        policy_mode=mode,
        env_name=_register_graph_env(cfg, run_dir, params),
        policies=policies,
        active_policies=policies,
        episode_seconds=episode_seconds(cfg),
        episode_steps=episode_steps(cfg),
    )
    return context, model_configs


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from sumo_rl.agents.dcrnn.rllib_module import build_dcrnn_dqn_module_spec

    context, model_configs = _build_graph_context(cfg, run_dir)
    callbacks_class = training_episode_summary_callbacks_class()
    params = dict(context.params)
    params["replay_buffer_config"] = build_replay_buffer_config(params)
    params.setdefault("dueling", False)
    params.setdefault("double_q", True)
    params.setdefault("num_atoms", 1)
    params.setdefault("epsilon", [(0, 0.1), (100000, 0.01)])
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
            "epsilon",
            "tau",
            "dueling",
            "double_q",
            "num_atoms",
            "td_error_loss_fn",
            "replay_buffer_config",
        ),
    )
    config = apply_multi_agent_settings(config, context)
    config = apply_standard_evaluation_settings(config, params)

    rl_module_specs = {
        policy_id: build_dcrnn_dqn_module_spec(
            policy_spec.observation_space,
            policy_spec.action_space,
            model_config=model_configs[policy_id],
        )
        for policy_id, policy_spec in context.active_policies.items()
    }
    config = config.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs))
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = extract_rllib_result_metrics(result, algorithm_kind=KIND, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/dcrnn/learners", out=metrics)
    replay_metrics = result.get("replay_buffer") or result.get("replay_buffers")
    if isinstance(replay_metrics, dict):
        flatten_numeric_metrics(replay_metrics, prefix="train/dcrnn/replay", out=metrics)
    return metrics


def train(
    algo,
    cfg: Any,
    *,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]] = None,
    validate: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> None:
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
