"""CoLight-specific RLlib config, training loop, and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ray.rllib.policy.policy import PolicySpec

from sumo_rl.agents.colight.graph_env import CoLightGraphParallelEnv, make_colight_observation_class
from sumo_rl.agents.colight.topology import render_colight_topology
from sumo_rl.agents.dqn.dqn import build_replay_buffer_config
from sumo_rl.agents.rllib_common import (
    RllibAlgorithmContext,
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    build_sumo_parallel_env,
    completed_training_episodes,
    emit_training_episode_rows,
    emit_validation_if_due,
    flatten_numeric_metrics,
    plain_dict,
    register_multi_agent_env,
    rllib_counter_metrics,
    scenario_factory_name,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
    episode_seconds,
    episode_steps,
)
from sumo_rl.experiments.runner import _prepare_env_kwargs


KIND = "colight"


def _colight_model_config(params: Dict[str, Any]) -> Dict[str, Any]:
    model_config = dict(params.get("model_config") or {})
    model_config.setdefault("architecture_tag", "colight_graph_attention")
    model_config.setdefault("include_phase", True)
    model_config.setdefault("phase_encoding", "one_hot")
    model_config.setdefault("vehicle_max", 1.0)
    model_config.setdefault("node_embedding_dims", [128, 128])
    model_config.setdefault("num_gat_layers", 1)
    model_config.setdefault("num_heads", 5)
    model_config.setdefault("head_dim", 16)
    model_config.setdefault("gat_output_dim", 128)
    model_config.setdefault("output_layers", [])
    for key in ("double_q", "dueling", "epsilon", "num_atoms", "v_min", "v_max"):
        if key in params and params[key] is not None:
            model_config[key] = params[key]
    return model_config


def _with_colight_observation(cfg: Any, run_dir: Path, model_config: Dict[str, Any], seed: Optional[int] = None):
    import sumo_rl

    kwargs = _prepare_env_kwargs(cfg, run_dir)
    seconds = episode_seconds(cfg)
    if seconds > 0 and "num_seconds" not in kwargs:
        kwargs["num_seconds"] = seconds
    if seed is not None:
        kwargs["sumo_seed"] = int(seed)
    kwargs["single_agent"] = False
    kwargs["observation_class"] = make_colight_observation_class(
        include_phase=bool(model_config.get("include_phase", True)),
        phase_encoding=str(model_config.get("phase_encoding", "one_hot")),
        vehicle_max=float(model_config.get("vehicle_max", 1.0)),
    )

    factory = str(getattr(getattr(cfg, "env", None), "factory", "parallel_env") or "parallel_env")
    if factory == "parallel_env":
        return sumo_rl.parallel_env(**kwargs)
    if factory == "env":
        return sumo_rl.parallel_env(**kwargs)
    constructor = getattr(sumo_rl, factory, None)
    if constructor is None:
        return build_sumo_parallel_env(cfg, run_dir, seed=seed)
    return constructor(parallel=True, **kwargs)


def build_colight_parallel_env(cfg: Any, run_dir: Path, model_config: Dict[str, Any], seed: Optional[int] = None):
    return CoLightGraphParallelEnv(_with_colight_observation(cfg, run_dir, model_config, seed=seed))


def build_eval_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
    model_config = _colight_model_config(params)
    return ParallelPettingZooEnv(build_colight_parallel_env(cfg, run_dir, model_config, seed=seed))


def _build_colight_context(cfg: Any, run_dir: Path, params: Dict[str, Any]) -> RllibAlgorithmContext:
    mode = str(params.get("policy_mode", "shared") or "shared").strip().lower()
    if mode != "shared":
        raise ValueError("CoLight must use algorithm.params.policy_mode=shared to preserve graph-level cooperation.")

    model_config = _colight_model_config(params)
    sample_env = build_colight_parallel_env(
        cfg,
        run_dir,
        model_config,
        seed=int(getattr(getattr(cfg, "experiment", None), "seed", 0) or 0),
    )
    try:
        first_agent = sample_env.possible_agents[0]
        shared_spec = PolicySpec(
            observation_space=sample_env.observation_space(first_agent),
            action_space=sample_env.action_space(first_agent),
        )
        policies = {
            agent_id: PolicySpec(
                observation_space=sample_env.observation_space(agent_id),
                action_space=sample_env.action_space(agent_id),
            )
            for agent_id in sample_env.possible_agents
        }
        if bool(params.get("render_topology", True)):
            env_kwargs = _prepare_env_kwargs(cfg, run_dir)
            net_file = str(env_kwargs.get("net_file", ""))
            if net_file:
                render_paths = render_colight_topology(sample_env, net_file, run_dir / "topology")
                print(f"[{KIND}] wrote topology overlay to {render_paths['svg']}")
    finally:
        sample_env.close()

    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    env_name = f"sumo_rl_{scenario_factory_name(cfg)}_{KIND}"

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            experiment = getattr(cfg, "experiment", None)
            seed = int(getattr(experiment, "seed", 0) or 0) + int(env_config.get("worker_index", 0) or 0)
        return ParallelPettingZooEnv(build_colight_parallel_env(cfg, run_dir, model_config, seed=seed))

    register_env(env_name, _creator)
    return RllibAlgorithmContext(
        cfg=cfg,
        run_dir=run_dir,
        algorithm_kind=KIND,
        params=params,
        policy_mode="shared",
        env_name=env_name,
        policies=policies,
        active_policies={"shared_policy": shared_spec},
        episode_seconds=episode_seconds(cfg),
        episode_steps=episode_steps(cfg),
    )


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from sumo_rl.agents.colight.rllib_module import build_colight_dqn_module_spec

    callbacks_class = training_episode_summary_callbacks_class()
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    params = dict(params)
    params.setdefault("policy_mode", "shared")
    params["replay_buffer_config"] = build_replay_buffer_config(params)
    params.setdefault("dueling", False)
    params.setdefault("double_q", True)
    params.setdefault("num_atoms", 1)
    params.setdefault("epsilon", [(0, 0.8), (100000, 0.01)])
    if "num_steps_sampled_before_learning_starts" in params:
        params["num_steps_sampled_before_learning_starts"] = max(
            int(params["num_steps_sampled_before_learning_starts"]),
            episode_steps(cfg) + 1,
        )

    context = _build_colight_context(cfg, run_dir, params)
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

    base_model_config = _colight_model_config(params)
    rl_module_specs = {
        policy_id: build_colight_dqn_module_spec(
            policy_spec.observation_space,
            policy_spec.action_space,
            model_config=base_model_config,
        )
        for policy_id, policy_spec in context.active_policies.items()
    }
    config = config.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs))
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = rllib_counter_metrics(result, algorithm_kind=KIND, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/colight/learners", out=metrics)
    replay_metrics = result.get("replay_buffer") or result.get("replay_buffers")
    if isinstance(replay_metrics, dict):
        flatten_numeric_metrics(replay_metrics, prefix="train/colight/replay", out=metrics)
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
