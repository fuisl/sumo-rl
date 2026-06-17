"""FGSv3-specific RLlib config, training loop, and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ray.rllib.policy.policy import PolicySpec

from sumo_rl.agents.fgs.graph_env import FGSGraphParallelEnv
from sumo_rl.agents.fgsv3.learner import FGSv3SACTorchLearner
from sumo_rl.agents.fgsv3.rllib_module import (
    build_fgsv3_ppo_module_spec,
    build_fgsv3_ppo_multi_module_spec,
    build_fgsv3_sac_module_spec,
    build_fgsv3_sac_multi_module_spec,
    normalize_fgsv3_model_config,
)
from sumo_rl.agents.rllib_common import (
    RllibAlgorithmContext,
    apply_env_runner_settings,
    apply_multi_agent_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    build_sumo_parallel_env,
    episode_seconds,
    episode_steps,
    plain_dict,
    scenario_factory_name,
    training_episode_summary_callbacks_class,
)
from sumo_rl.agents.ppo import ppo as ppo_agent
from sumo_rl.agents.sac import sac as sac_agent
from sumo_rl.experiments.runner import _prepare_env_kwargs


KIND = "fgsv3"
PPO_KIND = "fgsv3_ppo"


def _fgsv3_model_config(params: Dict[str, Any]) -> Dict[str, Any]:
    model_config = normalize_fgsv3_model_config(params.get("model_config") or {})
    for key in ("twin_q",):
        if key in params and params[key] is not None:
            model_config[key] = bool(params[key])
    return normalize_fgsv3_model_config(model_config)


def build_fgsv3_parallel_env(cfg: Any, run_dir: Path, model_config: Dict[str, Any], seed: Optional[int] = None):
    import sumo_rl

    kwargs = _prepare_env_kwargs(cfg, run_dir)
    seconds = episode_seconds(cfg)
    if seconds > 0 and "num_seconds" not in kwargs:
        kwargs["num_seconds"] = seconds
    if seed is not None:
        kwargs["sumo_seed"] = int(seed)
    kwargs["single_agent"] = False

    factory = str(getattr(getattr(cfg, "env", None), "factory", "parallel_env") or "parallel_env")
    if factory in {"parallel_env", "env"}:
        env = sumo_rl.parallel_env(**kwargs)
    else:
        constructor = getattr(sumo_rl, factory, None)
        env = constructor(parallel=True, **kwargs) if constructor is not None else build_sumo_parallel_env(cfg, run_dir, seed=seed)

    topology_config = dict(model_config.get("topology", {}) or {})
    render_dir = run_dir / "topology" if bool(topology_config.get("render", True)) else None
    return FGSGraphParallelEnv(
        env,
        net_file=str(kwargs.get("net_file", "")),
        topology_source=str(topology_config.get("source", "tls_super_edges")),
        render_topology_dir=render_dir,
    )


def build_eval_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {})
    return ParallelPettingZooEnv(build_fgsv3_parallel_env(cfg, run_dir, _fgsv3_model_config(params), seed=seed))


def _build_fgsv3_context(
    cfg: Any,
    run_dir: Path,
    params: Dict[str, Any],
    *,
    algorithm_kind: str = KIND,
) -> RllibAlgorithmContext:
    mode = str(params.get("policy_mode", "shared") or "shared").strip().lower()
    if mode != "shared":
        raise ValueError("FGSv3 must use algorithm.params.policy_mode=shared for decentralized actor parameter sharing.")

    model_config = _fgsv3_model_config(params)
    sample_env = build_fgsv3_parallel_env(
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
    finally:
        sample_env.close()

    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    env_name = f"sumo_rl_{scenario_factory_name(cfg)}_{algorithm_kind}"

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            experiment = getattr(cfg, "experiment", None)
            seed = int(getattr(experiment, "seed", 0) or 0) + int(env_config.get("worker_index", 0) or 0)
        return ParallelPettingZooEnv(build_fgsv3_parallel_env(cfg, run_dir, model_config, seed=seed))

    register_env(env_name, _creator)
    return RllibAlgorithmContext(
        cfg=cfg,
        run_dir=run_dir,
        algorithm_kind=algorithm_kind,
        params=params,
        policy_mode="shared",
        env_name=env_name,
        policies=policies,
        active_policies={"shared_policy": shared_spec},
        episode_seconds=episode_seconds(cfg),
        episode_steps=episode_steps(cfg),
    )


def build_config(cfg: Any, run_dir: Path, *, algorithm_kind: str = KIND):
    algorithm_kind = str(algorithm_kind or KIND).strip()
    if algorithm_kind == PPO_KIND:
        return build_ppo_config(cfg, run_dir)
    if algorithm_kind != KIND:
        raise ValueError(f"Unsupported FGSv3 algorithm kind: {algorithm_kind}")
    return build_sac_config(cfg, run_dir)


def build_sac_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.sac import SACConfig

    callbacks_class = training_episode_summary_callbacks_class()
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    params = dict(params)
    params.setdefault("policy_mode", "shared")
    params.setdefault("twin_q", True)
    params.setdefault("n_step", 1)
    if int(params.get("n_step", 1)) != 1:
        raise ValueError("FGSv3 factored_neighborhood critic requires algorithm.params.n_step=1.")
    params.setdefault("replay_buffer_type", "MultiAgentPrioritizedEpisodeReplayBuffer")
    params["replay_buffer_config"] = sac_agent.build_replay_buffer_config(params)
    params["model_config"] = _fgsv3_model_config(params)
    if "num_steps_sampled_before_learning_starts" in params:
        params["num_steps_sampled_before_learning_starts"] = max(
            int(params["num_steps_sampled_before_learning_starts"]),
            episode_steps(cfg) + 1,
        )

    context = _build_fgsv3_context(cfg, run_dir, params, algorithm_kind=KIND)
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

    rl_module_specs = {
        policy_id: build_fgsv3_sac_module_spec(
            policy_spec.observation_space,
            policy_spec.action_space,
            model_config=params["model_config"],
        )
        for policy_id, policy_spec in context.active_policies.items()
    }
    config = config.rl_module(
        rl_module_spec=build_fgsv3_sac_multi_module_spec(rl_module_specs, model_config=params["model_config"])
    )
    learners = getattr(config, "learners", None)
    if callable(learners):
        config = config.learners(learner_class=FGSv3SACTorchLearner)
    else:
        config = config.training(learner_class=FGSv3SACTorchLearner)
    return config.callbacks(callbacks_class)


def build_ppo_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.ppo import PPOConfig

    callbacks_class = training_episode_summary_callbacks_class()
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    params = dict(params)
    params.setdefault("policy_mode", "shared")
    params["model_config"] = _fgsv3_model_config(params)

    context = _build_fgsv3_context(cfg, run_dir, params, algorithm_kind=PPO_KIND)
    config = PPOConfig().framework("torch").environment(env=context.env_name, disable_env_checking=True)
    config = apply_env_runner_settings(config, params)
    config = apply_training_settings(
        config,
        params,
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
            "vf_loss_coeff",
            "vf_clip_param",
        ),
        aliases={
            "num_sgd_iter": "num_epochs",
            "sgd_minibatch_size": "minibatch_size",
        },
    )
    config = apply_multi_agent_settings(config, context)
    config = apply_standard_evaluation_settings(config, params)

    rl_module_specs = {
        policy_id: build_fgsv3_ppo_module_spec(
            policy_spec.observation_space,
            policy_spec.action_space,
            model_config=params["model_config"],
        )
        for policy_id, policy_spec in context.active_policies.items()
    }
    config = config.rl_module(
        rl_module_spec=build_fgsv3_ppo_multi_module_spec(rl_module_specs, model_config=params["model_config"])
    )
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    return sac_agent.extract_training_metrics(result, iteration, algorithm_kind=KIND)


def train(
    algo,
    cfg: Any,
    *,
    algorithm_kind: str = KIND,
    emit_metrics: Optional[Callable[[Dict[str, Any], int], None]] = None,
    validate: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> None:
    algorithm_kind = str(algorithm_kind or KIND).strip()
    if algorithm_kind == PPO_KIND:
        return ppo_agent.train(algo, cfg, algorithm_kind=PPO_KIND, emit_metrics=emit_metrics, validate=validate)
    return sac_agent.train(algo, cfg, algorithm_kind=KIND, emit_metrics=emit_metrics, validate=validate)
