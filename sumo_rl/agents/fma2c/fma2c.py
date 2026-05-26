"""FMA2C-specific RLlib config, environment registration, and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sumo_rl.agents.fma2c.env import is_manager_agent, wrap_fma2c_env
from sumo_rl.agents.rllib_common import (
    apply_env_runner_settings,
    apply_standard_evaluation_settings,
    apply_training_settings,
    build_sumo_parallel_env,
    completed_training_episodes,
    emit_training_episode_rows,
    emit_validation_if_due,
    flatten_numeric_metrics,
    plain_dict,
    extract_rllib_result_metrics,
    scenario_factory_name,
    training_episode_summary_callbacks_class,
    training_episode_target,
    training_should_stop,
)


KIND = "fma2c"


def _fma2c_params(cfg: Any) -> Dict[str, Any]:
    params = plain_dict(getattr(getattr(cfg, "algorithm", None), "params", {}) or {}) or {}
    params.setdefault("policy_mode", "independent")
    return params


def _net_file_from_cfg(cfg: Any) -> Optional[str]:
    env_cfg = getattr(cfg, "env", None)
    kwargs = getattr(env_cfg, "kwargs", None) if env_cfg is not None else None
    if kwargs is None:
        return None
    if isinstance(kwargs, dict):
        return kwargs.get("net_file")
    return getattr(kwargs, "net_file", None)


def build_fma2c_parallel_env(cfg: Any, run_dir: Path, seed: Optional[int] = None):
    base_env = build_sumo_parallel_env(cfg, run_dir, seed=seed)
    return wrap_fma2c_env(base_env, params=_fma2c_params(cfg), net_file=_net_file_from_cfg(cfg))


def build_eval_env(cfg: Any, run_dir: Path, seed: Optional[int] = None, *, pad_spaces: bool = False):
    del pad_spaces
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    return ParallelPettingZooEnv(build_fma2c_parallel_env(cfg, run_dir, seed=seed))


def policy_id_for_agent(agent_id: str, mode: str) -> str:
    mode = str(mode or "independent").strip().lower()
    if mode == "shared_by_level":
        return "manager_policy" if is_manager_agent(agent_id) else "worker_policy"
    return str(agent_id)


def build_policy_mapping(mode: str) -> Callable[..., str]:
    def _mapping_fn(agent_id, *args, **kwargs):
        del args, kwargs
        return policy_id_for_agent(str(agent_id), mode)

    return _mapping_fn


def _build_policies(cfg: Any, run_dir: Path):
    from ray.rllib.policy.policy import PolicySpec

    experiment = getattr(cfg, "experiment", None)
    sample_env = build_fma2c_parallel_env(
        cfg,
        run_dir,
        seed=int(getattr(experiment, "seed", 0) or 0),
    )
    try:
        policies = {
            agent_id: PolicySpec(
                observation_space=sample_env.observation_space(agent_id),
                action_space=sample_env.action_space(agent_id),
            )
            for agent_id in sample_env.possible_agents
        }
    finally:
        sample_env.close()

    params = _fma2c_params(cfg)
    mode = str(params.get("policy_mode", "independent")).strip().lower()
    if mode != "shared_by_level":
        return policies, mode

    worker_specs = [spec for agent_id, spec in policies.items() if not is_manager_agent(agent_id)]
    manager_specs = [spec for agent_id, spec in policies.items() if is_manager_agent(agent_id)]
    if not worker_specs or not manager_specs:
        return policies, "independent"
    worker_space = worker_specs[0].observation_space
    worker_action = worker_specs[0].action_space
    manager_space = manager_specs[0].observation_space
    manager_action = manager_specs[0].action_space
    if any(spec.observation_space != worker_space or spec.action_space != worker_action for spec in worker_specs):
        raise ValueError("FMA2C shared_by_level requires homogeneous worker observation/action spaces.")
    if any(spec.observation_space != manager_space or spec.action_space != manager_action for spec in manager_specs):
        raise ValueError("FMA2C shared_by_level requires homogeneous manager observation/action spaces.")
    return {
        "worker_policy": PolicySpec(observation_space=worker_space, action_space=worker_action),
        "manager_policy": PolicySpec(observation_space=manager_space, action_space=manager_action),
    }, mode


def _register_env(cfg: Any, run_dir: Path) -> str:
    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    env_name = f"sumo_rl_{scenario_factory_name(cfg)}_{KIND}"

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            experiment = getattr(cfg, "experiment", None)
            seed = int(getattr(experiment, "seed", 0) or 0) + int(env_config.get("worker_index", 0) or 0)
        return ParallelPettingZooEnv(build_fma2c_parallel_env(cfg, run_dir, seed=seed))

    register_env(env_name, _creator)
    return env_name


def build_config(cfg: Any, run_dir: Path):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    params = _fma2c_params(cfg)
    policies, policy_mode = _build_policies(cfg, run_dir)
    callbacks_class = training_episode_summary_callbacks_class()
    env_name = _register_env(cfg, run_dir)

    experiment = getattr(cfg, "experiment", None)
    episode_seconds = int(getattr(experiment, "episode_seconds", 1) or 1)
    env_cfg = getattr(cfg, "env", None)
    env_kwargs = getattr(env_cfg, "kwargs", {}) if env_cfg is not None else {}
    delta_time = getattr(env_kwargs, "delta_time", None)
    if isinstance(env_kwargs, dict):
        delta_time = env_kwargs.get("delta_time", delta_time)
    episode_steps = max(1, episode_seconds // int(delta_time or 5))

    params.setdefault("lr", 2.5e-4)
    params.setdefault("gamma", 0.96)
    params.setdefault("lambda_", 1.0)
    params.setdefault("use_kl_loss", False)
    params.setdefault("entropy_coeff", 0.01)
    params.setdefault("vf_loss_coeff", 0.5)
    params.setdefault("grad_clip", 40.0)
    params.setdefault("num_epochs", 1)
    params.setdefault("train_batch_size_per_learner", min(120, episode_steps))
    params.setdefault("minibatch_size", min(120, episode_steps))

    config = PPOConfig().framework("torch").environment(env=env_name, disable_env_checking=True)
    config = apply_env_runner_settings(config, params)
    config = apply_training_settings(
        config,
        params,
        episode_steps_value=episode_steps,
        allowed_keys=(
            "lr",
            "gamma",
            "lambda_",
            "use_gae",
            "use_critic",
            "use_kl_loss",
            "kl_coeff",
            "clip_param",
            "vf_clip_param",
            "entropy_coeff",
            "vf_loss_coeff",
            "grad_clip",
            "train_batch_size_per_learner",
            "num_epochs",
            "minibatch_size",
        ),
        aliases={
            "batch_size": "train_batch_size_per_learner",
        },
    )
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=build_policy_mapping(policy_mode),
        policies_to_train=list(policies.keys()),
    )
    model_config = dict(params.get("model_config") or {})
    train_batch_size = min(int(params.get("train_batch_size_per_learner", episode_steps)), episode_steps)
    max_seq_len = max(1, min(int(model_config.get("max_seq_len", 20)), episode_steps, train_batch_size))
    default_model_config = DefaultModelConfig(
        fcnet_hiddens=list(model_config.get("fcnet_hiddens", [64])),
        fcnet_activation=str(model_config.get("fcnet_activation", "tanh")),
        use_lstm=bool(model_config.get("use_lstm", True)),
        max_seq_len=max_seq_len,
        lstm_cell_size=int(model_config.get("lstm_cell_size", 64)),
        vf_share_layers=bool(model_config.get("vf_share_layers", True)),
    )
    config = config.rl_module(model_config=default_model_config)
    config = apply_standard_evaluation_settings(config, params)
    return config.callbacks(callbacks_class)


def extract_training_metrics(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    metrics = extract_rllib_result_metrics(result, algorithm_kind=KIND, iteration=iteration)
    learner_metrics = result.get("learners") or result.get("learner")
    if isinstance(learner_metrics, dict):
        flatten_numeric_metrics(learner_metrics, prefix="train/fma2c/learners", out=metrics)
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
