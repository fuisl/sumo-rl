from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig

# Ray computes its default storage path while importing modules, so point the
# process home directory at the workspace before any Ray import can happen.
os.environ["HOME"] = str(Path.cwd())
os.environ["USERPROFILE"] = str(Path.cwd())

from sumo_rl.experiments.runner import (
    _LocalMetricsCsvLogger,
    _aggregate_final_eval_rows,
    _build_final_eval_summary_row,
    _get_run_dir,
    _init_wandb,
    _log_episode_summary,
    _update_wandb_summary,
)
from sumo_rl.rllib.custom_sac import build_custom_sac_module_spec
from sumo_rl.rllib.envs import build_multi_agent_wrapper, scenario_factory_name


SUPPORTED_RLLIB_ALGORITHMS = {
    "ppo_rllib",
    "dqn_rllib",
    "sac_rllib_builtin",
    "sac_rllib_custom",
}


MULTI_AGENT_RLLIB_ALGORITHMS = {
    "ppo_rllib",
    "dqn_rllib",
    "sac_rllib_builtin",
    "sac_rllib_custom",
}


def _plain_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return dict(cfg)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _eval_seeds(cfg: DictConfig) -> list[int]:
    explicit = _plain_dict(getattr(cfg.experiment, "eval_seeds", None))
    eval_episodes = int(getattr(cfg.experiment, "eval_episodes", 0) or 0)
    if isinstance(explicit, list) and explicit:
        seeds = [int(seed) for seed in explicit]
        if eval_episodes > 0:
            return seeds[:eval_episodes]
        return seeds
    base_seed = int(cfg.experiment.seed) if getattr(cfg.experiment, "seed", None) is not None else 0
    count = max(1, eval_episodes or 1)
    return [base_seed + index for index in range(count)]


def _register_env(cfg: DictConfig, run_dir: Path, algorithm_kind: str, *, pad_spaces: bool = False):
    from ray.tune.registry import register_env

    env_name = f"sumo_rl_{scenario_factory_name(cfg)}_{algorithm_kind}"

    def _creator(env_config):
        env_config = dict(env_config or {})
        seed = env_config.get("seed")
        if seed is None:
            base_seed = int(cfg.experiment.seed) if getattr(cfg.experiment, "seed", None) is not None else 0
            seed = base_seed + int(env_config.get("worker_index", 0) or 0)
        if algorithm_kind in MULTI_AGENT_RLLIB_ALGORITHMS:
            return build_multi_agent_wrapper(cfg, run_dir, seed=seed, pad_spaces=pad_spaces)
        raise ValueError(f"Unsupported RLlib environment kind: {algorithm_kind}")

    register_env(env_name, _creator)
    return env_name


def _training_iterations(cfg: DictConfig) -> int:
    params = _plain_dict(getattr(cfg.algorithm, "params", {}) or {})
    value = params.get("train_iterations")
    if value is not None:
        return max(1, int(value))
    total_timesteps = int(getattr(cfg.experiment, "total_timesteps", 1) or 1)
    return max(1, min(20, total_timesteps // 1000 or 1))


def _representative_spaces(env) -> Tuple[Any, Any]:
    if hasattr(env, "observation_spaces") and hasattr(env, "action_spaces"):
        agent_id = next(iter(env.observation_spaces.keys()))
        return env.observation_spaces[agent_id], env.action_spaces[agent_id]
    return env.observation_space, env.action_space


def _policy_mode(params: Dict[str, Any]) -> str:
    return str(params.get("policy_mode", "independent") or "independent").strip().lower()


def _policy_id_for_agent(agent_id: str, policy_mode: str) -> str:
    if policy_mode == "shared":
        return "shared_policy"
    return str(agent_id)


def _build_policy_mapping(policy_mode: str):
    def _mapping_fn(agent_id, *args, **kwargs):
        del args, kwargs
        return _policy_id_for_agent(str(agent_id), policy_mode)

    return _mapping_fn


def _build_multi_agent_policies(
    cfg: DictConfig,
    run_dir: Path,
    params: Dict[str, Any],
    *,
    pad_spaces: bool,
):
    from ray.rllib.policy.policy import PolicySpec

    sample_env = build_multi_agent_wrapper(
        cfg,
        run_dir,
        seed=int(getattr(cfg.experiment, "seed", 0) or 0),
        pad_spaces=pad_spaces,
    )
    policies = {}
    for agent_id in sample_env.possible_agents:
        policies[str(agent_id)] = PolicySpec(
            observation_space=sample_env.observation_space(agent_id),
            action_space=sample_env.action_space(agent_id),
        )
    sample_env.close()
    return policies


def _build_shared_policy_dict(policies: Dict[str, Any]) -> Dict[str, Any]:
    first_spec = next(iter(policies.values()))
    return {"shared_policy": first_spec}


def _apply_common_env_runner_settings(config, params: Dict[str, Any]):
    num_env_runners = int(params.get("num_env_runners", 0) or 0)
    num_envs_per_runner = int(params.get("num_envs_per_env_runner", 1) or 1)
    learners = params.get("num_learners")
    if hasattr(config, "env_runners"):
        config = config.env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_runner,
        )
    if learners is not None and hasattr(config, "learners"):
        config = config.learners(num_learners=int(learners))
    return config


def _total_timesteps(cfg: DictConfig) -> int:
    return max(1, int(getattr(cfg.experiment, "total_timesteps", 1) or 1))


def _cap_to_horizon(value: Any, horizon: int) -> int:
    return max(1, min(int(value), int(horizon)))


def _apply_common_training_settings(config, params: Dict[str, Any], *, total_timesteps: int):
    training_kwargs: Dict[str, Any] = {}
    for key in (
        "lr",
        "gamma",
        "lambda_",
        "clip_param",
        "entropy_coeff",
        "grad_clip",
        "train_batch_size_per_learner",
        "num_epochs",
        "minibatch_size",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "tau",
        "initial_alpha",
        "target_entropy",
        "n_step",
        "training_intensity",
        "num_steps_sampled_before_learning_starts",
        "target_network_update_freq",
        "twin_q",
        "clip_actions",
    ):
        if key in params and params[key] is not None:
            training_kwargs[key] = params[key]
    if "train_batch_size_per_learner" in training_kwargs:
        training_kwargs["train_batch_size_per_learner"] = _cap_to_horizon(
            training_kwargs["train_batch_size_per_learner"],
            total_timesteps,
        )
    if "num_sgd_iter" in params and params["num_sgd_iter"] is not None:
        training_kwargs["num_epochs"] = params["num_sgd_iter"]
    if "sgd_minibatch_size" in params and params["sgd_minibatch_size"] is not None:
        training_kwargs["minibatch_size"] = _cap_to_horizon(
            params["sgd_minibatch_size"],
            int(training_kwargs.get("train_batch_size_per_learner", total_timesteps)),
        )
    if training_kwargs and hasattr(config, "training"):
        config = config.training(**training_kwargs)
    if hasattr(config, "reporting"):
        config = config.reporting(min_sample_timesteps_per_iteration=total_timesteps)
    return config


def _build_algorithm_config(cfg: DictConfig, run_dir: Path, algorithm_kind: str):
    params = _plain_dict(getattr(cfg.algorithm, "params", {}) or {})
    policy_mode = _policy_mode(params)
    env_name = _register_env(cfg, run_dir, algorithm_kind, pad_spaces=(policy_mode == "shared"))
    total_timesteps = _total_timesteps(cfg)

    if algorithm_kind == "ppo_rllib":
        from ray.rllib.algorithms.ppo import PPOConfig
        policies = _build_multi_agent_policies(cfg, run_dir, params, pad_spaces=(policy_mode == "shared"))

        config = PPOConfig().framework("torch").environment(env_name)
        config = _apply_common_env_runner_settings(config, params)
        config = _apply_common_training_settings(config, params, total_timesteps=total_timesteps)
        config = config.multi_agent(
            policies=(_build_shared_policy_dict(policies) if policy_mode == "shared" else policies),
            policy_mapping_fn=_build_policy_mapping(policy_mode),
            policies_to_train=(["shared_policy"] if policy_mode == "shared" else list(policies.keys())),
        )
        return config

    if algorithm_kind == "dqn_rllib":
        from ray.rllib.algorithms.dqn import DQNConfig
        policies = _build_multi_agent_policies(cfg, run_dir, params, pad_spaces=(policy_mode == "shared"))

        config = DQNConfig().framework("torch").environment(env_name)
        dqn_params = dict(params)
        if "num_steps_sampled_before_learning_starts" in dqn_params:
            dqn_params["num_steps_sampled_before_learning_starts"] = max(
                int(dqn_params["num_steps_sampled_before_learning_starts"]),
                total_timesteps + 1,
            )
        config = _apply_common_env_runner_settings(config, dqn_params)
        config = _apply_common_training_settings(config, dqn_params, total_timesteps=total_timesteps)
        config = config.multi_agent(
            policies=(_build_shared_policy_dict(policies) if policy_mode == "shared" else policies),
            policy_mapping_fn=_build_policy_mapping(policy_mode),
            policies_to_train=(["shared_policy"] if policy_mode == "shared" else list(policies.keys())),
        )
        return config

    if algorithm_kind == "sac_rllib_builtin":
        from ray.rllib.algorithms.sac import SACConfig
        policies = _build_multi_agent_policies(cfg, run_dir, params, pad_spaces=(policy_mode == "shared"))

        config = SACConfig().framework("torch").environment(env_name)
        config = _apply_common_env_runner_settings(config, params)
        config = _apply_common_training_settings(config, params, total_timesteps=total_timesteps)
        config = config.multi_agent(
            policies=(_build_shared_policy_dict(policies) if policy_mode == "shared" else policies),
            policy_mapping_fn=_build_policy_mapping(policy_mode),
            policies_to_train=(["shared_policy"] if policy_mode == "shared" else list(policies.keys())),
        )
        return config

    if algorithm_kind == "sac_rllib_custom":
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

        policies = _build_multi_agent_policies(cfg, run_dir, params, pad_spaces=(policy_mode == "shared"))
        active_policies = _build_shared_policy_dict(policies) if policy_mode == "shared" else policies
        rl_module_specs = {
            policy_id: build_custom_sac_module_spec(
                policy_spec.observation_space,
                policy_spec.action_space,
                model_config=params.get("model_config"),
            )
            for policy_id, policy_spec in active_policies.items()
        }

        config = SACConfig().framework("torch").environment(env_name)
        config = _apply_common_env_runner_settings(config, params)
        config = _apply_common_training_settings(config, params, total_timesteps=total_timesteps)
        config = config.multi_agent(
            policies=active_policies,
            policy_mapping_fn=_build_policy_mapping(policy_mode),
            policies_to_train=list(active_policies.keys()),
        )
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs)
        )
        return config

    raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")


def _compute_single_action(algo, obs, *, policy_id: Optional[str] = None):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        module = get_module(policy_id) if policy_id is not None else get_module()
        if module is not None and hasattr(module, "forward_inference"):
            import torch
            from ray.rllib.core.columns import Columns

            obs_batch = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = module.forward_inference({Columns.OBS: obs_batch})
                if Columns.ACTIONS not in output and Columns.ACTION_DIST_INPUTS in output:
                    action_dist = module.get_inference_action_dist_cls().from_logits(
                        output[Columns.ACTION_DIST_INPUTS]
                    )
                    output[Columns.ACTIONS] = action_dist.to_deterministic().sample()
            action = output[Columns.ACTIONS]
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            return np.asarray(action).reshape(-1)[0].item()

    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        if policy_id is None:
            action = compute_single_action(obs, explore=False)
        else:
            action = compute_single_action(obs, policy_id=policy_id, explore=False)
        return action[0] if isinstance(action, tuple) else action

    policy = algo.get_policy(policy_id) if policy_id else algo.get_policy()
    action = policy.compute_single_action(obs, explore=False)
    return action[0] if isinstance(action, tuple) else action


def _run_multi_agent_episode(algo, env, seed: int, *, policy_mode: str) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        actions = {}
        for agent_id, agent_obs in obs.items():
            if agent_id.startswith("__"):
                continue
            actions[agent_id] = _compute_single_action(
                algo,
                agent_obs,
                policy_id=_policy_id_for_agent(str(agent_id), policy_mode),
            )
        obs, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += float(sum(float(value) for value in rewards.values()))
        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or all(bool(terminations.get(agent_id, False)) for agent_id in env.possible_agents)
            or all(bool(truncations.get(agent_id, False)) for agent_id in env.possible_agents)
        )
    return total_reward


def _evaluate(
    cfg: DictConfig,
    run_dir: Path,
    algo,
    algorithm_kind: str,
    logging_cfg,
    *,
    wandb_run=None,
    csv_run=None,
) -> Dict[str, Any]:
    seed_rows = []
    eval_seeds = _eval_seeds(cfg)
    policy_mode = _policy_mode(_plain_dict(getattr(cfg.algorithm, "params", {}) or {}))
    for seed_index, seed in enumerate(eval_seeds):
        eval_env = build_multi_agent_wrapper(
            cfg,
            run_dir,
            seed=seed,
            pad_spaces=(policy_mode == "shared"),
        )
        episode_reward = _run_multi_agent_episode(algo, eval_env, seed, policy_mode=policy_mode)

        seed_row = _build_final_eval_summary_row(
            eval_env,
            algorithm_kind=algorithm_kind,
            eval_mean_reward=float(episode_reward),
            eval_std_reward=0.0,
            eval_episodes=1,
            logging_cfg=logging_cfg,
            extra={"eval/seed": float(seed), "eval/seed_index": float(seed_index)},
        )
        seed_rows.append(seed_row)
        _log_episode_summary(
            wandb_run,
            csv_run,
            seed_row,
            step=seed_index,
            logging_cfg=logging_cfg,
        )
        eval_env.close()

    eval_mean_reward = float(np.mean([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    eval_std_reward = float(np.std([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    return _aggregate_final_eval_rows(
        seed_rows,
        algorithm_kind=algorithm_kind,
        eval_mean_reward=eval_mean_reward,
        eval_std_reward=eval_std_reward,
        eval_episodes=len(eval_seeds),
    )


def train_rllib(cfg: DictConfig) -> Dict[str, Any]:
    algorithm_kind = str(getattr(cfg.algorithm, "kind", "") or "").strip()
    if algorithm_kind not in SUPPORTED_RLLIB_ALGORITHMS:
        raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")

    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg = cfg.logging
    wandb_run = _init_wandb(cfg, run_dir)
    csv_run = _LocalMetricsCsvLogger(run_dir / "csv" / f"{cfg.experiment.name}.csv")

    import ray

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    algo = None
    final_summary: Dict[str, Any] = {}
    try:
        config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else config.build()

        for iteration in range(_training_iterations(cfg)):
            result = algo.train()
            print(
                f"[{algorithm_kind}] iteration={iteration + 1} "
                f"result_keys={sorted(result.keys())[:8]}"
            )

        final_summary = _evaluate(
            cfg,
            run_dir,
            algo,
            algorithm_kind,
            logging_cfg,
            wandb_run=wandb_run,
            csv_run=csv_run,
        )
        _log_episode_summary(wandb_run, csv_run, final_summary, step=int(getattr(cfg.experiment, "total_timesteps", 0) or 0), logging_cfg=logging_cfg)
        _update_wandb_summary(wandb_run, final_summary)

        if bool(getattr(logging_cfg, "save_final_model", True)):
            checkpoint_dir = run_dir / "checkpoints" / algorithm_kind
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = None
            if hasattr(algo, "save_to_path"):
                checkpoint = algo.save_to_path(str(checkpoint_dir))
            elif hasattr(algo, "save"):
                checkpoint = algo.save(str(checkpoint_dir))
            if checkpoint is not None:
                print(f"[{algorithm_kind}] saved checkpoint to {checkpoint}")
        return final_summary
    finally:
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
