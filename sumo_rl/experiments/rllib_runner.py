from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
    _log_outputs,
    _log_episode_summary,
    _resolve_num_gpus,
    _update_wandb_summary,
)
from sumo_rl.agents.colight import colight as colight_agent
from sumo_rl.agents.dqn import dqn as dqn_agent
from sumo_rl.agents.frap import frap as frap_agent
from sumo_rl.agents.ppo import ppo as ppo_agent
from sumo_rl.agents.rllib_common import (
    build_rllib_parallel_env,
    build_policy_mapping as _build_policy_mapping,
    _possible_agents,
    plain_dict as _plain_dict,
    policy_id_for_agent as _policy_id_for_agent,
    policy_mode as _policy_mode,
    scenario_factory_name,
)
from sumo_rl.agents.sac import sac as sac_agent


SUPPORTED_RLLIB_ALGORITHMS = {
    ppo_agent.KIND,
    dqn_agent.KIND,
    frap_agent.KIND,
    colight_agent.KIND,
    *sac_agent.KINDS,
}


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


def _validation_log_freq_episodes(cfg: DictConfig) -> int:
    logging_cfg = getattr(cfg, "logging", None)
    value = getattr(logging_cfg, "validation_log_freq_episodes", 1) if logging_cfg is not None else 1
    return max(1, int(value or 1))


def _rllib_run_name(cfg: DictConfig, algorithm_kind: str) -> str:
    scenario_name = scenario_factory_name(cfg) or str(getattr(getattr(cfg, "scenario", None), "name", "scenario"))
    timestamp = datetime.now().strftime("%H%M%S")
    return f"{scenario_name}__{algorithm_kind}__{timestamp}"


def _algorithm_module(algorithm_kind: str):
    if algorithm_kind == ppo_agent.KIND:
        return ppo_agent
    if algorithm_kind == dqn_agent.KIND:
        return dqn_agent
    if algorithm_kind == frap_agent.KIND:
        return frap_agent
    if algorithm_kind == colight_agent.KIND:
        return colight_agent
    if algorithm_kind in sac_agent.KINDS:
        return sac_agent
    raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")


def _build_algorithm_config(cfg: DictConfig, run_dir: Path, algorithm_kind: str):
    module = _algorithm_module(algorithm_kind)
    if module is sac_agent:
        return module.build_config(cfg, run_dir, algorithm_kind=algorithm_kind)
    return module.build_config(cfg, run_dir)


def _train_algorithm(algo, cfg: DictConfig, algorithm_kind: str, emit_metrics, validate=None) -> None:
    module = _algorithm_module(algorithm_kind)
    if module is sac_agent:
        module.train(algo, cfg, algorithm_kind=algorithm_kind, emit_metrics=emit_metrics, validate=validate)
    else:
        module.train(algo, cfg, emit_metrics=emit_metrics, validate=validate)


def _compute_single_action(algo, obs, *, policy_id: Optional[str] = None):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        module = get_module(policy_id) if policy_id is not None else get_module()
        if module is not None and hasattr(module, "forward_inference"):
            import torch
            from ray.rllib.core.columns import Columns

            if isinstance(obs, dict):
                obs_batch = {
                    key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
                    for key, value in obs.items()
                }
            else:
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


def _build_eval_env(cfg: DictConfig, run_dir: Path, seed: int, *, algorithm_kind: str, policy_mode: str):
    module = _algorithm_module(algorithm_kind)
    build_eval_env = getattr(module, "build_eval_env", None)
    if callable(build_eval_env):
        return build_eval_env(cfg, run_dir, seed=seed)
    return build_rllib_parallel_env(
        cfg,
        run_dir,
        seed=seed,
        pad_spaces=(policy_mode == "shared"),
    )


def _run_multi_agent_episode(algo, env, seed: int, *, policy_mode: str) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    possible_agents = _possible_agents(env)
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
            or all(bool(terminations.get(agent_id, False)) for agent_id in possible_agents)
            or all(bool(truncations.get(agent_id, False)) for agent_id in possible_agents)
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
        eval_episode = seed_index + 1
        eval_env = _build_eval_env(
            cfg,
            run_dir,
            seed=seed,
            algorithm_kind=algorithm_kind,
            policy_mode=policy_mode,
        )
        try:
            episode_reward = _run_multi_agent_episode(algo, eval_env, seed, policy_mode=policy_mode)
        finally:
            # SUMO writes tripinfo XML on close; build summaries only after the
            # file has been flushed so RESCO trip metrics do not become NaN.
            eval_env.close()

        seed_row = _build_final_eval_summary_row(
            eval_env,
            algorithm_kind=algorithm_kind,
            eval_mean_reward=float(episode_reward),
            eval_std_reward=0.0,
            eval_episodes=1,
            logging_cfg=logging_cfg,
            extra={
                "eval/seed": float(seed),
                "eval/seed_index": float(seed_index),
                "eval/episode": float(eval_episode),
            },
        )
        seed_rows.append(seed_row)

    eval_mean_reward = float(np.mean([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    eval_std_reward = float(np.std([row["final/eval/mean_reward"] for row in seed_rows])) if seed_rows else 0.0
    summary = _aggregate_final_eval_rows(
        seed_rows,
        algorithm_kind=algorithm_kind,
        eval_mean_reward=eval_mean_reward,
        eval_std_reward=eval_std_reward,
        eval_episodes=len(eval_seeds),
    )
    summary["eval/episode"] = float(len(eval_seeds))
    return summary


def _validation_summary_row(summary: Dict[str, Any], *, step: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"validation/env_step": float(step)}
    for key, value in summary.items():
        if key == "algorithm/kind":
            row[key] = value
        elif key.startswith("final/eval/"):
            row[f"validation/eval/{key[len('final/eval/'):]}"] = value
        elif key.startswith("final/resco/"):
            row[f"validation/resco/{key[len('final/resco/'):]}"] = value
        elif key.startswith("final/efficiency/"):
            row[f"validation/efficiency/{key[len('final/efficiency/'):]}"] = value
        elif key.startswith("final/safety/"):
            row[f"validation/safety/{key[len('final/safety/'):]}"] = value
        elif key.startswith("tripinfo/"):
            row[f"validation/{key}"] = value
        elif key.startswith("warnings/"):
            row[f"validation/{key}"] = value
        elif key in {"episode/sim_time_abs", "episode/elapsed_seconds", "eval/episode"}:
            row[f"validation/{key}"] = value
    return row


def train_rllib(cfg: DictConfig) -> Dict[str, Any]:
    algorithm_kind = str(getattr(cfg.algorithm, "kind", "") or "").strip()
    if algorithm_kind not in SUPPORTED_RLLIB_ALGORITHMS:
        raise ValueError(f"Unsupported RLlib algorithm kind: {algorithm_kind}")

    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg = cfg.logging
    run_name = _rllib_run_name(cfg, algorithm_kind)
    wandb_run = _init_wandb(cfg, run_dir, run_name=run_name)
    csv_run = _LocalMetricsCsvLogger(run_dir / "csv" / f"{cfg.experiment.name}.csv")

    import ray

    params = _plain_dict(getattr(cfg.algorithm, "params", {}) or {})
    ray_num_gpus = _resolve_num_gpus(params.get("ray_num_gpus", params.get("num_gpus_per_learner", "auto")))
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_gpus=ray_num_gpus)
    algo = None
    final_summary: Dict[str, Any] = {}
    try:
        config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else config.build()

        _train_algorithm(
            algo,
            cfg,
            algorithm_kind,
            emit_metrics=lambda metrics, step: _log_outputs(wandb_run, csv_run, metrics, step=step),
            validate=lambda metrics, step: _log_outputs(
                wandb_run,
                csv_run,
                _validation_summary_row(
                    _evaluate(
                        cfg,
                        run_dir,
                        algo,
                        algorithm_kind,
                        logging_cfg,
                        wandb_run=wandb_run,
                        csv_run=csv_run,
                    ),
                    step=step,
                ),
                step=step,
            ),
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
        _log_episode_summary(wandb_run, csv_run, final_summary, step=len(_eval_seeds(cfg)), logging_cfg=logging_cfg)
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
