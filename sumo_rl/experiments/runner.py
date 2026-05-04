from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf


def _as_plain_dict(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _as_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_dict(item) for item in value]
    return value


def _repo_root() -> Path:
    import sumo_rl

    return Path(sumo_rl.__file__).resolve().parent.parent


def _resolve_sumo_path(raw_path: Any) -> Any:
    if not isinstance(raw_path, str):
        return raw_path
    if raw_path.startswith("sumo_rl/"):
        return str(_repo_root() / raw_path)
    return raw_path


def _prepare_env_kwargs(cfg: DictConfig, run_dir: Path) -> Dict[str, Any]:
    kwargs = dict(_as_plain_dict(cfg.env.kwargs or {}))
    for key, value in list(kwargs.items()):
        if key.endswith("_file") or key in {"net_file", "route_file", "sumo_cfg_file"}:
            kwargs[key] = _resolve_sumo_path(value)

    if not kwargs.get("out_csv_name"):
        kwargs["out_csv_name"] = str(run_dir / "csv" / cfg.experiment.name)

    if "sumo_seed" not in kwargs and cfg.experiment.seed is not None:
        kwargs["sumo_seed"] = int(cfg.experiment.seed)

    return kwargs


def _get_run_dir() -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / f"run_{timestamp}"


def _init_wandb(cfg: DictConfig, run_dir: Path):
    logging_cfg = cfg.logging
    if not logging_cfg.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Hydra logging is enabled but `wandb` is not installed. Install the `experiments` extra first."
        ) from exc

    run_name = logging_cfg.name or cfg.experiment.name
    project = logging_cfg.project or cfg.experiment.project
    group = logging_cfg.group or cfg.experiment.group
    tags = list(logging_cfg.tags or cfg.experiment.tags or [])

    return wandb.init(
        project=project,
        entity=logging_cfg.entity,
        name=run_name,
        group=group,
        tags=tags,
        job_type=logging_cfg.job_type,
        mode=logging_cfg.mode,
        dir=str(run_dir),
        config=_as_plain_dict(cfg),
        reinit="finish_previous",
    )


def _numeric_metrics(data: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            nested_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[nested_prefix] = float(value)
            elif isinstance(value, dict):
                metrics.update(_numeric_metrics(value, nested_prefix))
    return metrics


def _log_wandb(run, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None or not metrics:
        return
    run.log(metrics, step=step)


def _build_env(cfg: DictConfig, run_dir: Path):
    import sumo_rl
    from sumo_rl import SumoEnvironment

    kwargs = _prepare_env_kwargs(cfg, run_dir)
    factory = cfg.env.factory

    if factory == "sumo_env":
        return SumoEnvironment(**kwargs)
    if factory == "env":
        return sumo_rl.env(**kwargs)
    if factory == "parallel_env":
        return sumo_rl.parallel_env(**kwargs)
    if factory == "fixed_time_env":
        kwargs["fixed_ts"] = True
        return sumo_rl.env(**kwargs)
    if factory == "grid4x4":
        return sumo_rl.grid4x4(**kwargs)
    if factory == "arterial4x4":
        return sumo_rl.arterial4x4(**kwargs)
    if factory == "cologne1":
        return sumo_rl.cologne1(**kwargs)
    if factory == "cologne3":
        return sumo_rl.cologne3(**kwargs)
    if factory == "cologne8":
        return sumo_rl.cologne8(**kwargs)
    if factory == "ingolstadt1":
        return sumo_rl.ingolstadt1(**kwargs)
    if factory == "ingolstadt7":
        return sumo_rl.ingolstadt7(**kwargs)
    if factory == "ingolstadt21":
        return sumo_rl.ingolstadt21(**kwargs)

    raise ValueError(f"Unsupported env factory: {factory}")


def _get_final_info(env):
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "env") and getattr(env.unwrapped.env, "metrics", None):
        return env.unwrapped.env.metrics[-1]
    if getattr(env, "metrics", None):
        return env.metrics[-1]
    return {}


def _build_ql_agents_direct(env, cfg: DictConfig, initial_states: Dict[str, Any]):
    from sumo_rl.agents import QLAgent
    from sumo_rl.exploration import EpsilonGreedy

    params = _as_plain_dict(cfg.algorithm.params or {})
    exploration = EpsilonGreedy(
        initial_epsilon=float(params.get("epsilon", 0.05)),
        min_epsilon=float(params.get("min_epsilon", 0.005)),
        decay=float(params.get("decay", 1.0)),
    )

    return {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=float(params.get("alpha", 0.1)),
            gamma=float(params.get("gamma", 0.99)),
            exploration_strategy=exploration,
        )
        for ts in env.ts_ids
    }


def _build_ql_agents_aec(env, cfg: DictConfig):
    from sumo_rl.agents import QLAgent
    from sumo_rl.exploration import EpsilonGreedy

    params = _as_plain_dict(cfg.algorithm.params or {})
    exploration = EpsilonGreedy(
        initial_epsilon=float(params.get("epsilon", 0.05)),
        min_epsilon=float(params.get("min_epsilon", 0.005)),
        decay=float(params.get("decay", 1.0)),
    )

    return {
        ts: QLAgent(
            starting_state=env.unwrapped.env.encode(env.observe(ts), ts),
            state_space=env.observation_space(ts),
            action_space=env.action_space(ts),
            alpha=float(params.get("alpha", 0.1)),
            gamma=float(params.get("gamma", 0.99)),
            exploration_strategy=exploration,
        )
        for ts in env.agents
    }


def _run_direct_q_learning(cfg: DictConfig, run_dir: Path, wandb_run) -> None:
    env = _build_env(cfg, run_dir)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_runs = int(cfg.experiment.runs)
    total_episodes = int(cfg.experiment.episodes)
    fixed_ts = bool(cfg.experiment.fixed_ts)

    try:
        for run_idx in range(1, total_runs + 1):
            initial_states = env.reset()
            agents = _build_ql_agents_direct(env, cfg, initial_states)

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    initial_states = env.reset()
                    for agent_id in initial_states.keys():
                        agents[agent_id].state = env.encode(initial_states[agent_id], agent_id)

                episode_reward = 0.0
                done = {"__all__": False}
                info = {}

                if fixed_ts:
                    while not done["__all__"]:
                        _, _, done, info = env.step({})
                else:
                    while not done["__all__"]:
                        actions = {ts: agents[ts].act() for ts in agents.keys()}
                        next_state, reward, done, info = env.step(action=actions)
                        episode_reward += float(sum(reward.values()))

                        for agent_id in agents.keys():
                            agents[agent_id].learn(
                                next_state=env.encode(next_state[agent_id], agent_id),
                                reward=reward[agent_id],
                            )

                env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                metrics = {
                    "run/index": run_idx,
                    "episode/index": episode_idx,
                    "episode/reward": episode_reward,
                    "env/step": float(info.get("step", env.sim_step)),
                }
                metrics.update(_numeric_metrics(info, "env"))
                _log_wandb(wandb_run, metrics, step=(run_idx - 1) * total_episodes + episode_idx)
    finally:
        env.close()


def _run_aec_q_learning(cfg: DictConfig, run_dir: Path, wandb_run) -> None:
    env = _build_env(cfg, run_dir)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_runs = int(cfg.experiment.runs)
    total_episodes = int(cfg.experiment.episodes)
    fixed_ts = bool(cfg.experiment.fixed_ts)

    try:
        for run_idx in range(1, total_runs + 1):
            env.reset()
            agents = _build_ql_agents_aec(env, cfg)

            for episode_idx in range(1, total_episodes + 1):
                if episode_idx > 1:
                    env.reset()
                    for agent_id in env.agents:
                        agents[agent_id].state = env.unwrapped.env.encode(env.observe(agent_id), agent_id)

                episode_reward = 0.0

                if fixed_ts:
                    while env.agents:
                        env.step(None)
                else:
                    for agent in env.agent_iter():
                        observation, reward, terminated, truncated, _ = env.last()
                        done = terminated or truncated
                        if agents[agent].action is not None:
                            agents[agent].learn(
                                next_state=env.unwrapped.env.encode(observation, agent),
                                reward=reward,
                            )

                        action = agents[agent].act() if not done else None
                        env.step(action)
                        episode_reward += float(reward)

                env.unwrapped.env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                metrics = {
                    "run/index": run_idx,
                    "episode/index": episode_idx,
                    "episode/reward": episode_reward,
                }
                if env.unwrapped.env.metrics:
                    metrics.update(_numeric_metrics(env.unwrapped.env.metrics[-1], "env"))
                _log_wandb(wandb_run, metrics, step=(run_idx - 1) * total_episodes + episode_idx)
    finally:
        env.close()


def _run_fixed_time(cfg: DictConfig, run_dir: Path, wandb_run) -> None:
    env = _build_env(cfg, run_dir)
    csv_prefix = Path(_prepare_env_kwargs(cfg, run_dir)["out_csv_name"])
    total_runs = int(cfg.experiment.runs)
    total_episodes = int(cfg.experiment.episodes)

    try:
        for run_idx in range(1, total_runs + 1):
            for episode_idx in range(1, total_episodes + 1):
                episode_reward = 0.0

                if hasattr(env, "agent_iter"):
                    env.reset()
                    for agent in env.agent_iter():
                        _obs, _reward, terminated, truncated, _info = env.last()
                        done = bool(terminated or truncated)
                        action = None if done else env.action_space(agent).sample()
                        env.step(action)
                else:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        _obs, _info = reset_result
                    done = False
                    while not done:
                        next_step = env.step({})
                        if len(next_step) == 5:
                            _obs, reward, terminated, truncated, info = next_step
                            done = bool(terminated or truncated)
                        else:
                            _obs, reward, dones, info = next_step
                            done = bool(dones["__all__"])
                        if isinstance(reward, dict):
                            episode_reward += float(sum(reward.values()))
                        elif reward is not None:
                            episode_reward += float(reward)

                env.save_csv(str(csv_prefix), (run_idx - 1) * total_episodes + episode_idx)
                final_info = _get_final_info(env)
                metrics = {
                    "run/index": run_idx,
                    "episode/index": episode_idx,
                    "episode/reward": episode_reward,
                }
                metrics.update(_numeric_metrics(final_info, "env"))
                _log_wandb(wandb_run, metrics, step=(run_idx - 1) * total_episodes + episode_idx)
    finally:
        env.close()


class _SB3WandbCallback:
    def __init__(self, wandb_run, log_freq: int = 1000):
        self.wandb_run = wandb_run
        self.log_freq = max(1, int(log_freq))
        self._callback = None

    def build(self):
        from stable_baselines3.common.callbacks import BaseCallback

        wandb_run = self.wandb_run
        log_freq = self.log_freq

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                if wandb_run is not None and self.n_calls % log_freq == 0:
                    rewards = self.locals.get("rewards")
                    metrics = {"train/num_timesteps": float(self.num_timesteps)}
                    if rewards is not None:
                        metrics["train/reward_mean"] = float(np.mean(rewards))
                    wandb_run.log(metrics, step=self.num_timesteps)
                return True

        self._callback = Callback()
        return self._callback


def _run_sb3_dqn(cfg: DictConfig, run_dir: Path, wandb_run) -> None:
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy

    env = _build_env(cfg, run_dir)
    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)

    try:
        model = DQN(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            seed=int(cfg.experiment.seed) if cfg.experiment.seed is not None else None,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        callback = _SB3WandbCallback(wandb_run, log_freq=int(cfg.logging.log_freq or 1000)).build()
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps), callback=callback)

        eval_env = None
        try:
            eval_env = _build_env(cfg, run_dir)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
        finally:
            if eval_env is not None:
                eval_env.close()

        _log_wandb(
            wandb_run,
            {
                "eval/mean_reward": float(mean_reward),
                "eval/std_reward": float(std_reward),
                "train/total_timesteps": float(model.num_timesteps),
            },
            step=int(model.num_timesteps),
        )
    finally:
        env.close()


def _run_sb3_ppo(cfg: DictConfig, run_dir: Path, wandb_run) -> None:
    import supersuit as ss
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecMonitor

    env = _build_env(cfg, run_dir)
    params = _as_plain_dict(cfg.algorithm.params or {})
    eval_episodes = int(cfg.experiment.eval_episodes)
    num_envs = int(params.pop("num_envs", 2))

    try:
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env,
            num_envs,
            num_cpus=1,
            base_class="stable_baselines3",
        )
        if not hasattr(env, "render_mode"):
            env.render_mode = None
        env = VecMonitor(env)

        model = PPO(
            policy=params.pop("policy", "MlpPolicy"),
            env=env,
            tensorboard_log=str(run_dir / "tensorboard"),
            **params,
        )
        callback = _SB3WandbCallback(wandb_run, log_freq=int(cfg.logging.log_freq or 1000)).build()
        model.learn(total_timesteps=int(cfg.experiment.total_timesteps), callback=callback)

        eval_env = None
        try:
            eval_env = _build_env(cfg, run_dir)
            eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
            eval_env = ss.concat_vec_envs_v1(
                eval_env,
                num_envs,
                num_cpus=1,
                base_class="stable_baselines3",
            )
            if not hasattr(eval_env, "render_mode"):
                eval_env.render_mode = None
            eval_env = VecMonitor(eval_env)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
        finally:
            if eval_env is not None:
                eval_env.close()

        _log_wandb(
            wandb_run,
            {
                "eval/mean_reward": float(mean_reward),
                "eval/std_reward": float(std_reward),
                "train/total_timesteps": float(model.num_timesteps),
            },
            step=int(model.num_timesteps),
        )
    finally:
        env.close()


def run(cfg: DictConfig) -> None:
    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(cfg.experiment.seed) if cfg.experiment.seed is not None else 0)
    random.seed(int(cfg.experiment.seed) if cfg.experiment.seed is not None else 0)

    wandb_run = _init_wandb(cfg, run_dir)
    try:
        algorithm_kind = cfg.algorithm.kind
        if algorithm_kind == "q_learning":
            if cfg.env.factory == "env":
                _run_aec_q_learning(cfg, run_dir, wandb_run)
            else:
                _run_direct_q_learning(cfg, run_dir, wandb_run)
        elif algorithm_kind == "fixed_time":
            _run_fixed_time(cfg, run_dir, wandb_run)
        elif algorithm_kind == "dqn_sb3":
            _run_sb3_dqn(cfg, run_dir, wandb_run)
        elif algorithm_kind == "ppo_sb3":
            _run_sb3_ppo(cfg, run_dir, wandb_run)
        else:
            raise ValueError(f"Unsupported algorithm kind: {algorithm_kind}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
