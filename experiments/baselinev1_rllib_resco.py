"""Train the baseline-v1 Torch Geometric graph policy with RLlib on RESCO."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import sumo_rl
from sumo_rl.environment.rllib_graph_env import RLLibGraphObservationWrapper
from sumo_rl.models.baselinev1_tg import BaselineV1TorchGeometricModel
from sumo_rl.models.topology import build_resco_topology


SCENARIOS = (
    "grid4x4",
    "arterial4x4",
    "cologne1",
    "cologne3",
    "cologne8",
    "ingolstadt1",
    "ingolstadt7",
    "ingolstadt21",
)
MODEL_NAME = "baselinev1_torch_geometric"


class BaselineV1Callbacks(DefaultCallbacks):
    """Lift selected SUMO infos into RLlib custom metrics."""

    INFO_KEYS = (
        "system_total_stopped",
        "system_total_waiting_time",
        "system_mean_waiting_time",
        "system_mean_speed",
        "baselinev1_invalid_actions_total",
    )

    def on_episode_step(self, *, worker, base_env, episode, env_index: Optional[int] = None, **kwargs) -> None:
        del worker, base_env, env_index, kwargs
        get_agents = getattr(episode, "get_agents", None)
        agent_ids = get_agents() if callable(get_agents) else []
        for agent_id in agent_ids:
            try:
                info = episode.last_info_for(agent_id)
            except Exception:
                continue
            if not info:
                continue
            for key in self.INFO_KEYS:
                value = info.get(key)
                if isinstance(value, (int, float, np.number)):
                    episode.custom_metrics[key] = float(value)


def _resolve_base_env(env: Any) -> Any:
    current = env
    seen = set()
    for _ in range(12):
        if hasattr(current, "_net"):
            return current
        for attr in ("env", "aec_env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and id(candidate) not in seen:
                seen.add(id(candidate))
                current = candidate
                break
        else:
            break
    raise RuntimeError("Unable to resolve the underlying SUMO environment.")


def _make_base_env(config: Dict[str, Any]):
    scenario = str(config["scenario"])
    factory = getattr(sumo_rl, scenario)
    kwargs = {
        "parallel": True,
        "use_gui": bool(config.get("use_gui", False)),
        "out_csv_name": config.get("out_csv_name"),
        "yellow_time": int(config.get("yellow_time", 2)),
        "fixed_ts": bool(config.get("fixed_ts", False)),
        "sumo_warnings": bool(config.get("sumo_warnings", False)),
    }
    reward_fn = config.get("reward_fn")
    if reward_fn:
        kwargs["reward_fn"] = reward_fn
    return factory(**kwargs)


def make_graph_env(env_config: Dict[str, Any]):
    config = dict(env_config)
    out_csv_name = config.get("out_csv_name")
    if out_csv_name:
        worker_index = getattr(env_config, "worker_index", 0)
        vector_index = getattr(env_config, "vector_index", 0)
        config["out_csv_name"] = f"{out_csv_name}_w{worker_index}_v{vector_index}"

    base_env = _make_base_env(config)
    agent_ids = list(getattr(base_env, "possible_agents", getattr(base_env, "agents", [])))
    topology = build_resco_topology(_resolve_base_env(base_env)._net, agent_ids)
    return RLLibGraphObservationWrapper(base_env, topology)


def _probe_spaces_and_model_config(args: argparse.Namespace, out_dir: Path):
    probe_env = make_graph_env(
        {
            "scenario": args.scenario,
            "use_gui": False,
            "out_csv_name": None,
            "yellow_time": args.yellow_time,
            "fixed_ts": args.fixed_ts,
            "sumo_warnings": args.sumo_warnings,
            "reward_fn": args.reward_fn,
        }
    )
    try:
        first_agent = probe_env.possible_agents[0]
        obs_space = probe_env.observation_space(first_agent)
        action_space = probe_env.action_space(first_agent)
        model_config = probe_env.model_config(normalize_edge_attr=not args.no_normalize_edge_attr)
    finally:
        probe_env.close()

    model_config.update(
        {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "fusion_dim": args.fusion_dim,
            "actor_hidden": args.actor_hidden,
            "value_hidden": args.value_hidden,
            "heads": args.heads,
            "dropout": args.dropout,
            "add_self_loops": args.add_self_loops,
            "out_dir": str(out_dir),
        }
    )
    return obs_space, action_space, model_config


def _resolve_env_file(explicit_path: str) -> Optional[Path]:
    candidates = []
    if explicit_path:
        path = Path(explicit_path).expanduser()
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(Path(__file__).resolve().parents[1] / path.name)
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[1] / ".env")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_env_file(env_path: Optional[Path]) -> None:
    if env_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return
    load_dotenv(env_path, override=False)


def _wandb_callbacks(args: argparse.Namespace, out_dir: Path):
    env_path = _resolve_env_file(args.wandb_env_file)
    _load_env_file(env_path)

    project = args.wandb_project or os.getenv("WANDB_PROJECT")
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    api_key = args.wandb_api_key or os.getenv("WANDB_API_KEY")
    name = args.wandb_name or os.getenv("WANDB_NAME") or f"baselinev1-rllib-{args.algo.lower()}-{args.scenario}"

    should_enable = args.wandb == "on" or (args.wandb == "auto" and (project or api_key))
    if not should_enable:
        return []

    try:
        from ray.air.integrations.wandb import WandbLoggerCallback
    except ImportError:
        try:
            from ray.tune.integration.wandb import WandbLoggerCallback
        except ImportError as exc:
            raise RuntimeError("WandB logging was requested, but Ray's WandbLoggerCallback is unavailable.") from exc

    return [
        WandbLoggerCallback(
            project=project or "sumo-rl",
            entity=entity or None,
            api_key=api_key or None,
            name=name,
            group=args.wandb_group or args.scenario,
            log_config=True,
            upload_checkpoints=bool(args.wandb_upload_checkpoints),
            dir=str(out_dir / "wandb"),
            reinit="finish_previous",
        )
    ]


def _tune_storage_kwargs(path: Any) -> Dict[str, str]:
    signature = inspect.signature(tune.run)
    key = "storage_path" if "storage_path" in signature.parameters else "local_dir"
    return {key: str(Path(path).expanduser().resolve())}


def _disable_new_api_stack(config):
    api_stack = getattr(config, "api_stack", None)
    if callable(api_stack):
        try:
            return api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        except TypeError:
            return api_stack(enable_rl_module_and_learner=False)
    return config


def _resolve_num_gpus(value: Any) -> float:
    raw_value = str(value).strip().lower()
    if raw_value in {"", "auto"}:
        try:
            import torch
        except ImportError:
            return 0
        return 1 if torch.cuda.is_available() else 0
    if raw_value == "all":
        try:
            import torch
        except ImportError:
            return 0
        return float(torch.cuda.device_count())
    return float(raw_value)


def _configure_rollouts(config, args: argparse.Namespace):
    rollouts = getattr(config, "rollouts", None)
    if callable(rollouts):
        try:
            return rollouts(num_rollout_workers=args.num_workers, rollout_fragment_length=args.rollout_fragment_length)
        except (TypeError, ValueError):
            pass

    env_runners = getattr(config, "env_runners")
    return env_runners(num_env_runners=args.num_workers, rollout_fragment_length=args.rollout_fragment_length)


def _build_config(
    args: argparse.Namespace,
    env_name: str,
    obs_space: Any,
    action_space: Any,
    model_config: Dict[str, Any],
):
    shared_policy = {"shared_policy": (None, obs_space, action_space, {})}

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        del agent_id, episode, worker, kwargs
        return "shared_policy"

    common_model = {
        "custom_model": MODEL_NAME,
        "custom_model_config": model_config,
    }

    if args.algo == "SAC":
        config = _disable_new_api_stack(SACConfig())
        target_entropy: Any = args.target_entropy
        if target_entropy != "auto":
            target_entropy = float(target_entropy)
        config = config.training(
            gamma=args.gamma,
            tau=args.tau,
            twin_q=True,
            train_batch_size=args.train_batch_size,
            actor_lr=args.lr,
            critic_lr=args.critic_lr or args.lr,
            alpha_lr=args.alpha_lr or args.lr,
            initial_alpha=args.initial_alpha,
            target_entropy=target_entropy,
            num_steps_sampled_before_learning_starts=args.learning_starts,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": args.replay_capacity,
            },
            policy_model_config=common_model,
            q_model_config=common_model,
        )
    else:
        config = _disable_new_api_stack(PPOConfig())
        config = config.training(
            gamma=args.gamma,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            lambda_=args.lambda_,
            use_gae=True,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            model=common_model,
        )

    config = (
        config.environment(env=env_name, env_config=_env_config(args), disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .debugging(log_level=args.log_level)
        .experimental(_disable_preprocessor_api=True)
        .callbacks(BaselineV1Callbacks)
        .multi_agent(policies=shared_policy, policy_mapping_fn=policy_mapping_fn)
    )
    return _configure_rollouts(config, args)


def _env_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "scenario": args.scenario,
        "use_gui": args.gui,
        "out_csv_name": str(Path(args.out_dir) / "resco"),
        "yellow_time": args.yellow_time,
        "fixed_ts": args.fixed_ts,
        "sumo_warnings": args.sumo_warnings,
        "reward_fn": args.reward_fn,
    }


def train(args: argparse.Namespace):
    out_dir = Path(args.out_dir.format(scenario=args.scenario, algo=args.algo.lower()))
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    args.num_gpus = _resolve_num_gpus(args.num_gpus)

    ModelCatalog.register_custom_model(MODEL_NAME, BaselineV1TorchGeometricModel)
    obs_space, action_space, model_config = _probe_spaces_and_model_config(args, out_dir)

    env_name = f"baselinev1_{args.scenario}"
    register_env(env_name, lambda config: ParallelPettingZooEnv(make_graph_env(config)))

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        address=args.ray_address or None,
        ignore_reinit_error=True,
        include_dashboard=False,
        local_mode=args.ray_local_mode,
    )
    available_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    if args.num_gpus > available_gpus:
        raise RuntimeError(f"Requested {args.num_gpus:g} GPU(s), but Ray sees only {available_gpus:g}.")
    print(f"RLlib GPU allocation: requested {args.num_gpus:g}, Ray sees {available_gpus:g}.")

    config = _build_config(args, env_name, obs_space, action_space, model_config)
    stop = {"training_iteration": args.max_iters}
    if args.stop_timesteps > 0:
        stop["timesteps_total"] = args.stop_timesteps

    callbacks = _wandb_callbacks(args, out_dir)
    storage_path = args.storage_path or str(out_dir / "ray_results")
    try:
        return tune.run(
            args.algo,
            name=args.experiment_name or f"baselinev1_{args.algo.lower()}_{args.scenario}",
            stop=stop,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=args.checkpoint_at_end,
            config=config.to_dict(),
            callbacks=callbacks,
            verbose=args.tune_verbose,
            **_tune_storage_kwargs(storage_path),
        )
    finally:
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline-v1 with RLlib, PyTorch, Torch Geometric, and WandB.")
    parser.add_argument("--scenario", default="grid4x4", choices=SCENARIOS)
    parser.add_argument("--algo", default="SAC", choices=("SAC", "PPO"))
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--stop-timesteps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rollout-fragment-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=0.0)
    parser.add_argument("--alpha-lr", type=float, default=0.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.9)
    parser.add_argument("--clip-param", type=float, default=0.4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.25)
    parser.add_argument("--sgd-minibatch-size", type=int, default=64)
    parser.add_argument("--num-sgd-iter", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--actor-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--add-self-loops", action="store_true")
    parser.add_argument("--no-normalize-edge-attr", action="store_true")
    parser.add_argument("--yellow-time", type=int, default=2)
    parser.add_argument("--reward-fn", default="")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fixed-ts", action="store_true")
    parser.add_argument("--sumo-warnings", action="store_true")
    parser.add_argument("--num-gpus", default=os.environ.get("RLLIB_NUM_GPUS", "auto"))
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--ray-local-mode", action="store_true")
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--tune-verbose", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--checkpoint-at-end", action="store_true")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--out-dir", default="outputs/baselinev1_rllib/{scenario}")
    parser.add_argument("--storage-path", default="")
    parser.add_argument("--wandb", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--wandb-env-file", default=".env")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-api-key", default="")
    parser.add_argument("--wandb-upload-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
