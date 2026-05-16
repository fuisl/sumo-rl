"""Proof that RLlib SAC supports Discrete action spaces by default.

This is a minimal, self-contained RLlib run:
- a one-step discrete bandit
- SAC on the current Ray new API stack
- a short training loop followed by greedy action inspection

If the final greedy action matches the configured optimal action, that is a
practical demonstration that SAC handled the Discrete action space in this Ray
version without any custom discrete-action wrapper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ray computes default storage locations while importing modules, so keep its
# home directory inside the workspace for this proof run.
os.environ["HOME"] = str(Path.cwd())
os.environ["USERPROFILE"] = str(Path.cwd())

import gymnasium as gym
import ray
from gymnasium.spaces import Box, Discrete
from ray.tune.registry import register_env


class DiscreteBanditEnv(gym.Env):
    """A one-step bandit with a discrete action space."""

    metadata = {"render_modes": []}

    def __init__(self, num_actions: int = 3, optimal_action: int = 2):
        super().__init__()
        if num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        self.num_actions = int(num_actions)
        self.optimal_action = int(optimal_action) % self.num_actions
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._obs = np.zeros((1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        del options
        return self._obs.copy(), {}

    def step(self, action: Any):
        action_int = int(action)
        reward = 1.0 if action_int == self.optimal_action else 0.0
        info = {
            "action": float(action_int),
            "optimal_action": float(self.optimal_action),
        }
        return self._obs.copy(), reward, True, False, info


def _greedy_action(algo, obs):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        module = get_module()
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
        action = compute_single_action(obs, explore=False)
        return action[0] if isinstance(action, tuple) else action

    policy = algo.get_policy()
    action = policy.compute_single_action(obs, explore=False)
    return action[0] if isinstance(action, tuple) else action


def _build_config(env_name: str, args: argparse.Namespace):
    from ray.rllib.algorithms.sac import SACConfig

    config = SACConfig().framework("torch").environment(env_name)
    config = config.env_runners(num_env_runners=0, num_envs_per_env_runner=1)
    config = config.training(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        initial_alpha=args.initial_alpha,
        target_entropy="auto",
        train_batch_size_per_learner=args.train_batch_size,
        num_steps_sampled_before_learning_starts=0,
        twin_q=True,
        clip_actions=False,
    )
    config = config.reporting(min_sample_timesteps_per_iteration=1)
    return config


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations to run.")
    parser.add_argument("--num-actions", type=int, default=3, help="Number of discrete actions.")
    parser.add_argument("--optimal-action", type=int, default=2, help="Reward-maximizing action index.")
    parser.add_argument("--train-batch-size", type=int, default=64, help="SAC learner batch size.")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--initial-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env_name = "proof_discrete_bandit_v0"

    register_env(
        env_name,
        lambda _cfg: DiscreteBanditEnv(
            num_actions=args.num_actions,
            optimal_action=args.optimal_action,
        ),
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    algo = None
    try:
        config = _build_config(env_name, args)
        config = config.resources(num_gpus=0)
        config = config.debugging(seed=args.seed)
        build_algo = getattr(config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else config.build()

        obs = np.zeros((1,), dtype=np.float32)
        print(
            f"Discrete bandit proof: actions={args.num_actions}, optimal_action={args.optimal_action}",
            flush=True,
        )
        for iteration in range(1, args.iterations + 1):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get("episode_return_mean")
            learner_stats = result.get("learners", {})
            print(
                f"iter={iteration:02d} "
                f"reward_mean={reward_mean} "
                f"keys={sorted(result.keys())[:8]}",
                flush=True,
            )
            if isinstance(learner_stats, dict):
                train_result = learner_stats.get("default_policy")
                if isinstance(train_result, dict):
                    print(f"  learner_keys={sorted(train_result.keys())[:8]}", flush=True)

        greedy_action = _greedy_action(algo, obs)
        print(f"greedy_action={greedy_action}", flush=True)
        return 0
    finally:
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
