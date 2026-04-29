"""Train the local-neighbor graph SAC baseline on a RESCO scenario."""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sumo_rl
from sumo_rl.models import LocalNeighborGATDiscreteSAC, build_resco_topology


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    action_mask: torch.Tensor
    next_obs: torch.Tensor
    next_action_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        batch_size = min(batch_size, len(self._buf))
        return self._rng.sample(list(self._buf), batch_size)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class Optimizers:
    critic: torch.optim.Optimizer
    actor: torch.optim.Optimizer
    alpha: torch.optim.Optimizer


def _resolve_base_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    for _ in range(10):
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


def _make_env(name: str, *, use_gui: bool, out_csv_name: str, fixed_ts: bool = False):
    factory = getattr(sumo_rl, name)
    return factory(
        parallel=True,
        use_gui=use_gui,
        out_csv_name=out_csv_name,
        yellow_time=2,
        fixed_ts=fixed_ts,
    )


def _ordered_agent_ids(env: Any, observations: dict[str, np.ndarray]) -> list[str]:
    if getattr(env, "agents", None):
        return list(env.agents)
    return list(observations.keys())


def _build_action_mask(env: Any, agent_ids: list[str], num_actions: int) -> torch.Tensor:
    mask = torch.zeros((len(agent_ids), num_actions), dtype=torch.bool)
    for idx, agent_id in enumerate(agent_ids):
        space = env.action_space(agent_id) if hasattr(env, "action_space") else env.action_spaces[agent_id]
        mask[idx, : int(space.n)] = True
    return mask


def _stack_obs(observations: dict[str, np.ndarray], agent_ids: list[str]) -> torch.Tensor:
    rows = np.asarray([observations[agent_id] for agent_id in agent_ids], dtype=np.float32)
    return torch.from_numpy(rows)


def _dict_to_tensor(values: dict[str, float], agent_ids: list[str]) -> torch.Tensor:
    rows = [float(values[agent_id]) for agent_id in agent_ids]
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(-1)


def _transition_from_step(
    obs: torch.Tensor,
    action: torch.Tensor,
    rewards: dict[str, float],
    next_obs: torch.Tensor,
    terminated: dict[str, bool],
    truncated: dict[str, bool],
    action_mask: torch.Tensor,
    next_action_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    agent_ids: list[str],
) -> Transition:
    done_flag = float(any(terminated.values()) or any(truncated.values()))
    reward_tensor = _dict_to_tensor(rewards, agent_ids)
    done_tensor = torch.full_like(reward_tensor, done_flag)
    return Transition(
        obs=obs,
        action=action.long(),
        reward=reward_tensor,
        done=done_tensor,
        action_mask=action_mask,
        next_obs=next_obs,
        next_action_mask=next_action_mask,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def _sample_batch(replay: ReplayBuffer, batch_size: int) -> list[Transition]:
    return replay.sample(batch_size)


def sac_update(
    agent: LocalNeighborGATDiscreteSAC,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    optimizers: Optimizers,
) -> dict[str, float] | None:
    if len(replay) < batch_size:
        return None

    batch = _sample_batch(replay, batch_size)
    critic_losses: list[torch.Tensor] = []
    actor_losses: list[torch.Tensor] = []
    alpha_losses: list[torch.Tensor] = []
    q1_values: list[float] = []
    q2_values: list[float] = []
    entropies: list[float] = []

    for transition in batch:
        obs = transition.obs.to(device)
        action = transition.action.to(device)
        reward = transition.reward.to(device)
        done = transition.done.to(device)
        action_mask = transition.action_mask.to(device)
        next_obs = transition.next_obs.to(device)
        next_action_mask = transition.next_action_mask.to(device)
        edge_index = transition.edge_index.to(device)
        edge_attr = None if transition.edge_attr is None else transition.edge_attr.to(device)

        q1_all, q2_all = agent.critic_values(obs, edge_index, edge_attr)
        q1 = q1_all.gather(-1, action.unsqueeze(-1))
        q2 = q2_all.gather(-1, action.unsqueeze(-1))

        with torch.no_grad():
            _, next_probs, next_log_probs = agent.get_action_probs(next_obs, edge_index, edge_attr, next_action_mask)
            next_q1, next_q2 = agent.target_critic_values(next_obs, edge_index, edge_attr)
            next_q_min = torch.min(next_q1, next_q2)
            target = reward + (1.0 - done) * gamma * (next_probs * (next_q_min - agent.alpha.detach() * next_log_probs)).sum(dim=-1, keepdim=True)

        critic_losses.append(0.5 * (torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)))

        _, action_probs, log_action_probs = agent.get_action_probs(obs, edge_index, edge_attr, action_mask)
        with torch.no_grad():
            current_q1, current_q2 = agent.critic_values(obs, edge_index, edge_attr)
        current_q_min = torch.min(current_q1, current_q2)
        actor_losses.append((action_probs * (agent.alpha.detach() * log_action_probs - current_q_min)).sum(dim=-1).mean())
        ent = -(action_probs * log_action_probs).sum(dim=-1).mean()
        entropies.append(float(ent.item()))
        alpha_losses.append((agent.log_alpha * (ent.detach() - agent.target_entropy)).mean())
        q1_values.append(float(q1.mean().item()))
        q2_values.append(float(q2.mean().item()))

    critic_loss = torch.stack(critic_losses).mean()
    actor_loss = torch.stack(actor_losses).mean()
    alpha_loss = torch.stack(alpha_losses).mean()

    critic_params = list(agent.critic.parameters())
    actor_params = list(agent.local_encoder.parameters()) + list(agent.neighbor_encoder.parameters()) + list(agent.fusion.parameters()) + list(agent.actor.parameters())

    optimizers.critic.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=10.0)
    optimizers.critic.step()

    optimizers.actor.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=10.0)
    optimizers.actor.step()

    optimizers.alpha.zero_grad()
    alpha_loss.backward()
    optimizers.alpha.step()

    agent.soft_update_target()

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": float(alpha_loss.item()),
        "q1": float(np.mean(q1_values)),
        "q2": float(np.mean(q2_values)),
        "entropy": float(np.mean(entropies)),
        "alpha": float(agent.alpha.item()),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    env = _make_env(args.scenario, use_gui=args.gui, out_csv_name=str(out_dir / "resco"), fixed_ts=args.fixed_ts)
    observations, _ = env.reset(seed=args.seed)
    agent_ids = _ordered_agent_ids(env, observations)
    base_env = _resolve_base_env(env)

    topology = build_resco_topology(base_env._net, agent_ids)
    action_mask = _build_action_mask(
        env,
        agent_ids,
        max(int((env.action_space(agent_id) if hasattr(env, "action_space") else env.action_spaces[agent_id]).n) for agent_id in agent_ids),
    )
    obs_tensor = _stack_obs(observations, agent_ids)

    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=int(obs_tensor.shape[-1]),
        num_actions=int(action_mask.shape[-1]),
        local_encoder_cfg={"out_dim": int(args.latent_dim)},
        neighbor_encoder_cfg={"hidden_dim": int(args.hidden_dim), "out_dim": int(args.latent_dim), "heads": int(args.heads), "dropout": float(args.dropout)},
        fusion_cfg={"hidden_dim": int(args.hidden_dim), "out_dim": int(args.fusion_dim)},
        actor_cfg={"hidden_dim": int(args.actor_hidden)},
        critic_cfg={"hidden_dim": int(args.critic_hidden)},
        init_alpha=float(args.alpha),
        tau=float(args.tau),
    ).to(device)

    optimizers = Optimizers(
        critic=torch.optim.Adam(agent.critic.parameters(), lr=3e-4),
        actor=torch.optim.Adam(
            list(agent.local_encoder.parameters())
            + list(agent.neighbor_encoder.parameters())
            + list(agent.fusion.parameters())
            + list(agent.actor.parameters()),
            lr=3e-4,
        ),
        alpha=torch.optim.Adam([agent.log_alpha], lr=3e-4),
    )

    replay = ReplayBuffer(capacity=int(args.replay_capacity), seed=args.seed)
    best_reward = float("-inf")
    best_path = out_dir / "best_agent.pt"

    with log_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["episode", "episode_reward", "episode_steps", "critic_loss", "actor_loss", "alpha_loss", "entropy", "alpha"])
        writer.writeheader()

        for episode in range(1, int(args.episodes) + 1):
            observations, _ = env.reset(seed=args.seed + episode)
            agent_ids = _ordered_agent_ids(env, observations)
            obs_tensor = _stack_obs(observations, agent_ids)
            action_mask = _build_action_mask(env, agent_ids, agent.num_actions)
            episode_reward = 0.0
            episode_steps = 0
            last_metrics: dict[str, float] = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0, "entropy": 0.0, "alpha": float(agent.alpha.item())}

            while env.agents:
                obs_tensor = obs_tensor.to(device)
                actions, _ = agent.select_action(obs_tensor, topology.edge_index.to(device), None if topology.edge_attr is None else topology.edge_attr.to(device), action_mask.to(device), deterministic=False)
                action_dict = {agent_id: int(actions[idx].item()) for idx, agent_id in enumerate(agent_ids)}

                next_observations, rewards, terminated, truncated, infos = env.step(action_dict)
                next_agent_ids = _ordered_agent_ids(env, next_observations) if next_observations else agent_ids
                next_obs_tensor = _stack_obs(next_observations, next_agent_ids) if next_observations else obs_tensor.detach().cpu()
                next_action_mask = _build_action_mask(env, next_agent_ids, agent.num_actions)

                transition = _transition_from_step(
                    obs=obs_tensor.detach().cpu(),
                    action=actions.detach().cpu(),
                    rewards=rewards,
                    next_obs=next_obs_tensor.detach().cpu(),
                    terminated=terminated,
                    truncated=truncated,
                    action_mask=action_mask.detach().cpu(),
                    next_action_mask=next_action_mask.detach().cpu(),
                    edge_index=topology.edge_index,
                    edge_attr=topology.edge_attr,
                    agent_ids=agent_ids,
                )
                replay.push(transition)

                episode_reward += float(sum(rewards.values()))
                episode_steps += 1
                obs_tensor = next_obs_tensor
                action_mask = next_action_mask
                agent_ids = next_agent_ids

                if len(replay) >= int(args.warmup):
                    update_metrics = sac_update(agent, replay, int(args.batch_size), float(args.gamma), device, optimizers)
                    if update_metrics is not None:
                        last_metrics = update_metrics

                if not env.agents:
                    break

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(agent.state_dict(), best_path)

            writer.writerow(
                {
                    "episode": episode,
                    "episode_reward": round(float(episode_reward), 4),
                    "episode_steps": episode_steps,
                    "critic_loss": round(float(last_metrics["critic_loss"]), 6),
                    "actor_loss": round(float(last_metrics["actor_loss"]), 6),
                    "alpha_loss": round(float(last_metrics["alpha_loss"]), 6),
                    "entropy": round(float(last_metrics["entropy"]), 6),
                    "alpha": round(float(last_metrics["alpha"]), 6),
                }
            )
            file_handle.flush()

            if episode == 1 or episode % int(args.log_interval) == 0:
                print(
                    f"ep={episode:04d} reward={episode_reward:.2f} steps={episode_steps:04d} "
                    f"critic={last_metrics['critic_loss']:.4f} actor={last_metrics['actor_loss']:.4f} "
                    f"alpha={last_metrics['alpha']:.4f} entropy={last_metrics['entropy']:.4f}"
                )

    env.close()
    return {"best_checkpoint": str(best_path), "log_path": str(log_path), "best_reward": float(best_reward)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local-neighbor graph SAC baseline on RESCO.")
    parser.add_argument("--scenario", default="grid4x4", choices=["grid4x4", "arterial4x4", "cologne1", "cologne3", "cologne8", "ingolstadt1", "ingolstadt7", "ingolstadt21"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--actor-hidden", type=int, default=128)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="")
    parser.add_argument("--out-dir", default="outputs/local_neighbor_gat_discrete_sac/grid4x4")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fixed-ts", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
