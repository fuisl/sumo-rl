"""Helpers for Stable-Baselines3 experiment logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def safe_scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return None
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.generic):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.ndim != 0 and value.numel() != 1:
                return None
            return float(value.item())
    except Exception:
        pass

    return None


class SB3WandbCallback:
    _GENERIC_TRAIN_KEYS = {
        "train/loss",
        "train/learning_rate",
        "train/n_updates",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "rollout/exploration_rate",
    }
    _SAC_TRAIN_KEYS = {
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
        "train/n_updates",
        "train/learning_rate",
        "train/replay_buffer_size",
    }

    def __init__(
        self,
        wandb_run,
        csv_run,
        logging_cfg=None,
        log_freq: int = 1000,
        eval_env=None,
        eval_episodes: int = 0,
        eval_freq: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 0,
        save_checkpoints: bool = False,
        save_final_model: bool = True,
    ):
        self.wandb_run = wandb_run
        self.csv_run = csv_run
        self.logging_cfg = logging_cfg
        self.log_freq = max(1, int(log_freq))
        self.eval_env = eval_env
        self.eval_episodes = max(0, int(eval_episodes))
        self.eval_freq = max(1, int(eval_freq if eval_freq is not None else self.log_freq))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_freq = max(0, int(checkpoint_freq))
        self.save_checkpoints = bool(save_checkpoints)
        self.save_final_model = bool(save_final_model)
        self._callback = None

    def build(self):
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from sumo_rl.experiments.metric_utils import build_namespaced_metrics
        from sumo_rl.agents.sb3.wrappers import _resolve_base_env

        wandb_run = self.wandb_run
        csv_run = self.csv_run
        logging_cfg = self.logging_cfg
        log_freq = self.log_freq
        eval_env = self.eval_env
        eval_episodes = self.eval_episodes
        eval_freq = self.eval_freq
        last_train_step = -1
        last_eval_step = -1
        last_sac_metrics: Dict[str, Optional[float]] = {}
        last_checkpoint_step = -1

        def _is_truthy(flag_name: str, default: bool = False) -> bool:
            if logging_cfg is None or not hasattr(logging_cfg, flag_name):
                return default
            return bool(getattr(logging_cfg, flag_name))

        def _is_sac_model(model) -> bool:
            return model is not None and model.__class__.__name__.lower() == "sac"

        def _collect_logger_metrics(model) -> Dict[str, float]:
            metrics: Dict[str, float] = {}
            if _is_sac_model(model):
                return metrics
            logger = getattr(model, "logger", None)
            values = getattr(logger, "name_to_value", {}) if logger is not None else {}
            allowed_keys = self._GENERIC_TRAIN_KEYS
            for key, value in values.items():
                scalar_value = safe_scalar(value)
                if scalar_value is None:
                    continue
                if key in allowed_keys:
                    metrics[key] = scalar_value
            return metrics

        def _collect_sac_diagnostics(model) -> Dict[str, float]:
            if not _is_truthy("log_sac_diagnostics", True) or not _is_sac_model(model):
                return {}

            metrics: Dict[str, float] = {}
            logger = getattr(model, "logger", None)
            values = getattr(logger, "name_to_value", {}) if logger is not None else {}
            for key in self._SAC_TRAIN_KEYS:
                value = safe_scalar(values.get(key))
                if value is not None:
                    metrics[key] = value
                    last_sac_metrics[key] = value

            replay_buffer_size = _get_replay_buffer_size(model)
            if replay_buffer_size is not None:
                metrics["train/replay_buffer_size"] = replay_buffer_size
                last_sac_metrics["train/replay_buffer_size"] = replay_buffer_size
            return metrics

        def _get_replay_buffer_size(model) -> Optional[float]:
            replay_buffer = getattr(model, "replay_buffer", None)
            if replay_buffer is None:
                return None
            size_methods = ("size", "__len__")
            for method_name in size_methods:
                method = getattr(replay_buffer, method_name, None)
                if callable(method):
                    try:
                        return safe_scalar(method())
                    except Exception:
                        continue
            try:
                return safe_scalar(len(replay_buffer))
            except Exception:
                return None

        def _log_metrics(metrics: Dict[str, Any], step: int) -> None:
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if csv_run is not None:
                csv_run.log(metrics, step=step)

        def _save_model_checkpoint(model, step: int, *, final: bool = False) -> None:
            if self.checkpoint_dir is None:
                return
            if final:
                if not self.save_final_model:
                    return
            elif not self.save_checkpoints or self.checkpoint_freq <= 0:
                return

            algo_name = model.__class__.__name__.lower()
            save_dir = self.checkpoint_dir / algo_name
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = "final_model" if final else f"checkpoint_step_{step:09d}"
            model.save(str(save_dir / file_name))

        def _log_train_metrics(locals_dict, model, step: int, force: bool = False) -> None:
            nonlocal last_train_step, last_checkpoint_step
            if step <= 0:
                return
            if not force and step % log_freq != 0:
                return
            if step == last_train_step:
                return

            metrics = {"train/num_timesteps": float(step)}
            if _is_truthy("log_sb3_internal_metrics", True):
                metrics.update(_collect_logger_metrics(model))
            if _is_truthy("log_sac_diagnostics", True):
                metrics.update(_collect_sac_diagnostics(model))
            if _is_truthy("log_traffic_metrics_during_training", False):
                base_env = _resolve_base_env(getattr(model, "env", None))
                final_info = getattr(base_env, "metrics", [])[-1] if getattr(base_env, "metrics", None) else {}
                traffic_metrics, _ = build_namespaced_metrics(final_info, include_agent_metrics_local=False)
                metrics.update({f"train/traffic/{key}": float(value) for key, value in traffic_metrics.items()})
            _log_metrics(metrics, step)
            last_train_step = step
            if self.save_checkpoints and self.checkpoint_freq > 0 and step % self.checkpoint_freq == 0 and step != last_checkpoint_step:
                _save_model_checkpoint(model, step)
                last_checkpoint_step = step

        def _log_eval_metrics(model, step: int, force: bool = False) -> None:
            nonlocal last_eval_step
            if step <= 0:
                return
            if not force and step % eval_freq != 0:
                return
            if step == last_eval_step:
                return
            if eval_env is None or eval_episodes <= 0:
                return

            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
            metrics = {
                "eval/num_timesteps": float(step),
                "eval/mean_reward": float(mean_reward),
                "eval/std_reward": float(std_reward),
            }
            _log_metrics(metrics, step)
            last_eval_step = step

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                _log_train_metrics(self.locals, self.model, self.num_timesteps)
                _log_eval_metrics(self.model, self.num_timesteps)
                return True

            def _on_rollout_end(self) -> None:
                _log_eval_metrics(self.model, self.num_timesteps)
                return None

            def _on_training_end(self) -> None:
                _log_train_metrics(self.locals, self.model, self.num_timesteps, force=True)
                _log_eval_metrics(self.model, self.num_timesteps, force=True)
                _save_model_checkpoint(self.model, self.num_timesteps, final=True)
                if _is_sac_model(self.model) and _is_truthy("log_sac_diagnostics", True):
                    warnings: Dict[str, float] = {}
                    total_timesteps = float(getattr(self.model, "num_timesteps", self.num_timesteps))
                    learning_starts = float(getattr(self.model, "learning_starts", 0.0))
                    batch_size = safe_scalar(getattr(self.model, "batch_size", None))
                    replay_buffer_size = last_sac_metrics.get("train/replay_buffer_size", _get_replay_buffer_size(self.model))
                    actor_loss = last_sac_metrics.get("train/actor_loss")
                    critic_loss = last_sac_metrics.get("train/critic_loss")

                    if total_timesteps <= learning_starts:
                        warnings["warnings/sac_no_gradient_updates"] = True
                    if batch_size is not None and replay_buffer_size is not None and replay_buffer_size < batch_size:
                        warnings["warnings/sac_replay_buffer_smaller_than_batch"] = True
                    if actor_loss is None and critic_loss is None:
                        warnings["warnings/sac_losses_missing"] = True

                    if warnings:
                        _log_metrics(warnings, self.num_timesteps)
                return None

        self._callback = Callback()
        return self._callback
