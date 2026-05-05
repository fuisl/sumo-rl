"""Helpers for Stable-Baselines3 experiment logging."""

from __future__ import annotations

import numpy as np


class SB3WandbCallback:
    def __init__(self, wandb_run, csv_run, log_freq: int = 1000):
        self.wandb_run = wandb_run
        self.csv_run = csv_run
        self.log_freq = max(1, int(log_freq))
        self._callback = None

    def build(self):
        from stable_baselines3.common.callbacks import BaseCallback

        wandb_run = self.wandb_run
        csv_run = self.csv_run
        log_freq = self.log_freq

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                if self.n_calls % log_freq == 0:
                    rewards = self.locals.get("rewards")
                    metrics = {"train/num_timesteps": float(self.num_timesteps)}
                    if rewards is not None:
                        metrics["train/reward_mean"] = float(np.mean(rewards))
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=self.num_timesteps)
                    if csv_run is not None:
                        csv_run.log(metrics, step=self.num_timesteps)
                return True

        self._callback = Callback()
        return self._callback
