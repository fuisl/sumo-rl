"""RLlib RLModule wrapper for PPO with a shared DCRNN graph backbone."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sumo_rl.models.dcrnn import DCRNNBackbone


def build_ppo_dcrnn_module_class():
    from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.torch import TorchRLModule

    class PPODCRNNTorchRLModule(TorchRLModule, DefaultPPORLModule):
        """PPO RLModule with a shared MLP+DCRNN graph encoder."""

        def setup(self):
            self.backbone = DCRNNBackbone.from_model_config(self.observation_space, self.model_config)
            action_dim = int(self.action_space.n)
            value_hidden_dim = int(self.model_config.get("value_hidden_dim", self.backbone.hidden_dim))
            policy_hidden_dim = int(self.model_config.get("policy_hidden_dim", self.backbone.hidden_dim))
            from torch import nn

            self.policy_head = nn.Sequential(
                nn.Linear(self.backbone.output_dim, policy_hidden_dim),
                nn.ReLU(),
                nn.Linear(policy_hidden_dim, action_dim),
            )
            self.value_head = nn.Sequential(
                nn.Linear(self.backbone.output_dim, value_hidden_dim),
                nn.ReLU(),
                nn.Linear(value_hidden_dim, 1),
            )

        def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self.backbone(batch[Columns.OBS])
            return {Columns.ACTION_DIST_INPUTS: self.policy_head(latent)}

        def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self.backbone(batch[Columns.OBS])
            return {
                Columns.ACTION_DIST_INPUTS: self.policy_head(latent),
                Columns.VF_PREDS: self.value_head(latent).squeeze(-1),
            }

        def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self.backbone(batch[Columns.OBS])
            return {
                Columns.ACTION_DIST_INPUTS: self.policy_head(latent),
                Columns.VF_PREDS: self.value_head(latent).squeeze(-1),
            }

        def compute_values(self, batch: Dict[str, Any], embeddings=None):
            del embeddings
            latent = self.backbone(batch[Columns.OBS])
            return self.value_head(latent).squeeze(-1)

        def get_initial_state(self) -> dict:
            return {}

        def get_non_inference_attributes(self):
            return ["value_head"]

    PPODCRNNTorchRLModule.__name__ = "PPODCRNNTorchRLModule"
    return PPODCRNNTorchRLModule


def build_ppo_dcrnn_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    return RLModuleSpec(
        module_class=build_ppo_dcrnn_module_class(),
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {},
    )
