"""Custom PPO learner for the shared-backbone DCRNN graph variant."""

from __future__ import annotations

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import ALL_MODULES


class PPOSharedEncoderTorchLearner(PPOTorchLearner):
    """PPO learner that optimizes the shared backbone and all heads once."""

    def _make_module(self):
        module = super()._make_module()
        move_shared_backbone = getattr(module, "move_shared_backbone_to_device", None)
        if callable(move_shared_backbone):
            move_shared_backbone(self._device)
        return module

    def configure_optimizers(self) -> None:
        torch = __import__("torch")

        params = []
        seen_param_refs = set()

        def _append_unique_params(items) -> None:
            for param in items:
                if not getattr(param, "requires_grad", False):
                    continue
                param_ref = self.get_param_ref(param)
                if param_ref in seen_param_refs:
                    continue
                seen_param_refs.add(param_ref)
                params.append(param)

        shared_backbone = getattr(self.module, "shared_backbone", None)
        if shared_backbone is None:
            raise ValueError("Shared PPO learner requires a multi-module with a shared_backbone attribute.")
        _append_unique_params(shared_backbone.parameters())

        for module in self.module.values():
            _append_unique_params(module.parameters())

        optimizer = torch.optim.Adam(params)
        self.register_optimizer(
            module_id=ALL_MODULES,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=self.config.lr,
        )
