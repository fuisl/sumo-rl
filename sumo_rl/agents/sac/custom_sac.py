from __future__ import annotations

from typing import Any, Dict, Optional


def build_custom_sac_module_class():
    from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule

    class CustomSACTorchRLModule(DefaultSACTorchRLModule):
        """Project-owned SAC module boundary."""

        def setup(self):
            super().setup()
            self._architecture_tag = str(self.model_config.get("architecture_tag", "custom_sac_mlp"))

        def get_non_inference_attributes(self):
            try:
                non_inference_attributes = list(super().get_non_inference_attributes())
            except AttributeError:
                non_inference_attributes = []
            if "_architecture_tag" not in non_inference_attributes:
                non_inference_attributes.append("_architecture_tag")
            return non_inference_attributes

    CustomSACTorchRLModule.__name__ = "CustomSACTorchRLModule"
    return CustomSACTorchRLModule


def build_custom_sac_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    module_class = build_custom_sac_module_class()
    return RLModuleSpec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {"architecture_tag": "custom_sac_mlp"},
    )
