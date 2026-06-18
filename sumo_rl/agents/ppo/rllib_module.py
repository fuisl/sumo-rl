"""RLlib PPO modules for graph-observation DCRNN variants."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sumo_rl.models.dcrnn import DCRNNBackbone


_SHARED_BACKBONE_STATE_KEY = "__shared_backbone__"


def _build_mlp_heads(*, input_dim: int, policy_hidden_dim: int, value_hidden_dim: int, action_dim: int):
    from torch import nn

    policy_head = nn.Sequential(
        nn.Linear(input_dim, policy_hidden_dim),
        nn.ReLU(),
        nn.Linear(policy_hidden_dim, action_dim),
    )
    value_head = nn.Sequential(
        nn.Linear(input_dim, value_hidden_dim),
        nn.ReLU(),
        nn.Linear(value_hidden_dim, 1),
    )
    return policy_head, value_head


def build_ppo_dcrnn_module_class():
    from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.torch import TorchRLModule

    class PPODCRNNTorchRLModule(TorchRLModule, DefaultPPORLModule):
        """PPO RLModule with a per-policy DCRNN backbone."""

        def setup(self):
            self.backbone = DCRNNBackbone.from_model_config(self.observation_space, self.model_config)
            action_dim = int(self.action_space.n)
            value_hidden_dim = int(self.model_config.get("value_hidden_dim", self.backbone.hidden_dim))
            policy_hidden_dim = int(self.model_config.get("policy_hidden_dim", self.backbone.hidden_dim))
            self.policy_head, self.value_head = _build_mlp_heads(
                input_dim=self.backbone.output_dim,
                policy_hidden_dim=policy_hidden_dim,
                value_hidden_dim=value_hidden_dim,
                action_dim=action_dim,
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
                Columns.EMBEDDINGS: latent,
                Columns.ACTION_DIST_INPUTS: self.policy_head(latent),
                Columns.VF_PREDS: self.value_head(latent).squeeze(-1),
            }

        def compute_values(self, batch: Dict[str, Any], embeddings=None):
            latent = embeddings if embeddings is not None else self.backbone(batch[Columns.OBS])
            return self.value_head(latent).squeeze(-1)

        def get_initial_state(self) -> dict:
            return {}

        def get_non_inference_attributes(self):
            return ["value_head"]

    PPODCRNNTorchRLModule.__name__ = "PPODCRNNTorchRLModule"
    return PPODCRNNTorchRLModule


def build_ppo_dcrnn_shared_module_class():
    from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.torch import TorchRLModule

    class PPODCRNNSharedTorchRLModule(TorchRLModule, DefaultPPORLModule):
        """Per-agent PPO module with shared parent-owned DCRNN backbone."""

        def setup(self):
            self.agent_index = int(self.model_config["agent_index"])
            self.shared_backbone = None
            action_dim = int(self.action_space.n)
            hidden_dim = int(self.model_config.get("hid_dim", self.model_config.get("hidden_dim", 128)))
            output_dim = hidden_dim + int(
                self.model_config.get(
                    "input_dim",
                    getattr(self.observation_space, "shape", (0, 0, 0))[-1],
                )
            )
            pre_encoder = dict(self.model_config.get("pre_encoder", {}) or {})
            if bool(pre_encoder.get("enabled", False)):
                output_dim = hidden_dim + int(pre_encoder.get("hidden_dim", hidden_dim))
            value_hidden_dim = int(self.model_config.get("value_hidden_dim", hidden_dim))
            policy_hidden_dim = int(self.model_config.get("policy_hidden_dim", hidden_dim))
            self.policy_head, self.value_head = _build_mlp_heads(
                input_dim=output_dim,
                policy_hidden_dim=policy_hidden_dim,
                value_hidden_dim=value_hidden_dim,
                action_dim=action_dim,
            )

        def set_shared_backbone(self, backbone: DCRNNBackbone) -> None:
            object.__setattr__(self, "_shared_backbone_ref", backbone)

        def _require_shared_backbone(self) -> DCRNNBackbone:
            backbone = getattr(self, "_shared_backbone_ref", None)
            if backbone is None:
                raise RuntimeError("Shared PPO DCRNN backbone has not been attached to the module.")
            return backbone

        def _encode(self, batch: Dict[str, Any]):
            return self._require_shared_backbone().forward_for_agent(
                batch[Columns.OBS],
                agent_index=self.agent_index,
            )

        def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self._encode(batch)
            return {Columns.ACTION_DIST_INPUTS: self.policy_head(latent)}

        def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self._encode(batch)
            return {
                Columns.ACTION_DIST_INPUTS: self.policy_head(latent),
                Columns.VF_PREDS: self.value_head(latent).squeeze(-1),
            }

        def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            del kwargs
            latent = self._encode(batch)
            return {
                Columns.EMBEDDINGS: latent,
                Columns.ACTION_DIST_INPUTS: self.policy_head(latent),
                Columns.VF_PREDS: self.value_head(latent).squeeze(-1),
            }

        def compute_values(self, batch: Dict[str, Any], embeddings=None):
            latent = embeddings if embeddings is not None else self._encode(batch)
            return self.value_head(latent).squeeze(-1)

        def get_initial_state(self) -> dict:
            return {}

        def get_non_inference_attributes(self):
            return ["value_head"]

    PPODCRNNSharedTorchRLModule.__name__ = "PPODCRNNSharedTorchRLModule"
    return PPODCRNNSharedTorchRLModule


def build_ppo_dcrnn_shared_multi_module_class():
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    from ray.rllib.utils.numpy import convert_to_numpy
    from ray.rllib.utils.torch_utils import convert_to_torch_tensor

    class PPODCRNNSharedMultiRLModule(MultiRLModule):
        """Multi-agent PPO module that owns one shared DCRNN backbone."""

        def setup(self):
            self._rl_modules = {}
            module_specs = self.rl_module_specs
            framework = None
            first_spec = next(iter(module_specs.values()))
            model_config = dict(first_spec.model_config or {})
            self.shared_backbone = DCRNNBackbone.from_shared_ppo_model_config(
                first_spec.observation_space,
                model_config,
            )
            for module_id, rl_module_spec in module_specs.items():
                module = rl_module_spec.build()
                module.set_shared_backbone(self.shared_backbone)
                self._rl_modules[module_id] = module
                if framework is None:
                    framework = module.framework
                else:
                    assert module.framework in [None, framework]
            self.framework = framework

        def move_shared_backbone_to_device(self, device) -> None:
            self.shared_backbone.to(device)

        def get_state(
            self,
            components=None,
            *,
            not_components=None,
            inference_only: bool = False,
            **kwargs,
        ) -> Dict[str, Any]:
            state = super().get_state(
                components=components,
                not_components=not_components,
                inference_only=inference_only,
                **kwargs,
            )
            state[_SHARED_BACKBONE_STATE_KEY] = convert_to_numpy(self.shared_backbone.state_dict())
            return state

        def set_state(self, state: Dict[str, Any]) -> None:
            state = dict(state or {})
            shared_backbone_state = state.pop(_SHARED_BACKBONE_STATE_KEY, None)
            if shared_backbone_state is not None:
                try:
                    device = next(self.shared_backbone.parameters()).device
                except StopIteration:
                    device = None
                self.shared_backbone.load_state_dict(
                    convert_to_torch_tensor(shared_backbone_state, device=device),
                    strict=True,
                )
            super().set_state(state)

    PPODCRNNSharedMultiRLModule.__name__ = "PPODCRNNSharedMultiRLModule"
    return PPODCRNNSharedMultiRLModule


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


def build_ppo_dcrnn_shared_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    return RLModuleSpec(
        module_class=build_ppo_dcrnn_shared_module_class(),
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {},
    )


def build_ppo_dcrnn_shared_multi_module_spec(
    rl_module_specs: Dict[str, Any],
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

    return MultiRLModuleSpec(
        multi_rl_module_class=build_ppo_dcrnn_shared_multi_module_class(),
        rl_module_specs=rl_module_specs,
        model_config=model_config or {},
    )
