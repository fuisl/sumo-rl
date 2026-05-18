"""Extensible SAC RLModule boundary for the native-discrete RLlib path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional


DEFAULT_CUSTOM_SAC_MODEL_CONFIG: Dict[str, Any] = {
    "architecture_tag": "custom_sac_mlp",
    "actor": {
        "encoder": {
            "type": "mlp",
            "hidden_dims": [256, 256],
            "activation": "tanh",
        },
        "head": {
            "hidden_dims": [],
            "activation": "relu",
        },
    },
    "critic": {
        "encoder": {
            "type": "mlp",
            "hidden_dims": [256, 256],
            "activation": "tanh",
        },
        "head": {
            "hidden_dims": [],
            "activation": "relu",
        },
        "twin_q": True,
    },
    "communication": {
        "enabled": False,
        "type": "none",
        "apply_to": ["actor", "critic"],
        "scope": "module",
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _int_list(value: Any, *, field_name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list of positive integers.")
    result = [int(item) for item in value]
    if any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain positive integers.")
    return result


def normalize_custom_sac_model_config(model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return an RLlib-compatible model config plus project-owned SAC metadata."""

    try:
        from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    except ImportError as exc:  # pragma: no cover - exercised only without RLlib.
        raise ImportError("custom SAC requires Ray RLlib to be installed.") from exc

    incoming = dict(model_config or {})
    custom_updates = {
        key: incoming.pop(key)
        for key in list(incoming)
        if key in DEFAULT_CUSTOM_SAC_MODEL_CONFIG
    }
    custom_config = _deep_update(DEFAULT_CUSTOM_SAC_MODEL_CONFIG, custom_updates)

    actor_encoder = custom_config["actor"]["encoder"]
    actor_head = custom_config["actor"]["head"]
    critic_encoder = custom_config["critic"]["encoder"]
    critic_head = custom_config["critic"]["head"]
    communication = custom_config["communication"]

    if str(actor_encoder.get("type", "mlp")).lower() != "mlp":
        raise ValueError("custom SAC currently supports actor.encoder.type=mlp.")
    if str(critic_encoder.get("type", "mlp")).lower() != "mlp":
        raise ValueError("custom SAC currently supports critic.encoder.type=mlp.")

    communication_type = str(communication.get("type", "none") or "none").lower()
    if communication_type not in {"none", "identity", "message_passing", "gat"}:
        raise ValueError(
            "custom SAC communication.type must be one of: none, identity, message_passing, gat."
        )
    communication["type"] = communication_type
    communication["enabled"] = bool(communication.get("enabled", False)) and communication_type != "none"
    communication["apply_to"] = [
        str(item).lower()
        for item in communication.get("apply_to", ["actor", "critic"])
        if str(item).lower() in {"actor", "critic"}
    ]

    merged: Dict[str, Any] = asdict(DefaultModelConfig())
    merged.update(incoming)
    merged["architecture_tag"] = str(custom_config.get("architecture_tag", "custom_sac_mlp"))
    merged["custom_sac"] = custom_config
    merged["fcnet_hiddens"] = _int_list(actor_encoder.get("hidden_dims"), field_name="actor.encoder.hidden_dims")
    merged["fcnet_activation"] = str(actor_encoder.get("activation", "relu") or "relu")
    merged["head_fcnet_hiddens"] = _int_list(actor_head.get("hidden_dims"), field_name="actor.head.hidden_dims")
    merged["head_fcnet_activation"] = str(actor_head.get("activation", "relu") or "relu")
    merged["critic_fcnet_hiddens"] = _int_list(
        critic_encoder.get("hidden_dims"),
        field_name="critic.encoder.hidden_dims",
    )
    merged["critic_fcnet_activation"] = str(critic_encoder.get("activation", "relu") or "relu")
    merged["critic_head_fcnet_hiddens"] = _int_list(
        critic_head.get("hidden_dims"),
        field_name="critic.head.hidden_dims",
    )
    merged["critic_head_fcnet_activation"] = str(critic_head.get("activation", "relu") or "relu")
    merged["twin_q"] = bool(custom_config["critic"].get("twin_q", True))
    return merged


def build_custom_sac_catalog_class():
    import gymnasium as gym
    from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
    from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig

    class CustomSACCatalog(SACCatalog):
        """Catalog that lets actor and discrete twin-Q architectures diverge."""

        def __init__(self, observation_space, action_space, model_config_dict, view_requirements=None):
            super().__init__(observation_space, action_space, model_config_dict, view_requirements)
            critic_hidden = list(self._model_config_dict.get("critic_fcnet_hiddens") or [])
            if critic_hidden:
                self.qf_latent_dims = [critic_hidden[-1]]
                required_qf_output_dim = self.action_space.n if isinstance(self.action_space, gym.spaces.Discrete) else 1
                self.qf_head_config = MLPHeadConfig(
                    input_dims=self.qf_latent_dims,
                    hidden_layer_dims=self._model_config_dict.get("critic_head_fcnet_hiddens") or [],
                    hidden_layer_activation=self._model_config_dict.get("critic_head_fcnet_activation") or "relu",
                    output_layer_activation="linear",
                    output_layer_dim=required_qf_output_dim,
                )

        def _build_qf_encoder_discrete(self, framework: str):
            critic_hidden = list(self._model_config_dict.get("critic_fcnet_hiddens") or [])
            if not critic_hidden:
                return super()._build_qf_encoder_discrete(framework=framework)
            self.qf_encoder_config = MLPEncoderConfig(
                input_dims=self.observation_space.shape,
                hidden_layer_dims=critic_hidden[:-1],
                hidden_layer_activation=self._model_config_dict.get("critic_fcnet_activation") or "relu",
                output_layer_dim=critic_hidden[-1],
                output_layer_activation=self._model_config_dict.get("critic_fcnet_activation") or "relu",
            )
            return self.qf_encoder_config.build(framework=framework)

    CustomSACCatalog.__name__ = "CustomSACCatalog"
    return CustomSACCatalog


def build_custom_sac_module_class():
    from ray.rllib.algorithms.sac.sac_learner import (
        ACTION_LOG_PROBS,
        ACTION_LOG_PROBS_NEXT,
        ACTION_PROBS,
        ACTION_PROBS_NEXT,
        QF_PREDS,
        QF_TARGET_NEXT,
        QF_TWIN_PREDS,
    )
    from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.models.base import ENCODER_OUT
    from ray.rllib.utils.framework import try_import_torch

    torch, nn = try_import_torch()

    class CustomSACCommunicationBlock(nn.Module):
        """Swappable latent hook for future graph/message-passing operators."""

        def __init__(self, kind: str):
            super().__init__()
            self.kind = kind

        def forward(self, latent):
            return latent

    class CustomSACTorchRLModule(DefaultSACTorchRLModule):
        """Project-owned SAC module boundary for discrete-action traffic lights."""

        def __init__(self, *args, **kwargs):
            if kwargs.get("catalog_class") is None:
                kwargs["catalog_class"] = build_custom_sac_catalog_class()
            super().__init__(*args, **kwargs)

        def setup(self):
            super().setup()
            custom_config = self.model_config.get("custom_sac", {})
            communication = dict(custom_config.get("communication", {}) or {})
            self._architecture_tag = str(self.model_config.get("architecture_tag", "custom_sac_mlp"))
            self._actor_config = dict(custom_config.get("actor", {}) or {})
            self._critic_config = dict(custom_config.get("critic", {}) or {})
            self._communication_config = communication
            self._communication_enabled = bool(communication.get("enabled", False))
            self._communication_type = str(communication.get("type", "none") or "none")
            self._communication_apply_to = set(communication.get("apply_to", ["actor", "critic"]) or [])
            self.actor_communication = CustomSACCommunicationBlock(self._communication_type)
            self.critic_communication = CustomSACCommunicationBlock(self._communication_type)

        def _apply_actor_communication(self, latent):
            if self._communication_enabled and "actor" in self._communication_apply_to:
                return self.actor_communication(latent)
            return latent

        def _apply_critic_communication(self, latent):
            if self._communication_enabled and "critic" in self._communication_apply_to:
                return self.critic_communication(latent)
            return latent

        def _forward_inference(self, batch):
            pi_encoder_outs = self.pi_encoder(batch)
            actor_latent = self._apply_actor_communication(pi_encoder_outs[ENCODER_OUT])
            return {Columns.ACTION_DIST_INPUTS: self.pi(actor_latent)}

        def _forward_exploration(self, batch, **kwargs):
            del kwargs
            return self._forward_inference(batch)

        def _forward_train_discrete(self, batch):
            output = {}
            batch_curr = {Columns.OBS: batch[Columns.OBS]}
            batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

            pi_encoder_next_outs = self.pi_encoder(batch_next)
            actor_latent_next = self._apply_actor_communication(pi_encoder_next_outs[ENCODER_OUT])
            action_logits_next = self.pi(actor_latent_next)
            action_probs_next = torch.nn.functional.softmax(action_logits_next, dim=-1)

            output[ACTION_PROBS_NEXT] = action_probs_next
            output[ACTION_LOG_PROBS_NEXT] = action_probs_next.log()
            output[QF_TARGET_NEXT] = self.forward_target(batch_next, squeeze=False)

            output[QF_PREDS] = self._qf_forward_train_helper(
                batch_curr,
                self.qf_encoder,
                self.qf,
                squeeze=False,
            )
            if self.twin_q:
                output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                    batch_curr,
                    self.qf_twin_encoder,
                    self.qf_twin,
                    squeeze=False,
                )

            pi_encoder_outs = self.pi_encoder(batch_curr)
            actor_latent = self._apply_actor_communication(pi_encoder_outs[ENCODER_OUT])
            action_logits = self.pi(actor_latent)
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            output[ACTION_PROBS] = action_probs
            output[ACTION_LOG_PROBS] = action_probs.log()
            return output

        def _qf_forward_train_helper(self, batch, encoder, head, squeeze=True):
            qf_encoder_outs = encoder(batch)
            critic_latent = self._apply_critic_communication(qf_encoder_outs[ENCODER_OUT])
            qf_out = head(critic_latent)
            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out

        def get_non_inference_attributes(self):
            try:
                non_inference_attributes = list(super().get_non_inference_attributes())
            except AttributeError:
                non_inference_attributes = []
            for attr in (
                "_architecture_tag",
                "_actor_config",
                "_critic_config",
                "_communication_config",
                "_communication_enabled",
                "_communication_type",
                "_communication_apply_to",
                "actor_communication",
                "critic_communication",
            ):
                if attr not in non_inference_attributes:
                    non_inference_attributes.append(attr)
            return non_inference_attributes

    CustomSACTorchRLModule.__name__ = "CustomSACTorchRLModule"
    return CustomSACTorchRLModule


def build_custom_sac_multi_module_class():
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

    class CustomSACMultiRLModule(MultiRLModule):
        """Project-owned multi-agent SAC boundary for future shared communication."""

        def setup(self):
            super().setup()
            self._custom_sac_model_config = dict(self.model_config.get("custom_sac", {}) or {})
            communication = self._custom_sac_model_config.get("communication", {}) or {}
            self._communication_enabled = bool(communication.get("enabled", False))
            self._communication_type = str(communication.get("type", "none") or "none")

    CustomSACMultiRLModule.__name__ = "CustomSACMultiRLModule"
    return CustomSACMultiRLModule


def build_custom_sac_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    module_class = build_custom_sac_module_class()
    normalized_model_config = normalize_custom_sac_model_config(model_config)
    return RLModuleSpec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=normalized_model_config,
    )


def build_custom_sac_multi_module_spec(
    rl_module_specs: Dict[str, Any],
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

    return MultiRLModuleSpec(
        multi_rl_module_class=build_custom_sac_multi_module_class(),
        rl_module_specs=rl_module_specs,
        model_config=normalize_custom_sac_model_config(model_config),
    )
