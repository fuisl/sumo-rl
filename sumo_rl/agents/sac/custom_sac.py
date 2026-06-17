"""Extensible SAC RLModule boundary for the native-discrete RLlib path."""

from __future__ import annotations

import functools
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional

from sumo_rl.models.dcrnn import DCRNNBackbone


DEFAULT_CUSTOM_SAC_MODEL_CONFIG: Dict[str, Any] = {
    "architecture_tag": "sac_mlp",
    "encoder_layout": "separate",
    "shared_encoder": {},
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

_ENCODER_LAYOUTS = {"separate", "actor_only", "shared"}
_ACTOR_ONLY_ARCHITECTURES = {"sac_dcrnn_actor", "sac_dcrnn_actor_mlp"}
_SHARED_ARCHITECTURES = {"sac_dcrnn_shared_mlp"}


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


def _default_encoder_layout_for_architecture(architecture_tag: str) -> str:
    architecture_tag = str(architecture_tag or "sac_mlp")
    if architecture_tag in _ACTOR_ONLY_ARCHITECTURES:
        return "actor_only"
    if architecture_tag in _SHARED_ARCHITECTURES:
        return "shared"
    return "separate"


def normalize_custom_sac_model_config(model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return an RLlib-compatible model config plus project-owned SAC metadata."""

    try:
        from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    except ImportError as exc:  # pragma: no cover - exercised only without RLlib.
        raise ImportError("custom SAC requires Ray RLlib to be installed.") from exc

    incoming = dict(model_config or {})
    embedded_custom_sac = incoming.pop("custom_sac", None)
    top_level_custom_updates = {
        key: incoming.pop(key)
        for key in list(incoming)
        if key in DEFAULT_CUSTOM_SAC_MODEL_CONFIG
    }
    custom_updates = (
        _deep_update(dict(embedded_custom_sac or {}), top_level_custom_updates)
        if isinstance(embedded_custom_sac, dict)
        else top_level_custom_updates
    )
    custom_config = _deep_update(DEFAULT_CUSTOM_SAC_MODEL_CONFIG, custom_updates)

    actor_encoder = custom_config["actor"]["encoder"]
    actor_head = custom_config["actor"]["head"]
    critic_encoder = custom_config["critic"]["encoder"]
    critic_head = custom_config["critic"]["head"]
    communication = custom_config["communication"]
    architecture_tag = str(custom_config.get("architecture_tag", "sac_mlp") or "sac_mlp")
    encoder_layout = (
        str(custom_config.get("encoder_layout") or _default_encoder_layout_for_architecture(architecture_tag)).lower()
    )
    if "encoder_layout" not in custom_updates:
        encoder_layout = _default_encoder_layout_for_architecture(architecture_tag)
    if encoder_layout not in _ENCODER_LAYOUTS:
        raise ValueError("custom SAC encoder_layout must be one of: separate, actor_only, shared.")
    custom_config["encoder_layout"] = encoder_layout

    actor_encoder_type = str(actor_encoder.get("type", "mlp")).lower()
    critic_encoder_type = str(critic_encoder.get("type", "mlp")).lower()
    if actor_encoder_type not in {"mlp", "dcrnn"}:
        raise ValueError("custom SAC currently supports actor.encoder.type in {mlp, dcrnn}.")
    if critic_encoder_type not in {"mlp", "dcrnn"}:
        raise ValueError("custom SAC currently supports critic.encoder.type in {mlp, dcrnn}.")
    actor_encoder["type"] = actor_encoder_type
    critic_encoder["type"] = critic_encoder_type

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

    shared_encoder = dict(custom_config.get("shared_encoder", {}) or {})
    actor_encoder_supplied = (
        isinstance(top_level_custom_updates.get("actor"), dict) and "encoder" in top_level_custom_updates["actor"]
    )
    critic_encoder_supplied = (
        isinstance(top_level_custom_updates.get("critic"), dict) and "encoder" in top_level_custom_updates["critic"]
    )
    if encoder_layout == "shared":
        if actor_encoder_supplied or critic_encoder_supplied:
            raise ValueError(
                "custom SAC shared encoder layout does not accept actor.encoder or critic.encoder overrides."
            )
        if not shared_encoder:
            raise ValueError("custom SAC shared encoder layout requires model_config.shared_encoder.")
        shared_encoder_type = str(shared_encoder.get("type", "") or "").lower()
        if shared_encoder_type != "dcrnn":
            raise ValueError("custom SAC shared encoder layout currently requires shared_encoder.type='dcrnn'.")
        shared_encoder["type"] = shared_encoder_type
    custom_config["shared_encoder"] = shared_encoder

    merged: Dict[str, Any] = asdict(DefaultModelConfig())
    merged.update(incoming)
    merged["architecture_tag"] = architecture_tag
    merged["custom_sac"] = custom_config
    if encoder_layout == "shared":
        merged["fcnet_hiddens"] = []
        merged["fcnet_activation"] = "relu"
    elif actor_encoder_type == "mlp":
        merged["fcnet_hiddens"] = _int_list(actor_encoder.get("hidden_dims"), field_name="actor.encoder.hidden_dims")
        merged["fcnet_activation"] = str(actor_encoder.get("activation", "relu") or "relu")
    else:
        merged["fcnet_hiddens"] = []
        merged["fcnet_activation"] = "relu"
    merged["head_fcnet_hiddens"] = _int_list(actor_head.get("hidden_dims"), field_name="actor.head.hidden_dims")
    merged["head_fcnet_activation"] = str(actor_head.get("activation", "relu") or "relu")
    if encoder_layout == "shared":
        merged["critic_fcnet_hiddens"] = []
        merged["critic_fcnet_activation"] = "relu"
    elif critic_encoder_type == "mlp":
        merged["critic_fcnet_hiddens"] = _int_list(
            critic_encoder.get("hidden_dims"),
            field_name="critic.encoder.hidden_dims",
        )
        merged["critic_fcnet_activation"] = str(critic_encoder.get("activation", "relu") or "relu")
    else:
        merged["critic_fcnet_hiddens"] = []
        merged["critic_fcnet_activation"] = "relu"
    merged["critic_head_fcnet_hiddens"] = _int_list(
        critic_head.get("hidden_dims"),
        field_name="critic.head.hidden_dims",
    )
    merged["critic_head_fcnet_activation"] = str(critic_head.get("activation", "relu") or "relu")
    merged["twin_q"] = bool(custom_config["critic"].get("twin_q", True))
    return merged


def _flat_dim(shape: Any) -> int:
    dims = tuple(int(dim) for dim in tuple(shape or ()))
    if not dims:
        raise ValueError("custom SAC requires observations with a concrete shape.")
    return int(math.prod(dims))


def build_custom_sac_catalog_class():
    import gymnasium as gym
    from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.models.base import ENCODER_OUT
    from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig
    from ray.rllib.utils.framework import try_import_torch

    torch, nn = try_import_torch()

    class _TorchFlattenMLPEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: list[int], activation: str, output_dim: int):
            super().__init__()
            layers = []
            previous_dim = int(input_dim)
            activation_name = str(activation or "relu").lower()
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(previous_dim, int(hidden_dim)))
                if activation_name == "tanh":
                    layers.append(nn.Tanh())
                elif activation_name == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation_name in {"identity", "linear", "none"}:
                    layers.append(nn.Identity())
                else:
                    layers.append(nn.ReLU())
                previous_dim = int(hidden_dim)
            layers.append(nn.Linear(previous_dim, int(output_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, batch):
            obs = batch[Columns.OBS].float().reshape(batch[Columns.OBS].shape[0], -1)
            return {ENCODER_OUT: self.net(obs)}

    class CustomSACCatalog(SACCatalog):
        """Catalog that lets actor and discrete twin-Q architectures diverge."""

        def _custom_sac_config(self) -> Dict[str, Any]:
            custom_sac = dict(self._model_config_dict.get("custom_sac", {}) or {})
            return custom_sac

        def _actor_encoder_type(self) -> str:
            custom_sac = self._custom_sac_config()
            actor_config = dict(custom_sac.get("actor", {}) or {})
            encoder_config = dict(actor_config.get("encoder", {}) or {})
            return str(encoder_config.get("type", "mlp") or "mlp").lower()

        def _critic_encoder_type(self) -> str:
            custom_sac = self._custom_sac_config()
            critic_config = dict(custom_sac.get("critic", {}) or {})
            encoder_config = dict(critic_config.get("encoder", {}) or {})
            return str(encoder_config.get("type", "mlp") or "mlp").lower()

        def _encoder_layout(self) -> str:
            custom_sac = self._custom_sac_config()
            return str(custom_sac.get("encoder_layout", "separate") or "separate").lower()

        def _shared_encoder_type(self) -> str:
            custom_sac = self._custom_sac_config()
            shared_encoder = dict(custom_sac.get("shared_encoder", {}) or {})
            return str(shared_encoder.get("type", "none") or "none").lower()

        def _uses_graph_override(self) -> bool:
            return (
                self._actor_encoder_type() == "dcrnn"
                or self._critic_encoder_type() == "dcrnn"
                or (self._encoder_layout() == "shared" and self._shared_encoder_type() == "dcrnn")
            )

        def _determine_components_hook(self):
            if not self._uses_graph_override():
                return super()._determine_components_hook()

            self._action_dist_class_fn = functools.partial(
                self._get_dist_cls_from_action_space,
                action_space=self.action_space,
            )
            flat_obs_dim = _flat_dim(self.observation_space.shape)
            self._flat_obs_dim = flat_obs_dim
            self._encoder_config = None
            actor_hidden = list(self._model_config_dict.get("fcnet_hiddens") or [])
            actor_latent_dim = int(actor_hidden[-1]) if actor_hidden else flat_obs_dim
            self.latent_dims = (actor_latent_dim,)

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

        def build_encoder(self, framework: str):
            if not self._uses_graph_override():
                return super().build_encoder(framework=framework)
            actor_hidden = list(self._model_config_dict.get("fcnet_hiddens") or [])
            flat_obs_dim = getattr(self, "_flat_obs_dim", _flat_dim(self.observation_space.shape))
            latent_dim = int(actor_hidden[-1]) if actor_hidden else flat_obs_dim
            return _TorchFlattenMLPEncoder(
                flat_obs_dim,
                actor_hidden[:-1],
                self._model_config_dict.get("fcnet_activation") or "relu",
                latent_dim,
            )

        def _build_qf_encoder_discrete(self, framework: str):
            if self._uses_graph_override():
                critic_hidden = list(self._model_config_dict.get("critic_fcnet_hiddens") or [])
                flat_obs_dim = getattr(self, "_flat_obs_dim", _flat_dim(self.observation_space.shape))
                latent_dim = int(critic_hidden[-1]) if critic_hidden else flat_obs_dim
                return _TorchFlattenMLPEncoder(
                    flat_obs_dim,
                    critic_hidden[:-1],
                    self._model_config_dict.get("critic_fcnet_activation") or "relu",
                    latent_dim,
                )
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

    def _activation_layer(name: str):
        activation = str(name or "relu").lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation in {"identity", "linear", "none"}:
            return nn.Identity()
        raise ValueError(f"Unsupported SAC activation: {name!r}.")

    def _build_mlp(input_dim: int, hidden_dims: list[int], activation: str, output_dim: int):
        layers = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(_activation_layer(activation))
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, int(output_dim)))
        return nn.Sequential(*layers)

    class _TorchDCRNNEncoder(nn.Module):
        def __init__(self, backbone: DCRNNBackbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, batch):
            return {ENCODER_OUT: self.backbone(batch[Columns.OBS])}

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
            self._architecture_tag = str(self.model_config.get("architecture_tag", "sac_mlp"))
            self._actor_config = dict(custom_config.get("actor", {}) or {})
            self._critic_config = dict(custom_config.get("critic", {}) or {})
            self._shared_encoder_config = dict(custom_config.get("shared_encoder", {}) or {})
            self._encoder_layout = str(custom_config.get("encoder_layout", "separate") or "separate").lower()
            self._communication_config = communication
            self._communication_enabled = bool(communication.get("enabled", False))
            self._communication_type = str(communication.get("type", "none") or "none")
            self._communication_apply_to = set(communication.get("apply_to", ["actor", "critic"]) or [])
            self._actor_encoder_type = str(self._actor_config.get("encoder", {}).get("type", "mlp") or "mlp").lower()
            self._critic_encoder_type = str(self._critic_config.get("encoder", {}).get("type", "mlp") or "mlp").lower()
            self.actor_communication = CustomSACCommunicationBlock(self._communication_type)
            self.critic_communication = CustomSACCommunicationBlock(self._communication_type)
            self.shared_dcrnn_backbone = None
            self.actor_dcrnn_backbone = None
            self.actor_dcrnn_head = None
            self.qf_dcrnn_backbone = None
            self.qf_twin_dcrnn_backbone = None
            uses_dcrnn = (
                self._encoder_layout == "shared"
                or self._actor_encoder_type == "dcrnn"
                or self._critic_encoder_type == "dcrnn"
            )
            if uses_dcrnn:
                if not hasattr(self.action_space, "n"):
                    raise ValueError("custom SAC DCRNN encoders currently support discrete action spaces only.")
            if self._encoder_layout == "shared":
                actor_head_config = dict(self._actor_config.get("head", {}) or {})
                actor_head_hidden_dims = _int_list(
                    actor_head_config.get("hidden_dims"),
                    field_name="actor.head.hidden_dims",
                )
                actor_head_activation = str(actor_head_config.get("activation", "relu") or "relu")
                critic_head_config = dict(self._critic_config.get("head", {}) or {})
                critic_head_hidden_dims = _int_list(
                    critic_head_config.get("hidden_dims"),
                    field_name="critic.head.hidden_dims",
                )
                critic_head_activation = str(critic_head_config.get("activation", "relu") or "relu")
                self.shared_dcrnn_backbone = DCRNNBackbone.from_shared_sac_model_config(
                    self.observation_space,
                    self.model_config,
                )
                self.actor_dcrnn_backbone = self.shared_dcrnn_backbone
                self.qf_dcrnn_backbone = self.shared_dcrnn_backbone
                self.actor_dcrnn_head = _build_mlp(
                    self.shared_dcrnn_backbone.output_dim,
                    actor_head_hidden_dims,
                    actor_head_activation,
                    int(self.action_space.n),
                )
                self.qf_encoder = _TorchDCRNNEncoder(self.shared_dcrnn_backbone)
                self.qf = _build_mlp(
                    self.shared_dcrnn_backbone.output_dim,
                    critic_head_hidden_dims,
                    critic_head_activation,
                    int(self.action_space.n),
                )
                if self.twin_q:
                    self.qf_twin_dcrnn_backbone = self.shared_dcrnn_backbone
                    self.qf_twin_encoder = _TorchDCRNNEncoder(self.shared_dcrnn_backbone)
                    self.qf_twin = _build_mlp(
                        self.shared_dcrnn_backbone.output_dim,
                        critic_head_hidden_dims,
                        critic_head_activation,
                        int(self.action_space.n),
                    )
            elif self._actor_encoder_type == "dcrnn":
                head_config = dict(self._actor_config.get("head", {}) or {})
                head_hidden_dims = _int_list(head_config.get("hidden_dims"), field_name="actor.head.hidden_dims")
                head_activation = str(head_config.get("activation", "relu") or "relu")
                self.actor_dcrnn_backbone = DCRNNBackbone.from_actor_model_config(
                    self.observation_space,
                    self.model_config,
                )
                self.actor_dcrnn_head = _build_mlp(
                    self.actor_dcrnn_backbone.output_dim,
                    head_hidden_dims,
                    head_activation,
                    int(self.action_space.n),
                )
            if self._critic_encoder_type == "dcrnn":
                head_config = dict(self._critic_config.get("head", {}) or {})
                head_hidden_dims = _int_list(head_config.get("hidden_dims"), field_name="critic.head.hidden_dims")
                head_activation = str(head_config.get("activation", "relu") or "relu")
                self.qf_dcrnn_backbone = DCRNNBackbone.from_critic_model_config(
                    self.observation_space,
                    self.model_config,
                )
                self.qf_encoder = _TorchDCRNNEncoder(self.qf_dcrnn_backbone)
                self.qf = _build_mlp(
                    self.qf_dcrnn_backbone.output_dim,
                    head_hidden_dims,
                    head_activation,
                    int(self.action_space.n),
                )
                if self.twin_q:
                    self.qf_twin_dcrnn_backbone = DCRNNBackbone.from_critic_model_config(
                        self.observation_space,
                        self.model_config,
                    )
                    self.qf_twin_encoder = _TorchDCRNNEncoder(self.qf_twin_dcrnn_backbone)
                    self.qf_twin = _build_mlp(
                        self.qf_twin_dcrnn_backbone.output_dim,
                        head_hidden_dims,
                        head_activation,
                        int(self.action_space.n),
                    )

        def _apply_actor_communication(self, latent):
            if self._communication_enabled and "actor" in self._communication_apply_to:
                return self.actor_communication(latent)
            return latent

        def _apply_critic_communication(self, latent):
            if self._communication_enabled and "critic" in self._communication_apply_to:
                return self.critic_communication(latent)
            return latent

        def _actor_logits(self, batch):
            if self._encoder_layout == "shared" or self._actor_encoder_type == "dcrnn":
                actor_latent = self.actor_dcrnn_backbone(batch[Columns.OBS])
                actor_latent = self._apply_actor_communication(actor_latent)
                return self.actor_dcrnn_head(actor_latent)
            pi_encoder_outs = self.pi_encoder(batch)
            actor_latent = self._apply_actor_communication(pi_encoder_outs[ENCODER_OUT])
            return self.pi(actor_latent)

        def _forward_inference(self, batch):
            return {Columns.ACTION_DIST_INPUTS: self._actor_logits(batch)}

        def _forward_exploration(self, batch, **kwargs):
            del kwargs
            return self._forward_inference(batch)

        def _forward_train_discrete(self, batch):
            output = {}
            batch_curr = {Columns.OBS: batch[Columns.OBS]}
            batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

            action_logits_next = self._actor_logits(batch_next)
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

            action_logits = self._actor_logits(batch_curr)
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

        def _uses_dcrnn_critic_targets(self) -> bool:
            return self._encoder_layout == "shared" or self._critic_encoder_type == "dcrnn"

        def make_target_networks(self) -> None:
            if not self._uses_dcrnn_critic_targets():
                super().make_target_networks()
                return

            # Keep explicit target encoder ownership for each critic branch so
            # critic encoders follow SAC target-sync behavior rather than
            # gradient updates.
            self.target_qf_encoder = deepcopy(self.qf_encoder)
            self.target_qf = deepcopy(self.qf)
            if self.twin_q:
                self.target_qf_twin_encoder = deepcopy(self.qf_twin_encoder)
                self.target_qf_twin = deepcopy(self.qf_twin)

        def get_target_network_pairs(self):
            if not self._uses_dcrnn_critic_targets():
                return super().get_target_network_pairs()

            pairs = [(self.qf_encoder, self.target_qf_encoder), (self.qf, self.target_qf)]
            if self.twin_q:
                pairs.extend(
                    [
                        (self.qf_twin_encoder, self.target_qf_twin_encoder),
                        (self.qf_twin, self.target_qf_twin),
                    ]
                )
            return pairs

        def get_non_inference_attributes(self):
            try:
                non_inference_attributes = list(super().get_non_inference_attributes())
            except AttributeError:
                non_inference_attributes = []
            for attr in (
                "_architecture_tag",
                "_actor_config",
                "_critic_config",
                "_shared_encoder_config",
                "_encoder_layout",
                "_communication_config",
                "_communication_enabled",
                "_communication_type",
                "_communication_apply_to",
                "_actor_encoder_type",
                "_critic_encoder_type",
                "actor_communication",
                "critic_communication",
                "shared_dcrnn_backbone",
                "actor_dcrnn_backbone",
                "actor_dcrnn_head",
                "qf_dcrnn_backbone",
                "qf_twin_dcrnn_backbone",
            ):
                if attr not in non_inference_attributes:
                    non_inference_attributes.append(attr)
            if self._uses_dcrnn_critic_targets():
                for attr in ("target_qf_encoder", "target_qf", "target_qf_twin_encoder", "target_qf_twin"):
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
