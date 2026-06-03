"""RLlib SAC module for FGS."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import gymnasium as gym

from sumo_rl.agents.fgs.model import CentralGraphJointActionCritic, CentralGraphPolicyCritic, FGSGraphEncoder


FGS_ACTOR_QF_PREDS = "fgs_actor_qf_preds"
FGS_ACTOR_QF_TWIN_PREDS = "fgs_actor_qf_twin_preds"


DEFAULT_FGS_MODEL_CONFIG: Dict[str, Any] = {
    "architecture_tag": "fgs_frap_gnn_sac",
    "twin_q": True,
    "local_encoder": {
        "type": "frap",
        "output_dim": 128,
        "frap": {
            "demand_shape": 2,
            "demand_layout": "split",
            "observation_has_phase": True,
            "observation_has_min_green": True,
            "conv_units": 32,
        },
    },
    "communication": {
        "enabled": True,
        "type": "gat",
        "num_heads": 4,
        "head_dim": 16,
        "output_dim": 128,
    },
    "critic": {
        "type": "central_graph_joint_action",
        "hidden_dims": [256, 256],
        "activation": "relu",
    },
    "topology": {
        "source": "tls_super_edges",
    },
    "invalid_action_value": -1.0e9,
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_fgs_model_config(model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _deep_update(DEFAULT_FGS_MODEL_CONFIG, dict(model_config or {}))
    local_type = str(config["local_encoder"].get("type", "frap") or "frap").lower()
    if local_type not in {"frap", "mlp"}:
        raise ValueError("FGS local_encoder.type must be one of: frap, mlp.")
    config["local_encoder"]["type"] = local_type
    communication = config["communication"]
    communication_type = str(communication.get("type", "gat") or "gat").lower()
    if communication_type not in {"gat", "identity"}:
        raise ValueError("FGS communication.type must be one of: gat, identity.")
    communication["type"] = communication_type
    critic_type = str(config["critic"].get("type", "central_graph_joint_action") or "central_graph_joint_action")
    if critic_type not in {"central_graph_joint_action", "central_graph_policy_context"}:
        raise ValueError("FGS critic.type must be one of: central_graph_joint_action, central_graph_policy_context.")
    config["critic"]["type"] = critic_type
    topology_source = str(config["topology"].get("source", "tls_super_edges") or "tls_super_edges")
    if topology_source not in {"tls_super_edges", "direct_lane"}:
        raise ValueError("FGS topology.source must be one of: tls_super_edges, direct_lane.")
    config["topology"]["source"] = topology_source
    config["twin_q"] = bool(config.get("twin_q", True))
    return config


def build_fgs_sac_module_class():
    import torch
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
    from ray.rllib.models.torch.torch_distributions import TorchCategorical

    class FGSCatalog:
        def __init__(self, observation_space, action_space, model_config_dict):
            self.observation_space = observation_space
            self.action_space = action_space
            self.model_config_dict = model_config_dict

        def get_action_dist_cls(self, framework: str):
            del framework
            return TorchCategorical

    class FGSSACTorchRLModule(DefaultSACTorchRLModule):
        """Discrete SAC module with FRAP local encoding and centralized graph critics."""

        def __init__(self, *args, **kwargs):
            kwargs["catalog_class"] = FGSCatalog
            super().__init__(*args, **kwargs)

        def setup(self):
            if not isinstance(self.action_space, gym.spaces.Discrete):
                raise ValueError("FGS requires a discrete traffic-signal action space.")
            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise ValueError("FGS requires graph Dict observations from FGSGraphParallelEnv.")
            spaces = self.observation_space.spaces
            self.model_config = normalize_fgs_model_config(dict(self.model_config or {}))
            self.twin_q = bool(self.model_config.get("twin_q", True))
            self.num_nodes = int(spaces["node_features"].shape[0])
            self.node_feature_dim = int(spaces["node_features"].shape[-1])
            self.num_actions = int(self.action_space.n)
            self.invalid_action_value = float(self.model_config.get("invalid_action_value", -1.0e9))
            self.critic_type = str(self.model_config.get("critic", {}).get("type", "central_graph_joint_action"))

            self.pi_encoder = FGSGraphEncoder(
                node_feature_dim=self.node_feature_dim,
                num_nodes=self.num_nodes,
                num_actions=self.num_actions,
                model_config=self.model_config,
            )
            self.pi = torch.nn.Linear(self.pi_encoder.output_dim, self.num_actions)
            self.qf_encoder = FGSGraphEncoder(
                node_feature_dim=self.node_feature_dim,
                num_nodes=self.num_nodes,
                num_actions=self.num_actions,
                model_config=self.model_config,
            )
            critic_config = dict(self.model_config.get("critic", {}) or {})
            critic_cls = CentralGraphJointActionCritic if self.critic_type == "central_graph_joint_action" else CentralGraphPolicyCritic
            self.qf = critic_cls(
                graph_dim=self.qf_encoder.output_dim,
                num_nodes=self.num_nodes,
                num_actions=self.num_actions,
                hidden_dims=critic_config.get("hidden_dims", [256, 256]),
                activation=str(critic_config.get("activation", "relu")),
            )
            if self.twin_q:
                self.qf_twin_encoder = FGSGraphEncoder(
                    node_feature_dim=self.node_feature_dim,
                    num_nodes=self.num_nodes,
                    num_actions=self.num_actions,
                    model_config=self.model_config,
                )
                self.qf_twin = critic_cls(
                    graph_dim=self.qf_twin_encoder.output_dim,
                    num_nodes=self.num_nodes,
                    num_actions=self.num_actions,
                    hidden_dims=critic_config.get("hidden_dims", [256, 256]),
                    activation=str(critic_config.get("activation", "relu")),
                )

        def _masked_logits(self, logits, mask):
            if mask is None:
                return logits
            return logits.masked_fill((mask > 0).logical_not(), self.invalid_action_value)

        def _actor_outputs(self, obs: Dict[str, torch.Tensor]):
            encoded = self.pi_encoder(obs)
            graph_h = encoded["graph"]
            all_logits = self.pi(graph_h)
            node_action_mask = obs.get("node_action_mask")
            if node_action_mask is not None:
                all_logits = self._masked_logits(all_logits, node_action_mask.float())
            ego_index = obs["ego_index"].long().reshape(graph_h.shape[0]).clamp(0, self.num_nodes - 1)
            ego_logits = all_logits[torch.arange(graph_h.shape[0], device=graph_h.device), ego_index]
            ego_logits = self._masked_logits(ego_logits, obs.get("action_mask"))
            all_probs = torch.nn.functional.softmax(all_logits, dim=-1)
            ego_probs = torch.nn.functional.softmax(ego_logits, dim=-1)
            return ego_logits, ego_probs, torch.log(ego_probs.clamp_min(1e-12)), all_probs

        def _critic_outputs(self, obs: Dict[str, torch.Tensor], action_context, *, encoder, critic):
            encoded = encoder(obs)
            return self._critic_outputs_from_encoded(obs, action_context, encoded=encoded, critic=critic)

        def _critic_outputs_from_encoded(self, obs: Dict[str, torch.Tensor], action_context, *, encoded, critic):
            q_values = critic(encoded["graph"], action_context.detach(), obs["ego_index"])
            return self._masked_logits(q_values, obs.get("action_mask"))

        def _replay_joint_action_context(self, obs: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor], fallback):
            if self.critic_type != "central_graph_joint_action":
                return fallback
            context = next_obs.get("prev_joint_action")
            if context is None:
                context = obs.get("prev_joint_action")
            return context if context is not None else fallback

        def _forward_inference(self, batch):
            obs = batch[Columns.OBS]
            ego_logits, _, _, _ = self._actor_outputs(obs)
            return {Columns.ACTION_DIST_INPUTS: ego_logits}

        def _forward_exploration(self, batch, **kwargs):
            del kwargs
            return self._forward_inference(batch)

        def _forward_train_discrete(self, batch):
            obs = batch[Columns.OBS]
            next_obs = batch[Columns.NEXT_OBS]
            output = {}

            _, action_probs_next, action_log_probs_next, all_probs_next = self._actor_outputs(next_obs)
            output[ACTION_PROBS_NEXT] = action_probs_next
            output[ACTION_LOG_PROBS_NEXT] = action_log_probs_next
            output[QF_TARGET_NEXT] = self.forward_target({Columns.OBS: next_obs}, all_action_probs=all_probs_next)

            ego_logits, action_probs, action_log_probs, all_probs = self._actor_outputs(obs)
            del ego_logits
            output[ACTION_PROBS] = action_probs
            output[ACTION_LOG_PROBS] = action_log_probs
            replay_context = self._replay_joint_action_context(obs, next_obs, all_probs)
            qf_encoded = self.qf_encoder(obs)
            output[QF_PREDS] = self._critic_outputs_from_encoded(
                obs,
                replay_context,
                encoded=qf_encoded,
                critic=self.qf,
            )
            output[FGS_ACTOR_QF_PREDS] = self._critic_outputs_from_encoded(
                obs,
                all_probs,
                encoded=qf_encoded,
                critic=self.qf,
            )
            if self.twin_q:
                qf_twin_encoded = self.qf_twin_encoder(obs)
                output[QF_TWIN_PREDS] = self._critic_outputs_from_encoded(
                    obs,
                    replay_context,
                    encoded=qf_twin_encoded,
                    critic=self.qf_twin,
                )
                output[FGS_ACTOR_QF_TWIN_PREDS] = self._critic_outputs_from_encoded(
                    obs,
                    all_probs,
                    encoded=qf_twin_encoded,
                    critic=self.qf_twin,
                )
            return output

        def forward_target(self, batch: Dict[str, Any], *, squeeze: bool = False, all_action_probs=None):
            del squeeze
            obs = batch[Columns.OBS]
            if all_action_probs is None:
                _, _, _, all_action_probs = self._actor_outputs(obs)
            q_values = self._critic_outputs(
                obs,
                all_action_probs,
                encoder=self.target_qf_encoder,
                critic=self.target_qf,
            )
            if self.twin_q:
                twin_q_values = self._critic_outputs(
                    obs,
                    all_action_probs,
                    encoder=self.target_qf_twin_encoder,
                    critic=self.target_qf_twin,
                )
                q_values = torch.min(q_values, twin_q_values)
            return q_values

        def make_target_networks(self) -> None:
            self.target_qf_encoder = copy.deepcopy(self.qf_encoder)
            self.target_qf = copy.deepcopy(self.qf)
            if self.twin_q:
                self.target_qf_twin_encoder = copy.deepcopy(self.qf_twin_encoder)
                self.target_qf_twin = copy.deepcopy(self.qf_twin)

        def get_target_network_pairs(self):
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
            attrs = ["target_qf_encoder", "target_qf"]
            if self.twin_q:
                attrs.extend(["qf_twin_encoder", "qf_twin", "target_qf_twin_encoder", "target_qf_twin"])
            return attrs

    FGSSACTorchRLModule.__name__ = "FGSSACTorchRLModule"
    return FGSSACTorchRLModule


def build_fgs_sac_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    return RLModuleSpec(
        module_class=build_fgs_sac_module_class(),
        observation_space=observation_space,
        action_space=action_space,
        model_config=normalize_fgs_model_config(model_config),
    )


def build_fgs_sac_multi_module_spec(
    rl_module_specs: Dict[str, Any],
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

    return MultiRLModuleSpec(
        rl_module_specs=rl_module_specs,
        model_config=normalize_fgs_model_config(model_config),
    )
