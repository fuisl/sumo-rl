"""RLlib RLModule wrapper for CoLight."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from sumo_rl.agents.colight.model import CoLightQNetwork


def build_colight_dqn_module_class():
    import torch
    from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS
    from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.schedules.scheduler import Scheduler

    class CoLightDQNTorchRLModule(DefaultDQNTorchRLModule):
        """DQN RLModule whose Q-function is the shared CoLight graph network."""

        def setup(self):
            self.uses_dueling = False
            self.uses_double_q = bool(self.model_config.get("double_q", True))
            self.num_atoms = int(self.model_config.get("num_atoms", 1))
            if self.num_atoms != 1:
                raise ValueError("CoLight DQN currently supports expectation Q-learning only; set num_atoms=1.")
            self.epsilon_schedule = Scheduler(
                fixed_value_or_schedule=self.model_config.get("epsilon", 0.0),
                framework=self.framework,
            )
            self.q_net = CoLightQNetwork.from_model_config(self.observation_space, self.action_space, self.model_config)

        def get_non_inference_attributes(self):
            return ["_target_q_net"]

        def get_initial_state(self):
            return {}

        def make_target_networks(self) -> None:
            self._target_q_net = copy.deepcopy(self.q_net)

        def get_target_network_pairs(self):
            return [(self.q_net, self._target_q_net)]

        def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            return {QF_PREDS: self._target_q_net(batch[Columns.OBS])}

        def compute_q_values(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            return {QF_PREDS: self.q_net(batch[Columns.OBS])}

        def _forward_exploration(self, batch: Dict[str, Any], t: int) -> Dict[str, Any]:
            qf_outs = self.compute_q_values(batch)
            action_dist_cls = self.get_exploration_action_dist_cls()
            action_dist = action_dist_cls.from_logits(qf_outs[QF_PREDS])
            exploit_actions = action_dist.to_deterministic().sample()

            self.epsilon_schedule.update(t)
            epsilon = self.epsilon_schedule.get_current_value()
            action_mask = batch[Columns.OBS].get("action_mask")
            if action_mask is None:
                random_actions = torch.randint(
                    low=0,
                    high=qf_outs[QF_PREDS].shape[-1],
                    size=exploit_actions.shape,
                    device=qf_outs[QF_PREDS].device,
                )
            else:
                valid = (action_mask > 0).float()
                empty_rows = valid.sum(dim=-1) <= 0
                if torch.any(empty_rows):
                    valid[empty_rows] = 1.0
                random_actions = torch.multinomial(valid, num_samples=1).squeeze(1)

            actions = torch.where(
                torch.rand(exploit_actions.shape, device=qf_outs[QF_PREDS].device) < epsilon,
                random_actions,
                exploit_actions,
            )
            return {Columns.ACTIONS: actions}

        def compute_advantage_distribution(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            del batch
            raise NotImplementedError("CoLight DQN does not implement distributional advantage outputs.")

    CoLightDQNTorchRLModule.__name__ = "CoLightDQNTorchRLModule"
    return CoLightDQNTorchRLModule


def build_colight_dqn_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    module_class = build_colight_dqn_module_class()
    return RLModuleSpec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {"architecture_tag": "colight_graph_attention"},
    )

