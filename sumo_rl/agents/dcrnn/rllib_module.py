"""RLlib DQN RLModule wrapper for DCRNN Q-networks."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from sumo_rl.models.dcrnn import DCRNNQNetwork


def build_dcrnn_dqn_module_class():
    from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS
    from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.schedules.scheduler import Scheduler

    class DCRNNDQNTorchRLModule(DefaultDQNTorchRLModule):
        """DQN RLModule whose Q-function is a DCRNN graph encoder."""

        def setup(self):
            self.uses_dueling = False
            self.uses_double_q = bool(self.model_config.get("double_q", True))
            self.num_atoms = int(self.model_config.get("num_atoms", 1))
            if self.num_atoms != 1:
                raise ValueError("DCRNN DQN currently supports expectation Q-learning only; set num_atoms=1.")
            self.epsilon_schedule = Scheduler(
                fixed_value_or_schedule=self.model_config.get("epsilon", 0.0),
                framework=self.framework,
            )
            self.q_net = DCRNNQNetwork.from_model_config(self.observation_space, self.action_space, self.model_config)

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

        def compute_advantage_distribution(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            del batch
            raise NotImplementedError("DCRNN DQN does not implement distributional advantage outputs.")

    DCRNNDQNTorchRLModule.__name__ = "DCRNNDQNTorchRLModule"
    return DCRNNDQNTorchRLModule


def build_dcrnn_dqn_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    return RLModuleSpec(
        module_class=build_dcrnn_dqn_module_class(),
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {},
    )

