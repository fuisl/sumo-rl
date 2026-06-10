"""RLlib RLModule wrapper for FRAP."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from sumo_rl.agents.frap.model import FRAPQNetwork


def build_frap_dqn_module_class():
    from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS
    from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.schedules.scheduler import Scheduler

    class FRAPDQNTorchRLModule(DefaultDQNTorchRLModule):
        """DQN RLModule whose Q-function is the FRAP phase-competition network."""

        def setup(self):
            self.uses_dueling = False
            self.uses_double_q = bool(self.model_config.get("double_q", True))
            self.num_atoms = int(self.model_config.get("num_atoms", 1))
            if self.num_atoms != 1:
                raise ValueError("FRAP DQN currently supports expectation Q-learning only; set num_atoms=1.")
            self.epsilon_schedule = Scheduler(
                fixed_value_or_schedule=self.model_config.get("epsilon", 0.0),
                framework=self.framework,
            )
            self.q_net = FRAPQNetwork.from_model_config(self.observation_space, self.action_space, self.model_config)

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
            raise NotImplementedError("FRAP DQN does not implement distributional advantage outputs.")

    FRAPDQNTorchRLModule.__name__ = "FRAPDQNTorchRLModule"
    return FRAPDQNTorchRLModule


def build_frap_dqn_module_spec(
    observation_space,
    action_space,
    *,
    model_config: Optional[Dict[str, Any]] = None,
):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    module_class = build_frap_dqn_module_class()
    return RLModuleSpec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config or {"architecture_tag": "frap_phase_competition"},
    )
