"""FGSv2 SAC learner with separate replay and actor joint-action contexts."""

from __future__ import annotations

from typing import Any, Dict

from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    LOGPS_KEY,
    QF_LOSS_KEY,
    QF_MAX_KEY,
    QF_MEAN_KEY,
    QF_MIN_KEY,
    QF_PREDS,
    QF_TARGET_NEXT,
    QF_TWIN_LOSS_KEY,
    QF_TWIN_PREDS,
    TD_ERROR_MEAN_KEY,
)
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY
from ray.rllib.utils.metrics import TD_ERROR_KEY
from ray.rllib.utils.typing import ModuleID, TensorType

from sumo_rl.agents.fgsv2.rllib_module import FGSV2_ACTOR_QF_PREDS, FGSV2_ACTOR_QF_TWIN_PREDS


class FGSv2SACTorchLearner(SACTorchLearner):
    """Discrete SAC loss for FGSv2 action-token actor and centralized critic."""

    def _compute_loss_for_module_discrete(
        self,
        *,
        module_id: ModuleID,
        config,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        torch = __import__("torch")
        alpha = torch.exp(self.curr_log_alpha[module_id])

        action_probs_next = fwd_out[ACTION_PROBS_NEXT]
        action_log_probs_next = fwd_out[ACTION_LOG_PROBS_NEXT]
        next_q = fwd_out[QF_TARGET_NEXT]
        next_v = (action_probs_next * (next_q - alpha.detach() * action_log_probs_next)).sum(-1).squeeze(-1)
        next_v_masked = (1.0 - batch[Columns.TERMINATEDS].float()) * next_v
        target_q = (batch[Columns.REWARDS] + (config.gamma ** batch["n_step"]) * next_v_masked).detach()

        actions = batch[Columns.ACTIONS].to(dtype=torch.int64).unsqueeze(-1)
        qf_pred = fwd_out[QF_PREDS].gather(dim=-1, index=actions).squeeze(-1)
        if config.twin_q:
            qf_twin_pred = fwd_out[QF_TWIN_PREDS].gather(dim=-1, index=actions).squeeze(-1)

        td_error = torch.abs(qf_pred - target_q)
        if config.twin_q:
            td_error = 0.5 * (td_error + torch.abs(qf_twin_pred - target_q))

        critic_loss = torch.mean(
            batch["weights"] * torch.nn.HuberLoss(reduction="none", delta=1.0)(qf_pred, target_q)
        )
        if config.twin_q:
            critic_twin_loss = torch.mean(
                batch["weights"] * torch.nn.HuberLoss(reduction="none", delta=1.0)(qf_twin_pred, target_q)
            )

        action_probs = fwd_out[ACTION_PROBS]
        action_log_probs = fwd_out[ACTION_LOG_PROBS]
        actor_q = fwd_out.get(FGSV2_ACTOR_QF_PREDS, fwd_out[QF_PREDS])
        if config.twin_q:
            actor_q_twin = fwd_out.get(FGSV2_ACTOR_QF_TWIN_PREDS, fwd_out[QF_TWIN_PREDS])
            actor_q = torch.min(actor_q, actor_q_twin)
        actor_q = actor_q.detach()
        policy_loss = (action_probs * (alpha.detach() * action_log_probs - actor_q)).sum(-1).mean()

        entropy = (action_log_probs * action_probs).sum(-1)
        alpha_loss = -torch.mean(self.curr_log_alpha[module_id] * (entropy.detach() + self.target_entropy[module_id]))

        total_loss = policy_loss + critic_loss + alpha_loss
        if config.twin_q:
            total_loss += critic_twin_loss

        self.metrics.log_value(key=(module_id, TD_ERROR_KEY), value=td_error, reduce="item_series")
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: policy_loss,
                QF_LOSS_KEY: critic_loss,
                "alpha_loss": alpha_loss,
                "alpha_value": alpha[0],
                "log_alpha_value": torch.log(alpha)[0],
                "target_entropy": self.target_entropy[module_id],
                LOGPS_KEY: torch.mean(fwd_out[ACTION_LOG_PROBS]),
                QF_MEAN_KEY: torch.mean(fwd_out[QF_PREDS]),
                QF_MAX_KEY: torch.max(fwd_out[QF_PREDS]),
                QF_MIN_KEY: torch.min(fwd_out[QF_PREDS]),
                "fgsv2_actor_q_mean": torch.mean(actor_q),
                TD_ERROR_MEAN_KEY: torch.mean(td_error),
            },
            key=module_id,
            window=1,
        )

        self._temp_losses[(module_id, POLICY_LOSS_KEY)] = policy_loss
        self._temp_losses[(module_id, QF_LOSS_KEY)] = critic_loss
        self._temp_losses[(module_id, "alpha_loss")] = alpha_loss
        if config.twin_q:
            self.metrics.log_value(key=(module_id, QF_TWIN_LOSS_KEY), value=critic_twin_loss, window=1)
            self._temp_losses[(module_id, QF_TWIN_LOSS_KEY)] = critic_twin_loss
        return total_loss
