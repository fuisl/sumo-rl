# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
Box = gymnasium.spaces.Box
Discrete = gymnasium.spaces.Discrete


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.sac.custom_sac import build_custom_sac_module_spec, normalize_custom_sac_model_config


def test_custom_sac_module_spec_keeps_discrete_action_space():
    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)

    spec = build_custom_sac_module_spec(obs_space, action_space)

    assert spec.observation_space == obs_space
    assert spec.action_space == action_space
    assert spec.model_config["architecture_tag"] == "sac_mlp"
    assert spec.model_config["twin_q"] is True
    assert spec.model_config["custom_sac"]["critic"]["twin_q"] is True


def test_custom_sac_default_architecture_matches_builtin_sac_rlmodule_defaults():
    from ray.rllib.algorithms.sac.sac_catalog import SACCatalog

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)
    builtin_catalog = SACCatalog(obs_space, action_space, {"twin_q": True})
    custom_config = normalize_custom_sac_model_config({})

    assert custom_config["fcnet_hiddens"] == builtin_catalog._model_config_dict["fcnet_hiddens"]
    assert custom_config["fcnet_activation"] == builtin_catalog._model_config_dict["fcnet_activation"]
    assert custom_config["head_fcnet_hiddens"] == builtin_catalog._model_config_dict["head_fcnet_hiddens"]
    assert custom_config["head_fcnet_activation"] == builtin_catalog._model_config_dict["head_fcnet_activation"]
    assert custom_config["critic_fcnet_hiddens"] == builtin_catalog._model_config_dict["fcnet_hiddens"]
    assert custom_config["critic_fcnet_activation"] == builtin_catalog._model_config_dict["fcnet_activation"]
    assert custom_config["critic_head_fcnet_hiddens"] == builtin_catalog._model_config_dict["head_fcnet_hiddens"]
    assert custom_config["critic_head_fcnet_activation"] == builtin_catalog._model_config_dict["head_fcnet_activation"]


def test_custom_sac_module_spec_builds_and_exposes_actor_critic_hooks():
    torch = pytest.importorskip("torch")
    from ray.rllib.core.columns import Columns

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(4)
    spec = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "actor_small",
            "actor": {"encoder": {"hidden_dims": [32], "activation": "relu"}},
            "critic": {"encoder": {"hidden_dims": [64], "activation": "relu"}},
        },
    )

    module = spec.build()
    output = module.forward_inference({Columns.OBS: torch.zeros(2, 4)})

    assert module._architecture_tag == "actor_small"
    assert module._communication_enabled is False
    assert module.catalog.__class__.__name__ == "CustomSACCatalog"
    assert module.catalog.latent_dims == (32,)
    assert module.catalog.qf_latent_dims == [64]
    assert output[Columns.ACTION_DIST_INPUTS].shape == (2, 4)
    assert spec.model_config["fcnet_hiddens"] == [32]
    assert spec.model_config["critic_fcnet_hiddens"] == [64]


def test_custom_sac_forward_train_exposes_actor_twin_critic_outputs():
    torch = pytest.importorskip("torch")
    from ray.rllib.algorithms.sac.sac_learner import (
        ACTION_LOG_PROBS,
        ACTION_PROBS,
        QF_PREDS,
        QF_TARGET_NEXT,
        QF_TWIN_PREDS,
    )
    from ray.rllib.core.columns import Columns

    obs_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Discrete(3)
    module = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "actor": {"encoder": {"hidden_dims": [16]}},
            "critic": {
                "encoder": {"hidden_dims": [32]},
                "head": {"hidden_dims": [8]},
            },
            "communication": {
                "enabled": True,
                "type": "gat",
                "apply_to": ["actor", "critic"],
            },
        },
    ).build()
    module.make_target_networks()

    output = module.forward_train(
        {
            Columns.OBS: torch.zeros(5, 4),
            Columns.NEXT_OBS: torch.ones(5, 4),
        }
    )

    assert output[ACTION_PROBS].shape == (5, 3)
    assert output[ACTION_LOG_PROBS].shape == (5, 3)
    assert output[QF_PREDS].shape == (5, 3)
    assert output[QF_TWIN_PREDS].shape == (5, 3)
    assert output[QF_TARGET_NEXT].shape == (5, 3)
