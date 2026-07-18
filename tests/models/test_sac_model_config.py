from __future__ import annotations

import pytest

pytest.importorskip("ray")

from sumo_rl.agents.sac.custom_sac import normalize_custom_sac_model_config


def test_custom_sac_model_config_accepts_message_passing_placeholder():
    model_config = normalize_custom_sac_model_config(
        {
            "architecture_tag": "sac_mlp_variant",
            "actor": {"encoder": {"hidden_dims": [128]}},
            "communication": {
                "enabled": True,
                "type": "gat",
                "apply_to": ["actor"],
                "scope": "multi_agent",
            },
        }
    )

    assert model_config["architecture_tag"] == "sac_mlp_variant"
    assert model_config["custom_sac"]["communication"]["enabled"] is True
    assert model_config["custom_sac"]["communication"]["type"] == "gat"
    assert model_config["custom_sac"]["communication"]["apply_to"] == ["actor"]
    assert model_config["fcnet_hiddens"] == [128]


def test_custom_sac_model_config_accepts_independent_actor_and_critic_mlp_sizes():
    model_config = normalize_custom_sac_model_config(
        {
            "architecture_tag": "sac_mlp_variant",
            "actor": {
                "encoder": {
                    "hidden_dims": [32],
                    "activation": "relu",
                },
                "head": {"hidden_dims": [16], "activation": "relu"},
            },
            "critic": {
                "encoder": {
                    "hidden_dims": [24],
                    "activation": "tanh",
                },
                "head": {"hidden_dims": [12], "activation": "relu"},
            },
        }
    )

    assert model_config["architecture_tag"] == "sac_mlp_variant"
    assert model_config["custom_sac"]["critic"]["encoder"]["type"] == "mlp"
    assert model_config["fcnet_hiddens"] == [32]
    assert model_config["critic_fcnet_hiddens"] == [24]


def test_custom_sac_model_config_rejects_unsupported_critic_encoder():
    with pytest.raises(ValueError, match="critic.encoder.type='mlp' only"):
        normalize_custom_sac_model_config(
            {
                "critic": {
                    "encoder": {
                        "type": "cnn",
                    }
                }
            }
        )


def test_custom_sac_model_config_rejects_unsupported_actor_encoder():
    with pytest.raises(ValueError, match="actor.encoder.type='mlp' only"):
        normalize_custom_sac_model_config(
            {
                "actor": {
                    "encoder": {
                        "type": "dcrnn",
                    }
                }
            }
        )
