from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
Box = gymnasium.spaces.Box
Discrete = gymnasium.spaces.Discrete


def test_custom_sac_dcrnn_non_inference_attributes_keep_actor_path():
    pytest.importorskip("ray")

    from sumo_rl.agents.sac.custom_sac import build_custom_sac_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)
    action_space = Discrete(3)
    module = build_custom_sac_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "sac_dcrnn_full",
            "agent_index": 1,
            "num_nodes": 4,
            "input_dim": 4,
            "adjacency": np.eye(4, dtype=np.float32).tolist(),
            "actor": {
                "encoder": {
                    "type": "dcrnn",
                    "hidden_dim": 16,
                    "max_diffusion_step": 1,
                },
                "head": {"hidden_dims": [8], "activation": "relu"},
            },
            "critic": {
                "encoder": {
                    "type": "dcrnn",
                    "hidden_dim": 12,
                    "max_diffusion_step": 1,
                },
                "head": {"hidden_dims": [8], "activation": "relu"},
                "twin_q": True,
            },
        },
    ).build()
    module.make_target_networks()

    non_inference_attributes = set(module.get_non_inference_attributes())

    assert "actor_dcrnn_backbone" not in non_inference_attributes
    assert "actor_dcrnn_head" not in non_inference_attributes
    assert "shared_dcrnn_backbone" not in non_inference_attributes
    assert "_encoder_layout" not in non_inference_attributes
    assert "_actor_encoder_type" not in non_inference_attributes
    assert "_communication_enabled" not in non_inference_attributes
    assert "_communication_apply_to" not in non_inference_attributes
    assert "actor_communication" not in non_inference_attributes
    assert "qf_dcrnn_backbone" in non_inference_attributes
    assert "qf_twin_dcrnn_backbone" in non_inference_attributes
    assert "target_qf_encoder" in non_inference_attributes
    assert "target_qf" in non_inference_attributes
