from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict as DictSpace, Discrete


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

import sumo_rl
from sumo_rl.agents.fgs import fgs
from sumo_rl.agents.fgs.graph_env import FGSGraphParallelEnv
from sumo_rl.agents.fgs.model import FGSGraphEncoder, FRAPEmbeddingEncoder
from sumo_rl.agents.fgs.rllib_module import build_fgs_sac_module_spec
from sumo_rl.agents.fgs.topology import extract_tls_topology, render_fgs_topology
from sumo_rl.experiments import rllib_runner


def _graph_obs_space(num_nodes=2, node_dim=13, max_edges=2, num_actions=4):
    return DictSpace(
        {
            "node_features": Box(-np.inf, np.inf, shape=(num_nodes, node_dim), dtype=np.float32),
            "edge_index": Box(0, max(0, num_nodes - 1), shape=(2, max_edges), dtype=np.int64),
            "edge_mask": Box(0.0, 1.0, shape=(max_edges,), dtype=np.float32),
            "edge_weight": Box(0.0, np.inf, shape=(max_edges,), dtype=np.float32),
            "ego_index": Box(0, max(0, num_nodes - 1), shape=(), dtype=np.int64),
            "action_mask": Box(0.0, 1.0, shape=(num_actions,), dtype=np.float32),
            "node_action_mask": Box(0.0, 1.0, shape=(num_nodes, num_actions), dtype=np.float32),
        }
    )


def _graph_obs(batch_size=3, num_nodes=2, node_dim=13, max_edges=2, num_actions=4):
    obs = torch.zeros((batch_size, num_nodes, node_dim), dtype=torch.float32)
    obs[:, :, 0] = 1.0
    obs[:, :, 5:] = 0.25
    return {
        "node_features": obs,
        "edge_index": torch.tensor([[[0, 1], [1, 0]]] * batch_size, dtype=torch.long),
        "edge_mask": torch.ones((batch_size, max_edges), dtype=torch.float32),
        "edge_weight": torch.ones((batch_size, max_edges), dtype=torch.float32),
        "ego_index": torch.tensor([0, 1, 0], dtype=torch.long)[:batch_size],
        "action_mask": torch.ones((batch_size, num_actions), dtype=torch.float32),
        "node_action_mask": torch.ones((batch_size, num_nodes, num_actions), dtype=torch.float32),
    }


def test_fgs_frap_encoder_returns_finite_embeddings():
    obs = torch.zeros((6, 4 + 1 + 2 * 4), dtype=torch.float32)
    obs[:, 0] = 1.0
    obs[:, 5:] = 0.25
    encoder = FRAPEmbeddingEncoder(observation_dim=13, num_actions=4, output_dim=16)

    embeddings = encoder(obs)

    assert embeddings.shape == (6, 16)
    assert torch.isfinite(embeddings).all()


def test_fgs_graph_encoder_returns_graph_and_ego_embeddings():
    encoder = FGSGraphEncoder(
        node_feature_dim=13,
        num_nodes=2,
        num_actions=4,
        model_config={
            "local_encoder": {"type": "frap", "output_dim": 16, "frap": {"demand_shape": 2}},
            "communication": {"enabled": True, "type": "gat", "num_heads": 1, "head_dim": 4, "output_dim": 16},
        },
    )

    output = encoder(_graph_obs(batch_size=2))

    assert output["graph"].shape == (2, 2, 16)
    assert output["ego"].shape == (2, 16)
    assert torch.isfinite(output["graph"]).all()


def test_fgs_tls_super_edge_topology_connects_nearest_downstream_tls(tmp_path):
    net_file = tmp_path / "tiny.net.xml"
    net_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <edge id="ab" from="tls_0" to="mid" priority="1">
        <lane id="ab_0" index="0" speed="10.00" length="50.00" shape="0.00,0.00 50.00,0.00"/>
    </edge>
    <edge id="bc" from="mid" to="tls_1" priority="1">
        <lane id="bc_0" index="0" speed="10.00" length="50.00" shape="50.00,0.00 100.00,0.00"/>
    </edge>
    <connection from="ab" to="bc" fromLane="0" toLane="0"/>
    <tlLogic id="tls_0" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <tlLogic id="tls_1" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <junction id="tls_0" type="traffic_light" x="0.00" y="0.00" incLanes="" intLanes=""/>
    <junction id="mid" type="priority" x="50.00" y="0.00" incLanes="ab_0" intLanes=""/>
    <junction id="tls_1" type="traffic_light" x="100.00" y="0.00" incLanes="bc_0" intLanes=""/>
</net>
""",
        encoding="utf-8",
    )

    topology = extract_tls_topology(net_file)
    paths = render_fgs_topology(topology, tmp_path / "topology")

    assert topology.directed_edges == [("tls_0", "tls_1")]
    assert topology.super_edges[0].path_edge_ids == ["ab", "bc"]
    assert paths["json"].exists()
    assert paths["svg"].exists()


class _DummySignal:
    def __init__(self, lanes, out_lanes):
        self.lanes = lanes
        self.out_lanes = out_lanes


class _DummyBaseEnv:
    ts_ids = ["tls_0", "tls_1"]

    def __init__(self):
        self.traffic_signals = {
            "tls_0": _DummySignal(["a"], ["b"]),
            "tls_1": _DummySignal(["b"], ["c"]),
        }


class _DummyParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self, heterogeneous=False):
        self.env = _DummyBaseEnv()
        self.heterogeneous = heterogeneous

    def observation_space(self, agent_id):
        if self.heterogeneous and agent_id == "tls_1":
            return Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        return Box(0.0, 1.0, shape=(13,), dtype=np.float32)

    def action_space(self, agent_id):
        if self.heterogeneous and agent_id == "tls_1":
            return Discrete(2)
        return Discrete(4)

    def reset(self, seed=None, options=None):
        del seed, options
        obs = np.zeros(13, dtype=np.float32)
        obs[0] = 1.0
        if self.heterogeneous:
            small_obs = np.array([0.0, 1.0, 1.0, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            return {"tls_0": obs, "tls_1": small_obs}, {"tls_0": {}, "tls_1": {}}
        return {"tls_0": obs, "tls_1": obs + 0.25}, {"tls_0": {}, "tls_1": {}}

    def step(self, actions):
        self.last_actions = actions
        obs = np.zeros(13, dtype=np.float32)
        obs[0] = 1.0
        if self.heterogeneous:
            local_obs = {
                "tls_0": obs,
                "tls_1": np.array([1.0, 0.0, 1.0, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
            }
        else:
            local_obs = {"tls_0": obs, "tls_1": obs + 0.5}
        rewards = {"tls_0": 1.0, "tls_1": 2.0}
        terminations = {"tls_0": False, "tls_1": False, "__all__": False}
        truncations = {"tls_0": False, "tls_1": False, "__all__": False}
        infos = {"tls_0": {}, "tls_1": {}}
        return local_obs, rewards, terminations, truncations, infos

    def close(self):
        pass


def test_fgs_graph_wrapper_builds_stable_graph_observations():
    env = FGSGraphParallelEnv(_DummyParallelEnv(), topology_source="direct_lane")

    obs, _ = env.reset(seed=7)

    assert obs["tls_0"]["node_features"].shape == (2, 13)
    assert obs["tls_0"]["edge_mask"].tolist() == [1.0, 1.0]
    assert obs["tls_0"]["action_mask"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert obs["tls_1"]["ego_index"].item() == 1


def test_fgs_graph_wrapper_canonicalizes_heterogeneous_default_observations():
    env = FGSGraphParallelEnv(_DummyParallelEnv(heterogeneous=True), topology_source="direct_lane")

    obs, _ = env.reset(seed=7)

    assert obs["tls_0"]["node_features"].shape == (2, 13)
    assert obs["tls_1"]["action_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert obs["tls_1"]["node_action_mask"].tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    assert obs["tls_1"]["node_features"][1].tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0]
    )

    env.step({"tls_0": 3, "tls_1": 3})

    assert env.env.last_actions["tls_0"] == 3
    assert env.env.last_actions["tls_1"] == 1


def test_fgs_module_inference_and_train_outputs_discrete_sac_tensors():
    from ray.rllib.algorithms.sac.sac_learner import ACTION_PROBS, QF_PREDS, QF_TARGET_NEXT, QF_TWIN_PREDS
    from ray.rllib.core.columns import Columns

    module = build_fgs_sac_module_spec(
        _graph_obs_space(),
        Discrete(4),
        model_config={
            "local_encoder": {"type": "frap", "output_dim": 16, "frap": {"demand_shape": 2}},
            "communication": {"enabled": True, "type": "gat", "num_heads": 1, "head_dim": 4, "output_dim": 16},
            "critic": {"hidden_dims": [32]},
        },
    ).build()
    module.make_target_networks()

    obs = _graph_obs(batch_size=2)
    inference = module.forward_inference({Columns.OBS: obs})
    train_out = module.forward_train({Columns.OBS: obs, Columns.NEXT_OBS: obs})

    assert inference[Columns.ACTION_DIST_INPUTS].shape == (2, 4)
    assert train_out[ACTION_PROBS].shape == (2, 4)
    assert train_out[QF_PREDS].shape == (2, 4)
    assert train_out[QF_TWIN_PREDS].shape == (2, 4)
    assert train_out[QF_TARGET_NEXT].shape == (2, 4)


def test_rllib_runner_supports_fgs_algorithm_kind():
    assert "fgs" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_fgs_build_config_registers_shared_custom_rl_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyParallelEnv())
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="fgs_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "shared",
                "num_env_runners": 0,
                "num_envs_per_env_runner": 1,
                "model_config": {
                    "local_encoder": {"type": "frap", "output_dim": 16},
                    "communication": {"enabled": False, "type": "identity"},
                    "critic": {"hidden_dims": [32]},
                    "topology": {"source": "direct_lane", "render": False},
                },
            }
        ),
    )

    config = fgs.build_config(cfg, tmp_path)

    assert set(config.rl_module_spec.rl_module_specs.keys()) == {"shared_policy"}
    spec = config.rl_module_spec.rl_module_specs["shared_policy"]
    assert spec.module_class.__name__ == "FGSSACTorchRLModule"
    assert spec.model_config["architecture_tag"] == "fgs_frap_gnn_sac"


def test_fgs_rejects_independent_policy_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyParallelEnv())
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="fgs_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "model_config": {"topology": {"source": "direct_lane", "render": False}},
            }
        ),
    )

    with pytest.raises(ValueError, match="policy_mode=shared"):
        fgs.build_config(cfg, tmp_path)
