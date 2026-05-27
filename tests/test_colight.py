from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from gymnasium.spaces import Box, Dict as DictSpace, Discrete

import sumo_rl
from sumo_rl.agents.colight import colight
from sumo_rl.agents.colight.graph_env import CoLightGraphParallelEnv, make_colight_observation_class
from sumo_rl.agents.colight.model import CoLightGATLayer, CoLightQNetwork
from sumo_rl.agents.colight.rllib_module import build_colight_dqn_module_spec
from sumo_rl.agents.colight.topology import render_colight_topology
from sumo_rl.agents.graph_attention import CoLightPyGAttentionLayer
from sumo_rl.experiments import rllib_runner


def _graph_obs_space(num_nodes=2, node_dim=4, max_edges=2, max_actions=4):
    return DictSpace(
        {
            "node_features": Box(-np.inf, np.inf, shape=(num_nodes, node_dim), dtype=np.float32),
            "edge_index": Box(0, max(0, num_nodes - 1), shape=(2, max_edges), dtype=np.int64),
            "edge_mask": Box(0.0, 1.0, shape=(max_edges,), dtype=np.float32),
            "ego_index": Box(0, max(0, num_nodes - 1), shape=(), dtype=np.int64),
            "action_mask": Box(0.0, 1.0, shape=(max_actions,), dtype=np.float32),
        }
    )


def _graph_obs(batch_size=3, num_nodes=2, node_dim=4, max_edges=2, max_actions=4):
    return {
        "node_features": torch.ones((batch_size, num_nodes, node_dim), dtype=torch.float32),
        "edge_index": torch.tensor([[[0, 1], [1, 0]]] * batch_size, dtype=torch.long),
        "edge_mask": torch.ones((batch_size, max_edges), dtype=torch.float32),
        "ego_index": torch.tensor([0, 1, 0], dtype=torch.long)[:batch_size],
        "action_mask": torch.ones((batch_size, max_actions), dtype=torch.float32),
    }


def test_colight_q_network_outputs_one_q_value_per_action():
    model = CoLightQNetwork(
        node_feature_dim=4,
        num_nodes=2,
        num_actions=4,
        node_embedding_dims=[8],
        num_gat_layers=1,
        num_heads=2,
        head_dim=4,
        gat_output_dim=8,
    )

    q_values = model(_graph_obs())

    assert q_values.shape == (3, 4)
    assert torch.isfinite(q_values).all()


def test_colight_gat_layer_supports_self_loops_without_edges():
    layer = CoLightGATLayer(input_dim=4, head_dim=2, output_dim=6, num_heads=2)
    x = torch.ones((3, 4), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    output = layer(x, edge_index)

    assert isinstance(layer, CoLightPyGAttentionLayer)
    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()


def test_colight_batched_edge_flattening_keeps_samples_disconnected():
    model = CoLightQNetwork(
        node_feature_dim=4,
        num_nodes=2,
        num_actions=4,
        node_embedding_dims=[8],
        num_heads=1,
        head_dim=4,
        gat_output_dim=8,
    )
    obs = _graph_obs(batch_size=2)

    edge_index = model._flatten_edges(obs)

    assert edge_index.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]


def test_repo_gat_code_does_not_import_torch_scatter_directly():
    agent_dir = ROOT / "sumo_rl" / "agents"
    offenders = []
    for path in agent_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "torch_scatter" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_colight_action_mask_makes_invalid_q_values_very_negative():
    model = CoLightQNetwork(
        node_feature_dim=4,
        num_nodes=2,
        num_actions=4,
        node_embedding_dims=[8],
        num_heads=1,
        head_dim=4,
        gat_output_dim=8,
    )
    obs = _graph_obs(batch_size=1)
    obs["action_mask"] = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    q_values = model(obs)

    assert q_values[0, 1].item() < -1e8
    assert q_values[0, 3].item() < -1e8


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

    def save_csv(self, out_csv_name, episode):
        del out_csv_name, episode


class _DummyLaneDomain:
    def getLastStepVehicleIDs(self, lane_id):
        assert lane_id == "lane_0"
        return ["veh_0", "veh_1", "veh_2", "ghost_0"]

    def getLastStepLength(self, lane_id):
        assert lane_id == "lane_0"
        return 5.0


def test_colight_observation_normalizes_lane_counts_by_capacity():
    ts = SimpleNamespace(
        green_phase=1,
        num_green_phases=2,
        lanes=["lane_0"],
        lanes_length={"lane_0": 75.0},
        MIN_GAP=2.5,
        sumo=SimpleNamespace(lane=_DummyLaneDomain()),
    )
    observation_fn = make_colight_observation_class(vehicle_max="capacity", clip_vehicle_counts=True)(ts)

    obs = observation_fn()

    assert obs.tolist() == pytest.approx([0.0, 1.0, 0.3])
    assert observation_fn.observation_space().high.tolist() == [1.0, 1.0, 1.0]


class _DummyParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self):
        self.env = _DummyBaseEnv()

    def observation_space(self, agent_id):
        return Box(0.0, np.inf, shape=(3 if agent_id == "tls_0" else 2,), dtype=np.float32)

    def action_space(self, agent_id):
        return Discrete(4 if agent_id == "tls_0" else 2)

    def reset(self, seed=None, options=None):
        del seed, options
        return {
            "tls_0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "tls_1": np.array([4.0, 5.0], dtype=np.float32),
        }, {"tls_0": {}, "tls_1": {}}

    def step(self, actions):
        self.last_actions = actions
        obs = {
            "tls_0": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "tls_1": np.array([5.0, 6.0], dtype=np.float32),
        }
        rewards = {"tls_0": 1.0, "tls_1": 2.0}
        terminations = {"tls_0": False, "tls_1": False, "__all__": False}
        truncations = {"tls_0": False, "tls_1": False, "__all__": False}
        infos = {"tls_0": {}, "tls_1": {}}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        pass


def test_colight_graph_wrapper_builds_stable_graph_observations_and_masks_actions():
    base_env = _DummyParallelEnv()
    env = CoLightGraphParallelEnv(base_env)

    obs, _ = env.reset(seed=7)

    assert set(obs.keys()) == {"tls_0", "tls_1"}
    assert obs["tls_0"]["node_features"].shape == (2, 3)
    assert obs["tls_0"]["edge_mask"].tolist() == [1.0, 1.0]
    assert obs["tls_1"]["action_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]

    _, _, _, _, infos = env.step({"tls_0": 3, "tls_1": 3})

    assert base_env.last_actions["tls_0"] == 3
    assert base_env.last_actions["tls_1"] == 1
    assert infos["tls_1"]["colight/action_clipped"] == 1.0
    assert infos["tls_1"]["colight/raw_action"] == 3.0


def test_colight_topology_renderer_writes_svg_and_edge_list(tmp_path):
    net_file = tmp_path / "tiny.net.xml"
    net_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,100.00,0.00"
        origBoundary="0.00,0.00,100.00,0.00" projParameter="!"/>
    <edge id="ab" from="tls_0" to="tls_1" priority="1">
        <lane id="ab_0" index="0" speed="13.89" length="100.00" shape="0.00,0.00 100.00,0.00"/>
    </edge>
    <junction id="tls_0" type="traffic_light" x="0.00" y="0.00" incLanes="" intLanes="" shape="-5,-5 -5,5 5,5 5,-5"/>
    <junction id="tls_1" type="traffic_light" x="100.00" y="0.00" incLanes="ab_0" intLanes="" shape="95,-5 95,5 105,5 105,-5"/>
</net>
""",
        encoding="utf-8",
    )
    env = CoLightGraphParallelEnv(_DummyParallelEnv())

    paths = render_colight_topology(env, str(net_file), tmp_path / "topology")

    svg = paths["svg"].read_text(encoding="utf-8")
    edge_json = paths["json"].read_text(encoding="utf-8")
    assert "marker-end" in svg
    assert "tls_0" in svg
    assert '"index": 0' in edge_json
    assert '"source": "tls_0"' in edge_json


def test_colight_module_spec_uses_dict_observation_space():
    obs_space = _graph_obs_space()
    action_space = Discrete(4)

    spec = build_colight_dqn_module_spec(
        obs_space,
        action_space,
        model_config={
            "architecture_tag": "colight_graph_attention",
            "epsilon": 0.0,
            "double_q": True,
            "num_atoms": 1,
        },
    )

    assert spec.observation_space == obs_space
    assert spec.action_space == action_space
    assert spec.model_config["architecture_tag"] == "colight_graph_attention"


def test_rllib_runner_supports_colight_algorithm_kind():
    assert "colight" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_colight_build_config_registers_shared_custom_rl_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyParallelEnv())
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="colight_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "shared",
                "model_config": {
                    "node_embedding_dims": [8],
                    "num_heads": 1,
                    "head_dim": 4,
                    "gat_output_dim": 8,
                },
            }
        ),
    )

    config = colight.build_config(cfg, tmp_path)
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"shared_policy"}
    spec = multi_spec.rl_module_specs["shared_policy"]
    assert spec.model_config["architecture_tag"] == "colight_graph_attention"
    assert spec.model_config["epsilon"] == [(0, 0.8), (200000, 0.01)]
    assert config.replay_buffer_config["type"] == "MultiAgentEpisodeReplayBuffer"


def test_colight_rejects_independent_policy_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyParallelEnv())
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="single_intersection"),
        experiment=SimpleNamespace(name="colight_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(params={"policy_mode": "independent"}),
    )

    with pytest.raises(ValueError, match="policy_mode=shared"):
        colight.build_config(cfg, tmp_path)
