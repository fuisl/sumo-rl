# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gymnasium.spaces import Box, Discrete

import sumo_rl
from sumo_rl.environment.graph_env import GraphParallelEnv
from sumo_rl.models.graph import GraphObservationHistory, build_traffic_signal_graph, pack_density_queue_features


class _FakeTrafficSignal:
    def __init__(
        self,
        ts_id,
        lanes,
        out_lanes,
        density,
        queue,
        num_green_phases=3,
        green_phase=0,
        min_green=5,
        yellow_time=2,
        time_since_last_phase_change=0,
    ):
        self.id = ts_id
        self.lanes = list(lanes)
        self.out_lanes = list(out_lanes)
        self.num_green_phases = num_green_phases
        self.green_phase = green_phase
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.time_since_last_phase_change = time_since_last_phase_change
        self._density = list(density)
        self._queue = list(queue)

    def get_lanes_density(self):
        return self._density

    def get_lanes_queue(self):
        return self._queue


def _fake_signals():
    return [
        _FakeTrafficSignal(
            "tls_0",
            ["in_0"],
            ["lane_0_1"],
            [0.25],
            [0.5],
            num_green_phases=2,
            green_phase=1,
            time_since_last_phase_change=10,
        ),
        _FakeTrafficSignal(
            "tls_1",
            ["lane_0_1", "in_1"],
            ["out_1"],
            [0.75, 0.1],
            [0.2, 0.3],
            num_green_phases=3,
            green_phase=2,
            time_since_last_phase_change=1,
        ),
    ]


def test_graph_topology_construction_adds_virtual_nodes_and_self_loops():
    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)

    assert graph.ts_ids == ("tls_0", "tls_1")
    assert graph.num_nodes == 4
    assert graph.max_lanes == 2
    assert graph.max_green_phases == 3
    assert graph.adjacency[graph.incoming_node_index, graph.ts_index["tls_0"]] == 1.0
    assert graph.adjacency[graph.ts_index["tls_0"], graph.ts_index["tls_1"]] == 1.0
    assert graph.adjacency[graph.ts_index["tls_1"], graph.outgoing_node_index] == 1.0
    assert np.all(np.diag(graph.adjacency) == 1.0)


def test_graph_feature_packing_and_history_repeat_padding():
    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)
    features = pack_density_queue_features(_fake_signals(), graph)

    assert features.shape == (4, 8)
    assert features[graph.ts_index["tls_0"]].tolist() == [0.0, 1.0, 0.0, 1.0, 0.25, 0.0, 0.5, 0.0]
    assert np.allclose(features[graph.ts_index["tls_1"]], [0.0, 0.0, 1.0, 0.0, 0.75, 0.1, 0.2, 0.3])
    assert np.allclose(features[graph.incoming_node_index], np.zeros(8, dtype=np.float32))

    history = GraphObservationHistory(3, graph)
    stacked = history.reset(features)

    assert stacked.shape == (3, 4, 8)
    assert np.allclose(stacked[0], features)
    assert np.allclose(stacked[1], features)
    assert np.allclose(stacked[2], features)


def test_graph_feature_packing_supports_legacy_density_queue_layout():
    graph = build_traffic_signal_graph(
        _fake_signals(),
        include_virtual_nodes=True,
        feature_layout="density_queue",
    )
    features = pack_density_queue_features(_fake_signals(), graph)

    assert graph.feature_layout == "density_queue"
    assert features.shape == (4, 4)
    assert features[graph.ts_index["tls_0"]].tolist() == [0.25, 0.0, 0.5, 0.0]
    assert np.allclose(features[graph.ts_index["tls_1"]], [0.75, 0.1, 0.2, 0.3])


def test_graph_topology_construction_uses_fgsv3_tls_super_edges_when_net_file_is_available(tmp_path):
    net_file = tmp_path / "tiny_tls.net.xml"
    net_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <edge id="in_0" from="src_0" to="tls_0" priority="1">
        <lane id="in_0_0" index="0" speed="10.00" length="30.00" shape="-30.00,0.00 0.00,0.00"/>
    </edge>
    <edge id="mid" from="tls_0" to="tls_1" priority="1">
        <lane id="mid_0" index="0" speed="10.00" length="50.00" shape="0.00,0.00 50.00,0.00"/>
    </edge>
    <edge id="out_1" from="tls_1" to="sink_1" priority="1">
        <lane id="out_1_0" index="0" speed="10.00" length="30.00" shape="50.00,0.00 80.00,0.00"/>
    </edge>
    <connection from="in_0" to="mid" fromLane="0" toLane="0" tl="tls_0" linkIndex="0" dir="s" state="O"/>
    <connection from="mid" to="out_1" fromLane="0" toLane="0" tl="tls_1" linkIndex="0" dir="s" state="O"/>
    <tlLogic id="tls_0" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <tlLogic id="tls_1" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <junction id="src_0" type="dead_end" x="-30.00" y="0.00" incLanes="" intLanes=""/>
    <junction id="tls_0" type="traffic_light" x="0.00" y="0.00" incLanes="in_0_0" intLanes=""/>
    <junction id="tls_1" type="traffic_light" x="50.00" y="0.00" incLanes="mid_0" intLanes=""/>
    <junction id="sink_1" type="dead_end" x="80.00" y="0.00" incLanes="out_1_0" intLanes=""/>
</net>
""",
        encoding="utf-8",
    )

    graph = build_traffic_signal_graph(_fake_signals(), net_file=net_file, include_virtual_nodes=True)

    assert graph.topology_source == "tls_super_edges"
    assert graph.num_nodes == 2
    assert graph.incoming_node_index is None
    assert graph.outgoing_node_index is None
    assert graph.adjacency[graph.ts_index["tls_0"], graph.ts_index["tls_1"]] == 1.0
    assert graph.adjacency[graph.ts_index["tls_1"], graph.ts_index["tls_0"]] == 0.0
    assert np.all(np.diag(graph.adjacency) == 1.0)


def test_graph_parallel_env_uses_base_env_net_file_for_dcrnn_graph(tmp_path):
    net_file = tmp_path / "mismatched_tls.net.xml"
    net_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <edge id="in_a" from="source" to="cluster_a" priority="1">
        <lane id="in_a_0" index="0" speed="10.00" length="30.00" shape="-30.00,0.00 0.00,0.00"/>
    </edge>
    <edge id="ab" from="cluster_a" to="mid" priority="1">
        <lane id="ab_0" index="0" speed="10.00" length="50.00" shape="0.00,0.00 50.00,0.00"/>
    </edge>
    <edge id="bc" from="mid" to="tls_b" priority="1">
        <lane id="bc_0" index="0" speed="10.00" length="50.00" shape="50.00,0.00 100.00,0.00"/>
    </edge>
    <connection from="in_a" to="ab" fromLane="0" toLane="0" tl="program_a" linkIndex="0" dir="s" state="O"/>
    <connection from="ab" to="bc" fromLane="0" toLane="0"/>
    <tlLogic id="program_a" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <tlLogic id="tls_b" type="static" programID="0" offset="0"><phase duration="30" state="G"/></tlLogic>
    <junction id="source" type="dead_end" x="-30.00" y="0.00" incLanes="" intLanes=""/>
    <junction id="cluster_a" type="traffic_light" x="0.00" y="0.00" incLanes="in_a_0" intLanes=""/>
    <junction id="mid" type="priority" x="50.00" y="0.00" incLanes="ab_0" intLanes=""/>
    <junction id="tls_b" type="traffic_light" x="100.00" y="0.00" incLanes="bc_0" intLanes=""/>
</net>
""",
        encoding="utf-8",
    )

    class _NetAwareParallelEnv(_DummyDCRNNParallelEnv):
        possible_agents = ["program_a", "tls_b"]
        agents = ["program_a", "tls_b"]

        def __init__(self):
            signals = [
                _FakeTrafficSignal("program_a", ["in_a"], ["ab"], [0.2], [0.1], num_green_phases=2),
                _FakeTrafficSignal("tls_b", ["bc"], ["out_b"], [0.3], [0.2], num_green_phases=2),
            ]
            self._net = str(net_file)
            self.ts_ids = [signal.id for signal in signals]
            self.traffic_signals = {signal.id: signal for signal in signals}

        def observation_space(self, agent_id):
            del agent_id
            return Box(low=0.0, high=1.0, shape=(5, 2, 4), dtype=np.float32)

        def action_space(self, agent_id):
            del agent_id
            return Discrete(2)

    env = GraphParallelEnv(_NetAwareParallelEnv(), history_len=3)

    assert env.graph.topology_source == "tls_super_edges"
    assert env.graph.ts_ids == ("program_a", "tls_b")
    assert env.graph.adjacency[env.graph.ts_index["program_a"], env.graph.ts_index["tls_b"]] == 1.0
    assert np.all(np.diag(env.graph.adjacency) == 1.0)


def test_dcrnn_q_network_outputs_one_q_value_per_action():
    torch = pytest.importorskip("torch")
    from sumo_rl.models.dcrnn import DCRNNBackbone, DCRNNQNetwork

    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)
    backbone = DCRNNBackbone(
        input_dim=graph.feature_dim,
        adjacency=graph.adjacency,
        num_nodes=graph.num_nodes,
        agent_index=graph.ts_index["tls_1"],
        hidden_dim=16,
        max_diffusion_step=1,
    )
    model = DCRNNQNetwork(
        input_dim=graph.feature_dim,
        adjacency=graph.adjacency,
        num_nodes=graph.num_nodes,
        agent_index=graph.ts_index["tls_1"],
        num_actions=3,
        hidden_dim=16,
        max_diffusion_step=1,
    )

    obs = torch.zeros((2, 5, graph.num_nodes, graph.feature_dim), dtype=torch.float32)
    backbone_latent = backbone(obs)
    q_values = model(obs)

    assert backbone_latent.shape == (2, 24)
    assert q_values.shape == (2, 3)
    assert torch.isfinite(q_values).all()


def test_dcrnn_backbone_pre_encoder_projects_features_before_recurrent_stack():
    torch = pytest.importorskip("torch")
    from sumo_rl.models.dcrnn import DCRNNBackbone

    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)
    backbone = DCRNNBackbone(
        input_dim=graph.feature_dim,
        adjacency=graph.adjacency,
        num_nodes=graph.num_nodes,
        agent_index=graph.ts_index["tls_1"],
        hidden_dim=16,
        max_diffusion_step=1,
        pre_encoder_enabled=True,
        pre_encoder_hidden_dim=12,
        pre_encoder_activation="relu",
    )

    obs = torch.zeros((2, 5, graph.num_nodes, graph.feature_dim), dtype=torch.float32)
    projected = backbone._encode_observations(obs)
    latent = backbone(obs)

    assert projected.shape == (2, 5, graph.num_nodes, 12)
    assert latent.shape == (2, 28)
    assert torch.isfinite(latent).all()


class _DummyDCRNNParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        signals = _fake_signals()
        self.ts_ids = [signal.id for signal in signals]
        self.traffic_signals = {signal.id: signal for signal in signals}

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)

    def action_space(self, agent_id):
        return Discrete(2 if agent_id == "tls_0" else 3)

    def close(self):
        pass


def test_rllib_runner_supports_dcrnn_algorithm_kind():
    pytest.importorskip("ray")
    from sumo_rl.experiments import rllib_runner

    assert "dqn_dcrnn" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS
    assert "dqn_dcrnn_mlp" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS
    assert "dcrnn" in rllib_runner.SUPPORTED_RLLIB_ALGORITHMS


def test_dcrnn_build_config_registers_graph_rl_modules(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.dcrnn import dcrnn

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDCRNNParallelEnv(**kwargs))
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name="dcrnn_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "history_len": 5,
                "model_config": {
                    "hid_dim": 16,
                    "max_diffusion_step": 1,
                    "num_rnn_layers": 1,
                },
            }
        ),
    )

    config = dcrnn.build_config(cfg, tmp_path)
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    assert multi_spec.rl_module_specs["tls_0"].model_config["agent_index"] == 0
    assert multi_spec.rl_module_specs["tls_1"].model_config["agent_index"] == 1
    assert multi_spec.rl_module_specs["tls_0"].model_config["architecture_tag"] == "dqn_dcrnn"
    assert multi_spec.rl_module_specs["tls_0"].model_config["feature_layout"] == "phase_min_green_density_queue"


def test_dcrnn_mlp_build_config_enables_pre_encoder(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from sumo_rl.agents.dcrnn import dcrnn

    monkeypatch.setattr(sumo_rl, "parallel_env", lambda **kwargs: _DummyDCRNNParallelEnv(**kwargs))
    cfg = SimpleNamespace(
        scenario=SimpleNamespace(name="resco_grid4x4"),
        experiment=SimpleNamespace(name="dcrnn_mlp_test", seed=7, episode_seconds=60),
        env=SimpleNamespace(factory="parallel_env", kwargs={}),
        algorithm=SimpleNamespace(
            params={
                "policy_mode": "independent",
                "history_len": 5,
                "model_config": {
                    "hid_dim": 16,
                    "max_diffusion_step": 1,
                    "num_rnn_layers": 1,
                },
            }
        ),
    )

    config = dcrnn.build_config(cfg, tmp_path, algorithm_kind=dcrnn.MLP_KIND)
    multi_spec = config.get_multi_rl_module_spec(env=None, spaces=None, inference_only=False)

    assert set(multi_spec.rl_module_specs.keys()) == {"tls_0", "tls_1"}
    spec = multi_spec.rl_module_specs["tls_0"]
    assert spec.model_config["architecture_tag"] == "dqn_dcrnn_mlp"
    assert spec.model_config["pre_encoder"]["enabled"] is True


def test_dcrnn_module_uses_distinct_target_network_copy():
    pytest.importorskip("ray")
    pytest.importorskip("torch")
    from sumo_rl.agents.dcrnn.rllib_module import build_dcrnn_dqn_module_spec

    obs_space = Box(low=0.0, high=1.0, shape=(5, 4, 8), dtype=np.float32)
    action_space = Discrete(3)
    module = build_dcrnn_dqn_module_spec(
        obs_space,
        action_space,
        model_config={
            "agent_index": 1,
            "num_nodes": 4,
            "input_dim": 8,
            "adjacency": np.eye(4, dtype=np.float32).tolist(),
            "hid_dim": 16,
            "max_diffusion_step": 1,
        },
    ).build()
    module.make_target_networks()

    assert module.q_net is not module._target_q_net
    online_param_ids = {id(param) for param in module.q_net.parameters()}
    target_param_ids = {id(param) for param in module._target_q_net.parameters()}
    assert online_param_ids.isdisjoint(target_param_ids)
