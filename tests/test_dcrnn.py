from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gymnasium.spaces import Box, Discrete

import sumo_rl
from sumo_rl.models.graph import GraphObservationHistory, build_traffic_signal_graph, pack_density_queue_features


class _FakeTrafficSignal:
    def __init__(self, ts_id, lanes, out_lanes, density, queue, num_green_phases=3):
        self.id = ts_id
        self.lanes = list(lanes)
        self.out_lanes = list(out_lanes)
        self.num_green_phases = num_green_phases
        self._density = list(density)
        self._queue = list(queue)

    def get_lanes_density(self):
        return self._density

    def get_lanes_queue(self):
        return self._queue


def _fake_signals():
    return [
        _FakeTrafficSignal("tls_0", ["in_0"], ["lane_0_1"], [0.25], [0.5], num_green_phases=2),
        _FakeTrafficSignal("tls_1", ["lane_0_1", "in_1"], ["out_1"], [0.75, 0.1], [0.2, 0.3], num_green_phases=3),
    ]


def test_graph_topology_construction_adds_virtual_nodes_and_self_loops():
    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)

    assert graph.ts_ids == ("tls_0", "tls_1")
    assert graph.num_nodes == 4
    assert graph.max_lanes == 2
    assert graph.adjacency[graph.incoming_node_index, graph.ts_index["tls_0"]] == 1.0
    assert graph.adjacency[graph.ts_index["tls_0"], graph.ts_index["tls_1"]] == 1.0
    assert graph.adjacency[graph.ts_index["tls_1"], graph.outgoing_node_index] == 1.0
    assert np.all(np.diag(graph.adjacency) == 1.0)


def test_graph_feature_packing_and_history_repeat_padding():
    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)
    features = pack_density_queue_features(_fake_signals(), graph)

    assert features.shape == (4, 4)
    assert features[graph.ts_index["tls_0"]].tolist() == [0.25, 0.0, 0.5, 0.0]
    assert np.allclose(features[graph.ts_index["tls_1"]], [0.75, 0.1, 0.2, 0.3])
    assert np.allclose(features[graph.incoming_node_index], [0.0, 0.0, 0.0, 0.0])

    history = GraphObservationHistory(3, graph)
    stacked = history.reset(features)

    assert stacked.shape == (3, 4, 4)
    assert np.allclose(stacked[0], features)
    assert np.allclose(stacked[1], features)
    assert np.allclose(stacked[2], features)


def test_dcrnn_q_network_outputs_one_q_value_per_action():
    torch = pytest.importorskip("torch")
    from sumo_rl.models.dcrnn import DCRNNQNetwork

    graph = build_traffic_signal_graph(_fake_signals(), include_virtual_nodes=True)
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
    q_values = model(obs)

    assert q_values.shape == (2, 3)
    assert torch.isfinite(q_values).all()


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
    assert multi_spec.rl_module_specs["tls_0"].model_config["architecture_tag"] == "dcrnn_dqn"
