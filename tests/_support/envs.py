from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete


class DummyDiscreteParallelEnv:
    possible_agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return Discrete(3)

    def close(self):
        pass


class DummyHeterogeneousSharedEnv:
    possible_agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def observation_space(self, agent_id):
        if agent_id == "tls_0":
            return Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        return Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

    def action_space(self, agent_id):
        if agent_id == "tls_0":
            return Discrete(4)
        return Discrete(5)

    def close(self):
        pass


class FakeGraphTrafficSignal:
    def __init__(self, ts_id, lanes, out_lanes, density, queue):
        self.id = ts_id
        self.lanes = list(lanes)
        self.out_lanes = list(out_lanes)
        self._density = list(density)
        self._queue = list(queue)

    def get_lanes_density(self):
        return self._density

    def get_lanes_queue(self):
        return self._queue


class DummyGraphParallelEnv:
    possible_agents = ["tls_0", "tls_1"]
    agents = ["tls_0", "tls_1"]

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        signals = [
            FakeGraphTrafficSignal("tls_0", ["in_0"], ["lane_0_1"], [0.25], [0.5]),
            FakeGraphTrafficSignal("tls_1", ["lane_0_1", "in_1"], ["out_1"], [0.75, 0.1], [0.2, 0.3]),
        ]
        self.ts_ids = [signal.id for signal in signals]
        self.traffic_signals = {signal.id: signal for signal in signals}

    def observation_space(self, agent_id):
        del agent_id
        return Box(low=0.0, high=1.0, shape=(5, 4, 4), dtype=np.float32)

    def action_space(self, agent_id):
        del agent_id
        return Discrete(3)

    def close(self):
        pass
