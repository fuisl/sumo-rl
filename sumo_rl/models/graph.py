"""Graph topology and feature helpers for traffic-signal networks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Mapping, Sequence

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class TrafficSignalGraph:
    """Static graph metadata derived from SUMO traffic-signal lane links."""

    ts_ids: tuple[str, ...]
    ts_index: dict[str, int]
    num_nodes: int
    max_lanes: int
    adjacency: np.ndarray
    edge_index: np.ndarray
    incoming_node_index: int | None = None
    outgoing_node_index: int | None = None

    @property
    def feature_dim(self) -> int:
        return 2 * self.max_lanes

    def model_config(self, agent_id: str, **extra: Any) -> dict[str, Any]:
        config = {
            "agent_id": str(agent_id),
            "agent_index": int(self.ts_index[str(agent_id)]),
            "num_nodes": int(self.num_nodes),
            "input_dim": int(self.feature_dim),
            "adjacency": self.adjacency.astype(np.float32).tolist(),
            "ts_ids": list(self.ts_ids),
        }
        config.update(extra)
        return config


def _ordered_traffic_signals(traffic_signals: Mapping[str, Any] | Sequence[Any]) -> list[Any]:
    if isinstance(traffic_signals, Mapping):
        return [traffic_signals[key] for key in sorted(traffic_signals)]
    return list(traffic_signals)


def _signal_id(ts: Any) -> str:
    return str(getattr(ts, "id"))


def build_traffic_signal_graph(
    traffic_signals: Mapping[str, Any] | Sequence[Any],
    *,
    include_virtual_nodes: bool = True,
    add_self_loops: bool = True,
) -> TrafficSignalGraph:
    """Build a deterministic directed graph from traffic signal in/out lanes."""

    ts_list = _ordered_traffic_signals(traffic_signals)
    if not ts_list:
        raise ValueError("Cannot build a traffic-signal graph without traffic signals.")

    ts_ids = tuple(_signal_id(ts) for ts in ts_list)
    ts_index = {ts_id: index for index, ts_id in enumerate(ts_ids)}
    max_lanes = max(1, max(len(getattr(ts, "lanes", []) or []) for ts in ts_list))

    lanes = []
    for ts in ts_list:
        lanes.extend(getattr(ts, "lanes", []) or [])
        lanes.extend(getattr(ts, "out_lanes", []) or [])
    lane_index = {lane_id: index for index, lane_id in enumerate(sorted(set(lanes)))}
    lane_edges = [[-1, -1] for _ in lane_index]

    for ts in ts_list:
        index = ts_index[_signal_id(ts)]
        for lane_id in getattr(ts, "lanes", []) or []:
            lane_edges[lane_index[lane_id]][1] = index
        for lane_id in getattr(ts, "out_lanes", []) or []:
            lane_edges[lane_index[lane_id]][0] = index

    incoming_node_index = len(ts_ids) if include_virtual_nodes else None
    outgoing_node_index = len(ts_ids) + 1 if include_virtual_nodes else None
    num_nodes = len(ts_ids) + (2 if include_virtual_nodes else 0)
    edges: list[tuple[int, int]] = []

    for source, target in lane_edges:
        if source == -1 and target == -1:
            continue
        if source == -1:
            if incoming_node_index is not None:
                edges.append((incoming_node_index, target))
            continue
        if target == -1:
            if outgoing_node_index is not None:
                edges.append((source, outgoing_node_index))
            continue
        edges.append((source, target))

    if add_self_loops:
        edges.extend((index, index) for index in range(num_nodes))

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in edges:
        if source >= 0 and target >= 0:
            adjacency[source, target] = 1.0

    edge_index = np.asarray(np.nonzero(adjacency), dtype=np.int64)
    return TrafficSignalGraph(
        ts_ids=ts_ids,
        ts_index=ts_index,
        num_nodes=num_nodes,
        max_lanes=max_lanes,
        adjacency=adjacency,
        edge_index=edge_index,
        incoming_node_index=incoming_node_index,
        outgoing_node_index=outgoing_node_index,
    )


def pack_density_queue_features(
    traffic_signals: Mapping[str, Any] | Sequence[Any],
    graph: TrafficSignalGraph,
) -> np.ndarray:
    """Pack current density and queue features into a graph node matrix."""

    ts_by_id = {_signal_id(ts): ts for ts in _ordered_traffic_signals(traffic_signals)}
    features = np.zeros((graph.num_nodes, graph.feature_dim), dtype=np.float32)
    for ts_id in graph.ts_ids:
        ts = ts_by_id[ts_id]
        density = np.asarray(ts.get_lanes_density(), dtype=np.float32).reshape(-1)
        queue = np.asarray(ts.get_lanes_queue(), dtype=np.float32).reshape(-1)
        node_index = graph.ts_index[ts_id]
        density_width = min(graph.max_lanes, density.size)
        queue_width = min(graph.max_lanes, queue.size)
        features[node_index, :density_width] = density[:density_width]
        features[node_index, graph.max_lanes : graph.max_lanes + queue_width] = queue[:queue_width]
    return features


class GraphObservationHistory:
    """Rolling graph-feature buffer with repeat padding at episode start."""

    def __init__(self, history_len: int, graph: TrafficSignalGraph):
        self.history_len = max(1, int(history_len))
        self.graph = graph
        self._frames: Deque[np.ndarray] = deque(maxlen=self.history_len)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_len, self.graph.num_nodes, self.graph.feature_dim),
            dtype=np.float32,
        )

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        clean_frame = np.asarray(frame, dtype=np.float32)
        for _ in range(self.history_len):
            self._frames.append(clean_frame.copy())
        return self.as_array()

    def append(self, frame: np.ndarray) -> np.ndarray:
        clean_frame = np.asarray(frame, dtype=np.float32)
        if not self._frames:
            return self.reset(clean_frame)
        self._frames.append(clean_frame.copy())
        return self.as_array()

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(
                (self.history_len, self.graph.num_nodes, self.graph.feature_dim),
                dtype=np.float32,
            )
        frames = list(self._frames)
        while len(frames) < self.history_len:
            frames.insert(0, frames[0].copy())
        return np.stack(frames, axis=0).astype(np.float32, copy=False)


def traffic_signals_from_base_env(base_env: Any) -> list[Any]:
    ts_ids: Iterable[str] = getattr(base_env, "ts_ids", None) or []
    traffic_signals = getattr(base_env, "traffic_signals", None)
    if isinstance(traffic_signals, Mapping) and ts_ids:
        return [traffic_signals[ts_id] for ts_id in ts_ids]
    if isinstance(traffic_signals, Mapping):
        return _ordered_traffic_signals(traffic_signals)
    return _ordered_traffic_signals(traffic_signals or [])

