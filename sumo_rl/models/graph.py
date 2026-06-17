"""Graph topology and feature helpers for traffic-signal networks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Mapping, Sequence

import numpy as np
from gymnasium import spaces


def _normalize_feature_layout(feature_layout: str | None) -> str:
    layout = str(feature_layout or "phase_min_green_density_queue").strip().lower()
    aliases = {
        "phase_min_green_density_queue": "phase_min_green_density_queue",
        "full_tls_state": "phase_min_green_density_queue",
        "density_queue": "density_queue",
    }
    if layout not in aliases:
        raise ValueError(
            "Unsupported graph feature_layout. Expected one of: "
            "phase_min_green_density_queue, full_tls_state, density_queue."
        )
    return aliases[layout]


@dataclass(frozen=True)
class TrafficSignalGraph:
    """Static graph metadata derived from SUMO traffic-signal lane links."""

    ts_ids: tuple[str, ...]
    ts_index: dict[str, int]
    num_nodes: int
    max_lanes: int
    max_green_phases: int
    adjacency: np.ndarray
    edge_index: np.ndarray
    feature_layout: str = "phase_min_green_density_queue"
    incoming_node_index: int | None = None
    outgoing_node_index: int | None = None

    @property
    def phase_dim(self) -> int:
        return 0 if self.feature_layout == "density_queue" else self.max_green_phases

    @property
    def feature_dim(self) -> int:
        if self.feature_layout == "density_queue":
            return 2 * self.max_lanes
        return self.max_green_phases + 1 + 2 * self.max_lanes

    @property
    def min_green_index(self) -> int | None:
        if self.feature_layout == "density_queue":
            return None
        return self.max_green_phases

    @property
    def density_offset(self) -> int:
        return 0 if self.feature_layout == "density_queue" else self.max_green_phases + 1

    @property
    def queue_offset(self) -> int:
        return self.density_offset + self.max_lanes

    def phase_slice(self) -> slice:
        return slice(0, self.phase_dim)

    def model_config(self, agent_id: str, **extra: Any) -> dict[str, Any]:
        config = {
            "agent_id": str(agent_id),
            "agent_index": int(self.ts_index[str(agent_id)]),
            "num_nodes": int(self.num_nodes),
            "input_dim": int(self.feature_dim),
            "adjacency": self.adjacency.astype(np.float32).tolist(),
            "ts_ids": list(self.ts_ids),
            "feature_layout": self.feature_layout,
            "max_lanes": int(self.max_lanes),
            "max_green_phases": int(self.max_green_phases),
            "density_offset": int(self.density_offset),
            "queue_offset": int(self.queue_offset),
        }
        if self.min_green_index is not None:
            config["min_green_index"] = int(self.min_green_index)
        config.update(extra)
        return config


def _ordered_traffic_signals(traffic_signals: Mapping[str, Any] | Sequence[Any]) -> list[Any]:
    if isinstance(traffic_signals, Mapping):
        return [traffic_signals[key] for key in sorted(traffic_signals)]
    return list(traffic_signals)


def _signal_id(ts: Any) -> str:
    return str(getattr(ts, "id"))


def _phase_one_hot(ts: Any, max_green_phases: int) -> np.ndarray:
    encoded = np.zeros(max_green_phases, dtype=np.float32)
    if max_green_phases <= 0:
        return encoded
    green_phase = int(getattr(ts, "green_phase", 0) or 0)
    if 0 <= green_phase < max_green_phases:
        encoded[green_phase] = 1.0
    return encoded


def _min_green_feature(ts: Any) -> float:
    if not all(hasattr(ts, name) for name in ("time_since_last_phase_change", "min_green", "yellow_time")):
        return 0.0
    return float(
        0
        if float(getattr(ts, "time_since_last_phase_change")) < float(getattr(ts, "min_green")) + float(getattr(ts, "yellow_time"))
        else 1
    )


def _pack_feature_row(ts: Any, graph: TrafficSignalGraph) -> np.ndarray:
    density = np.asarray(ts.get_lanes_density(), dtype=np.float32).reshape(-1)
    queue = np.asarray(ts.get_lanes_queue(), dtype=np.float32).reshape(-1)
    features = np.zeros(graph.feature_dim, dtype=np.float32)
    if graph.feature_layout == "phase_min_green_density_queue":
        features[graph.phase_slice()] = _phase_one_hot(ts, graph.max_green_phases)
        if graph.min_green_index is not None:
            features[graph.min_green_index] = _min_green_feature(ts)
    density_width = min(graph.max_lanes, density.size)
    queue_width = min(graph.max_lanes, queue.size)
    features[graph.density_offset : graph.density_offset + density_width] = density[:density_width]
    features[graph.queue_offset : graph.queue_offset + queue_width] = queue[:queue_width]
    return features


def build_traffic_signal_graph(
    traffic_signals: Mapping[str, Any] | Sequence[Any],
    *,
    include_virtual_nodes: bool = True,
    add_self_loops: bool = True,
    feature_layout: str = "phase_min_green_density_queue",
) -> TrafficSignalGraph:
    """Build a deterministic directed graph from traffic signal in/out lanes."""

    ts_list = _ordered_traffic_signals(traffic_signals)
    if not ts_list:
        raise ValueError("Cannot build a traffic-signal graph without traffic signals.")

    normalized_feature_layout = _normalize_feature_layout(feature_layout)
    ts_ids = tuple(_signal_id(ts) for ts in ts_list)
    ts_index = {ts_id: index for index, ts_id in enumerate(ts_ids)}
    max_lanes = max(1, max(len(getattr(ts, "lanes", []) or []) for ts in ts_list))
    max_green_phases = max(1, max(int(getattr(ts, "num_green_phases", 1) or 1) for ts in ts_list))

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
        max_green_phases=max_green_phases,
        adjacency=adjacency,
        edge_index=edge_index,
        feature_layout=normalized_feature_layout,
        incoming_node_index=incoming_node_index,
        outgoing_node_index=outgoing_node_index,
    )


def pack_graph_features(
    traffic_signals: Mapping[str, Any] | Sequence[Any],
    graph: TrafficSignalGraph,
) -> np.ndarray:
    """Pack current graph-state features into a graph node matrix."""

    ts_by_id = {_signal_id(ts): ts for ts in _ordered_traffic_signals(traffic_signals)}
    features = np.zeros((graph.num_nodes, graph.feature_dim), dtype=np.float32)
    for ts_id in graph.ts_ids:
        ts = ts_by_id[ts_id]
        node_index = graph.ts_index[ts_id]
        features[node_index] = _pack_feature_row(ts, graph)
    return features


def pack_density_queue_features(
    traffic_signals: Mapping[str, Any] | Sequence[Any],
    graph: TrafficSignalGraph,
) -> np.ndarray:
    """Backward-compatible wrapper around ``pack_graph_features``."""

    return pack_graph_features(traffic_signals, graph)


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
