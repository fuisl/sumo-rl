"""Topology extraction for RESCO SUMO networks."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

try:
    import sumolib
except ImportError:  # pragma: no cover - handled at runtime where SUMO is required
    sumolib = None


@dataclass(frozen=True)
class GraphTopology:
    edge_index: Tensor
    edge_attr: Tensor | None
    node_ids: tuple[str, ...]


def _iter_neighbor_nodes(node):
    neighbors = []
    for edge in node.getOutgoing():
        next_node = edge.getToNode()
        if next_node.getID() == node.getID():
            continue
        neighbors.append((next_node, float(edge.getLength()), int(edge.getLaneNumber())))
    for edge in node.getIncoming():
        next_node = edge.getFromNode()
        if next_node.getID() == node.getID():
            continue
        neighbors.append((next_node, float(edge.getLength()), int(edge.getLaneNumber())))
    return neighbors


def _walk_to_neighbor_tls(source_id: str, start_node, agent_ids: set[str]) -> list[tuple[str, float, int]]:
    heap: list[tuple[float, float, str, object]] = [(0.0, float("inf"), start_node.getID(), start_node)]
    best_dist: dict[str, float] = {start_node.getID(): 0.0}
    best_terminal: dict[str, tuple[float, int]] = {}

    while heap:
        dist, bottleneck_lanes, node_id, node = heapq.heappop(heap)
        if dist > best_dist.get(node_id, float("inf")):
            continue

        if node_id in agent_ids and node_id != source_id:
            lane_count = 0 if bottleneck_lanes == float("inf") else int(bottleneck_lanes)
            prev = best_terminal.get(node_id)
            if prev is None or dist < prev[0] or (dist == prev[0] and lane_count > prev[1]):
                best_terminal[node_id] = (dist, lane_count)
            continue

        for next_node, edge_dist, edge_lanes in _iter_neighbor_nodes(node):
            next_id = next_node.getID()
            next_dist = dist + edge_dist
            if next_dist >= best_dist.get(next_id, float("inf")):
                continue
            best_dist[next_id] = next_dist
            next_bottleneck = float(edge_lanes) if bottleneck_lanes == float("inf") else min(bottleneck_lanes, float(edge_lanes))
            heapq.heappush(heap, (next_dist, next_bottleneck, next_id, next_node))

    return [(node_id, dist, lane_count) for node_id, (dist, lane_count) in best_terminal.items()]


def build_resco_topology(net_file: str, agent_ids: Sequence[str]) -> GraphTopology:
    if sumolib is None:
        raise ImportError("sumolib is required to build RESCO topology.")

    net = sumolib.net.readNet(net_file, withInternal=False)
    nodes = {node.getID(): node for node in net.getNodes()}
    ordered_agent_ids = [str(agent_id) for agent_id in agent_ids]
    agent_id_set = set(ordered_agent_ids)
    id_to_index = {agent_id: idx for idx, agent_id in enumerate(ordered_agent_ids)}

    src_list: list[int] = []
    dst_list: list[int] = []
    attrs: list[list[float]] = []

    for source_id in ordered_agent_ids:
        node = nodes.get(source_id)
        if node is None:
            continue
        for neighbor_id, distance, lane_count in _walk_to_neighbor_tls(source_id, node, agent_id_set):
            if neighbor_id not in id_to_index:
                continue
            src_list.append(id_to_index[source_id])
            dst_list.append(id_to_index[neighbor_id])
            attrs.append([distance, float(lane_count)])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None

    return GraphTopology(edge_index=edge_index, edge_attr=edge_attr, node_ids=tuple(ordered_agent_ids))
