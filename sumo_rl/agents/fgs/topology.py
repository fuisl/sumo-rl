"""Traffic-light graph construction for FGS.

This mirrors the TLS super-edge idea from HMARL-TSC: parse the SUMO network,
follow legal road-edge transitions, and connect each signal to the nearest
downstream signal reachable from each outgoing edge.
"""

from __future__ import annotations

import heapq
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

Edge = tuple[str, str]
Point = tuple[float, float]


@dataclass(frozen=True)
class RoadEdge:
    edge_id: str
    source_node: str
    target_node: str
    length: float
    travel_time: float
    lane_count: int
    shape: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class TLSSuperEdge:
    source: str
    target: str
    length: float
    travel_time: float
    lane_count: int
    path_edge_ids: list[str]
    path_node_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "length": self.length,
            "travel_time": self.travel_time,
            "lane_count": self.lane_count,
            "path_edge_ids": list(self.path_edge_ids),
            "path_node_ids": list(self.path_node_ids),
        }


@dataclass(frozen=True)
class TLSTopology:
    workers: list[str]
    directed_edges: list[Edge]
    edges: list[Edge]
    edge_weights: dict[Edge, float]
    super_edges: list[TLSSuperEdge]
    positions: dict[str, Point]
    road_polylines: list[list[Point]]

    def to_dict(self) -> dict:
        return {
            "workers": list(self.workers),
            "directed_edges": [list(edge) for edge in self.directed_edges],
            "edges": [list(edge) for edge in self.edges],
            "edge_weights": {"{}--{}".format(*edge): weight for edge, weight in sorted(self.edge_weights.items())},
            "super_edges": [edge.to_dict() for edge in self.super_edges],
            "positions": {worker: [x, y] for worker, (x, y) in sorted(self.positions.items())},
        }


def extract_tls_topology(net_file: str | Path) -> TLSTopology:
    """Extract a contracted TLS graph from a SUMO ``.net.xml`` file."""

    root = ET.parse(str(net_file)).getroot()
    junction_positions = _extract_positions(root)
    edge_catalog, outgoing_by_node = _build_edge_catalog(root, junction_positions)
    tls_junctions = _extract_tls_junctions(root, junction_positions, edge_catalog)
    tls_ids = sorted(tls_junctions)
    transitions = _build_edge_transitions(root, edge_catalog)
    super_edges = _build_super_edges(
        edge_catalog=edge_catalog,
        outgoing_by_node=outgoing_by_node,
        transitions=transitions,
        tls_ids=tls_ids,
        tls_junctions=tls_junctions,
    )

    directed_edges = sorted({(edge.source, edge.target) for edge in super_edges})
    undirected_edges = sorted({_canonical_edge(edge.source, edge.target) for edge in super_edges})
    edge_weights: dict[Edge, float] = {}
    for edge in super_edges:
        key = _canonical_edge(edge.source, edge.target)
        weight = 1.0 / edge.travel_time if edge.travel_time > 0 else 1.0
        edge_weights[key] = edge_weights.get(key, 0.0) + weight

    return TLSTopology(
        workers=tls_ids,
        directed_edges=directed_edges,
        edges=undirected_edges,
        edge_weights={edge: edge_weights[edge] for edge in undirected_edges},
        super_edges=[edge for edge in super_edges],
        positions={worker: junction_positions[tls_junctions[worker]] for worker in tls_ids},
        road_polylines=_extract_road_polylines(root, junction_positions),
    )


def bidirectional_message_edges(topology: TLSTopology, agent_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Return bidirectional message-passing edges constrained to known agents."""

    agent_set = {str(agent_id) for agent_id in agent_ids}
    edges = set()
    for source, target in topology.directed_edges:
        if source in agent_set and target in agent_set and source != target:
            edges.add((source, target))
            edges.add((target, source))
    return sorted(edges)


def render_fgs_topology(topology: TLSTopology, output_dir: Path, *, width: int = 1200) -> dict[str, Path]:
    """Write JSON and SVG topology artifacts for auditability."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fgs_topology_edges.json"
    svg_path = output_dir / "fgs_topology.svg"
    json_path.write_text(json.dumps(topology.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    svg_path.write_text(_svg_document(topology, width=width), encoding="utf-8")
    return {"json": json_path, "svg": svg_path}


def _extract_tls_junctions(
    root: ET.Element,
    positions: dict[str, Point],
    edge_catalog: dict[str, RoadEdge],
) -> dict[str, str]:
    tls_junctions: dict[str, str] = {}
    for tls in root.findall("tlLogic"):
        tls_id = tls.get("id")
        if not tls_id:
            continue
        if tls_id in positions:
            tls_junctions[tls_id] = tls_id
            continue
        junction_id = _infer_tls_junction(root, tls_id, positions, edge_catalog)
        if junction_id is not None:
            tls_junctions[tls_id] = junction_id
    if not tls_junctions:
        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            if junction_id and junction.get("type") == "traffic_light" and junction_id in positions:
                tls_junctions[junction_id] = junction_id
    if not tls_junctions:
        raise ValueError("No traffic light IDs with junction positions found in SUMO net.")
    return dict(sorted(tls_junctions.items()))


def _infer_tls_junction(
    root: ET.Element,
    tls_id: str,
    positions: dict[str, Point],
    edge_catalog: dict[str, RoadEdge],
) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id:
            continue
        from_edge = edge_catalog.get(str(connection.get("from") or ""))
        if from_edge is not None and from_edge.target_node in positions:
            counts[from_edge.target_node] += 1
        to_edge = edge_catalog.get(str(connection.get("to") or ""))
        if to_edge is not None and to_edge.source_node in positions:
            counts[to_edge.source_node] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _extract_positions(root: ET.Element) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id or junction.get("type") == "internal":
            continue
        x = _safe_float(junction.get("x"))
        y = _safe_float(junction.get("y"))
        if x is not None and y is not None:
            positions[junction_id] = (x, y)
    return positions


def _build_edge_catalog(root: ET.Element, positions: dict[str, Point]) -> tuple[dict[str, RoadEdge], dict[str, list[str]]]:
    catalog: dict[str, RoadEdge] = {}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge_elem in root.findall("edge"):
        edge_id = edge_elem.get("id")
        source = edge_elem.get("from")
        target = edge_elem.get("to")
        if (
            not edge_id
            or edge_elem.get("function") == "internal"
            or not source
            or not target
            or source not in positions
            or target not in positions
        ):
            continue
        length, travel_time, lane_count, shape = _edge_metrics(edge_elem, positions)
        catalog[edge_id] = RoadEdge(edge_id, source, target, length, travel_time, lane_count, shape)
        outgoing[source].append(edge_id)
    return catalog, {node: sorted(edge_ids) for node, edge_ids in outgoing.items()}


def _build_edge_transitions(root: ET.Element, edge_catalog: dict[str, RoadEdge]) -> dict[str, list[str]]:
    transitions: dict[str, set[str]] = {edge_id: set() for edge_id in edge_catalog}
    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        if from_edge in edge_catalog and to_edge in edge_catalog:
            transitions[from_edge].add(to_edge)
    return {edge_id: sorted(next_edges) for edge_id, next_edges in transitions.items()}


def _build_super_edges(
    *,
    edge_catalog: dict[str, RoadEdge],
    outgoing_by_node: dict[str, list[str]],
    transitions: dict[str, list[str]],
    tls_ids: list[str],
    tls_junctions: dict[str, str],
) -> list[TLSSuperEdge]:
    node_to_tls: dict[str, list[str]] = defaultdict(list)
    for tls_id in tls_ids:
        node_to_tls[tls_junctions[tls_id]].append(tls_id)
    for node_id in node_to_tls:
        node_to_tls[node_id] = sorted(node_to_tls[node_id])
    tls_node_set = set(node_to_tls)
    best_by_pair: dict[Edge, TLSSuperEdge] = {}
    for source in tls_ids:
        source_node = tls_junctions[source]
        for start_edge in outgoing_by_node.get(source_node, []):
            result = _find_nearest_tls(edge_catalog, transitions, tls_node_set, source_node, start_edge)
            if result is None:
                continue
            target_node, path_edge_ids = result
            for target in node_to_tls[target_node]:
                candidate = _super_edge_from_path(source, target, path_edge_ids, edge_catalog)
                if candidate.source == candidate.target:
                    continue
                key = (candidate.source, candidate.target)
                existing = best_by_pair.get(key)
                if existing is None or candidate.travel_time < existing.travel_time:
                    best_by_pair[key] = candidate
    return [best_by_pair[key] for key in sorted(best_by_pair)]


def _find_nearest_tls(
    edge_catalog: dict[str, RoadEdge],
    transitions: dict[str, list[str]],
    tls_node_set: set[str],
    source_node: str,
    start_edge: str,
) -> tuple[str, list[str]] | None:
    queue: list[tuple[float, str]] = []
    best_cost = {start_edge: float(edge_catalog[start_edge].travel_time)}
    predecessor: dict[str, str | None] = {start_edge: None}
    heapq.heappush(queue, (best_cost[start_edge], start_edge))

    while queue:
        current_cost, current_edge = heapq.heappop(queue)
        if current_cost != best_cost.get(current_edge):
            continue
        target_node = edge_catalog[current_edge].target_node
        if target_node in tls_node_set and target_node != source_node:
            path_edge_ids = []
            cursor: str | None = current_edge
            while cursor is not None:
                path_edge_ids.append(cursor)
                cursor = predecessor[cursor]
            path_edge_ids.reverse()
            return target_node, path_edge_ids
        for next_edge in transitions.get(current_edge, []):
            next_cost = current_cost + float(edge_catalog[next_edge].travel_time)
            if next_cost >= best_cost.get(next_edge, math.inf):
                continue
            best_cost[next_edge] = next_cost
            predecessor[next_edge] = current_edge
            heapq.heappush(queue, (next_cost, next_edge))
    return None


def _super_edge_from_path(
    source: str,
    target: str,
    path_edge_ids: list[str],
    edge_catalog: dict[str, RoadEdge],
) -> TLSSuperEdge:
    length = 0.0
    travel_time = 0.0
    lane_count = 0
    path_node_ids: list[str] = []
    for index, edge_id in enumerate(path_edge_ids):
        edge = edge_catalog[edge_id]
        if index == 0:
            path_node_ids.append(edge.source_node)
        path_node_ids.append(edge.target_node)
        length += edge.length
        travel_time += edge.travel_time
        lane_count = max(lane_count, edge.lane_count)
    return TLSSuperEdge(source, target, length, travel_time, lane_count, list(path_edge_ids), path_node_ids)


def _edge_metrics(edge_elem: ET.Element, positions: dict[str, Point]) -> tuple[float, float, int, list[Point]]:
    lengths: list[float] = []
    travel_times: list[float] = []
    speeds: list[float] = []
    shapes: list[list[Point]] = []
    for lane in edge_elem.findall("lane"):
        length = _safe_float(lane.get("length"))
        speed = _safe_float(lane.get("speed"))
        if length is not None:
            lengths.append(length)
        if speed is not None and speed > 0:
            speeds.append(speed)
            if length is not None:
                travel_times.append(length / speed)
        shape = _parse_shape(lane.get("shape"))
        if shape:
            shapes.append(shape)
    source = edge_elem.get("from")
    target = edge_elem.get("to")
    length = sum(lengths) / len(lengths) if lengths else _distance(source, target, positions)
    if travel_times:
        travel_time = sum(travel_times) / len(travel_times)
    elif speeds and length > 0:
        travel_time = length / max(speeds)
    else:
        travel_time = max(length, 1.0)
    shape = shapes[0] if shapes else _shape_from_nodes(source, target, positions)
    return length, travel_time, len(edge_elem.findall("lane")), shape


def _extract_road_polylines(root: ET.Element, positions: dict[str, Point]) -> list[list[Point]]:
    polylines: list[list[Point]] = []
    for edge_elem in root.findall("edge"):
        if edge_elem.get("function") == "internal":
            continue
        shape = None
        for lane in edge_elem.findall("lane"):
            shape = _parse_shape(lane.get("shape"))
            if shape:
                break
        if not shape:
            shape = _shape_from_nodes(edge_elem.get("from"), edge_elem.get("to"), positions)
        if len(shape) >= 2:
            polylines.append(shape)
    return polylines


def _parse_shape(raw: str | None) -> list[Point]:
    if not raw:
        return []
    points = []
    for item in raw.split():
        if "," not in item:
            continue
        x, y = item.split(",", 1)
        try:
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def _shape_from_nodes(source: str | None, target: str | None, positions: dict[str, Point]) -> list[Point]:
    if source in positions and target in positions:
        return [positions[source], positions[target]]
    return []


def _distance(source: str | None, target: str | None, positions: dict[str, Point]) -> float:
    if source in positions and target in positions:
        sx, sy = positions[source]
        tx, ty = positions[target]
        return float(math.hypot(tx - sx, ty - sy))
    return 0.0


def _safe_float(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _canonical_edge(source: str, target: str) -> Edge:
    return (source, target) if source <= target else (target, source)


def _svg_document(topology: TLSTopology, *, width: int) -> str:
    positions = topology.positions
    points = [point for polyline in topology.road_polylines for point in polyline] + list(positions.values())
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    height = max(360, min(1600, int(width * (max_y - min_y) / max(max_x - min_x, 1e-6))))
    padding = 48
    scale = min((width - 2 * padding) / (max_x - min_x), (height - 2 * padding) / (max_y - min_y))

    def project(point: Point) -> Point:
        x = padding + (point[0] - min_x) * scale
        y = height - (padding + (point[1] - min_y) * scale)
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="#2563eb" /></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<g fill="none" stroke="#94a3b8" stroke-width="1.2" stroke-opacity="0.55">',
    ]
    for polyline in topology.road_polylines:
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in [project(point) for point in polyline])
        lines.append(f'<polyline points="{path}" />')
    lines.append("</g>")
    lines.append('<g fill="none" stroke="#2563eb" stroke-width="2.4" stroke-opacity="0.85">')
    for source, target in topology.directed_edges:
        if source not in positions or target not in positions:
            continue
        sx, sy = project(positions[source])
        tx, ty = project(positions[target])
        lines.append(f'<line x1="{sx:.2f}" y1="{sy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" marker-end="url(#arrow)" />')
    lines.append("</g>")
    lines.append('<g font-family="Arial, sans-serif" font-size="11">')
    for worker, point in positions.items():
        x, y = project(point)
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="#0f766e" stroke="#ffffff" />')
        lines.append(f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" fill="#0f172a">{worker}</text>')
    lines.append("</g></svg>")
    return "\n".join(lines) + "\n"
