"""Render CoLight topology overlays from SUMO networks."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Optional


Point = tuple[float, float]


def _base_sumo_env(env: Any) -> Any:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "traffic_signals") and hasattr(current, "ts_ids"):
            return current
        for attr in ("env", "par_env", "aec_env", "base_env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    return env


def _load_sumo_net(net_file: str):
    import sumolib

    return sumolib.net.readNet(str(net_file))


def _edge_shapes(net: Any) -> list[list[Point]]:
    shapes = []
    for edge in net.getEdges():
        try:
            shape = [(float(x), float(y)) for x, y in edge.getShape()]
        except Exception:
            shape = []
        if len(shape) >= 2:
            shapes.append(shape)
    return shapes


def _lane_shape_points(net: Any, lane_ids: Iterable[str]) -> list[Point]:
    points: list[Point] = []
    for lane_id in lane_ids:
        try:
            lane = net.getLane(str(lane_id))
            points.extend((float(x), float(y)) for x, y in lane.getShape())
        except Exception:
            continue
    return points


def _mean_point(points: list[Point]) -> Optional[Point]:
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _traffic_signal_positions(net: Any, base_env: Any, agent_ids: list[str]) -> dict[str, Point]:
    traffic_signals = getattr(base_env, "traffic_signals", {}) or {}
    positions: dict[str, Point] = {}
    for agent_id in agent_ids:
        try:
            x, y = net.getNode(str(agent_id)).getCoord()
            positions[agent_id] = (float(x), float(y))
            continue
        except Exception:
            pass

        signal = traffic_signals.get(agent_id)
        if signal is None:
            continue
        lane_ids = list(getattr(signal, "lanes", []) or []) + list(getattr(signal, "out_lanes", []) or [])
        mean = _mean_point(_lane_shape_points(net, lane_ids))
        if mean is not None:
            positions[agent_id] = mean
    return positions


def _bounds(shapes: list[list[Point]], positions: dict[str, Point]) -> tuple[float, float, float, float]:
    points = [point for shape in shapes for point in shape] + list(positions.values())
    if not points:
        return (0.0, 1.0, 0.0, 1.0)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def _transformer(bounds: tuple[float, float, float, float], width: int, height: int, padding: int):
    min_x, max_x, min_y, max_y = bounds
    scale = min((width - 2 * padding) / (max_x - min_x), (height - 2 * padding) / (max_y - min_y))
    x_offset = (width - scale * (max_x - min_x)) / 2.0
    y_offset = (height - scale * (max_y - min_y)) / 2.0

    def _project(point: Point) -> Point:
        x = x_offset + (point[0] - min_x) * scale
        y = height - (y_offset + (point[1] - min_y) * scale)
        return x, y

    return _project


def _polyline(points: list[Point]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _offset_line(start: Point, end: Point, amount: float) -> tuple[Point, Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx = -dy / length
    ny = dx / length
    return (start[0] + nx * amount, start[1] + ny * amount), (end[0] + nx * amount, end[1] + ny * amount)


def _shorten_line(start: Point, end: Point, amount: float) -> tuple[Point, Point]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux = dx / length
    uy = dy / length
    return (start[0] + ux * amount, start[1] + uy * amount), (end[0] - ux * amount, end[1] - uy * amount)


def _svg_document(
    *,
    road_shapes: list[list[Point]],
    positions: dict[str, Point],
    topology_edges: list[tuple[str, str]],
    width: int,
    height: int,
) -> str:
    bounds = _bounds(road_shapes, positions)
    project = _transformer(bounds, width, height, padding=48)
    projected_roads = [[project(point) for point in shape] for shape in road_shapes]
    projected_positions = {agent_id: project(point) for agent_id, point in positions.items()}
    topology_set = set(topology_edges)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "<defs>",
        '<marker id="topology-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="#dc2626" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<g id="sumo-road-network" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-opacity="0.55">',
    ]
    for shape in projected_roads:
        lines.append(f'<polyline points="{_polyline(shape)}" />')
    lines.append("</g>")
    lines.append('<g id="colight-topology" fill="none" stroke="#dc2626" stroke-width="2.6" stroke-opacity="0.88">')
    for source_id, target_id in topology_edges:
        if source_id not in projected_positions or target_id not in projected_positions:
            continue
        start = projected_positions[source_id]
        end = projected_positions[target_id]
        offset = 4.5 if (target_id, source_id) in topology_set and source_id < target_id else -4.5
        start, end = _offset_line(start, end, offset)
        start, end = _shorten_line(start, end, 13.0)
        lines.append(
            f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" '
            f'x2="{end[0]:.2f}" y2="{end[1]:.2f}" marker-end="url(#topology-arrow)" />'
        )
    lines.append("</g>")
    lines.append('<g id="traffic-signal-nodes" font-family="Arial, sans-serif" font-size="11">')
    for agent_id, point in projected_positions.items():
        label = html.escape(str(agent_id))
        lines.append(f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="5.5" fill="#0f766e" stroke="#ffffff" />')
        lines.append(
            f'<text x="{point[0] + 7:.2f}" y="{point[1] - 7:.2f}" '
            f'fill="#0f172a" paint-order="stroke" stroke="#ffffff" stroke-width="3">{label}</text>'
        )
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_colight_topology(
    graph_env: Any,
    net_file: str,
    output_dir: Path,
    *,
    width: int = 1200,
) -> dict[str, Path]:
    """Write an SVG topology overlay and JSON edge list for a CoLight graph env."""

    base_env = _base_sumo_env(graph_env)
    agent_ids = [str(agent_id) for agent_id in getattr(graph_env, "possible_agents", [])]
    index_to_agent = {int(index): agent_id for agent_id, index in getattr(graph_env, "_agent_to_index", {}).items()}
    topology_edges = [
        (index_to_agent[source], index_to_agent[target])
        for source, target in getattr(graph_env, "_edges", [])
        if source in index_to_agent and target in index_to_agent
    ]

    net = _load_sumo_net(net_file)
    road_shapes = _edge_shapes(net)
    positions = _traffic_signal_positions(net, base_env, agent_ids)
    min_x, max_x, min_y, max_y = _bounds(road_shapes, positions)
    aspect = (max_y - min_y) / max(max_x - min_x, 1e-6)
    height = max(480, min(1600, int(width * aspect)))

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "colight_topology.svg"
    json_path = output_dir / "colight_topology_edges.json"
    svg_path.write_text(
        _svg_document(
            road_shapes=road_shapes,
            positions=positions,
            topology_edges=topology_edges,
            width=width,
            height=height,
        ),
        encoding="utf-8",
    )
    payload = {
        "net_file": str(net_file),
        "num_nodes": len(agent_ids),
        "num_directed_edges": len(topology_edges),
        "nodes": [{"id": agent_id, "position": positions.get(agent_id)} for agent_id in agent_ids],
        "directed_edges": [{"source": source, "target": target} for source, target in topology_edges],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"svg": svg_path, "json": json_path}

