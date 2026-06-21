from __future__ import annotations

import argparse
import html
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.fgs.topology import TLSTopology, extract_tls_topology


Point = tuple[float, float]

DEFAULT_NET_FILE = ROOT / "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml"
DEFAULT_OUTPUT_DIR = ROOT / "visualization/outputs/ingolstadt21"


@dataclass(frozen=True)
class JunctionInfo:
    junction_id: str
    junction_type: str
    position: Point
    tls_program_ids: tuple[str, ...]


@dataclass(frozen=True)
class NodeExtraction:
    junctions: list[JunctionInfo]
    road_polylines: list[list[Point]]
    tls_program_to_junction: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "counts": {
                "junctions": len(self.junctions),
                "road_polylines": len(self.road_polylines),
                "tls_programs": len(self.tls_program_to_junction),
                "tls_program_controlled_junctions": len(set(self.tls_program_to_junction.values())),
            },
            "tls_program_to_junction": dict(sorted(self.tls_program_to_junction.items())),
            "junctions": [
                {
                    "id": junction.junction_id,
                    "type": junction.junction_type,
                    "position": [junction.position[0], junction.position[1]],
                    "tls_program_ids": list(junction.tls_program_ids),
                }
                for junction in self.junctions
            ],
        }


def render_fgs_visualization(net_file: str | Path, output_dir: str | Path, *, width: int = 1400) -> dict[str, Path]:
    """Render the node extraction and final FGS topology stages for a SUMO net."""

    net_path = Path(net_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_extraction = extract_node_stage(net_path)
    topology = extract_tls_topology(net_path)

    node_svg = out_dir / "01_node_extraction.svg"
    topology_svg = out_dir / "02_fgs_extracted_topology.svg"
    node_json = out_dir / "node_extraction.json"
    topology_json = out_dir / "fgs_topology.json"

    node_svg.write_text(_node_extraction_svg(node_extraction, net_path=net_path, width=width), encoding="utf-8")
    topology_svg.write_text(
        _topology_svg(topology, node_extraction=node_extraction, net_path=net_path, width=width),
        encoding="utf-8",
    )
    node_json.write_text(json.dumps(node_extraction.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    topology_json.write_text(json.dumps(topology.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    return {
        "node_svg": node_svg,
        "topology_svg": topology_svg,
        "node_json": node_json,
        "topology_json": topology_json,
    }


def extract_node_stage(net_file: str | Path) -> NodeExtraction:
    root = ET.parse(str(net_file)).getroot()
    junction_positions: dict[str, Point] = {}
    junction_types: dict[str, str] = {}
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id or junction.get("type") == "internal":
            continue
        x = _safe_float(junction.get("x"))
        y = _safe_float(junction.get("y"))
        if x is None or y is None:
            continue
        junction_positions[junction_id] = (x, y)
        junction_types[junction_id] = str(junction.get("type") or "")

    edge_endpoints = _edge_endpoints(root, junction_positions)
    tls_program_to_junction = _tls_program_junctions(root, junction_positions, edge_endpoints)
    junction_to_tls: dict[str, list[str]] = defaultdict(list)
    for tls_id, junction_id in tls_program_to_junction.items():
        junction_to_tls[junction_id].append(tls_id)

    junctions = [
        JunctionInfo(
            junction_id=junction_id,
            junction_type=junction_types[junction_id],
            position=junction_positions[junction_id],
            tls_program_ids=tuple(sorted(junction_to_tls.get(junction_id, []))),
        )
        for junction_id in sorted(junction_positions)
    ]
    return NodeExtraction(
        junctions=junctions,
        road_polylines=_road_polylines(root, junction_positions),
        tls_program_to_junction=dict(sorted(tls_program_to_junction.items())),
    )


def _edge_endpoints(root: ET.Element, positions: dict[str, Point]) -> dict[str, tuple[str, str]]:
    endpoints = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        source = edge.get("from")
        target = edge.get("to")
        if (
            edge_id
            and edge.get("function") != "internal"
            and source in positions
            and target in positions
        ):
            endpoints[edge_id] = (str(source), str(target))
    return endpoints


def _tls_program_junctions(
    root: ET.Element,
    positions: dict[str, Point],
    edge_endpoints: dict[str, tuple[str, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tls in root.findall("tlLogic"):
        tls_id = tls.get("id")
        if not tls_id:
            continue
        if tls_id in positions:
            mapping[tls_id] = tls_id
            continue
        inferred = _infer_controlled_junction(root, tls_id, positions, edge_endpoints)
        if inferred is not None:
            mapping[tls_id] = inferred
    if mapping:
        return mapping

    for junction_id in sorted(positions):
        junction = root.find(f"junction[@id='{junction_id}']")
        if junction is not None and junction.get("type") == "traffic_light":
            mapping[junction_id] = junction_id
    return mapping


def _infer_controlled_junction(
    root: ET.Element,
    tls_id: str,
    positions: dict[str, Point],
    edge_endpoints: dict[str, tuple[str, str]],
) -> Optional[str]:
    counts: dict[str, int] = defaultdict(int)
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id:
            continue
        from_edge = edge_endpoints.get(str(connection.get("from") or ""))
        if from_edge is not None and from_edge[1] in positions:
            counts[from_edge[1]] += 1
        to_edge = edge_endpoints.get(str(connection.get("to") or ""))
        if to_edge is not None and to_edge[0] in positions:
            counts[to_edge[0]] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _road_polylines(root: ET.Element, positions: dict[str, Point]) -> list[list[Point]]:
    polylines: list[list[Point]] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        shape = []
        for lane in edge.findall("lane"):
            shape = _parse_shape(lane.get("shape"))
            if shape:
                break
        if not shape:
            source = edge.get("from")
            target = edge.get("to")
            if source in positions and target in positions:
                shape = [positions[source], positions[target]]
        if len(shape) >= 2:
            polylines.append(shape)
    return polylines


def _node_extraction_svg(extraction: NodeExtraction, *, net_path: Path, width: int) -> str:
    points = [point for road in extraction.road_polylines for point in road] + [
        junction.position for junction in extraction.junctions
    ]
    project, height = _projector(points, width)
    lines = _svg_header(width, height, title="FGS node extraction")
    lines.extend(_summary_block([
        f"Node extraction: {net_path.name}",
        f"Junctions: {len(extraction.junctions)}",
        f"Road polylines: {len(extraction.road_polylines)}",
        f"TLS programs: {len(extraction.tls_program_to_junction)}",
    ]))
    lines.append('<g id="underlying-network" fill="none" stroke="#9ca3af" stroke-width="1.1" stroke-opacity="0.45">')
    for road in extraction.road_polylines:
        lines.append(f'<polyline points="{_polyline([project(point) for point in road])}" />')
    lines.append("</g>")
    lines.append('<g id="junctions" stroke="#ffffff" stroke-width="0.8">')
    for junction in extraction.junctions:
        x, y = project(junction.position)
        if junction.tls_program_ids:
            fill = "#0f766e"
            radius = 4.8
            css_class = "tls-program-controlled"
        elif junction.junction_type == "traffic_light":
            fill = "#f59e0b"
            radius = 4.0
            css_class = "traffic-light-junction"
        else:
            fill = "#64748b"
            radius = 2.4
            css_class = "ordinary-junction"
        lines.append(
            f'<circle class="{css_class}" cx="{x:.2f}" cy="{y:.2f}" r="{radius:.1f}" fill="{fill}" />'
        )
    lines.append("</g>")
    lines.append('<g id="tls-program-labels" font-family="Arial, sans-serif" font-size="10">')
    for junction in extraction.junctions:
        if not junction.tls_program_ids:
            continue
        x, y = project(junction.position)
        label = html.escape(",".join(junction.tls_program_ids))
        lines.append(
            f'<text x="{x + 6:.2f}" y="{y - 6:.2f}" fill="#0f172a" '
            f'paint-order="stroke" stroke="#ffffff" stroke-width="3">{label}</text>'
        )
    lines.append("</g>")
    lines.extend(_legend([
        ("#64748b", "Non-internal junction"),
        ("#f59e0b", "Traffic-light junction"),
        ("#0f766e", "TLS-program-controlled junction"),
    ], x=24, y=height - 92))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _topology_svg(topology: TLSTopology, *, node_extraction: NodeExtraction, net_path: Path, width: int) -> str:
    points = [point for road in node_extraction.road_polylines for point in road] + list(topology.positions.values())
    project, height = _projector(points, width)
    edge_by_pair = {(edge.source, edge.target): edge for edge in topology.super_edges}
    max_weight = max((1.0 / edge.travel_time for edge in topology.super_edges if edge.travel_time > 0), default=1.0)

    lines = _svg_header(width, height, title="FGS extracted topology")
    lines.extend(_summary_block([
        f"FGS extracted topology: {net_path.name}",
        f"Junctions: {len(node_extraction.junctions)}",
        f"Road polylines: {len(node_extraction.road_polylines)}",
        f"TLS workers: {len(topology.workers)}",
        f"Directed super-edges: {len(topology.directed_edges)}",
    ]))
    lines.append('<g id="underlying-network" fill="none" stroke="#cbd5e1" stroke-width="1.0" stroke-opacity="0.28">')
    for road in node_extraction.road_polylines:
        lines.append(f'<polyline points="{_polyline([project(point) for point in road])}" />')
    lines.append("</g>")
    lines.append('<g id="fgs-super-edges" fill="none" stroke="#2563eb" stroke-linecap="round">')
    for source, target in topology.directed_edges:
        if source not in topology.positions or target not in topology.positions:
            continue
        super_edge = edge_by_pair.get((source, target))
        weight = 1.0 / super_edge.travel_time if super_edge is not None and super_edge.travel_time > 0 else 1.0
        strength = weight / max(max_weight, 1e-9)
        stroke_width = 1.8 + 3.2 * strength
        opacity = 0.42 + 0.46 * strength
        start = project(topology.positions[source])
        end = project(topology.positions[target])
        start, end = _offset_if_bidirectional(start, end, source, target, topology.directed_edges)
        start, end = _shorten_line(start, end, 8.5)
        title = html.escape(f"{source} -> {target}")
        lines.append(
            f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
            f'stroke-width="{stroke_width:.2f}" stroke-opacity="{opacity:.2f}" marker-end="url(#fgs-arrow)">'
            f"<title>{title}</title></line>"
        )
    lines.append("</g>")
    lines.append('<g id="fgs-tls-nodes" font-family="Arial, sans-serif" font-size="10">')
    for worker, point in sorted(topology.positions.items()):
        x, y = project(point)
        label = html.escape(worker)
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.0" fill="#0f766e" stroke="#ffffff" stroke-width="1.4" />')
        lines.append(
            f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" fill="#0f172a" '
            f'paint-order="stroke" stroke="#ffffff" stroke-width="3">{label}</text>'
        )
    lines.append("</g>")
    lines.extend(_legend([
        ("#cbd5e1", "Underlying SUMO network"),
        ("#2563eb", "FGS directed super-edge"),
        ("#0f766e", "FGS TLS worker"),
    ], x=24, y=height - 92))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_header(width: int, height: int, *, title: str) -> list[str]:
    escaped_title = html.escape(title)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f"<title>{escaped_title}</title>",
        "<defs>",
        '<marker id="fgs-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L9,4.5 L0,9 Z" fill="#2563eb" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]


def _summary_block(lines: list[str]) -> list[str]:
    output = ['<g id="summary" font-family="Arial, sans-serif">']
    output.append('<rect x="18" y="18" width="390" height="112" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" />')
    for index, line in enumerate(lines):
        size = 15 if index == 0 else 12
        weight = "700" if index == 0 else "400"
        output.append(
            f'<text x="32" y="{44 + index * 19}" font-size="{size}" font-weight="{weight}" fill="#0f172a">'
            f"{html.escape(line)}</text>"
        )
    output.append("</g>")
    return output


def _legend(items: list[tuple[str, str]], *, x: int, y: int) -> list[str]:
    height = 26 + 20 * len(items)
    lines = ['<g id="legend" font-family="Arial, sans-serif" font-size="12">']
    lines.append(
        f'<rect x="{x}" y="{y}" width="260" height="{height}" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" />'
    )
    lines.append(f'<text x="{x + 12}" y="{y + 20}" font-weight="700" fill="#0f172a">Legend</text>')
    for index, (color, label) in enumerate(items):
        item_y = y + 42 + index * 20
        lines.append(f'<circle cx="{x + 18}" cy="{item_y - 4}" r="5" fill="{color}" />')
        lines.append(f'<text x="{x + 32}" y="{item_y}" fill="#0f172a">{html.escape(label)}</text>')
    lines.append("</g>")
    return lines


def _projector(points: list[Point], width: int) -> tuple[Callable[[Point], Point], int]:
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    height = max(520, min(1800, int(width * (max_y - min_y) / max(max_x - min_x, 1e-6))))
    padding = 56
    scale = min((width - 2 * padding) / (max_x - min_x), (height - 2 * padding) / (max_y - min_y))

    def project(point: Point) -> Point:
        x = padding + (point[0] - min_x) * scale
        y = height - (padding + (point[1] - min_y) * scale)
        return x, y

    return project, height


def _offset_if_bidirectional(
    start: Point,
    end: Point,
    source: str,
    target: str,
    directed_edges: list[tuple[str, str]],
) -> tuple[Point, Point]:
    if (target, source) not in set(directed_edges):
        return start, end
    amount = 5.0 if source < target else -5.0
    return _offset_line(start, end, amount)


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


def _polyline(points: list[Point]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _parse_shape(raw: Optional[str]) -> list[Point]:
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


def _safe_float(raw: Optional[str]) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FGS topology extraction SVGs for a SUMO .net.xml file.")
    parser.add_argument("--net-file", type=Path, default=DEFAULT_NET_FILE, help="SUMO .net.xml file to visualize.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for SVG and JSON artifacts.")
    parser.add_argument("--width", type=int, default=1400, help="SVG width in pixels.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = render_fgs_visualization(args.net_file, args.output_dir, width=args.width)
    print(f"Wrote node extraction SVG: {paths['node_svg']}")
    print(f"Wrote FGS topology SVG: {paths['topology_svg']}")
    print(f"Wrote metadata JSON: {paths['node_json']}")
    print(f"Wrote topology JSON: {paths['topology_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
