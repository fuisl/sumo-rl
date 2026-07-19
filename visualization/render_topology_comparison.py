from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INGOLSTADT_NET = ROOT / "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml"
DEFAULT_OUTPUT_DIR = ROOT / "visualization/outputs/topology_comparison"
GUDANG_PLACE = "古荡街道, 西湖区, 杭州市, 浙江省, 中国"
DEFAULT_GUDANG_OSM_CACHE = ROOT / "cache/cad4a1545036256a274d63a3bda6e005e6884935.json"
GUDANG_FIGURE_BOUNDS_WGS84 = (120.087, 30.263, 120.127, 30.300)
GUDANG_CONTROL_BOUNDS_WGS84 = (120.095, 30.274, 120.120, 30.290)
COLIGHT_HANGZHOU_ROADNET_URL = (
    "https://raw.githubusercontent.com/wingsweihua/colight/master/data/Hangzhou/4_4/roadnet_4_4.json"
)
OSM_HIGHWAY_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
DESIGN_STYLE = {
    "ink": "#141413",
    "paper": "#faf9f5",
    "card": "#e8e6dc",
    "mid_gray": "#b0aea5",
    "orange": "#d97757",
    "blue": "#6a9bcc",
    "green": "#788c5d",
}
FONT_FAMILY = ["Liberitus Sans", "Libertinus Sans", "Liberation Sans", "DejaVu Sans"]
LOCAL_FONT_PATHS = [
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Regular.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Bold.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Italic.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Regular.ttf",
]


Point = tuple[float, float]


@dataclass(frozen=True)
class TopologyRenderData:
    roads: list[list[Point]]
    signal_points: list[Point]
    metadata: dict[str, object]
    bounds: tuple[float, float, float, float] | None = None


def render_topology_comparison(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    ingolstadt_net: str | Path = DEFAULT_INGOLSTADT_NET,
    gudang_roadnet: str | Path | None = None,
    gudang_osm_cache: str | Path = DEFAULT_GUDANG_OSM_CACHE,
    gudang_place: str = GUDANG_PLACE,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ingolstadt_roads, ingolstadt_signals = _sumo_topology(Path(ingolstadt_net))
    if gudang_roadnet:
        gudang_roadnet_path = Path(gudang_roadnet)
        gudang = _cityflow_topology(gudang_roadnet_path, gudang_place=gudang_place)
    else:
        gudang = _osm_gudang_original_topology(Path(gudang_osm_cache), gudang_place=gudang_place)

    written: dict[str, Path] = {}
    for fmt in formats:
        extension = fmt.lstrip(".")
        ingolstadt_path = out_dir / f"ingolstadt21_topology.{extension}"
        gudang_path = out_dir / f"gudang_subdistrict_topology.{extension}"
        comparison_path = out_dir / f"ingolstadt21_vs_gudang_topology.{extension}"
        _plot_single_topology(
            ingolstadt_roads,
            ingolstadt_signals,
            ingolstadt_path,
            bounds=None,
            aspect_ratio=1.65,
            line_width=1.6,
            line_alpha=0.78,
        )
        _plot_single_topology(
            gudang.roads,
            gudang.signal_points,
            gudang_path,
            bounds=gudang.bounds,
            aspect_ratio=1.65,
            line_width=1.05,
            line_alpha=0.70,
        )
        _plot_topology_pair(
            ingolstadt_roads,
            ingolstadt_signals,
            None,
            gudang.roads,
            gudang.signal_points,
            gudang.bounds,
            comparison_path,
            aspect_ratio=2.8,
        )
        written[f"ingolstadt21_{extension}"] = ingolstadt_path
        written[f"gudang_{extension}"] = gudang_path
        written[f"comparison_{extension}"] = comparison_path

    metadata = {
        "ingolstadt21": {
            "source": str(Path(ingolstadt_net)),
            "road_polylines": len(ingolstadt_roads),
            "points": sum(len(road) for road in ingolstadt_roads),
            "signalized_intersections": len(ingolstadt_signals),
        },
        "gudang_subdistrict": {
            **gudang.metadata,
            "road_polylines": len(gudang.roads),
            "points": sum(len(road) for road in gudang.roads),
            "signalized_intersections": len(gudang.signal_points),
        },
        "style": {
            "background": DESIGN_STYLE["card"],
            "road": DESIGN_STYLE["ink"],
            "signalized_intersection": DESIGN_STYLE["orange"],
        },
    }
    metadata_path = out_dir / "topology_comparison_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    written["metadata"] = metadata_path
    return written


def _sumo_topology(net_file: Path) -> tuple[list[list[Point]], list[Point]]:
    root = ET.parse(str(net_file)).getroot()
    junction_positions = _sumo_junction_positions(root)
    roads: list[list[Point]] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        shape: list[Point] = []
        for lane in edge.findall("lane"):
            shape = _parse_shape(lane.get("shape"))
            if shape:
                break
        if len(shape) >= 2:
            roads.append(shape)
    if not roads:
        raise ValueError(f"No road polylines found in {net_file}.")
    signal_points = _sumo_signal_points(root, junction_positions)
    return roads, signal_points


def _sumo_junction_positions(root: ET.Element) -> dict[str, Point]:
    positions: dict[str, Point] = {}
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id or junction.get("type") == "internal":
            continue
        try:
            positions[junction_id] = (float(junction.get("x") or 0.0), float(junction.get("y") or 0.0))
        except ValueError:
            continue
    return positions


def _sumo_signal_points(root: ET.Element, positions: dict[str, Point]) -> list[Point]:
    edge_endpoints = {
        str(edge.get("id")): (str(edge.get("from")), str(edge.get("to")))
        for edge in root.findall("edge")
        if edge.get("id")
        and edge.get("function") != "internal"
        and edge.get("from") in positions
        and edge.get("to") in positions
    }
    signal_points: list[Point] = []
    for tls in root.findall("tlLogic"):
        tls_id = str(tls.get("id") or "")
        junction_id = tls_id if tls_id in positions else _infer_controlled_junction(root, tls_id, positions, edge_endpoints)
        if junction_id and junction_id in positions:
            signal_points.append(positions[junction_id])
    unique = []
    seen = set()
    for point in signal_points:
        key = (round(point[0], 3), round(point[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _infer_controlled_junction(
    root: ET.Element,
    tls_id: str,
    positions: dict[str, Point],
    edge_endpoints: dict[str, tuple[str, str]],
) -> str | None:
    counts: dict[str, int] = {}
    for connection in root.findall("connection"):
        if connection.get("tl") != tls_id:
            continue
        from_edge = edge_endpoints.get(str(connection.get("from") or ""))
        to_edge = edge_endpoints.get(str(connection.get("to") or ""))
        if from_edge is not None and from_edge[1] in positions:
            counts[from_edge[1]] = counts.get(from_edge[1], 0) + 1
        if to_edge is not None and to_edge[0] in positions:
            counts[to_edge[0]] = counts.get(to_edge[0], 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _cached_colight_hangzhou_roadnet(output_dir: Path) -> Path:
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / "roadnet_4_4.json"
    if not path.exists():
        urlretrieve(COLIGHT_HANGZHOU_ROADNET_URL, path)
    return path


def _osm_gudang_original_topology(cache_path: Path, *, gudang_place: str) -> TopologyRenderData:
    from pyproj import Transformer

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Gudang OSM cache not found at {cache_path}. "
            "Regenerate it with OSMnx/Overpass or pass --gudang-roadnet for the CityFlow roadnet."
        )

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    elements = list(data.get("elements") or [])
    node_lonlat = {
        int(element["id"]): (float(element["lon"]), float(element["lat"]))
        for element in elements
        if element.get("type") == "node" and "id" in element and "lon" in element and "lat" in element
    }
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    figure_bounds = _project_bounds(GUDANG_FIGURE_BOUNDS_WGS84, transformer)

    roads: list[list[Point]] = []
    for way in elements:
        if way.get("type") != "way":
            continue
        highway = str((way.get("tags") or {}).get("highway") or "")
        if highway not in OSM_HIGHWAY_TYPES:
            continue
        lonlat = [node_lonlat[node_id] for node_id in way.get("nodes", []) if node_id in node_lonlat]
        if len(lonlat) < 2 or not _line_intersects_wgs84_bounds(lonlat, GUDANG_FIGURE_BOUNDS_WGS84):
            continue
        roads.append([transformer.transform(lon, lat) for lon, lat in lonlat])
    if not roads:
        raise ValueError(f"No Gudang OSM road polylines found in {cache_path}.")

    raw_signals = [
        (float(element["lon"]), float(element["lat"]))
        for element in elements
        if element.get("type") == "node"
        and (element.get("tags") or {}).get("highway") == "traffic_signals"
        and _point_in_wgs84_bounds((float(element["lon"]), float(element["lat"])), GUDANG_FIGURE_BOUNDS_WGS84)
    ]
    raw_signal_points = [transformer.transform(lon, lat) for lon, lat in raw_signals]
    clustered_signals = _cluster_points(raw_signal_points, eps=60.0)
    selected_signal_points = _select_gudang_control_points(clustered_signals, transformer)
    selected_lonlat = [transformer.transform(x, y, direction="INVERSE") for x, y in selected_signal_points]

    return TopologyRenderData(
        roads=roads,
        signal_points=selected_signal_points,
        metadata={
            "source": "OpenStreetMap Overpass cache",
            "source_file": str(cache_path),
            "place_query": gudang_place,
            "figure_bounds_wgs84": GUDANG_FIGURE_BOUNDS_WGS84,
            "control_target_bounds_wgs84": GUDANG_CONTROL_BOUNDS_WGS84,
            "raw_osm_signal_nodes": len(raw_signal_points),
            "clustered_osm_signal_intersections": len(clustered_signals),
            "selected_signal_points_wgs84": [
                [round(float(lon), 7), round(float(lat), 7)] for lon, lat in selected_lonlat
            ],
            "signal_selection": "Nearest OSM traffic-signal clusters to the paper-visible 4x4 Gudang control footprint.",
        },
        bounds=figure_bounds,
    )


def _cityflow_topology(roadnet_path: Path, *, gudang_place: str) -> TopologyRenderData:
    data = json.loads(roadnet_path.read_text(encoding="utf-8"))
    roads: list[list[Point]] = []
    for road in data.get("roads") or []:
        points = [
            (float(point["x"]), float(point["y"]))
            for point in road.get("points", [])
            if "x" in point and "y" in point
        ]
        if len(points) >= 2:
            roads.append(points)
    if not roads:
        raise ValueError(f"No road polylines found in {roadnet_path}.")

    signal_points = []
    for intersection in data.get("intersections") or []:
        if intersection.get("virtual"):
            continue
        point = intersection.get("point") or {}
        if "x" in point and "y" in point:
            signal_points.append((float(point["x"]), float(point["y"])))
    return TopologyRenderData(
        roads=roads,
        signal_points=signal_points,
        metadata={
            "source": "CoLight Hangzhou 4x4 roadnet",
            "source_file": str(roadnet_path),
            "source_url": COLIGHT_HANGZHOU_ROADNET_URL,
            "place_query": gudang_place,
        },
    )


def _plot_single_topology(
    roads: list[list[Point]],
    signal_points: list[Point],
    output_path: Path,
    *,
    bounds: tuple[float, float, float, float] | None,
    aspect_ratio: float,
    line_width: float,
    line_alpha: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _set_matplotlib_style(font_manager)
    fig, ax = plt.subplots(
        figsize=(8.25, 8.25 / aspect_ratio),
        constrained_layout=True,
        facecolor=DESIGN_STYLE["card"],
    )
    ax.set_facecolor(DESIGN_STYLE["card"])
    _draw_roads(ax, roads, linewidth=line_width, alpha=line_alpha)
    _draw_signal_points(ax, signal_points)
    _finish_topology_axes(ax, roads, signal_points, bounds=bounds)
    fig.savefig(output_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_topology_pair(
    left_roads: list[list[Point]],
    left_signal_points: list[Point],
    left_bounds: tuple[float, float, float, float] | None,
    right_roads: list[list[Point]],
    right_signal_points: list[Point],
    right_bounds: tuple[float, float, float, float] | None,
    output_path: Path,
    *,
    aspect_ratio: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _set_matplotlib_style(font_manager)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 11.2 / aspect_ratio),
        constrained_layout=True,
        facecolor=DESIGN_STYLE["card"],
    )
    for ax, roads, signals, bounds, width, alpha in (
        (axes[0], left_roads, left_signal_points, left_bounds, 1.45, 0.78),
        (axes[1], right_roads, right_signal_points, right_bounds, 0.95, 0.70),
    ):
        ax.set_facecolor(DESIGN_STYLE["card"])
        _draw_roads(ax, roads, linewidth=width, alpha=alpha)
        _draw_signal_points(ax, signals)
        _finish_topology_axes(ax, roads, signals, bounds=bounds)
    fig.savefig(output_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_roads(ax, roads: list[list[Point]], *, linewidth: float, alpha: float) -> None:
    for road in roads:
        if len(road) < 2:
            continue
        xs = [point[0] for point in road]
        ys = [point[1] for point in road]
        ax.plot(
            xs,
            ys,
            color=DESIGN_STYLE["ink"],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def _draw_signal_points(ax, points: list[Point]) -> None:
    if not points:
        return
    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=52,
        marker="o",
        facecolor=DESIGN_STYLE["orange"],
        edgecolor=DESIGN_STYLE["paper"],
        linewidth=1.15,
        alpha=0.96,
        zorder=5,
    )


def _finish_topology_axes(
    ax,
    roads: list[list[Point]],
    signal_points: list[Point],
    *,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    min_x, max_x, min_y, max_y = bounds if bounds is not None else _bounds(roads, signal_points)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    pad = 0.055 * max(width, height)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _bounds(roads: list[list[Point]], signal_points: list[Point]) -> tuple[float, float, float, float]:
    points = [point for road in roads for point in road] + list(signal_points)
    if not points:
        raise ValueError("Cannot plot topology without points.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _project_bounds(bounds_wgs84: tuple[float, float, float, float], transformer) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    corners = [
        transformer.transform(min_lon, min_lat),
        transformer.transform(min_lon, max_lat),
        transformer.transform(max_lon, min_lat),
        transformer.transform(max_lon, max_lat),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), max(xs), min(ys), max(ys)


def _line_intersects_wgs84_bounds(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> bool:
    min_lon, min_lat, max_lon, max_lat = bounds
    line_min_lon = min(lon for lon, _lat in points)
    line_max_lon = max(lon for lon, _lat in points)
    line_min_lat = min(lat for _lon, lat in points)
    line_max_lat = max(lat for _lon, lat in points)
    return not (
        line_max_lon < min_lon
        or line_min_lon > max_lon
        or line_max_lat < min_lat
        or line_min_lat > max_lat
    )


def _point_in_wgs84_bounds(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    lon, lat = point
    min_lon, min_lat, max_lon, max_lat = bounds
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _cluster_points(points: list[Point], *, eps: float) -> list[Point]:
    unused = set(range(len(points)))
    centers: list[Point] = []
    while unused:
        seed = unused.pop()
        group = {seed}
        changed = True
        while changed:
            changed = False
            for index in list(unused):
                if any(_distance(points[index], points[member]) <= eps for member in group):
                    unused.remove(index)
                    group.add(index)
                    changed = True
        xs = [points[index][0] for index in group]
        ys = [points[index][1] for index in group]
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centers


def _select_gudang_control_points(signal_centers: list[Point], transformer) -> list[Point]:
    if len(signal_centers) < 16:
        raise ValueError(f"Expected at least 16 Gudang OSM signal clusters, found {len(signal_centers)}.")
    min_lon, min_lat, max_lon, max_lat = GUDANG_CONTROL_BOUNDS_WGS84
    target_points = [
        transformer.transform(
            min_lon + (max_lon - min_lon) * col / 3.0,
            max_lat - (max_lat - min_lat) * row / 3.0,
        )
        for row in range(4)
        for col in range(4)
    ]
    selected: list[Point] = []
    remaining = list(signal_centers)
    for target in target_points:
        point = min(remaining, key=lambda candidate: _distance(candidate, target))
        selected.append(point)
        remaining.remove(point)
    return selected


def _distance(first: Point, second: Point) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _set_matplotlib_style(font_manager) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": [_available_font(font_manager)],
            "figure.facecolor": DESIGN_STYLE["card"],
            "axes.facecolor": DESIGN_STYLE["card"],
            "savefig.facecolor": DESIGN_STYLE["card"],
            "savefig.edgecolor": DESIGN_STYLE["card"],
        }
    )


def _available_font(font_manager) -> str:
    regular_name = None
    for path in LOCAL_FONT_PATHS:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            if "Regular" in path.name:
                regular_name = font_manager.FontProperties(fname=str(path)).get_name()
    if regular_name:
        return regular_name
    for family in FONT_FAMILY:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _parse_shape(raw: str | None) -> list[Point]:
    if not raw:
        return []
    points: list[Point] = []
    for item in raw.split():
        if "," not in item:
            continue
        x, y = item.split(",", 1)
        try:
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render topology-only Ingolstadt21 and Gudang comparison figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ingolstadt-net", type=Path, default=DEFAULT_INGOLSTADT_NET)
    parser.add_argument("--gudang-roadnet", type=Path, default=None)
    parser.add_argument("--gudang-osm-cache", type=Path, default=DEFAULT_GUDANG_OSM_CACHE)
    parser.add_argument("--gudang-place", default=GUDANG_PLACE)
    parser.add_argument("--format", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = render_topology_comparison(
        output_dir=args.output_dir,
        ingolstadt_net=args.ingolstadt_net,
        gudang_roadnet=args.gudang_roadnet,
        gudang_osm_cache=args.gudang_osm_cache,
        gudang_place=args.gudang_place,
        formats=tuple(args.format or ["png", "pdf"]),
    )
    for key, path in sorted(written.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
