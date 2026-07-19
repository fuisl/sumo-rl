from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_ROOT = ROOT / "visualization" / "outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "mfd"
DEFAULT_FIGSIZE = (8.8, 5.8)
DEFAULT_INGOLSTADT21_PPO_DENSITY_FLOW_STEM = "resco_ingolstadt21_mfd_density_flow_fgsv3_ppo_vs_ppo"
FONT_FAMILY = ["Liberitus Sans", "Libertinus Sans", "Liberation Sans", "DejaVu Sans"]
LOCAL_FONT_PATHS = [
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Regular.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Bold.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Italic.otf",
    ROOT / "visualization" / "assets" / "fonts" / "LibertinusSans-Regular.ttf",
]
CONTROL_ORDER = [
    "fixed_time",
    "static_max_pressure",
    "dqn",
    "ppo",
    "sac_builtin",
    "colight",
    "fgs_mlp_gatv2_ppo",
    "fgsv3_frap_gatv2_ppo",
    "fgsv3_frap_gatv2_sac",
    "fgsv3_frap_gatv2_sac_ti4_big",
]
CONTROL_PALETTE = {
    "fixed_time": "#7A7A7A",
    "static_max_pressure": "#111111",
    "dqn": "#59A14F",
    "ppo": "#4C78A8",
    "sac_builtin": "#F58518",
    "colight": "#B279A2",
    "fgs_mlp_gatv2_ppo": "#2A9D8F",
    "fgsv3_frap_gatv2_ppo": "#1F77B4",
    "fgsv3_frap_gatv2_sac": "#D62728",
    "fgsv3_frap_gatv2_sac_ti4_big": "#9467BD",
}
CONTROL_DISPLAY_NAMES = {
    "fixed_time": "FixedTime",
    "static_max_pressure": "Static max pressure",
    "dqn": "DQN",
    "ppo": "PPO",
    "sac_builtin": "SAC",
    "colight": "CoLight",
    "fgs_mlp_gatv2_ppo": "PPO original",
    "fgsv3_frap_gatv2_ppo": "FGSv3 PPO",
    "fgsv3_frap_gatv2_sac": "FGSv3 SAC",
    "fgsv3_frap_gatv2_sac_ti4_big": "FGSv3 SAC TI4",
}
CONTROL_MARKERS = {
    "fixed_time": "o",
    "static_max_pressure": "X",
    "dqn": "^",
    "ppo": "D",
    "sac_builtin": "v",
    "colight": "p",
    "fgs_mlp_gatv2_ppo": "X",
    "fgsv3_frap_gatv2_ppo": "s",
    "fgsv3_frap_gatv2_sac": "P",
    "fgsv3_frap_gatv2_sac_ti4_big": "D",
}
DESIGN_STYLE = {
    "ink": "#141413",
    "paper": "#faf9f5",
    "card": "#e8e6dc",
    "card_alt": "#f1efe7",
    "grid": "#faf9f5",
    "mid_gray": "#b0aea5",
    "orange": "#d97757",
    "blue": "#6a9bcc",
    "green": "#788c5d",
}
INGOLSTADT21_PPO_DENSITY_FLOW_TRACES = [
    (
        "FGSv3 PPO",
        DEFAULT_INPUT_ROOT / "resco_ingolstadt21__fgsv3_frap_gatv2_ppo" / "trip_trace.json",
        DESIGN_STYLE["orange"],
        CONTROL_MARKERS["fgsv3_frap_gatv2_ppo"],
    ),
    (
        "PPO",
        DEFAULT_INPUT_ROOT / "resco_ingolstadt21__fgs_mlp_gatv2_ppo" / "trip_trace.json",
        DESIGN_STYLE["blue"],
        CONTROL_MARKERS["fgs_mlp_gatv2_ppo"],
    ),
]


@dataclass(frozen=True)
class NetworkScale:
    lane_length_km: float
    lane_count: int
    source: str


def discover_trace_paths(input_root: str | Path = DEFAULT_INPUT_ROOT) -> list[Path]:
    root = Path(input_root)
    return sorted(path for path in root.glob("*/trip_trace.json") if path.is_file())


def build_mfd_rows(trace_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        path = Path(trace_path)
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_rows_for_trace(trace, path))
    return rows


def write_mfd_plots(
    trace_paths: Iterable[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    min_warmup_seconds: float = 0.0,
    bins: int = 24,
) -> dict[str, Path]:
    rows = build_mfd_rows(trace_paths)
    if not rows:
        raise ValueError("No trace rows found. Generate trip_trace.json files before plotting MFD diagrams.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    frame = pd.DataFrame(rows)
    if min_warmup_seconds > 0:
        frame = frame[frame["elapsed_seconds"] >= float(min_warmup_seconds)].copy()
    if frame.empty:
        raise ValueError("All MFD rows were filtered out by --min-warmup-seconds.")

    rows_path = out_dir / "mfd_rows.csv"
    frame.to_csv(rows_path, index=False)

    metadata_path = out_dir / "mfd_metadata.json"
    metadata = _summary_metadata(frame)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    written: dict[str, Path] = {
        "rows": rows_path,
        "metadata": metadata_path,
    }
    for scenario, scenario_frame in sorted(frame.groupby("scenario"), key=lambda item: str(item[0])):
        slug = _slugify(str(scenario))
        written.update(
            _plot_scenario_mfd(
                scenario_frame.copy(),
                scenario=str(scenario),
                output_dir=out_dir,
                stem=f"{slug}_mfd",
                formats=tuple(formats),
                bins=bins,
            )
        )

    if frame["scenario"].nunique() > 1:
        written.update(
            _plot_scenario_mfd(
                frame.copy(),
                scenario="all_scenarios",
                output_dir=out_dir,
                stem="all_scenarios_mfd",
                formats=tuple(formats),
                bins=bins,
            )
        )
    return written


def write_ingolstadt21_ppo_density_flow_comparison(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    output_stem: str = DEFAULT_INGOLSTADT21_PPO_DENSITY_FLOW_STEM,
    formats: Iterable[str] = ("png", "pdf"),
    bins: int = 24,
) -> dict[str, Path]:
    import pandas as pd

    specs = [spec for spec in INGOLSTADT21_PPO_DENSITY_FLOW_TRACES if spec[1].exists()]
    if not specs:
        raise ValueError("No Ingolstadt21 PPO comparison traces were found.")

    trace_paths = [trace_path for _label, trace_path, _color, _marker in specs]
    rows = build_mfd_rows(trace_paths)
    if not rows:
        raise ValueError("No density-flow rows could be built from the Ingolstadt21 PPO traces.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)
    label_by_path = {str(trace_path): label for label, trace_path, _color, _marker in specs}
    color_by_label = {label: color for label, _trace_path, color, _marker in specs}
    marker_by_label = {label: marker for label, _trace_path, _color, marker in specs}
    frame = frame[frame["trace_path"].isin(label_by_path)].copy()
    frame["control_display"] = frame["trace_path"].map(label_by_path)
    frame["color"] = frame["control_display"].map(color_by_label)
    frame["marker"] = frame["control_display"].map(marker_by_label)

    plot_frame = frame.copy()
    plot_frame["control"] = plot_frame["control_display"]
    binned = _binned_means(
        plot_frame,
        x_col="lane_density_veh_per_km",
        y_col="lane_flow_veh_per_hour",
        bins=bins,
    )
    if not binned.empty:
        binned["control_display"] = binned["control"]
        binned["color"] = binned["control_display"].map(color_by_label)
        binned["marker"] = binned["control_display"].map(marker_by_label)

    rows_path = out_dir / f"{output_stem}_rows.csv"
    binned_path = out_dir / f"{output_stem}_binned.csv"
    frame.to_csv(rows_path, index=False)
    binned.to_csv(binned_path, index=False)

    written: dict[str, Path] = {
        "rows": rows_path,
        "binned": binned_path,
    }
    for fmt in formats:
        path = out_dir / f"{output_stem}.{fmt.lstrip('.')}"
        _plot_design_density_flow_comparison(frame, binned, path)
        written[fmt.lstrip(".")] = path
    return written


def _rows_for_trace(trace: dict[str, Any], trace_path: Path) -> list[dict[str, Any]]:
    metadata = dict(trace.get("metadata") or {})
    network = dict(trace.get("network") or {})
    frames = list(trace.get("frames") or [])
    if not frames:
        return []

    scale = _network_scale(network)
    scenario = str(metadata.get("scenario") or _infer_scenario(trace_path))
    control = _infer_control_label(trace_path, metadata)
    algorithm_kind = str(metadata.get("algorithm_kind") or control)
    seed = metadata.get("seed")
    first_time = min(float(frame.get("time", 0.0) or 0.0) for frame in frames)

    rows: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        vehicles = list(frame.get("vehicles") or [])
        speeds = [max(0.0, float(vehicle.get("speed", 0.0) or 0.0)) for vehicle in vehicles]
        vehicle_count = len(speeds)
        speed_sum_mps = float(sum(speeds))
        mean_speed_mps = speed_sum_mps / vehicle_count if vehicle_count else 0.0
        density = vehicle_count / scale.lane_length_km if scale.lane_length_km > 0 else math.nan
        mean_speed_kmh = mean_speed_mps * 3.6
        rows.append(
            {
                "trace_path": str(trace_path),
                "scenario": scenario,
                "control": control,
                "algorithm_kind": algorithm_kind,
                "seed": seed,
                "frame_index": frame_index,
                "time_seconds": float(frame.get("time", 0.0) or 0.0),
                "elapsed_seconds": float(frame.get("time", 0.0) or 0.0) - first_time,
                "vehicle_count": vehicle_count,
                "stopped_count": sum(1 for speed in speeds if speed <= 0.1),
                "mean_speed_mps": mean_speed_mps,
                "mean_speed_kmh": mean_speed_kmh,
                "lane_length_km": scale.lane_length_km,
                "lane_count": scale.lane_count,
                "lane_density_veh_per_km": density,
                "lane_flow_veh_per_hour": density * mean_speed_kmh if math.isfinite(density) else math.nan,
                "production_veh_km_per_hour": speed_sum_mps * 3.6,
                "network_scale_source": scale.source,
            }
        )
    return rows


def _network_scale(network: dict[str, Any]) -> NetworkScale:
    net_file = network.get("net_file")
    if net_file:
        path = Path(str(net_file))
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            scale = _sumo_net_lane_scale(path)
            if scale.lane_length_km > 0:
                return scale

    polyline_length_m = 0.0
    for road in network.get("road_polylines") or []:
        points = [(float(point[0]), float(point[1])) for point in road if len(point) >= 2]
        for start, end in zip(points, points[1:]):
            polyline_length_m += math.dist(start, end)
    return NetworkScale(
        lane_length_km=max(polyline_length_m / 1000.0, 1e-9),
        lane_count=0,
        source="road_polylines",
    )


def _sumo_net_lane_scale(net_file: Path) -> NetworkScale:
    lane_length_m = 0.0
    lane_count = 0
    for _event, element in ET.iterparse(net_file, events=("end",)):
        if element.tag != "edge":
            continue
        edge_id = str(element.attrib.get("id", ""))
        edge_function = str(element.attrib.get("function", ""))
        if edge_id.startswith(":") or edge_function == "internal":
            element.clear()
            continue
        for lane in element.findall("lane"):
            try:
                lane_length_m += float(lane.attrib.get("length", "0") or 0.0)
                lane_count += 1
            except ValueError:
                continue
        element.clear()
    return NetworkScale(lane_length_km=lane_length_m / 1000.0, lane_count=lane_count, source=str(net_file))


def _plot_scenario_mfd(
    frame: Any,
    *,
    scenario: str,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    bins: int,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = [_available_font(font_manager)]
    controls = _ordered_controls(frame["control"].dropna().unique())
    display_order = [_control_display_name(control) for control in controls]
    frame["control_display"] = frame["control"].map(_control_display_name)
    palette = {
        _control_display_name(control): CONTROL_PALETTE.get(control, "#555555")
        for control in controls
    }
    markers = {
        _control_display_name(control): CONTROL_MARKERS.get(control, "o")
        for control in controls
    }
    written: dict[str, Path] = {}
    specs = [
        (
            "density_flow",
            "lane_density_veh_per_km",
            "lane_flow_veh_per_hour",
            "Lane density (veh/km/lane)",
            "Lane flow (veh/h/lane)",
        ),
        (
            "accumulation_production",
            "vehicle_count",
            "production_veh_km_per_hour",
            "Accumulation (vehicles)",
            "Network production (veh-km/h)",
        ),
    ]
    for suffix, x_col, y_col, x_label, y_label in specs:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True)
        sns.scatterplot(
            data=frame,
            x=x_col,
            y=y_col,
            hue="control_display",
            style="control_display",
            palette=palette,
            markers=markers,
            hue_order=display_order,
            style_order=display_order,
            alpha=0.32,
            s=28,
            linewidth=0,
            ax=ax,
        )
        binned = _binned_means(frame, x_col=x_col, y_col=y_col, bins=bins)
        if not binned.empty:
            binned["control_display"] = binned["control"].map(_control_display_name)
            sns.lineplot(
                data=binned,
                x=x_col,
                y=y_col,
                hue="control_display",
                style="control_display",
                palette=palette,
                markers=markers,
                dashes=False,
                hue_order=display_order,
                style_order=display_order,
                linewidth=2.0,
                legend=False,
                ax=ax,
            )
        _add_dense_inset(
            ax,
            frame,
            binned,
            scenario=scenario,
            suffix=suffix,
            x_col=x_col,
            y_col=y_col,
            palette=palette,
            markers=markers,
            display_order=display_order,
        )
        ax.set_title(f"{_display_name(scenario)} MFD")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(title="Control", loc="best", frameon=True)
        ax.margins(x=0.04, y=0.08)
        for fmt in formats:
            path = output_dir / f"{stem}_{suffix}.{fmt.lstrip('.')}"
            fig.savefig(path, dpi=220)
            written[f"{stem}_{suffix}_{fmt.lstrip('.')}"] = path
        plt.close(fig)
    return written


def _plot_design_density_flow_comparison(frame: Any, binned: Any, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update(
        {
            "font.family": [_available_font(font_manager)],
            "figure.facecolor": DESIGN_STYLE["card"],
            "axes.facecolor": DESIGN_STYLE["card"],
            "axes.edgecolor": DESIGN_STYLE["ink"],
            "axes.labelcolor": DESIGN_STYLE["ink"],
            "xtick.color": DESIGN_STYLE["ink"],
            "ytick.color": DESIGN_STYLE["ink"],
            "text.color": DESIGN_STYLE["ink"],
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, constrained_layout=True, facecolor=DESIGN_STYLE["card"])
    _style_design_mfd_axes(ax)
    _draw_design_density_flow_layers(ax, frame, binned, include_labels=True)
    _add_design_density_flow_inset(ax, frame, binned)
    ax.set_xlabel("Lane density (veh/km/lane)")
    ax.set_ylabel("Lane flow (veh/h/lane)")
    legend = ax.legend(title="Control", loc="upper right", frameon=True, borderpad=0.8, labelspacing=0.7)
    legend.get_frame().set_facecolor(DESIGN_STYLE["card_alt"])
    legend.get_frame().set_edgecolor(DESIGN_STYLE["ink"])
    legend.get_frame().set_linewidth(0.8)
    ax.margins(x=0.04, y=0.08)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _style_design_mfd_axes(ax: Any) -> None:
    ax.grid(True, color=DESIGN_STYLE["grid"], linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(DESIGN_STYLE["ink"])
        spine.set_linewidth(0.9)


def _draw_design_density_flow_layers(ax: Any, frame: Any, binned: Any, *, include_labels: bool) -> None:
    for control, group in frame.groupby("control_display", sort=False):
        group = group.sort_values("elapsed_seconds")
        color = str(group["color"].iloc[0])
        marker = str(group["marker"].iloc[0])
        ax.scatter(
            group["lane_density_veh_per_km"],
            group["lane_flow_veh_per_hour"],
            color=color,
            marker=marker,
            s=28,
            alpha=0.32,
            edgecolor="none",
            linewidth=0,
            label=str(control) if include_labels else None,
            zorder=2,
        )
    if binned.empty:
        return
    for control, group in binned.groupby("control_display", sort=False):
        group = group.sort_values("lane_density_veh_per_km")
        color = str(group["color"].iloc[0])
        marker = str(group["marker"].iloc[0])
        ax.plot(
            group["lane_density_veh_per_km"],
            group["lane_flow_veh_per_hour"],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6.6,
            markeredgecolor=DESIGN_STYLE["paper"],
            markeredgewidth=1.0,
            label=None,
            zorder=5,
        )


def _add_design_density_flow_inset(ax: Any, frame: Any, binned: Any) -> None:
    from matplotlib.patches import Rectangle

    x_col = "lane_density_veh_per_km"
    y_col = "lane_flow_veh_per_hour"
    x_min, x_max = 0.9, 3.0
    y_min = 35.0
    focus_frame = frame[frame[x_col].between(x_min, x_max)]
    if focus_frame.empty:
        return
    y_max = min(90.0, max(y_min + 8.0, float(focus_frame[y_col].max()) + 4.0))
    y_max = min(90.0, math.ceil(y_max / 5.0) * 5.0)
    dense_frame = focus_frame[focus_frame[y_col].between(y_min, y_max)]
    if dense_frame.empty:
        return

    inset = ax.inset_axes([0.51, 0.48, 0.31, 0.34])
    inset.set_facecolor(DESIGN_STYLE["card"])
    _style_design_mfd_axes(inset)
    _draw_design_density_flow_layers(inset, frame, binned, include_labels=False)
    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min, y_max)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.tick_params(labelsize=8)
    for spine in inset.spines.values():
        spine.set_color(DESIGN_STYLE["blue"])
        spine.set_linewidth(1.0)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=DESIGN_STYLE["blue"],
            linewidth=1.0,
            alpha=0.68,
            clip_on=True,
            zorder=4,
        )
    )


def _add_dense_inset(
    ax: Any,
    frame: Any,
    binned: Any,
    *,
    scenario: str,
    suffix: str,
    x_col: str,
    y_col: str,
    palette: dict[str, str],
    markers: dict[str, str],
    display_order: list[str],
) -> None:
    if str(scenario) != "resco_ingolstadt21":
        return

    dense_limits = {
        "density_flow": ((0.9, 3.0), (35.0, 90.0)),
        "accumulation_production": ((130.0, 460.0), (5200.0, 13250.0)),
    }
    limits = dense_limits.get(suffix)
    if limits is None:
        return

    (x_min, x_max), (y_min, y_max) = limits
    dense_frame = frame[
        frame[x_col].between(x_min, x_max)
        & frame[y_col].between(y_min, y_max)
    ]
    if dense_frame.empty:
        return

    import seaborn as sns

    inset_bounds = {
        "density_flow": (7.0, 38.0, 4.8, 36.0),
        "accumulation_production": (980.0, 5600.0, 650.0, 5600.0),
    }
    inset = ax.inset_axes(inset_bounds[suffix], transform=ax.transData)
    sns.scatterplot(
        data=frame,
        x=x_col,
        y=y_col,
        hue="control_display",
        style="control_display",
        palette=palette,
        markers=markers,
        hue_order=display_order,
        style_order=display_order,
        alpha=0.26,
        s=20,
        linewidth=0,
        legend=False,
        ax=inset,
    )
    if not binned.empty:
        sns.lineplot(
            data=binned,
            x=x_col,
            y=y_col,
            hue="control_display",
            style="control_display",
            palette=palette,
            markers=markers,
            dashes=False,
            hue_order=display_order,
            style_order=display_order,
            linewidth=1.7,
            legend=False,
            ax=inset,
        )
    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min, y_max)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.tick_params(labelsize=8)
    inset.grid(True, linewidth=0.7, alpha=0.55)
    ax.indicate_inset_zoom(inset, edgecolor="#525252", alpha=0.65)


def _available_font(font_manager: Any) -> str:
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


def _binned_means(frame: Any, *, x_col: str, y_col: str, bins: int) -> Any:
    import pandas as pd

    grouped_frames = []
    for control, control_frame in frame.groupby("control"):
        control_frame = control_frame[[x_col, y_col]].dropna().copy()
        if control_frame.empty:
            continue
        unique_x = control_frame[x_col].nunique()
        if unique_x <= 1:
            control_frame["control"] = control
            grouped_frames.append(control_frame.groupby("control", as_index=False).mean(numeric_only=True))
            continue
        control_frame["_bin"] = pd.cut(control_frame[x_col], bins=min(int(bins), int(unique_x)), duplicates="drop")
        means = control_frame.groupby("_bin", observed=True).agg({x_col: "mean", y_col: "mean"}).reset_index(drop=True)
        means["control"] = control
        grouped_frames.append(means)
    if not grouped_frames:
        return pd.DataFrame(columns=["control", x_col, y_col])
    return pd.concat(grouped_frames, ignore_index=True).sort_values(["control", x_col])


def _summary_metadata(frame: Any) -> dict[str, Any]:
    grouped = frame.groupby(["scenario", "control"], dropna=False)
    traces = []
    for (scenario, control), group in grouped:
        traces.append(
            {
                "scenario": str(scenario),
                "control": str(control),
                "frames": int(len(group)),
                "trace_paths": sorted(str(path) for path in group["trace_path"].dropna().unique()),
                "lane_length_km": float(group["lane_length_km"].iloc[0]),
                "lane_count": int(group["lane_count"].iloc[0]),
                "max_vehicle_count": int(group["vehicle_count"].max()),
                "max_lane_flow_veh_per_hour": float(group["lane_flow_veh_per_hour"].max()),
                "max_production_veh_km_per_hour": float(group["production_veh_km_per_hour"].max()),
            }
        )
    return {
        "formula": {
            "lane_density_veh_per_km": "vehicle_count / total_non_internal_lane_length_km",
            "lane_flow_veh_per_hour": "lane_density_veh_per_km * mean_speed_kmh",
            "production_veh_km_per_hour": "sum(vehicle_speed_mps) * 3.6",
        },
        "traces": traces,
    }


def _infer_scenario(trace_path: Path) -> str:
    name = trace_path.parent.name
    return name.split("__", 1)[0] if "__" in name else name


def _infer_control_label(trace_path: Path, metadata: dict[str, Any]) -> str:
    name = trace_path.parent.name
    if "__" in name:
        return name.split("__", 1)[1]
    return str(metadata.get("algorithm_kind") or name)


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def _display_name(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _control_display_name(value: Any) -> str:
    control = str(value)
    return CONTROL_DISPLAY_NAMES.get(control, _display_name(control))


def _ordered_controls(values: Iterable[Any]) -> list[str]:
    controls = [str(value) for value in values]
    known = [control for control in CONTROL_ORDER if control in controls]
    extra = sorted(control for control in controls if control not in CONTROL_ORDER)
    return known + extra


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create seaborn MFD diagrams from visualization trip traces.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Directory containing */trip_trace.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for MFD plots and CSV rows.")
    parser.add_argument(
        "--ingolstadt21-ppo-density-flow",
        action="store_true",
        help="Render only the Ingolstadt21 FGSv3 PPO vs PPO density-flow comparison.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_INGOLSTADT21_PPO_DENSITY_FLOW_STEM,
        help="Output stem for --ingolstadt21-ppo-density-flow.",
    )
    parser.add_argument("--trace", action="append", type=Path, default=None, help="Explicit trip_trace.json path. May repeat.")
    parser.add_argument("--format", action="append", default=None, help="Output format such as png or pdf. May repeat.")
    parser.add_argument("--min-warmup-seconds", type=float, default=0.0, help="Drop early frames before plotting.")
    parser.add_argument("--bins", type=int, default=24, help="Number of bins for the smoothed MFD line.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formats = tuple(args.format or ["png", "pdf"])
    if args.ingolstadt21_ppo_density_flow:
        written = write_ingolstadt21_ppo_density_flow_comparison(
            args.output_dir,
            output_stem=args.output_stem,
            formats=formats,
            bins=args.bins,
        )
    else:
        trace_paths = args.trace if args.trace else discover_trace_paths(args.input_root)
        written = write_mfd_plots(
            trace_paths,
            args.output_dir,
            formats=formats,
            min_warmup_seconds=args.min_warmup_seconds,
            bins=args.bins,
        )
    for key, path in sorted(written.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
