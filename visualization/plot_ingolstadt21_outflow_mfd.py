from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.plot_mfd import _available_font


OUTPUT_DIR = ROOT / "visualization" / "outputs" / "mfd"
DEFAULT_OUTPUT_STEM = "resco_ingolstadt21_density_outflow_selected_controls"
DEFAULT_PPO_COMPARISON_STEM = "resco_ingolstadt21_mfd_fgsv3_ppo_vs_ppo"
FONT_FAMILY = ["Liberitus Sans", "Libertinus Sans", "Liberation Sans", "DejaVu Sans"]
FIGSIZE = (10.0, 6.2)
DESIGN_STYLE = {
    "ink": "#141413",
    "paper": "#faf9f5",
    "card": "#e8e6dc",
    "card_alt": "#f1efe7",
    "mid_gray": "#b0aea5",
    "orange": "#d97757",
    "blue": "#6a9bcc",
    "green": "#788c5d",
}
CONTROL_SPECS = [
    (
        "FGSv3 PPO",
        ROOT / "visualization/outputs/resco_ingolstadt21__fgsv3_frap_gatv2_ppo/trip_trace.json",
        "#1F77B4",
        "o",
    ),
    (
        "FGSv3 SAC",
        ROOT / "visualization/outputs/resco_ingolstadt21__fgsv3_frap_gatv2_sac/trip_trace.json",
        "#D62728",
        "+",
    ),
    (
        "FixedTime",
        ROOT / "visualization/outputs/resco_ingolstadt21__fixed_time/trip_trace.json",
        "#7A7A7A",
        "x",
    ),
    (
        "DQN",
        ROOT / "visualization/outputs/resco_ingolstadt21__dqn/trip_trace.json",
        "#59A14F",
        "^",
    ),
]
PPO_COMPARISON_SPECS = [
    (
        "FGSv3 PPO",
        ROOT / "visualization/outputs/resco_ingolstadt21__fgsv3_frap_gatv2_ppo/trip_trace.json",
        DESIGN_STYLE["blue"],
        "o",
    ),
    (
        "PPO",
        ROOT / "visualization/outputs/resco_ingolstadt21__fgs_mlp_gatv2_ppo/trip_trace.json",
        DESIGN_STYLE["orange"],
        "s",
    ),
]


def render_ingolstadt21_outflow_mfd(
    output_dir: str | Path = OUTPUT_DIR,
    *,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    bin_seconds: float = 300.0,
    control_specs: list[tuple[str, Path, str, str]] | None = None,
) -> dict[str, Path]:
    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    specs = CONTROL_SPECS if control_specs is None else control_specs
    for label, trace_path, color, marker in specs:
        if not trace_path.exists():
            continue
        points, summaries = _rows_for_control(label, trace_path, color, marker, bin_seconds=bin_seconds)
        point_rows.extend(points)
        summary_rows.extend(summaries)
    if not point_rows:
        raise ValueError("No Ingolstadt21 traces were found for the selected controls.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_csv_path = out_dir / f"{output_stem}_points.csv"
    summary_csv_path = out_dir / f"{output_stem}_summary.csv"

    import pandas as pd

    points = pd.DataFrame(point_rows)
    summary = pd.DataFrame(summary_rows)
    points.to_csv(points_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    _plot(points, summary, png_path, pdf_path)
    return {"png": png_path, "pdf": pdf_path, "points_csv": points_csv_path, "summary_csv": summary_csv_path}


def render_ingolstadt21_ppo_comparison_mfd(
    output_dir: str | Path = OUTPUT_DIR,
    *,
    output_stem: str = DEFAULT_PPO_COMPARISON_STEM,
    bin_seconds: float = 300.0,
) -> dict[str, Path]:
    return render_ingolstadt21_outflow_mfd(
        output_dir,
        output_stem=output_stem,
        bin_seconds=bin_seconds,
        control_specs=PPO_COMPARISON_SPECS,
    )


def _rows_for_control(
    label: str,
    trace_path: Path,
    color: str,
    marker: str,
    *,
    bin_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    frames = list(trace.get("frames") or [])
    if not frames:
        return []
    network = dict(trace.get("network") or {})
    road_length_km = _road_length_km(Path(str(network["net_file"])))
    metadata_path = trace_path.parent / "trip_animation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrivals = _arrival_times(Path(str(metadata["tripinfo_path"])))

    start_time = min(float(frame.get("time", 0.0) or 0.0) for frame in frames)
    end_time = max(float(frame.get("time", 0.0) or 0.0) for frame in frames)
    point_rows = []
    summary_rows = []
    bin_index = 0
    current = start_time
    while current < end_time:
        next_time = min(current + bin_seconds, end_time + 1e-9)
        bin_frames = [
            frame
            for frame in frames
            if current <= float(frame.get("time", 0.0) or 0.0) < next_time
        ]
        if not bin_frames:
            current = next_time
            bin_index += 1
            continue
        vehicle_count = sum(len(frame.get("vehicles") or []) for frame in bin_frames) / len(bin_frames)
        arrived = sum(1 for arrival in arrivals if current <= arrival < next_time)
        duration = max(next_time - current, 1e-9)
        outflow = arrived * 3600.0 / duration
        summary_rows.append(
            {
                "control": label,
                "trace_path": str(trace_path),
                "bin_index": bin_index,
                "bin_start": current,
                "bin_end": next_time,
                "mean_vehicle_count": vehicle_count,
                "vehicle_density_v_per_km": vehicle_count / road_length_km,
                "network_outflow_veh_per_h": outflow,
                "arrived_count": arrived,
                "road_length_km": road_length_km,
                "color": color,
                "marker": marker,
            }
        )
        for frame in bin_frames:
            time = float(frame.get("time", 0.0) or 0.0)
            count = len(frame.get("vehicles") or [])
            point_rows.append(
                {
                    "control": label,
                    "trace_path": str(trace_path),
                    "bin_index": bin_index,
                    "time": time,
                    "vehicle_count": count,
                    "vehicle_density_v_per_km": count / road_length_km,
                    "network_outflow_veh_per_h": outflow,
                    "arrived_count": arrived,
                    "road_length_km": road_length_km,
                    "color": color,
                    "marker": marker,
                }
            )
        current = next_time
        bin_index += 1
    return point_rows, summary_rows


def _plot(points: Any, summary: Any, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update(
        {
            "font.family": [_available_font(font_manager)],
            "figure.facecolor": DESIGN_STYLE["paper"],
            "axes.facecolor": DESIGN_STYLE["paper"],
            "axes.edgecolor": DESIGN_STYLE["ink"],
            "axes.labelcolor": DESIGN_STYLE["ink"],
            "xtick.color": DESIGN_STYLE["ink"],
            "ytick.color": DESIGN_STYLE["ink"],
            "text.color": DESIGN_STYLE["ink"],
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True, facecolor=DESIGN_STYLE["paper"])
    _style_axes(ax)
    _draw_mfd_layers(ax, points, summary, include_labels=True)
    _add_zoom_inset(ax, points, summary)
    controls = [str(control) for control in summary["control"].dropna().unique()]
    title_suffix = " vs ".join(controls) if len(controls) <= 3 else "Selected controls"
    ax.set_title(f"Ingolstadt21 MFD: {title_suffix}", loc="left", pad=14)
    ax.set_xlabel("Vehicle density (v/km)")
    ax.set_ylabel("Network outflow (vehicles/h)")
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(handle, label) for handle, label in zip(handles, labels) if not label.endswith(" samples")]
    if filtered:
        legend = ax.legend(
            [item[0] for item in filtered],
            [item[1] for item in filtered],
            title="Control",
            loc="upper right",
            frameon=True,
            borderpad=0.8,
            labelspacing=0.7,
            handlelength=2.0,
        )
        legend.get_frame().set_facecolor(DESIGN_STYLE["card_alt"])
        legend.get_frame().set_edgecolor(DESIGN_STYLE["ink"])
        legend.get_frame().set_linewidth(0.8)
    ax.margins(x=0.05, y=0.08)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def _style_axes(ax: Any) -> None:
    ax.grid(True, color=DESIGN_STYLE["card"], linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(DESIGN_STYLE["ink"])
        spine.set_linewidth(0.9)


def _draw_mfd_layers(ax: Any, points: Any, summary: Any, *, include_labels: bool) -> None:
    for control, group in points.groupby("control", sort=False):
        group = group.sort_values(["bin_index", "time"])
        color = str(group["color"].iloc[0])
        marker = str(group["marker"].iloc[0])
        ax.scatter(
            group["vehicle_density_v_per_km"],
            group["network_outflow_veh_per_h"],
            color=color,
            marker=marker,
            s=34,
            alpha=0.22,
            edgecolor=DESIGN_STYLE["paper"],
            linewidth=0.25,
            label=f"{control} samples" if include_labels else None,
            zorder=2,
        )
    for control, group in summary.groupby("control", sort=False):
        group = group.sort_values("bin_index")
        color = str(group["color"].iloc[0])
        marker = str(group["marker"].iloc[0])
        ax.plot(
            group["vehicle_density_v_per_km"],
            group["network_outflow_veh_per_h"],
            label=str(control) if include_labels else None,
            color=color,
            marker=marker,
            linewidth=2.8,
            markersize=7.2,
            alpha=0.96,
            markeredgecolor=DESIGN_STYLE["paper"],
            markeredgewidth=1.1,
            zorder=5,
        )


def _add_zoom_inset(ax: Any, points: Any, summary: Any) -> None:
    inset = ax.inset_axes([0.55, 0.48, 0.31, 0.32])
    inset.set_facecolor(DESIGN_STYLE["card_alt"])
    _style_axes(inset)
    _draw_mfd_layers(inset, points, summary, include_labels=False)
    inset.set_xlim(-0.2, 6.2)
    inset.set_ylim(650, 5700)
    inset.set_title("Zoom", fontsize=9, pad=5)
    inset.tick_params(labelsize=8)
    ax.indicate_inset_zoom(inset, edgecolor=DESIGN_STYLE["ink"], alpha=0.55, linewidth=0.8)


def _road_length_km(net_file: Path) -> float:
    if not net_file.is_absolute():
        net_file = ROOT / net_file
    edge_length_m = 0.0
    for _event, element in ET.iterparse(net_file, events=("end",)):
        if element.tag != "edge":
            continue
        edge_id = str(element.attrib.get("id", ""))
        edge_function = str(element.attrib.get("function", ""))
        if edge_id.startswith(":") or edge_function == "internal":
            element.clear()
            continue
        lane_lengths = []
        for lane in element.findall("lane"):
            try:
                lane_lengths.append(float(lane.attrib.get("length", "0") or 0.0))
            except ValueError:
                continue
        if lane_lengths:
            edge_length_m += max(lane_lengths)
        element.clear()
    if edge_length_m <= 0:
        raise ValueError(f"Unable to read non-internal road length from {net_file}")
    return edge_length_m / 1000.0


def _arrival_times(tripinfo_path: Path) -> list[float]:
    arrivals = []
    for _event, element in ET.iterparse(tripinfo_path, events=("end",)):
        if element.tag != "tripinfo":
            element.clear()
            continue
        arrival = element.attrib.get("arrival")
        if arrival is not None:
            try:
                value = float(arrival)
            except ValueError:
                element.clear()
                continue
            if math.isfinite(value) and value >= 0:
                arrivals.append(value)
        element.clear()
    return arrivals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Ingolstadt21 density/outflow MFD for selected controls.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--bin-seconds", type=float, default=300.0)
    parser.add_argument("--ppo-comparison", action="store_true", help="Render only FGSv3 PPO versus PPO.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.ppo_comparison:
        output_stem = args.output_stem
        if output_stem == DEFAULT_OUTPUT_STEM:
            output_stem = DEFAULT_PPO_COMPARISON_STEM
        paths = render_ingolstadt21_ppo_comparison_mfd(
            args.output_dir,
            output_stem=output_stem,
            bin_seconds=args.bin_seconds,
        )
    else:
        paths = render_ingolstadt21_outflow_mfd(
            args.output_dir,
            output_stem=args.output_stem,
            bin_seconds=args.bin_seconds,
        )
    for key, path in sorted(paths.items()):
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
