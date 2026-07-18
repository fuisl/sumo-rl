from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib
from matplotlib.patches import FancyArrowPatch
from omegaconf import OmegaConf


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "sumo_rl").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate the repo root from the current working directory.")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.agents.rllib_common import build_sumo_parallel_env
from sumo_rl.agents.static import MaxPressurePolicy
from sumo_rl.agents.fgs.topology import extract_tls_topology
from sumo_rl.environment.graph_env import resolve_sumo_base_env
from sumo_rl.experiments.runner import _fixed_time_step_action
from sumo_rl.models.graph import build_traffic_signal_graph, traffic_signals_from_base_env


DEFAULT_ARTIFACT_DIR = ROOT / "experiments" / "artifacts" / "spatiotemporal_dependency"


def load_scenario_cfg(
    *,
    scenario_name: str = "resco_cologne8",
    controller: str = "fixed_time",
    episode_seconds: int = 3600,
    seed: int = 1,
    use_gui: bool = False,
    additional_sumo_cmd: str = "--no-step-log true --duration-log.disable true",
) -> Any:
    base_cfg = OmegaConf.load(ROOT / "configs" / "base.yaml")
    scenario_cfg = OmegaConf.load(ROOT / "configs" / "scenario" / f"{scenario_name}.yaml")
    algorithm_name = "fixed_time" if controller == "fixed_time" else "static_max_pressure"
    algorithm_cfg = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{algorithm_name}.yaml")
    cfg = OmegaConf.merge(base_cfg, scenario_cfg, algorithm_cfg)
    cfg.experiment.name = f"notebook_{scenario_name}_{controller}_dependency"
    cfg.experiment.seed = int(seed)
    cfg.experiment.fixed_ts = bool(controller == "fixed_time")
    cfg.env.kwargs.num_seconds = int(episode_seconds)
    cfg.env.kwargs.fixed_ts = bool(controller == "fixed_time")
    cfg.env.kwargs.use_gui = bool(use_gui)
    cfg.env.kwargs.add_system_info = True
    cfg.env.kwargs.add_per_agent_info = True
    cfg.env.kwargs.sumo_warnings = False
    cfg.env.kwargs.additional_sumo_cmd = str(additional_sumo_cmd)
    cfg.env.kwargs.out_csv_name = None
    cfg.env.kwargs.tripinfo_output_name = None
    cfg.env.kwargs.statistic_output_name = None
    return cfg


def metric_column(signal_metric: str, aggregation: str) -> str:
    metric = str(signal_metric).strip().lower()
    agg = str(aggregation).strip().lower()
    if metric not in {"density", "queue"}:
        raise ValueError("signal_metric must be 'density' or 'queue'.")
    if agg not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'mean' or 'sum'.")
    return f"{metric}_{agg}"


def metric_label(signal_metric: str, aggregation: str) -> str:
    metric = str(signal_metric).strip().lower()
    agg = str(aggregation).strip().lower()
    if metric == "density":
        base = "incoming-lane density"
    else:
        base = "incoming-lane queue"
    if agg == "mean":
        return f"mean {base}"
    return f"sum {base}"


def _aggregate(values: list[float], aggregation: str) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    if str(aggregation).strip().lower() == "sum":
        return float(np.sum(array))
    return float(np.mean(array))


def _snapshot_signal_rows(
    base_env: Any,
    *,
    decision_index: int,
    aggregation: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elapsed_seconds = float(base_env.sim_step) - float(base_env.begin_time)
    for signal_id in getattr(base_env, "ts_ids", []):
        traffic_signal = base_env.traffic_signals[str(signal_id)]
        lane_density = [float(value) for value in traffic_signal.get_lanes_density()]
        lane_queue = [float(value) for value in traffic_signal.get_lanes_queue()]
        rows.append(
            {
                "decision_index": int(decision_index),
                "sim_time_abs_seconds": float(base_env.sim_step),
                "elapsed_seconds": float(elapsed_seconds),
                "signal_id": str(signal_id),
                "num_incoming_lanes": int(len(lane_density)),
                "density_mean": _aggregate(lane_density, "mean"),
                "density_sum": _aggregate(lane_density, "sum"),
                "queue_mean": _aggregate(lane_queue, "mean"),
                "queue_sum": _aggregate(lane_queue, "sum"),
            }
        )
    return rows


def collect_signal_trace(
    *,
    cfg: Any,
    controller: str = "fixed_time",
    seed: int | None = None,
    use_libsumo: bool = False,
    aggregation: str = "mean",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir or (ROOT / "outputs" / "_notebooks"))
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_sumo_parallel_env(cfg, run_dir, seed=seed, use_libsumo=use_libsumo)
    base_env = resolve_sumo_base_env(env)
    agent_ids = [str(agent_id) for agent_id in getattr(base_env, "ts_ids", [])]
    policy = MaxPressurePolicy() if controller == "static_max_pressure" else None
    trace_rows: list[dict[str, Any]] = []

    try:
        env.reset(seed=seed)
        topology = extract_tls_topology(getattr(base_env, "_net"))
        graph = build_traffic_signal_graph(
            traffic_signals_from_base_env(base_env),
            net_file=getattr(base_env, "_net", None),
            include_virtual_nodes=False,
            add_self_loops=False,
        )

        decision_index = 0
        trace_rows.extend(_snapshot_signal_rows(base_env, decision_index=decision_index, aggregation=aggregation))
        while getattr(env, "agents", None):
            if controller == "fixed_time":
                actions = _fixed_time_step_action(env, base_env, agent_ids)
            elif controller == "static_max_pressure":
                actions = {agent_id: policy.select_action(base_env.traffic_signals[agent_id]) for agent_id in agent_ids}
            else:
                raise ValueError("controller must be 'fixed_time' or 'static_max_pressure'.")
            env.step(actions)
            decision_index += 1
            trace_rows.extend(_snapshot_signal_rows(base_env, decision_index=decision_index, aggregation=aggregation))
    finally:
        env.close()

    trace_df = pd.DataFrame(trace_rows).sort_values(["decision_index", "signal_id"]).reset_index(drop=True)
    return {
        "cfg": cfg,
        "controller": str(controller),
        "seed": int(seed if seed is not None else getattr(cfg.experiment, "seed", 0) or 0),
        "trace_df": trace_df,
        "graph": graph,
        "topology": topology,
        "decision_seconds": int(getattr(base_env, "delta_time", 5)),
        "begin_time": float(getattr(base_env, "begin_time", 0.0)),
        "scenario_name": str(getattr(getattr(cfg, "scenario", None), "name", "scenario")),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def autocorrelation(series: np.ndarray, max_lag_steps: int, decision_seconds: int) -> pd.DataFrame:
    values = np.asarray(series, dtype=float)
    rows = []
    for lag_step in range(max(0, int(max_lag_steps)) + 1):
        if lag_step == 0:
            corr = _safe_corr(values, values)
        else:
            corr = _safe_corr(values[:-lag_step], values[lag_step:])
        rows.append(
            {
                "lag_step": int(lag_step),
                "lag_seconds": int(lag_step) * int(decision_seconds),
                "correlation": corr,
            }
        )
    return pd.DataFrame(rows)


def lagged_cross_correlation(
    source_series: np.ndarray,
    target_series: np.ndarray,
    max_lag_steps: int,
    decision_seconds: int,
) -> pd.DataFrame:
    source_values = np.asarray(source_series, dtype=float)
    target_values = np.asarray(target_series, dtype=float)
    rows = []
    for lag_step in range(max(0, int(max_lag_steps)) + 1):
        if lag_step == 0:
            corr = _safe_corr(source_values, target_values)
        else:
            corr = _safe_corr(source_values[:-lag_step], target_values[lag_step:])
        rows.append(
            {
                "lag_step": int(lag_step),
                "lag_seconds": int(lag_step) * int(decision_seconds),
                "correlation": corr,
            }
        )
    return pd.DataFrame(rows)


def _adjacency_for_distance(adjacency: np.ndarray, *, directed: bool) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float32).copy()
    np.fill_diagonal(matrix, 0.0)
    if not directed:
        matrix = np.maximum(matrix, matrix.T)
    return matrix


def shortest_path_distances(adjacency: np.ndarray, *, directed: bool = True) -> np.ndarray:
    matrix = _adjacency_for_distance(adjacency, directed=directed)
    num_nodes = int(matrix.shape[0])
    distances = np.full((num_nodes, num_nodes), np.inf, dtype=float)
    for start in range(num_nodes):
        distances[start, start] = 0.0
        queue = deque([int(start)])
        while queue:
            node = queue.popleft()
            for next_node in np.where(matrix[node] > 0)[0]:
                next_node = int(next_node)
                if np.isfinite(distances[start, next_node]):
                    continue
                distances[start, next_node] = distances[start, node] + 1.0
                queue.append(next_node)
    return distances


def _distance_category(distance: float) -> str:
    if distance == 1:
        return "1-hop"
    if distance == 2:
        return "2-hop"
    return "Distant"


def _pivot_trace(trace_df: pd.DataFrame, *, value_col: str, signal_ids: list[str]) -> pd.DataFrame:
    return (
        trace_df.pivot(index="decision_index", columns="signal_id", values=value_col)
        .reindex(columns=signal_ids)
        .sort_index()
    )


def compute_pair_metrics(
    trace_df: pd.DataFrame,
    graph: Any,
    *,
    value_col: str,
    decision_seconds: int,
    max_lag_steps: int,
    distance_mode: str = "directed",
) -> dict[str, Any]:
    directed = str(distance_mode).strip().lower() != "undirected"
    signal_ids = list(graph.ts_ids)
    pivot_df = _pivot_trace(trace_df, value_col=value_col, signal_ids=signal_ids)
    distances = shortest_path_distances(graph.adjacency, directed=directed)
    rows: list[dict[str, Any]] = []

    for source_index, source_id in enumerate(signal_ids):
        source_values = pivot_df[source_id].to_numpy(dtype=float)
        for target_index, target_id in enumerate(signal_ids):
            if source_index == target_index:
                continue
            target_values = pivot_df[target_id].to_numpy(dtype=float)
            curve_df = lagged_cross_correlation(source_values, target_values, max_lag_steps, decision_seconds)
            valid_curve = curve_df.dropna(subset=["correlation"])
            if valid_curve.empty:
                max_corr = float("nan")
                best_lag_step = float("nan")
                best_lag_seconds = float("nan")
            else:
                best_row = valid_curve.loc[valid_curve["correlation"].idxmax()]
                max_corr = float(best_row["correlation"])
                best_lag_step = float(best_row["lag_step"])
                best_lag_seconds = float(best_row["lag_seconds"])

            positive_curve = curve_df[curve_df["lag_step"] > 0].dropna(subset=["correlation"])
            if positive_curve.empty:
                max_positive_corr = float("nan")
                best_positive_lag_step = float("nan")
                best_positive_lag_seconds = float("nan")
            else:
                best_positive_row = positive_curve.loc[positive_curve["correlation"].idxmax()]
                max_positive_corr = float(best_positive_row["correlation"])
                best_positive_lag_step = float(best_positive_row["lag_step"])
                best_positive_lag_seconds = float(best_positive_row["lag_seconds"])

            distance = float(distances[source_index, target_index])
            rows.append(
                {
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "distance_mode": "directed" if directed else "undirected",
                    "graph_distance": distance,
                    "distance_category": _distance_category(distance),
                    "max_corr": max_corr,
                    "best_lag_step": best_lag_step,
                    "best_lag_seconds": best_lag_seconds,
                    "max_positive_lag_corr": max_positive_corr,
                    "best_positive_lag_step": best_positive_lag_step,
                    "best_positive_lag_seconds": best_positive_lag_seconds,
                }
            )

    pair_metrics_df = pd.DataFrame(rows).sort_values(
        ["distance_category", "graph_distance", "source_id", "target_id"]
    ).reset_index(drop=True)
    return {
        "pivot_df": pivot_df,
        "distances": distances,
        "pair_metrics_df": pair_metrics_df,
    }


def choose_representative_pair(pair_metrics_df: pd.DataFrame) -> tuple[str, str] | None:
    direct_pairs = pair_metrics_df[pair_metrics_df["distance_category"] == "1-hop"].copy()
    if not direct_pairs.empty:
        direct_pairs = direct_pairs.sort_values(
            ["max_positive_lag_corr", "max_corr", "best_positive_lag_seconds", "source_id", "target_id"],
            ascending=[False, False, True, True, True],
        )
        top_row = direct_pairs.iloc[0]
        return str(top_row["source_id"]), str(top_row["target_id"])

    if pair_metrics_df.empty:
        return None
    top_row = pair_metrics_df.sort_values(["max_corr", "source_id", "target_id"], ascending=[False, True, True]).iloc[0]
    return str(top_row["source_id"]), str(top_row["target_id"])


def choose_representative_signal(
    pivot_df: pd.DataFrame,
    *,
    decision_seconds: int,
    max_lag_steps: int,
    preferred_signal_id: str | None = None,
) -> str:
    if preferred_signal_id is not None and preferred_signal_id in pivot_df.columns:
        return str(preferred_signal_id)

    best_signal_id = None
    best_score = float("-inf")
    for signal_id in pivot_df.columns:
        acf_df = autocorrelation(pivot_df[signal_id].to_numpy(dtype=float), max_lag_steps, decision_seconds)
        positive_lags = acf_df[acf_df["lag_step"] > 0]["correlation"].dropna()
        score = float(positive_lags.mean()) if not positive_lags.empty else float("-inf")
        if score > best_score:
            best_score = score
            best_signal_id = str(signal_id)
    if best_signal_id is None:
        raise RuntimeError("Could not choose a representative signal from the trace.")
    return best_signal_id


def analyze_spatiotemporal_dependency(
    trace_bundle: dict[str, Any],
    *,
    signal_metric: str = "density",
    aggregation: str = "mean",
    max_lag_seconds: int = 60,
    distance_mode: str = "directed",
    representative_pair: tuple[str, str] | None = None,
    representative_signal_id: str | None = None,
) -> dict[str, Any]:
    decision_seconds = int(trace_bundle["decision_seconds"])
    max_lag_steps = max(1, int(max_lag_seconds) // max(1, decision_seconds))
    value_col = metric_column(signal_metric, aggregation)
    pair_bundle = compute_pair_metrics(
        trace_bundle["trace_df"],
        trace_bundle["graph"],
        value_col=value_col,
        decision_seconds=decision_seconds,
        max_lag_steps=max_lag_steps,
        distance_mode=distance_mode,
    )
    pair_metrics_df = pair_bundle["pair_metrics_df"]
    representative_pair = representative_pair or choose_representative_pair(pair_metrics_df)
    preferred_signal_id = representative_pair[0] if representative_pair is not None else None
    representative_signal_id = choose_representative_signal(
        pair_bundle["pivot_df"],
        decision_seconds=decision_seconds,
        max_lag_steps=max_lag_steps,
        preferred_signal_id=representative_signal_id or preferred_signal_id,
    )
    autocorr_df = autocorrelation(
        pair_bundle["pivot_df"][representative_signal_id].to_numpy(dtype=float),
        max_lag_steps,
        decision_seconds,
    )
    if representative_pair is None:
        raise RuntimeError("Could not identify a representative signal pair.")
    crosscorr_df = lagged_cross_correlation(
        pair_bundle["pivot_df"][representative_pair[0]].to_numpy(dtype=float),
        pair_bundle["pivot_df"][representative_pair[1]].to_numpy(dtype=float),
        max_lag_steps,
        decision_seconds,
    )
    return {
        "signal_metric": str(signal_metric),
        "aggregation": str(aggregation),
        "value_col": value_col,
        "value_label": metric_label(signal_metric, aggregation),
        "decision_seconds": decision_seconds,
        "max_lag_seconds": int(max_lag_steps) * int(decision_seconds),
        "max_lag_steps": int(max_lag_steps),
        "distance_mode": str(distance_mode).strip().lower(),
        "pivot_df": pair_bundle["pivot_df"],
        "distances": pair_bundle["distances"],
        "pair_metrics_df": pair_metrics_df,
        "representative_pair": tuple(representative_pair),
        "representative_signal_id": str(representative_signal_id),
        "autocorr_df": autocorr_df,
        "crosscorr_df": crosscorr_df,
    }


def _bounds(road_polylines: list[list[tuple[float, float]]], positions: dict[str, tuple[float, float]]):
    points = [point for polyline in road_polylines for point in polyline] + list(positions.values())
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def plot_topology_map(
    ax: Any,
    trace_bundle: dict[str, Any],
    analysis_bundle: dict[str, Any],
    *,
    draw_all_edges: bool = True,
) -> Any:
    topology = trace_bundle["topology"]
    graph = trace_bundle["graph"]
    road_polylines = topology.road_polylines
    positions = topology.positions
    min_x, max_x, min_y, max_y = _bounds(road_polylines, positions)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    for polyline in road_polylines:
        xy = np.asarray(polyline, dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=1.0, alpha=0.75, zorder=1)

    if draw_all_edges:
        for source_index, target_index in np.argwhere(graph.adjacency > 0):
            source_id = graph.ts_ids[int(source_index)]
            target_id = graph.ts_ids[int(target_index)]
            if source_id == target_id:
                continue
            start = positions.get(source_id)
            end = positions.get(target_id)
            if start is None or end is None:
                continue
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#93c5fd",
                alpha=0.7,
                zorder=2,
            )
            ax.add_patch(arrow)

    representative_signal_id = analysis_bundle["representative_signal_id"]
    representative_pair = analysis_bundle["representative_pair"]
    for signal_id, point in positions.items():
        is_highlight_signal = str(signal_id) == str(representative_signal_id)
        facecolor = "#f59e0b" if is_highlight_signal else "#0f766e"
        size = 85 if is_highlight_signal else 45
        ax.scatter(point[0], point[1], s=size, color=facecolor, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(
            point[0] + 0.012 * span_x,
            point[1] + 0.018 * span_y,
            str(signal_id),
            fontsize=8,
            color="#0f172a",
            zorder=5,
        )

    pair_start = positions.get(representative_pair[0])
    pair_end = positions.get(representative_pair[1])
    if pair_start is not None and pair_end is not None:
        ax.add_patch(
            FancyArrowPatch(
                pair_start,
                pair_end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2.5,
                color="#dc2626",
                alpha=0.95,
                zorder=6,
            )
        )

    ax.set_title("Cologne8 TLS Topology and Selected 1-hop Pair")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_autocorrelation(ax: Any, analysis_bundle: dict[str, Any]) -> Any:
    autocorr_df = analysis_bundle["autocorr_df"]
    signal_id = analysis_bundle["representative_signal_id"]
    ax.plot(
        autocorr_df["lag_seconds"],
        autocorr_df["correlation"],
        marker="o",
        linewidth=2.0,
        color="#0f766e",
    )
    ax.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Panel A: Temporal Dependency at TLS {signal_id}")
    ax.grid(alpha=0.18)
    return ax


def plot_cross_correlation(ax: Any, analysis_bundle: dict[str, Any]) -> Any:
    crosscorr_df = analysis_bundle["crosscorr_df"]
    source_id, target_id = analysis_bundle["representative_pair"]
    ax.plot(
        crosscorr_df["lag_seconds"],
        crosscorr_df["correlation"],
        marker="o",
        linewidth=2.0,
        color="#dc2626",
    )
    ax.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Panel B: {source_id} -> {target_id}")
    ax.grid(alpha=0.18)
    return ax


def plot_dependency_panels(
    trace_bundle: dict[str, Any],
    analysis_bundle: dict[str, Any],
    *,
    figsize: tuple[float, float] = (14, 5.5),
) -> tuple[Any, np.ndarray]:
    figure, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    plot_topology_map(axes[0], trace_bundle, analysis_bundle)
    plot_autocorrelation(axes[1], analysis_bundle)
    plot_cross_correlation(axes[2], analysis_bundle)
    return figure, axes


def plot_distance_boxplot(
    ax: Any,
    analysis_bundle: dict[str, Any],
    *,
    use_positive_lag_only: bool = False,
) -> Any:
    pair_metrics_df = analysis_bundle["pair_metrics_df"]
    value_key = "max_positive_lag_corr" if use_positive_lag_only else "max_corr"
    categories = ["1-hop", "2-hop", "Distant"]
    values_by_category = [
        pair_metrics_df.loc[pair_metrics_df["distance_category"] == category, value_key].dropna().to_numpy(dtype=float)
        for category in categories
    ]
    boxplot_kwargs = {
        "patch_artist": True,
        "boxprops": {"facecolor": "#bfdbfe", "edgecolor": "#1d4ed8"},
        "medianprops": {"color": "#1e3a8a", "linewidth": 2.0},
        "whiskerprops": {"color": "#1d4ed8"},
        "capprops": {"color": "#1d4ed8"},
        "flierprops": {"markerfacecolor": "#1d4ed8", "markeredgecolor": "#1d4ed8", "alpha": 0.4},
    }
    try:
        ax.boxplot(values_by_category, labels=categories, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(values_by_category, tick_labels=categories, **boxplot_kwargs)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Maximum lagged correlation")
    mode_label = "positive lags only" if use_positive_lag_only else "lags 0..K"
    ax.set_title(f"Network-wide Evidence by Graph Distance ({mode_label})")
    ax.grid(axis="y", alpha=0.18)
    return ax


def plot_pairwise_heatmap(ax: Any, analysis_bundle: dict[str, Any], *, use_positive_lag_only: bool = False) -> Any:
    pair_metrics_df = analysis_bundle["pair_metrics_df"]
    pivot_df = analysis_bundle["pivot_df"]
    signal_ids = list(pivot_df.columns)
    value_key = "max_positive_lag_corr" if use_positive_lag_only else "max_corr"
    matrix = np.full((len(signal_ids), len(signal_ids)), np.nan, dtype=float)
    index_by_id = {signal_id: index for index, signal_id in enumerate(signal_ids)}
    for _, row in pair_metrics_df.iterrows():
        matrix[index_by_id[str(row["source_id"])], index_by_id[str(row["target_id"])]] = float(row[value_key])

    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(signal_ids)))
    ax.set_yticks(range(len(signal_ids)))
    ax.set_xticklabels(signal_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(signal_ids, fontsize=8)
    ax.set_title("Pairwise Maximum Lagged Correlation")
    ax.set_xlabel("Target TLS")
    ax.set_ylabel("Source TLS")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def export_dependency_artifacts(
    trace_bundle: dict[str, Any],
    analysis_bundle: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir or DEFAULT_ARTIFACT_DIR / trace_bundle["scenario_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = output_dir / "signal_trace.csv"
    pair_csv = output_dir / "pair_metrics.csv"
    autocorr_csv = output_dir / "representative_autocorrelation.csv"
    crosscorr_csv = output_dir / "representative_cross_correlation.csv"
    trace_bundle["trace_df"].to_csv(trace_csv, index=False)
    analysis_bundle["pair_metrics_df"].to_csv(pair_csv, index=False)
    analysis_bundle["autocorr_df"].to_csv(autocorr_csv, index=False)
    analysis_bundle["crosscorr_df"].to_csv(crosscorr_csv, index=False)

    panels_path = output_dir / "dependency_panels.png"
    boxplot_path = output_dir / "distance_boxplot.png"
    heatmap_path = output_dir / "pairwise_heatmap.png"

    panels_figure, _ = plot_dependency_panels(trace_bundle, analysis_bundle)
    panels_figure.savefig(panels_path, dpi=250, bbox_inches="tight")
    plt.close(panels_figure)

    boxplot_figure, boxplot_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_distance_boxplot(boxplot_ax, analysis_bundle)
    boxplot_figure.savefig(boxplot_path, dpi=250, bbox_inches="tight")
    plt.close(boxplot_figure)

    heatmap_figure, heatmap_ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    plot_pairwise_heatmap(heatmap_ax, analysis_bundle)
    heatmap_figure.savefig(heatmap_path, dpi=250, bbox_inches="tight")
    plt.close(heatmap_figure)

    return {
        "trace_csv": trace_csv,
        "pair_csv": pair_csv,
        "autocorr_csv": autocorr_csv,
        "crosscorr_csv": crosscorr_csv,
        "panels_png": panels_path,
        "boxplot_png": boxplot_path,
        "heatmap_png": heatmap_path,
    }
