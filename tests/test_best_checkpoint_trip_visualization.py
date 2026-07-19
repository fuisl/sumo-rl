from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.render_best_checkpoint_trip import _projector, render_trip_animation, select_best_checkpoint


def test_select_best_checkpoint_uses_ranked_metadata(tmp_path):
    metadata_dir = tmp_path / "checkpoints" / "fgs_ppo" / "best_validation"
    metadata_dir.mkdir(parents=True)
    checkpoint_1 = metadata_dir / "rank1"
    checkpoint_2 = metadata_dir / "rank2"
    metadata = {
        "metric_name": "validation/resco_delay_mean",
        "retained": [
            {
                "rank": 2,
                "checkpoint_path": str(checkpoint_2),
                "metric_name": "validation/resco_delay_mean",
                "metric_value": 18.0,
                "validation_pass_index": 6,
                "validation_env_step": 21600.0,
            },
            {
                "rank": 1,
                "checkpoint_path": str(checkpoint_1),
                "metric_name": "validation/resco_delay_mean",
                "metric_value": 17.95,
                "validation_pass_index": 2,
                "validation_env_step": 7200.0,
            },
        ],
    }
    (metadata_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    selected = select_best_checkpoint(tmp_path)

    assert selected.rank == 1
    assert selected.checkpoint_path == checkpoint_1
    assert selected.metric_value == 17.95
    assert selected.validation_pass_index == 2


def test_render_trip_animation_draws_vehicle_dots_and_pressure_bars(tmp_path):
    trace = {
        "metadata": {
            "experiment": "synthetic_best_checkpoint",
            "checkpoint_rank": 1,
            "metric_value": 17.95,
        },
        "network": {
            "road_polylines": [
                [[0.0, 0.0], [100.0, 0.0]],
                [[50.0, -40.0], [50.0, 40.0]],
            ],
            "tls_positions": {
                "tls_a": [50.0, 0.0],
            },
        },
        "frames": [
            {
                "time": 0.0,
                "vehicles": [
                    {"id": "veh_1", "x": 10.0, "y": 0.0, "speed": 0.0},
                    {"id": "veh_2", "x": 30.0, "y": 0.0, "speed": 8.0},
                ],
                "pressures": {"tls_a": -6.0},
            },
            {
                "time": 5.0,
                "vehicles": [
                    {"id": "veh_1", "x": 35.0, "y": 0.0, "speed": 5.0},
                    {"id": "veh_2", "x": 70.0, "y": 0.0, "speed": 10.0},
                ],
                "pressures": {"tls_a": 3.0},
            },
        ],
    }

    output = render_trip_animation(trace, tmp_path / "trip.gif", width=520, fps=4, frame_count=2)

    image = Image.open(output)
    assert image.is_animated
    assert image.n_frames == 2
    first_frame = image.convert("RGB")
    pixels = list(first_frame.getdata())
    queue_pixels = sum(1 for r, g, b in pixels if r > 175 and 70 < g < 155 and b < 130)
    flow_pixels = sum(1 for r, g, b in pixels if 80 < r < 140 and 120 < g < 180 and b > 155)
    road_pixels = sum(1 for r, g, b in pixels if 120 < r < 190 and 115 < g < 185 and 105 < b < 175)
    assert queue_pixels > 0
    assert flow_pixels > 0
    assert road_pixels > 0


def test_render_trip_animation_clears_previous_vehicle_positions(tmp_path):
    trace = {
        "metadata": {
            "experiment": "synthetic_vehicle_motion",
            "checkpoint_rank": 1,
            "metric_value": 1.0,
        },
        "network": {
            "road_polylines": [[[0.0, 0.0], [100.0, 0.0]]],
            "tls_positions": {"tls_a": [50.0, 0.0]},
        },
        "frames": [
            {
                "time": 0.0,
                "vehicles": [{"id": "veh_1", "x": 10.0, "y": 0.0, "speed": 0.0}],
                "pressures": {"tls_a": 0.0},
            },
            {
                "time": 5.0,
                "vehicles": [{"id": "veh_1", "x": 90.0, "y": 0.0, "speed": 0.0}],
                "pressures": {"tls_a": 0.0},
            },
        ],
    }

    output = render_trip_animation(
        trace,
        tmp_path / "motion.gif",
        width=520,
        fps=4,
        frame_count=2,
        show_overlay=False,
        show_legend=False,
    )
    image = Image.open(output)
    image.seek(1)
    second_frame = image.convert("RGB")
    project, _ = _projector([(0.0, 0.0), (100.0, 0.0), (50.0, 0.0), (10.0, 0.0), (90.0, 0.0)], 520)
    old_x, old_y = (int(round(value)) for value in project((10.0, 0.0)))
    new_x, new_y = (int(round(value)) for value in project((90.0, 0.0)))

    def red_pixel_count(cx: int, cy: int) -> int:
        count = 0
        for x in range(cx - 7, cx + 8):
            for y in range(cy - 7, cy + 8):
                r, g, b = second_frame.getpixel((x, y))
                if r > 175 and 70 < g < 155 and b < 130:
                    count += 1
        return count

    assert red_pixel_count(old_x, old_y) == 0
    assert red_pixel_count(new_x, new_y) > 20


def test_render_trip_animation_supports_requested_aspect_ratio(tmp_path):
    trace = {
        "metadata": {"experiment": "synthetic_aspect_ratio"},
        "network": {
            "road_polylines": [[[0.0, 0.0], [100.0, 0.0]]],
            "tls_positions": {"tls_a": [50.0, 0.0]},
        },
        "frames": [
            {
                "time": 0.0,
                "vehicles": [{"id": "veh_1", "x": 50.0, "y": 0.0, "speed": 8.0}],
                "pressures": {"tls_a": 1.0},
            }
        ],
    }

    output = render_trip_animation(trace, tmp_path / "wide.gif", width=660, aspect_ratio=1.65, frame_count=1)

    image = Image.open(output)
    assert image.size == (660, 400)


def test_projector_centers_network_inside_requested_aspect_ratio():
    project, height = _projector([(0.0, 0.0), (0.0, 100.0)], 1320, aspect_ratio=1.65)

    bottom_x, bottom_y = project((0.0, 0.0))
    top_x, top_y = project((0.0, 100.0))

    assert height == 800
    assert abs(bottom_x - 660.0) < 8.0
    assert abs(top_x - 660.0) < 8.0
    assert abs(bottom_y - 736.0) < 1.0
    assert abs(top_y - 64.0) < 1.0
