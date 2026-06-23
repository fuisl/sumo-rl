from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sumo_rl.experiments.live_trip_visualization import (
    _capture_frame,
    _projector,
    render_trip_animation,
    select_best_checkpoint,
)


def test_select_best_checkpoint_uses_ranked_metadata(tmp_path: Path):
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


def test_render_trip_animation_draws_vehicle_dots_and_pressure_bars(tmp_path: Path):
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
    red_pixels = sum(1 for r, g, b in pixels if r > 180 and g < 90 and b < 90)
    blue_pixels = sum(1 for r, g, b in pixels if b > 140 and r < 120)
    road_pixels = sum(1 for r, g, b in pixels if 70 < r < 130 and 80 < g < 150 and 100 < b < 170)
    assert red_pixels > 0
    assert blue_pixels > 0
    assert road_pixels > 0


def test_render_trip_animation_clears_previous_vehicle_positions(tmp_path: Path):
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

    output = render_trip_animation(trace, tmp_path / "motion.gif", width=520, fps=4, frame_count=2)
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
                if r > 180 and g < 100 and b < 100:
                    count += 1
        return count

    assert red_pixel_count(old_x, old_y) == 0
    assert red_pixel_count(new_x, new_y) > 20


def test_capture_frame_collects_time_vehicle_and_pressure_fields():
    class DummyVehicleAPI:
        def getIDList(self):
            return ["veh_a"]

        def getPosition(self, vehicle_id):
            assert vehicle_id == "veh_a"
            return (12.5, 7.25)

        def getSpeed(self, vehicle_id):
            assert vehicle_id == "veh_a"
            return 4.5

        def getRoadID(self, vehicle_id):
            assert vehicle_id == "veh_a"
            return "edge_0"

        def getLaneID(self, vehicle_id):
            assert vehicle_id == "veh_a"
            return "lane_0"

    class DummySignal:
        def get_pressure(self):
            return -3.0

    class DummySumo:
        vehicle = DummyVehicleAPI()

    class DummyEnv:
        sumo = DummySumo()
        sim_step = 42.0
        ts_ids = ["tls_1"]
        traffic_signals = {"tls_1": DummySignal()}

    frame = _capture_frame(DummyEnv())

    assert frame["time"] == 42.0
    assert frame["pressures"] == {"tls_1": -3.0}
    assert frame["vehicles"] == [
        {
            "id": "veh_a",
            "x": 12.5,
            "y": 7.25,
            "speed": 4.5,
            "edge": "edge_0",
            "lane": "lane_0",
        }
    ]
