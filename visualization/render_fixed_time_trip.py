from __future__ import annotations

import argparse
from pathlib import Path

from render_best_checkpoint_trip import (
    DEFAULT_FIXED_TIME_CONFIG,
    DEFAULT_FIXED_TIME_OUTPUT_DIR,
    run_fixed_time_trip_visualization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fixed-time SUMO trip animation with pressure bars.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIXED_TIME_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--config-name", default=DEFAULT_FIXED_TIME_CONFIG, help="Hydra config name to compose.")
    parser.add_argument("--config-file", type=Path, default=None, help="Optional resolved Hydra config YAML.")
    parser.add_argument("--seed", type=int, default=None, help="SUMO seed. Defaults to the first configured eval seed.")
    parser.add_argument("--width", type=int, default=1200, help="Animation width in pixels.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--frame-count", type=int, default=160, help="Maximum rendered GIF frame count.")
    parser.add_argument("--max-render-vehicles", type=int, default=1200, help="Maximum vehicle dots drawn per frame.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_fixed_time_trip_visualization(
        args.output_dir,
        config_name=args.config_name,
        config_file=args.config_file,
        seed=args.seed,
        width=args.width,
        fps=args.fps,
        frame_count=args.frame_count,
        max_render_vehicles=args.max_render_vehicles,
    )
    print(f"Wrote trip animation GIF: {paths['animation']}")
    print(f"Wrote live trace JSON: {paths['trace']}")
    print(f"Wrote metadata JSON: {paths['metadata']}")
    print(f"Retained tripinfo XML: {paths['tripinfo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
