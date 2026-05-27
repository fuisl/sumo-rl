from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.experiments.rllib_runner import (
    _build_algorithm_config,
    _evaluate_with_details,
    _restore_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore an RLlib checkpoint and run one evaluation pass.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Hydra run directory that contains .hydra/config.yaml for the original training run.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory to restore, for example a best_validation checkpoint path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    algorithm_kind = str(cfg.algorithm.kind)

    import ray

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_gpus=0)
    algo = None
    try:
        algo_config = _build_algorithm_config(cfg, run_dir, algorithm_kind)
        build_algo = getattr(algo_config, "build_algo", None)
        algo = build_algo() if callable(build_algo) else algo_config.build()

        _restore_checkpoint(algo, checkpoint_path)

        summary, seed_rows, _ = _evaluate_with_details(
            cfg,
            run_dir,
            algo,
            algorithm_kind,
            cfg.logging,
            include_validation_metrics=True,
        )

        print("Validation summary:")
        pprint(summary)
        print("\nPer-seed rows:")
        pprint(seed_rows)
    finally:
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
