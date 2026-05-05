---
title: Thesis Experiments
firstpage:
---

# Thesis Experiments

This page contains the thesis-specific experiment workflow built on top of the upstream SUMO-RL examples.

If you are looking for fixed-time/manual traffic control, read [docs/thesis/manual_control.md](manual_control.md) after this page.
If you want the RESCO static baselines, read [docs/thesis/static_baselines.md](static_baselines.md) next.
If you want the RESCO 4x4 SB3 examples, read [docs/thesis/third_party_sb3.md](third_party_sb3.md) next.

The 4x4 grid presets in this thesis use the RESCO grid4x4 assets by default, not the Lucas 4x4 network.

## Hydra
Hydra is used as the experiment composition layer.

- Each runnable example has a Hydra config in `configs/`
- Configs define the environment, algorithm, and logging settings
- Command-line overrides let you change seeds, paths, and hyperparameters without editing code
- Each run gets its own output directory under `outputs/<experiment-name>/<timestamp>/`
- A local metrics CSV is written to `outputs/<experiment-name>/<timestamp>/logs/metrics.csv` for quick debugging
- The runner now logs episode-end RESCO summaries only, using:
  - `resco_avg_delay` from SUMO tripinfo `timeLoss`
  - `resco_trip_time` from SUMO tripinfo `duration`
  - `resco_wait` from SUMO tripinfo `waitingTime`
  - `resco_queue` and `resco_max_queue` from the live queue metrics
  - the raw tripinfo XML files are stored under `outputs/<experiment-name>/<timestamp>/tripinfo/`
- The config layout is split into:
  - `configs/scenario/` for network and road-network setup
  - `configs/algorithm/` for algorithm kind and default hyperparameters
  - top-level preset files such as `configs/sb3_grid4x4.yaml` that combine the pieces

Example:
```bash
python experiments/dqn_2way-single-intersection.py experiment.seed=7
```

The 4x4 presets on this repo point to:
- RESCO `grid4x4` for the thesis configs
- Lucas `4x4-Lucas` only in older upstream-style example paths, if you choose to use them manually

## Weights & Biases
Weights & Biases is used for experiment tracking.

- W&B logs configs, metrics, and run metadata
- The repo defaults to offline mode so local runs do not require an API key
- To use online logging, authenticate outside the repo with `wandb login` or set `WANDB_API_KEY` in your environment

Example:
```bash
python experiments/dqn_2way-single-intersection.py logging.mode=online logging.project=my-thesis
```

## Optional Install
To use the Hydra and W&B experiment layer, install the optional extras:
```bash
pip install -e ".[experiments]"
```

## Notes
- These additions do not replace the upstream SUMO-RL API.
- The existing environment and algorithm examples still run through the same underlying SUMO-RL code paths.
- The RESCO summary log is the canonical run artifact for comparing against the benchmark formulas.
