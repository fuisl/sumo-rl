---
title: Thesis Experiments
firstpage:
---

# Thesis Experiments

This page contains the thesis-specific experiment workflow built on top of the upstream SUMO-RL examples.

If you are looking for fixed-time/manual traffic control, read [docs/thesis/manual_control.md](manual_control.md) after this page.
If you want the RESCO static baselines, read [docs/thesis/static_baselines.md](static_baselines.md) next.
If you want the SB3 examples, read [docs/thesis/third_party_sb3.md](third_party_sb3.md) next.
If you want the third-party MARL replication path for IDQN, MPLight, and IPPO, read [docs/thesis/third_party_marl.md](third_party_marl.md) next.

The thesis launchers now use generic method entrypoints plus Hydra scenario overrides for the scenario-first names `resco_cologne1`, `resco_cologne3`, `resco_ingolstadt1`, and `resco_ingolstadt7`.

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
  - scenario-first presets such as `configs/presets/two_way_single_intersection/dqn.yaml`, `configs/presets/resco_cologne1/ppo.yaml`, and `configs/presets/resco_grid4x4/static_greedy.yaml`

Example:
```bash
python experiments/dqn.py scenario=resco_cologne3
```

## Weights & Biases
Weights & Biases is used for experiment tracking.

- W&B logs configs, metrics, and run metadata
- The repo defaults to offline mode so local runs do not require an API key
- To use online logging, authenticate outside the repo with `wandb login` or set `WANDB_API_KEY` in your environment

Example:
```bash
python experiments/dqn.py scenario=resco_ingolstadt7 logging.mode=online logging.project=my-thesis
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
- Run names now put the scenario first, for example `resco_grid4x4__static_greedy`, `resco_cologne3__dqn`, or `resco_grid4x4__libsignal_mplight`.
