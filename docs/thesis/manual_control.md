---
title: Manual Traffic Control
firstpage:
---

# Manual Traffic Control

This guide explains how to run SUMO-RL in fixed-time mode with Hydra and inspect the results in Weights & Biases.

The fixed-time presets in this thesis use five seeds, one episode per seed, and the runner writes a final average summary.
The summary is computed with RESCO-style formulas:

- `resco_avg_delay` from tripinfo `timeLoss`
- `resco_trip_time` from tripinfo `duration`
- `resco_wait` from tripinfo `waitingTime`
- `resco_queue` and `resco_max_queue` from the live queue counts in the simulator

The per-run identifier in the logs is `run_seed`, not the base config seed.

## How to Read These Docs

Read in this order:

1. [`README.md`](../../README.md) for the upstream SUMO-RL overview and original examples.
2. [`docs/thesis/experiments.md`](experiments.md) for the thesis-specific Hydra and W&B setup.
3. This page for fixed-time/manual control commands.
4. [`configs/`](../../configs) if you want to inspect or override the exact experiment presets.

## What "Manual" Means Here

In this project, manual traffic control means fixed-time control:

- the signal phases come from the SUMO network or route definition
- the environment ignores RL actions
- no external RL library is required
- you still get CSV logs and optional W&B tracking

## Run Single Intersection

Use the generic fixed-time launcher with the desired scenario:

```bash
python experiments/fixed_time.py scenario=single_intersection
```

If you want a different seed or GUI mode, override Hydra values:

```bash
python experiments/fixed_time.py scenario=single_intersection experiment.seed=7 env.kwargs.use_gui=true
```

The preset uses:

- `seeds: [1, 2, 3, 4, 5]`
- `runs: 5`
- `episodes: 1`
- `num_seconds: 3600`

## Run Grid Fixed-Time

Use the RESCO grid scenario with the same generic launcher:

```bash
python experiments/fixed_time.py scenario=resco_grid4x4
```

This preset uses:
- `sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml`
- `sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml`

It does not use the Lucas `4x4-Lucas` files.

You can also override values directly:

```bash
python experiments/fixed_time.py scenario=resco_grid4x4 experiment.seed=13 logging.mode=online
```

Like the single-intersection version, the grid preset runs five seeds by default and averages the final result.

## Inspect with W&B

The default W&B mode is offline, so local runs work without a key.

To inspect a run:

- Look at the resolved config in the W&B run page
- Check the episode-level metrics like waiting time, queue, speed, and reward
- Compare runs by seed or preset name

If you want online logging:

1. Authenticate once with `wandb login`, or set `WANDB_API_KEY` in your shell.
2. Run with `logging.mode=online`.
3. Open the W&B project page and compare runs there.

If you stayed offline, you can later sync runs with:

```bash
wandb sync <path-to-offline-run>
```

For quick debugging, the same metrics are also written locally to:

- `outputs/<experiment-name>/<timestamp>/logs/metrics.csv`
- `outputs/<experiment-name>/<timestamp>/tripinfo/` for the raw SUMO tripinfo XML

The CSV and W&B logs now use the RESCO summary fields directly so you can compare them against the benchmark formulas.
If you turn on per-agent logging, those extra agent metrics stay in the local CSV only.

## Notes

- Fixed-time presets are driven by Hydra config files in `configs/`.
- The 4x4 fixed-time preset combines `configs/scenario/resco_grid4x4.yaml` with `configs/algorithm/fixed_time.yaml`.
- The fixed-time presets use five seeds and one episode per seed so the final result is an average over the five runs.
- The output CSV still lands under the Hydra run directory, alongside the W&B metadata.
- If a run does not look right, first inspect the resolved Hydra config and then the saved CSV metrics.
