---
title: Static Baselines
firstpage:
---

# Static Baselines

This page covers the non-learning static baselines for the thesis RESCO scenarios.
It also matches the fixed-time benchmark style used in the thesis, which follows the same five-seed averaging pattern.
The logged summary values follow the RESCO formulas:

- `resco_avg_delay` from tripinfo `timeLoss`
- `resco_trip_time` from tripinfo `duration`
- `resco_wait` from tripinfo `waitingTime`
- `resco_queue` and `resco_max_queue` from live queue counts in the simulator

The per-run identifier in the logs is `run_seed`, not the base config seed.

## What to Run

The static baselines in this thesis use:

- `resco_cologne1`
- `resco_cologne3`
- `cologne8`
- `resco_ingolstadt1`
- `resco_ingolstadt7`
- `ingolstadt21`
- `num_seconds: 3600`
- `episodes: 1`
- `seeds: [1, 2, 3, 4, 5]`
- `eval_seeds: [1, 2, 3, 4, 5]`

The runner executes one validation-style episode per seed, logs RLlib-style `validation/*` metrics, and then writes a summary average across the five runs.

## Max Pressure

Run:

```bash
python experiments/static_max_pressure.py
python experiments/static_max_pressure.py scenario=resco_ingolstadt7
python experiments/static_max_pressure.py -m scenario=resco_cologne1,resco_cologne3,cologne8,resco_ingolstadt1,resco_ingolstadt7,ingolstadt21
```

What it uses:
- [`configs/presets/resco_grid4x4/static_max_pressure.yaml`](../../configs/presets/resco_grid4x4/static_max_pressure.yaml)
- [`configs/scenario/resco_grid4x4.yaml`](../../configs/scenario/resco_grid4x4.yaml)
- the new static Max Pressure controller in `sumo_rl/agents/static/`

## Outputs

Each run writes:

- Hydra output under `outputs/<experiment-name>/<timestamp>/`
- per-run CSV metrics under `logs/metrics.csv`
- raw SUMO tripinfo XML under `tripinfo/` only when `logging.save_tripinfo_output=true`
- a final summary row with the average across the five seeds
- optional W&B logs if enabled
- the RESCO summary fields are logged directly, so the CSV and W&B logs match the benchmark formulas
- RLlib-style `validation/*` rows so the static baselines can share W&B panels with RLlib runs
- agent-level metrics stay local in the CSV when you enable them, and are not sent to W&B

## Horizontal Baselines In W&B

To draw fixed-time or max-pressure as horizontal baselines in the same validation panels as RLlib runs:

- set `logging.baseline_line_max_episode_index=<training_episode_budget>`
- optionally set `logging.baseline_line_episode_stride=<validation_cadence>`

This re-logs the same aggregated `validation/*` values at repeated `validation/episode_index` anchors, so W&B renders a flat comparison line instead of a single point.

## Suggested Reading Order

1. [`docs/thesis/experiments.md`](experiments.md)
2. This page
3. [`docs/thesis/manual_control.md`](manual_control.md)

## Notes

- This is a static benchmark layer, separate from the RL baselines.
- The static policies are intended to be simple, reproducible comparators against fixed-time control.
- The scenario-specific static presets now live under `configs/presets/<scenario>/`.
