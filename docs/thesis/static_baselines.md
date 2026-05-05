---
title: Static Baselines
firstpage:
---

# Static Baselines

This page covers the non-learning static baselines for the RESCO `grid4x4` scenario.
It also matches the fixed-time benchmark style used in the thesis, which now follows the same five-seed averaging pattern.
The logged summary values follow the RESCO formulas:

- `resco_avg_delay` from tripinfo `timeLoss`
- `resco_trip_time` from tripinfo `duration`
- `resco_wait` from tripinfo `waitingTime`
- `resco_queue` and `resco_max_queue` from live queue counts in the simulator

The per-run identifier in the logs is `run_seed`, not the base config seed.

## What to Run

The static baselines in this thesis use:

- RESCO `grid4x4`
- `num_seconds: 3600`
- `episodes: 1`
- `seeds: [1, 2, 3, 4, 5]`

The runner executes one episode per seed and then writes a summary average across the five runs.

## Max Pressure

Run:

```bash
python experiments/static_max_pressure_resco_grid4x4.py
```

What it uses:
- [`configs/static/max_pressure_resco_grid4x4.yaml`](../../configs/static/max_pressure_resco_grid4x4.yaml)
- [`configs/scenario/resco_grid4x4.yaml`](../../configs/scenario/resco_grid4x4.yaml)
- the new static Max Pressure controller in `sumo_rl/agents/static/`

## Greedy

Run:

```bash
python experiments/static_greedy_resco_grid4x4.py
```

What it uses:
- [`configs/static/greedy_resco_grid4x4.yaml`](../../configs/static/greedy_resco_grid4x4.yaml)
- [`configs/scenario/resco_grid4x4.yaml`](../../configs/scenario/resco_grid4x4.yaml)
- the queue-based Greedy controller in `sumo_rl/agents/static/`

## Outputs

Each run writes:

- Hydra output under `outputs/<experiment-name>/<timestamp>/`
- per-run CSV metrics under `logs/metrics.csv`
- raw SUMO tripinfo XML under `tripinfo/`
- a final summary row with the average across the five seeds
- optional W&B logs if enabled
- the RESCO summary fields are logged directly, so the CSV and W&B logs match the benchmark formulas
- agent-level metrics stay local in the CSV when you enable them, and are not sent to W&B

## Suggested Reading Order

1. [`docs/thesis/experiments.md`](experiments.md)
2. This page
3. [`docs/thesis/manual_control.md`](manual_control.md)

## Notes

- This is a static benchmark layer, separate from the RL baselines.
- The static policies are intended to be simple, reproducible comparators against Q-learning and SB3.
