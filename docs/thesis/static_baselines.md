---
title: Static Baselines
firstpage:
---

# Static Baselines

This page covers the non-learning static baselines for the thesis RESCO scenarios.
It also matches the fixed-time benchmark style used in the thesis, which follows the same five-seed averaging pattern.
The logged summary values follow the RESCO formulas:

- `resco_avg_delay` from all non-ghost tripinfo rows written by SUMO: `timeLoss + departDelay`
- `resco_trip_time` from all non-ghost tripinfo rows written by SUMO: `duration`
- `resco_wait` from all non-ghost tripinfo rows written by SUMO: `waitingTime`
- `resco_queue` and `resco_max_queue` from live queue counts in the simulator

Tripinfo rows for vehicles that are still running or never departed when the episode ends are counted separately and included in the RESCO averages, matching the benchmark tripinfo parser.

## What to Run

The static baselines in this thesis use:

- `resco_cologne1`
- `resco_cologne3`
- `resco_cologne8`
- `resco_ingolstadt1`
- `resco_ingolstadt7`
- `resco_ingolstadt21`
- `num_seconds: 3600`
- `episodes: 1`
- `seeds: [1, 2, 3, 4, 5]`
- `eval_seeds: [1, 2, 3, 4, 5]`

The runner executes one validation-style episode per seed, averages those seed results, and replays the aggregated RLlib-style `validation/*` metrics on the episode axis for W&B comparison.

## Max Pressure

Run:

```bash
python experiments/static_max_pressure.py
python experiments/static_max_pressure.py scenario=resco_ingolstadt7
python experiments/static_max_pressure.py -m scenario=resco_cologne1,resco_cologne3,resco_cologne8,resco_ingolstadt1,resco_ingolstadt7,resco_ingolstadt21
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
- optional W&B logs if enabled
- one averaged RLlib-style `validation/*` baseline row replayed every 5 episodes through episode 500 by default
- agent-level metrics stay local in the CSV when you enable them, and are not sent to W&B
- one RLlib-style validation media bundle per run: action share, action timeline, phase queue, and tripinfo distributions

## Horizontal Baselines In W&B

To draw fixed-time or max-pressure as horizontal baselines in the same validation panels as RLlib runs:

- the default span is `logging.baseline_line_max_episode_index=500`
- the default cadence is `logging.baseline_line_episode_stride=5`

This re-logs the same aggregated `validation/*` values at `validation/episode_index = 5, 10, ..., 500`, so W&B renders a flat comparison line instead of a single point. The validation media artefacts are still logged once per run.

## Suggested Reading Order

1. [`docs/thesis/experiments.md`](experiments.md)
2. This page
3. [`docs/thesis/manual_control.md`](manual_control.md)

## Notes

- This is a static benchmark layer, separate from the RL baselines.
- The static policies are intended to be simple, reproducible comparators against fixed-time control.
- The scenario-specific static presets now live under `configs/presets/<scenario>/`.
