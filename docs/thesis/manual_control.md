---
title: Manual Traffic Control
firstpage:
---

# Manual Traffic Control

This guide explains how to run SUMO-RL in fixed-time mode with Hydra and inspect the results in Weights & Biases.

The fixed-time 4x4 preset in this thesis uses the RESCO `grid4x4` network and route files.

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

Use the dedicated fixed-time launcher:

```bash
python experiments/fixed_time_single_intersection.py
```

If you want a different seed or GUI mode, override Hydra values:

```bash
python experiments/fixed_time_single_intersection.py experiment.seed=7 env.kwargs.use_gui=true
```

## Run Grid Fixed-Time

Use the dedicated grid launcher:

```bash
python experiments/fixed_time_4x4grid.py
```

This preset uses:
- `sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml`
- `sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml`

It does not use the Lucas `4x4-Lucas` files.

You can also override values directly:

```bash
python experiments/fixed_time_4x4grid.py experiment.seed=13 logging.mode=online
```

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

## Notes

- Fixed-time presets are driven by Hydra config files in `configs/`.
- The output CSV still lands under the Hydra run directory, alongside the W&B metadata.
- If a run does not look right, first inspect the resolved Hydra config and then the saved CSV metrics.
