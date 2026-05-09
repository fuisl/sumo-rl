---
title: Third-Party SB3
firstpage:
---

# Third-Party SB3

This page covers the working Stable-Baselines3 examples for the RESCO 4x4 grid.
If you are looking for the third-party MARL replication track for `IDQN`, `MPLight`, and `IPPO`, read [docs/thesis/third_party_marl.md](third_party_marl.md) instead.

## What to Run

The thesis configs use the RESCO `grid4x4` assets, not the older Lucas `4x4-Lucas` network.

## Install

To run the SB3 examples, install the optional dependencies:

```bash
pip install -e ".[experiments]"
```

This includes Hydra, W&B, and the extra `supersuit` dependency needed for the multi-agent PPO example.

The config layout follows the same split used elsewhere in the thesis:

- `configs/scenario/` for the RESCO `grid4x4` network
- `configs/algorithm/` for the SB3 defaults
- the scenario-first `configs/presets/resco_grid4x4/ppo.yaml` preset for the runnable command

### Stable-Baselines3 PPO on RESCO 4x4

This is the main multi-agent example for the thesis:

```bash
python experiments/ppo.py scenario=resco_grid4x4 env.factory=grid4x4
```

What it uses:
- [`configs/presets/resco_grid4x4/ppo.yaml`](../../configs/presets/resco_grid4x4/ppo.yaml)
- RESCO `grid4x4` network and route files
- PettingZoo parallel environment wrapped for SB3
- `stable-baselines3` and `supersuit`

### Alternate Launcher for Stable-Baselines3 PPO

This is the same SB3 multi-agent setup under the generic launcher with overrides:

```bash
python experiments/ppo.py scenario=resco_grid4x4 env.factory=grid4x4
```

What it uses:
- [`configs/presets/resco_grid4x4/ppo.yaml`](../../configs/presets/resco_grid4x4/ppo.yaml) plus the `scenario=resco_grid4x4` override
- RESCO `grid4x4` through the helper factory
- vectorized PettingZoo -> SB3 conversion
- `stable-baselines3` and `supersuit`

## How to Read the Outputs

Use the Hydra output directory for the resolved config and logs:

- `outputs/<experiment-name>/<timestamp>/`

Use W&B to compare:

- training curves
- run metadata
- resolved config
- differences between SB3 runs and manual baselines

## Suggested Reading Order

1. [`docs/thesis/experiments.md`](experiments.md)
2. This page
3. [`docs/thesis/manual_control.md`](manual_control.md)

## Notes

- The SB3 example is the primary multi-agent third-party baseline for RESCO 4x4.
- The alternate launcher is useful as a sanity check and naming alias.
- Both examples are already wired to the shared Hydra/W&B experiment runner.
