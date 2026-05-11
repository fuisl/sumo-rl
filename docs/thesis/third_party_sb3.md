---
title: Third-Party SB3
firstpage:
---

# Third-Party SB3

This page covers the Stable-Baselines3 examples for the RESCO thesis scenarios.
If you are looking for the third-party MARL replication track for `IDQN`, `MPLight`, and `IPPO`, read [docs/thesis/third_party_marl.md](third_party_marl.md) instead.

## What To Run

The main SB3 entrypoints are:

```bash
python experiments/dqn.py
python experiments/ql.py
python experiments/sac.py
python experiments/ppo.py
```

The canonical configs now live under the scenario-first preset folders:

- `configs/presets/resco_cologne1/`
- `configs/presets/resco_cologne3/`
- `configs/presets/resco_ingolstadt1/`
- `configs/presets/resco_ingolstadt7/`
- The exact folder pattern is documented in [`configs/presets/README.md`](../../configs/presets/README.md)

The DQN, Q-learning, PPO, and SAC presets use the discrete RESCO traffic-signal control setups in those folders.
SAC uses the parallel PettingZoo environment through a thin joint-action wrapper so it can train on the same discrete traffic-signal actions as the other methods.

## Install

To run the SB3 examples, install the optional dependencies:

```bash
pip install -e ".[experiments]"
```

This includes Hydra, W&B, and the extra `supersuit` dependency needed for the multi-agent SB3 examples.

## Metrics

The runner logs both summary and training traces:

- `resco_*` for benchmark-style episode summaries
- `efficiency_*` for queue, speed, and throughput aggregates
- `fairness_*` for Jain fairness over per-agent waiting times
- `safety_*` for emergency braking and unsafe-event proxies
- `train/*` for learning traces
- `eval/*` for rolling evaluation checks during training

## Example Runs

```bash
python experiments/dqn.py scenario=resco_cologne3
python experiments/ql.py scenario=resco_ingolstadt1
python experiments/ppo.py scenario=resco_ingolstadt7
python experiments/sac.py scenario=resco_ingolstadt7
```

The PPO example is now scenario-first as well, so you can point it at `resco_cologne1`, `resco_cologne3`, `resco_ingolstadt1`, or `resco_ingolstadt7`.

## How To Read The Outputs

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

- The SB3 examples now focus on the scenario-first RESCO presets instead of a single legacy `grid4x4` path.
- The SAC path is deliberately centralized through the joint-action wrapper so it can smoke-test on the same discrete action set as the other methods.
- All examples are wired to the shared Hydra/W&B experiment runner.
