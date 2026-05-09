---
title: Third-Party MARL Replication
firstpage:
---

# Third-Party MARL Replication

This page covers the thesis replication track for `IDQN`, `MPLight`, and `IPPO`.
The goal is to use external libraries or codebases with the least possible glue, while keeping the SUMO-RL thesis workflow reproducible and easy to explain.

## What to Run

The recommended third-party paths are:

- `IDQN` -> **LibSignal** as the primary path, **RESCO** as the benchmark reference
- `MPLight` -> **LibSignal** as the primary path
- `IPPO` -> **LibSignal** as the primary path, **EPyMARL** as the fallback if the LibSignal path is not sufficient for the thesis scenario

`FMA2C` is intentionally out of scope for this track.

## Install

Install LibSignal separately from its upstream repository, then point this repo at that checkout:

- `LIBSIGNAL_ROOT=/path/to/LibSignal`
- or `algorithm.params.libsignal_root=/path/to/LibSignal` in the Hydra config

LibSignal itself expects its own dependencies and SUMO setup. For the `IPPO` path, make sure the upstream `pfrl` dependency is available in the LibSignal environment.

This repo only provides the launcher and logging bridge.

## Run

Use the dedicated thesis launchers:

```bash
python experiments/libsignal_idqn.py
python experiments/libsignal_mplight.py
python experiments/libsignal_ippo.py
```

All three launchers use the RESCO `grid4x4` scenario, run one episode per seed by default, and keep the five-seed thesis pattern.
Their canonical config files now live under `configs/presets/resco_grid4x4/` so the scenario-specific settings sit together in one folder.

The LibSignal agent names used by these launchers are:

- `IDQN` -> `dqn`
- `MPLight` -> `mplight`
- `IPPO` -> `ppo_pfrl`

## Why These Choices

LibSignal is the most viable single upstream for the three remaining models because its documentation explicitly lists:

- `IDQN`
- `IPPO`
- `MPLight`

That makes it the best fit for a fast third-party replication pass with one simulator stack and one config style.

RESCO stays in scope for `IDQN` because it provides a direct SUMO benchmark reference and a documented quick-start for `@IDQN`.

EPyMARL stays in scope for `IPPO` only as a fallback path, since it already supports `IPPO` and custom Gym/PettingZoo environments.

## How To Interpret Fidelity

Use this distinction when writing up the thesis:

- **Exact third-party reproduction** means the algorithm runs from the upstream project with its native training code and configuration style.
- **Closest practical replication** means the algorithm is reproduced with the nearest viable third-party backend when the original paper code is not readily runnable in this repo.

For this thesis track:

- `MPLight` is expected to be the cleanest exact third-party run through LibSignal.
- `IDQN` is exact if you use RESCO or LibSignal directly on the matching SUMO scenario.
- `IPPO` is exact if LibSignal is sufficient; otherwise it becomes a closest-practical replication through EPyMARL.

## Logging

The phase 5 bridge keeps the static-baseline logging shape:

- one trace row per seed in both CSV and W&B
- a final seed-average summary row
- the upstream LibSignal `TRAIN` or `TEST` row is preserved as the source of truth

The logged row includes both sets of fields:

- upstream LibSignal metrics: `libsignal/*`
- thesis-side RESCO proxy metrics: `resco_avg_delay`, `resco_trip_time`, `resco_wait`, `resco_queue`, `resco_max_queue`

The RESCO fields are proxies derived from LibSignal's summary log, not raw tripinfo XML, so the thesis text should describe them as benchmark-style comparisons rather than byte-for-byte RESCO reproduction.
The copied upstream log and the normalized per-seed summary are stored under the Hydra run directory in `phase5/libsignal/`.

## Suggested Run Order

1. Start with `MPLight` in LibSignal, because it is explicitly documented and most likely to be plug-and-play.
2. Run `IDQN` next, using RESCO as the benchmark reference and LibSignal as the thesis-friendly path.
3. Run `IPPO` last, first in LibSignal and then in EPyMARL only if the LibSignal setup needs too much adaptation.

## Outputs To Keep

For each run, keep the same artifacts the rest of the thesis uses:

- the resolved experiment config
- a dedicated run directory
- local metrics logs
- any benchmark summary tables or training curves from the upstream library

When you compare results, note which backend produced them:

- `LibSignal`
- `RESCO`
- `EPyMARL`

That keeps the thesis results easy to audit later.

## Suggested Reading Order

1. [`docs/thesis/experiments.md`](experiments.md)
2. This page
3. [`docs/thesis/third_party_sb3.md`](third_party_sb3.md)
4. [`docs/thesis/manual_control.md`](manual_control.md)

## Notes

- This page is intentionally narrower than the SB3 page and focuses only on the three MARL targets.
- The repo stays unchanged for the upstream SUMO-RL examples.
- If a third-party backend needs a small wrapper later, add only the thinnest possible launcher/config layer needed for reproducibility.
