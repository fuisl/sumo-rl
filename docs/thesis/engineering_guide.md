---
title: Thesis Engineering Guide
firstpage:
---

# Thesis Engineering Guide

This guide is written for someone joining the repo and needing to become productive quickly.
It focuses on the parts that matter most for thesis experiments:

- how the runner works
- how different RL backends are wrapped
- how metrics flow into CSV and W&B
- what is shared across methods
- what you must configure per model
- how to add a new model safely

## Start Here

Read the code in this order:

1. `experiments/*.py`
2. `sumo_rl/experiments/runner.py`
3. `sumo_rl/experiments/rllib_runner.py`
4. `sumo_rl/environment/env.py`
5. `sumo_rl/environment/traffic_signal.py`
6. `sumo_rl/experiments/metric_utils.py`
7. `sumo_rl/rllib/envs.py`
8. `sumo_rl/rllib/custom_sac.py`
9. the matching config files in `configs/`

If you only have one hour, read:

- `runner.py`
- `env.py`
- `metric_utils.py`
- the preset you are about to run

## Mental Model

The repo has one shared experiment spine:

1. a thin launcher under `experiments/`
2. a Hydra config assembled from `configs/`
3. `sumo_rl.experiments.runner.run(cfg)`
4. an algorithm-specific branch inside the runner
5. shared metric helpers and shared output layout

The key idea is that most methods should differ in only three places:

- how the environment is wrapped
- how the model is trained or evaluated
- how external outputs are normalized before logging

Everything else should stay shared.

## How The Runner Works

### 1. Entry points

Files like `experiments/fixed_time.py`, `experiments/static_max_pressure.py`, and `experiments/rllib.py` do almost nothing.
They:

- check `SUMO_HOME`
- load a Hydra config
- call `run(cfg)`

That means the real behavior lives in the runner, not the launcher script.

The same pattern now applies to `experiments/rllib.py`, which calls
`sumo_rl.experiments.rllib_runner.train_rllib(cfg)`.

### 2. Shared setup in `runner.run`

`sumo_rl/experiments/runner.py` does the common setup:

- creates the Hydra run directory
- seeds NumPy and Python random
- starts W&B if enabled
- creates the local metrics CSV logger
- dispatches to the correct algorithm branch using `cfg.algorithm.kind`

The shared output layout is:

- `outputs/<experiment-name>/<timestamp>/logs/metrics.csv`
- `outputs/<experiment-name>/<timestamp>/csv/`
- `outputs/<experiment-name>/<timestamp>/tripinfo/`
- `outputs/<experiment-name>/<timestamp>/checkpoints/`
- `outputs/<experiment-name>/<timestamp>/ray_results/` when Ray writes trial state

### 3. Environment construction

The runner builds envs with `_build_env(cfg, run_dir, seed=None)`.
That function reads:

- `cfg.env.factory`
- `cfg.env.kwargs`
- auto-filled output paths such as `out_csv_name` and `tripinfo_output_name`

Important factories:

- `sumo_env`: direct `SumoEnvironment`
- `env`: PettingZoo AEC env
- `parallel_env`: PettingZoo parallel env
- `fixed_time_env`: PettingZoo env with `fixed_ts=True`

### 4. Algorithm dispatch

The runner has one branch per integration style:

- Q-learning: `_run_direct_q_learning`, `_run_aec_q_learning`
- Static baselines: `_run_fixed_time`, `_run_static_policy`
- RLlib methods: `sumo_rl.experiments.rllib_runner.train_rllib`

When you add a new model, you are usually adding one new branch here.

## How The Environment Works

`sumo_rl/environment/env.py` is the source of truth for simulation-side metrics.

Each step does this:

1. advance SUMO
2. compute observations
3. compute rewards
4. compute `info`
5. append the `info` snapshot to `self.metrics`

Important state in `SumoEnvironment`:

- `metrics`: list of live step-info rows for the current episode
- `last_episode_summary`: cached episode summary from the completed episode
- `last_episode_final_info`: cached final live-info row from the completed episode
- `last_episode_lane_waiting_times`: cached lane waiting-time snapshot from the completed episode

Those last three caches matter because some wrappers and evaluation APIs auto-reset the env after `done`.

## How Metrics Flow

### Live step metrics

`_compute_info()` builds the per-step `info` dict.
It can include:

- `step`
- `system_*` when `add_system_info=true`
- per-agent waiting-time fields when `add_per_agent_info=true`

### Episode summary metrics

At episode end, `finalize_episode_summary()` builds the benchmark summary from:

- tripinfo XML for `resco_*`
- recorded queue statistics for `resco_queue` and `resco_max_queue`
- cached final live-info state for `efficiency_*` and `safety_*`

The runner then converts that cached episode state into the final summary row with:

- `_build_resco_summary_row(...)`

### W&B and CSV

There are two logging sinks:

- `_LocalMetricsCsvLogger`
- W&B via `wandb_run.log(...)`

The final episode summary rows also update `wandb.run.summary`.
That was added specifically to avoid the old "history looked fine but W&B summary showed zero" failure mode.

## How Different RL Libraries Are Wrapped

### Direct Q-learning

This is the simplest path.

- env is either direct gym-style or PettingZoo AEC
- the repo creates `QLAgent` objects directly
- the runner controls the train loop itself
- metric logging happens in the runner

Use this path as the easiest reference when learning the codebase.

### DQN and PPO through RLlib

These methods use the PettingZoo parallel env and then hand it to RLlib as a shared-policy multi-agent environment.
In the current thesis config, PPO and DQN default to independent policies, so each
traffic signal gets its own policy ID. You can still switch back to a shared policy
by setting `algorithm.params.policy_mode=shared`.

The flow is:

1. build parallel env
2. optionally pad heterogeneous spaces with SuperSuit
3. register the env with RLlib
4. map every traffic-signal agent to one shared policy
5. train through the RLlib algorithm config

Key files:

- `sumo_rl/experiments/rllib_runner.py`
- `sumo_rl/rllib/envs.py`

The runner logs:

- `train/*`
- evaluation summaries
- final RESCO summaries

For thesis-style runs, evaluation should use a reproducible seed schedule rather than repeating one seed for every eval episode. The repo supports `experiment.eval_seeds`, and the RESCO benchmark presets use the fixed-time pattern `[1, 2, 3, 4, 5]` so eval mean/std and the final traffic summary are both computed over distinct seeded episodes.

The final benchmark row is not created by the callback.
It is created by the runner after a dedicated final evaluation pass.

### SAC through RLlib

RLlib SAC expects a continuous action space.

For the compatibility baseline, the repo uses a joint-action adapter so the same
traffic-signal scenario can be trained without rewriting the simulator:

- flattens all agent observations into one vector
- exposes one continuous action vector
- decodes each action slice with `argmax` into a discrete action per signal
- sums the per-agent rewards by default

That means RLlib is not "converting" discrete actions into continuous actions on
its own. Our adapter changes the environment interface from discrete per-signal
actions to one continuous Box action space, and SAC learns on that Box space.

### Custom SAC module

The custom SAC path keeps the same joint-action environment but swaps in a
project-owned RLModule boundary.

This is the right place to change:

- the encoder architecture
- the policy and Q-head layout
- the training loop cadence in `rllib_runner.py`
- checkpoint timing and evaluation behavior

## Shared Pieces Across Methods

These are the parts you should preserve when adding new algorithms:

- Hydra config layout
- run directory structure
- tripinfo output storage
- local CSV metrics logger
- `resco_*` benchmark fields
- namespaced `efficiency/*` and `safety/*`
- final W&B summary update

If a new method cannot produce the shared schema directly, add a small normalization layer.
Do not fork the whole logging design unless absolutely necessary.

## What Must Be Configured Per Model

Every method needs its own answer for these questions:

### Environment API shape

- direct gym-like env
- PettingZoo AEC env
- PettingZoo parallel env
- external library env

### Reward semantics

- scalar reward
- reward dict
- averaged reward
- upstream third-party reward

### Training loop ownership

- runner-owned loop
- RLlib-owned algorithm loop with a thin project wrapper

### Evaluation ownership

- runner evaluates manually
- runner evaluates at the end of the training run
- custom logic can still add periodic evaluation later

### Config files

At minimum you normally need:

- one launcher in `experiments/`
- one algorithm config in `configs/algorithm/`
- one shared RLlib root config in `configs/rllib.yaml`
- optional scenario presets in `configs/presets/<scenario>/`

## How To Add A New Model From An Existing RL Library

Use this recipe if the library can already consume a Gym, VecEnv, or PettingZoo-compatible env.

1. Decide the env shape the library expects.
2. Reuse an existing wrapper if possible.
3. If needed, write the thinnest new wrapper in `sumo_rl/rllib/`.
4. Add a new runner branch in `rllib_runner.py`.
5. Reuse `_build_resco_summary_row(...)` and `_log_episode_summary(...)` for the final benchmark row.
6. Make sure the method logs final summaries from the completed episode cache, not from the post-reset env state.
7. Add the launcher script.
8. Add the algorithm config.
9. Add matching scenario presets.
10. Add a short smoke-test command to your notes or docs.

Good fit for this path:

- another RLlib algorithm
- a library that accepts Gymnasium, VecEnv, or PettingZoo with only a small adapter

## How To Add A New Model From Outside The Current RL Stack

Use this recipe if the model should stay in its original repository.

1. Keep the external training code external.
2. Add a thin launcher or CLI bridge under `sumo_rl/integrations/`.
3. Export one normalized summary artifact from the external run.
4. Convert that artifact into the shared thesis row shape.
5. Copy the raw upstream logs into the Hydra run directory for auditability.
6. Log one per-seed row and one final aggregate row if the run is seed-based.
7. Document clearly which metrics are native upstream fields and which are thesis-side proxies.

Good fit for this path:

- external libraries with their own trainer and summary format
- codebases with their own trainer, logger, and simulator setup

## Metrics Checklist For Any New Method

Before you trust a new model, manually check these files:

1. `outputs/<run>/logs/metrics.csv`
2. `outputs/<run>/tripinfo/*.xml`
3. W&B history plots
4. W&B run summary

Confirm all of the following:

1. `resco_avg_delay`, `resco_trip_time`, and `resco_wait` are non-zero when vehicles completed trips.
2. `efficiency_*` and `safety_*` are present when the env logged system info.
3. the final W&B summary matches the final CSV summary row
4. the reward curves and the benchmark metrics tell a consistent story
5. multi-seed methods keep one per-seed row plus one final aggregate row

If the final summary row is zero again, debug in this order:

1. confirm the evaluation episode actually ran to completion
2. inspect `last_episode_summary` and `last_episode_final_info`
3. check whether the env auto-reset before the final row was built
4. confirm `tripinfo_output_name` was set and the XML file exists
5. confirm the external bridge or wrapper is not dropping the final `info` state

## Most Important Files To Know

If you are going to work on training and logging often, keep these open side by side:

- `sumo_rl/experiments/runner.py`
- `sumo_rl/experiments/rllib_runner.py`
- `sumo_rl/experiments/metric_utils.py`
- `sumo_rl/environment/env.py`
- `sumo_rl/environment/traffic_signal.py`
- `sumo_rl/rllib/envs.py`
- `sumo_rl/rllib/custom_sac.py`

That set covers almost every debugging session you will have in this repo.

## Recommended First Tasks For A New Contributor

If you are onboarding, these are good first exercises:

1. Run one short RLlib DQN preset and inspect the output directory.
2. Trace where one metric such as `resco_avg_delay` is created and logged.
3. Trace how PPO reaches RLlib from the launcher.
4. Trace how SAC converts multi-agent discrete actions into a single continuous joint action space.
5. Read one RLlib preset and compare it with one static baseline preset.

After that, the rest of the codebase gets much easier to navigate.
