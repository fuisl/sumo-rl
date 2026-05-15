---
hide-toc: true
firstpage:
lastpage:
---

# Metrics Reference

This page documents the metric pipeline used by the thesis runner.
It covers:

- the formula behind each metric
- the source of each metric input
- when the metric is logged
- the corrections made after the final-summary caching bug

The current source of truth is:

- `sumo_rl/environment/env.py`
- `sumo_rl/environment/traffic_signal.py`
- `sumo_rl/experiments/metric_utils.py`
- `sumo_rl/experiments/runner.py`
- `sumo_rl/experiments/rllib_runner.py`

## Logging Stages

The repo logs metrics at four different stages.

| Stage | Producer | Step axis | Main payload |
| --- | --- | --- | --- |
| Live env step | `SumoEnvironment._compute_info()` | `info["step"]` in SUMO seconds | `system_*`, optional per-agent waiting-time fields |
| Training trace | algorithm runner | training timesteps or episode boundary | `train/*`, `eval/*`, per-episode summaries |
| Episode summary | `runner._build_resco_summary_row(...)` | final episode step or final training timestep | `resco_*`, `efficiency_*`, `safety_*`, reward metadata |
| Run summary | runner aggregate helpers plus `wandb.run.summary` | final run write | `summary/*` seed averages for multi-run methods and pinned final W&B summary values |

## Namespaces

The runner uses these namespaces:

- `resco/*`: benchmark-style episode summary
- `efficiency/*`: network flow and throughput diagnostics
- `safety/*`: safety and instability proxies
- `train/*`: training traces
- `eval/*`: periodic evaluation traces
- `summary/*`: run-level aggregate rows, mostly seed averages

The underscore aliases such as `resco_avg_delay` are still logged for compatibility.
W&B gets the slash aliases too, so the dashboard groups stay readable.

## Reward Signals

The environment reward is produced by `TrafficSignal.compute_reward()`.
For thesis presets, the default reward is `diff-waiting-time` unless a preset overrides it.

The runner also logs reward metadata in the final episode summary:

- `reward/name`
- `reward/formula`
- `reward/source`
- `reward/scope`

### Built-in reward functions

| Reward | Formula in code | Inputs | Output meaning |
| --- | --- | --- | --- |
| `diff-waiting-time` | `last_ts_waiting_time - current_ts_waiting_time` | `current_ts_waiting_time = sum(get_accumulated_waiting_time_per_lane()) / 100` | Positive is better |
| `average-speed` | `TrafficSignal.get_average_speed()` | per-vehicle `speed / allowed_speed` on incoming lanes | Higher is better |
| `queue` | `-TrafficSignal.get_total_queued()` | halting-vehicle count on incoming lanes | Higher is better because the queue is negated |
| `pressure` | `TrafficSignal.get_pressure()` | outgoing vehicle count minus incoming vehicle count | Diagnostic reward with implementation-specific sign |
| `co2` | `-TrafficSignal.get_total_co2()` | SUMO CO2 emissions on incoming lanes | Higher is better because emissions are negated |

### Training reward traces

| Metric | Producer | Inputs | Logged when |
| --- | --- | --- | --- |
| `train/reward_mean` | algorithm runner | latest reward batch from the trainer result | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `train/episode_reward` | algorithm runner | latest episode return snapshot from the trainer result | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `train/resco/*` | algorithm runner | cached RESCO episode summary from the completed training episode | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `eval/mean_reward` | algorithm runner or final evaluation pass | evaluation rollout output | final evaluation summary only |
| `eval/std_reward` | algorithm runner or final evaluation pass | evaluation rollout output | final evaluation summary only |

For SAC, the joint wrapper reduces the per-agent reward dictionary to one scalar before handing it to the learner.
That means SAC training rewards are not directly comparable to the per-agent raw reward vector.

## Shared Episode Summary Metrics

The final episode summary is built from cached data from the last completed episode.
This matters because evaluation envs auto-reset after a done episode.
The runner now uses cached episode data instead of reading the fresh post-reset state.

### Benchmark metrics

These are the main thesis comparison metrics.

| Metric | Formula | Inputs | Logged when |
| --- | --- | --- | --- |
| `resco_avg_delay` | `mean(timeLoss + departDelay)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_trip_time` | `mean(duration)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_wait` | `mean(waitingTime)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_tripinfo_count` | count of completed non-ghost tripinfo rows | SUMO tripinfo XML | episode summary |
| `resco_queue` | mean of recorded queue-per-signal values across the episode | `system_mean_queued`, else `system_total_queued / num_signals`, else `system_total_stopped / num_signals` | episode summary |
| `resco_max_queue` | max recorded queue-per-signal value across the episode | `system_max_queue` from live step info | episode summary |

Interpretation:

- lower is better for delay, trip time, wait, and queue
- `resco_queue` is an episode average
- `resco_max_queue` is a worst-case episode peak

### Efficiency and safety metrics

These come from the final cached live-info row of the completed episode, not from tripinfo.

| Metric | Formula or definition | Inputs | Logged when |
| --- | --- | --- | --- |
| `efficiency_total_running` | count of non-ghost vehicles currently in network | `sumo.vehicle.getIDList()` | episode summary and any direct live-info logging |
| `efficiency_total_backlogged` | count of pending non-ghost vehicles not yet inserted | `sumo.simulation.getPendingVehicles()` | episode summary and live info |
| `efficiency_total_stopped` | count of vehicles with `speed < 0.1` | per-vehicle SUMO speed | episode summary and live info |
| `efficiency_total_queued` | sum of `TrafficSignal.get_total_queued()` over signals | per-signal halting counts | episode summary and live info |
| `efficiency_mean_queued` | mean queue count per signal | per-signal halting counts | episode summary and live info |
| `efficiency_max_queue` | max queue count among signals | per-signal halting counts | episode summary and live info |
| `efficiency_total_arrived` | cumulative arrived vehicles | `sumo.simulation.getArrivedNumber()` | episode summary and live info |
| `efficiency_total_departed` | cumulative departed vehicles | `sumo.simulation.getDepartedNumber()` | episode summary and live info |
| `efficiency_total_waiting_time` | sum of vehicle `waitingTime` values in the current step | per-vehicle SUMO waiting time | episode summary and live info |
| `efficiency_mean_waiting_time` | mean of vehicle `waitingTime` values in the current step | per-vehicle SUMO waiting time | episode summary and live info |
| `efficiency_mean_speed` | mean raw vehicle speed in m/s in the current step | per-vehicle SUMO speed | episode summary and live info |
| `efficiency_mean_average_speed` | mean normalized speed ratio per signal | `TrafficSignal.get_average_speed()` | episode summary and live info |
| `efficiency_mean_pressure` | mean raw signal pressure | `TrafficSignal.get_pressure()` | episode summary and live info |
| `safety_total_teleported` | cumulative teleported vehicles | `sumo.simulation.getEndingTeleportNumber()` | episode summary and live info |
| `safety_total_emergency_brake` | cumulative emergency stopping events | `sumo.simulation.getEmergencyStoppingVehiclesNumber()` | episode summary and live info |
| `safety_total_collisions` | cumulative colliding vehicles | `sumo.simulation.getCollidingVehiclesNumber()` or `sumo.simulation.getCollidingVehiclesIDList()` | episode summary and live info |
| `final/resco_wait_std` | std of per-vehicle `waitingTime` in the eval episode | tripinfo XML | final evaluation summary |
| `final/resco_avg_delay_std` | std of per-vehicle `timeLoss + departDelay` in the eval episode | tripinfo XML | final evaluation summary |

Important interpretation note:

- `efficiency_mean_speed` is raw network speed in m/s
- `efficiency_mean_average_speed` is a normalized signal-local ratio in `[0, 1]`
- `efficiency_mean_pressure` is a diagnostic sign-sensitive quantity, not a universal "lower is better" score

For pressure, values closer to zero are usually easier to compare than raw signed values by themselves.

## Algorithm-Specific Notes

### Q-learning

The direct and AEC Q-learning runners log:

- `train/episode_reward`
- `train/td_error_mean`
- `train/td_error_abs_mean`
- one episode summary row per episode
- one `summary/*` aggregate row across collected episodes

### DQN and PPO through RLlib

These runners use:

- PettingZoo parallel env
- SuperSuit padding when the RESCO scenario has heterogeneous agent spaces
- RLlib shared-policy multi-agent training
- a final explicit evaluation pass after training

For multi-seed evaluation, the runner builds one cached summary per eval seed and then averages the numeric `final/*`, `tripinfo/*`, and episode-time fields into one final row.
Warnings are kept if any eval seed shows the problem.

### SAC through RLlib

SAC uses a single joint-action adapter around the same PettingZoo env:

- flattened joint observation
- continuous Box action vector decoded back to per-agent discrete actions
- scalar reward equal to the sum of the per-agent rewards by default

## Problems Found And Corrected

The repo previously had a few metric-logging problems.

### 1. Final summary rows could read the new reset episode instead of the completed one

What happened:

- evaluation envs auto-reset after `done`
- the old summary builder sometimes read `env.metrics` and lane snapshots after that reset
- that produced zeros or default values in the final summary row even when the training curves looked fine

Current fix:

- `SumoEnvironment.finalize_episode_summary()` now caches:
  - `last_episode_summary`
  - `last_episode_final_info`
  - `last_episode_lane_waiting_times`
- the runner summary path reads those cached values first

### 2. W&B summary values depended too much on history-last-value behavior

What happened:

- W&B uses the last logged value for a metric key unless you set `run.summary` explicitly
- that is fragile for final benchmark metrics

Current fix:

- final episode summary rows now also update `wandb.run.summary`
- aggregate `summary/*` rows also update `wandb.run.summary`

### 3. Metric helper logic was duplicated

What happened:

- metric formula helpers existed in both `runner.py` and `metric_utils.py`
- that makes drift easy

Current fix:

- the runner now delegates the shared metric math to `metric_utils.py`

### 4. The pressure reward text had the wrong sign

What happened:

- the old docs text described pressure as inbound minus outbound
- the actual code is outgoing minus incoming

Current fix:

- the docs and reward metadata now use the real formula from `TrafficSignal.get_pressure()`

## Manual Validation Checklist For New Models

Use this checklist whenever you add PPO, DQN, SAC, or any future method.

1. Make sure the env config keeps `add_system_info: true`, `add_per_agent_info: true`, and a non-null `tripinfo_output_name`.
2. Run a short smoke experiment first, not the full horizon.
3. If you need to inspect raw XML, run with `logging.save_tripinfo_output=true`, then open `outputs/<run>/tripinfo/*.xml` and confirm completed vehicles are present.
4. Open `outputs/<run>/logs/metrics.csv` and confirm the final row has non-zero `resco_*` and non-empty `efficiency_*` and `safety_*` fields when traffic exists.
5. In W&B, compare the run summary values against the final CSV row. They should agree for the final benchmark metrics.
6. If you use a separate evaluation env, make sure the final summary is built from the last completed episode cache, not from the post-reset live env state.
7. If the library has its own reward scale, keep reward plots separate from benchmark metrics in your interpretation.
8. If the library is external, normalize its outputs into the shared schema before logging to W&B.

## Literature And Benchmark Guidance

The current metric choices follow the common traffic-signal-control practice of separating:

- optimization reward
- benchmark traffic outcomes such as delay, travel time, queue, and waiting time
- diagnostic quantities such as safety proxies

For thesis reporting, the strongest comparisons are still:

- same scenario
- same demand file
- same simulation horizon
- multiple seeds
- per-seed rows preserved, with final tables built from those rows

That is also the safer evaluation pattern recommended by deep-RL reproducibility papers: do not rely on a single seed or on reward alone when comparing methods.

## References

- RESCO benchmark paper: James Ault and Guni Sharon, "Reinforcement Learning Benchmarks for Traffic Signal Control", NeurIPS Datasets and Benchmarks 2021.
- SUMO TripInfo fields: https://eclipse.dev/sumo/docs/Simulation/Output/TripInfo.html
- W&B summary behavior: https://docs.wandb.ai/guides/track/log/log-summary/
- Deep RL evaluation guidance:
  - Peter Henderson et al., "Deep Reinforcement Learning that Matters", AAAI 2018
  - Rishabh Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
