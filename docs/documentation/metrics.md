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
| Training trace | algorithm runner | training timesteps or episode boundary | shared `train/*` metrics plus optional `debug/*` diagnostics |
| Episode summary | `runner._build_episode_benchmark_summary_row(...)` | final episode step or final training timestep | `resco_*`, `efficiency_*`, `safety_*`, reward metadata |
| Run summary | runner aggregate helpers plus `wandb.run.summary` | final run write | `summary/*` seed averages for multi-run methods and pinned final W&B summary values |

## Namespaces

The runner uses these namespaces:

- `train/*`: training-trace namespace for shared episode-level metrics such as `train/resco_*`, selected throughput totals, and `train/safety_*`
- `validation/*`: periodic eval namespace for train-comparable episode metrics averaged across the eval seeds in that validation pass
- `debug/*`: per-agent reward traces, end-of-episode snapshot diagnostics, and debug-only trainer diagnostics
- `final/*`: final explicit evaluation summary metrics
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

## Trace Modes

Training trace logging is controlled by `logging.trace_mode`.
The default is `training`.

| Mode | Always logged | Extra logged | Main intent |
| --- | --- | --- | --- |
| `training` | shared `train/*` metrics and `debug/reward/<agent_id>` | none | keep the thesis-facing training trace compact |
| `debug` | the same shared metrics and `debug/reward/<agent_id>` | trainer diagnostics under `debug/*` | inspect learning dynamics and RLlib internals |

The shared training metrics are:

- `train/episode_index`
- `train/env_step`
- `train/reward_mean`
- `train/reward_max`
- `train/reward_std`
- `train/resco_delay_mean`
- `train/resco_delay_max`
- `train/resco_delay_std`
- `train/resco_wait_mean`
- `train/resco_wait_max`
- `train/resco_wait_std`
- `train/resco_trip_time_mean`
- `train/resco_queue_mean`
- `train/resco_queue_max`
- `train/resco_tripinfo_count`
- `train/efficiency_total_arrived`
- `train/efficiency_total_departed`
- `train/safety_*`
- `debug/reward/<agent_id>`

The shared validation metrics are:

- `validation/env_step`
- `validation/pass_index`
- `validation/reward_mean`
- `validation/reward_max`
- `validation/reward_std`
- `validation/resco_delay_mean`
- `validation/resco_delay_max`
- `validation/resco_delay_std`
- `validation/resco_wait_mean`
- `validation/resco_wait_max`
- `validation/resco_wait_std`
- `validation/resco_trip_time_mean`
- `validation/resco_queue_mean`
- `validation/resco_queue_max`
- `validation/resco_tripinfo_count`
- `validation/efficiency_total_arrived`
- `validation/efficiency_total_departed`
- `validation/safety_total_teleported`
- `validation/safety_total_emergency_brake`
- `validation/safety_total_collisions`
- `validation/eval/episode`
- `validation/episode/sim_time_abs`
- `validation/episode/elapsed_seconds`
- `validation/warnings/*`

Validation-only W&B media can also be logged under:

- `validation/actions_share/<agent_id>`
- `validation/actions_timeline/<agent_id>`

These are history-backed per-agent validation plot images for the validation passes.
They are not scalar metrics, so they are intended for W&B panels rather than CSV analysis.

The debug-only training metrics are:

- `debug/efficiency_total_running`
- `debug/efficiency_total_backlogged`
- `debug/efficiency_total_stopped`
- `debug/efficiency_total_queued`
- `debug/efficiency_total_waiting_time`
- `debug/efficiency_mean_speed`
- `debug/efficiency_mean_average_speed`
- `debug/efficiency_mean_pressure`
- `debug/rllib/*`
- `debug/env_steps_sampled`
- `debug/agent_steps_sampled`
- `debug/episodes_total`
- `debug/episode_return_mean`
- `debug/episode_return_min`
- `debug/episode_return_max`
- `debug/episode_len_mean`
- `debug/ppo/learners/*`
- `debug/dqn/learners/*`
- `debug/dqn/replay/*`
- `debug/sac/learners/*`
- `debug/sac/replay/*`
- `debug/ppo/entropy_mean`
- `debug/sac/entropy_mean`
- `debug/td_error_mean`
- `debug/td_error_abs_mean`

Removed from the always-on RLlib training trace:

- `train/episode_summary_available`
- `train/episode_reward`
- trainer-return aliases under `train/*`

### Training reward traces

| Metric | Producer | Inputs | Logged when |
| --- | --- | --- | --- |
| `train/reward_mean` | algorithm runner | mean of per-agent episode reward totals from the completed episode | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `train/reward_max` | algorithm runner | max of per-agent episode reward totals from the completed episode | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `train/reward_std` | algorithm runner | std of per-agent episode reward totals from the completed episode | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `debug/reward/<agent_id>` | algorithm runner | completed-episode reward total for one signal | every `logging.train_log_freq_episodes` completed episodes in both trace modes |
| `train/resco_*` | algorithm runner | cached completed-episode benchmark summary | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `train/efficiency_total_arrived` / `train/efficiency_total_departed` | algorithm runner | cumulative throughput totals from the completed episode summary | every `logging.train_log_freq_episodes` completed episodes; default is every episode |
| `debug/efficiency_*` | algorithm runner | final live-system snapshot captured at the end of the completed episode | every `logging.train_log_freq_episodes` completed episodes in both trace modes |
| `debug/episode_return_*` | RLlib runners | trainer-return summary for the training iteration | debug trace mode only |
| `debug/ppo/entropy_mean` | PPO runner | trainer-level learner entropy diagnostic when available | debug trace mode only |
| `debug/sac/entropy_mean` | SAC runner | trainer-level learner entropy diagnostic when available | debug trace mode only |
| `final/eval/mean_reward` | algorithm runner or final evaluation pass | evaluation rollout output | final evaluation summary only |
| `final/eval/std_reward` | algorithm runner or final evaluation pass | evaluation rollout output | final evaluation summary only |

For SAC, RLlib now consumes the same multi-agent discrete action setup as PPO and DQN.
That means the logged SAC training rewards come from RLlib's multi-agent training result,
not from a project-side joint-action reward reduction wrapper.

### Validation traces

Periodic RLlib validation logs train-comparable metrics under `validation/*`.
For metrics that also exist in training, the formulas match the training trace:

- `validation/reward_mean|max|std` come from the completed eval episode summary's per-agent reward totals
- `validation/resco_*` come from the same completed eval episode summary fields used for `train/resco_*`
- `validation/efficiency_total_arrived|departed` and `validation/safety_*` come from the eval episode's cached `system_*` summary fields

When validation uses multiple eval seeds, the runner builds one per-seed summary row and then logs one validation point whose scalar values are the arithmetic mean across those seeds.
Redundant `validation/eval/*`, `validation/resco/*`, `validation/efficiency/*`, `validation/safety/*`, and `validation/tripinfo/*` aliases are intentionally omitted from the periodic validation surface.
The legacy `eval/*` alias namespace is also intentionally omitted; final evaluation metrics live under `final/*` and periodic comparisons live under `validation/*`.

#### Validation action-distribution plots

RLlib validation can also log two action-usage payloads per traffic-signal agent:

- `validation/actions_share/<agent_id>`
- `validation/actions_timeline/<agent_id>`

`validation/actions_share/<agent_id>` is a stacked area chart image built from the validation rollout's chosen discrete actions:

- x-axis: environment time over the validation rollout
- y-axis: sliding-window action proportion
- one colored band per discrete action / green phase
- the stacked values sum to `1.0` at each plotted step

The rolling window represents one minute of environment time:

- `logging.log_validation_action_plots`
- `logging.validation_action_plot_max_agents`

The runner derives the share window from the decision interval:

- `window_steps = round(60 / decision_interval_seconds)`

So with `delta_time=5`, the plotted share window spans `12` decisions.

`validation/actions_timeline/<agent_id>` is a whole-episode phase timeline image:

- x-axis: environment time over the validation rollout
- y-axis: discrete phase index
- each colored block shows which phase was active between adjacent decision steps
- exactly one phase is active for each interval

For multi-seed validation, the runner:

1. builds one action trace per seed
2. converts each trace into sliding-window proportions for the share plot
3. averages the per-step proportions across seeds for the share plot
4. uses a per-step majority vote across seeds for the timeline plot

The public step key for browsing validation plot versions is:

- `validation/pass_index`

This is a dense monotonic counter for validation passes.
It is separate from `validation/env_step`, which stays as the comparable training-progress axis.

#### How to view the W&B slider under the action plots

The runner logs one image per validation pass under the same media key for each plot type.
W&B keeps the history of those images, so you can browse the validation passes with the slider under the media panel.

Recommended setup:

1. Open the run in W&B.
2. Add a panel.
3. Select one logged media key such as:
   - `validation/actions_share/<agent_id>`
   - `validation/actions_timeline/<agent_id>`
4. Open that panel as an image/media panel.
5. Use the slider under the panel to move across validation passes.

In both cases:

- the slider moves between validation passes
- the chart x-axis inside each image still shows within-rollout environment time for the selected pass
- one panel corresponds to one traffic-light agent

### Best validation checkpoints

RLlib runs can also retain the top validation checkpoints under:

- `outputs/<run>/checkpoints/<algorithm_kind>/best_validation/`

The retention settings are controlled by:

- `logging.save_best_validation_checkpoints`
- `logging.best_validation_checkpoint_count`
- `logging.best_validation_metric`

The current runner ranks retained checkpoints by lower `validation/resco_delay_mean`.
Each retained entry is listed in `best_validation/metadata.json` together with:

- checkpoint path
- validation pass index
- `validation/env_step`
- metric value used for ranking
- the aggregated validation summary for that checkpoint, without any `final/*` aliases
- the per-seed eval rows used to compute the aggregate, also without `final/*` aliases

The runner keeps only the best `N` validation checkpoints on disk and rewrites the metadata file after every retained update.

### Loading saved weights correctly

To reload one of the retained best-validation checkpoints:

1. Read `best_validation/metadata.json` and pick the retained checkpoint entry you want.
2. Rebuild the same RLlib algorithm kind with a compatible config for the same scenario and reward setup.
3. Restore the checkpoint with the RLlib checkpoint API, not by manually loading partial module weights.
4. Evaluate using the same eval seed schedule if you want comparable `validation/resco_delay_mean` values.

The intended restore pattern is:

```python
algo = config.build()
algo.restore_from_path(checkpoint_path)
```

If the algorithm only exposes `restore(...)`, use that fallback instead.
For reproducible comparison, keep the same:

- algorithm kind
- scenario/env wiring
- reward definition
- eval seed schedule

Restored evaluation results may not be bit-identical across machines or library versions, but they should stay close to the stored validation result within a documented tolerance.

## Shared Episode Summary Metrics

The final episode summary is built from cached data from the last completed episode.
This matters because evaluation envs auto-reset after a done episode.
The runner now uses cached episode data instead of reading the fresh post-reset state.

### Benchmark metrics

These are the main thesis comparison metrics.

| Metric | Formula | Inputs | Logged when |
| --- | --- | --- | --- |
| `resco_avg_delay` | `mean(timeLoss + departDelay)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_delay_mean` | same value as `resco_avg_delay` | SUMO tripinfo XML | episode summary and training trace |
| `resco_delay_max` | `max(timeLoss + departDelay)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary and training trace |
| `resco_delay_std` | std of `timeLoss + departDelay` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary and training trace |
| `resco_trip_time` | `mean(duration)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_trip_time_mean` | same value as `resco_trip_time` | SUMO tripinfo XML | episode summary and training trace |
| `resco_wait` | `mean(waitingTime)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary |
| `resco_wait_mean` | same value as `resco_wait` | SUMO tripinfo XML | episode summary and training trace |
| `resco_wait_max` | `max(waitingTime)` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary and training trace |
| `resco_wait_std` | std of `waitingTime` over completed non-ghost vehicles | SUMO tripinfo XML | episode summary and training trace |
| `resco_tripinfo_count` | count of completed non-ghost tripinfo rows | SUMO tripinfo XML | episode summary |
| `resco_queue` | mean of recorded queue-per-signal values across the episode | `system_mean_queued`, else `system_total_queued / num_signals`, else `system_total_stopped / num_signals` | episode summary |
| `resco_max_queue` | max recorded queue-per-signal value across the episode | `system_max_queue` from live step info | episode summary |
| `resco_queue_mean` | same value as `resco_queue` | live queue metrics | episode summary and training trace |
| `resco_queue_max` | same value as `resco_max_queue` | live queue metrics | episode summary and training trace |

Interpretation:

- lower is better for delay, trip time, wait, and queue
- `resco_queue` is an episode average
- `resco_max_queue` is a worst-case episode peak

For the training trace, the runner remaps the completed-episode summary into:

- `train/resco_delay_mean|max|std`
- `train/resco_wait_mean|max|std`
- `train/resco_trip_time_mean`
- `train/resco_queue_mean|max`
- `train/resco_tripinfo_count`

These `train/*` fields are not recomputed from a separate source.
They are copied from the cached completed-episode summary that the environment builds at episode end.

Only the episode-facing throughput totals stay in `train/*`.
The end-of-episode live-state efficiency snapshot fields move to `debug/*`
because they describe the network at the horizon boundary rather than the
whole-episode outcome.

### Entropy diagnostics

Entropy is logged only for PPO and SAC, and only in `debug` mode.

The runner does not compute entropy itself from action probabilities.
Instead, it searches the RLlib learner metrics for numeric fields whose key path contains `entropy`,
while skipping control/configuration fields such as `target_entropy`, `entropy_coeff`, and `curr_kl_coeff`.

When multiple entropy-like fields are present, the code prefers them in this order:

1. keys ending with `entropy_mean`
2. keys ending with `curr_entropy`
3. any other entropy-like numeric field

The selected value is then logged as:

- `debug/ppo/entropy_mean`
- `debug/sac/entropy_mean`

This keeps entropy as a debug diagnostic instead of treating it as a benchmark metric.

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
- one forced last validation pass at the end of training

For multi-seed evaluation, the runner builds one cached summary per eval seed and then averages the validation-facing scalar metrics into one validation row.
Warnings are kept if any eval seed shows the problem.
The last forced validation pass is the canonical end-of-run RLlib result; the RLlib path no longer publishes a separate final benchmark namespace.
If validation action plots are enabled, that forced last validation pass also logs the final `validation/actions_share/<agent_id>` and `validation/actions_timeline/<agent_id>` payloads and advances `validation/pass_index` once more.

### SAC through RLlib

SAC now uses the same PettingZoo parallel env and RLlib multi-agent policy setup
as PPO and DQN:

- PettingZoo parallel env
- SuperSuit padding when shared-policy mode needs aligned spaces
- RLlib multi-agent discrete policies
- one forced last validation pass at the end of training

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
