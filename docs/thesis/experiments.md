---
title: Thesis Experiments
firstpage:
---

# Thesis Experiments

This page contains the thesis-specific experiment workflow built on top of the upstream SUMO-RL examples.
If you are onboarding to the codebase, read [docs/thesis/engineering_guide.md](engineering_guide.md) first.

If you are looking for fixed-time/manual traffic control, read [docs/thesis/manual_control.md](manual_control.md) after this page.
If you want the RESCO static baselines, read [docs/thesis/static_baselines.md](static_baselines.md) next.

The thesis launchers now expose the fixed-time and max-pressure RESCO presets plus a shared RLlib launcher for PPO, DQN, and SAC.

## Hydra
Hydra is used as the experiment composition layer.

- Each runnable example has a Hydra config in `configs/`
- Configs define the environment, algorithm, and logging settings
- Command-line overrides let you change seeds, paths, and hyperparameters without editing code
- Each run gets its own output directory under `outputs/<experiment-name>/<timestamp>/`
- A local metrics CSV is written to `outputs/<experiment-name>/<timestamp>/logs/metrics.csv` for quick debugging
- Episode horizon is configured in seconds with `experiment.episode_seconds`. If you need the decision-step horizon, divide by `delta_time`; for example, `3600` seconds with `delta_time=5` is about `720` steps.
- RLlib validation is episode-based by default with `experiment.validation_interval_episodes=5`; `logging.eval_freq` is only the step-based fallback when the episode interval is unset.
- The runner now logs episode-end RESCO summaries plus namespaced efficiency and safety metrics, using:
  - `resco_avg_delay` from SUMO tripinfo `timeLoss`
  - `resco_trip_time` from SUMO tripinfo `duration`
  - `resco_wait` from SUMO tripinfo `waitingTime`
  - `resco_queue` and `resco_max_queue` from the live queue metrics
  - `efficiency_*` for queue, speed, waiting-time, and throughput aggregates
  - `safety_*` for emergency-brake and teleport/unsafe-event counts
  - tripinfo XML is generated to compute metrics and deleted by default; set `logging.save_tripinfo_output=true` to keep the raw XML files under `outputs/<experiment-name>/<timestamp>/tripinfo/`
- The config layout is split into:
  - `configs/scenario/` for network and road-network setup
  - `configs/algorithm/` for the method hyperparameters
  - `configs/rllib.yaml` for the shared RLlib launcher
  - scenario-first presets such as `configs/presets/resco_grid4x4/fixed_time.yaml` and `configs/presets/resco_cologne1/static_max_pressure.yaml`
  - the canonical layout guide in [`configs/presets/README.md`](../../configs/presets/README.md), which now also explains the RLlib algorithm files

Example:
```bash
python experiments/fixed_time.py scenario=resco_grid4x4
```

Other common entrypoints:

```bash
python experiments/static_max_pressure.py scenario=resco_cologne1
python experiments/rllib.py algorithm=ppo scenario=resco_grid4x4
python experiments/rllib.py algorithm=dqn scenario=resco_cologne1
python experiments/rllib.py algorithm=sac_builtin scenario=resco_ingolstadt1
python experiments/rllib.py algorithm=sac_custom scenario=resco_ingolstadt7
```

PPO and DQN default to independent policies. To switch to a shared policy, override
`algorithm.params.policy_mode=shared` on the command line.

SAC now uses RLlib's native discrete-action support. The repo hands each traffic
signal its own discrete action space through the multi-agent RLlib wrapper, and
the built-in/custom SAC paths train directly on those discrete policies.

That means there is no project-side joint Box action adapter in the current SAC
path. If SAC fails, the issue is in the RLlib discrete SAC path or the env/policy
setup, not in a custom continuous-action wrapper.

## Weights & Biases
Weights & Biases is used for experiment tracking.

- W&B logs configs, metrics, and run metadata
- The repo defaults to offline mode so local runs do not require an API key
- To use online logging, authenticate outside the repo with `wandb login` or set `WANDB_API_KEY` in your environment

Example:
```bash
python experiments/static_max_pressure.py scenario=resco_ingolstadt7 logging.mode=online logging.project=my-thesis
```

## Optional Install
To use the Hydra and W&B experiment layer, install the optional extras:
```bash
pip install -e ".[experiments]"
pip install -e ".[rllib]"
pip install -e ".[rllib-custom]"
```

## Notes
- These additions do not replace the upstream SUMO-RL API.
- The existing environment and algorithm examples still run through the same underlying SUMO-RL code paths.
- The RESCO summary log is the canonical run artifact for comparing against the benchmark formulas.
- Run names now put the scenario first, for example `resco_grid4x4__fixed_time` or `resco_cologne1__static_max_pressure`.
- Short smoke runs should watch the `train/` and `eval/` traces in addition to the episode-end summary rows.
