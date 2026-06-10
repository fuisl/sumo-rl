# Scenario-First Presets

This folder is the canonical thesis experiment layout for the fixed-time and static
baseline recipes.

The rule of thumb is:

- `configs/base.yaml` holds shared defaults.
- `configs/scenario/` holds the road-network and environment definitions.
- `configs/algorithm/` holds method hyperparameters.
- `configs/presets/<scenario>/` holds the runnable experiment recipes.
- `experiments/*.py` are thin launchers that point at one preset by default.
- `configs/rllib.yaml` is the shared RLlib launcher config for PPO, DQN, and SAC.
- `experiments/rllib.py` is the shared RLlib launcher.

The target RESCO scenarios are:

- `resco_cologne1`
- `resco_cologne3`
- `resco_cologne8`
- `resco_ingolstadt1`
- `resco_ingolstadt7`
- `resco_ingolstadt21`

Each target scenario folder is meant to contain the same method names, so the layout is easy to scan:

```text
configs/presets/<scenario>/
  fixed_time.yaml
  static_max_pressure.yaml
```

FGS RLlib presets are available for the grid and Cologne8 benchmarks:

```text
configs/presets/resco_grid4x4/
  fgs_frap_gat_sac.yaml
  fgs_mlp_gat_sac.yaml
configs/presets/resco_cologne8/
  sac_builtin.yaml
  fgs_frap_gat_sac.yaml
  fgs_mlp_gat_sac.yaml
  fgs_frap_gatv2_sac.yaml
  fgs_mlp_gatv2_sac.yaml
```

The static baseline presets now follow the RLlib validation seed layout:

- `experiment.eval_seeds` is used when present
- the thesis presets pin `eval_seeds: [1, 2, 3, 4, 5]`
- the baseline logs only the averaged `validation/*` result, replayed every 5 episodes through episode 500 by default

RLlib methods are named in `configs/algorithm/` instead:

```text
configs/algorithm/
  ppo.yaml
  dqn.yaml
  dqn_dcrnn.yaml
  fgs.yaml
  sac_builtin.yaml
  sac_mlp.yaml
  sac_dcrnn_actor.yaml
  sac_dcrnn_full.yaml
```

The older `dcrnn.yaml` and `sac_custom.yaml` files are kept as compatibility
aliases, but the canonical public names are `dqn_dcrnn` and `sac_mlp`.

How to read one preset:

1. Open the launcher in `experiments/`.
2. Open the matching file in `configs/presets/<scenario>/`.
3. Follow the `defaults` chain into `configs/scenario/` and `configs/algorithm/`.

For RLlib runs, open `configs/rllib.yaml` together with the algorithm file you want.
Launch RLlib presets from the repository config root by passing the preset path
as the Hydra config name, for example:

```bash
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_mlp_gat_sac
python experiments/rllib.py --config-name presets/resco_cologne8/sac_builtin
```

RLlib training length is controlled by `experiment.episodes`. The episode horizon
is configured in seconds with `experiment.episode_seconds`, and the decision-step
horizon is derived from the environment `delta_time` when needed. For example,
`3600` episode seconds with `delta_time=5` gives about `3600 / 5 = 720` decision
steps. Training logs use sampled env steps (`logging.train_log_freq_steps`), while
RLlib validation cadence is controlled by `experiment.validation_interval_episodes`
by default. The step-based `logging.eval_freq` remains a fallback when the episode
interval is not set. The shared RLlib config also caps local CPU use with
`resources.ray_num_cpus=2` and `resources.native_num_threads=1`, but RLlib
presets default to `resources.ray_address=auto`; start one shared Ray head with
the desired CPU/GPU capacity before launching jobs. To pin the shared head to
physical GPU 1, start it with
`CUDA_VISIBLE_DEVICES=1 ray start --head --num-cpus=8 --num-gpus=1`. Override
`resources.ray_address=null` for standalone local debugging. The shared default
also uses one remote EnvRunner and one remote Learner per experiment so Ray can
account for CPU/GPU reservations; the default
`algorithm.params.num_gpus_per_learner=0.25` lets several learner actors share
one selected GPU, while `1` reserves it exclusively. Set
`resources.cuda_visible_devices` in `configs/rllib.yaml`
or on the command line to choose the physical GPU; the selected GPU is exposed
inside the run as local CUDA index 0, so `algorithm.params.local_gpu_idx` should
usually stay `0`.

The launcher name tells you the method family.
The folder name tells you the scenario.
The file name tells you the exact recipe.
