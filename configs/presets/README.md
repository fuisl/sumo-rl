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

Support terminology in this file matches the thesis support matrix in
`docs/thesis/experiments.md`:

- `supported`: part of the supported thesis baseline
- `experimental`: available for research or ablation work, but not part of the supported thesis baseline
- `alias`: compatibility-only name for a canonical method
- `preset-backed`: supported through a scenario-first preset file under `configs/presets/<scenario>/`
- `shared-launcher`: supported through `experiments/rllib.py` plus `configs/algorithm/`, even when no scenario-first preset file exists

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
  fgs_frap_gatv2_ppo.yaml
  fgs_mlp_gat_ppo.yaml
  fgs_mlp_gatv2_ppo.yaml
configs/presets/resco_ingolstadt21/
  fgs_frap_gatv2_ppo.yaml
  fgs_mlp_gat_ppo.yaml
  fgs_mlp_gatv2_ppo.yaml
  fgs_mlp_gatv2_sac.yaml
```

These FGS and FGS PPO recipes are `supported` only where they are preset-backed
today. Their additional ablation shapes still count as benchmark support only
through the scenario folders listed above, not as a promise that every
algorithm variant has full scenario parity.

The full FGS v1 startup and training pipeline is documented in
`docs/thesis/fgs_v1_pipeline.md`.

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
  fgs_ppo.yaml
  sac_builtin.yaml
  sac_mlp.yaml
```

For Phase 2 support status, read that list as follows:

- `supported` via the shared launcher: `ppo`, `dqn`, `dqn_dcrnn`, `frap`, `colight`, `fgs`, `fgs_ppo`, `sac_builtin`, `sac_mlp`
- `experimental` via the shared launcher: `ppo_dcrnn_mlp`, `ppo_dcrnn_shared_mlp`, `dqn_dcrnn_mlp`, `fgsv2`
- `alias`: `dcrnn` for `dqn_dcrnn`, and `sac_custom` for `sac_mlp`

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

If a method is listed as `supported` in the thesis matrix but does not have a
scenario-first preset file here, interpret that support as `shared-launcher`
support rather than `preset-backed` benchmark support.

RLlib training length is controlled by `experiment.episodes`. The episode horizon
is configured in seconds with `experiment.episode_seconds`, and the decision-step
horizon is derived from the environment `delta_time` when needed. For example,
`3600` episode seconds with `delta_time=5` gives about `3600 / 5 = 720` decision
steps. Training logs use sampled env steps (`logging.train_log_freq_steps`), while
RLlib validation cadence is controlled by `experiment.validation_interval_episodes`
by default. The step-based `logging.eval_freq` remains a fallback when the episode
interval is not set. The shared RLlib config starts a local Ray instance with
conservative resource defaults. For SLURM-specific overrides, use
`docs/thesis/remote_server.md`.

SAC presets inherit `algorithm.params.training_intensity=1.0` and
`algorithm.params.train_batch_size_per_learner=64` from `sac_builtin`. This
keeps RLlib's off-policy replay/learner work bounded on Cologne-sized
multi-agent runs; raise these values only when you intentionally want more
gradient work per collected sample.

The launcher name tells you the method family.
The folder name tells you the scenario.
The file name tells you the exact recipe.
