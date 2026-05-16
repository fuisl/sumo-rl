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
- `resco_ingolstadt1`
- `resco_ingolstadt7`

Each target scenario folder is meant to contain the same method names, so the layout is easy to scan:

```text
configs/presets/<scenario>/
  fixed_time.yaml
  static_max_pressure.yaml
```

RLlib methods are named in `configs/algorithm/` instead:

```text
configs/algorithm/
  ppo.yaml
  dqn.yaml
  sac_builtin.yaml
  sac_custom.yaml
```

How to read one preset:

1. Open the launcher in `experiments/`.
2. Open the matching file in `configs/presets/<scenario>/`.
3. Follow the `defaults` chain into `configs/scenario/` and `configs/algorithm/`.

For RLlib runs, open `configs/rllib.yaml` together with the algorithm file you want.
RLlib training length is controlled by `experiment.episodes`. The episode horizon
is configured in seconds with `experiment.episode_seconds`, and the decision-step
horizon is derived from the environment `delta_time` when needed. For example,
`3600` episode seconds with `delta_time=5` gives about `3600 / 5 = 720` decision
steps. Training logs use sampled env steps (`logging.train_log_freq_steps`), while
validation logs use evaluation episodes (`logging.validation_log_freq_episodes`).

The launcher name tells you the method family.
The folder name tells you the scenario.
The file name tells you the exact recipe.
