# Scenario-First Presets

This folder is the canonical thesis experiment layout.

The rule of thumb is:

- `configs/base.yaml` holds shared defaults.
- `configs/scenario/` holds the road-network and environment definitions.
- `configs/algorithm/` holds method hyperparameters.
- `configs/presets/<scenario>/` holds the runnable experiment recipes.
- `experiments/*.py` are thin launchers that point at one preset by default.

The target RESCO scenarios are:

- `resco_cologne1`
- `resco_cologne3`
- `resco_ingolstadt1`
- `resco_ingolstadt7`

Each target scenario folder is meant to contain the same method names, so the layout is easy to scan:

```text
configs/presets/<scenario>/
  dqn.yaml
  baselinev1_rllib.yaml
  ppo.yaml
  ql.yaml
  sac.yaml
  libsignal_idqn.yaml
  libsignal_ippo.yaml
  libsignal_mplight.yaml
  static_greedy.yaml
  static_max_pressure.yaml
```

How to read one preset:

1. Open the launcher in `experiments/`.
2. Open the matching file in `configs/presets/<scenario>/`.
3. Follow the `defaults` chain into `configs/scenario/` and `configs/algorithm/`.

The launcher name tells you the method family.
The folder name tells you the scenario.
The file name tells you the exact recipe.
