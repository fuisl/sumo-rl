# Experiments Directory

Top-level `experiments/` is the supported launcher and utility surface for the
thesis workflow.

Supported top-level entrypoints include:

- `fixed_time.py`
- `static_max_pressure.py`
- `rllib.py`
- `validate_methods.py`
- `record_rollout.py`
- `record_rllib_rollout.py`
- `restore_rllib_checkpoint.py`
- `download_wandb_runs.py`

Research notebooks and archive/reference utilities live under
`experiments/archive/`. Local analysis outputs for those archive workflows may
be written under the ignored `experiments/artifacts/` tree.
