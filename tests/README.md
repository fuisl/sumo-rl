# Tests

This directory is organized by test ownership rather than by historical file growth.

## Layout

| Directory | Usual marker | What belongs here |
| --- | --- | --- |
| `tests/core/` | none or `core_fast` | deterministic pure-Python checks for config defaults, metric formulas/parsing, reward logic, connection-mode behavior, and utility code |
| `tests/runner/` | `research_heavy` | RLlib runner orchestration, policy mapping, evaluation env selection, training/evaluation cadence, checkpointing, resume behavior, and W&B/CSV wiring |
| `tests/models/` | `research_heavy` | model and algorithm contract tests for DCRNN, PPO-DCRNN, SAC, CoLight, FRAP, FGS, and FGSV2 |
| `tests/integration_local/` | `local_heavy` | SUMO-backed smoke tests and environment integration checks that depend on local runtime setup |
| `tests/_support/` | not collected | shared dummy envs and helper objects that are reused by collected tests |

## Where To Add Tests

- Add config, metric, reward, env-wiring, and utility checks to `tests/core/`.
- Add runner scheduling, checkpointing, evaluation, logging, and policy-mapping checks to `tests/runner/`.
- Add module-shape, forward-pass, and algorithm-contract checks to `tests/models/`.
- Add real SUMO-backed smoke coverage to `tests/integration_local/`.

## Transition Map

| Old file | New home |
| --- | --- |
| `tests/test_rllib_runner.py` | `tests/runner/` split by runner concern |
| `tests/test_sac_discrete.py` | `tests/models/` split by SAC context, config, forward behavior, build config, and runner contract |
| `tests/test_experiment_metrics.py` | `tests/core/test_experiment_metrics.py` |
| `tests/test_validation_metrics.py` | `tests/core/test_validation_metrics.py` |
| `tests/gym_test.py`, `tests/pz_test.py` | `tests/integration_local/` |
