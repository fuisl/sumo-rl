# PLANS.md

## Roadmap
This project will be organized in phases so the thesis work stays incremental and easy to review.

### Phase 1: Hydra + W&B Integration
- Status: implemented in code; runtime smoke tests still depend on SUMO and the optional experiment extras.
- Add a shared experiment configuration layer with Hydra.
- Add W&B logging for resolved configs, seeds, metrics, and run metadata.
- Refactor the current example scripts to call a shared training core instead of duplicating setup.
- Keep the old examples runnable during the transition.

### Phase 2: Manual Traffic Control
- Status: implemented in code and docs.
- Add or document a fixed-time traffic control mode.
- Provide a clear guide for running manual traffic control on the existing single-intersection and grid examples.
- Make sure the fixed-time path works without RL dependencies.

### Phase 3: Third-Party RL Libraries
Status: implemented in code and docs for the current working examples; future extensions remain open.
Implement external RL integrations in small steps, starting with the simplest and quickest ones.

#### Phase 3.1: Fastest Integrations First
- Implemented for the thesis examples.
- Stable-Baselines3 DQN on the single-intersection environment.
- Stable-Baselines3 PPO on a single-agent or vectorized setup if needed.

#### Phase 3.2: Multi-Agent Library Integration
- Implemented for the RESCO 4x4 examples.
- Stable-Baselines3 PPO on the multi-agent RESCO 4x4 grid setup.
- Keep the PettingZoo wrapper path documented and reusable.

#### Phase 3.3: Future Extensions
- Add any additional imported third-party methods only after the core integrations are stable.
- If a method such as DAC is not already present in this repo, treat it as a separate addition with its own adapter and dependency check.

## Assumptions
- `SUMO_HOME` remains required for all simulation runs.
- W&B should support disabled or offline mode for local development.
- Hydra configs should drive the experiments, not replace the existing environment API.
