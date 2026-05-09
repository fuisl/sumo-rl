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
- Added fixed-time and static presets for the RESCO `cologne1` and `ingolstadt1` networks.
- RESCO-aligned episode summaries are now logged from tripinfo XML and live system metrics.
- Fixed-time and static runs log one row per seed plus a final 5-seed summary.
- The current logging schema keeps benchmark metrics centered on RESCO-style delay, trip time, waiting time, and queue.

### Phase 3: Third-Party SB3 Integrations
Status: implemented in code and docs for the current working examples.
Implement external Stable-Baselines3 integrations in small steps, starting with the simplest and quickest ones.

#### Phase 3.1: Fastest Integrations First
- Implemented for the thesis examples.
- Stable-Baselines3 DQN on the single-intersection environment.
- Stable-Baselines3 PPO on a single-agent or vectorized setup if needed.

#### Phase 3.2: Multi-Agent SB3 Integration
- Implemented for the RESCO 4x4 examples.
- Stable-Baselines3 PPO on the multi-agent RESCO 4x4 grid setup.
- Keep the PettingZoo wrapper path documented and reusable.

#### Phase 3.3: Future Extensions
- Add any additional imported third-party methods only after the core integrations are stable.
- If a method such as DAC is not already present in this repo, treat it as a separate addition with its own adapter and dependency check.

### Phase 4: Benchmark Comparison Cleanup
- Status: in progress; the logging path is in place, but the benchmark audit and comparison write-up still need to be finished.
- Compare the handwritten max-pressure and greedy baselines against RESCO's benchmark definitions.
- Keep the RESCO comparison fields documented and easy to verify against the raw tripinfo XML.
- Keep W&B and CSV schemas narrow for benchmark runs, while preserving enough system metrics for debugging.
- Add a short benchmark comparison note or table that states which fields are canonical for thesis reporting and how they map to the raw RESCO tripinfo values.
- Verify the max-pressure and greedy presets still produce the intended five-seed summary behavior on the RESCO scenarios.

### Phase 5: Third-Party MARL Replication
Status: implemented in code and docs for the LibSignal bridge.
Implement the non-SB3 third-party replication track for `IDQN`, `MPLight`, and `IPPO`.

#### Phase 5.1: LibSignal Primary Path
- Implemented with dedicated launchers and a shared LibSignal bridge.
- Use LibSignal as the main backend for `IDQN`, `MPLight`, and `IPPO` where it is sufficient.
- Keep the commands and expected outputs documented so the runs are easy to reproduce.
- The Phase 5 logging follows the static-run pattern: per-seed trace rows in both CSV and W&B, plus a final seed-average summary.
- Keep both the RESCO proxy metrics and the extra upstream metrics in the logged traces so the runs stay comparable and auditable.

#### Phase 5.2: Reference and Fallback Paths
- Keep RESCO as the reference path for `IDQN`.
- Keep EPyMARL as the fallback path for `IPPO` if LibSignal needs too much adaptation.
- Document clearly which backend is considered exact reproduction and which is closest-practical replication.
- Use the LibSignal DTL log as the source of truth for the upstream metrics and label the RESCO values as thesis-side proxies.

## Assumptions
- `SUMO_HOME` remains required for all simulation runs.
- W&B should support disabled or offline mode for local development.
- Hydra configs should drive the experiments, not replace the existing environment API.
- For fixed-time and static baselines, per-seed logging stays in both CSV and W&B, and the final summary should average the seed runs.
- The MARL replication track is a separate phase from benchmark cleanup so the thesis scope stays easy to review.
