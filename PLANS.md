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

### Phase 4: Benchmark Comparison Cleanup
- Status: in progress; the logging path is in place, but the benchmark audit and comparison write-up still need to be finished.
- Compare the handwritten max-pressure and greedy baselines against RESCO's benchmark definitions.
- Keep the RESCO comparison fields documented and easy to verify against the raw tripinfo XML.
- Keep W&B and CSV schemas narrow for benchmark runs, while preserving enough system metrics for debugging.
- Add a short benchmark comparison note or table that states which fields are canonical for thesis reporting and how they map to the raw RESCO tripinfo values.
- Verify the max-pressure and greedy presets still produce the intended five-seed summary behavior on the RESCO scenarios.

## Assumptions
- `SUMO_HOME` remains required for all simulation runs.
- W&B should support disabled or offline mode for local development.
- Hydra configs should drive the experiments, not replace the existing environment API.
- For fixed-time and static baselines, per-seed logging stays in both CSV and W&B, and the final summary should average the seed runs.
