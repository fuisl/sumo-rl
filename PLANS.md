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
- Compare the handwritten max-pressure baseline against RESCO's benchmark definitions.
- Keep the RESCO comparison fields documented and easy to verify against the raw tripinfo XML.
- Keep W&B and CSV schemas narrow for benchmark runs, while preserving enough system metrics for debugging.
- Add a short benchmark comparison note or table that states which fields are canonical for thesis reporting and how they map to the raw RESCO tripinfo values.
- Verify the max-pressure presets still produce the intended five-seed summary behavior on the RESCO scenarios.

### Phase 5: Runner-Native Graph Models
- Status: initial DCRNN path implemented; runtime smoke tests still depend on SUMO, Ray/RLlib, and PyTorch.
- Add reusable traffic-signal graph topology and density/queue graph observation utilities.
- Add a graph-observation wrapper for RLlib runners without changing existing PPO, DQN, FRAP, SAC, or static paths.
- Add the first graph algorithm as `algorithm=dqn_dcrnn`, using a DCRNN Q-network through the current runner, W&B, CSV, evaluation, and checkpoint flow.
- Keep v1 restricted to independent policies; shared graph data passing into current models is a future extension.

### Phase 5: FGS FRAP-GNN-SAC
- Status: implemented as a project-owned, third-party-inspired RLlib integration.
- FGS combines FRAP-style local phase competition, CoLight-style GAT communication, and discrete SAC.
- It is not a handwritten baseline; it is a modular learning method whose components can be ablated through Hydra config.
- The v1 critic defaults to a joint-action CTDE form over graph embeddings and same-transition joint action context, while execution remains decentralized through a shared policy.
- The default reward remains SUMO-RL's existing `diff-waiting-time`.
- Cologne8 FGS presets include PyG `GATv2Conv` communication ablations for both FRAP and MLP local encoders.
- FGSv3 is implemented as `algorithm=fgsv3`: it keeps FRAP action tokens local,
  communicates phase-demand plus own previous phase through weighted GATv2, and
  uses a neighborhood-factored SAC critic for Cologne8 and Ingolstadt21 runs.

## Assumptions
- `SUMO_HOME` remains required for all simulation runs.
- W&B should support disabled or offline mode for local development.
- Hydra configs should drive the experiments, not replace the existing environment API.
- For fixed-time and static baselines, per-seed logging stays in both CSV and W&B, and the final summary should average the seed runs.
