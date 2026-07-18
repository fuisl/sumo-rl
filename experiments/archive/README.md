# Experiment Archive

This directory holds research/reference experiment assets that are kept for
thesis reproducibility but are not part of the supported top-level experiment
launcher surface.

Top-level `experiments/` is reserved for supported launchers and utilities such
as `fixed_time.py`, `static_max_pressure.py`, `rllib.py`,
`validate_methods.py`, `record_rollout.py`, and related thin wrappers.

Archive assets may:

- depend on analysis-specific workflows
- write local artifacts under `experiments/artifacts/`
- require notebook-driven or one-off manual inspection

Treat files here as reference material unless a canonical doc explicitly calls
out a supported workflow.

## Archive decisions

- `dcrnn_resource_smoke.py`: retain as a durable reference utility because `docs/thesis/resource_usage_smoke.md` points to it as the canonical smoke workflow for DCRNN resource checks.
- `spatiotemporal_dependency.py`: retain as a durable reference helper because it contains the reusable analysis logic behind the Cologne8 spatiotemporal dependency workflow.
- `cologne8_spatiotemporal_dependency.ipynb`: retain as a durable thesis reference because it turns the dependency helper into presentation-ready evidence for Cologne8 temporal and spatial propagation claims.
- `manual_checkpoint_evaluation.ipynb`: retain as a reference notebook because it is a thin front-end over `experiments/validate_methods.py` for manual inspection of validation outputs without changing the supported CLI surface.
- `visualize_best_checkpoint_trip.ipynb`: retain as a reference notebook because it supports qualitative replay and GIF export for best-checkpoint inspection when thesis results need a visual explanation.
- `visualize_dcrnn_graph.ipynb`: retain as a durable reference notebook because it documents and visualizes the exact traffic-signal graph topology used by the DCRNN observation path.
- `compare_wandb_runs.ipynb`: treat as a disposable archive artifact because it is a generic local plotting notebook for already-downloaded W&B exports and does not protect a unique thesis workflow.

If we want to shrink the archive further in a later cleanup pass, `compare_wandb_runs.ipynb` is the first deletion candidate. The retained files stay because they either back a canonical thesis doc or preserve unique analysis context that is harder to reconstruct from the supported launchers alone.
