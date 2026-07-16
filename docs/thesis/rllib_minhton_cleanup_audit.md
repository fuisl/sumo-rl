# RLlib-Minhton Cleanup Audit

This note records the first cleanup pass after merging `rllib-minhton` into `main`.
The goal is to break follow-up work into reviewable slices instead of mixing feature code,
generated artifacts, documentation experiments, and research variants in one branch.

## Current State

- Local `main` now contains `origin/main` plus the merged `rllib-minhton` history.
- The engineering workflow changes were moved to the separate branch `chore-engineering-quality-gates`.
- This branch, `cleanup-rllib-minhton-audit`, is reserved for cleanup planning and follow-up cleanup commits.

## Highest-Priority Cleanup Targets

### 1. Tracked experiment artifacts and notebooks

These files are large, review-heavy, and not part of the core runtime:

- `experiments/visualize_dcrnn_graph.ipynb`
- `experiments/compare_wandb_runs.ipynb`
- `experiments/manual_checkpoint_evaluation.ipynb`
- `experiments/visualize_best_checkpoint_trip.ipynb`
- `experiments/dcrnn_resource_smoke.py`
- `experiments/validate_methods.py`
- `docs/thesis/resource_usage_smoke.md`

Recommended action:

- Decide which of these are thesis reference material versus temporary analysis artifacts.
- Move durable workflow docs into normal markdown documentation.
- Remove or archive notebooks and one-off smoke scripts that are not required to run experiments.

### 2. RLlib algorithm-surface explosion

The merged tree now exposes many closely related algorithm variants:

- `dqn`, `dqn_dcrnn`, `dqn_dcrnn_mlp`
- `ppo`, `ppo_dcrnn_mlp`, `ppo_dcrnn_shared_mlp`
- `fgs`, `fgs_ppo`, `fgsv2`
- `sac_builtin`, `sac_mlp`, `sac_dcrnn_actor`, `sac_dcrnn_actor_mlp`, `sac_dcrnn_full`, `sac_dcrnn_full_mlp`, `sac_dcrnn_shared_mlp`
- `colight`, `frap`

Recommended action:

- Define a supported thesis baseline set and an experimental set.
- Keep docs, presets, and CI aligned only with the supported set.
- Demote or quarantine variants that are only ablation branches unless they are actively used in results.

### 3. Redundant documentation paths

Thesis behavior is currently spread across:

- `README.md`
- `PLANS.md`
- `docs/thesis/experiments.md`
- `docs/thesis/engineering_guide.md`
- `docs/thesis/fgs_v1_pipeline.md`
- `docs/thesis/architecture_diagrams.md`
- `docs/thesis/pseudocode.md`

Recommended action:

- Keep the README focused on install and supported entrypoints.
- Keep one thesis experiment guide as the canonical runtime doc.
- Keep architecture-specific notes only where they provide unique value.
- Remove repetition between README and thesis docs wherever the same commands or explanations appear twice.

### 4. Runtime and evaluation complexity

The largest ongoing maintenance surface appears in:

- `sumo_rl/experiments/rllib_runner.py`
- `sumo_rl/agents/sac/sac.py`
- `sumo_rl/agents/ppo/ppo.py`
- `sumo_rl/agents/dcrnn/dcrnn.py`
- `sumo_rl/agents/fgs/fgs.py`

Recommended action:

- Separate supported runtime behavior from experimental convenience features.
- Reduce special-case branching where multiple algorithm aliases resolve to near-identical paths.
- Prefer shared helpers only where behavior is genuinely shared; otherwise keep variant logic isolated and explicit.

## Proposed Step-by-Step Cleanup Sequence

1. Artifact cleanup
- Remove or archive non-essential notebooks and smoke-analysis artifacts.
- Update docs so any removed artifact still has an equivalent documented workflow if needed.

2. Supported algorithm matrix
- Write down the thesis-supported algorithms, presets, and scenarios.
- Mark the rest as experimental in docs and tests.

3. Documentation consolidation
- Reduce duplication between README and thesis docs.
- Keep one canonical experiment workflow page.

4. RLlib runner simplification
- Audit aliases, duplicated branches, and evaluation-specific special cases.
- Remove dead or low-value code paths only after the supported matrix is explicit.

5. Test-suite alignment
- Align tests with the supported matrix and artifact cleanup decisions.
- Demote experimental-variant tests that do not protect supported workflows.

## Merge Strategy For Follow-Up

Each cleanup item should land in its own review branch and PR-sized commit sequence:

- `cleanup-artifacts`
- `cleanup-supported-algorithm-matrix`
- `cleanup-doc-consolidation`
- `cleanup-rllib-runner`
- `cleanup-test-alignment`

This keeps the repository reviewable while preserving the merged `rllib-minhton` baseline.
