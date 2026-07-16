# AGENTS.md

## Purpose
This repository is a SUMO-RL thesis codebase. Keep changes small, reproducible, and easy to explain to the group.

## Working Rules
- Prefer the existing examples and environment APIs before introducing new abstractions.
- Keep training code configurable and avoid hard-coding experiment values in scripts.
- Use `Hydra` for experiment composition and `wandb` for run tracking once integrated.
- Preserve current example behavior unless a change is explicitly requested.
- Do not delete or rewrite unrelated files or user changes.
- When aligning with RESCO, keep the metric formulas and field names synchronized with the benchmark source and update docs at the same time.
- For fixed-time and static baselines, keep per-seed trace rows in both W&B and CSV, and keep the final summary as a seed average.
- Keep active planning local and lightweight. Do not turn tracked repo docs into a running scratchpad for unrelated work.

## Code Style
- Use ASCII unless a file already uses non-ASCII.
- Keep comments brief and only where the code is not obvious.
- Prefer clear module boundaries over large monolithic scripts.

## Git Workflow
- Follow conventional commit style for commit messages.
- Use prefixes such as `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, and similar standard forms.
- Keep commit messages short, specific, and scoped to the actual change.

## Experiment Workflow
- Document new entrypoints in `README.md` or stable docs when they affect how experiments are run.
- Keep outputs, logs, and checkpoints under dedicated run directories.
- When adding an algorithm, note whether it is:
  - a handwritten baseline,
  - a Stable-Baselines3 integration,
  - a third-party SB3 integration,
  - or a future/optional extension.
- When adding benchmark presets like new road networks, add the launcher script and the matching Hydra config together.

## Planning And Documentation
- Use local `PLANS.md` for in-progress planning and working notes.
- Use `docs/architecture/` for durable design material that should outlive a feature plan.
- Use `docs/decisions/` for repository-level technical decisions with lasting impact.
- Use stable docs such as `README.md` and `docs/thesis/` for contributor-facing workflows and canonical run instructions.
- Keep local plans short and practical: goal, scope, steps, open questions, and validation.
- Do not use tracked docs as personal diaries of every command or minor experiment.
- After a workstream is complete, extract only the durable parts into permanent docs and discard obsolete planning detail.

## Verification
- Prefer small smoke tests for environment and trainer wiring.
- For experiment scripts, verify that they run on a short horizon before scaling up.
