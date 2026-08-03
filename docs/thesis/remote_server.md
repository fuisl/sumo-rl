# Remote Server Setup

Use this workflow for a Linux or SLURM server without root, conda, or
micromamba. The supported install is a project-local Python `venv` with SUMO
provided by the `eclipse-sumo` wheel.

## Install

```bash
cd ~/projects/sumo-rl
bash scripts/setup_remote_venv.sh
source .venv/bin/activate
export SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
```

The script creates `.venv`, installs `python -m pip install -e ".[server]"`,
and verifies `sumo`, `sumolib`, `traci`, `libsumo`, Ray, W&B, and Torch.

Use a specific interpreter when needed:

```bash
PYTHON_BIN=python3.11 bash scripts/setup_remote_venv.sh
```

## Smoke Tests

Interactive Cologne3 smoke:

```bash
bash scripts/smoke_cologne3.sh
```

Shorter smoke while iterating:

```bash
EPISODES=3 EPISODE_SECONDS=120 RAY_CPUS=4 bash scripts/smoke_cologne3.sh
```

SLURM batch smoke:

```bash
sbatch scripts/slurm_cologne3_smoke.sh
```

The smoke scripts run fixed-time, static max-pressure, PPO, and DQN on
Cologne3. Defaults are 15 episodes and 300 simulation seconds.

## SLURM Notes

- Let SLURM own GPU visibility; keep `resources.cuda_visible_devices=null`.
- Pass CPU allocation with `resources.ray_num_cpus="${SLURM_CPUS_PER_TASK:-8}"`.
- Keep `algorithm.params.local_gpu_idx=0`; SLURM usually exposes the allocated
  GPU as local CUDA device `0`.
- Keep `resources.native_num_threads=1` to avoid CPU oversubscription.
- Use `resources.ray_address=null` unless the cluster explicitly provides a
  shared Ray head.

## Results

Hydra writes runs under:

```text
outputs/<experiment-name>/<timestamp>/
```

For thesis results, keep `.hydra/config.yaml`, `logs/metrics.csv`,
`checkpoints/`, W&B output, and the Git commit:

```bash
git rev-parse HEAD
python -m pip freeze > outputs/server-venv-requirements.txt
```
