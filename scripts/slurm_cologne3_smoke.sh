#!/usr/bin/env bash
#SBATCH --job-name=sumo-rl-cologne3-smoke
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=sumo-rl-cologne3-smoke-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export RAY_TMPDIR="${SLURM_TMPDIR:-$PWD/.ray_tmp}"
export RAY_CPUS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p outputs/slurm "$RAY_TMPDIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SUMO_HOME=$SUMO_HOME"
echo "RAY_TMPDIR=$RAY_TMPDIR"

bash scripts/smoke_cologne3.sh
