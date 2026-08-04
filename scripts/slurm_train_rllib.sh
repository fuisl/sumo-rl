#!/usr/bin/env bash
#SBATCH --job-name=sumo-rl-rllib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

if [[ -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
  echo "ERROR: SLURM_CPUS_PER_TASK is not set." >&2
  exit 1
fi

if [[ -z "${SLURM_MEM_PER_NODE:-}" ]]; then
  echo "ERROR: SLURM_MEM_PER_NODE is not set. Request memory with #SBATCH --mem or sbatch --mem." >&2
  exit 1
fi

mkdir -p outputs/slurm

echo "=== SLURM ALLOCATION ==="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job name:      ${SLURM_JOB_NAME:-local}"
echo "Node list:     ${SLURM_JOB_NODELIST:-local}"
echo "Tasks:         ${SLURM_NTASKS:-unset}"
echo "CPUs/task:     ${SLURM_CPUS_PER_TASK:-unset}"
echo "Memory/node:   ${SLURM_MEM_PER_NODE:-unset} MiB"
echo "Memory/CPU:    ${SLURM_MEM_PER_CPU:-unset} MiB"
echo "Visible GPUs:  ${CUDA_VISIBLE_DEVICES:-unset}"

if command -v scontrol >/dev/null 2>&1; then
  scontrol show job "$SLURM_JOB_ID" |
    grep -E 'JobId=|ReqTRES=|AllocTRES=|TresPerNode=|MinMemory' || true
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export RAY_TMPDIR="${SLURM_TMPDIR:-$PWD/.ray_tmp}"

mkdir -p "$RAY_TMPDIR"

echo "SUMO_HOME=$SUMO_HOME"
echo "RAY_TMPDIR=$RAY_TMPDIR"

python -u experiments/rllib.py \
  "$@" \
  "resources.ray_num_cpus=${SLURM_CPUS_PER_TASK}" \
  resources.native_num_threads=1 \
  resources.cuda_visible_devices=null
