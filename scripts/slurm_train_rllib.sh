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

mkdir -p outputs/slurm

echo "=== SLURM JOB ALLOCATION ==="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job name:      ${SLURM_JOB_NAME:-local}"
echo "Node list:     ${SLURM_JOB_NODELIST:-local}"
echo "Tasks:         ${SLURM_NTASKS:-unset}"
echo "CPUs/task:     ${SLURM_CPUS_PER_TASK:-unset}"
echo "Memory/node:   ${SLURM_MEM_PER_NODE:-unset} MiB"
echo "Memory/CPU:    ${SLURM_MEM_PER_CPU:-unset} MiB"
echo "Visible GPUs:  ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -z "${SLURM_MEM_PER_NODE:-}" && -z "${SLURM_MEM_PER_CPU:-}" ]]; then
  echo "WARNING: Slurm did not export memory environment variables; checking scontrol/sacct output instead." >&2
fi

if command -v scontrol >/dev/null 2>&1; then
  echo "=== SLURM JOB TRES ==="
  scontrol show job "$SLURM_JOB_ID" |
    grep -E 'JobId=|ReqTRES=|AllocTRES=|TresPerNode=|MinMemory' || true
fi

echo "=== NODE MEMORY SNAPSHOT ==="
if command -v free >/dev/null 2>&1; then
  free -h || true
else
  echo "free command not available"
fi

echo "=== NODE SLURM SNAPSHOT ==="
if command -v scontrol >/dev/null 2>&1; then
  scontrol show node "${SLURM_JOB_NODELIST}" |
    grep -E 'NodeName=|State=|CPUAlloc=|CPUTot=|CPULoad=|RealMemory=|AllocMem=|FreeMem=|Gres=|GresUsed=' || true
else
  echo "scontrol command not available"
fi

echo "=== GPU SNAPSHOT ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv || nvidia-smi || true
else
  echo "nvidia-smi command not available"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_ENV_FILE="${WANDB_ENV_FILE:-.env}"
  if [[ -f "$WANDB_ENV_FILE" ]]; then
    echo "Loading W&B environment from $WANDB_ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    source "$WANDB_ENV_FILE"
    set +a
  fi
fi

if [[ " $* " != *" logging=disabled "* ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'
  else
    echo "WANDB_API_KEY is not set; using any existing W&B login on this server."
  fi
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export RAY_TMPDIR="${SLURM_TMPDIR:-$PWD/.ray_tmp}"

mkdir -p "$RAY_TMPDIR"

echo "SUMO_HOME=$SUMO_HOME"
echo "RAY_TMPDIR=$RAY_TMPDIR"
echo "After completion, inspect usage with:"
echo "sacct -j ${SLURM_JOB_ID} --format=JobID,JobName%25,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AveRSS"

python -u experiments/rllib.py \
  "$@" \
  "resources.ray_num_cpus=${SLURM_CPUS_PER_TASK}" \
  resources.native_num_threads=1 \
  resources.cuda_visible_devices=null
