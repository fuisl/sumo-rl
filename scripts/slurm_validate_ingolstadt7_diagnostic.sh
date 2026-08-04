#!/usr/bin/env bash
#SBATCH --job-name=sumo-rl-validate-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/validate-diagnostic-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

for arg in "$@"; do
  case "$arg" in
    RUN_DIR=*) RUN_DIR="${arg#RUN_DIR=}" ;;
    SEED=*) SEED="${arg#SEED=}" ;;
    JUNCTION=*) JUNCTION="${arg#JUNCTION=}" ;;
    MAX_DECISION_STEPS=*) MAX_DECISION_STEPS="${arg#MAX_DECISION_STEPS=}" ;;
    PROGRESS_LOG_STEPS=*) PROGRESS_LOG_STEPS="${arg#PROGRESS_LOG_STEPS=}" ;;
    RAY_NUM_GPUS=*) RAY_NUM_GPUS="${arg#RAY_NUM_GPUS=}" ;;
    RAY_NUM_CPUS=*) RAY_NUM_CPUS="${arg#RAY_NUM_CPUS=}" ;;
    NATIVE_NUM_THREADS=*) NATIVE_NUM_THREADS="${arg#NATIVE_NUM_THREADS=}" ;;
    DIAGNOSTIC_DEMAND_ABLATION=*) DIAGNOSTIC_DEMAND_ABLATION="${arg#DIAGNOSTIC_DEMAND_ABLATION=}" ;;
    *)
      echo "ERROR: Unsupported validation argument: $arg" >&2
      echo "Use KEY=VALUE arguments such as RUN_DIR=outputs/rllib/<run> JUNCTION=gneJ143." >&2
      exit 1
      ;;
  esac
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export RAY_TMPDIR="${SLURM_TMPDIR:-$PWD/.ray_tmp}"

mkdir -p outputs/slurm "$RAY_TMPDIR"

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: RUN_DIR is required." >&2
  echo "Example: RUN_DIR=outputs/rllib/2026-08-04_03-47-17" >&2
  exit 1
fi

SEED="${SEED:-0}"
JUNCTION="${JUNCTION:-gneJ143}"
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-30}"
PROGRESS_LOG_STEPS="${PROGRESS_LOG_STEPS:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-4}}"
NATIVE_NUM_THREADS="${NATIVE_NUM_THREADS:-1}"
DIAGNOSTIC_DEMAND_ABLATION="${DIAGNOSTIC_DEMAND_ABLATION:-none}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SUMO_HOME=$SUMO_HOME"
echo "RAY_TMPDIR=$RAY_TMPDIR"
echo "RUN_DIR=$RUN_DIR"
echo "SEED=$SEED"
echo "JUNCTION=$JUNCTION"
echo "MAX_DECISION_STEPS=$MAX_DECISION_STEPS"
echo "PROGRESS_LOG_STEPS=$PROGRESS_LOG_STEPS"
echo "RAY_NUM_GPUS=$RAY_NUM_GPUS"
echo "RAY_NUM_CPUS=$RAY_NUM_CPUS"
echo "NATIVE_NUM_THREADS=$NATIVE_NUM_THREADS"
echo "DIAGNOSTIC_DEMAND_ABLATION=$DIAGNOSTIC_DEMAND_ABLATION"
nvidia-smi -L || true

python -u experiments/validate_methods.py \
  --controller rllib \
  --run-dir "$RUN_DIR" \
  --checkpoint-selector best \
  --seeds "$SEED" \
  --parallel-workers 1 \
  --diagnostic-junction "$JUNCTION" \
  --diagnostic-demand-ablation "$DIAGNOSTIC_DEMAND_ABLATION" \
  --max-decision-steps "$MAX_DECISION_STEPS" \
  --progress-log-steps "$PROGRESS_LOG_STEPS" \
  --ray-num-gpus "$RAY_NUM_GPUS" \
  --ray-num-cpus "$RAY_NUM_CPUS" \
  --native-num-threads "$NATIVE_NUM_THREADS"
