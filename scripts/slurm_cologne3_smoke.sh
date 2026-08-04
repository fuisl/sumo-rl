#!/usr/bin/env bash
#SBATCH --job-name=sumo-rl-ppo-cologne3-smoke
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/ppo-cologne3-smoke-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export RAY_TMPDIR="${SLURM_TMPDIR:-$PWD/.ray_tmp}"

mkdir -p outputs/slurm "$RAY_TMPDIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SUMO_HOME=$SUMO_HOME"
echo "RAY_TMPDIR=$RAY_TMPDIR"

python experiments/rllib.py \
  algorithm=ppo \
  scenario=resco_cologne3 \
  experiment.episodes=10 \
  experiment.episode_seconds=300 \
  experiment.validation_interval_episodes=5 \
  experiment.eval_episodes=2 \
  logging=disabled \
  logging.checkpoint_every_episodes=5 \
  "resources.ray_num_cpus=${SLURM_CPUS_PER_TASK:-8}" \
  resources.native_num_threads=1 \
  resources.cuda_visible_devices=null
