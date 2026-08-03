#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-15}"
EPISODE_SECONDS="${EPISODE_SECONDS:-300}"
RAY_CPUS="${RAY_CPUS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

python -m pytest tests/core/test_rllib_config_defaults.py

python experiments/fixed_time.py \
  scenario=resco_cologne3 \
  "experiment.episodes=${EPISODES}" \
  "experiment.episode_seconds=${EPISODE_SECONDS}" \
  logging=disabled

python experiments/static_max_pressure.py \
  scenario=resco_cologne3 \
  "experiment.episodes=${EPISODES}" \
  "experiment.episode_seconds=${EPISODE_SECONDS}" \
  logging=disabled

python experiments/rllib.py \
  algorithm=ppo \
  scenario=resco_cologne3 \
  "experiment.episodes=${EPISODES}" \
  "experiment.episode_seconds=${EPISODE_SECONDS}" \
  experiment.validation_interval_episodes=5 \
  experiment.eval_episodes=2 \
  logging=disabled \
  logging.checkpoint_every_episodes=5 \
  "resources.ray_num_cpus=${RAY_CPUS}" \
  resources.native_num_threads=1 \
  resources.cuda_visible_devices=null

python experiments/rllib.py \
  algorithm=dqn \
  scenario=resco_cologne3 \
  "experiment.episodes=${EPISODES}" \
  "experiment.episode_seconds=${EPISODE_SECONDS}" \
  experiment.validation_interval_episodes=5 \
  experiment.eval_episodes=2 \
  algorithm.params.num_steps_sampled_before_learning_starts=0 \
  logging=disabled \
  "resources.ray_num_cpus=${RAY_CPUS}" \
  resources.native_num_threads=1 \
  resources.cuda_visible_devices=null
