#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/submit_slurm.sh [--profile] <slurm-script> [sbatch-options] [hydra-overrides...]

Examples:
  bash scripts/submit_slurm.sh --profile scripts/slurm_train_rllib.sh \
    algorithm=ppo scenario=resco_ingolstadt7

  bash scripts/submit_slurm.sh scripts/slurm_train_rllib.sh \
    --mem=14G --time=08:00:00 --job-name=sumo-rl-ppo-ingolstadt7 \
    algorithm=ppo scenario=resco_ingolstadt7 experiment.episodes=200
USAGE
}

profile=false

if [[ $# -gt 0 && "$1" == "--profile" ]]; then
  profile=true
  shift
fi

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script="$1"
shift

if [[ ! -f "$script" ]]; then
  echo "ERROR: Slurm script not found: $script" >&2
  exit 1
fi

required=(
  "--job-name"
  "--nodes"
  "--ntasks"
  "--cpus-per-task"
  "--mem"
  "--gres"
  "--time"
  "--output"
)

for option in "${required[@]}"; do
  if ! grep -qE "^#SBATCH[[:space:]]+${option}(=|[[:space:]])" "$script"; then
    echo "ERROR: Missing #SBATCH ${option} in ${script}" >&2
    exit 1
  fi
done

sbatch_args=()
hydra_args=()
has_mem=false
has_time=false
has_job_name=false

takes_value() {
  case "$1" in
    --account|--begin|--constraint|--cpus-per-task|--dependency|--exclude|--gres|--job-name|--mail-type|--mail-user|--mem|--nodes|--ntasks|--output|--partition|--qos|--time|--wckey)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_sbatch_flag() {
  case "$1" in
    --exclusive|--no-requeue|--requeue|--test-only|--wait)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

remember_override() {
  case "$1" in
    --mem) has_mem=true ;;
    --time) has_time=true ;;
    --job-name) has_job_name=true ;;
  esac
}

while [[ $# -gt 0 ]]; do
  arg="$1"
  shift

  if [[ "$arg" == --*=* ]]; then
    option_name="${arg%%=*}"
    if takes_value "$option_name"; then
      sbatch_args+=("$arg")
      remember_override "$option_name"
    else
      hydra_args+=("$arg")
    fi
  elif takes_value "$arg"; then
    if [[ $# -lt 1 ]]; then
      echo "ERROR: Missing value for sbatch option ${arg}" >&2
      exit 1
    fi
    sbatch_args+=("$arg")
    remember_override "$arg"
    sbatch_args+=("$1")
    shift
  elif is_sbatch_flag "$arg"; then
    sbatch_args+=("$arg")
  else
    hydra_args+=("$arg")
  fi
done

profile_sbatch_args=()
profile_hydra_args=()

if [[ "$profile" == true ]]; then
  if [[ "$has_mem" == false ]]; then
    profile_sbatch_args+=(--mem=16G)
  fi
  if [[ "$has_time" == false ]]; then
    profile_sbatch_args+=(--time=00:30:00)
  fi
  if [[ "$has_job_name" == false ]]; then
    profile_sbatch_args+=(--job-name=sumo-rl-profile)
  fi

  profile_hydra_args+=(
    experiment.episodes=5
    experiment.validation_interval_episodes=5
    experiment.eval_episodes=1
    logging=disabled
  )
fi

echo "Submitting ${script}"
if [[ "$profile" == true ]]; then
  echo "Mode: profile"
fi
if [[ ${#sbatch_args[@]} -gt 0 || ${#profile_sbatch_args[@]} -gt 0 ]]; then
  echo "SBATCH options: ${profile_sbatch_args[*]} ${sbatch_args[*]}"
fi
if [[ ${#profile_hydra_args[@]} -gt 0 || ${#hydra_args[@]} -gt 0 ]]; then
  echo "Hydra overrides: ${hydra_args[*]} ${profile_hydra_args[*]}"
fi

sbatch "${profile_sbatch_args[@]}" "${sbatch_args[@]}" "$script" "${hydra_args[@]}" "${profile_hydra_args[@]}"
