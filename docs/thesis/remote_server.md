# Remote Server Setup

Use this workflow for a Linux or SLURM server without root, conda, or
micromamba. The supported install is a project-local Python `venv` with SUMO
provided by the `eclipse-sumo` wheel.

For the day-to-day SLURM process, start with the root [RUNNING.md](../../RUNNING.md)
guide.

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

## Profile Runs

SLURM profile smoke:

```bash
bash scripts/submit_slurm.sh --profile scripts/slurm_train_rllib.sh \
  algorithm=ppo \
  scenario=resco_cologne3
```

Profile mode uses the reusable RLlib runner for a 5-episode SLURM job with
W&B disabled. Use it to check wiring, runtime, and memory before full training.

## SLURM Workflow

Treat every SLURM script as a resource contract: request the resources, measure
what the job used, then update the next request from evidence. Use the tracked
generic runner instead of editing or copying per-experiment scripts:

```bash
bash scripts/submit_slurm.sh --profile scripts/slurm_train_rllib.sh \
  algorithm=ppo \
  scenario=resco_ingolstadt7
```

Profile mode submits a 5-episode inspection run with:

```text
experiment.episodes=5
experiment.validation_interval_episodes=5
experiment.eval_episodes=1
logging=disabled
--mem=16G
--time=00:30:00
```

Inspect the completed job before submitting the full run:

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName%25,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AveRSS
```

`MaxRSS` is usually most useful on the `.batch` or job-step row. Request RAM as
the observed peak times 1.25-1.5, and scale wall time from the profile runtime
with about 1.2-1.3 headroom. Do not request exactly the observed peak because
checkpointing, evaluation, SUMO scenario variance, and Ray object-store peaks
can increase memory use.

Submit the full run with explicit resource overrides, without editing tracked
scripts. If `WANDB_API_KEY` is set in the environment, the SLURM runner logs in
before starting online W&B training:

```bash
bash scripts/submit_slurm.sh scripts/slurm_train_rllib.sh \
  --mem=14G \
  --time=08:00:00 \
  --job-name=sumo-rl-ppo-ingolstadt7 \
  algorithm=ppo \
  scenario=resco_ingolstadt7 \
  experiment.episodes=200 \
  logging=wandb \
  logging.mode=online
```

For repeated usage checks, add this helper to `~/.bashrc` on the server:

```bash
slurm_usage() {
  local job_id="${1:?Usage: slurm_usage JOB_ID}"

  sacct -j "$job_id" \
    --format=JobID%18,JobName%28,State%12,Elapsed,AllocCPUS,ReqMem,MaxRSS,AveRSS
}
```

Then run:

```bash
slurm_usage <JOB_ID>
```

The standard loop is:

```text
5-episode profile -> inspect MaxRSS/runtime -> add headroom -> full run -> record usage
```

## SLURM Notes

- Let SLURM own GPU visibility; keep `resources.cuda_visible_devices=null`.
- Pass CPU allocation with `resources.ray_num_cpus="${SLURM_CPUS_PER_TASK}"`.
- Keep `algorithm.params.local_gpu_idx=0`; SLURM usually exposes the allocated
  GPU as local CUDA device `0`.
- Keep `resources.native_num_threads=1` to avoid CPU oversubscription.
- Use `resources.ray_address=null` unless the cluster explicitly provides a
  shared Ray head.
- Use `--mem`, not `--mem-per-cpu`, for these single-node Ray/SUMO jobs.
- Some clusters do not export `SLURM_MEM_PER_NODE`; trust `sacct ReqMem` and
  the allocation report in the job log when the wrapper submitted an explicit
  `--mem`.
- Keep custom generated job scripts under `outputs/slurm/` if they are needed;
  tracked scripts should stay reusable.

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
