# Running Training On SLURM

Use this workflow for remote training:

```text
resource check -> profile/probe -> live/end resource check -> full training
```

## 1. Check Current Resources

Before submitting, inspect the partition and the node:

```bash
sinfo -o "%P %N %t %c %m %G"
scontrol show node gpu-a240 | grep -E "NodeName=|State=|CPUAlloc=|CPUTot=|CPULoad=|RealMemory=|AllocMem=|FreeMem=|Gres=|GresUsed="
squeue -w gpu-a240
free -h
nvidia-smi
```

Read memory with care:

- `free` can be small because Linux uses RAM as cache.
- `available` from `free -h` is more useful than `free`.
- Slurm `AllocMem` shows memory already allocated to jobs.
- `sacct MaxRSS` from your own completed job is the best sizing evidence.

## 2. Run A Profile/Probe Job

Profile mode is a short 5-episode Slurm job. It disables W&B logging and uses
default probe resources unless you override them:

```bash
bash scripts/submit_slurm.sh --profile scripts/slurm_train_rllib.sh \
  algorithm=ppo \
  scenario=resco_cologne3 \
  +env.kwargs.reward_fn=weighted-nash-average-speed \
  +env.kwargs.reward_nash_epsilon=0.001
```

The profile defaults are:

```text
--mem=16G
--time=00:30:00
experiment.episodes=5
experiment.validation_interval_episodes=5
experiment.eval_episodes=1
logging=disabled
```

## 3. Check Usage Live And After Completion

While the job is running:

```bash
squeue -j <JOB_ID> -o "%.18i %.30j %.8T %.10M %.20R"
watch -n 5 nvidia-smi
```

After the job finishes:

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName%25,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AveRSS
```

For GPU allocation and tracked TRES usage:

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName%25,State,Elapsed,AllocTRES%80,ReqTRES%80,TRESUsageInMax%120,TRESUsageInAve%120
```

If Slurm does not report GPU memory or utilization, use `nvidia-smi` during the
next profile job.

## 4. Choose Full-Run Resources

Use these rules of thumb:

```text
RAM request = observed MaxRSS * 1.25 to 1.5, rounded up
Wall time   = profile runtime scaled by episodes, then * 1.2 to 1.3
CPUs        = smallest value that completes the profile without stalling Ray
GPU         = smallest GPU or MIG profile with enough VRAM and throughput
```

For the Cologne3 PPO profile that used about `2.4G` RAM, `--mem=6G` is a
reasonable full-run starting point.

Example full training run:

```bash
bash scripts/submit_slurm.sh scripts/slurm_train_rllib.sh \
  --job-name=sumo-rl-ppo-cologne3-nsw \
  --cpus-per-task=4 \
  --mem=6G \
  --gres=gpu:a100_2g.10gb:1 \
  --time=02:00:00 \
  algorithm=ppo \
  scenario=resco_cologne3 \
  experiment.episodes=200 \
  experiment.validation_interval_episodes=5 \
  experiment.eval_episodes=2 \
  logging=wandb \
  logging.mode=online \
  logging.checkpoint_every_episodes=5 \
  +env.kwargs.reward_fn=weighted-nash-average-speed \
  +env.kwargs.reward_nash_epsilon=0.001
```

If `WANDB_API_KEY` is set, the Slurm runner logs in before online W&B training.
Profile mode always appends `logging=disabled`.

## 5. Record The Result

After every completed full run, save the useful row:

```text
Scenario        Algorithm  Reward                       CPUs  MemReq  MaxRSS  GPU                 Runtime
resco_cologne3  PPO        weighted-nash-average-speed  4     6G      <fill>  a100_2g.10gb:1      <fill>
```

Use this record to pick resources for the next run instead of guessing.

## 6. Run A Checkpoint Validation Diagnostic

Use the validation diagnostic Slurm script to inspect one junction from a saved
RLlib run without keeping an SSH terminal busy. `RUN_DIR` is required and should
point to the Hydra run directory that contains `.hydra/` and `checkpoints/`.

```bash
bash scripts/submit_slurm.sh scripts/slurm_validate_ingolstadt7_diagnostic.sh \
  --job-name=sumo-rl-validate-gneJ143 \
  --cpus-per-task=4 \
  --mem=8G \
  --gres=gpu:a100_2g.10gb:1 \
  --time=00:30:00 \
  RUN_DIR=outputs/rllib/2026-08-04_03-47-17 \
  SEED=0 \
  JUNCTION=gneJ143 \
  MAX_DECISION_STEPS=30 \
  PROGRESS_LOG_STEPS=1 \
  RAY_NUM_GPUS=1 \
  RAY_NUM_CPUS=4 \
  NATIVE_NUM_THREADS=1 \
  DIAGNOSTIC_DEMAND_ABLATION=none
```

The diagnostic writes the usual validation outputs plus per-junction traces
under the validation run directory:

```text
seed_<seed>/diagnostics/<junction>_decisions.csv
diagnostics/<junction>_decisions_all_seeds.csv
```
