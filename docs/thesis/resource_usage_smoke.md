# DCRNN Resource-Usage Smoke

Use `experiments/dcrnn_resource_smoke.py` for short resource-usage checks on the
DCRNN RLlib variants before launching longer Experiment Group C jobs.

Example:

```bash
python experiments/dcrnn_resource_smoke.py ^
  --variants ppo_dcrnn_mlp ppo_dcrnn_shared_mlp ^
  --scenario resco_grid4x4 ^
  --episodes 2 ^
  --episode-seconds 300 ^
  --train-batch-size 256 ^
  --sgd-minibatch-size 64 ^
  --num-sgd-iter 1
```

The script writes one subdirectory per variant under
`outputs/dcrnn_resource_smoke/<timestamp>/` and records:

- `resource_usage_summary.csv` and `resource_usage_summary.json`
- per-variant `resource_usage/summary.json`
- per-variant `resource_usage/gpu_samples.csv` when `nvidia-smi` is available

Reported smoke metrics:

- `parameter_count`
- `parameter_encoder_count`
- `parameter_actor_count`
- `parameter_critic_count`
- `parameter_other_count`
- `history_len`, `num_nodes`, `feature_dim`, `num_policies`
- `per_sample_obs_bytes`, `rollout_obs_bytes`, `minibatch_obs_bytes`
- `train_batch_size_per_learner_actual`, `minibatch_size_actual`
- `cuda_post_build_*`, `cuda_after_warmup_inference_*`
- `cuda_train_post_build_*`, `cuda_after_first_train_iteration_*`, `cuda_after_training_end_*`
- `shared_forward_inference_*`, `shared_forward_train_first_iteration_*`, `shared_forward_train_total_*`
- `env_pipeline`, `env_base_factory`, `env_wrapper`, `env_observation_mode`
- `wall_clock_training_seconds`
- `inference_joint_decision_ms`
- `inference_agent_action_ms`
- `gpu_peak_memory_delta_mb`
- `gpu_average_utilization_pct`

The default smoke settings intentionally shrink the PPO learner batch and SGD
loop so the script reaches at least one learner update quickly without turning
the smoke run into a full experiment.

`per_sample_obs_bytes` is computed directly from the Gymnasium observation
space when possible, so flat PPO observations and graph-history DCRNN
observations are both covered. The `history_len` / `num_nodes` /
`feature_dim` fields are populated only for graph-style `[H, N, F]`
observations.
