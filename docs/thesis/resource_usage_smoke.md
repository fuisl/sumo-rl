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
- per-variant `resource_usage/episode_rows.csv` and `resource_usage/episode_rows.json`

Recommended comparison fields:

- `parameter_count`
- `parameter_encoder_count`
- `parameter_actor_count`
- `parameter_critic_count`
- `parameter_other_count`
- `per_sample_obs_bytes`, `rollout_obs_bytes`, `minibatch_obs_bytes`
- `used_cuda`
- `driver_cuda_post_build_*`, `driver_cuda_after_warmup_inference_*`, `driver_cuda_after_training_end_*`
- `env_pipeline`, `env_base_factory`, `env_wrapper`, `env_observation_mode`
- `inference_joint_decision_ms`
- `inference_agent_action_ms`
- `inference_encoder_call_count`
- `inference_encoder_time_ms`
- `inference_encoder_calls_per_joint_decision`
- `inference_shared_forward_hit_rate`
- `episode_row_count`
- `episode_wall_clock_seconds_mean`
- `episode_shared_forward_hit_rate_mean`
- `episode_inference_encoder_call_count_mean`
- `episode_inference_encoder_time_ms_mean`
- `episode_inference_encoder_calls_per_joint_decision_mean`
- `episode_inference_shared_forward_hit_rate_mean`
- `wall_clock_training_seconds`

Per-episode resource rows now record completed environment episodes rather than
learner iterations. The final per-variant summary keeps run-level metadata and
adds episode-averaged fields such as:

- `episode_row_count`
- `episode_wall_clock_seconds_mean`
- `episode_shared_forward_hit_rate_mean`
- `episode_inference_encoder_call_count_mean`
- `episode_inference_encoder_time_ms_mean`
- `episode_inference_encoder_calls_per_joint_decision_mean`

Peak-style internal CUDA episode metrics also keep a run max in the final
summary, for example `episode_driver_cuda_max_memory_allocated_mb_run_max`.

The default smoke settings intentionally shrink the PPO learner batch and SGD
loop so the script reaches at least one learner update quickly without turning
the smoke run into a full experiment.

`per_sample_obs_bytes` is computed directly from the Gymnasium observation
space when possible, so flat PPO observations and graph-history DCRNN
observations are both covered. Device-level `nvidia-smi` sampling is no longer
part of the main smoke comparison because it is too noisy on shared servers;
prefer the driver-local `torch.cuda.*` snapshot fields instead. `used_cuda`
is runner-aware and may be `1` even when the driver-local `driver_cuda_*`
fields are `None`, because Ray learner or worker processes can own the actual
CUDA memory. Encoder call/time metrics are measured through explicit driver-side
inference probes so you can compare how often each variant executes
`DCRNNBackbone.encode_graph()` per joint decision even when learner-side CUDA
memory lives inside Ray worker processes.
