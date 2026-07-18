# DCRNN Resource-Usage Smoke

Use `experiments/archive/dcrnn_resource_smoke.py` for short resource-usage
checks on the DCRNN RLlib variants before launching longer Experiment Group C
jobs. This is a retained reference utility, not a supported top-level launcher.

Example:

```bash
python experiments/archive/dcrnn_resource_smoke.py ^
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

## Parameter counting

`experiments/archive/dcrnn_resource_smoke.py` counts trainable parameters only and
deduplicates shared tensors by `(data_ptr, numel)` before summing them. This
matters for shared-backbone variants, because the same module can be reachable
through more than one child policy/module reference.

The exported fields mean:

- `parameter_count`: unique trainable parameters across the resolved module set
- `parameter_encoder_count`: unique trainable parameters under
  `backbone` or `shared_backbone`
- `parameter_actor_count`: unique trainable parameters under `policy_head`
- `parameter_critic_count`: unique trainable parameters under `value_head`
- `parameter_other_count`: `parameter_count - encoder - actor - critic`

For the DCRNN PPO variants, this split matches the implementation directly:

- encoder = `DCRNNBackbone`
- actor = PPO policy head MLP
- critic = PPO value head MLP

For baseline PPO, the current smoke script does not break out RLlib's default
shared MLP trunk separately, so `parameter_encoder_count` is reported as `0`.
That is a reporting convention in the smoke script, not a claim that baseline
PPO has no shared feature extractor.

## Conceptual PPO split

If you want a fair architecture-level comparison against DCRNN PPO, treat the
baseline PPO MLP trunk as the encoder:

- encoder = all shared hidden layers before the policy/value outputs
- actor = policy output layer
- critic = value output layer

For a dense layer with bias:

```text
params = in_features * out_features + out_features
```

RLlib PPO currently defaults to `fcnet_hiddens=[256, 256]`, so for one policy
with flat observation width `obs_dim` and discrete action count `action_dim`:

```text
encoder_per_policy =
    (obs_dim * 256 + 256)
  + (256 * 256 + 256)
  = 256 * obs_dim + 66,048

actor_per_policy =
    256 * action_dim + action_dim

critic_per_policy =
    256 * 1 + 1
  = 257
```

In independent-policy PPO, sum those counts across all traffic-signal policies.
In shared-policy PPO, compute them once for the merged shared policy.

## Current scenario examples

The values below use the current repo's probed PPO observation/action spaces and
the default RLlib PPO `256,256` MLP trunk.

### Cologne1

Current repo space:

- one policy
- `obs_dim=21`
- `action_dim=4`

Counts:

```text
encoder = (21 * 256 + 256) + (256 * 256 + 256) = 71,424
actor   = 256 * 4 + 4 = 1,028
critic  = 257
total   = 71,424 + 1,028 + 257 = 72,709
```

If you see an older row with total `72,452`, that row came from a slightly
different action-space or PPO configuration than the current repo state.

### Cologne8

Current per-policy `(obs_dim, action_dim)` pairs:

```text
(17,4), (11,2), (10,3), (17,4), (12,3), (7,2), (12,3), (13,4)
```

Summed counts:

```text
encoder = 553,728
actor   = 6,425
critic  = 2,056
total   = 562,209
```

### Ingolstadt21

Current per-policy `(obs_dim, action_dim)` pairs:

```text
(18,3), (16,3), (12,3), (16,3), (21,4), (20,3), (20,3), (17,2), (18,3),
(21,4), (16,3), (19,4), (16,3), (33,4), (28,3), (22,3), (18,3), (14,3),
(24,3), (20,3), (14,3)
```

Summed counts:

```text
encoder = 1,490,176
actor   = 16,962
critic  = 5,397
total   = 1,512,535
```

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
