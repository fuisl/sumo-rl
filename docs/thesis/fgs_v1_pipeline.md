---
title: FGS v1 Pipeline
firstpage:
---

# FGS v1 Pipeline

This page documents the first FGS implementation in this thesis repo:
`algorithm=fgs`, the FRAP-GNN-SAC path under RLlib.

FGS v1 means:

- FRAP-style or MLP local encoding per traffic signal.
- CoLight-style graph attention, or the later GATv2 ablation, over a traffic-light graph.
- A shared decentralized discrete SAC actor.
- Centralized graph critics during training.
- Hydra-driven startup through `experiments/rllib.py`.

`algorithm=fgs_ppo` is a sibling ablation that reuses the same graph wrapper,
topology extraction, local encoder, and GAT/GATv2 encoder, but swaps the SAC
actor/critic/learner for PPO policy and value heads. The rest of this document
focuses on the original SAC v1 path unless it says otherwise.

## How It Is Started

FGS v1 does not have a standalone launcher. It starts through the shared RLlib
entrypoint:

```bash
python experiments/rllib.py algorithm=fgs scenario=resco_grid4x4
```

For a short smoke run:

```bash
source .venv/bin/activate
python experiments/rllib.py \
  algorithm=fgs \
  scenario=single_intersection \
  experiment.episodes=1 \
  experiment.episode_seconds=60 \
  logging=disabled \
  resources.ray_address=null
```

For the scenario-first presets:

```bash
python experiments/rllib.py --config-name presets/resco_grid4x4/fgs_frap_gat_sac
python experiments/rllib.py --config-name presets/resco_grid4x4/fgs_mlp_gat_sac
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_frap_gat_sac
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_mlp_gat_sac
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_frap_gatv2_sac
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_mlp_gatv2_sac
```

RLlib configs default to a local Ray instance. On SLURM, use the remote-server
runbook:

```bash
sbatch scripts/slurm_cologne3_smoke.sh
```

## Startup Chain

The startup path is:

1. `experiments/rllib.py`
2. Hydra composes `configs/rllib.yaml`, one `configs/scenario/*.yaml`, and
   `configs/algorithm/fgs.yaml` or a scenario-first preset.
3. `experiments/rllib.py` checks `SUMO_HOME` and calls
   `sumo_rl.experiments.rllib_runner.train_rllib(cfg)`.
4. `train_rllib()` validates `cfg.algorithm.kind == "fgs"`, creates the Hydra
   run directory, initializes W&B and the local CSV logger, and starts or joins Ray.
5. `rllib_runner._algorithm_module("fgs")` imports `sumo_rl.agents.fgs.fgs`.
6. `fgs.build_config(cfg, run_dir)` dispatches to `build_sac_config()`.
7. `build_sac_config()` normalizes the model config, enforces FGS v1
   constraints, builds a sample graph env to discover spaces, registers the env
   with RLlib, installs the custom RLModule and learner, and returns `SACConfig`.
8. `train_rllib()` builds the RLlib algorithm from that config.
9. `fgs.train()` delegates to the shared SAC training loop.
10. The runner performs periodic validation, final validation, checkpointing,
    W&B summary updates, CSV logging, and Ray shutdown.

In code, the important files are:

```text
experiments/rllib.py
sumo_rl/experiments/rllib_runner.py
sumo_rl/agents/fgs/fgs.py
sumo_rl/agents/fgs/graph_env.py
sumo_rl/agents/fgs/topology.py
sumo_rl/agents/fgs/model.py
sumo_rl/agents/fgs/rllib_module.py
sumo_rl/agents/fgs/learner.py
configs/algorithm/fgs.yaml
configs/presets/resco_grid4x4/fgs_frap_gat_sac.yaml
configs/presets/resco_grid4x4/fgs_mlp_gat_sac.yaml
configs/presets/resco_cologne8/fgs_*_sac.yaml
```

## Config Composition

The default FGS config is `configs/algorithm/fgs.yaml`.

The important defaults are:

- `algorithm.kind: fgs`
- `algorithm.params.policy_mode: shared`
- `algorithm.params.n_step: 1`
- `algorithm.params.twin_q: true`
- `algorithm.params.replay_buffer_type: MultiAgentPrioritizedEpisodeReplayBuffer`
- `algorithm.params.model_config.local_encoder.type: frap`
- `algorithm.params.model_config.communication.type: gat`
- `algorithm.params.model_config.critic.type: central_graph_joint_action`
- `algorithm.params.model_config.topology.source: tls_super_edges`

FGS v1 forces `policy_mode=shared` because the actor is decentralized by ego
embedding but uses one shared parameter set across all traffic signals. It also
requires `n_step=1` for the default `central_graph_joint_action` critic because
the critic consumes the same-transition joint-action context stored in the graph
observation.

Presets override only the recipe-specific pieces. For example,
`configs/presets/resco_cologne8/fgs_mlp_gat_sac.yaml` selects
`scenario=resco_cologne8`, keeps `algorithm=fgs`, and changes the local encoder
from FRAP to an MLP while leaving the startup, graph wrapper, SAC learner, and
metric path unchanged.

## Environment Pipeline

FGS wraps the normal SUMO-RL PettingZoo parallel environment.

`fgs.build_fgs_parallel_env()`:

1. Calls `_prepare_env_kwargs(cfg, run_dir)` to resolve SUMO files, output paths,
   tripinfo paths, and env kwargs.
2. Sets `num_seconds` from `experiment.episode_seconds` when needed.
3. Sets `sumo_seed` for training or validation envs.
4. Builds a SUMO-RL parallel env from `cfg.env.factory`.
5. Wraps it with `FGSGraphParallelEnv`.

`FGSGraphParallelEnv` converts per-agent local observations into full-graph
observations for every traffic signal. Each agent still receives its own
observation dict, but the dict contains the full traffic-light graph plus an
`ego_index` telling the shared policy which node is acting.

The graph observation fields are:

- `node_features`: padded canonical features for all traffic signals.
- `edge_index`: message-passing edges as source and target node indices.
- `edge_mask`: marks real edges inside the padded edge tensor.
- `edge_weight`: topology-derived edge weights.
- `ego_index`: the node index for the current acting agent.
- `action_mask`: valid phase actions for the ego traffic signal.
- `node_action_mask`: valid phase actions for every node.
- `phase_pair_mask`: phase-to-movement mask used by the FRAP encoder.
- `phase_competition_mask`: phase competition mask used by the FRAP encoder.
- `prev_joint_action`: one-hot joint action from the previous environment step.

The wrapper canonicalizes heterogeneous default SUMO-RL observations into:

```text
[phase_one_hot padded to max_actions, min_green, density, queue]
```

It also clips outgoing actions to each traffic signal's real action-space size
before stepping SUMO.

## Topology Pipeline

FGS v1 defaults to `model_config.topology.source=tls_super_edges`.

`sumo_rl/agents/fgs/topology.py` builds the topology from the SUMO `.net.xml`:

1. Parse junction positions and traffic-light IDs.
2. Build a catalog of non-internal road edges.
3. Build legal road-edge transitions from SUMO `<connection>` elements.
4. For each traffic light and each outgoing edge, search downstream legal paths
   until the nearest traffic light is found.
5. Contract those road paths into directed TLS super-edges.
6. Convert directed super-edges into bidirectional message-passing edges.
7. Weight each undirected edge by inverse travel time.

When rendering is enabled, each run writes:

```text
outputs/<experiment-name>/<timestamp>/topology/fgs_topology.svg
outputs/<experiment-name>/<timestamp>/topology/fgs_topology_edges.json
```

Those files are the audit trail for checking whether FGS communicated over the
expected traffic-light graph.

`model_config.topology.source=direct_lane` skips the super-edge parser and falls
back to direct lane overlap between a source signal's outgoing lanes and a
target signal's incoming lanes.

## Model Pipeline

The SAC RLModule is built in `sumo_rl/agents/fgs/rllib_module.py`.

For each shared-policy batch:

1. `FGSGraphEncoder` reads `node_features`, masks, edges, and `ego_index`.
2. The local encoder maps each node's canonical local observation to an
   embedding.
3. Optional graph communication updates node embeddings with GAT or GATv2.
4. The actor head produces logits for every node.
5. The module selects the ego node logits with `ego_index`.
6. Invalid padded actions are masked with `invalid_action_value`.
7. The categorical discrete SAC policy samples or selects one phase action.

The default local encoder is FRAP:

- It treats the observation tail as movement demand.
- It uses phase-pair masks to map phases to movements.
- It builds ordered phase competitions.
- It pools valid phase competitions into one local embedding.

The MLP ablation uses the same graph wrapper and action masks, but replaces the
FRAP local encoder with a feed-forward MLP.

The communication block is controlled by
`algorithm.params.model_config.communication.type`:

- `gat`: project CoLight-style PyG `MessagePassing` attention.
- `gatv2`: PyTorch Geometric `GATv2Conv` ablation.
- `identity`: no inter-node communication.

## Critic And Learner Pipeline

FGS v1 keeps execution decentralized but trains with centralized graph context.

The default critic is `central_graph_joint_action`:

1. The encoder returns all graph node embeddings.
2. The critic flattens the whole graph embedding.
3. It receives a joint-action context for all nodes.
4. It receives the ego embedding and an ego one-hot node ID.
5. It evaluates candidate ego actions and returns Q-values for the ego action
   space.

The replay TD loss uses `prev_joint_action` from the next graph observation when
available. This is the same-transition joint action that was actually applied
in SUMO. Actor and target-value paths instead use the current policy's
all-node action probabilities as a tractable policy context.

`FGSSACTorchLearner` extends RLlib's discrete SAC learner so that:

- critic TD loss uses replay joint-action context;
- actor loss uses the actor-specific Q outputs built with current policy
  probabilities;
- twin-Q remains enabled by default;
- the learner logs standard SAC losses plus `fgs_actor_q_mean`.

This split is the main v1 CTDE detail: the critic learns from replayed joint
actions, while the actor is optimized against the critic under current policy
context without enumerating the full joint action space.

## Training And Validation Loop

After RLlib builds the FGS algorithm, `fgs.train()` delegates to the shared SAC
training loop. The outer runner still owns validation and outputs.

During training:

- training length is controlled by `experiment.episodes`;
- episode horizon is controlled by `experiment.episode_seconds`;
- RLlib sampling uses the registered FGS graph env;
- training metrics are emitted through the algorithm module;
- W&B and local CSV logging are handled by `rllib_runner.train_rllib()`.

During validation:

1. `rllib_runner` calls `fgs.build_eval_env()`.
2. The eval env is another `FGSGraphParallelEnv`, seeded with the validation
   seed.
3. `_compute_single_action()` uses the RLModule inference path when available.
4. The runner records action traces, phase queue traces, tripinfo distributions,
   RESCO metrics, efficiency metrics, and safety metrics.
5. The final validation row updates W&B summary and the local CSV.

Validation cadence is episode-based by default through
`experiment.validation_interval_episodes`. The final validation pass is always
run unless the most recent validation already matches the final training step
and episode index.

## Output Artifacts

A normal FGS run writes under the Hydra run directory:

```text
outputs/<experiment-name>/<timestamp>/
  .hydra/
  csv/
  checkpoints/fgs/
  topology/fgs_topology.svg
  topology/fgs_topology_edges.json
  tripinfo/                  # only when configured to keep raw tripinfo
```

The important metric sinks are:

- W&B history and summary, if logging is enabled.
- The local CSV logger under `csv/`.
- RESCO, efficiency, and safety validation rows from `rllib_runner`.
- Optional best-validation checkpoints under `checkpoints/fgs/best_validation/`
  when that logging option is enabled.

## Minimal Verification

Use these commands after changing FGS v1 wiring:

```bash
source .venv/bin/activate
python -m pytest tests/models/test_fgs.py
python -m pytest tests/core/test_rllib_config_defaults.py
python experiments/rllib.py \
  algorithm=fgs \
  scenario=single_intersection \
  experiment.episodes=1 \
  experiment.episode_seconds=60 \
  logging=disabled \
  resources.ray_address=null
```

Use the larger regression bundle when changes touch shared RLlib SAC, FRAP,
CoLight-style attention, or runner metrics:

```bash
source .venv/bin/activate
python -m pytest \
  tests/models/test_fgs.py \
  tests/models/test_sac_build_config.py \
  tests/models/test_sac_model_config.py \
  tests/models/test_frap.py \
  tests/models/test_colight.py \
  tests/runner/test_train_rllib.py
```

## Common Failure Points

If FGS does not start, check these in order:

1. `SUMO_HOME` is set before running `experiments/rllib.py`.
2. The selected scenario has valid `net_file` and `route_file` paths.
3. `resources.ray_address` matches the intended mode: `auto` or an existing
   cluster for shared runs, `null` for local-only debugging.
4. `algorithm.params.policy_mode` is still `shared`.
5. `algorithm.params.n_step` is still `1` for the default joint-action critic.
6. The graph topology artifacts contain the expected traffic-light IDs and
   edges.
7. The observation layout is still compatible with
   `[phase_one_hot, min_green, density, queue]`.
8. Action masks are valid for heterogeneous action spaces.
9. On SLURM GPU runs, keep `resources.cuda_visible_devices=null` and
   `algorithm.params.local_gpu_idx=0`.
