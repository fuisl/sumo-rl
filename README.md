<img src="docs/_static/logo.png" align="right" width="30%"/>

[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml/badge.svg)](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/sumo-rl/blob/main/LICENSE)

# SUMO-RL

<!-- start intro -->

SUMO-RL provides a simple interface to instantiate Reinforcement Learning (RL) environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Support Multiagent RL
- Compatibility with `gymnasium.Env`, PettingZoo, and RLlib-based learning stacks
- Easy customisation: state and reward definitions are easily modifiable

The main class is [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py).
If instantiated with parameter 'single-agent=True', it behaves like a regular [Gymnasium Env](https://github.com/Farama-Foundation/Gymnasium).
For multiagent environments, use [env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) or [parallel_env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) to instantiate a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) environment with AEC or Parallel API, respectively.
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

For more details, check the [documentation online](https://lucasalegre.github.io/sumo-rl/).

<!-- end intro -->

## Install

<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Important: for the thesis RLlib experiments, backend selection is controlled through Hydra config instead of the global `LIBSUMO_AS_TRACI` environment variable. The PPO, DQN, and selected SAC algorithm configs enable Libsumo for training with `env.kwargs.use_libsumo=true`, while validation stays on TraCI by default through `logging.eval_use_libsumo=false`.

### Install SUMO-RL

Stable release version is available through pip
```bash
pip install sumo-rl
```

Alternatively, you can install using the latest (unreleased) version
```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```

For the thesis RLlib experiments, install the extra dependencies as needed:

```bash
pip install -e ".[experiments]"
pip install -e ".[rllib]"
pip install -e ".[rllib-custom]"
```

<!-- end install -->

## MDP - Observations, Actions and Rewards

### Observation

<!-- start observation -->

The default observation for each traffic signal agent is a vector:
```python
    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
```
- ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
- ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
- ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
- ```lane_i_queue```is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

You can define your own observation by implementing a class that inherits from [ObservationFunction](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py) and passing it to the environment constructor.

<!-- end observation -->

### Action

<!-- start action -->

The action space is discrete.
Every 'delta_time' seconds, each traffic signal agent can choose the next green phase configuration.

E.g.: In the 2-way single intersection there are |A| = 4 discrete actions, corresponding to the following green phase configurations:

<p align="center">
<img src="docs/_static/actions.png" align="center" width="75%"/>
</p>

Important: every time a phase change occurs, the next phase is preeceded by a yellow phase lasting ```yellow_time``` seconds.

<!-- end action -->

### Rewards

<!-- start reward -->

The default reward function is the change in cumulative vehicle delay:

<p align="center">
<img src="docs/_static/reward.png" align="center" width="25%"/>
</p>

That is, the reward is how much the total delay (sum of the waiting times of all approaching vehicles) changed in relation to the previous time-step.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py)) with the parameter `reward_fn` in the [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) constructor.

It is also possible to implement your own reward function:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

<!-- end reward -->

## API's (Gymnasium and PettingZoo)

### Gymnasium Single-Agent API

<!-- start gymnasium -->

If your network only has ONE traffic light, then you can instantiate a standard Gymnasium env (see [Gymnasium API](https://gymnasium.farama.org/api/env/)):
```python
import gymnasium as gym
import sumo_rl
env = gym.make('sumo-rl-v0',
                net_file='path_to_your_network.net.xml',
                route_file='path_to_your_routefile.rou.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

<!-- end gymnasium -->

### PettingZoo Multi-Agent API

<!-- start pettingzoo -->

For multi-agent environments, you can use the PettingZoo API (see [Petting Zoo API](https://pettingzoo.farama.org/api/parallel/)):

```python
import sumo_rl
env = sumo_rl.parallel_env(net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

<!-- end pettingzoo -->

### RESCO Benchmarks

In the folder [nets/RESCO](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets/RESCO) you can find the network and route files from [RESCO](https://github.com/jault/RESCO) (Reinforcement Learning Benchmarks for Traffic Signal Control), which was built on top of SUMO-RL. See their [paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf) for results.

<p align="center">
<img src="sumo_rl/nets/RESCO/maps.png" align="center" width="60%"/>
</p>

### Experiments

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/main/experiments) for examples on how to instantiate an environment and train your RL agent. In the thesis configs, the 4x4 grid presets use the RESCO `grid4x4` assets rather than the older Lucas `4x4-Lucas` network. Thesis-specific Hydra and W&B notes are documented separately in [docs/thesis/experiments.md](docs/thesis/experiments.md).

If you want to push runs to Weights & Biases, put the credentials/config in a local `.env` at the repo root, for example:

```bash
WANDB_API_KEY=...
WANDB_PROJECT=sumo-rl
WANDB_ENTITY=your-entity
```

### Fixed-time control in a RESCO scenario:
```bash
python experiments/fixed_time.py scenario=resco_grid4x4
python experiments/fixed_time.py scenario=resco_cologne1
python experiments/fixed_time.py -m scenario=resco_cologne1,resco_cologne3,resco_cologne8,resco_ingolstadt1,resco_ingolstadt7,resco_ingolstadt21
```

### Max-pressure control in a RESCO scenario:
```bash
python experiments/static_max_pressure.py scenario=resco_cologne1
python experiments/static_max_pressure.py scenario=resco_ingolstadt7
python experiments/static_max_pressure.py -m scenario=resco_cologne1,resco_cologne3,resco_cologne8,resco_ingolstadt1,resco_ingolstadt7,resco_ingolstadt21
```

### RLlib PPO, DQN, FRAP, DQN+DCRNN, CoLight, and SAC:
```bash
python experiments/rllib.py algorithm=ppo scenario=resco_grid4x4
python experiments/rllib.py algorithm=ppo_dcrnn_mlp scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=ppo_dcrnn_shared_mlp scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=dqn scenario=resco_cologne1
python experiments/rllib.py algorithm=frap scenario=resco_grid4x4
python experiments/rllib.py algorithm=dqn_dcrnn scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=dqn_dcrnn_mlp scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=colight scenario=resco_grid4x4
python experiments/rllib.py algorithm=sac_builtin scenario=resco_ingolstadt1
python experiments/rllib.py algorithm=sac_mlp scenario=resco_ingolstadt7
python experiments/rllib.py algorithm=sac_dcrnn_actor scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=sac_dcrnn_actor_mlp scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=sac_dcrnn_full scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=sac_dcrnn_full_mlp scenario=resco_grid4x4 experiment.episodes=1
python experiments/rllib.py algorithm=sac_dcrnn_shared_mlp scenario=resco_grid4x4 experiment.episodes=1
```

Scenario-first RLlib presets are launched by keeping the config root at
`configs/` and passing the preset path as the config name:

```bash
python experiments/rllib.py --config-name presets/resco_cologne8/fgs_mlp_gat_sac
python experiments/rllib.py --config-name presets/resco_cologne8/sac_builtin
```

To manually restore a saved RLlib checkpoint and run the repo's current
evaluation helper from a notebook, open
`experiments/manual_checkpoint_evaluation.ipynb` and set `RUN_DIR` plus
`CHECKPOINT_PATH`.

For thesis-style validation across RLlib checkpoints and static baselines,
use the unified validation CLI:

```bash
python experiments/validate_methods.py --controller rllib --run-dir outputs/rllib/2026-06-21_12-00-00 --checkpoint-selector best --seeds 1 2 3
python experiments/validate_methods.py --controller fixed_time --scenario resco_grid4x4 --seeds 1 2 3
python experiments/validate_methods.py --controller static_max_pressure --scenario resco_grid4x4 --seeds 1 2 3
```

The CLI writes a compact terminal table, per-seed CSV/JSON artifacts, and
validation plots under a dedicated output directory.

To export an MP4 rollout from a trained RLlib checkpoint, use:

```bash
python experiments/record_rollout.py --controller rllib --run-dir outputs/rllib/2026-06-21_12-00-00 --checkpoint outputs/rllib/2026-06-21_12-00-00/checkpoints/ppo/checkpoint_000001 --output outputs/rllib/2026-06-21_12-00-00/videos/rollout.mp4
```

The recorder restores the checkpoint, runs one evaluation rollout with
`render_mode=rgb_array`, and writes an MP4 file. Use `--frame-skip` to reduce
video size or `--max-steps` for a short smoke recording.
The MP4 writer needs either OpenCV or `imageio` plus `imageio-ffmpeg`
installed; the `.[experiments]`, `.[rendering]`, and `.[all]` extras now
include the `imageio` path.

For static baselines, use the same recorder with a different controller:

```bash
python experiments/record_rollout.py --controller fixed_time --scenario resco_grid4x4 --output outputs/recordings/fixed_time.mp4
python experiments/record_rollout.py --controller static_max_pressure --scenario resco_grid4x4 --output outputs/recordings/max_pressure.mp4
```

You can pass extra Hydra overrides for static controllers with repeated
`--override` flags, for example `--override env.kwargs.num_seconds=600`.

RLlib runs default to `resources.ray_address=null`, so experiments start a
local Ray instance unless you opt into cluster discovery. To share one Ray
scheduler across multiple jobs, start a shared Ray head first, for example
`CUDA_VISIBLE_DEVICES=1 ray start --head --num-cpus=8 --num-gpus=1`, then
launch one or more jobs with `resources.ray_address=auto` or an explicit head
address. In cluster mode the head's resources come from `ray start`, not from
`resources.ray_num_cpus` or `algorithm.params.ray_num_gpus`. Local runs default
to a small CPU budget where `resources.ray_num_cpus=2` advertises
two logical CPUs to Ray and `resources.native_num_threads=1` caps
OpenMP/BLAS/Torch-style thread pools.
The shared config keeps sampling in the local process by default
(`algorithm.params.num_env_runners=0`) and uses one learner actor so `ray status`
still reports reserved learner resources. The default learner reservation is
fractional (`algorithm.params.num_gpus_per_learner=0.1`) so several runs can
share the selected GPU when memory headroom is available; override it to `1`
for exclusive GPU use.
GPU selection should be pinned with `resources.cuda_visible_devices`; for example
`resources.cuda_visible_devices=1` exposes physical GPU 1 as local CUDA index 0,
so keep `algorithm.params.local_gpu_idx=0`.
For RLlib W&B titles, set `logging.name` for an explicit display name, or set a
non-default `experiment.name`; the default `experiment.name=rllib` keeps the
generated `scenario__algorithm__time` title.

`ppo_dcrnn_mlp` uses the same graph-history wrapper as the DCRNN DQN/SAC
variants. The wrapper builds one directed traffic-signal graph from
incoming/outgoing lane connectivity and returns rolling
`[history_len, num_nodes, phase_one_hot_min_green_density_queue_features]`
tensors. Each PPO
policy sees the full graph history, performs diffusion inside the DCRNN, and
then acts from the ego node latent through separate policy and value heads.
This graph PPO variant keeps decentralized policies with centralized graph
observations and does not add a separate GAT layer.
The PPO+DCRNN config now uses `sgd_minibatch_size=64` to reduce learner
activation memory without changing the backbone or rollout horizon.

`ppo_dcrnn_shared_mlp` keeps the same graph-history wrapper and independent
per-agent policy IDs, but lifts the DCRNN+MLP encoder into one shared
multi-module backbone. Each traffic signal still keeps its own PPO actor and
value heads, and one optimizer updates the shared encoder plus all heads
together.

FRAP is implemented as a DQN-family RLlib module with the phase-competition
Q-network from Zheng et al. and the LibSignal FRAP implementation. By default it
uses the SUMO-RL observation tail as per-movement demand features
`[density_i, queue_i]` from the default split observation layout; override
`algorithm.params.model_config.phase_pairs` when a network needs custom
movement-pair ordering.

DQN+DCRNN is implemented as a DQN-family RLlib module with a graph-observation
wrapper. It builds a traffic-signal graph from incoming/outgoing lanes and feeds
rolling full traffic-light state histories
`[phase_one_hot, min_green, density, queue]` to a diffusion-convolutional
recurrent Q-network. The graph can include virtual source/sink nodes plus
self-loops, and neighbor influence is carried through the DCRNN diffusion
operator rather than through attention. The Q-head uses the ego node latent plus
the ego node's most recent features. Use `algorithm=dqn_dcrnn` as the canonical name;
`algorithm=dcrnn` is kept as a backward-compatible alias. The first version
keeps decentralized policies with centralized graph observations.

The DQN+DCRNN defaults are intentionally lighter than the plain DQN defaults on
large RESCO maps: they now use `history_len=3`, `train_batch_size_per_learner=8`,
which helps because the full graph-history observation is duplicated once per
controlled traffic signal. On a 21-signal map such as `resco_ingolstadt21`,
episode replay can otherwise drive very large GPU allocations during learner
batches.

`dqn_dcrnn_mlp` keeps the same graph-history wrapper, but inserts one node-wise
MLP layer before the DCRNN stack.

CoLight is available as `algorithm=colight`. It is a DQN-family RLlib method
with one shared graph-attention Q-network over all controlled intersections.
The wrapper gives each traffic signal the full node-feature graph plus an ego
index and action mask, so the shared policy remains faithful to the CoLight
paper's network-level cooperation rather than independent per-agent DQN.
Each CoLight run writes `topology/colight_topology.svg` and
`topology/colight_topology_edges.json` under the Hydra output directory; set
`algorithm.params.render_topology=false` to skip this artifact.

FGS Cologne8 presets include both the original CoLight-style custom GAT
communication and PyTorch Geometric `GATv2Conv` ablations:
`configs/presets/resco_cologne8/fgs_frap_gatv2_sac.yaml` and
`configs/presets/resco_cologne8/fgs_mlp_gatv2_sac.yaml`.

SAC now uses RLlib's native discrete-action support for the traffic-light
policies in this repo, so it does not depend on a custom joint continuous-action
adapter anymore.
Use `algorithm=sac_builtin` as the reference RLlib baseline. Use
`algorithm=sac_mlp` when you want to expose and modify the SAC RLModule
architecture through `algorithm.params.model_config`, including actor, twin
critic, and future message-passing/GAT hook settings. `algorithm=sac_custom`
is kept as a backward-compatible alias so older launch commands still work.
Important: the generic SAC communication hook is currently a placeholder
identity block, so configuring `communication.type=gat` there does not yet give
you real graph attention.
The default SAC config sets `algorithm.params.training_intensity=1.0` and
`algorithm.params.train_batch_size_per_learner=64` because RLlib's natural
off-policy replay ratio can make Cologne-sized multi-agent discrete SAC spend
most of its wall time in learner/replay updates. Keep `twin_q` enabled for
`sac_builtin`; RLlib's discrete SAC learner expects twin-Q outputs on this path.
Use `algorithm=sac_dcrnn_actor` when you want the graph-history DCRNN encoder
on the SAC actor while keeping the SAC critics on the current MLP path. The
actor sees the same graph-history observation as DQN+DCRNN and performs
diffusion message passing before reducing to the ego node latent. The critics do
not return to the original local flat observation; they still consume the graph
history, then flatten it into the MLP SAC critic path. Treat this as an
experimental ablation. Use
`algorithm=sac_dcrnn_full` when you want separate DCRNN encoders on the SAC
actor, `qf`, and `qf_twin` branches while keeping the standard SAC target-copy
behavior for the critic targets. This is the canonical thesis DCRNN SAC path.
`algorithm=sac_dcrnn_actor_mlp` and `algorithm=sac_dcrnn_full_mlp` keep those
same branch layouts, but add one node-wise MLP layer before each enabled DCRNN
encoder. Use `algorithm=sac_dcrnn_shared_mlp` when you want one shared
DCRNN+MLP backbone across the SAC actor, `qf`, and `qf_twin` while keeping
separate actor and critic heads. Treat this as an experimental parameter-sharing
ablation. Across all of these SAC DCRNN variants, graph communication currently
means DCRNN diffusion over graph-history observations, not GAT.
The SAC+DCRNN configs now use `train_batch_size_per_learner=32` as a lighter
default than the base SAC batch size, which reduces learner and replay memory
pressure without changing the architecture.

If you want the concrete SAC-side reference for graph attention and explicit
message passing, use `algorithm=fgs`. FGS applies a local FRAP or MLP encoder to
per-node default SUMO-RL observations `[phase_one_hot, min_green, density,
queue]`, then runs a CoLight-style GAT or `GATv2Conv` over the traffic-signal
graph before the discrete SAC actor and centralized critics.

### Proof that SAC supports `Discrete` by default:
```bash
python proofs/rllib_sac_discrete/sac_discrete_proof.py --iterations 10
```

## Citing

<!-- start citation -->

If you use this repository in your research, please cite:
```bibtex
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```

<!-- end citation -->

<!-- start list of publications -->

List of publications that use SUMO-RL (please open a pull request to add missing entries):
- [Quantifying the impact of non-stationarity in reinforcement learning-based traffic signal control (Alegre et al., 2021)](https://peerj.com/articles/cs-575/)
- [Information-Theoretic State Space Model for Multi-View Reinforcement Learning (Hwang et al., 2023)](https://openreview.net/forum?id=jwy77xkyPt)
- [A citywide TD-learning based intelligent traffic signal control for autonomous vehicles: Performance evaluation using SUMO (Reza et al., 2023)](https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.13301)
- [Handling uncertainty in self-adaptive systems: an ontology-based reinforcement learning model (Ghanadbashi et al., 2023)](https://link.springer.com/article/10.1007/s40860-022-00198-x)
- [Multiagent Reinforcement Learning for Traffic Signal Control: a k-Nearest Neighbors Based Approach (Almeida et al., 2022)](https://ceur-ws.org/Vol-3173/3.pdf)
- [From Local to Global: A Curriculum Learning Approach for Reinforcement Learning-based Traffic Signal Control (Zheng et al., 2022)](https://ieeexplore.ieee.org/abstract/document/9832372)
- [Poster: Reliable On-Ramp Merging via Multimodal Reinforcement Learning (Bagwe et al., 2022)](https://ieeexplore.ieee.org/abstract/document/9996639)
- [Using ontology to guide reinforcement learning agents in unseen situations (Ghanadbashi & Golpayegani, 2022)](https://link.springer.com/article/10.1007/s10489-021-02449-5)
- [Information upwards, recommendation downwards: reinforcement learning with hierarchy for traffic signal control (Antes et al., 2022)](https://www.sciencedirect.com/science/article/pii/S1877050922004185)
- [A Comparative Study of Algorithms for Intelligent Traffic Signal Control (Chaudhuri et al., 2022)](https://link.springer.com/chapter/10.1007/978-981-16-7996-4_19)
- [An Ontology-Based Intelligent Traffic Signal Control Model (Ghanadbashi & Golpayegani, 2021)](https://ieeexplore.ieee.org/abstract/document/9564962)
- [Reinforcement Learning Benchmarks for Traffic Signal Control (Ault & Sharon, 2021)](https://openreview.net/forum?id=LqRSh6V0vR)
- [EcoLight: Reward Shaping in Deep Reinforcement Learning for Ergonomic Traffic Signal Control (Agand et al., 2021)](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2021/43/paper.pdf)

<!-- end list of publications -->
