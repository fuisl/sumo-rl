<img src="docs/_static/logo.png" align="right" width="30%"/>

[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/fuisl/sumo-rl/actions/workflows/core-fast-tests.yml/badge.svg)](https://github.com/fuisl/sumo-rl/actions/workflows/core-fast-tests.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
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

### Development hooks

Install the repo hooks once after cloning:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

The hook split is intentionally lightweight:

- `pre-commit` runs fast quality checks with Ruff, codespell, pyright, and file hygiene hooks.
- `pre-push` runs local heavy validation for real SUMO-backed smoke tests on the developer machine.

Manual commands:

```bash
pre-commit run --all-files
pytest -m core_fast tests
pre-commit run --hook-stage pre-push local-heavy-validation
```

Test taxonomy:

- `core_fast`: required CI coverage for lightweight pure-Python and mocked wiring tests
- `local_heavy`: developer-machine SUMO smoke checks run from `pre-push`
- `research_heavy`: broader RLlib and model-variant coverage kept out of required CI

Run the heavy local validation before opening a PR whenever you change SUMO environment code, runtime wiring, or RLlib integration paths. Use `--no-verify` only for exceptional cases where you knowingly need to bypass the local heavy hook.

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

### Thesis workflow

The thesis workflow is intentionally documented outside the top-level README so
there is one canonical place to maintain commands and support status.

Doc map:

- installation and optional thesis extras: [docs/thesis/experiments.md](docs/thesis/experiments.md)
- experiment launch, validation, resume, rollout export, and W&B: [docs/thesis/experiments.md](docs/thesis/experiments.md)
- fixed-time/manual control reference: [docs/thesis/manual_control.md](docs/thesis/manual_control.md)
- static baseline reference: [docs/thesis/static_baselines.md](docs/thesis/static_baselines.md)
- contributor architecture and debugging notes: [docs/thesis/engineering_guide.md](docs/thesis/engineering_guide.md)
- deeper reference material for FGS, DCRNN resource smoke, diagrams, and pseudocode: `docs/thesis/`

Supported-surface note:

- The canonical thesis support matrix lives in [docs/thesis/experiments.md](docs/thesis/experiments.md).
- Treat `fixed_time`, `static_max_pressure`, `ppo`, `dqn`, `dqn_dcrnn`, `frap`, `colight`, `fgs`, `fgs_ppo`, `sac_builtin`, `sac_mlp`, and `sac_dcrnn_full` as the supported thesis-facing methods.
- Treat DCRNN/FGS/SAC ablation variants such as `ppo_dcrnn_mlp`, `dqn_dcrnn_mlp`, `fgsv2`, `sac_dcrnn_actor`, and `sac_dcrnn_shared_mlp` as experimental unless that matrix explicitly promotes them later.
- Treat `dcrnn` as an alias for `dqn_dcrnn` and `sac_custom` as an alias for `sac_mlp`.

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
