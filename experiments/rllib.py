import os
import sys

import sumo_rl
import ray
import sys

print("DEBUG sumo_rl:", sumo_rl.__file__)
print("DEBUG ray:", ray.__version__)
print("DEBUG python:", sys.executable)


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import hydra
from omegaconf import DictConfig

from sumo_rl.experiments.rllib_runner import train_rllib


@hydra.main(version_base=None, config_path="../configs", config_name="rllib")
def main(cfg: DictConfig) -> None:
    train_rllib(cfg)


if __name__ == "__main__":
    main()
