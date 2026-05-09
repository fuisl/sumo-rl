import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import hydra
from omegaconf import DictConfig

from sumo_rl.experiments.runner import run


@hydra.main(version_base=None, config_path="../configs", config_name="presets/two_way_single_intersection/dqn")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
