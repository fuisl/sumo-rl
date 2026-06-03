import os
import sys
from pathlib import Path


_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

for env_var in _CPU_THREAD_ENV_VARS:
    os.environ.setdefault(env_var, "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sumo_rl


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import hydra
from omegaconf import DictConfig

print("DEBUG sumo_rl:", sumo_rl.__file__)
print("DEBUG python:", sys.executable)


@hydra.main(version_base=None, config_path="../configs", config_name="rllib")
def main(cfg: DictConfig) -> None:
    cuda_visible_devices = getattr(getattr(cfg, "resources", None), "cuda_visible_devices", None)
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip().lower() not in {"", "none", "null"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    from sumo_rl.experiments.rllib_runner import train_rllib

    train_rllib(cfg)


if __name__ == "__main__":
    main()
