from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides or [])


def _assert_shared_ray_defaults(cfg):
    assert cfg.resources.ray_address == "auto"
    assert cfg.algorithm.params.ray_num_gpus == "auto"
    assert cfg.algorithm.params.num_env_runners == 1
    assert cfg.algorithm.params.num_envs_per_env_runner == 1
    assert cfg.algorithm.params.num_cpus_per_env_runner == 1
    assert cfg.algorithm.params.num_gpus_per_env_runner == 0
    assert cfg.algorithm.params.num_learners == 1
    assert cfg.algorithm.params.num_cpus_per_learner == 1
    assert cfg.algorithm.params.num_gpus_per_learner == 0.25


def test_rllib_direct_algorithm_uses_shared_ray_defaults():
    cfg = _compose("rllib", ["algorithm=ppo", "scenario=cologne3"])

    assert cfg.algorithm.kind == "ppo"
    assert cfg.scenario.name == "resco_cologne3"
    _assert_shared_ray_defaults(cfg)


def test_rllib_preset_uses_shared_ray_defaults_after_algorithm_override():
    cfg = _compose("presets/resco_cologne8/fgs_mlp_gat_sac")

    assert cfg.algorithm.kind == "fgs"
    assert cfg.scenario.name == "resco_cologne8"
    _assert_shared_ray_defaults(cfg)
