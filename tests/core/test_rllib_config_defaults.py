from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


pytestmark = pytest.mark.core_fast


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides or [])


def _assert_rllib_ray_defaults(cfg):
    assert cfg.resources.ray_address is None
    assert cfg.algorithm.params.ray_num_gpus == "auto"
    assert cfg.algorithm.params.num_env_runners == 0
    assert cfg.algorithm.params.num_envs_per_env_runner == 1
    assert cfg.algorithm.params.num_cpus_per_env_runner == 1
    assert cfg.algorithm.params.num_gpus_per_env_runner == 0
    assert cfg.algorithm.params.num_learners == 1
    assert cfg.algorithm.params.num_cpus_per_learner == 1
    assert cfg.algorithm.params.num_gpus_per_learner == 0.1


def _assert_shared_ray_defaults(cfg):
    _assert_rllib_ray_defaults(cfg)


def test_rllib_direct_algorithm_uses_local_ray_defaults():
    cfg = _compose("rllib", ["algorithm=ppo", "scenario=cologne3"])

    assert cfg.algorithm.kind == "ppo"
    assert cfg.scenario.name == "resco_cologne3"
    assert cfg.env.kwargs.use_libsumo is True
    assert cfg.logging.eval_use_libsumo is False
    assert cfg.logging.resume_from_checkpoint is None
    assert cfg.logging.checkpoint_every_episodes == 50
    assert cfg.logging.save_periodic_checkpoints is True
    _assert_rllib_ray_defaults(cfg)


def test_rllib_base_config_stays_backend_neutral_for_other_algorithms():
    cfg = _compose("rllib", ["algorithm=colight", "scenario=resco_grid4x4"])

    assert cfg.algorithm.kind == "colight"
    assert "use_libsumo" not in cfg.env.kwargs


def test_rllib_preset_uses_shared_ray_defaults_after_algorithm_override():
    cfg = _compose("presets/resco_cologne8/fgs_mlp_gat_sac")

    assert cfg.algorithm.kind == "fgs"
    assert cfg.scenario.name == "resco_cologne8"
    _assert_rllib_ray_defaults(cfg)


def test_fgs_ppo_algorithm_uses_shared_ray_defaults():
    cfg = _compose("rllib", ["algorithm=fgs_ppo", "scenario=resco_cologne8"])

    assert cfg.algorithm.kind == "fgs_ppo"
    assert cfg.algorithm.params.policy_mode == "shared"
    assert cfg.algorithm.params.model_config.architecture_tag == "fgs_frap_gnn_ppo"
    _assert_shared_ray_defaults(cfg)


def test_fgs_ablation_presets_compose():
    expected = {
        "presets/resco_cologne8/fgs_frap_gatv2_ppo": ("resco_cologne8", "fgs_ppo", "frap", "gatv2"),
        "presets/resco_cologne8/fgs_mlp_gat_ppo": ("resco_cologne8", "fgs_ppo", "mlp", "gat"),
        "presets/resco_cologne8/fgs_mlp_gatv2_ppo": ("resco_cologne8", "fgs_ppo", "mlp", "gatv2"),
        "presets/resco_cologne8/fgs_mlp_gatv2_sac": ("resco_cologne8", "fgs", "mlp", "gatv2"),
        "presets/resco_ingolstadt21/fgs_frap_gatv2_ppo": ("resco_ingolstadt21", "fgs_ppo", "frap", "gatv2"),
        "presets/resco_ingolstadt21/fgs_mlp_gat_ppo": ("resco_ingolstadt21", "fgs_ppo", "mlp", "gat"),
        "presets/resco_ingolstadt21/fgs_mlp_gatv2_ppo": ("resco_ingolstadt21", "fgs_ppo", "mlp", "gatv2"),
        "presets/resco_ingolstadt21/fgs_mlp_gatv2_sac": ("resco_ingolstadt21", "fgs", "mlp", "gatv2"),
    }

    for config_name, (scenario_name, algorithm_kind, local_type, communication_type) in expected.items():
        cfg = _compose(config_name)

        assert cfg.scenario.name == scenario_name
        assert cfg.algorithm.kind == algorithm_kind
        assert cfg.algorithm.params.model_config.local_encoder.type == local_type
        assert cfg.algorithm.params.model_config.communication.type == communication_type
        _assert_shared_ray_defaults(cfg)


def test_sac_builtin_uses_bounded_replay_training_intensity():
    cfg = _compose("rllib", ["algorithm=sac_builtin", "scenario=resco_cologne8"])

    assert cfg.algorithm.kind == "sac_builtin"
    assert cfg.algorithm.params.train_batch_size_per_learner == 64
    assert cfg.algorithm.params.training_intensity == 1.0
