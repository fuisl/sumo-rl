from __future__ import annotations

from pathlib import Path

import pytest


RESEARCH_HEAVY_FILES = {
    "test_colight.py",
    "test_dcrnn.py",
    "test_dcrnn_memory.py",
    "test_fgs.py",
    "test_fgsv2.py",
    "test_frap.py",
    "test_ppo_dcrnn.py",
    "test_rllib_env_construction.py",
    "test_rllib_env_factory.py",
    "test_rllib_libsumo_validation.py",
    "test_rllib_runner.py",
    "test_rllib_traci_isolation.py",
    "test_sac_dcrnn_inference_attrs.py",
    "test_sac_discrete.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(item.fspath).name in RESEARCH_HEAVY_FILES:
            item.add_marker(pytest.mark.research_heavy)
