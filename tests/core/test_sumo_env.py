# ruff: noqa: E402

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.environment.env import SumoEnvironment

pytestmark = pytest.mark.core_fast


def _stub_env(**overrides):
    env = SumoEnvironment.__new__(SumoEnvironment)
    defaults = {
        "_sumo_binary": "sumo",
        "_net": "network.net.xml",
        "_route": "routes.rou.xml",
        "max_depart_delay": -1,
        "waiting_time_memory": 1000,
        "time_to_teleport": -1,
        "begin_time": 25200,
        "sim_max_time": 28800,
        "sumo_seed": 7,
        "sumo_warnings": True,
        "additional_sumo_cmd": None,
        "tripinfo_output_name": None,
        "statistic_output_name": None,
        "label": "0",
        "episode": 1,
    }
    for key, value in defaults.items():
        setattr(env, key, value)
    for key, value in overrides.items():
        setattr(env, key, value)
    return env


def test_build_sumo_cmd_includes_begin_and_end_times():
    env = _stub_env(tripinfo_output_name=os.path.join("outputs", "tripinfo", "fixed_time"))

    sumo_cmd = env._build_sumo_cmd()

    assert "-b" in sumo_cmd
    assert sumo_cmd[sumo_cmd.index("-b") + 1] == "25200"
    assert "--end" in sumo_cmd
    assert sumo_cmd[sumo_cmd.index("--end") + 1] == "28800"
    assert "--tripinfo-output" in sumo_cmd


def test_build_sumo_cmd_preserves_explicit_end_override():
    env = _stub_env(additional_sumo_cmd="--end 29999")

    sumo_cmd = env._build_sumo_cmd()

    assert sumo_cmd.count("--end") == 1
    assert sumo_cmd[sumo_cmd.index("--end") + 1] == "29999"
