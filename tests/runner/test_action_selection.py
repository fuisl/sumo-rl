# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("ray")

from sumo_rl.experiments import rllib_runner


def test_sync_env_runner_weights_for_evaluation_uses_learner_weights():
    calls = []

    class DummyLearnerGroup:
        def get_weights(self):
            return {"module": {"weight": 123}}

    class DummyEnvRunner:
        def set_weights(self, weights):
            calls.append(weights)

    algo = SimpleNamespace(
        learner_group=DummyLearnerGroup(),
        env_runner=DummyEnvRunner(),
    )

    synced = rllib_runner._sync_env_runner_weights_for_evaluation(algo)

    assert synced is True
    assert calls == [{"module": {"weight": 123}}]


def test_sync_env_runner_weights_for_evaluation_returns_false_without_sync_api():
    algo = SimpleNamespace(
        learner_group=object(),
        env_runner=object(),
    )

    synced = rllib_runner._sync_env_runner_weights_for_evaluation(algo)

    assert synced is False


def test_compute_single_action_prefers_module_forward_over_algo_compute_single_action():
    class DummyColumns:
        ACTIONS = "actions"
        OBS = "obs"

    class DummyTensor:
        def __init__(self, values):
            self._values = values

        def unsqueeze(self, dim):
            del dim
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class DummyTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            del dtype, device
            return DummyTensor(values)

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def no_grad():
            class _NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _NoGrad()

    class DummyRayColumnsModule:
        Columns = DummyColumns

    class DummyModule:
        def __init__(self):
            self.calls = []

        def forward_inference(self, batch):
            self.calls.append(batch)
            return {DummyColumns.ACTIONS: DummyTensor([2])}

    class DummyAlgo:
        def __init__(self):
            self.calls = []
            self.module = DummyModule()

        def compute_single_action(self, obs, policy_id=None, explore=None):
            self.calls.append(
                {
                    "obs": obs,
                    "policy_id": policy_id,
                    "explore": explore,
                }
            )
            return 3

        def get_module(self, policy_id=None):
            self.calls.append({"get_module_policy_id": policy_id})
            return self.module

    algo = DummyAlgo()

    original_torch = sys.modules.get("torch")
    original_ray = sys.modules.get("ray")
    original_ray_rllib = sys.modules.get("ray.rllib")
    original_ray_rllib_core = sys.modules.get("ray.rllib.core")
    original_ray_rllib_core_columns = sys.modules.get("ray.rllib.core.columns")
    try:
        sys.modules["torch"] = DummyTorch
        sys.modules["ray"] = SimpleNamespace()
        sys.modules["ray.rllib"] = SimpleNamespace()
        sys.modules["ray.rllib.core"] = SimpleNamespace()
        sys.modules["ray.rllib.core.columns"] = DummyRayColumnsModule

        action = rllib_runner._compute_single_action(algo, {"graph": [1, 2, 3]}, policy_id="tls_1")
    finally:
        for name, original in (
            ("torch", original_torch),
            ("ray", original_ray),
            ("ray.rllib", original_ray_rllib),
            ("ray.rllib.core", original_ray_rllib_core),
            ("ray.rllib.core.columns", original_ray_rllib_core_columns),
        ):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    assert action == 2
    assert algo.calls == [{"get_module_policy_id": "tls_1"}]


def test_compute_single_action_uses_module_forward_when_rllib_compute_single_action_hits_env_runner_gap():
    class DummyColumns:
        ACTIONS = "actions"
        OBS = "obs"

    class DummyTensor:
        def __init__(self, values):
            self._values = values

        def unsqueeze(self, dim):
            del dim
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class DummyTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            del dtype, device
            return DummyTensor(values)

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def no_grad():
            class _NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            return _NoGrad()

    class DummyRayColumnsModule:
        Columns = DummyColumns

    class DummyModule:
        def forward_inference(self, batch):
            del batch
            return {DummyColumns.ACTIONS: DummyTensor([2])}

    class DummyAlgo:
        def compute_single_action(self, obs, policy_id=None, explore=None):
            del obs, policy_id, explore
            raise AttributeError("'MultiAgentEnvRunner' object has no attribute 'get_policy'")

        def get_module(self, policy_id=None):
            del policy_id
            return DummyModule()

    original_torch = sys.modules.get("torch")
    original_ray = sys.modules.get("ray")
    original_ray_rllib = sys.modules.get("ray.rllib")
    original_ray_rllib_core = sys.modules.get("ray.rllib.core")
    original_ray_rllib_core_columns = sys.modules.get("ray.rllib.core.columns")
    try:
        sys.modules["torch"] = DummyTorch
        sys.modules["ray"] = SimpleNamespace()
        sys.modules["ray.rllib"] = SimpleNamespace()
        sys.modules["ray.rllib.core"] = SimpleNamespace()
        sys.modules["ray.rllib.core.columns"] = DummyRayColumnsModule

        action = rllib_runner._compute_single_action(DummyAlgo(), {"graph": [1, 2, 3]}, policy_id="tls_1")
    finally:
        for name, original in (
            ("torch", original_torch),
            ("ray", original_ray),
            ("ray.rllib", original_ray_rllib),
            ("ray.rllib.core", original_ray_rllib_core),
            ("ray.rllib.core.columns", original_ray_rllib_core_columns),
        ):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    assert action == 2


def test_compute_single_action_falls_back_to_algo_compute_single_action_without_module():
    class DummyAlgo:
        def __init__(self):
            self.calls = []

        def get_module(self, policy_id=None):
            del policy_id
            raise RuntimeError("module is unavailable")

        def compute_single_action(self, obs, policy_id=None, explore=None):
            self.calls.append(
                {
                    "obs": obs,
                    "policy_id": policy_id,
                    "explore": explore,
                }
            )
            return 3

    algo = DummyAlgo()

    action = rllib_runner._compute_single_action(algo, {"graph": [1, 2, 3]}, policy_id="tls_1")

    assert action == 3
    assert algo.calls == [
        {
            "obs": {"graph": [1, 2, 3]},
            "policy_id": "tls_1",
            "explore": False,
        }
    ]
