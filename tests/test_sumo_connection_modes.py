import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sumo_rl.environment import env as env_mod


class _DummyConnection:
    def __init__(self, on_close=None):
        self._on_close = on_close

    def close(self):
        if self._on_close is not None:
            self._on_close()
        return None


class _DummyTrafficSignal:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _DummyLibsumoModule:
    def close(self):
        return None


def test_sumo_environment_uses_labeled_traci_connections_when_libsumo_is_disabled(monkeypatch):
    start_calls = []

    monkeypatch.setattr(env_mod.sumolib, "checkBinary", lambda name: name)
    monkeypatch.setattr(env_mod, "TrafficSignal", _DummyTrafficSignal)
    monkeypatch.setattr(env_mod.traci, "switch", lambda label: None, raising=False)
    monkeypatch.setattr(env_mod.traci, "close", lambda: None)

    def fake_start(backend, cmd, *, label=None):
        start_calls.append({"backend": backend, "cmd": list(cmd), "label": label})
        return _DummyConnection()

    monkeypatch.setattr(env_mod, "_start_traci_with_retries", fake_start)

    env = env_mod.SumoEnvironment(
        net_file="net.xml",
        route_file="route.xml",
        ts_ids=["tls_0"],
        use_libsumo=False,
    )
    env._start_simulation()

    assert start_calls[0]["label"].startswith("init_connection")
    assert start_calls[1]["label"] == env.label
    env.close()


def test_sumo_environment_uses_process_global_libsumo_when_enabled(monkeypatch):
    start_calls = []

    monkeypatch.setattr(env_mod.sumolib, "checkBinary", lambda name: name)
    monkeypatch.setattr(env_mod, "TrafficSignal", _DummyTrafficSignal)
    monkeypatch.setattr(env_mod.traci, "close", lambda: None)
    monkeypatch.setattr(env_mod, "libsumo", _DummyLibsumoModule())

    def fake_start(backend, cmd, *, label=None):
        start_calls.append({"backend": backend, "cmd": list(cmd), "label": label})
        return _DummyConnection()

    monkeypatch.setattr(env_mod, "_start_traci_with_retries", fake_start)

    env = env_mod.SumoEnvironment(
        net_file="net.xml",
        route_file="route.xml",
        ts_ids=["tls_0"],
        use_libsumo=True,
    )
    env._start_simulation()

    assert start_calls[0]["label"] is None
    assert start_calls[1]["label"] is None
    env.close()


def test_sumo_environment_defaults_to_traci_when_backend_is_not_configured(monkeypatch):
    start_calls = []

    monkeypatch.setattr(env_mod.sumolib, "checkBinary", lambda name: name)
    monkeypatch.setattr(env_mod, "TrafficSignal", _DummyTrafficSignal)
    monkeypatch.setattr(env_mod.traci, "close", lambda: None)

    def fake_start(backend, cmd, *, label=None):
        start_calls.append({"backend": backend, "cmd": list(cmd), "label": label})
        return _DummyConnection()

    monkeypatch.setattr(env_mod, "_start_traci_with_retries", fake_start)

    env = env_mod.SumoEnvironment(
        net_file="net.xml",
        route_file="route.xml",
        ts_ids=["tls_0"],
    )

    assert env.use_libsumo is False
    assert start_calls[0]["label"].startswith("init_connection")
    env.close()


def test_close_switches_traci_only_for_isolated_connections(monkeypatch):
    switch_calls = []
    close_calls = []

    monkeypatch.setattr(env_mod.sumolib, "checkBinary", lambda name: name)
    monkeypatch.setattr(env_mod, "TrafficSignal", _DummyTrafficSignal)
    monkeypatch.setattr(env_mod, "libsumo", _DummyLibsumoModule())
    monkeypatch.setattr(env_mod, "_start_traci_with_retries", lambda backend, cmd, *, label=None: _DummyConnection())
    monkeypatch.setattr(env_mod.traci, "switch", lambda label: switch_calls.append(label), raising=False)
    monkeypatch.setattr(env_mod.traci, "close", lambda: close_calls.append("close"))

    traci_env = env_mod.SumoEnvironment(
        net_file="net.xml",
        route_file="route.xml",
        ts_ids=["tls_0"],
        use_libsumo=False,
    )
    traci_env.sumo = object()
    traci_env.close()

    libsumo_env = env_mod.SumoEnvironment(
        net_file="net.xml",
        route_file="route.xml",
        ts_ids=["tls_0"],
        use_libsumo=True,
    )
    libsumo_env.sumo = _DummyConnection(lambda: close_calls.append("libsumo_close"))
    libsumo_env.close()

    assert switch_calls == [traci_env.label]
    assert close_calls == ["close", "libsumo_close"]
