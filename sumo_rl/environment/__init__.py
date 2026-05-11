"""SUMO Environment for Traffic Signal Control."""

try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError:
    register = None


if register is not None:
    register(
        id="sumo-rl-v0",
        entry_point="sumo_rl.environment.env:SumoEnvironment",
        kwargs={"single_agent": True},
    )
