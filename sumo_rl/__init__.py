"""Import all the necessary modules for the sumo_rl package."""

try:
    from sumo_rl.environment.env import (
        ObservationFunction,
        SumoEnvironment,
        TrafficSignal,
        env,
        parallel_env,
    )
    from sumo_rl.environment.resco_envs import (
        arterial4x4,
        cologne1,
        cologne3,
        cologne8,
        grid4x4,
        ingolstadt1,
        ingolstadt7,
        ingolstadt21,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"gymnasium", "pettingzoo", "sumolib", "traci"}:
        raise

    ObservationFunction = None
    SumoEnvironment = None
    TrafficSignal = None
    env = None
    parallel_env = None
    arterial4x4 = None
    cologne1 = None
    cologne3 = None
    cologne8 = None
    grid4x4 = None
    ingolstadt1 = None
    ingolstadt7 = None
    ingolstadt21 = None


__version__ = "1.4.5"
