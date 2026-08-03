"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""

import os
import sys
from collections import deque
from collections.abc import Callable
from math import ceil

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces

from ..util.tripinfo import is_ghost_vehicle as _is_ghost_vehicle


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in
      incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class.
    See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward
    function by implementing a custom reward function and passing it to the
    constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5
    NASH_AVERAGE_SPEED_EPSILON = 0.1

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: str | Callable | list,
        reward_weights: list[float],
        reward_penalty_lambda: float,
        reward_nash_epsilon: float,
        reward_nsw_window_cycle_multiplier: float,
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string
                with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            reward_penalty_lambda (float): Coefficient for penalty-based reward functions.
            reward_nash_epsilon (float): Positive smoothing term added to Nash-style phase utilities.
            reward_nsw_window_cycle_multiplier (float): Multiplier applied to the fixed-time cycle length
                to size the rolling NSW reward window.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.reward_penalty_lambda = float(reward_penalty_lambda)
        self.reward_nash_epsilon = float(reward_nash_epsilon)
        self.reward_nsw_window_cycle_multiplier = float(reward_nsw_window_cycle_multiplier)
        self.sumo = sumo
        self._last_fixed_cycle_phase_index = None
        self._phase_stats_cache_step = None
        self._phase_stats_cache = None
        self.fixed_cycle_length_seconds = 1.0
        self.reward_nsw_window_seconds = 1
        self._nsw_window_samples = deque(maxlen=1)

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}
        self.phase_lanes = self._build_phase_lanes()

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.fixed_cycle_length_seconds = self._derive_fixed_cycle_length_seconds(phases)
        self.reward_nsw_window_seconds = self._compute_nsw_window_seconds(
            self.fixed_cycle_length_seconds,
            self.reward_nsw_window_cycle_multiplier,
        )
        self._nsw_window_samples = deque(maxlen=self.reward_nsw_window_seconds)
        if self.env.fixed_ts:
            self.fixed_cycle_phases = list(phases)
            self.green_phases = [phase for index, phase in enumerate(phases) if index % 2 == 0]
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            self.sync_fixed_time_state()
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @staticmethod
    def _derive_fixed_cycle_length_seconds(phases) -> float:
        cycle_length = 0.0
        for phase in phases:
            try:
                cycle_length += float(getattr(phase, "duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return max(1.0, cycle_length)

    @staticmethod
    def _compute_nsw_window_seconds(cycle_length_seconds: float, cycle_multiplier: float) -> int:
        return max(1, int(ceil(float(cycle_length_seconds) * float(cycle_multiplier))))

    def _build_phase_lanes(self) -> list[list[str]]:
        phase_lanes = []
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.id)
        for phase in getattr(self, "green_phases", []):
            lanes = []
            state = getattr(phase, "state", "")
            for signal_index, signal_state in enumerate(state):
                if signal_state not in ("G", "g"):
                    continue
                if signal_index >= len(controlled_links):
                    continue
                links = controlled_links[signal_index] or []
                for link in links:
                    incoming_lane = link[0]
                    if incoming_lane not in lanes:
                        lanes.append(incoming_lane)
            phase_lanes.append(lanes)
        return phase_lanes

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def sync_fixed_time_state(self):
        """Synchronize cached fixed-time phase state with the active SUMO program."""

        if not self.env.fixed_ts:
            return

        current_phase_index = int(self.sumo.trafficlight.getPhase(self.id))
        if self._last_fixed_cycle_phase_index is None or self._last_fixed_cycle_phase_index != current_phase_index:
            self.time_since_last_phase_change = 0
        else:
            self.time_since_last_phase_change += 1
        self._last_fixed_cycle_phase_index = current_phase_index

        current_state = self.sumo.trafficlight.getRedYellowGreenState(self.id)
        matched_green_phase = next(
            (index for index, phase in enumerate(self.green_phases) if getattr(phase, "state", "") == current_state),
            None,
        )
        if matched_green_phase is None and current_phase_index > 0:
            previous_state = getattr(self.fixed_cycle_phases[current_phase_index - 1], "state", "")
            matched_green_phase = next(
                (index for index, phase in enumerate(self.green_phases) if getattr(phase, "state", "") == previous_state),
                None,
            )
        if matched_green_phase is None:
            matched_green_phase = min(current_phase_index // 2, max(0, self.num_green_phases - 1))

        self.green_phase = int(matched_green_phase)
        self.is_yellow = "y" in current_state.lower()

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)

        # Ensure max green time is enforced if needed
        if self.enforce_max_green and new_phase == self.green_phase and self.time_since_last_phase_change >= self.max_green:
            new_phase = (self.green_phase + 1) % self.num_green_phases  # Next phase is activated

        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self) -> float | np.ndarray:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
        else:
            self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards

        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _nash_average_speed_reward(self):
        epsilon = float(getattr(self, "reward_nash_epsilon", self.NASH_AVERAGE_SPEED_EPSILON))
        phase_average_speeds, _phase_max_waiting_times = self.get_windowed_phase_speed_wait_stats()
        if not phase_average_speeds:
            return self.get_average_speed() + epsilon

        phase_utilities = np.asarray(phase_average_speeds, dtype=np.float64) + epsilon
        return float(np.exp(np.mean(np.log(phase_utilities))))

    def _weighted_nash_average_speed_reward(self):
        epsilon = float(getattr(self, "reward_nash_epsilon", self.NASH_AVERAGE_SPEED_EPSILON))
        phase_average_speeds, phase_max_waiting_times = self.get_windowed_phase_speed_wait_stats()
        if not phase_average_speeds:
            return self.get_average_speed() + epsilon

        phase_utilities = np.asarray(phase_average_speeds, dtype=np.float64) + epsilon
        phase_max_waiting_times = np.asarray(phase_max_waiting_times, dtype=np.float64)

        if phase_max_waiting_times.size != phase_utilities.size:
            phase_max_waiting_times = np.zeros_like(phase_utilities)

        total_max_waiting_time = float(np.sum(phase_max_waiting_times))
        if total_max_waiting_time > 0.0:
            phase_weights = phase_max_waiting_times / total_max_waiting_time
        else:
            phase_weights = np.full_like(phase_utilities, 1.0 / float(phase_utilities.size))

        return float(np.exp(np.sum(phase_weights * np.log(phase_utilities))))

    def _queue_reward(self):
        return -self.get_total_queued()

    def _normalized_queue_reward(self):
        lanes_queue = self.get_lanes_queue()
        if not lanes_queue:
            return 0.0
        return -float(np.mean(lanes_queue))

    def _normalized_pressure_reward(self):
        incoming = self.get_lanes_density()
        outgoing = self.get_out_lanes_density()
        incoming_mean = 0.0 if not incoming else float(np.mean(incoming))
        outgoing_mean = 0.0 if not outgoing else float(np.mean(outgoing))
        return outgoing_mean - incoming_mean

    def _co2_reward(self):
        return -self.get_total_co2()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def _diff_waiting_time_with_unchosen_phase_penalty_reward(self):
        lane_waiting_times = self.get_accumulated_waiting_time_per_lane()
        ts_wait = sum(lane_waiting_times) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        penalty = self._get_max_unchosen_phase_wait_penalty(lane_waiting_times)
        self.last_ts_waiting_time = ts_wait
        return reward - (self.reward_penalty_lambda * penalty)

    def _get_max_unchosen_phase_wait_penalty(self, lane_waiting_times: list[float]) -> float:
        if not self.phase_lanes:
            return 0.0

        lane_wait_map = {lane: wait for lane, wait in zip(self.lanes, lane_waiting_times)}
        phase_queued_counts = self.get_phase_queued_counts()
        phase_penalties = []

        for phase_index, phase_lanes in enumerate(self.phase_lanes):
            if phase_index == self.green_phase:
                continue

            queue_length = phase_queued_counts[phase_index] if phase_index < len(phase_queued_counts) else 0
            if queue_length <= 0:
                continue

            cumulative_wait = sum(lane_wait_map.get(lane, 0.0) for lane in phase_lanes)
            phase_penalties.append(cumulative_wait / float(queue_length))

        return max(phase_penalties, default=0.0)

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> list[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = [veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)]
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(
            len([veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)])
            for lane in self.out_lanes
        ) - sum(
            len([veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)])
            for lane in self.lanes
        )

    def get_out_lanes_density(self) -> list[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            len([veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)])
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> list[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            len([veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)])
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> list[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the
        number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            sum(
                1
                for veh in self.sumo.lane.getLastStepVehicleIDs(lane)
                if not _is_ghost_vehicle(veh) and self.sumo.vehicle.getSpeed(veh) < 0.1
            )
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(
            1
            for lane in self.lanes
            for veh in self.sumo.lane.getLastStepVehicleIDs(lane)
            if not _is_ghost_vehicle(veh) and self.sumo.vehicle.getSpeed(veh) < 0.1
        )

    def get_phase_queued_counts(self) -> list[int]:
        """Returns queued-vehicle counts for the lane groups served by each green phase."""
        return [
            sum(
                1
                for lane in phase_lanes
                for veh in self.sumo.lane.getLastStepVehicleIDs(lane)
                if not _is_ghost_vehicle(veh) and self.sumo.vehicle.getSpeed(veh) < 0.1
            )
            for phase_lanes in self.phase_lanes
        ]

    def get_phase_average_speeds(self) -> list[float]:
        """Returns mean normalized speed ratios for the vehicles served by each green phase."""
        return [stats["average_speed"] for stats in self._get_phase_speed_wait_stats()]

    def get_phase_max_waiting_times(self) -> list[float]:
        """Returns the max current waiting time among the vehicles served by each green phase."""
        return [stats["max_waiting_time"] for stats in self._get_phase_speed_wait_stats()]

    def record_nsw_window_sample(self) -> None:
        """Record one SUMO-second sample for windowed NSW rewards."""

        stats = self._get_phase_speed_wait_stats()
        self._nsw_window_samples.append(
            {
                "average_speeds": [float(item["average_speed"]) for item in stats],
                "max_waiting_times": [float(item["max_waiting_time"]) for item in stats],
            }
        )

    def get_windowed_phase_speed_wait_stats(self) -> tuple[list[float], list[float]]:
        """Returns per-phase mean speeds and max waits over the configured NSW window."""

        raw_samples = getattr(self, "_nsw_window_samples", None)
        if raw_samples is None:
            speeds = [float(value) for value in self.get_phase_average_speeds()]
            try:
                waits = [float(value) for value in self.get_phase_max_waiting_times()]
            except AttributeError:
                waits = [0.0 for _ in speeds]
            return speeds, waits

        samples = list(raw_samples)
        if not samples:
            stats = self._get_phase_speed_wait_stats()
            return (
                [float(item["average_speed"]) for item in stats],
                [float(item["max_waiting_time"]) for item in stats],
            )

        phase_count = max((len(sample["average_speeds"]) for sample in samples), default=0)
        if phase_count == 0:
            return [], []

        window_average_speeds = []
        window_max_waiting_times = []
        for phase_index in range(phase_count):
            speeds = [
                sample["average_speeds"][phase_index] for sample in samples if phase_index < len(sample["average_speeds"])
            ]
            waits = [
                sample["max_waiting_times"][phase_index]
                for sample in samples
                if phase_index < len(sample["max_waiting_times"])
            ]
            window_average_speeds.append(1.0 if not speeds else float(np.mean(speeds)))
            window_max_waiting_times.append(0.0 if not waits else float(max(waits)))
        return window_average_speeds, window_max_waiting_times

    def get_total_co2(self) -> float:
        """Returns the total CO2 emissions (mg/s) of the vehicles in the incoming lanes of the intersection."""
        return sum(self.sumo.vehicle.getCO2Emission(veh) for veh in self._get_veh_list())

    def _get_unique_phase_vehicle_ids(self, phase_lanes: list[str]) -> list[str]:
        seen = set()
        phase_vehicles = []
        for lane in phase_lanes:
            for veh in self.sumo.lane.getLastStepVehicleIDs(lane):
                if _is_ghost_vehicle(veh) or veh in seen:
                    continue
                seen.add(veh)
                phase_vehicles.append(veh)
        return phase_vehicles

    def _get_phase_speed_wait_stats(self) -> list[dict]:
        cache_step = self._get_phase_stats_cache_step()
        cached_stats = getattr(self, "_phase_stats_cache", None)
        cached_step = getattr(self, "_phase_stats_cache_step", None)
        if cached_stats is not None and cache_step is not None and cached_step == cache_step:
            return cached_stats

        phase_stats = []
        for phase_lanes in self.phase_lanes:
            phase_vehicles = self._get_unique_phase_vehicle_ids(phase_lanes)
            if not phase_vehicles:
                phase_stats.append({"average_speed": 1.0, "max_waiting_time": 0.0})
                continue

            normalized_speeds = []
            max_waiting_time = 0.0
            for veh in phase_vehicles:
                allowed_speed = self.sumo.vehicle.getAllowedSpeed(veh)
                normalized_speeds.append(0.0 if allowed_speed <= 0.0 else self.sumo.vehicle.getSpeed(veh) / allowed_speed)
                max_waiting_time = max(max_waiting_time, float(self.sumo.vehicle.getWaitingTime(veh)))
            phase_stats.append(
                {
                    "average_speed": float(np.mean(normalized_speeds)),
                    "max_waiting_time": max_waiting_time,
                }
            )

        if cache_step is not None:
            self._phase_stats_cache_step = cache_step
            self._phase_stats_cache = phase_stats
        else:
            self._phase_stats_cache_step = None
            self._phase_stats_cache = None
        return phase_stats

    def _get_phase_stats_cache_step(self):
        env = getattr(self, "env", None)
        return getattr(env, "sim_step", None)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += [veh for veh in self.sumo.lane.getLastStepVehicleIDs(lane) if not _is_ghost_vehicle(veh)]
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "diff-waiting-time-with-unchosen-phase-penalty": _diff_waiting_time_with_unchosen_phase_penalty_reward,
        "average-speed": _average_speed_reward,
        "nash-average-speed": _nash_average_speed_reward,
        "weighted-nash-average-speed": _weighted_nash_average_speed_reward,
        "queue": _queue_reward,
        "normalized-queue": _normalized_queue_reward,
        "pressure": _pressure_reward,
        "normalized-pressure": _normalized_pressure_reward,
        "co2": _co2_reward,
    }
