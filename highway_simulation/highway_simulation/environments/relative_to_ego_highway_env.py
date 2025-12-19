"""Gymnasium environment for the highway simulation."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from highway_simulation.scripts.highway import Highway
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import Config
from highway_simulation.scripts.util.metrics import Metrics
# from highway_simulation.scripts.planning.live_plotter import LivePlotter
class HighwayEnv(gym.Env):
    """Highway simulation environment compatible with Gymnasium."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        self.config = self.default_config()
        self.highway = Highway(self.config)
        self.action_space = spaces.Discrete(len(Action))

        # Define the observation space
        V = 5  # Number of vehicles (including ego)
        F = 3  # Features per vehicle (x, y, v_x)
        # Define observation space with shape (V, F), with normalization range in [0, 1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(V * F,), dtype=np.float32
        )

        # Environment-specific parameters
        self.current_state = None
        self.done = False
        self.seed_val = None

        # statistics
        self.ego_speeds = []
        self.start_time = time.time()
        self.avg_vehicle_speeds = []
        self.avg_time_gaps = []
        self.avg_ttc = []

        self.times = []
        self.positions_x = []
        self.velocities_x = []
        self.accelerations_x = []
        self.positions_y = []

        # self.plotter = LivePlotter()

    @classmethod
    def default_config(cls) -> Config:
        return Config(
            min_vel=13,
            max_vel=40,
            min_rewardable_vel=28,
            max_rewardable_vel=36,
            collision_threshold=2,
            lane_change_duration=12,
            num_of_vehicles=30,
            road_length=51000,
            vehicle_width=4.5,
            vehicle_height=2,
            screen_width=1499,
            screen_height=1000,
            lane_width=3.5,
            time_step=0.1,
            num_lanes=3,
            effective_sim_length=7500,
            effective_sim_time=180,
            aggresive_driver=True,
            ego_drives_with_mobil=False,
            evaluation_mode=False,
        )

    def set_config(self, config: Config) -> None:
        self.config = config
        self.highway = Highway(self.config)

    def seed(self, seed: Optional[int] = None) -> None:
        self.seed_val = seed

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        no_vehicles: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=self.seed_val)
        self.current_state = self.highway.reset(self.seed_val, no_vehicles)
        return self.current_state, {}  # Return the initial state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action and return the result."""
        # Take a step in the simulation
        new_state, reward, done, _ = self.highway.step(action)

        # Extract current state
        time_step = len(self.times) * self.config.time_step
        self.times.append(time_step)
        self.positions_x.append(self.highway.lane_manager.ego_vehicle.x)
        self.velocities_x.append(self.highway.lane_manager.ego_vehicle.speed)
        self.accelerations_x.append(self.highway.lane_manager.ego_vehicle.acc)
        self.positions_y.append(self.highway.lane_manager.ego_vehicle.y)

        # Update the live plot
        # self.plotter.update(self.times, self.positions_x, self.velocities_x, self.accelerations_x, self.positions_y)

        if self.config.evaluation_mode:
            self.avg_vehicle_speeds.append(
                self.highway.lane_manager.avg_speed_of_all_vehicles * 36 / 10
            )
            self.avg_time_gaps.append(self.highway.lane_manager.avg_time_gap_per_lane)
            self.avg_ttc.append(self.highway.reward_calculator.ttc_metric)

            if done:
                self.print_summary()

        # Return the new state, reward, done flag, and additional info (empty here)
        return new_state, reward, done, False, {}

    def render(self, mode: str = "human") -> None:
        """Render the environment to the screen."""
        self.highway.render()  # Render the highway environment

    def close(self) -> None:
        """Clean up resources if necessary."""
        self.highway.close()

    def print_summary(self) -> None:

        def get_lane_time_distribution(time_in_lanes: Dict[int, float]) -> Dict[int, float]:
            """Calculate the percentage of time spent in each lane based on simulation time."""
            total_time = sum(time_in_lanes.values())
            if total_time == 0:
                return {lane: 0.0 for lane in time_in_lanes}

            distribution = {
                lane: (time / total_time) * 100
                for lane, time in time_in_lanes.items()
            }
            return distribution
        
        end_time = time.time()
        total_time = end_time - self.start_time

        avg_ego_speed = (
            self.highway.lane_manager.ego_vehicle.history_trajectory.return_avg_vehicle_speed
        )
        avg_vehicle_speed = (
            sum(self.avg_vehicle_speeds) / len(self.avg_vehicle_speeds)
            if self.avg_vehicle_speeds
            else 0
        )

        ## TTC
        avg_time_gap = (
            sum(self.avg_time_gaps) / len(self.avg_time_gaps) if self.avg_time_gaps else 0
        )

        num_infinite = sum(1 for ttc in self.avg_ttc if ttc == float('inf'))
        # Calculate the percentage of infinite values
        percentage_infinite = (num_infinite / len(self.avg_ttc)) * 100

        # Get finite values and compute the average
        finite_values = [ttc for ttc in self.avg_ttc if ttc != float("inf")]
        average_finite_ttc = sum(finite_values) / len(finite_values) if finite_values else 0
        distribution = get_lane_time_distribution(
            self.highway.lane_manager.time_in_lanes
        )

        metrics = Metrics(
            seed=self.seed_val,
            num_of_vehicles=self.config.num_of_vehicles,
            avg_ego_speed=avg_ego_speed,
            num_of_lane_changes_ego=self.highway.lane_manager.ego_lane_changes,
            wall_time_spent=total_time,
            ego_vehicle_travelled_percentage=self.highway.lane_manager.ego_vehicle.x
            / self.config.effective_sim_length,
            avg_vehicle_speed=avg_vehicle_speed,
            avg_time_gap=avg_time_gap,
            ttc_infinite_percentage=percentage_infinite,
            ttc_finite_average=average_finite_ttc,
            lane_time_distribution=distribution,
            acceleration_distribution=self.highway.lane_manager.ego_vehicle.history_trajectory.return_acceleration_distribution,
            successful_run=True
            if round(
                self.highway.lane_manager.ego_vehicle.x / self.config.effective_sim_length
            )
            == 1
            else False,
            is_aggresive=self.config.aggresive_driver,
            is_driven_by_mobil=self.config.ego_drives_with_mobil,
        )
        metrics.print()
        metrics.save()

        # self.highway.lane_manager.ego_vehicle.history_trajectory.plot_trajectory(plot_history_of_data=True)


if __name__ == "__main__":
    env = HighwayEnv()
    # print("checkenv")
    # check_env(env.unwrapped)
    # print("check over")

    obs = env.reset()
    terminated = False
    while not terminated:
        act = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(act)
