"""Core simulation orchestration."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.planning.decision_to_trajectory import DecisionToTrajectory
from highway_simulation.scripts.planning.state import Trajectory
from highway_simulation.scripts.plotting.highwayPlotter import HighwayPlotter
from highway_simulation.scripts.rewards.rewardCalculator import RewardCalculator
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import Config
from highway_simulation.scripts.util.metrics import Metrics
from highway_simulation.scripts.vehicle.vehicle import Vehicle

class Highway:
    """Main simulation loop and state management."""

    def __init__(self, config: Config, render_mode: str = "human") -> None:
        self.config = config
        self.vehicle_width = config.vehicle_width
        self.num_lanes = config.num_lanes
        self.effective_sim_length = config.effective_sim_length
        self.effective_sim_time = config.effective_sim_time

        Vehicle.set_config(config)
        LaneManager.set_config(config)
        RewardCalculator.set_config(config)
        HighwayPlotter.set_config(config)
        DecisionToTrajectory.set_config(config)
        Trajectory.set_config(config)
        Metrics.set_config(config)
        self.lane_manager = LaneManager()
        self.reward_calculator = RewardCalculator(self.lane_manager)
        self.highway_plotter = HighwayPlotter(self.lane_manager)
        self.decision_to_trajectory = DecisionToTrajectory()

    def reset(self, seed: int, no_vehicles: Optional[bool] = None) -> np.ndarray:
        if hasattr(self.lane_manager, "ego_vehicle"):
            del self.lane_manager.ego_vehicle
        self.lane_manager.remove_all_vehicles()
        self.lane_manager.add_vehicles_to_sim(seed, no_vehicles)
        self.reward_calculator.sim_start_time = time.time()
        self.lane_manager.ego_lane_changes = 0
        self.lane_manager.lane_change_in_progress = False
        return self.get_state()

    def reset_for_test_cases(self):
        self.lane_manager.remove_all_vehicles()
        test_case_name = self.lane_manager.add_vehicles_to_sim_from_test_case()

        return self.get_state(), test_case_name

    def update(self) -> None:
        """Update vehicle positions and clear lanes for re-sorting."""
        self.lane_manager.update()

    def find_vehicle_state(self, vehicle: Optional[Vehicle]) -> Tuple[float, float]:
        if vehicle is None:
            return 1, 1
        return (
            abs(self.lane_manager.ego_vehicle.x - vehicle.x),
            abs(self.lane_manager.ego_vehicle.speed - vehicle.speed),
        )

    def step(self, action: int):
        previous_ego = (
            self.lane_manager.ego_vehicle.x,
            self.lane_manager.ego_vehicle.speed,
            self.lane_manager.ego_vehicle.lane,
        )
        bad_action = self.take_action(action)
        self.update()
        reward, done = self.calculate_reward(previous_ego, bad_action)
        new_state = self.get_state()
        return new_state, reward, done, {}

    def render(self) -> None:
        self.highway_plotter.render()

    def close(self) -> None:
        self.highway_plotter.close()

        
    """ Q LEARNING FUNCTIONS """
    def return_env_state(self) -> None:
        pass

    def normalize_xyv(self, x: float, y: float, v: float) -> Tuple[float, float, float]:
        x_max = 400  # absolute value of relative coordinates
        y_mean = ((self.config.num_lanes - 1) * self.config.lane_width) / 2
        return (
            np.clip((x + x_max) / (2 * x_max), 0, 1),
            np.clip(y / (2 * y_mean), 0, 1),
            self.normalize_v(v),
        )

    def normalize_v(self, v: float) -> float:
        v_mean = self.config.max_vel / 2  # because negative speed does not exist
        return np.clip(v / (2*v_mean), 0, 1)

    def get_state(self) -> np.ndarray:
        V = 5  # Number of vehicles (including ego)
        F = 3  # Features per vehicle

        # Ego vehicle state
        ego_vehicle = self.lane_manager.ego_vehicle
        ego_x, ego_y = ego_vehicle.relative_x, ego_vehicle.y
        ego_vx = ego_vehicle.speed  # Assuming no lateral velocity
    
        surrounding_vehicles = self.lane_manager.get_nearby_vehicles(V - 1)
        state = np.zeros((V, F), dtype=np.float32)
        ego_x_norm, ego_y_norm, ego_vx_norm = self.normalize_xyv(
            ego_x, ego_y, ego_vx
        )
        state[0] = [ego_x_norm, ego_y_norm, ego_vx_norm]
        
        #print(surrounding_vehicles)
        for i, vehicle in enumerate(surrounding_vehicles):
            x = vehicle.relative_x 
            y = vehicle.y
            vx = vehicle.speed

            # Normalize features
            x_norm, y_norm, vx_norm = self.normalize_xyv(x, y, vx)
            # Add normalized data to the state array
            state[i + 1] = [x_norm, y_norm, vx_norm]

        #print((state.flatten()))
        return np.array(state.flatten())

    def take_action(self, action: int):
        """Execute the chosen action. Can not do a lane change while doing a lane change, can not go out of map"""
        assert action in range(0, len(Action))
        action = Action(action)
        if self.config.ego_drives_with_mobil:
            return

        if self.lane_manager.lane_change_in_progress and (
            action == Action.CHANGE_LANE_LEFT or action == Action.CHANGE_LANE_RIGHT
        ):
            action = Action.NO_ACTION
        if (
            action == Action.CHANGE_LANE_LEFT
            and self.lane_manager.is_ego_in_leftmost_lane()
        ) or (
            action == Action.CHANGE_LANE_RIGHT
            and self.lane_manager.is_ego_in_rightmost_lane()
        ):
            action = Action.NO_ACTION

        elif action == Action.ACCELERATE:
            self.lane_manager.ego_vehicle.acc = 2
            return
           
        elif action == Action.DECELERATE:
            self.lane_manager.ego_vehicle.acc = -2
            return
        elif action == Action.EMERGENCY_BRAKE:
            self.lane_manager.ego_vehicle.acc = -4
            return
        if action != Action.NO_ACTION and self.lane_manager.ego_vehicle.trajectory.is_trajectory_empty():
            if action == Action.CHANGE_LANE_LEFT:
                self.lane_manager.ego_vehicle.target_lane -= 1
            if action == Action.CHANGE_LANE_RIGHT:
                self.lane_manager.ego_vehicle.target_lane += 1
            trajectory = self.decision_to_trajectory.process_decision(
                self.lane_manager.ego_vehicle.return_state, action
            )
            #trajectory.plot_trajectory()
            self.lane_manager.ego_vehicle.trajectory = trajectory
            self.lane_manager.ego_vehicle.ongoing_trajectory = True
            self.lane_manager.ego_vehicle.stored_planned_trajectory = [state for state in trajectory.trajectory]
            self.lane_manager.ego_vehicle.current_trajectory_start_index = len(
                self.lane_manager.ego_vehicle.history_trajectory.trajectory
            )  # Store start index
            self.lane_manager.lane_change_in_progress = True
            return
            #logger.info(f"updated trajectory, taken action is: {action}")
        if action == Action.NO_ACTION:
            self.lane_manager.ego_vehicle.acc = 0
            return    

        return 0  # change later TODO

    def calculate_reward(self, previous_ego, bad_action):
        """Calculate the reward based on the ego vehicle's situation."""
        self.reward_calculator.set_ego_vehicle(self.lane_manager.ego_vehicle)
        reward, done = self.reward_calculator.calculate(previous_ego, bad_action)
        return reward, done
