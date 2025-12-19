"""Vehicle dynamics and control behavior."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from highway_simulation.scripts.vehicle.mpc import MPCController
from highway_simulation.scripts.vehicle.pure_pursuit import PurePursuit
from highway_simulation.scripts.vehicle.util import IDM_PARAM, MOBIL_PARAM, random_color
from highway_simulation.scripts.planning.state import (
    Acc,
    Angles,
    Jerk,
    Pos,
    State,
    Trajectory,
    Vel,
)
from highway_simulation.scripts.util.config import Config


class Vehicle:
    config = Config

    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def __init__(
        self,
        x: float,
        lane: int,
        speed: float,
        v_max: float,
        idm_param: IDM_PARAM = IDM_PARAM(
            a_max=0.7, s0=2.0, T=1.6, b=1.7, delta=4
        ),
        is_ego: bool = False,
        color: Optional[Tuple[int, int, int]] = None,
        mobil_param: MOBIL_PARAM = MOBIL_PARAM(politeness=0.5, a_thr=0.2, b_safe=2.0),
        L_f: float = 2.5,
        L_r: float = 2.5,
    ) -> None:
        self.ongoing_trajectory = False
        self.trajectory_completed = False  # only true for one time stamp per trajectory
        self.x = x
        self.length = self.config.vehicle_width
        self.lane = lane
        self.target_lane = lane  # will be useful during lane change
        self.y = lane * self.config.lane_width

        self.speed = speed * 10 / 36  # convert to meters/second
        self.lateral_speed = 0
        self.one_step_movement = 0
        self.is_ego = is_ego
        self.color = self.config.colors["BLUE"] if is_ego else (color or random_color())
        self.trajectory: Trajectory = Trajectory()
        self.current_trajectory_start_index = 0
        self.stored_planned_trajectory = []
        self.relative_x = self.x  # for plotting only
        
        self.L_f = L_f  # Distance from the center of gravity to the front axle
        self.L_r = L_r  # Distance from the center of gravity to the rear axle
        self.steering_angle = 0.0  # Steering angle in radians
        self.theta = 0.0  # Vehicle heading angle in radians
        self.pure_pursuit = PurePursuit(1.9, 2.5)
        self.mpc_controller = MPCController(horizon=10, dt=self.config.time_step)

        ### IDM PARAM
        self.a_max = idm_param.a_max
        self.v_max = v_max * 10 / 36  # convert to meters/second
        self.initial_v_max = self.v_max  # used for returning to original velocity after takeover
        self.s0 = idm_param.s0  # min dist to ahead vehicle
        self.T = idm_param.T  # safe time headway
        self.b = idm_param.b  # breaking decelaration
        self.delta = idm_param.delta  # accel exponent
        ## MOBIL PARAM
        self.politeness = mobil_param.politeness
        self.a_thr = mobil_param.a_thr
        self.b_safe = mobil_param.b_safe
        self.number_of_lane_changes = 0
        self.take_over_time_counter = 0

        self.history_trajectory: Trajectory = Trajectory()

        self.vehicle_ahead = None
        self.acc = 0
        self.lateral_acc = 0

        
    def update(self) -> None:
        
        if not self.trajectory.is_trajectory_empty():
            self.trajectory.use_next_state()

        dt = self.config.time_step
        if self.is_ego:  # change later
            acc = self.acc
            self.speed += dt * acc
            # self.y = self.y + one_step_lateral_movement  # Update position
            self.history_trajectory.trajectory.append(self.return_state)
            #logger.info(f"Ego state Pos x {self.x},y {self.y} Vel x {self.speed},y {self.lateral_speed} Acc x {self.acc},y {self.lateral_acc}, Steering angle {self.steering_angle}")

        else:
            acc = self.calculate_accel(self.vehicle_ahead)
            self.speed = max(0.1, self.speed + acc * dt)  # Ensure speed doesn't go below 0

        self.update_steering_angle()
        self.bicycle_model(self.steering_angle)

    def update_ego_driven_with_mobil(self) -> None:
        def map_to_closest(value):
            discrete_values = [-4, -2,-1,-0.5, 0, 0.1, 0.25, 0.5,1, 2]
            return min(discrete_values, key=lambda x: abs(x - value))
        """
        only used for ego driven by mobil
        """
        if not self.trajectory.is_trajectory_empty():
            self.trajectory.use_next_state()

        dt = self.config.time_step
        
        acc = self.calculate_accel(self.vehicle_ahead)
        ## map acc to action space of RL for fair comparison.
        if self.is_ego:
            acc = map_to_closest(acc)

        self.speed = max(0.1, self.speed + acc * dt)  # Ensure speed doesn't go below 0

        self.update_steering_angle()
        self.bicycle_model(self.steering_angle)
        
        if self.is_ego:
            self.acc = acc  # plotting purposes
            self.history_trajectory.trajectory.append(self.return_state)

    def update_steering_angle(self) -> None:

        #if self.is_ego:
        #    _, self.steering_angle = self.mpc_controller.compute_controls(self.return_state, self.trajectory)
        #
        # else:
        self.steering_angle = self.pure_pursuit.compute_steering_angle(
            self.return_state, self.trajectory
        )
            #print(f"pure pursuit angle: {self.steering_angle} mpc angle: {angle}")
        #self.steering_angle, self.acc = self.mpc.compute_control(self.return_state, self.trajectory)

    def bicycle_model(self, steering_angle: float) -> None:
        dt = self.config.time_step

        L = self.L_f + self.L_r  # Total wheelbase
        beta = np.arctan((self.L_r / L) * np.tan(steering_angle))

        # Update vehicle state
        self.x += self.speed * np.cos(self.theta + beta) * dt
        self.y += self.speed * np.sin(self.theta + beta) * dt
        self.theta += (self.speed / L) * np.sin(steering_angle) * dt
        
        if self.trajectory.is_trajectory_empty() and not self.trajectory_completed and self.ongoing_trajectory:
            # we want to capture the momemnt when the trajectory is completed. 
            # So we want is_trajectory_empty to be True and trajectory_completed to be False
            self.theta = 0.0
            self.trajectory_completed = True
            self.ongoing_trajectory = False
            #self.y = min([0.0,3.5,7.0], key=lambda x: abs(self.y - x))
            self.number_of_lane_changes += 1
            #self.plot_trajectories()

    def __repr__(self) -> str:
        return f"Vehicle with x: {self.x}, with relative_x: {self.relative_x}, y_position {self.y} speed: {self.speed * 36/10} km/h, lane: {self.lane}, is_ego: {self.is_ego},"

    
    
    def calculate_mobil_lane_change(
        self,
        vehicle_ahead_current: Optional["Vehicle"],
        vehicle_behind_current: Optional["Vehicle"],
        vehicle_ahead_target: Optional["Vehicle"],
        vehicle_behind_target: Optional["Vehicle"],
    ) -> bool:
        """Calculate whether to change lanes using the MOBIL model."""
        if not self.trajectory.is_trajectory_empty():
            return False
        current_acc = self.calculate_accel(vehicle_ahead_current)
        new_acc = self.calculate_accel(vehicle_ahead_target)

        acc_gain = new_acc - current_acc

        if acc_gain < self.a_thr:
            return False

        acc_loss = 0
        if vehicle_behind_current:
            acc_loss += self.politeness * (
                vehicle_behind_current.calculate_accel()
                - vehicle_behind_current.calculate_accel(vehicle_ahead_current)
            )

        if vehicle_behind_target:
            acc_loss += self.politeness * (
                vehicle_behind_target.calculate_accel()
                - vehicle_behind_target.calculate_accel(self)
            )

        # do not allow cut-in
        if vehicle_behind_target is not None and (self.x - vehicle_behind_target.x) < 20:
            return False
        
        if self.is_ego:
            print(f"current_acc: {current_acc}, new_acc: {new_acc}, acc_loss: {acc_loss} ")
            
        return acc_gain - acc_loss > 0
    
    def visualize_lane_change_decision(
        self,
        vehicle_ahead_current: Optional["Vehicle"],
        vehicle_behind_current: Optional["Vehicle"],
        vehicle_ahead_target: Optional["Vehicle"],
        vehicle_behind_target: Optional["Vehicle"],
    ) -> None:
        """Visualize the lane change decision using matplotlib."""
        fig, ax = plt.subplots()

        if vehicle_ahead_current:
            ax.add_patch(patches.Rectangle((vehicle_ahead_current.relative_x, vehicle_ahead_current.y), 
                                           self.config.vehicle_width, 
                                           self.config.vehicle_height, 
                                           edgecolor='black', facecolor='blue'))
        if vehicle_behind_current:
            ax.add_patch(patches.Rectangle((vehicle_behind_current.relative_x, vehicle_behind_current.y), 
                                            self.config.vehicle_width, 
                                           self.config.vehicle_height, 
                                           edgecolor='black', facecolor='blue'))

        # Draw target lane vehicles
        if vehicle_ahead_target:
            ax.add_patch(patches.Rectangle((vehicle_ahead_target.relative_x, vehicle_ahead_target.y), 
                                            self.config.vehicle_width, 
                                           self.config.vehicle_height, 
                                           edgecolor='black', facecolor='green'))
        if vehicle_behind_target:
            ax.add_patch(patches.Rectangle((vehicle_behind_target.relative_x, vehicle_behind_target.y), 
                                            self.config.vehicle_width, 
                                           self.config.vehicle_height, 
                                           edgecolor='black', facecolor='green'))

        # Draw the vehicle making the decision
        ax.add_patch(
            patches.Rectangle(
                (self.x, self.lane * self.y),
                self.config.vehicle_width,
                self.config.vehicle_height,
                edgecolor="black",
                facecolor="red",
            )
        )

        plt.xlim(0, 250)
        plt.ylim(0, self.config.num_lanes * self.config.lane_width)
        plt.xlabel('Distance (m)')
        plt.ylabel('Lane')
        plt.title('Lane Change Decision Visualization')
        plt.show()

    def calculate_accel(self, vehicle_ahead: Optional["Vehicle"] = None) -> float:
        """ Calculate Accel using IDM """
        if vehicle_ahead is not None:
            delta_x = max(0.1, vehicle_ahead.x - self.x - vehicle_ahead.length)
            delta_v = self.speed - vehicle_ahead.speed
        else:
            delta_x = float("inf")
            delta_v = 0
        s_star = self.s0 + max(
            0, self.speed * self.T + (self.speed * delta_v) / (2 * math.sqrt(self.a_max * self.b))
        )
        accel = self.a_max * (
            1 - (self.speed / self.v_max) ** self.delta - (s_star / delta_x) ** 2
        )
        return max(accel, -4)  # bug fix for swolloving vehicles 
    
    @property
    def center_coordinates(self) -> Tuple[float, float]:
        return (self.x + self.length/2, self.config.lane_width * self.lane + self.config.vehicle_height/2)
    
    @property
    def return_state(self) -> State:
        return State(
            Pos(self.x, self.y),
            Vel(self.speed, self.lateral_speed),
            Acc(self.acc, self.lateral_acc),
            Jerk(0, 0),
            Angles(self.theta, self.steering_angle),
        )
    
    def apply_state(self, state: State) -> None:
        ## these values are currently not used
        self.acc = state.acc.x
        self.lateral_acc = state.acc.y
        self.lateral_speed = state.vel.y
        
    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return (
                other.x == self.x
                and other.y == self.y
                and other.speed == self.speed
                and other.relative_x == self.relative_x
            )
        
    def plot_trajectories(self) -> None:
        """
        Plots:
        1. Planned vs. Actuated Y Position over Time
        2. Steering Angle over Time
        """
        if not self.is_ego:
            return
        # Extract planned Y trajectory (saved before execution)
        planned_y = [state.pos.y for state in self.stored_planned_trajectory]

        # Extract actuated Y trajectory (only from the latest trajectory execution)
        actuated_y = [state.pos.y for state in self.history_trajectory.trajectory[self.current_trajectory_start_index:]]

        # Extract steering angles from the actuated trajectory
        steering_angles = [state.angles.steering_angle for state in self.history_trajectory.trajectory[self.current_trajectory_start_index:]]
        
        heading_angles = [state.angles.heading for state in self.history_trajectory.trajectory[self.current_trajectory_start_index:]]

        heading_angles[-1] = 0
        # Generate time axis
        time_steps = np.arange(len(steering_angles)) * self.config.time_step * 1.25

        # Ensure planned trajectory has the same time scale
        planned_time_steps = np.linspace(0, time_steps[-1], len(planned_y))

        
        # Create a figure with two subplots
        plt.figure(figsize=(10, 5))
        plt.plot(planned_time_steps, planned_y, 'b-', label="Planned Lateral Position", alpha=0.7)
        plt.plot(time_steps, actuated_y, 'r-', label="Actuated Lateral Position", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Y Position (m)")
        plt.title("Planned vs Actuated Lateral Position Over Time")
        plt.legend()
        plt.grid()
        plt.show()

        ## 2. Plot Steering Angle over Time
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, steering_angles, 'g-', label="Steering Angle", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (radians)")
        plt.title("Steering Angle Over Time")
        plt.legend()
        plt.grid()
        plt.show()

        ## 3. Plot Heading Angle (Theta) over Time
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, heading_angles, "m-", label="Heading Angle (Theta)", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Heading Angle (radians)")
        plt.title("Heading Angle (Theta) Over Time")
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, heading_angles, 'm-', label="Heading Angle (Theta)", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Heading Angle (radians)")
        plt.title("Heading Angle (Theta) Over Time")
        plt.legend()
        plt.grid()
        plt.show()
