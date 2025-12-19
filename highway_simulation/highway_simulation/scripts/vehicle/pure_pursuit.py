"""Pure pursuit steering controller."""

from __future__ import annotations

import numpy as np

from highway_simulation.scripts.planning.state import State, Trajectory

class PurePursuit:
    """Basic pure pursuit controller for trajectory tracking."""

    def __init__(self, look_ahead_distance: float, wheelbase: float) -> None:
        """
        Initialize the Pure Pursuit controller.
        :param look_ahead_distance: Look-ahead distance for trajectory tracking (L_d).
        :param wheelbase: Wheelbase of the vehicle (L_f + L_r).
        """
        self.look_ahead_distance = look_ahead_distance
        self.wheelbase = wheelbase

    def compute_steering_angle(
        self, current_state: State, trajectory: Trajectory
    ) -> float:
        """
        Compute the steering angle to follow the trajectory.
        :param current_state: The current state of the vehicle.
        :param trajectory: The planned trajectory.
        :return: Steering angle (radians).
        """
        if trajectory.is_trajectory_empty():
            #print("Trajectory is empty!")
            return 0.0

        # Extract vehicle position
        vehicle_x, vehicle_y = current_state.pos.x, current_state.pos.y

        # Find the closest look-ahead point
        look_ahead_point = None
        for state in trajectory.trajectory:
            dist = np.sqrt((state.pos.x - vehicle_x) ** 2 + (state.pos.y - vehicle_y) ** 2)
            if dist >= self.look_ahead_distance:
                look_ahead_point = state
                break

        if look_ahead_point is None:
            # If no point is found within the look-ahead distance, aim for the last point
            look_ahead_point = trajectory.trajectory[-1]

        # Transform look-ahead point to the vehicle's local frame
        dx = look_ahead_point.pos.x - vehicle_x
        dy = look_ahead_point.pos.y - vehicle_y
        theta = np.arctan2(dy, dx)  # Direction to the look-ahead point
        heading_error = theta - current_state.angles.heading  # Heading error relative to vehicle's orientation

        # Calculate steering angle using pure pursuit formula
        ld_squared = self.look_ahead_distance ** 2
        steering_angle = np.arctan2(2 * self.wheelbase * np.sin(heading_error), ld_squared)

        return steering_angle
