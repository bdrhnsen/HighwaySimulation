"""Model predictive controller for trajectory tracking."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from highway_simulation.scripts.planning.state import State, Trajectory


class MPCController:
    """Simple MPC controller based on a discrete bicycle model."""

    def __init__(self, horizon: int = 5, dt: float = 0.1) -> None:
        """
        Initialize the MPC Controller.
        :param horizon: Prediction horizon (number of steps).
        :param dt: Time step for discretization.
        """
        self.init_horizon = horizon
        self.horizon = horizon
        self.dt = dt
        self.L = 2.5  # Wheelbase length
        self.max_steering = np.radians(25)
        self.max_acceleration = 3.0

    def vehicle_dynamics(
        self, state: Sequence[float], control: Sequence[float]
    ) -> np.ndarray:
        """
        Discrete bicycle model dynamics.
        :param state: [x, y, theta, v]
        :param control: [acceleration, steering]
        :return: Next state
        """
        x, y, theta, v = state
        a, delta = control
        next_x = x + v * np.cos(theta) * self.dt
        next_y = y + v * np.sin(theta) * self.dt
        next_theta = theta + (v / self.L) * np.tan(delta) * self.dt
        next_v = v + a * self.dt
        return np.array([next_x, next_y, next_theta, next_v])

    def cost_function(
        self,
        u: np.ndarray,
        initial_state: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        ref_v: np.ndarray,
    ) -> float:
        """
        Cost function for MPC optimization.
        :param u: Flattened control inputs [a_0, delta_0, ..., a_N, delta_N]
        :param initial_state: Current state [x, y, theta, v]
        :param ref_x, ref_y, ref_v: Reference trajectory
        :return: Scalar cost
        """
        u = u.reshape(self.horizon, 2)
        state = initial_state.copy()
        cost = 0

        for i in range(self.horizon):
            state = self.vehicle_dynamics(state, u[i])
            cost += (state[1] - ref_y[i])**2  # Position error
            #cost += (state[0] - ref_x[i])**2 + (state[1] - ref_y[i])**2  # Position error
            
            #cost += (state[3] - ref_v[i])**2  # Velocity error
            cost += 0.1 * (u[i, 0])**2 + 0.1 * (u[i, 1])**2  # Penalize large controls

        return cost

    def compute_controls(
        self, current_state: State, trajectory: Trajectory
    ) -> Tuple[float, float]:
        """
        Compute optimal steering and acceleration using SciPy's optimizer.
        :param current_state: Current state of the vehicle.
        :param trajectory: Planned trajectory.
        :return: Optimal acceleration and steering angle.
        """

        if trajectory.is_trajectory_empty():
            return 0.0, 0.0  # No control if trajectory is empty
        
        self.horizon = min(self.init_horizon, trajectory.trajectory_length)
        # Extract reference trajectory
        ref_x = np.array([state.pos.x for state in trajectory.trajectory[:self.horizon]])
        ref_y = np.array([state.pos.y for state in trajectory.trajectory[:self.horizon]])
        ref_v = np.array([state.vel.x for state in trajectory.trajectory[:self.horizon]])

        # Initial control inputs (zero acceleration and steering)
        u0 = np.zeros((self.horizon, 2)).flatten()

        # Bounds on acceleration and steering
        bounds = [
            (-self.max_acceleration, self.max_acceleration),
            (-self.max_steering, self.max_steering),
        ] * self.horizon

        # Solve optimization
        result = minimize(
            self.cost_function,
            u0,
            args=(
                np.array(
                    [
                        current_state.pos.x,
                        current_state.pos.y,
                        current_state.angles.heading,
                        current_state.vel.x,
                    ]
                ),
                ref_x,
                ref_y,
                ref_v,
            ),
            bounds=bounds,
            method="SLSQP"
        )

        if not result.success:
            return 0.0, 0.0  # Default if solver fails

        optimal_control = result.x.reshape(self.horizon, 2)
        return optimal_control[0, 0], optimal_control[0, 1]  # Return first step of acceleration and steering
