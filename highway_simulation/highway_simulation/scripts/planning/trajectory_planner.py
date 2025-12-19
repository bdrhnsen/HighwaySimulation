"""Trajectory generation utilities."""

from __future__ import annotations

import numpy as np

from highway_simulation.scripts.planning.state import Acc, Jerk, Pos, State, Trajectory, Vel
from highway_simulation.scripts.util.config import Config
class TrajectoryPlanner:
    config = Config

    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config
    
    def __init__(self, time_horizon: float = 3) -> None:
        """
        Initialize the trajectory planner.
        :param lane_width: Width of a single lane (meters).
        :param time_horizon: Time horizon for trajectory planning (seconds).
        :param time_step: Time step for generating waypoints (seconds).
        """
        self.time_horizon = time_horizon

    def quintic_polynomial(
        self, init_state: State, final_state: State, T: float
    ) -> Trajectory:
        """
        Generate a trajectory using a quintic polynomial.
        
        Args:
            x0 (float): Initial position.
            v0 (float): Initial velocity.
            a0 (float): Initial acceleration.
            xT (float): Final position.
            vT (float): Final velocity.
            aT (float): Final acceleration.
            T (float): Duration of the trajectory in seconds.
            num_points (int): Number of points in the trajectory.
            
        Returns:
            Trajectory: List of states
        """
        # Time powers matrix for quintic polynomial
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3],
        ])
        
        # Boundary conditions
        b_lat = np.array(
            [
                init_state.pos.y,
                init_state.vel.y,
                init_state.acc.y,
                final_state.pos.y,
                final_state.vel.y,
                final_state.acc.y,
            ]
        )
        
        # Solve for coefficients
        coeffs_lat = np.linalg.solve(A, b_lat)
        
        # Boundary conditions
        b_lon = np.array(
            [
                init_state.pos.x,
                init_state.vel.x,
                init_state.acc.x,
                final_state.pos.x,
                final_state.vel.x,
                final_state.acc.x,
            ]
        )
        
        # Solve for coefficients
        coeffs_lon = np.linalg.solve(A, b_lon)
        # Generate trajectory points
        times = np.linspace(0, T, int(T / self.config.time_step))

        # Precompute time matrices
        pos_matrix = np.vstack(
            [np.ones_like(times), times, times**2, times**3, times**4, times**5]
        ).T
        vel_matrix = np.vstack(
            [
                np.zeros_like(times),
                np.ones_like(times),
                2 * times,
                3 * times**2,
                4 * times**3,
                5 * times**4,
            ]
        ).T
        acc_matrix = np.vstack(
            [
                np.zeros_like(times),
                np.zeros_like(times),
                2 * np.ones_like(times),
                6 * times,
                12 * times**2,
                20 * times**3,
            ]
        ).T
        jerk_matrix = np.vstack(
            [
                np.zeros_like(times),
                np.zeros_like(times),
                np.zeros_like(times),
                6 * np.ones_like(times),
                24 * times,
                60 * times**2,
            ]
        ).T

        lon_pos = pos_matrix @ coeffs_lon
        lat_pos = pos_matrix @ coeffs_lat

        lon_vel = vel_matrix @ coeffs_lon
        lat_vel = vel_matrix @ coeffs_lat
        
        lon_acc = acc_matrix @ coeffs_lon
        lat_acc = acc_matrix @ coeffs_lat

        lon_jerk = jerk_matrix @ coeffs_lon
        lat_jerk = jerk_matrix @ coeffs_lat
        
        trajectory = Trajectory()
        for i in range(len(times)):
            trajectory.trajectory.append(
                State(
                    Pos(lon_pos[i], lat_pos[i]),
                    Vel(lon_vel[i], lat_vel[i]),
                    Acc(lon_acc[i], lat_acc[i]),
                    Jerk(lon_jerk[i], lat_jerk[i]),
                )
            )
        
        return trajectory

    def plan_between_points(self, init_state: State, final_state: State) -> Trajectory:

        trajectory = self.quintic_polynomial(init_state, final_state, T=3)

        return trajectory
