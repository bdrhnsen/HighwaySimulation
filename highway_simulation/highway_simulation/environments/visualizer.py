"""Observation visualization utilities."""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class ObservationVisualizer:
    """Visualize normalized observations in the simulation space."""

    def __init__(self, config) -> None:
        self.config = config
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.config.road_length // 10, self.config.road_length // 10)
        self.ax.set_ylim(0, self.config.num_lanes * self.config.lane_width)
        self.ax.set_aspect('equal')
        self.rectangles = []

    def denormalize(self, normalized_obs: np.ndarray) -> np.ndarray:
        """Reverse the normalization."""
        V = 5  # Number of vehicles
        F = 3  # Features per vehicle (x, y, v_x)
        obs = normalized_obs.reshape((V, F))
        
        x_max = 400  # Absolute value of relative coordinates
        y_mean = ((self.config.num_lanes - 1) * self.config.lane_width) / 2
        v_mean = self.config.max_vel / 2

        denormalized_obs = []
        for x_norm, y_norm, v_norm in obs:
            x = x_norm * (2 * x_max) - x_max
            y = y_norm * (2 * y_mean)
            v = v_norm * (2 * v_mean)
            denormalized_obs.append((x, y, v))

        return np.array(denormalized_obs)

    def plot_observation(self, observation: np.ndarray) -> None:
        """Plot the denormalized observation as rectangles."""
        denormalized_obs = self.denormalize(observation)
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, self.config.num_lanes * self.config.lane_width)
        self.ax.set_title("RL Agent's Observations")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        # Draw lanes
        for i in range(1, self.config.num_lanes):
            self.ax.axhline(i * self.config.lane_width, color="gray", linestyle="--")

        # Draw ego vehicle and surrounding vehicles
        for i, (x, y, v) in enumerate(denormalized_obs):
            color = "red" if i == 0 else "blue"  # Ego vehicle is red
            rect = patches.Rectangle(
                (x - self.config.vehicle_width / 2, y - self.config.vehicle_height / 2),
                self.config.vehicle_width,
                self.config.vehicle_height,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=0.7,
            )
            self.ax.add_patch(rect)
            self.ax.text(x, y, f"v={v:.1f}", color="black", fontsize=8, ha="center", va="center")

        plt.pause(0.01)

    def close(self) -> None:
        plt.close(self.fig)
