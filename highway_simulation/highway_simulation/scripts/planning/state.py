"""Trajectory data structures and plotting helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from highway_simulation.scripts.util.config import Config
@dataclass
class Vector2D:
    x: float = 0.0
    y: float = 0.0

@dataclass
class Pos(Vector2D):
    pass

@dataclass
class Vel(Vector2D):
    pass

@dataclass
class Acc(Vector2D):
    pass

@dataclass
class Jerk(Vector2D):
    pass

@dataclass
class Angles:
    heading: float = 0.0
    steering_angle: float = 0.0


@dataclass
class State:
    pos: Pos = field(default_factory=Pos)
    vel: Vel = field(default_factory=Vel)
    acc: Acc = field(default_factory=Acc)
    jerk: Jerk = field(default_factory=Jerk)
    angles: Angles = field(default_factory=Angles)

    def extract(self) -> Tuple[float, float, float, float, float, float, float, float]:
        return (
            self.pos.x,
            self.pos.y,
            self.vel.x,
            self.vel.y,
            self.acc.x,
            self.acc.y,
            self.angles.heading,
            self.angles.steering_angle,
        )

@dataclass
class Trajectory:
    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config
    trajectory: List[State] = field(default_factory=list)
    acc_category_counts: Dict[str, int] = field(default_factory=dict)

    def use_next_state(self) -> State:
        state = self.trajectory[0]
        self.trajectory.pop(0)
        return state

    def update_trajectory(self, input_trajectory: List[State]) -> None:
        self.trajectory = input_trajectory
    
    def is_trajectory_empty(self) -> bool:
        return len(self.trajectory) == 0
    
    @property
    def trajectory_length(self) -> int:
        return len(self.trajectory)
    
    @property
    def last_state(self) -> Optional[State]:
        return self.trajectory[-1] if self.trajectory else None
    
    @property  
    def return_pos(self) -> List[Pos]:
        return [state.pos for state in self.trajectory]
        
    def measure_quality_of_trajectory(
        self, max_allowed_jerk: float, max_allowed_acc: float, time_step: float
    ) -> Dict[str, float]:
        """
        Assess the quality of the trajectory by calculating the maximum jerk and acceleration values.
        
        :param max_allowed_jerk: Maximum allowable jerk.
        :param max_allowed_acc: Maximum allowable acceleration.
        :param time_step: Time interval between trajectory points.
        :return: A dictionary containing the assessment results.
        """
        max_acc = 0
        max_jerk = 0
        jerk_violations = 0
        acc_violations = 0

        for i in range(1, len(self.trajectory)):
            curr_state = self.trajectory[i]

            # Calculate acceleration (longitudinal and lateral)
            acc_x = curr_state.acc.x
            acc_y = curr_state.acc.y
            
            jerk_x = curr_state.jerk.x
            jerk_y = curr_state.jerk.y

            acc_violations += int(acc_x > max_allowed_acc) + int(
                acc_y > max_allowed_acc
            )
            jerk_violations += int(jerk_x > max_allowed_jerk) + int(
                jerk_y > max_allowed_jerk
            )

            if abs(acc_x) > max_acc or abs(acc_y) > (max_acc):
                max_acc = abs(acc_x) if abs(acc_x) > abs(acc_y) else abs(acc_y)

            if abs(jerk_x) > max_jerk or abs(jerk_y) > (max_jerk):
                max_jerk = abs(jerk_x) if abs(jerk_x) > abs(jerk_y) else abs(jerk_y)
        return {
            "max_acc": max_acc,
            "max_jerk": max_jerk,
            "acc_violations": acc_violations,
            "jerk_violations": jerk_violations,
            "is_trajectory_valid": acc_violations == 0 and jerk_violations == 0
        }
    
    @property
    def return_avg_vehicle_speed(self) -> float:
        velocities_x_kmh = [state.vel.x * 3.6 for state in self.trajectory]
        avg_ego_speed = sum(velocities_x_kmh) / len(velocities_x_kmh) if velocities_x_kmh else 0
        return avg_ego_speed
    @property
    
    def return_acceleration_distribution(self) -> Dict[str, float]:

        accelerations_x = [state.acc.x for state in self.trajectory]
        total_acc_samples = len(accelerations_x) 
        
        if total_acc_samples == 0:  # Avoid division by zero
            return {key: 0.0 for key in ["Strong Braking", "Moderate Braking", "No Acceleration", "Moderate Acceleration", "Strong Acceleration"]}

        categories = {
            "Strong Braking": lambda a: a < -2.5,
            "Moderate Braking": lambda a: -2.5 <= a < -0.1,
            "No Acceleration": lambda a: -0.1 <= a < 0.1,
            "Moderate Acceleration": lambda a: 0.1 <= a < 1.5,
            "Strong Acceleration": lambda a: a >= 1.5,
            }
        if self.config.ego_drives_with_mobil:
            acc_category_counts = {key: sum(1 for a in accelerations_x if condition(a)) for key, condition in categories.items()}
        else:
            mapping = {
                -4: "Strong Braking",
                -2: "Moderate Braking",
                0: "No Acceleration",
                2: "Moderate Acceleration",
                4: "Strong Acceleration" # does not exist in RL model
            }
            # Count occurrences of discrete accelerations
            acceleration_counts = Counter(accelerations_x)
            acc_category_counts = {key: sum(acceleration_counts[a] for a in mapping if mapping[a] == key) for key in categories.keys()}
        
        acc_category_percentages = {
            key: (count / total_acc_samples) * 100 for key, count in acc_category_counts.items()
        }
        return acc_category_percentages
    
    def plot_trajectory(
        self, plot_heading: bool = False, plot_history_of_data: bool = False
    ) -> None:
        """
        Plot the trajectory showing position, velocity, acceleration, and jerk
        in both longitudinal (x) and lateral (y) directions.
        """
        if not self.trajectory:
            print("No trajectory data to plot.")
            return

        # Extract data(f"Acceleration Category Distribution ({'MOBIL' 
        times = [i * 0.1 for i in range(len(self.trajectory))]  # Assuming a time step of 0.1 seconds
        positions_x = [state.pos.x for state in self.trajectory]
        positions_y = [state.pos.y for state in self.trajectory]
        velocities_x = [state.vel.x for state in self.trajectory]
        velocities_y = [state.vel.y for state in self.trajectory]
        accelerations_x = [state.acc.x for state in self.trajectory]
        accelerations_y = [state.acc.y for state in self.trajectory]
        jerks_x = [state.jerk.x for state in self.trajectory]
        jerks_y = [state.jerk.y for state in self.trajectory]


        # Create subplots
        if plot_heading:
            fig, axes = plt.subplots(5, 2, figsize=(12, 16))
        if plot_history_of_data:
            fig, axes = plt.subplots(2, 2, figsize=(12, 16))
            fig.suptitle("Trajectory Visualization", fontsize=16)
            
            # Position plots
            axes[0, 0].plot(times, positions_x, label="x-position")
            axes[0, 0].set_title("Position (Longitudinal - x)")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Position (m)")
            axes[0, 0].grid()
            axes[0, 0].legend()
            
            velocities_x_kmh = [v * 3.6 for v in velocities_x]
            lane_indices = [round(y / self.config.lane_width) for y in positions_y]
            unique_lanes = sorted(set(lane_indices))  # Get distinct lane numbers
            axes[1, 1].step(times, lane_indices, where="post", label="Lane", color="orange", linewidth=2)

            # Add horizontal lane lines for clarity
            for lane in unique_lanes:
                axes[0, 0].axhline(y=lane, color="gray", linestyle="--", linewidth=0.5)

            axes[1, 1].set_title("Lane Changes")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Lane Index")
            axes[1, 1].set_yticks(unique_lanes)  # Set y-axis ticks to show discrete lanes
            axes[1, 1].grid()
            axes[1, 1].legend()

            # Velocity plots
            axes[0, 1].plot(times, velocities_x_kmh, label="longitudinal velocity")
            axes[0, 1].set_title("Velocity (Longitudinal - x)")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Velocity (km/h)")
            axes[0, 1].grid()
            axes[0, 1].legend()
            
            if self.config.ego_drives_with_mobil:
                # Use normal plotting since acceleration values are not discrete
                
                axes[1, 0].plot(times, accelerations_x, color="red", label="Longitudinal Acceleration", linewidth=1.5)
                
                axes[1, 0].set_title("Acceleration (Longitudinal - x) [MOBIL Mode]", fontsize=12)
                axes[1, 0].set_xlabel("Time (s)", fontsize=10)
                axes[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=10)
                axes[1, 0].grid()
                axes[1, 0].legend()
            else:
                # longitudinal acc
                axes[1, 0].scatter(times, accelerations_x, color="red", label="Longitudinal Acceleration", s=10, alpha=0.7)

                # Set y-axis ticks to only discrete acceleration values
                axes[1, 0].set_yticks([-4, -2, 0, 2])  

                # Add horizontal reference lines
                for acc in [-4, -2, 0, 2]:
                    axes[1, 0].axhline(y=acc, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                axes[1, 0].set_title("Acceleration (Longitudinal - x)", fontsize=12)
                axes[1, 0].set_xlabel("Time (s)", fontsize=10)
                axes[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=10)
                axes[1, 0].grid()

            # Move legend to top left
            axes[1, 0].legend(loc="center", fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            return
        else:
            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
        fig.suptitle("Trajectory Visualization", fontsize=16)

        # Position plots
        axes[0, 0].plot(times, positions_x, label="x-position")
        axes[0, 0].set_title("Position (Longitudinal - x)")
        #axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Position (m)")
        axes[0, 0].grid()
        axes[0, 0].legend()

        axes[0, 1].plot(times, positions_y, label="y-position", color="orange")
        axes[0, 1].set_title("Position (Lateral - y)")
        #axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Position (m)")
        axes[0, 1].grid()
        axes[0, 1].legend()

        # Velocity plots
        axes[1, 0].plot(times, velocities_x, label="x-velocity")
        axes[1, 0].set_title("Velocity (Longitudinal - x)")
        #axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Velocity (m/s)")
        axes[1, 0].grid()
        axes[1, 0].legend()

        axes[1, 1].plot(times, velocities_y, label="y-velocity", color="orange")
        axes[1, 1].set_title("Velocity (Lateral - y)")
        #axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Velocity (m/s)")
        axes[1, 1].grid()
        axes[1, 1].legend()

        # Acceleration plots
        axes[2, 0].plot(times, accelerations_x, label="x-acceleration")
        axes[2, 0].set_title("Acceleration (Longitudinal - x)")
        #axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Acceleration (m/s²)")
        axes[2, 0].grid()
        axes[2, 0].legend()

        axes[2, 1].plot(times, accelerations_y, label="y-acceleration", color="orange")
        axes[2, 1].set_title("Acceleration (Lateral - y)")
        #axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_ylabel("Acceleration (m/s²)")
        axes[2, 1].grid()
        axes[2, 1].legend()

        # Jerk plots
        axes[3, 0].plot(times, jerks_x, label="x-jerk")
        axes[3, 0].set_title("Jerk (Longitudinal - x)")
        #axes[3, 0].set_xlabel("Time (s)")
        axes[3, 0].set_ylabel("Jerk (m/s³)")
        axes[3, 0].grid()
        axes[3, 0].legend()

        axes[3, 1].plot(times, jerks_y, label="y-jerk", color="orange")
        axes[3, 1].set_title("Jerk (Lateral - y)")
        #axes[3, 1].set_xlabel("Time (s)")
        axes[3, 1].set_ylabel("Jerk (m/s³)")
        axes[3, 1].grid()
        axes[3, 1].legend()

        if plot_heading:
            headings = [state.angles.heading for state in self.trajectory]
            axes[4, 0].plot(times, headings, label="heading")
            axes[4, 0].set_title("Heading (theta)")
            axes[4, 0].set_xlabel("Time (s)")
            axes[4, 0].set_ylabel("Heading (rad)")
            axes[4, 0].grid()
            axes[4, 0].legend()
            steering_angles = [state.angles.steering_angle for state in self.trajectory]
            axes[4, 1].plot(times, steering_angles, label="steering angle")
            axes[4, 1].set_title("Steering Angle")
            axes[4, 1].set_xlabel("Time (s)")
            axes[4, 1].set_ylabel("(rad)")
            axes[4, 1].grid()
            axes[4, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
