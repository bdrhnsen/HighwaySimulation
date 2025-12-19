"""Metrics capture and reporting utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from typing import Dict, List, Union

import pandas as pd
from tabulate import tabulate

from highway_simulation.scripts.util.config import Config

@dataclass
class Metrics:
    seed: int
    num_of_vehicles: int
    avg_ego_speed: float
    num_of_lane_changes_ego: int
    wall_time_spent: float
    ego_vehicle_travelled_percentage: float
    avg_vehicle_speed: float
    avg_time_gap: float
    ttc_infinite_percentage: float
    ttc_finite_average: float
    lane_time_distribution: Dict[Union[int, str], float]
    acceleration_distribution: Dict[str, float]
    successful_run: bool
    is_aggresive: bool
    is_driven_by_mobil: bool
    
    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def print(self) -> None:
        ego_driven_by = "reinforcement learning" if not self.is_driven_by_mobil else "Mobil+IDM"
        is_aggresive = "Agressively" if self.is_aggresive else "Not agressively"
        print("\n--- Episode Summary ---")
        print(f"Seed number : {self.seed}")
        print(f"number of vehicles {self.num_of_vehicles}")
        print(f"Ego driven by {ego_driven_by} ")
        print(f"Ego is driven {is_aggresive}")
        print(f"Average Ego Speed: {self.avg_ego_speed:.2f} km/h")
        print(f"Number of lane changes ego made: {self.num_of_lane_changes_ego}")
        print(f"Wall time spent: {self.wall_time_spent:.2f} seconds")
    
        print(f"Ego vehicle completed {self.ego_vehicle_travelled_percentage * 100:.2f}% of the track")
        print(f"Average All Vehicles Speed: {self.avg_vehicle_speed:.2f} m/s")
        print("-----------------------\n")
        
        print(f"Ego vehicle drove in the leftmost lane {self.lane_time_distribution[0]:.2f}% of the time.")
        print(f"Ego vehicle drove in the middle lane {self.lane_time_distribution[1]:.2f}% of the time.")
        print(f"Ego vehicle drove in the rightmost lane {self.lane_time_distribution[2]:.2f}% of the time.")
        print("-----------------------\n")

        print(f"Ego vehicle drove with strong braking acceleration {self.acceleration_distribution['Strong Braking']:.2f}% of the time.")
        print(f"Ego vehicle drove with moderate braking acceleration {self.acceleration_distribution['Moderate Braking']:.2f}% of the time.")
        print(f"Ego vehicle drove with no acceleration {self.acceleration_distribution['No Acceleration']:.2f}% of the time.")
        print(f"Ego vehicle drove with moderate acceleration {self.acceleration_distribution['Moderate Acceleration']:.2f}% of the time.")
        print(f"Ego vehicle drove with strong acceleration {self.acceleration_distribution['Strong Acceleration']:.2f}% of the time.")

        print("-----------------------\n")
        
        print(" TTC ")
        print(f"Average Time Gap: {self.avg_time_gap:.2f} s")
        print(f"TTC was infinite in: {self.ttc_infinite_percentage:.2f}% timestamps")
        print(f"Average of finite TTC values: {self.ttc_finite_average:.2f}")
        print("-----------------------\n")

    def save(self, file_path: str = "simulation_metrics.json") -> None:
        """Save metrics to a JSON file without overwriting previous results."""

        # Convert dataclass to dictionary and add a timestamp for uniqueness
        metrics_data = asdict(self)
        metrics_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if the file exists and load existing data
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            data = []

        # Append new metrics to the list
        data.append(metrics_data)

        # Save updated list back to file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Metrics saved to {file_path}")


def read_metrics_from_json(file_path: str) -> List[Metrics]:
    """Read metrics from a JSON file and return a list of Metrics objects."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return []

    with open(file_path, "r") as file:
        data = json.load(file)
    for entry in data:
        entry.pop("timestamp", None)

    # Convert dictionary entries back to Metrics objects
    metrics_list = [Metrics(**entry) for entry in data]
    return metrics_list


def print_all_metrics(file_path: str) -> None:
    """Reads and prints all metrics stored in the JSON file."""
    metrics_list = read_metrics_from_json(file_path)

    if not metrics_list:
        print("No metrics found in the JSON file.")
        return

    for index, metrics in enumerate(metrics_list, start=1):
        print(f"\n### Metrics Entry {index} ###")
        metrics.print()


def generate_tables_by_seed(file_path: str) -> None:
    """
    Reads a JSON file and categorizes metrics into four groups based on:

    1. is_aggresive = True,  is_driven_by_mobil = True
    2. is_aggresive = True,  is_driven_by_mobil = False
    3. is_aggresive = False, is_driven_by_mobil = True
    4. is_aggresive = False, is_driven_by_mobil = False

    Generates separate tables for each seed showing avg_ego_speed, num_of_lane_changes,
    and avg_time_gap.
    """
    _ = file_path


def categorize_and_compare_metrics(file_path: str) -> None:
    """Reads metrics from JSON, categorizes them by seed, and compares configurations."""

    # Read metrics from the JSON file
    metrics_list = read_metrics_from_json(file_path)
    
    if not metrics_list:
        print("No metrics found in the JSON file.")
        return
    
    # Group metrics by seed number
    grouped_by_seed = defaultdict(list)
    for metrics in metrics_list:
        grouped_by_seed[metrics.seed].append(metrics)

    # Define configurations
    configurations = [
        (False, False),  # Not Aggressive, RL
        (False, True),   # Not Aggressive, MOBIL
        (True, False),   # Aggressive, RL
        (True, True)     # Aggressive, MOBIL
    ]
    headers = [
        "Configuration",
        "Avg Ego Speed (km/h)",
        "Ego Lane Changes",
        "Collision Rate",
        "TTC Infinite %",
        "TTC Finite Avg",
        "Time Spent in Leftmost Lane",
        "Time Spent in Middle Lane",
        "Time Spent in Rightmost Lane",
        "Strong Braking %",
        "Moderate Braking %",
        "No Acceleration %",
        "Moderate Acceleration %",
        "Strong Acceleration %",
    ]

    # Dictionary to store aggregated data for averaging
    aggregated_data = defaultdict(list)

    # Process each seed separately
    for seed, seed_metrics in grouped_by_seed.items():
        print(f"\n### Seed: {seed} ###")
        viable_seed = True
        for metric in seed_metrics:
            if not metric.successful_run:
                viable_seed = False

        if not viable_seed:
            continue

        # Create a dictionary to store results for this seed
        table_data = []

        # Iterate over each configuration and extract corresponding values
        for is_aggresive, is_driven_by_mobil in configurations:
            filtered_metrics = [
                m
                for m in seed_metrics
                if m.is_aggresive == is_aggresive
                and m.is_driven_by_mobil == is_driven_by_mobil
            ]

            # If no metrics found for this configuration, use NaN placeholders
            if filtered_metrics:
                metric = filtered_metrics[0]
                avg_ego_speed = metric.avg_ego_speed
                ttc_infinite_percentage = metric.ttc_infinite_percentage
                ttc_finite_average = metric.ttc_finite_average
                collision_rate = 0
                ego_lane_changes = metric.num_of_lane_changes_ego
                time_spent_in_leftmost = metric.lane_time_distribution["0"]
                time_spent_in_middle = metric.lane_time_distribution["1"]
                time_spent_in_rightmost = metric.lane_time_distribution["2"]
                acceleration_distribution = metric.acceleration_distribution
            else:
                avg_ego_speed = float("nan")
                ttc_infinite_percentage = float("nan")
                ttc_finite_average = float("nan")
                collision_rate = 0
                ego_lane_changes = float("nan")
                time_spent_in_leftmost = float("nan")
                time_spent_in_middle = float("nan")
                time_spent_in_rightmost = float("nan")
                acceleration_distribution = {
                    "Strong Braking": float("nan"),
                    "Moderate Braking": float("nan"),
                    "No Acceleration": float("nan"),
                    "Moderate Acceleration": float("nan"),
                    "Strong Acceleration": float("nan"),
                }

            category = "RL" if not is_driven_by_mobil else "MOBIL+IDM"
            aggresive = " Aggressive" if is_aggresive else " Not Aggressive"
            row_name = category + aggresive

            # Append row data
            table_data.append({
                "name": row_name,
                "Avg Ego Speed (km/h)": round(avg_ego_speed, 2),
                "Collision Rate": 0,
                "Ego Lane Changes": ego_lane_changes,
                "TTC Infinite %": round(ttc_infinite_percentage, 2),
                "TTC Finite Avg": round(ttc_finite_average, 2),
                "Time Spent in Leftmost Lane": time_spent_in_leftmost,
                "Time Spent in Middle Lane": time_spent_in_middle,
                "Time Spent in Rightmost Lane": time_spent_in_rightmost,
                "Strong Braking %": acceleration_distribution["Strong Braking"],
                "Moderate Braking %": acceleration_distribution["Moderate Braking"],
                "No Acceleration %": acceleration_distribution["No Acceleration"],
                "Moderate Acceleration %": acceleration_distribution["Moderate Acceleration"],
                "Strong Acceleration %": acceleration_distribution["Strong Acceleration"],
            })

            # Store data for averaging
            aggregated_data[row_name].append({
                "Avg Ego Speed (km/h)": avg_ego_speed,
                "Ego Lane Changes": ego_lane_changes,
                "TTC Infinite %": ttc_infinite_percentage,
                "TTC Finite Avg": ttc_finite_average,
                "Time Spent in Leftmost Lane": time_spent_in_leftmost,
                "Time Spent in Middle Lane": time_spent_in_middle,
                "Time Spent in Rightmost Lane": time_spent_in_rightmost,
                "Strong Braking %": acceleration_distribution["Strong Braking"],
                "Moderate Braking %": acceleration_distribution["Moderate Braking"],
                "No Acceleration %": acceleration_distribution["No Acceleration"],
                "Moderate Acceleration %": acceleration_distribution["Moderate Acceleration"],
                "Strong Acceleration %": acceleration_distribution["Strong Acceleration"],
            })

        # Convert to DataFrame and display
        table = tabulate(
            pd.DataFrame(table_data), headers=headers, tablefmt="grid"
        )
        print(table)

    # Calculate and display averaged values across all seeds
    print("\n### Averaged Values Across All Seeds ###")
    averaged_table_data = []

    for config, data_list in aggregated_data.items():
        avg_ego_speed = sum(d["Avg Ego Speed (km/h)"] for d in data_list) / len(data_list)
        avg_ttc_infinite = sum(d["TTC Infinite %"] for d in data_list) / len(data_list)
        avg_ttc_finite = sum(d["TTC Finite Avg"] for d in data_list) / len(data_list)
        avg_leftmost = sum(d["Time Spent in Leftmost Lane"] for d in data_list) / len(data_list)
        avg_middle = sum(d["Time Spent in Middle Lane"] for d in data_list) / len(data_list)
        avg_rightmost = sum(d["Time Spent in Rightmost Lane"] for d in data_list) / len(data_list)
        avg_lane_change = sum(d["Ego Lane Changes"] for d in data_list) / len(data_list)
 
        averaged_table_data.append({
            "Configuration": config,
            "Avg Ego Speed (km/h)": round(avg_ego_speed, 2),
            "Ego Lane Changes": avg_lane_change,
            "Collision Rate": 0,
            "TTC Infinite %": round(avg_ttc_infinite, 2),
            "TTC Finite Avg": round(avg_ttc_finite, 2),
            "Time Spent in Leftmost Lane": round(avg_leftmost, 2),
            "Time Spent in Middle Lane": round(avg_middle, 2),
            "Time Spent in Rightmost Lane": round(avg_rightmost, 2),
            "Strong Braking %": round(
                sum(d["Strong Braking %"] for d in data_list) / len(data_list), 2
            ),
            "Moderate Braking %": round(
                sum(d["Moderate Braking %"] for d in data_list) / len(data_list), 2
            ),
            "No Acceleration %": round(
                sum(d["No Acceleration %"] for d in data_list) / len(data_list), 2
            ),
            "Moderate Acceleration %": round(
                sum(d["Moderate Acceleration %"] for d in data_list) / len(data_list), 2
            ),
            "Strong Acceleration %": round(
                sum(d["Strong Acceleration %"] for d in data_list) / len(data_list), 2
            ),
        })

    # Convert to DataFrame and display
    
    averaged_table = tabulate(
        pd.DataFrame(averaged_table_data), headers=headers, tablefmt="grid"
    )
    print(averaged_table)


def find_collision_rate(file_path: str) -> None:

    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Count total runs and failed runs for each configuration
    config_counts = {
        "Aggressive RL": {"total": 0, "collided": 0},
        "Aggressive MOBIL": {"total": 0, "collided": 0},
        "Non-Aggressive RL": {"total": 0, "collided": 0},
        "Non-Aggressive MOBIL": {"total": 0, "collided": 0},
    }

    # Process the data
    for entry in data:
        key = (
            "Aggressive RL"
            if entry["is_aggresive"] and not entry["is_driven_by_mobil"]
            else "Aggressive MOBIL"
            if entry["is_aggresive"] and entry["is_driven_by_mobil"]
            else "Non-Aggressive RL"
            if not entry["is_aggresive"] and not entry["is_driven_by_mobil"]
            else "Non-Aggressive MOBIL"
        )

        config_counts[key]["total"] += 1
        if not entry["successful_run"]:  # Collision occurred
            config_counts[key]["collided"] += 1

    # Compute percentages
    collision_percentages = {
        key: (value["collided"] / value["total"] * 100 if value["total"] > 0 else 0)
        for key, value in config_counts.items()
    }

    print(collision_percentages)
# Example usage
file_path = "simulation_metrics.json"
# categorize_and_compare_metrics(file_path)
# find_collision_rate(file_path=file_path)
# print_all_metrics(file_path)
