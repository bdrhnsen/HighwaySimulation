"""Near-collision risk calculation utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

def calculate_continuous_risk(
    position_difference_at_ttc: float,
    min_distance: float = 2.0,
    max_distance: float = 10.0,
) -> float:
    """
    Calculate a continuous risk of collision based on the position difference at TTC with two thresholds.

    Parameters:
    - position_difference_at_ttc (float): Projected position difference at TTC.
    - min_distance (float): Minimum allowed distance below which the risk is 1.
    - max_distance (float): Maximum distance beyond which the risk is 0.

    Returns:
    - float: Continuous risk of collision (from 0 to 1).
    """
    if position_difference_at_ttc <= min_distance:
        return 1.0
    if position_difference_at_ttc >= max_distance:
        return 0.0
    # Calculate risk using a nonlinear curve between min_distance and max_distance.
    normalized_distance = (position_difference_at_ttc - min_distance) / (
        max_distance - min_distance
    )
    risk = 1.0 - (normalized_distance ** 0.4)
    return risk


def plot() -> None:
    # Define a range of position differences at TTC (from close to far distances)
    position_diffs = np.linspace(
        0, 60, 200
    )  # Distance range from -1m (overlap) to 12m (far apart)
    min_distance = 5.0  # Minimum allowed distance (high risk below this)
    max_distance = 50.0  # Maximum distance (no risk beyond this)

    # Calculate risk for each position difference
    risks = [calculate_continuous_risk(diff, min_distance, max_distance) for diff in position_diffs]

    # Plot the risk curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        position_diffs,
        risks,
        label=f"Min Distance = {min_distance} m, Max Distance = {max_distance} m",
        color="b",
    )
    plt.axvline(
        x=min_distance,
        color="red",
        linestyle="--",
        label=f"Min Distance ({min_distance} m)",
    )
    plt.axvline(
        x=max_distance,
        color="green",
        linestyle="--",
        label=f"Max Distance ({max_distance} m)",
    )
    plt.title("Continuous Collision Risk Based on Position Difference at TTC")
    plt.xlabel("Position Difference at TTC (m)")
    plt.ylabel("Collision Risk")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
