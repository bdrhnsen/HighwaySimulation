"""Lane data model and statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import List

from highway_simulation.scripts.vehicle.vehicle import Vehicle

@dataclass
class Lane:
    id: int
    road_length: float
    vehicle_width: float
    how_far_to_ego: int = 99
    vehicles: List[Vehicle] = field(default_factory=list)

    @property
    def num_vehicles(self) -> int:
        """Return the number of vehicles in the lane."""
        return len(self.vehicles)

    @property
    def avg_speed(self) -> float:
        """Return the average speed of vehicles in the lane."""
        if self.vehicles:
            return sum(v.speed for v in self.vehicles) / len(self.vehicles)
        return 0.0

    @property
    def avg_time_gap(self) -> float:
        """Return the average time gap between longitudinally adjacent cars."""
        if len(self.vehicles) < 2:
            return 0
        vehicles_sorted = sorted(self.vehicles, key=lambda v: v.x)
        time_gaps = []
        for i in range(len(vehicles_sorted) - 1):
            current_vehicle = vehicles_sorted[i]
            next_vehicle = vehicles_sorted[i + 1]
            time_gaps.append((next_vehicle.x - current_vehicle.x) / current_vehicle.speed)
        return statistics.fmean(time_gaps)
