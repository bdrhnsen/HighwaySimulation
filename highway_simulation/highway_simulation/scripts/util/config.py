"""Configuration dataclass for simulation parameters."""

from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

@dataclass
class Config:
    """Container for simulation configuration values."""

    min_vel: int  # max and min velocities for ego vehicle
    max_vel: int
    min_rewardable_vel: int
    max_rewardable_vel: int
    collision_threshold: float
    lane_change_duration: int  # how many timestamps
    num_of_vehicles: int
    road_length: float
    vehicle_width: float
    vehicle_height: float
    screen_width: int
    screen_height: int
    lane_width: float
    time_step: float
    num_lanes: int
    effective_sim_length: int
    effective_sim_time: int
    ego_drives_with_mobil: bool
    aggresive_driver: bool
    evaluation_mode: bool
    ego_vehicle_color: ClassVar[Tuple[int, int, int]] = (255, 0, 0)
    colors: ClassVar[Dict[str, Tuple[int, ...]]] = {
        "WHITE": (255, 255, 255),
        "BLUE": (0, 0, 255),
        "BLACK": (0, 0, 0),
        "GRAY": (169, 169, 169),
        "TRANSPARENT_GRAY": (169, 169, 169, 50),
    }

    def __post_init__(self) -> None:
        """Adjust rewardable velocities if the driver is aggressive."""
        if self.aggresive_driver:
            self.max_rewardable_vel = self.max_rewardable_vel + 5
            self.min_rewardable_vel = self.min_rewardable_vel + 8
            self.collision_threshold = self.collision_threshold / 2
            self.lane_change_duration = self.lane_change_duration * 0.75


# default_config is only used for tests. Never change here if you did not change tests.
default_config = Config(
    min_vel=13,
    max_vel=36,
    min_rewardable_vel=21,
    max_rewardable_vel=29,
    collision_threshold=2,
    lane_change_duration=6,
    num_of_vehicles=35,
    road_length=51000,
    vehicle_width=4.5,
    vehicle_height=2,
    screen_width=1499,
    screen_height=1000,
    lane_width=3.5,
    time_step=0.3,
    num_lanes=3,
    effective_sim_length=20000,
    effective_sim_time=120,
    aggresive_driver=False,
    ego_drives_with_mobil=False,
    evaluation_mode=False,
)
