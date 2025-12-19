"""Action enum for the highway environment."""

from enum import Enum

class Action(Enum):
    """Discrete actions available to the agent."""

    NO_ACTION = 0
    CHANGE_LANE_RIGHT = 1
    CHANGE_LANE_LEFT = 2
    ACCELERATE = 3
    DECELERATE = 4
    EMERGENCY_BRAKE = 5
