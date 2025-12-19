"""Vehicle model parameter helpers."""

from dataclasses import dataclass
import random


@dataclass
class IDM_PARAM:
    a_max: float
    s0: float  # min dist to ahead vehicle
    T: float  # safe time headway
    b: float  # breaking decelaration
    delta: float  # accel exponent


@dataclass
class MOBIL_PARAM:
    politeness: float
    a_thr: float
    b_safe: float


def random_color() -> tuple[int, int, int]:
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
