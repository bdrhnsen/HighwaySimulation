"""Utility helpers for state encoding and mappings."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def find_mapping(longitudinal_dist: float, speed_diff: float) -> Tuple[int, int]:
    """Map longitudinal distance and speed difference to discrete bins."""
    # already absolute values
    if longitudinal_dist > 600 or speed_diff > 40:
        return -1, -1

    # Define boundaries for position and speed differences
    pos_diff_boundaries = [200, 400, 600]
    vel_diff_boundaries = [20, 40]

    # Map longitudinal_dist to pos_dif
    pos_dif = next((i for i, boundary in enumerate(pos_diff_boundaries) if longitudinal_dist < boundary), 2)

    # Map speed_diff to vel_dif
    vel_dif = next((i for i, boundary in enumerate(vel_diff_boundaries) if speed_diff < boundary), 1)

    return pos_dif, vel_dif


def find_mapping_for_ego(ego_vehicle) -> int:
    """Map ego vehicle speed to a discrete bin."""
    # Define speed boundaries for ego vehicle
    ego_speed_boundaries = [10, 20, 30, 40, 50]

    # Map ego_vehicle speed to the corresponding index
    mapped = next((i for i, boundary in enumerate(ego_speed_boundaries) if ego_vehicle.speed < boundary), 4)

    return mapped

"""
ego_vel 0

left_lane_front_vehicle -> pos_diff, vel_diff 1, 2 
left_lane_back_vehicle -> pos_diff, vel_diff 3, 4

current_lane_front_vehicle -> pos_diff, vel_diff 5, 6
current_lane_back_vehicle -> pos_diff, vel_diff 7, 8

right_lane_front_vehicle -> pos_diff, vel_diff 9, 10
right_lane_back_vehicle -> pos_diff, vel_diff 11, 12

so state is an 13 element flattened array. Which will be converted to a single value later.
if the vehicle on front and back does not exist or it is too far value will be -1

vel_dif can be [0,20] [20,40] 2 possible values + 1 for not existant 3 
pos_dif can be [0,200], [200,400],  [400, 600] so 3 possible values + 1 for not existant 4

ego_vel can be [0,50] and do not need to hold ego_pos. so 5 possible values for ego_vel  
"""
radices = [5, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3]


def array_to_index(array: Sequence[int], radices: Sequence[int] = radices) -> int:
    """Convert a mixed-radix array to a single index."""
    index = 0
    product = 1  # To store the product of previous radices
    for i in range(len(array)):
        index += array[i] * product
        product *= radices[i]  # Update product for the next position
    return index


def index_to_array(index: int, radices: Sequence[int] = radices) -> List[int]:
    """Convert a single index back to its mixed-radix representation."""
    array: List[int] = []
    for i in range(len(radices)):
        array.append(index % radices[i])
        index //= radices[i]  # Update index for the next value
    return array


"""
array = [4,3,2,3,2,3,2,3,2,3,2,3,2]
idx= array_to_index(array, radices)
converted_array = index_to_array(idx, radices)
print(idx)
print(converted_array)
print(converted_array==array)

array = [0,2,2,2,2,1,2,0,2,0,2,1,0]
idx= array_to_index(array, radices)
converted_array = index_to_array(idx, radices)
print(idx)
print(converted_array)
print(converted_array==array)

array = [0]*13
idx= array_to_index(array, radices)
converted_array = index_to_array(idx, radices)
print(idx)
print(converted_array)
print(converted_array==array)
"""
