"""Helpers for generating initial vehicle layouts."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from highway_simulation.scripts.util.config import Config
from highway_simulation.scripts.vehicle.vehicle import Vehicle


class HighwayHelper:
    def __init__(self, config: Config) -> None:
        self.config = config

    def is_position_available(
        self, position: float, lane: int, min_distance: float = 20
    ) -> bool:
        for vehicle in self.vehicle_list:
            if vehicle.lane == lane and abs(vehicle.x - position) < min_distance:
                return False
        return True
        
    def generate_vehicle_list(
        self,
        seed: int,
        num_vehicles: int,
        no_vehicles: Optional[bool] = None,
        ego_position: float = 50,
        ego_velocity_range: Tuple[int, int] = (90, 120),
        generation_range: int = 200,
    ) -> List[Vehicle]:
        # for ego # here speeds are given in screen units. 10 units are 1 meters. 120km/h 33.33m/s so 333 in game units. User input will be in terms of kmh
        # inside the sim I will arrange values. 
        ## saniyorum ki programin bazi noktalarda takilmasinin sebebi arabalara yer bulamiyor olmasi

        random.seed(seed)
        
        self.vehicle_list = []    
        ego_vel = random.randint(*ego_velocity_range)
        if self.config.ego_drives_with_mobil:
            ego_vel = self.config.max_rewardable_vel * 36 / 10

        elif self.config.aggresive_driver:
            ego_velocity_range = (120, 130)
            ego_vel = random.randint(*ego_velocity_range)
            
        ego_lane = random.randint(1, self.config.num_lanes - 1)
        ego_vehicle = Vehicle(ego_position, ego_lane, ego_vel, v_max=ego_vel, is_ego=True)
        if self.config.ego_drives_with_mobil and self.config.aggresive_driver:
            ego_velocity_range = (120, 130)
            ego_vehicle.speed = random.randint(*ego_velocity_range) * 10 / 36

        self.vehicle_list.append(ego_vehicle)

        if no_vehicles:
            return self.vehicle_list
        
        for _ in range(num_vehicles):
            lane = random.randint(1, self.config.num_lanes - 1)  # do not generate vehicles on left lane
            position = random.randint(int(ego_position) - 3000, int(ego_position) + 2000)
            vel_boundaries = 90, 120
            velocity = random.randint(*vel_boundaries)
            
            counter = 0
            while not self.is_position_available(position, lane):
                counter += 1
                position = random.randint(int(ego_position) - 3000, int(ego_position) + 2000)
                if counter == 10: 
                    print("I can not find a place for this vehicle, I will skip it")
                    break
            self.vehicle_list.append(Vehicle(position, lane, velocity, v_max=velocity))

        sorted_vehicle_list = sorted(self.vehicle_list, key=lambda veh: veh.x)
        x_list = [vehicle.x for vehicle in sorted_vehicle_list]
        if len(np.unique(x_list)) != len(x_list):
            Exception("problem") 
        return self.vehicle_list
