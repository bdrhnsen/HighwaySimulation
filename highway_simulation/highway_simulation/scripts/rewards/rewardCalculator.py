"""Reward calculation logic for the highway environment."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.rewards.near_collision import calculate_continuous_risk
from highway_simulation.scripts.util.config import Config
from highway_simulation.scripts.vehicle.vehicle import Vehicle
class RewardCalculator:
    config = Config
    
    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def __init__(self, lane_manager: LaneManager) -> None:
        self.lane_manager = lane_manager
        self.effective_sim_time = self.config.effective_sim_time
        self.sim_start_time = time.time()
        self.counter = 0
        self.done_in_3_steps = False
        self.ttc_metric = 0

    def set_ego_vehicle(self, ego_vehicle_sim: Vehicle) -> None:
        self.ego_vehicle = ego_vehicle_sim

    def calculate(self, previous_ego, bad_action):
        ## NEW REWARD FUNCTION INSPIRED BY FARAMA
        done = False
        reward, coll = self.normalized_reward_function()
        self.ttc_metric = self.calculate_ttc_metric()
        #reward += self.calculate_lane_penalty(previous_ego)
        #reward += self.one_step_movement_reward()
        #reward += self.bad_action(bad_action)
        #reward += self.calculate_penalty_due_to_different_speed()
        #reward += self.calculate_left_lane_penalty()
        #reward += self.calculate_penalty_for_too_slow()
        #collison_penalty, done=self.check_collision_reward()
        #reward +=collison_penalty
        if coll:
            self.done_in_3_steps = True

        if self.done_in_3_steps:
            self.counter += 1
            if self.counter == 3:
                done = True
                self.done_in_3_steps = False
                self.counter = 0

        return reward, self.is_done() or done

    def normalized_reward_function(self) -> Tuple[float, bool]:
        a = 0.1
        b = 1

        _, coll = self.check_collision_reward()
        if coll:
            print("coll")
        near_coll = self.calculate_near_collision_risk()
        

        b = 10 * near_coll  # if risk is 100% count as collision and abandon collision check.
        if (
            self.ego_vehicle.speed >= self.config.min_rewardable_vel
            and self.ego_vehicle.speed <= self.config.max_rewardable_vel
        ):
            a = 0.8
            if self.ego_vehicle.lane != 0 and not self.config.aggresive_driver:
                # no bonus for driving not on the leftmost lane for aggrresive driver
                a = 1
        
        reward = a * (self.ego_vehicle.speed - self.config.min_vel) / (
            self.config.max_vel - self.config.min_vel
        ) - b * (coll or near_coll)
        if self.config.aggresive_driver:
            mean_vel = (self.config.max_vel + self.config.min_vel) /2
            reward = a * (self.ego_vehicle.speed - mean_vel) / (
                self.config.max_vel - self.config.min_vel
            ) - b * (coll or near_coll)
            reward = np.clip(reward, -1, 1.5)
        else:    
            reward = np.clip(reward, -1, 1)
        return reward, coll
    
    def bad_action(self, bad_action) -> float:
        return -10 if bad_action else 0
        # left lane change when you are in the leftmost lane or right lane change when you are in rightmost lane

    def calculate_ttc_metric(self) -> float:
        ego_vehicle = self.ego_vehicle
        front_vehicle = self.lane_manager.find_vehicle_ahead(ego_vehicle, ego_vehicle.lane)
        
        if not front_vehicle:
            return float("inf")
    
        relative_distance = front_vehicle.x - self.lane_manager.ego_vehicle.x
        relative_speed = self.lane_manager.ego_vehicle.speed - front_vehicle.speed

        if relative_speed <= 0:
            return float("inf")
        
        ttc = relative_distance / relative_speed
        if ttc > 70:
            return float("inf")
        return ttc


    def calculate_near_collision_risk(self) -> float:
        """
        Calculate the risk of near-collision based on distance, relative speed, and TTC.
        Returns a risk score between 0 and 1, where 1 is a high risk of collision.
        """    
        if self.lane_manager.lane_change_in_progress:
            return 0.0

        ego_vehicle = self.ego_vehicle
        front_vehicle = self.lane_manager.find_vehicle_ahead(ego_vehicle, ego_vehicle.lane)

        # Parameters for risk calculation
        max_safe_distance = self.lane_manager.ego_vehicle.speed * 36 / 10 / 2
        min_safe_distance = 5.0  # Minimum safe distance in meters
        min_ttc = 2.0  # Minimum time-to-collision threshold in seconds

        # Initialize risk score
        front_risk = 0
        back_risk = 0
        
        # Check front vehicle risk
        if front_vehicle:
            distance = front_vehicle.relative_x - ego_vehicle.relative_x - self.config.vehicle_width
            relative_speed = ego_vehicle.speed - front_vehicle.speed
            position_difference_at_ttc = distance - relative_speed * min_ttc
            front_risk = calculate_continuous_risk(
                position_difference_at_ttc,
                min_distance=5,
                max_distance=self.ego_vehicle.speed * 36 / 10 / 2,
            )
            if self.config.aggresive_driver:
                front_risk = calculate_continuous_risk(
                    position_difference_at_ttc,
                    min_distance=2,
                    max_distance=self.ego_vehicle.speed * 36 / 10 / 5,
                )
            
        return max(front_risk, back_risk)


    def one_step_movement_reward(self) -> float:
        if self.ego_vehicle.speed > 21 and self.ego_vehicle.speed < 25:
            reward = self.ego_vehicle.speed / 10.0

            if self.ego_vehicle.lane != 0:
                reward=reward*1.3

            return reward
        else:
            return -0.1

    def calculate_penalty_due_to_different_speed(self) -> float:
        # penalty due to wanted speed and ego speed is different
        return -abs(self.ego_vehicle.speed - self.ego_vehicle.v_max) / 5.0

    def calculate_left_lane_penalty(self) -> float:
        return -0.5 if self.ego_vehicle.lane == 0 else 0
  
    def calculate_penalty_for_too_slow(self) -> float:
        return -10 if self.ego_vehicle.v_max < 10 else 0

    
    def calculate_lane_penalty(self, previous_ego) -> float:
        if self.ego_vehicle.lane != previous_ego[2]:  # punish lane change
            return -1  # temp
        else:
            return 0

    def is_done(self) -> bool:
        return (
            self.ego_vehicle.x > self.config.effective_sim_length
            or self.ego_vehicle.speed < self.config.min_vel
            or (time.time() - self.sim_start_time) > self.effective_sim_time
        )

    
    def check_collision_reward(self) -> Tuple[float, bool]:
        if not self.lane_manager.lane_change_in_progress:
            front, back = self.lane_manager.find_front_back_vehicles(
                self.lane_manager.lanes[self.lane_manager.ego_vehicle.lane]
            )
            vehicles = [v for v in (front, back) if v is not None]
            for vehicle in vehicles:
                if (
                    not vehicle.is_ego
                    and abs(vehicle.relative_x - self.lane_manager.ego_vehicle.relative_x)
                    < (self.config.collision_threshold + self.config.vehicle_width)
                ):
                    return -1000, True
                    
        else:
            front, back = self.lane_manager.find_front_back_vehicles(
                self.lane_manager.lanes[self.lane_manager.ego_vehicle.lane]
            )
            front2, back2 = self.lane_manager.find_front_back_vehicles(
                self.lane_manager.lanes[self.lane_manager.ego_vehicle.target_lane]
            )
            target_list = [v for v in (front, back, front2, back2) if v is not None]
            for vehicle in target_list:
                if (
                    not vehicle.is_ego
                    and abs(vehicle.relative_x - self.lane_manager.ego_vehicle.relative_x)
                    < (self.config.collision_threshold + self.config.vehicle_width)
                    and abs(vehicle.y - self.lane_manager.ego_vehicle.y)
                    < 0.75 * self.config.vehicle_height
                ):
                    return -1000, True
        return 0, False
