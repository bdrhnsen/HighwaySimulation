"""Tests for reward calculation behavior."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.rewards.rewardCalculator import RewardCalculator
from highway_simulation.scripts.util.config import default_config
from highway_simulation.scripts.vehicle.vehicle import Vehicle

class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.config = default_config
        LaneManager.set_config(self.config)
        RewardCalculator.set_config(self.config)
        Vehicle.set_config(self.config)
        self.lane_manager = LaneManager()
        self.reward_calculator = RewardCalculator(self.lane_manager)
        self.reward_calculator.effective_sim_time = 9999999999

    def add_ego_vehicle(self, x, lane, speed):
        ego_vehicle = Vehicle(x=x, lane=lane, speed=speed, v_max=speed, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

    def add_vehicle(self, x, lane, speed):
        vehicle = Vehicle(x=x, lane=lane, speed=speed, v_max=speed)
        self.lane_manager.add_vehicle(vehicle)
        return vehicle

    def clean_vehicles(self):
        self.lane_manager.remove_all_vehicles()
    def test_reward_high_speed(self):
        """Test reward when ego vehicle is at high speed and in the right lane."""
        self.add_ego_vehicle(x=100, lane=2, speed=self.config.max_rewardable_vel*36/10) # max reward vel
        reward, _ = self.reward_calculator.calculate(previous_ego=self.lane_manager.ego_vehicle, bad_action=False)
        self.assertAlmostEqual(round(reward,2), 0.7)
        self.clean_vehicles()
    def test_collision_penalty(self):
        """Test penalty when collision occurs."""
        self.add_ego_vehicle(x=100, lane=1, speed=20)
        self.add_vehicle(x=101, lane=1, speed=15)  # Close enough for collision
        reward, done = self.reward_calculator.calculate(self.lane_manager.ego_vehicle, bad_action = False)
        self.assertEqual(reward, -10)
        self.assertTrue(done)
        self.clean_vehicles()
    def test_near_collision_penalty(self):
        """Test penalty when near collisoin occurs."""
        # near collision due to low distance difference. 
        self.add_ego_vehicle(x=100, lane=1, speed=100)
        self.add_vehicle(x=109, lane=1, speed=99)  # Close enough for near collision
        reward, done = self.reward_calculator.calculate(self.lane_manager.ego_vehicle, bad_action = False)
        self.assertEqual(round(reward), -9)
        self.assertFalse(done)
        self.clean_vehicles()

        # near collision due to low ttc
        self.add_ego_vehicle(x=100, lane=1, speed=self.config.max_rewardable_vel*36/10)
        self.add_vehicle(x=120, lane=1, speed=60)
        reward, done = self.reward_calculator.calculate(self.lane_manager.ego_vehicle, bad_action = False)
        self.assertEqual(round(reward), -9)
        self.assertFalse(done)
        self.clean_vehicles()  
    """def test_near_collision_penalty_back_vehicle(self):
        
        # near collision due to low distance difference. 
        self.add_ego_vehicle(x=100, lane=1, speed=100)
        self.add_vehicle(x=91, lane=1, speed=101)  # Close enough for near collision
        reward, done = self.reward_calculator.calculate(self.lane_manager.ego_vehicle, bad_action = False)
        self.assertEqual(round(reward), -4)
        self.assertFalse(done)
        self.clean_vehicles()

        # near collision due to low ttc
        self.add_ego_vehicle(x=120, lane=1, speed=60)
        self.add_vehicle(x=100, lane=1, speed=self.config.max_rewardable_vel*36/10)
        reward, done = self.reward_calculator.calculate(self.lane_manager.ego_vehicle, bad_action = False)
        self.assertEqual(round(reward), -5)
        self.assertFalse(done)  """

   
if __name__ == "__main__":
    unittest.main()
