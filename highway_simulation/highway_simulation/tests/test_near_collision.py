"""Tests for near-collision risk scoring."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.rewards.rewardCalculator import RewardCalculator
from highway_simulation.scripts.util.config import default_config
from highway_simulation.scripts.vehicle.vehicle import Vehicle

class TestNearCollisionRisk(unittest.TestCase):

    def setUp(self):
        self.config = default_config
        Vehicle.set_config(self.config)
        LaneManager.set_config(self.config)
        RewardCalculator.set_config(self.config)
        self.lane_manager = LaneManager()
        self.reward_calculator = RewardCalculator(self.lane_manager)

    def test_no_nearby_vehicles(self):
        """Test when there are no nearby vehicles, the risk score should be 0.0."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertEqual(risk_score, 0.0)
        self.lane_manager.remove_all_vehicles()
    def test_front_vehicle_10m_away(self):
        """Test with near vehicle same speed, risk high"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=108, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 1)
        self.lane_manager.remove_all_vehicles()

    def test_front_vehicle_20m_away(self):
        """Test with 20m away vehicle same speed, risk medium"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=120, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 0.4)
        self.lane_manager.remove_all_vehicles()
    def test_front_vehicle_30m_away(self):
        """Test with 20m away vehicle same speed, risk lower"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=130, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 0.3)
        self.lane_manager.remove_all_vehicles()


    def test_front_vehicle_relative_speed_10m_away(self):
        """Same test scenerios but this time front vehicle has different speed"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=110, lane=1, speed=110, v_max=110)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 0.6)
        self.lane_manager.remove_all_vehicles()

        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=110, lane=1, speed=90, v_max=90)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 1)
        self.lane_manager.remove_all_vehicles()

    def test_front_vehicle_relative_speed_20m_away(self):
        """Same test scenerios but this time front vehicle has different speed"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=120, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 0.4)
        self.lane_manager.remove_all_vehicles()
        
        
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=120, lane=1, speed=80, v_max=80)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 1)
        self.lane_manager.remove_all_vehicles()
    def test_front_vehicle_relative_speed_30m_away(self):
        """Same test scenerios but this time front vehicle has different speed"""
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=130, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 0.3)
        self.lane_manager.remove_all_vehicles()
        
        
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=130, lane=1, speed=60, v_max=60)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(round(risk_score,1), 1)
        self.lane_manager.remove_all_vehicles()

    def test_no_risk(self):
        """no risk since front car is 55 meters away which is larger than 100/2 """
        ego_vehicle = Vehicle(x=100, lane=1, speed=100, v_max=100, is_ego=True)
        front_vehicle = Vehicle(x=155, lane=1, speed=100, v_max=100)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.reward_calculator.set_ego_vehicle(ego_vehicle)

        risk_score = self.reward_calculator.calculate_near_collision_risk()
        self.assertAlmostEqual(risk_score, 0.00)
        self.lane_manager.remove_all_vehicles()


if __name__ == '__main__':
    unittest.main()
