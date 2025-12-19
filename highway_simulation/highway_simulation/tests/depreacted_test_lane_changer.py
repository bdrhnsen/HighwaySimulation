"""Deprecated tests for lane change manager."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.vehicle import Vehicle
from highway_simulation.scripts.laneChanger import LaneChangeManager
from highway_simulation.scripts.util.config import Config, default_config


class TestLaneChanger(unittest.TestCase):

    def setUp(self):
        self.config = default_config
        LaneManager.set_config(self.config)
        LaneChangeManager.set_config(self.config)
        Vehicle.set_config(self.config)
        self.lane_manager = LaneManager()
        self.lane_changer = LaneChangeManager(self.lane_manager)

        # Initialize ego vehicle

    def test_initiate_lane_change_left(self):
        """Test that the ego vehicle can initiate a left lane change correctly."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_changer.initiate_lane_change(target_lane=0)

        self.assertTrue(self.lane_manager.lane_change_in_progress)
        self.assertEqual(self.lane_manager.ego_vehicle.target_lane, 0)
        self.assertNotEqual(ego_vehicle.lane, ego_vehicle.target_lane)
        self.lane_manager.remove_all_vehicles()

    def test_initiate_lane_change_right(self):
        """Test that the ego vehicle can initiate a right lane change correctly."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_changer.initiate_lane_change(target_lane=2)

        self.assertTrue(self.lane_manager.lane_change_in_progress)
        self.assertEqual(self.lane_manager.ego_vehicle.target_lane, 2)
        self.assertNotEqual(self.lane_manager.ego_vehicle.lane, self.lane_manager.ego_vehicle.target_lane)
        self.lane_manager.remove_all_vehicles()
    def test_invalid_lane_change(self):
        """Test that invalid lane changes (out of bounds) are not initiated."""
        # Attempt to change left when already in the leftmost lane
        ego_vehicle = Vehicle(x=100, lane=0, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        with self.assertRaises(Exception):
            self.lane_changer.initiate_lane_change(target_lane=-1)
        
        self.assertFalse(self.lane_manager.lane_change_in_progress)
        self.lane_manager.remove_all_vehicles()
        
        ego_vehicle = Vehicle(x=100, lane=2, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        with self.assertRaises(Exception):
            self.lane_changer.initiate_lane_change(target_lane=3)
        self.assertFalse(self.lane_manager.lane_change_in_progress)
        self.lane_manager.remove_all_vehicles()
    def test_lane_change_progression(self):
        """Test that the ego vehicle moves smoothly during lane change."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_changer.initiate_lane_change(target_lane=0)

        # Simulate the lane change progression over several steps
        for _ in range(self.config.lane_change_duration):
            self.lane_manager.update_positions_relative_to_ego()

        # Check if the ego vehicle is in the target lane
        self.assertEqual(self.lane_manager.ego_vehicle.lane, 0)
        self.assertFalse(self.lane_manager.lane_change_in_progress)
        self.lane_manager.remove_all_vehicles()

    def test_lane_change_completion(self):
        """Test that the lane change is marked as complete when the vehicle reaches the target lane."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)

        self.lane_changer.initiate_lane_change(target_lane=2)

        # Simulate the lane change completion
        for _ in range(self.config.lane_change_duration):
            self.lane_manager.update_positions_relative_to_ego()

        # Verify that lane change is complete
        self.assertEqual(self.lane_manager.ego_vehicle.lane, 2)
        self.assertFalse(self.lane_manager.lane_change_in_progress)
        self.assertEqual(self.lane_manager.ego_vehicle.y, self.lane_manager.ego_vehicle.lane * self.config.lane_width)

if __name__ == "__main__":
    unittest.main()
