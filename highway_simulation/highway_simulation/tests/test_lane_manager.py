"""Tests for LaneManager."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.vehicle.vehicle import Vehicle
from highway_simulation.scripts.util.config import default_config


class TestLaneManager(unittest.TestCase):

    def setUp(self):
        self.config = default_config
        LaneManager.set_config(self.config)
        Vehicle.set_config(self.config)
        self.lane_manager = LaneManager()

    def test_find_front_back_vehicles(self):
        ego_vehicle = Vehicle(x=100, lane=1, speed=20, v_max=33.33, is_ego=True)
        front_vehicle = Vehicle(x=150, lane=1, speed=20, v_max=33.33)
        back_vehicle = Vehicle(x=50, lane=1, speed=20, v_max=33.33)

        self.lane_manager.add_vehicle(ego_vehicle)
        self.lane_manager.add_vehicle(front_vehicle)
        self.lane_manager.add_vehicle(back_vehicle)

        front, back = self.lane_manager.find_front_back_vehicles(self.lane_manager.lanes[ego_vehicle.lane])
        self.assertEqual(front, front_vehicle)
        self.assertEqual(back, back_vehicle)
        self.lane_manager.remove_all_vehicles()

    def test_add_vehicle(self):
        """Test that vehicles are added correctly to the lane."""
        num_vehicles_before = len(self.lane_manager.lanes[1].vehicles)
        new_vehicle = Vehicle(x=200, lane=1, speed=25, v_max=33.33)
        self.lane_manager.add_vehicle(new_vehicle)
        num_vehicles_after = len(self.lane_manager.lanes[1].vehicles)

        self.assertEqual(num_vehicles_after, num_vehicles_before + 1)
        self.assertIn(new_vehicle, self.lane_manager.lanes[1].vehicles)
        self.lane_manager.remove_all_vehicles()

    def test_remove_vehicle(self):
        vehicle = Vehicle(x=200, lane=1, speed=20, v_max=33.33)
        self.lane_manager.add_vehicle(vehicle)

        num_vehicles_before = len(self.lane_manager.lanes[1].vehicles)
        self.lane_manager.destroy_vehicle(vehicle)
        num_vehicles_after = len(self.lane_manager.lanes[1].vehicles)

        self.assertEqual(num_vehicles_after, num_vehicles_before - 1)
        self.lane_manager.remove_all_vehicles()

    def test_update_lane_attributes(self):
        """Test that lane attributes like 'how_far_to_ego' are updated correctly."""
        front_vehicle = Vehicle(x=200, lane=2, speed=20, v_max=33.33)
        ego_vehicle = Vehicle(x=250, lane=2, speed=25, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(front_vehicle)
        self.lane_manager.add_vehicle(ego_vehicle)

        self.lane_manager.update_lane_attributes()

        #Check the attribute 'how_far_to_ego'
        self.assertEqual(self.lane_manager.lanes[0].how_far_to_ego, -2)
        self.assertEqual(self.lane_manager.lanes[1].how_far_to_ego, -1)
        self.assertEqual(self.lane_manager.lanes[2].how_far_to_ego, 0)
        self.lane_manager.remove_all_vehicles()
        # CHECK WHAT HAPPENS AFTER LANE CHANGE EGO VEHICLE DOES A LANE CHANGE while testing lane_changer
    
    def test_reset_positions_wrt_ego(self):
        """Test that vehicle positions are correctly reset relative to the ego vehicle."""
        front_vehicle = Vehicle(x=300, lane=1, speed=20, v_max=33.33)
        self.lane_manager.add_vehicle(front_vehicle)
        ego_vehicle = Vehicle(x=250, lane=2, speed=25, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)

        self.lane_manager.reset_positions_wrt_ego()

        expected_relative_x = ego_vehicle.relative_x + (front_vehicle.x - ego_vehicle.x)
        self.assertEqual(front_vehicle.relative_x, expected_relative_x)
        self.lane_manager.remove_all_vehicles()

    def test_is_in_range(self):
        """Test that the 'is_in_range' function correctly identifies in-range vehicles."""
        in_range_vehicle = Vehicle(x=300, lane=1, speed=20, v_max=33.33)
        out_of_range_vehicle = Vehicle(x=4500, lane=1, speed=20, v_max=33.33)
        ego_vehicle = Vehicle(x=250, lane=2, speed=25, v_max=33.33, is_ego=True)
        self.lane_manager.add_vehicle(ego_vehicle)
        self.assertTrue(self.lane_manager.is_in_range(in_range_vehicle))
        self.assertFalse(self.lane_manager.is_in_range(out_of_range_vehicle))
        self.lane_manager.remove_all_vehicles()

    def test_get_nearby_vehicles(self):
        v1 = Vehicle(x=250, lane=0, speed=20, v_max=33.33)
        v2 = Vehicle(x=350, lane=1, speed=20, v_max=33.33)
        v3 = Vehicle(x=150, lane=2, speed=20, v_max=33.33)
        v4 = Vehicle(x=400, lane=0, speed=20, v_max=33.33)
        v5 = Vehicle(x=450, lane=1, speed=20, v_max=33.33)
        v6 = Vehicle(x=600, lane=2, speed=20, v_max=33.33)
        ego_vehicle = Vehicle(x=250, lane=2, speed=25, v_max=33.33, is_ego=True)
        veh_list = [v1, v2, v3, v4, v5, v6, ego_vehicle]
        for veh in veh_list:
            self.lane_manager.add_vehicle(veh)
        nearby_vehicles = self.lane_manager.get_nearby_vehicles(4)
        self.assertTrue(v1 and v2 and v3 and v4 in nearby_vehicles)
        self.lane_manager.remove_all_vehicles()



    # MOBIL FUNCTIONS
    def test_find_vehicle_ahead(self):
        # Create vehicles
        vehicle1 = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle2 = Vehicle(x=100, lane=1, speed=40.0, v_max=40.0)
        vehicle3 = Vehicle(x=150, lane=1, speed=30.0, v_max=30.0)

        # Add vehicles to lane manager
        self.lane_manager.add_vehicle(vehicle1)
        self.lane_manager.add_vehicle(vehicle2)
        self.lane_manager.add_vehicle(vehicle3)

        # Test find_vehicle_ahead
        self.assertEqual(self.lane_manager.find_vehicle_ahead(vehicle1, 1), vehicle2)
        self.assertEqual(self.lane_manager.find_vehicle_ahead(vehicle2, 1), vehicle3)
        self.assertIsNone(self.lane_manager.find_vehicle_ahead(vehicle3, 1))
        self.lane_manager.remove_all_vehicles()

    def test_find_vehicle_behind(self):
        # Create vehicles
        vehicle1 = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle2 = Vehicle(x=100, lane=1, speed=40.0, v_max=40.0)
        vehicle3 = Vehicle(x=150, lane=1, speed=30.0, v_max=30.0)


        # Add vehicles to lane manager
        self.lane_manager.add_vehicle(vehicle1)
        self.lane_manager.add_vehicle(vehicle2)
        self.lane_manager.add_vehicle(vehicle3)

        # Test find_vehicle_behind
        self.assertEqual(self.lane_manager.find_vehicle_behind(vehicle3, 1), vehicle2)
        self.assertEqual(self.lane_manager.find_vehicle_behind(vehicle2, 1), vehicle1)
        self.assertIsNone(self.lane_manager.find_vehicle_behind(vehicle1, 1))
        self.lane_manager.remove_all_vehicles()

if __name__ == '__main__':
    unittest.main()
