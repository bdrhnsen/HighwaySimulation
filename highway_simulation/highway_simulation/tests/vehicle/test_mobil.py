"""Tests for MOBIL lane change logic."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.planning.decision_to_trajectory import DecisionToTrajectory
from highway_simulation.scripts.util.config import Config, default_config
from highway_simulation.scripts.vehicle.vehicle import Vehicle


class TestMobilLaneChange(unittest.TestCase):
    def setUp(self):
        self.config: Config = default_config
        Vehicle.set_config(self.config)
        LaneManager.set_config(self.config)
        DecisionToTrajectory.set_config(self.config)
        self.lane_manager = LaneManager()
        self.lane_manager.set_config(self.config)
        

    def test_lane_change_benefit(self):
        # Test Case 1: Vehicle should change lanes due to slow vehicle ahead
        vehicle = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle_ahead_current = Vehicle(x=75, lane=1, speed=40.0, v_max=40.0)
        vehicle_behind_current = None
        vehicle_ahead_target = None
        vehicle_behind_target = None
        self.lane_manager.add_vehicle(vehicle)
        self.lane_manager.add_vehicle(vehicle_ahead_current)

        self.assertTrue(vehicle.calculate_mobil_lane_change(vehicle_ahead_current, vehicle_behind_current, vehicle_ahead_target, vehicle_behind_target))

    def test_lane_change_benefit2(self):
        # Test Case 2: Vehicle want to do a lane change but there is a faster vehicle behind in target lane
        ## Result, still changes lane since the increase in self speed is more important than the decrease in speed of the vehicle behind
        vehicle = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle_ahead_current = Vehicle(x=75, lane=1, speed=40.0, v_max=40.0)
        vehicle_behind_current = None
        vehicle_ahead_target = None
        vehicle_behind_target = Vehicle(x=30, lane=0, speed=50.0, v_max=75.0)
        self.lane_manager.add_vehicle(vehicle)
        self.lane_manager.add_vehicle(vehicle_ahead_current)

        self.assertTrue(vehicle.calculate_mobil_lane_change(vehicle_ahead_current, vehicle_behind_current, vehicle_ahead_target, vehicle_behind_target))

    def test_lane_change_benefit3(self):
        # Test Case 3: Test edge cases of lane change decision
        ## no lane change since vehicle is far away
        vehicle = Vehicle(x=25, lane=1, speed=50.0, v_max=50.0)
        vehicle_ahead_current = Vehicle(x=75, lane=1, speed=50.0, v_max=50.0)
        vehicle_behind_current = None
        vehicle_ahead_target = None
        vehicle_behind_target = None
        self.lane_manager.add_vehicle(vehicle)
        self.lane_manager.add_vehicle(vehicle_ahead_current)

        self.assertFalse(vehicle.calculate_mobil_lane_change(vehicle_ahead_current, vehicle_behind_current, vehicle_ahead_target, vehicle_behind_target))

        self.lane_manager.remove_all_vehicles()
        ## not choose to do a lane change self speed is low
        vehicle = Vehicle(x=50, lane=1, speed=30.0, v_max=30.0)
        vehicle_ahead_current = Vehicle(x=75, lane=1, speed=50.0, v_max=50.0)
        vehicle_behind_current = None
        vehicle_ahead_target = None
        vehicle_behind_target = None
        self.lane_manager.add_vehicle(vehicle)
        self.lane_manager.add_vehicle(vehicle_ahead_current)

        self.assertFalse(vehicle.calculate_mobil_lane_change(vehicle_ahead_current, vehicle_behind_current, vehicle_ahead_target, vehicle_behind_target))

        self.lane_manager.remove_all_vehicles()
        ## it would rather change lanes since vehicle ahead in the target lane is further away.
        vehicle = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle_ahead_current = Vehicle(x=75, lane=1, speed=50.0, v_max=50.0)
        vehicle_behind_current = None
        vehicle_ahead_target = Vehicle(x=100, lane=0, speed=50.0, v_max=50.0)
        vehicle_behind_target = None
        self.lane_manager.add_vehicle(vehicle)
        self.lane_manager.add_vehicle(vehicle_ahead_current)

        self.assertTrue(vehicle.calculate_mobil_lane_change(vehicle_ahead_current, vehicle_behind_current, vehicle_ahead_target, vehicle_behind_target))
    def test_update_non_ego_lane_changes(self):
        # Create vehicles
        vehicle1 = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0)
        vehicle2 = Vehicle(x=100, lane=1, speed=40.0, v_max=40.0)
        vehicle3 = Vehicle(x=150, lane=1, speed=30.0, v_max=30.0)
        vehicle4 = Vehicle(x=200, lane=0, speed=60.0, v_max=60.0)

        # Add vehicles to lane manager
        self.lane_manager.add_vehicle(vehicle1)
        self.lane_manager.add_vehicle(vehicle2)
        self.lane_manager.add_vehicle(vehicle3)
        self.lane_manager.add_vehicle(vehicle4)

        # Test update_non_ego_lane_changes
        self.lane_manager.update_non_ego_lane_changes()

        # Check if vehicle1 has a new trajectory for lane change
        self.assertFalse(vehicle1.trajectory.is_trajectory_empty())
        self.assertEqual(vehicle1.target_lane, 0)  

        self.assertFalse(vehicle2.trajectory.is_trajectory_empty())
        self.assertEqual(vehicle2.target_lane, 0)  

        self.assertTrue(vehicle3.trajectory.is_trajectory_empty())
    
    def test_left_lane_only_for_takeover(self):
        # Create vehicles
        vehicle1 = Vehicle(x=50, lane=0, speed=50.0, v_max=50.0)
        # Add vehicles to lane manager
        self.lane_manager.add_vehicle(vehicle1)

        # Test update_non_ego_lane_changes
        self.lane_manager.update_non_ego_lane_changes()

        # Check if vehicle1 has a new trajectory for lane change, it should change to right lane
        self.assertFalse(vehicle1.trajectory.is_trajectory_empty())
        self.assertEqual(vehicle1.target_lane, 1)  

if __name__ == "__main__":
    unittest.main()
