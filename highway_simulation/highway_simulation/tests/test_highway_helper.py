"""Tests for HighwayHelper."""

import random
import unittest

from highway_simulation.scripts.reset.highwayHelper import HighwayHelper
from highway_simulation.scripts.vehicle.vehicle import Vehicle
from highway_simulation.scripts.util.config import default_config


class TestHighwayHelper(unittest.TestCase):
    # TODO can be improved
    def setUp(self) -> None:
        self.config = default_config
        Vehicle.set_config(self.config)
        self.highway_helper = HighwayHelper(self.config)
        self.highway_helper.vehicle_list = []

    def test_is_position_available_empty_lane(self) -> None:
        """Test that a position is available when there are no vehicles in the lane."""
        position = 100
        lane = 1
        self.assertTrue(self.highway_helper.is_position_available(position, lane))

    def test_is_position_available_with_vehicle(self) -> None:
        """Test that a position is not available when a vehicle is present nearby."""
        vehicle = Vehicle(x=150, lane=1, speed=20, v_max=33.33)
        self.highway_helper.vehicle_list.append(vehicle)

        # Test a position close to the existing vehicle
        position = 140
        lane = 1
        self.assertFalse(self.highway_helper.is_position_available(position, lane))

    def test_generate_vehicle_list_with_same_seed(self) -> None:
        """Test that vehicle lists generated with the same seed are identical."""
        seed = 42
        num_vehicles = 5

        vehicles1 = self.highway_helper.generate_vehicle_list(seed, num_vehicles)
        vehicles2 = self.highway_helper.generate_vehicle_list(seed, num_vehicles)

        # Check that both lists have the same number of vehicles
        self.assertEqual(vehicles1, vehicles2)


    def test_generate_vehicle_list_with_different_seeds(self) -> None:
        """Test that vehicle lists generated with different seeds are different."""
        seed1 = 42
        seed2 = 100
        num_vehicles = 5

        vehicles1 = self.highway_helper.generate_vehicle_list(seed1, num_vehicles)
        vehicles2 = self.highway_helper.generate_vehicle_list(seed2, num_vehicles)

        # Check that both lists have the same number of vehicles
        self.assertNotEqual(vehicles1, vehicles2)

    def test_seeding_consistency(self) -> None:
        """Test that setting a random seed produces consistent results for the same seed."""
        seed = 123
        num_vehicles = 10

        random.seed(seed)
        vehicles1 = self.highway_helper.generate_vehicle_list(seed, num_vehicles)

        # Reset the seed and generate the list again
        random.seed(seed)
        vehicles2 = self.highway_helper.generate_vehicle_list(seed, num_vehicles)
        self.assertEqual(vehicles1, vehicles2)
        
if __name__ == "__main__":
    unittest.main()
