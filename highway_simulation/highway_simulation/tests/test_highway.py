"""Tests for the Highway class."""

import unittest

import numpy as np

from highway_simulation.scripts.highway import Highway
from highway_simulation.scripts.vehicle.vehicle import Vehicle
from highway_simulation.scripts.util.config import default_config


class TestHighway(unittest.TestCase):

    def setUp(self) -> None:
        self.config = default_config
        self.highway = Highway(self.config)
        self.highway.reset(seed=42)

    def test_reset(self) -> None:
        """Test that the environment resets correctly and returns an initial state."""
        initial_state = self.highway.reset(seed=42)
        self.assertIsNotNone(initial_state, "Initial state should not be None.")
        self.assertEqual(len(initial_state), 15, "Initial state should have 15 elements.")
        self.assertTrue(np.all((0 <= initial_state) & (initial_state <= 1)), "State values should be normalized.")

    def test_step_action(self) -> None:
        """Test that taking a valid action updates the state correctly."""
        # Test with a 'no action' (maintain speed)
        action = 0
        state, reward, done, _ = self.highway.step(action)
        self.assertIsInstance(state, np.ndarray, "State should be a numpy array.")
        self.assertEqual(len(state), 15, "State should have 15 elements.")
        self.assertIsInstance(reward, float, "Reward should be a float.")
        self.assertIsInstance(done, bool, "Done should be a boolean.")

    def test_invalid_action(self) -> None:
        """Test that an invalid action is handled gracefully."""
        invalid_action = 6  # Assuming action space is [0, 5]
        with self.assertRaises(AssertionError):
            self.highway.step(invalid_action)

    def test_state_representation(self) -> None:
        """Test that the state representation is consistent and normalized."""
        state = self.highway.get_state()
        self.assertEqual(state.shape, (15,), "State should have a shape of (15,).")
        self.assertTrue(np.all((0 <= state) & (state <= 1)), "State values should be within [0, 1].")

    def test_episode_termination(self) -> None:
        """Test that the environment terminates correctly based on conditions."""
        # Scenario 1: Reaching effective simulation length
        self.highway.lane_manager.ego_vehicle.x = self.config.effective_sim_length + 10
        _, done = self.highway.calculate_reward((0,0,0),0)
        self.assertTrue(done, "Episode should terminate when effective simulation length is exceeded.")

        # Scenario 2: Time limit exceeded
        self.highway.reward_calculator.sim_start_time -= self.config.effective_sim_time + 10
        _, done = self.highway.calculate_reward((0,0,0),0)
        self.assertTrue(done, "Episode should terminate when the time limit is exceeded.")

    def test_take_action_accelerate(self) -> None:
        """Test that the 'accelerate' action increases the vehicle speed."""
        initial_speed = self.highway.lane_manager.ego_vehicle.speed
        self.highway.step(3)  # Action 3 corresponds to 'accelerate'
        new_speed = self.highway.lane_manager.ego_vehicle.speed
        self.assertGreater(new_speed, initial_speed, "Speed should increase after the 'accelerate' action.")

    def test_take_action_decelerate(self) -> None:
        """Test that the 'decelerate' action decreases the vehicle speed."""
        self.highway.lane_manager.ego_vehicle.speed = 20  # Set an initial speed
        self.highway.step(4)  # Action 4 corresponds to 'decelerate'
        new_speed = self.highway.lane_manager.ego_vehicle.speed
        self.assertLess(new_speed, 20, "Speed should decrease after the 'decelerate' action.")
        
    def test_normalize_xyv(self) -> None:
        x = 200
        y = 3.5
        v = 18
        norm_x, norm_y, norm_vx = self.highway.normalize_xyv(x, y, v)
        self.assertEqual(norm_x, 0.75)
        self.assertEqual(norm_y, 0.5)
        self.assertEqual(norm_vx, 0.5)

        x = 400
        y = 7
        v = 36
        norm_x, norm_y, norm_vx = self.highway.normalize_xyv(x, y, v)
        norm_vx = self.highway.normalize_v(v)
        self.assertEqual(norm_x, 1)
        self.assertEqual(norm_y, 1)
        self.assertEqual(norm_vx, 1)
        
        x = -400
        y = 0
        v = 0
        norm_x, norm_y, norm_vx = self.highway.normalize_xyv(x, y, v)
        norm_vx = self.highway.normalize_v(v)
        self.assertEqual(norm_x, 0)
        self.assertEqual(norm_y, 0)
        self.assertEqual(norm_vx, 0)

    def test_get_state_normalization(self) -> None:
        """Test the normalization of the state representation."""
        # Set up ego vehicle
        
        self.highway.lane_manager.remove_all_vehicles()
        ego_vehicle = Vehicle(x=50, lane=1, speed=36, v_max=36, is_ego=True)
        self.highway.lane_manager.add_vehicle(ego_vehicle)

        # Create nearby vehicles
        closest_veh = Vehicle(x=60, lane=1, speed=15, v_max=15)
        second_closest_veh = Vehicle(x=150, lane=1, speed=25, v_max=25)
        third_closest_veh = Vehicle(x=-60, lane=0, speed=18, v_max=18)
        forth_closest_veh = Vehicle(x=-200, lane=2, speed=22, v_max=22)

        self.highway.lane_manager.add_vehicle(closest_veh)
        self.highway.lane_manager.add_vehicle(second_closest_veh)
        self.highway.lane_manager.add_vehicle(third_closest_veh)
        self.highway.lane_manager.add_vehicle(forth_closest_veh)

        # Get the state from the environment
        state = self.highway.get_state()

        # Manually calculate the expected normalized values
        # Ego vehicle speed (normalized)
        ego_x = self.highway.lane_manager.ego_vehicle.relative_x
        ego_y = self.highway.lane_manager.ego_vehicle.y
        ego_speed = self.highway.lane_manager.ego_vehicle.speed

        # Expected state array (flattened)
        expected_state = np.array([self.highway.normalize_xyv(ego_x, ego_y,ego_speed),
        self.highway.normalize_xyv(closest_veh.relative_x, closest_veh.y,closest_veh.speed),
        self.highway.normalize_xyv(second_closest_veh.relative_x, second_closest_veh.y,second_closest_veh.speed), 
        self.highway.normalize_xyv(third_closest_veh.relative_x, third_closest_veh.y,third_closest_veh.speed),
        self.highway.normalize_xyv(forth_closest_veh.relative_x, forth_closest_veh.y,forth_closest_veh.speed)],dtype=np.float32)
        expected_state= expected_state.flatten()
        # Check that the state matches the expected values
        np.testing.assert_almost_equal(state, expected_state, decimal=4, err_msg="The state representation is not correctly normalized.")
    def test_get_state_normalization_with_missing_vehicles(self):
        """When a vehicle is missing, all values filled with 0s. This means rl model will imagine a vehicle at -400m in the middle lane and velocity 0."""
        # Set up ego vehicle
        
        self.highway.lane_manager.remove_all_vehicles()
        ego_vehicle = Vehicle(x=50, lane=1, speed=36, v_max=36, is_ego=True)
        self.highway.lane_manager.add_vehicle(ego_vehicle)

        # Create nearby vehicles
        closest_veh = Vehicle(x=60, lane=1, speed=15, v_max=15)
        second_closest_veh = Vehicle(x=150, lane=1, speed=25, v_max=25)
        third_closest_veh = Vehicle(x=-60, lane=0, speed=18, v_max=18)


        self.highway.lane_manager.add_vehicle(closest_veh)
        self.highway.lane_manager.add_vehicle(second_closest_veh)
        self.highway.lane_manager.add_vehicle(third_closest_veh)

        # Get the state from the environment
        state = self.highway.get_state()

        # Manually calculate the expected normalized values
        # Ego vehicle speed (normalized)
        ego_x = self.highway.lane_manager.ego_vehicle.relative_x
        ego_y = self.highway.lane_manager.ego_vehicle.y
        ego_speed = self.highway.lane_manager.ego_vehicle.speed

        # Expected state array (flattened)
        expected_state = np.array([self.highway.normalize_xyv(ego_x, ego_y,ego_speed),
        self.highway.normalize_xyv(closest_veh.relative_x, closest_veh.y,closest_veh.speed),
        self.highway.normalize_xyv(second_closest_veh.relative_x, second_closest_veh.y,second_closest_veh.speed), 
        self.highway.normalize_xyv(third_closest_veh.relative_x, third_closest_veh.y,third_closest_veh.speed),
       (0,0,0)],dtype=np.float32)
        expected_state= expected_state.flatten()
        # Check that the state matches the expected values
        np.testing.assert_almost_equal(state, expected_state, decimal=4, err_msg="The state representation is not correctly normalized.")

if __name__ == '__main__':
    unittest.main()
