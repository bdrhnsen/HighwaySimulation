"""Tests for action reward ordering."""

import unittest

from highway_simulation.scripts.highway import Highway
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import default_config
from highway_simulation.scripts.vehicle.vehicle import Vehicle

class TestActionRewards(unittest.TestCase):

    def setUp(self):
        self.config = default_config
        self.highway = Highway(self.config)
        self.highway.reset(seed=42)

    def reset_environment(self, ego_vehicle_speed):
        """Helper function to reset the environment and add vehicles."""
        ego_vehicle = Vehicle(x=100, lane=1, speed=ego_vehicle_speed, v_max=ego_vehicle_speed, is_ego=True)
        self.highway.lane_manager.remove_all_vehicles()
        self.highway.lane_manager.add_vehicle(ego_vehicle)

    def reset_environment_in_near_collision(self, ego_vehicle_speed, distance_x, speed_diff):
        ego_vehicle = Vehicle(x=100, lane=1, speed=ego_vehicle_speed, v_max=ego_vehicle_speed, is_ego=True)
        front_vehicle = Vehicle(
            x=100 + distance_x,
            lane=1,
            speed=ego_vehicle_speed + speed_diff,
            v_max=ego_vehicle_speed + speed_diff,
        )
        self.highway.lane_manager.remove_all_vehicles()
        self.highway.lane_manager.add_vehicle(ego_vehicle)
        self.highway.lane_manager.add_vehicle(front_vehicle)
        self.highway.lane_manager.lane_change_in_progress = False

    def test_which_action_when_max_speed(self):
        """Going at max rewardable speed. So max rewardable action should be 0 acc, 
        it should be ranked as keep vel, decelarate, emergency brake, accelarate"""
        
        self.reset_environment(self.config.max_rewardable_vel * 36 / 10)
        no_action_reward = self.highway.step(Action.NO_ACTION.value)[1]
        self.reset_environment(self.config.max_rewardable_vel * 36 / 10)
        accelerate_reward = self.highway.step(Action.ACCELERATE.value)[1]
        self.reset_environment(self.config.max_rewardable_vel * 36 / 10)
        decelerate_reward = self.highway.step(Action.DECELERATE.value)[1]
        self.reset_environment(self.config.max_rewardable_vel * 36 / 10)
        emergancy_brake_reward = self.highway.step(Action.EMERGENCY_BRAKE.value)[1]
        #print(no_action_reward, accelerate_reward, decelerate_reward, emergancy_brake_reward)
        self.assertGreater(no_action_reward, decelerate_reward)
        self.assertGreater(decelerate_reward, emergancy_brake_reward)
        self.assertGreater(emergancy_brake_reward, accelerate_reward)


    def test_which_action_when_min_speed(self):
        """Going at min rewardable speed. So max rewardable action should be accelarate, 
        it should be ranked as accelarate, keep vel, decelarate, emergency brake"""
        
        self.reset_environment(self.config.min_rewardable_vel * 36 / 10)
        no_action_reward = self.highway.step(Action.NO_ACTION.value)[1]
        self.reset_environment(self.config.min_rewardable_vel * 36 / 10)
        accelerate_reward = self.highway.step(Action.ACCELERATE.value)[1]
        self.reset_environment(self.config.min_rewardable_vel * 36 / 10)
        decelerate_reward = self.highway.step(Action.DECELERATE.value)[1]
        self.reset_environment(self.config.min_rewardable_vel * 36 / 10)
        emergancy_brake_reward = self.highway.step(Action.EMERGENCY_BRAKE.value)[1]
        #print(no_action_reward, accelerate_reward, decelerate_reward, emergancy_brake_reward)
        self.assertGreater(accelerate_reward, no_action_reward)
        self.assertGreater(no_action_reward, decelerate_reward)
        self.assertGreater(decelerate_reward, emergancy_brake_reward)

    def test_which_action_when_middle_speed(self):
        """Going at mean rewardable speed. So max rewardable action should be accelarate, 
        it should be ranked as accelarate, keep vel, decelarate, emergency brake"""
        speed = (self.config.max_rewardable_vel + self.config.min_rewardable_vel) / 2 * 36 / 10
        self.reset_environment(speed)
        no_action_reward = self.highway.step(Action.NO_ACTION.value)[1]
        self.reset_environment(speed)
        accelerate_reward = self.highway.step(Action.ACCELERATE.value)[1]
        self.reset_environment(speed)
        decelerate_reward = self.highway.step(Action.DECELERATE.value)[1]
        self.reset_environment(speed)
        emergancy_brake_reward = self.highway.step(Action.EMERGENCY_BRAKE.value)[1]
   
        self.assertGreater(accelerate_reward, no_action_reward)
        self.assertGreater(no_action_reward, decelerate_reward)
        self.assertGreater(decelerate_reward, emergancy_brake_reward)

    def test_which_action_when_near_collision(self):
        """Going at mean rewardable speed. Critical position, can easily crash. Max rewardable should be lane change,
        emergency brake
        1. Lane change 2. emergency brake, 3. decelarate, 4. no action, 5. accelarate"""
        speed = (self.config.max_rewardable_vel + self.config.min_rewardable_vel) / 2 * 36 / 10

        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=10, speed_diff=-2)
        no_action_reward = self.highway.step(Action.NO_ACTION.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=10, speed_diff=-2)
        accelerate_reward = self.highway.step(Action.ACCELERATE.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=10, speed_diff=-2)
        decelerate_reward = self.highway.step(Action.DECELERATE.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=10, speed_diff=-2)
        emergancy_brake_reward = self.highway.step(Action.EMERGENCY_BRAKE.value)[1]

        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=7, speed_diff=-2)
        lane_change_left_reward = self.highway.step(Action.CHANGE_LANE_LEFT.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=7, speed_diff=-2)
        lane_change_right_reward = self.highway.step(Action.CHANGE_LANE_RIGHT.value)[1]
        
        #print(no_action_reward, accelerate_reward, decelerate_reward, emergancy_brake_reward, lane_change_left_reward)
        self.assertEqual(lane_change_left_reward, lane_change_right_reward)
        self.assertGreater(lane_change_left_reward, emergancy_brake_reward)
        self.assertGreater(emergancy_brake_reward, decelerate_reward)
        self.assertGreater(decelerate_reward, no_action_reward)
        self.assertGreater(decelerate_reward, accelerate_reward)
        ## TODO right now accelerating in this scenerio gives more reward then doing nothing, 
        # this is because reward in this speed range makes more differences then near collision penalty
        self.assertAlmostEqual(no_action_reward, accelerate_reward, delta=0.03)

        


    def test_which_action_when_near_collision_farther(self):
        """Going at mean rewardable speed. semi -critical position. Max rewardable should be lane change
        1. Lane change 2. emergency brake, 3. decelarate, 4. no action, 5. accelarate"""
        speed = (self.config.max_rewardable_vel + self.config.min_rewardable_vel) / 2 * 36 / 10

        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        no_action_reward = self.highway.step(Action.NO_ACTION.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        accelerate_reward = self.highway.step(Action.ACCELERATE.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        decelerate_reward = self.highway.step(Action.DECELERATE.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        emergancy_brake_reward = self.highway.step(Action.EMERGENCY_BRAKE.value)[1]

        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        lane_change_left_reward = self.highway.step(Action.CHANGE_LANE_LEFT.value)[1]
        self.reset_environment_in_near_collision(ego_vehicle_speed=speed, distance_x=20, speed_diff=-2)
        lane_change_right_reward = self.highway.step(Action.CHANGE_LANE_RIGHT.value)[1]
        
        #print(no_action_reward, accelerate_reward, decelerate_reward, emergancy_brake_reward, lane_change_left_reward)
        self.assertEqual(lane_change_left_reward, lane_change_right_reward)
        self.assertGreater(lane_change_left_reward, emergancy_brake_reward)
        self.assertGreater(emergancy_brake_reward, decelerate_reward)
        self.assertGreater(decelerate_reward, no_action_reward)
        self.assertGreater(no_action_reward, accelerate_reward)

if __name__ == "__main__":
    unittest.main()
