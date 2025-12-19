"""Tests for the kinematic bicycle model behavior."""

import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.planning.decision_to_trajectory import DecisionToTrajectory
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import Config, default_config
from highway_simulation.scripts.vehicle.vehicle import Vehicle


class TestKinematicBicycleModel(unittest.TestCase):

    def setUp(self):
        self.config: Config = default_config
        Vehicle.set_config(self.config)
        LaneManager.set_config(self.config)
        DecisionToTrajectory.set_config(self.config)
        self.vehicle = Vehicle(x=50, lane=1, speed=80, v_max=80, is_ego=True)
        self.lane_manager = LaneManager()
        self.decision_to_trajectory = DecisionToTrajectory()

    def test_lateral_motion(self):
        action = Action.CHANGE_LANE_RIGHT
        trajectory = self.decision_to_trajectory.process_decision(
            self.vehicle.return_state, action
        )
        self.vehicle.trajectory = trajectory

        # trajectory.plot_trajectory()
        for _ in range(100):
            if not self.vehicle.trajectory.is_trajectory_empty():
                self.vehicle.apply_state(self.vehicle.trajectory.use_next_state())
            self.vehicle.update()
            # if i!=0 and i%10 == 0:
            #     self.vehicle.history_trajectory.plot_trajectory(plot_heading=True)


    def test_lateral_motion_LEFT(self):
        action = Action.CHANGE_LANE_LEFT
        trajectory = self.decision_to_trajectory.process_decision(
            self.vehicle.return_state, action
        )
        self.vehicle.trajectory = trajectory

        trajectory.plot_trajectory()
        for _ in range(100):
            if not self.vehicle.trajectory.is_trajectory_empty():
                self.vehicle.apply_state(self.vehicle.trajectory.use_next_state())
            self.vehicle.update()
            # if i!=0 and i%10 == 0:
            #     self.vehicle.history_trajectory.plot_trajectory(plot_heading=True)

if __name__ == "__main__":
    unittest.main()
