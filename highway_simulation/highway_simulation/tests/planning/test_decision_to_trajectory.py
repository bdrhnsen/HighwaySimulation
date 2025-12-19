"""Tests for decision-to-trajectory conversion."""

import unittest

from highway_simulation.scripts.planning.decision_to_trajectory import DecisionToTrajectory
from highway_simulation.scripts.planning.state import Acc, Pos, State, Vel
from highway_simulation.scripts.planning.trajectory_planner import TrajectoryPlanner
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import default_config


class TestDecisionToTrajectory(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment."""
        self.planner = TrajectoryPlanner()
        TrajectoryPlanner.set_config(default_config)
        self.config = default_config
        self.config.lane_width = 3.5  # Example lane width
        self.config.time_step = 0.1  # Example time step
        DecisionToTrajectory.set_config(self.config)

        self.trajectory_planner = TrajectoryPlanner()
        self.decision_to_trajectory = DecisionToTrajectory()

        # Example initial state
        self.current_state = State(
            Pos(0, 0), Vel(10, 0), Acc(0, 0)
        )

    def test_calculate_lane_change_end_state(self):
        """
        Test lane change end state calculation.
        """
        # Left lane change
        result = self.decision_to_trajectory.calculate_lane_change_end_state(self.current_state, -1)
        self.assertAlmostEqual(result.pos.y, -3.5)
        self.assertAlmostEqual(result.pos.x, 30)  # 10 * 3 + 0.5 * 0 * 3^2
        self.assertAlmostEqual(result.vel.y, 0)
        self.assertAlmostEqual(result.acc.y, 0)

        # Right lane change
        result = self.decision_to_trajectory.calculate_lane_change_end_state(self.current_state, 1)
        self.assertAlmostEqual(result.pos.y, 3.5)

    def test_calculate_end_state(self):
        """
        Test end state calculation for longitudinal actions.
        """
        # Accelerate
        result = self.decision_to_trajectory.calculate_end_state(self.current_state, 5)
        self.assertAlmostEqual(result.pos.x, 37.5)  # 0.5*(10 + 15)*3
        self.assertAlmostEqual(result.vel.x, 15)  # 10 + 5 
        self.assertAlmostEqual(result.acc.x, 0)

        # Decelerate
        result = self.decision_to_trajectory.calculate_end_state(self.current_state, -5)
        self.assertAlmostEqual(result.pos.x, 22.5)  # 0.5 * (10+5) * 3
        self.assertAlmostEqual(result.vel.x, 5)  # 10-5 

    def test_process_decision(self):
        """
        Test processing of decisions and generation of trajectories.
        """
        # Test lane change left
        trajectory = self.decision_to_trajectory.process_decision(self.current_state, Action.CHANGE_LANE_RIGHT)
        self.assertEqual(len(trajectory.trajectory), int(self.trajectory_planner.time_horizon / self.config.time_step))
        trajectory.plot_trajectory()
        # Check the final state in the trajectory
        final_state = trajectory.trajectory[-1]
        self.assertAlmostEqual(final_state.pos.y, 3.5)
        
        # Test acceleration
        trajectory = self.decision_to_trajectory.process_decision(self.current_state, Action.ACCELERATE)
        #trajectory.plot_trajectory()
        final_state = trajectory.trajectory[-1]
        self.assertAlmostEqual(final_state.pos.x, 37.5)  # x position after accelerating

        
if __name__ == "__main__":
    unittest.main()
