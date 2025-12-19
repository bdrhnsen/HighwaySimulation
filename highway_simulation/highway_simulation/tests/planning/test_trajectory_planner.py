"""Tests for trajectory planning."""

import unittest
from math import isclose

from highway_simulation.scripts.planning.state import Acc, Pos, State, Vel
from highway_simulation.scripts.planning.trajectory_planner import TrajectoryPlanner
from highway_simulation.scripts.util.config import default_config


class TestTrajectoryPlanner(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.planner = TrajectoryPlanner()
        TrajectoryPlanner.set_config(default_config)
        # Define a sample start and end state
        self.start_state = State(
            pos=Pos(x=0.0, y=0.0),
            vel=Vel(x=15.0, y=0.0),
            acc=Acc(x=0.0, y=0.0)
        )
        self.end_state = State(
            pos=Pos(x=50, y=3.5),
            vel=Vel(x=15.0, y=0.0),
            acc=Acc(x=0.0, y=0.0)
        )


        # Action space scenarios
        self.action_space = [
            {"name": "lane_change_left", "start": State(Pos(0, 0), Vel(15, 0), Acc(0, 0)),
             "end": State(Pos(40, -3.5), Vel(15, 0), Acc(0, 0))},  # Move to the left lane
            {"name": "lane_change_right", "start": State(Pos(0, 0), Vel(15, 0), Acc(0, 0)),
             "end": State(Pos(40, 3.5), Vel(15, 0), Acc(0, 0))},  # Move to the right lane
            {"name": "accelerate", "start": State(Pos(0, 0), Vel(20, 0), Acc(0, 0)),
             "end": State(Pos(75, 0), Vel(25, 0), Acc(0, 0))},  # Increase speed
            {"name": "decelerate", "start": State(Pos(0, 0), Vel(20, 0), Acc(0, 0)),
             "end": State(Pos(55, 0), Vel(15, 0), Acc(0, 0))},  # Decrease speed
            {"name": "maintain_speed", "start": State(Pos(0, 0), Vel(15, 0), Acc(0, 0)),
             "end": State(Pos(55, 0), Vel(15, 0), Acc(0, 0))}  # Maintain current speed
        ]

    def test_plan_path_between_points(self):
        """Test the full trajectory planning."""
        trajectory = self.planner.plan_between_points(self.start_state, self.end_state)

        # Check that each point is a State object
        for state in trajectory.trajectory:
            self.assertIsInstance(state, State)
            self.assertIsInstance(state.pos, Pos)
            self.assertIsInstance(state.vel, Vel)
            self.assertIsInstance(state.acc, Acc)


        # End state of the trajectory is only constrained by pos.y, and vel.x and vel.y is assumed to be 0s
        self.assertTrue(isclose(trajectory.trajectory[-1].pos.y, self.end_state.pos.y, abs_tol=0.1))
        self.assertTrue(isclose(trajectory.trajectory[-1].vel.x, self.end_state.vel.x, abs_tol=0.1))
        self.assertTrue(isclose(trajectory.trajectory[-1].vel.y, self.end_state.vel.y, abs_tol=0.1))

    
    def test_trajectory_for_action_space(self):
        "Test trajectory generation for each action in the action space."
        for case in self.action_space:
            action_name = case["name"]
            start_state = case["start"]
            end_state = case["end"]


            trajectory = self.planner.plan_between_points(start_state, end_state)

            self.assertTrue(
                isclose(trajectory.trajectory[0].pos.x, start_state.pos.x, abs_tol=0.1),
                f"Failed for {action_name}: Start position x mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[0].pos.y, start_state.pos.y, abs_tol=0.1),
                f"Failed for {action_name}: Start position y mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[0].vel.x, start_state.vel.x, abs_tol=0.1),
                f"Failed for {action_name}: Start velocity x mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[0].vel.y, start_state.vel.y, abs_tol=0.1),
                f"Failed for {action_name}: Start velocity y mismatch"
            )

            # Verify end state
            self.assertTrue(
                isclose(trajectory.trajectory[-1].pos.x, end_state.pos.x, abs_tol=0.1),
                f"Failed for {action_name}: End position x mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[-1].pos.y, end_state.pos.y, abs_tol=0.1),
                f"Failed for {action_name}: End position y mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[-1].vel.x, end_state.vel.x, abs_tol=0.1),
                f"Failed for {action_name}: End velocity x mismatch"
            )
            self.assertTrue(
                isclose(trajectory.trajectory[-1].vel.y, end_state.vel.y, abs_tol=0.1),
                f"Failed for {action_name}: End velocity y mismatch"
            )

    def test_smoothness_of_trajectory(self):
        "Ensure trajectories are smooth with continuous velocity and acceleration changes."
        for case in self.action_space:
            action_name = case["name"]
            start_state = case["start"]
            end_state = case["end"]

            # Generate trajectory
            trajectory = self.planner.plan_between_points(start_state, end_state)
            #trajectory.plot_trajectory()
            for i in range(1, len(trajectory.trajectory)):
                prev_state = trajectory.trajectory[i - 1]
                curr_state = trajectory.trajectory[i]

                # Ensure velocity change is continuous
                self.assertTrue(
                    abs(curr_state.vel.x - prev_state.vel.x) < 5,
                    f"Failed for {action_name}: Velocity x discontinuity at step {i}"
                )
                self.assertTrue(
                    abs(curr_state.vel.y - prev_state.vel.y) < 5,
                    f"Failed for {action_name}: Velocity y discontinuity at step {i}"
                )

                # Ensure acceleration change is continuous
                self.assertTrue(
                    abs(curr_state.acc.x - prev_state.acc.x) < 7,
                    f"Failed for {action_name}: Acceleration x discontinuity at step {i}"
                )
                self.assertTrue(
                    abs(curr_state.acc.y - prev_state.acc.y) < 4,
                    f"Failed for {action_name}: Acceleration y discontinuity at step {i}"
                )
    
if __name__ == "__main__":
    unittest.main()
