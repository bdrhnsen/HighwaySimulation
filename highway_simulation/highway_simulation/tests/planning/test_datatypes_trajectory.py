"""Tests for trajectory data types and quality metrics."""

import unittest

from highway_simulation.scripts.planning.state import Acc, Jerk, Pos, State, Trajectory, Vel


class TestTrajectory(unittest.TestCase):

    def setUp(self):
        """Set up example trajectories and parameters."""
        self.time_step = 0.1
        self.max_allowed_jerk = 5.0
        self.max_allowed_acc = 2.0

        self.real_trajectory =Trajectory([
            State(pos=Pos(x=(0.0), y=(0.0)),
                  vel=Vel(x=(14.999999999999998), y=(-5.921189464667501e-16)),
                  acc=Acc(x=(0.0), y=(-1.1842378929335002e-15)),
                  jerk=Jerk(x=(11.111111111111097), y=(7.777777777777779))),
            State(pos=Pos(x=(5.057663973987704), y=(0.04036478179139333)),
                  vel=Vel(x=(15.48773052888279), y=(0.3414113702179537)),
                  acc=Acc(x=(2.560585276634656), y=(1.7924096936442606)),
                  jerk=Jerk(x=(4.526748971193411), y=(3.1687242798353914))),
            State(pos=Pos(x=(10.382055580958184), y=(0.2674389066707304)),
                  vel=Vel(x=(16.493674744703547), y=(1.045572321292485)),
                  acc=Acc(x=(3.2007315957933207), y=(2.240512117055327)),
                  jerk=Jerk(x=(-0.41152263374485365), y=(-0.2880658436213981))),
            State(pos=Pos(x=(16.04938271604938), y=(0.7345679012345667)),
                  vel=Vel(x=(17.469135802469133), y=(1.7283950617283939)),
                  acc=Acc(x=(2.4691358024691343), y=(1.7283950617283947)),
                  jerk=Jerk(x=(-3.7037037037036953), y=(-2.5925925925925917))),
            State(pos=Pos(x=(21.98343748412335), y=(1.3884062388863472)),
                  vel=Vel(x=(18.048315805517447), y=(2.133821063862216)),
                  acc=Acc(x=(0.9144947416552363), y=(0.6401463191586636)),
                  jerk=Jerk(x=(-5.3497942386831205), y=(-3.7448559670781894))),
            State(pos=Pos(x=(28.016562515876636), y=(2.1115937611136486)),
                  vel=Vel(x=(18.048315805517447), y=(2.133821063862216)),
                  acc=Acc(x=(-0.9144947416552327), y=(-0.6401463191586654)),
                  jerk=Jerk(x=(-5.3497942386831205), y=(-3.744855967078191))),
            State(pos=Pos(x=(33.95061728395061), y=(2.76543209876543)),
                  vel=Vel(x=(17.469135802469136), y=(1.7283950617283939)),
                  acc=Acc(x=(-2.4691358024691255), y=(-1.7283950617283939)),
                  jerk=Jerk(x=(-3.7037037037036953), y=(-2.5925925925925917))),
            State(pos=Pos(x=(39.617944419041805), y=(3.232561093329265)),
                  vel=Vel(x=(16.493674744703547), y=(1.045572321292486)),
                  acc=Acc(x=(-3.2007315957933145), y=(-2.240512117055321)),
                  jerk=Jerk(x=(-0.4115226337448554), y=(-0.2880658436214034))),
            State(pos=Pos(x=(44.942336026012285), y=(3.459635218208602)),
                  vel=Vel(x=(15.487730528882803), y=(0.341411370217962)),
                  acc=Acc(x=(-2.5605852766346544), y=(-1.7924096936442666)),
                  jerk=Jerk(x=(4.526748971193406), y=(3.168724279835388))),
            State(pos=Pos(x=(50.0), y=(3.5)),
                  vel=Vel(x=(15.0), y=(0.0)),
                  acc=Acc(x=(0.0), y=(0.0)),
                  jerk=Jerk(x=(11.1111111111111), y=(7.777777777777779))),
        ])
        # Valid trajectory
        self.valid_trajectory = Trajectory([
            State(Pos(0, 0), Vel(10, 0), Acc(1, 1), Jerk(1, 0.5)),
            State(Pos(1, 0.1), Vel(11, 0.1), Acc(1.9, 1.5), Jerk(1.2, 0.6)),
            State(Pos(2, 0.2), Vel(12, 0.2), Acc(1.9, 2), Jerk(1.4, 0.7)),
        ])

        # Trajectory with jerk violations
        self.jerk_violation_trajectory = Trajectory([
            State(Pos(0, 0), Vel(10, 0), Acc(1.9, 1), Jerk(5, 1)),
            State(Pos(1, 0.1), Vel(11, 0.1), Acc(2.5, 1.5), Jerk(6, 1.5)),
            State(Pos(2, 0.2), Vel(12, 0.2), Acc(3, 1.9), Jerk(7, 2)),
        ])

        # Trajectory with mixed violations
        self.mixed_violation_trajectory = Trajectory([
            State(Pos(0, 0), Vel(10, 0), Acc(5, 1), Jerk(5, 1)),
            State(Pos(1, 0.1), Vel(11, 0.1), Acc(6, 1.5), Jerk(6, 1.5)),
            State(Pos(2, 0.2), Vel(12, 0.2), Acc(7, 2), Jerk(7, 2)),
        ])

    def test_valid_trajectory(self):
        """Test a trajectory with no violations."""
        results = self.valid_trajectory.measure_quality_of_trajectory(
            self.max_allowed_jerk, self.max_allowed_acc, self.time_step
        )
        self.assertEqual(results["acc_violations"], 0)
        self.assertEqual(results["jerk_violations"], 0)
        self.assertLessEqual(results["max_acc"], self.max_allowed_acc)
        self.assertLessEqual(results["max_jerk"], self.max_allowed_jerk)
        self.assertTrue(results["is_trajectory_valid"])
    def test_jerk_violation_trajectory(self):
        """Test a trajectory with jerk violations."""
        results = self.jerk_violation_trajectory.measure_quality_of_trajectory(
            self.max_allowed_jerk, self.max_allowed_acc, self.time_step
        )
        self.assertEqual(results["acc_violations"], 2)
        self.assertGreater(results["jerk_violations"], 0)
        self.assertFalse(results["is_trajectory_valid"])
        

    def test_mixed_violation_trajectory(self):
        """Test a trajectory with both jerk and acceleration violations."""
        results = self.mixed_violation_trajectory.measure_quality_of_trajectory(
            self.max_allowed_jerk, self.max_allowed_acc, self.time_step
        )
        self.assertGreater(results["acc_violations"], 0)
        self.assertGreater(results["jerk_violations"], 0)
        self.assertFalse(results["is_trajectory_valid"])
        self.assertGreater(results["max_acc"], self.max_allowed_acc)
        self.assertGreater(results["max_jerk"], self.max_allowed_jerk)
        self.assertFalse(results["is_trajectory_valid"])

    def test_provided_trajectory(self):
        """Test the provided trajectory against quality metrics."""
        results = self.real_trajectory.measure_quality_of_trajectory(
            self.max_allowed_jerk, self.max_allowed_acc, self.time_step
        )

        # Verify results
        self.assertAlmostEqual(results["max_acc"], 3.2007, places=1)
        self.assertAlmostEqual(results["max_jerk"], 11.11111, places=2)
        self.assertEqual(results["acc_violations"], 4)
        self.assertEqual(results["jerk_violations"], 2)
        self.assertFalse(results["is_trajectory_valid"])

    def test_trajectory_plot(self):
        pass
        #self.real_trajectory.plot_trajectory()


if __name__ == "__main__":
    unittest.main()
