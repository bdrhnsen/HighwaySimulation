"""Translate discrete decisions into trajectories."""

from __future__ import annotations

from highway_simulation.scripts.planning.state import Acc, Pos, State, Trajectory, Vel
from highway_simulation.scripts.planning.trajectory_planner import TrajectoryPlanner
from highway_simulation.scripts.util.action import Action
from highway_simulation.scripts.util.config import Config


class DecisionToTrajectory:
    config = Config

    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def __init__(self) -> None:
        TrajectoryPlanner.set_config(self.config)
        self.trajectory_planner = TrajectoryPlanner()

    def calculate_lane_change_end_state(
        self, current_state: State, y_diff: int
    ) -> State:
        """
        lane0 y0
        lane 1 y3.5
        lane 2 y7
        Calculates the end state with current state and target lane

        """
        T = 3
        x0, y0, vx0, vy0, accx0, accy0, _, _ = current_state.extract()

        ## longitudinal
        xT = x0 + vx0 * T
        vxT = vx0
        accxT = 0

        ## lateral
        yT = y0 + y_diff * self.config.lane_width
        vyT = 0
        accyT = 0
        
        return State(Pos(xT, yT), Vel(vxT, vyT), Acc(accxT, accyT))
    
    def calculate_end_state(self, current_state: State, delta_speed: float) -> State:
        """
        lane0 y0
        lane 1 y3.5
        lane 2 y7
        Calculates the end state with current state and target lane

        """
        T = 3
        x0, y0, vx0, vy0, accx0, accy0, _, _ = current_state.extract()

        ## longitudinal
        xT = x0 + 0.5 * (vx0 + (vx0 + delta_speed)) * T  # avg_speed * time
        vxT = vx0 + delta_speed
        accxT = 0

        ## lateral
        yT = y0
        vyT = 0
        accyT = 0
        
        return State(Pos(xT, yT), Vel(vxT, vyT), Acc(accxT, accyT))
    
    def calculate_lane_change_trajectory(self, vehicle, target_lane: int) -> Trajectory:
        """
        to be used for non ego vehicles, lane_change_delta means the lane change direction
        1 for right, -1 for left

        """
        current_state = vehicle.return_state
        lane_change_delta = target_lane - vehicle.lane
        end_state = self.calculate_lane_change_end_state(current_state, lane_change_delta)
        return self.trajectory_planner.plan_between_points(current_state, end_state)
       
    def process_decision(self, current_state: State, action: Action) -> Trajectory:
        #action = Action.CHANGE_LANE_LEFT ## manual override TODO

        if action == Action.CHANGE_LANE_LEFT:  # Left lane change
            end_state = self.calculate_lane_change_end_state(current_state, -1)
        elif action == Action.CHANGE_LANE_RIGHT:  # Right lane change
            end_state = self.calculate_lane_change_end_state(current_state, 1)
            
        elif action == Action.NO_ACTION:  
            end_state = self.calculate_end_state(current_state, 0)
            
        elif action == Action.ACCELERATE:
            end_state = self.calculate_end_state(current_state, 5)
           
        elif action == Action.DECELERATE:
            end_state = self.calculate_end_state(current_state, -5)
            
        elif action == Action.EMERGENCY_BRAKE:
            end_state = self.calculate_end_state(current_state, -9)
            
        return self.trajectory_planner.plan_between_points(current_state, end_state)
