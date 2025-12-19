"""Scenario definitions for highway test cases."""

from __future__ import annotations

from typing import List, Tuple

from highway_simulation.scripts.vehicle.vehicle import Vehicle


class HighwayTestCases:
    """Create deterministic scenario lists for testing."""

    def __init__(self, config) -> None:
        self.config = config
        self.test_cases: List[Tuple[str, List[Vehicle]]] = []

    def define_test_cases(self) -> List[Tuple[str, List[Vehicle]]]:

        ## Front Vehicle slowing down.
        front_vehicle = Vehicle(1500, 0, 70, 50)
        front_vehicle = Vehicle(700, 1, 70, 80)
        ego_vehicle = Vehicle(100, 1, 70, 70, is_ego=True)

        self.test_cases.append(("FrontVehicleSlowingDown", [front_vehicle, ego_vehicle]))
        
        ## Back vehicle speeding up (Reward function does not include this)
        
        back_vehicle = Vehicle(200, 1, 70, 70)
        ego_vehicle = Vehicle(400, 1, 70, 90, is_ego=True)

        self.test_cases.append(("BackVehicleSpeedingUp", [back_vehicle, ego_vehicle]))
        
        ## Ego is at right There are 2 front vehicles in middle and right lane. See if it will do a double lane change and pass them
        front_vehicle_same_lane = Vehicle(700,1,70,50)
        front_vehicle_next_lane = Vehicle(700,2,70,50)
        ego_vehicle = Vehicle(500, 2, 70, 70, is_ego=True)

        self.test_cases.append(("EncourgeDoubleLaneChange", [front_vehicle_same_lane, front_vehicle_next_lane, ego_vehicle]))

        ##Traffic Jam Ahead
        stopped_vehicle_1 = Vehicle(1800, 1, 10, 10)
        stopped_vehicle_2 = Vehicle(1850, 0, 10, 10)
        stopped_vehicle_2 = Vehicle(1850, 2, 10, 10)

        ego_vehicle = Vehicle(600, 1, 70, 70, is_ego=True)

        self.test_cases.append(("TrafficJamAhead", [stopped_vehicle_1, stopped_vehicle_2, ego_vehicle]))

        ## Single lane blocked ahead
        blocked_vehicle = Vehicle(600, 1, 5, 5)  # Stopped vehicle blocking lane
        ego_vehicle = Vehicle(200, 1, 70, 70, is_ego=True)

        self.test_cases.append(("LaneBlockedAhead", [blocked_vehicle, ego_vehicle]))

        ## Complex lane change
        veh_in_lane_0 = Vehicle(600, 0, 90, 90)
        veh_in_lane_1 = Vehicle(900, 1, 90, 90)
        veh_in_lane_2 = Vehicle(1200, 2, 90, 90)      
        ego_vehicle = Vehicle(200, 1, 70, 70, is_ego=True)

        self.test_cases.append(("ComplexLaneChange", [veh_in_lane_0,veh_in_lane_1,veh_in_lane_2, ego_vehicle]))
      
        ## Complex lane change2
        veh_in_lane_0 = Vehicle(600, 2, 90, 90)
        veh_in_lane_1 = Vehicle(900, 1, 90, 90)
        veh_in_lane_2 = Vehicle(700, 0, 90, 90)      
        ego_vehicle = Vehicle(200, 1, 70, 70, is_ego=True)

        self.test_cases.append(("ComplexLaneChange2", [veh_in_lane_0,veh_in_lane_1,veh_in_lane_2, ego_vehicle]))
       
        ## Complex lane change3
        veh_in_lane_0 = Vehicle(600, 2, 90, 90)
        veh_in_lane_0_1 = Vehicle(700, 2, 90, 90)
        veh_in_lane_1 = Vehicle(900, 1, 90, 90)
        veh_in_lane_1_1 = Vehicle(450, 1, 90, 90)
        veh_in_lane_2 = Vehicle(600, 0, 90, 90)
        veh_in_lane_2_1 = Vehicle(700, 0, 90, 90)      
        ego_vehicle = Vehicle(50, 1, 70, 70, is_ego=True)

        self.test_cases.append(("ComplexLaneChange3", [veh_in_lane_0,veh_in_lane_0_1,veh_in_lane_1,veh_in_lane_1_1,veh_in_lane_2,veh_in_lane_2_1, ego_vehicle]))

        ## Complex lane change4
        veh_in_lane_0 = Vehicle(400, 2, 90, 70)
        veh_in_lane_0_1 = Vehicle(700, 2, 90, 90)
        veh_in_lane_1 = Vehicle(900, 1, 90, 70)
        veh_in_lane_1_1 = Vehicle(350, 1, 90, 95)
        veh_in_lane_2 = Vehicle(800, 0, 90, 90)
        veh_in_lane_2_1 = Vehicle(1500, 0, 90, 130)      
        ego_vehicle = Vehicle(50, 1, 70, 70, is_ego=True)

        self.test_cases.append(("ComplexLaneChange4", [veh_in_lane_0,veh_in_lane_0_1,veh_in_lane_1,veh_in_lane_1_1,veh_in_lane_2,veh_in_lane_2_1, ego_vehicle]))

        return self.test_cases
