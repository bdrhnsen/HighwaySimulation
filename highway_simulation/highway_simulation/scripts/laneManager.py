"""Lane management and vehicle updates."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from highway_simulation.scripts.lane import Lane
from highway_simulation.scripts.planning.decision_to_trajectory import DecisionToTrajectory
from highway_simulation.scripts.reset.highwayHelper import HighwayHelper
from highway_simulation.scripts.util.config import Config
from highway_simulation.scripts.vehicle.vehicle import Vehicle
from highway_simulation.testing.highwayTestCases import HighwayTestCases

class LaneManager:
    config = Config

    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def __init__(self) -> None:
        self.road_length = self.config.road_length
        self.vehicle_width = self.config.vehicle_width
        self.num_lanes = self.config.num_lanes
        self.lanes = [
            Lane(id=i, road_length=self.road_length, vehicle_width=self.vehicle_width)
            for i in range(self.num_lanes)
        ]
        self.lane_change_in_progress = False

        highway_test_cases = HighwayTestCases(self.config)
        self.test_cases = highway_test_cases.define_test_cases()
        self.highway_helper = HighwayHelper(self.config)
        self.decision_to_trajectory = DecisionToTrajectory()
        self.num_of_vehicles = self.config.num_of_vehicles

        ## For metrics
        self.ego_lane_changes = 0
        self.avg_speed_of_all_vehicles = 0
        self.avg_lane_change_per_vehicle = 0
        self.avg_time_gap_per_lane = 0
        self.time_in_lanes = {i: 0 for i in range(self.num_lanes)}  # Time spent in each lane

    def update_time_in_lane(self) -> None:
        self.time_in_lanes[self.ego_vehicle.lane] += 1

    def add_vehicle(self, vehicle: Vehicle) -> None:
        self.lanes[vehicle.lane].vehicles.append(vehicle)
        if vehicle.is_ego:
            self.ego_vehicle = vehicle

    def destroy_vehicle(self, vehicle: Vehicle) -> None:
        self.lanes[vehicle.lane].vehicles.remove(vehicle)

    def update_lane_attributes(self) -> None:
        assert hasattr(self, "ego_vehicle")
        ego_lane_id = self.ego_vehicle.lane
        for lane in self.lanes:
            lane.how_far_to_ego = lane.id - ego_lane_id

    def update(self) -> None:  # MAIN LOOP OF SIMULATION
        """Update vehicle positions and clear lanes for re-sorting."""
        self.find_ahead_vehicles()
        if self.config.ego_drives_with_mobil:
            self.update_positions_mobil()
        else:
            self.update_positions_relative_to_ego()
        self.reset_positions_wrt_ego()
        self.check_relative_x()
        self.update_lane_attributes()
        self.update_non_ego_lane_changes()
        self.update_statistics()
    
    ## TO USE IN MOBIL ALGORITHM
    def find_vehicle_ahead(self, vehicle: Vehicle, lane: int) -> Optional[Vehicle]:
        """Find the vehicle ahead in the specified lane."""
        vehicles_in_lane = sorted(self.lanes[lane].vehicles, key=lambda v: v.x)
        if vehicle.lane == self.lanes[lane].id:
            vehicles_in_lane.remove(vehicle)
        for v in vehicles_in_lane:
            if v.x > vehicle.x:
                if v.x - vehicle.x > 150: # if it is too far away it does not count as a vehicle ahead
                    return None
                return v
            
            
        return None

    def find_vehicle_behind(self, vehicle: Vehicle, lane: int) -> Optional[Vehicle]:
        """Find the vehicle behind in the specified lane."""
        vehicles_in_lane = sorted(self.lanes[lane].vehicles, key=lambda v: v.x, reverse=True)
        if vehicle.lane == self.lanes[lane].id:
            vehicles_in_lane.remove(vehicle)
        for v in vehicles_in_lane:
            if v.x < vehicle.x:
                if v.x - vehicle.x < -150:
                    return None
                return v
        return None

    def update_non_ego_lane_changes(self) -> None:
        """Update lane changes for non-ego vehicles using the MOBIL model."""
        for lane in self.lanes:
            for vehicle in lane.vehicles:
                if not self.config.ego_drives_with_mobil and vehicle.is_ego:
                    continue
                if not vehicle.trajectory.is_trajectory_empty():
                    continue

                vehicle_ahead_current = self.find_vehicle_ahead(vehicle, vehicle.lane)
                vehicle_behind_current = self.find_vehicle_behind(vehicle, vehicle.lane)
                left_lane = vehicle.lane - 1 if vehicle.lane > 0 else None
                right_lane = vehicle.lane + 1 if vehicle.lane < self.num_lanes - 1 else None

                ## USE LEFT LANE ONLY FOR TAKEOVER
                if not (vehicle.is_ego and self.config.aggresive_driver):
                    if vehicle.lane == 0 and right_lane is not None:
                        vehicle_ahead_target = self.find_vehicle_ahead(vehicle, right_lane)
                        vehicle_behind_target = self.find_vehicle_behind(
                            vehicle, right_lane
                        )
                        if (
                            vehicle_ahead_target is not None
                            and abs(vehicle_ahead_target.x - vehicle.x) > 100
                        ) or vehicle_ahead_target is None:
                            trajectory = (
                                self.decision_to_trajectory.calculate_lane_change_trajectory(
                                    vehicle, right_lane
                                )
                            )
                            vehicle.target_lane = right_lane
                            vehicle.trajectory = trajectory
                            vehicle.ongoing_trajectory = True
                            vehicle.v_max = vehicle.initial_v_max
                            continue
                        else:
                            vehicle.v_max += 0.05

                ## MOBIL    
                for target_lane in [left_lane, right_lane]:
                    if target_lane is not None and target_lane in range(self.num_lanes):
                        vehicle_ahead_target = self.find_vehicle_ahead(vehicle, target_lane)
                        vehicle_behind_target = self.find_vehicle_behind(vehicle, target_lane)
                        if vehicle.calculate_mobil_lane_change(
                            vehicle_ahead_current,
                            vehicle_behind_current,
                            vehicle_ahead_target,
                            vehicle_behind_target,
                        ):
                            trajectory = (
                                self.decision_to_trajectory.calculate_lane_change_trajectory(
                                    vehicle, target_lane
                                )
                            )
                            vehicle.target_lane = target_lane
                            vehicle.trajectory = trajectory
                            vehicle.ongoing_trajectory = True
                            break

    ###### END OF MOBIL ALGORITHM

    def find_ahead_vehicles(self) -> None:
        """adds ahead vehicle object to vehicles. If ahead vehicle does not exist it assigns None """
        for lane in self.lanes:
            lane.vehicles.sort(key=lambda v: v.x)
            if lane.vehicles:
                for i in range(len(lane.vehicles) - 1):
                    lane.vehicles[i].vehicle_ahead = lane.vehicles[i + 1]
                lane.vehicles[-1].vehicle_ahead = None

    def reset_positions_wrt_ego(self) -> None:
        assert hasattr(self, "ego_vehicle")
        for lane in self.lanes:
            for vehicle in lane.vehicles:
                if not vehicle.is_ego:
                    pos_dif = vehicle.x - self.ego_vehicle.x
                    vehicle.relative_x = self.ego_vehicle.relative_x + pos_dif

    def check_relative_x(self) -> None:
        assert hasattr(self, "ego_vehicle")
        for lane in self.lanes:
            for vehicle in lane.vehicles:
                if not vehicle.is_ego:
                    if round(vehicle.relative_x - self.ego_vehicle.relative_x) != round(
                        vehicle.x - self.ego_vehicle.x
                    ):
                        self.reset_positions_wrt_ego()
                        print("Exception(Relative Position wrt ego does not work")
                        break

    def update_positions_mobil(self) -> None:
        for lane in self.lanes:
            for vehicle in lane.vehicles:
                # if not self.is_in_update_range(vehicle):
                #     continue ## only update vehicles that are in the range of the ego vehicle
                vehicle.update_ego_driven_with_mobil()
                if vehicle.trajectory_completed:
                    self.handle_trajectory_complete(vehicle)
            lane.vehicles = [v for v in lane.vehicles if self.is_in_range(v)]

    def update_positions_relative_to_ego(self) -> None:
        assert hasattr(self, "ego_vehicle")
        self.ego_vehicle.update()
        if self.ego_vehicle.trajectory_completed:
            self.handle_trajectory_complete(self.ego_vehicle)
            self.lane_change_in_progress = False

        for lane in self.lanes:
            for vehicle in lane.vehicles:
                # if not self.is_in_update_range(vehicle):
                #     continue ## only update vehicles that are in the range of the ego vehicle
                if not vehicle.is_ego:
                    vehicle.update()
                    if vehicle.trajectory_completed:
                        self.handle_trajectory_complete(vehicle)
            lane.vehicles = [v for v in lane.vehicles if self.is_in_range(v)]

    def handle_trajectory_complete(self, vehicle: Vehicle) -> None:
        self.lanes[vehicle.lane].vehicles.remove(vehicle)
        self.lanes[vehicle.target_lane].vehicles.append(vehicle)
        self.lanes[vehicle.target_lane].vehicles[-1].lane = vehicle.target_lane
        self.lanes[vehicle.target_lane].vehicles[-1].trajectory_completed = False
        
        if vehicle.is_ego:
            self.lane_change_in_progress = False
            self.ego_lane_changes += 1
            self.ego_vehicle.lane = self.ego_vehicle.target_lane
            self.ego_vehicle.trajectory_completed = False
        #print(f"Vehicle {vehicle} has completed its trajectory and changed lanes.")
    
    def get_nearby_vehicles(self, num_vehicles: int) -> List[Vehicle]:
        assert hasattr(self, "ego_vehicle")
        vehicles = []
        for lane in self.lanes:
            vehicles.extend(lane.vehicles)
        # Exclude the ego vehicle and sort by distance to the ego
        vehicles = [v for v in vehicles if not v.is_ego]
        vehicles.sort(key=lambda v: abs(v.relative_x - self.ego_vehicle.relative_x))
        
        # Return the closest `num_vehicles` vehicles
        #print(len(vehicles))
        return vehicles[:num_vehicles]


    ## USED IN THE COLLISION CALCULATION
    def find_front_back_vehicles(self, lane: Lane) -> Tuple[Optional[Vehicle], Optional[Vehicle]]:
        assert hasattr(self, "ego_vehicle")
        vehicle_list = [v for v in lane.vehicles if not v.is_ego and self.is_in_range(v)]
        if not vehicle_list:
            return None, None
        if self.ego_vehicle in vehicle_list:
            vehicle_list.remove(self.ego_vehicle)
        sorted_vehicles = sorted(vehicle_list, key=lambda vehicle: abs(vehicle.x - self.ego_vehicle.x))
        #sorts from small to large
        front, back = None, None
        for vehicle in sorted_vehicles:
            if vehicle.x - self.ego_vehicle.x > 0:
                front = vehicle
            else:
                back = vehicle
            if front is not None and back is not None:
                break
        return front, back

    def is_ego_in_leftmost_lane(self) -> bool:
        return self.ego_vehicle.lane == 0

    def is_ego_in_rightmost_lane(self) -> bool:
        return self.ego_vehicle.lane == (self.config.num_lanes - 1)

    def is_in_range(self, vehicle: Vehicle) -> bool:
        assert hasattr(self, "ego_vehicle")
        return abs(self.ego_vehicle.x - vehicle.x) < 20000

    def is_in_update_range(self, vehicle: Vehicle) -> bool:
        assert hasattr(self, "ego_vehicle")
        return abs(self.ego_vehicle.x - vehicle.x) < 1000

    def calculate_lane_statistics(self) -> List[Tuple[int, float, float, int]]:
        """Return statistics for each lane (number of vehicles and average speed)."""
        stats = [
            (lane.num_vehicles, lane.avg_speed, lane.avg_time_gap, lane.how_far_to_ego)
            for lane in self.lanes
        ]
        return stats


    def add_vehicles_to_sim_from_test_case(self):
        test_case_name, vehicle_list = self.test_cases[0]

        for vehicle in vehicle_list:
            self.add_vehicle(vehicle)
        self.test_cases.pop(0)
        return test_case_name

    def add_vehicles_to_sim(self, seed: int, no_vehicles: Optional[bool]) -> None:
        vehicle_list = self.highway_helper.generate_vehicle_list(
            seed, self.num_of_vehicles, no_vehicles
        )
        for vehicle in vehicle_list:
            self.add_vehicle(vehicle)

    def remove_all_vehicles(self) -> None:
        for lane in self.lanes:
            lane.vehicles = []
    

    def update_statistics(self) -> None:
        lane_avg_speed = [lane.avg_speed for lane in self.lanes if lane.avg_speed != 0.0]
        lane_avg_time_gap = [lane.avg_time_gap for lane in self.lanes if lane.avg_time_gap != 0.0]

        self.avg_speed_of_all_vehicles = sum(lane_avg_speed) / len(lane_avg_speed) if lane_avg_speed else 0
        self.avg_time_gap_per_lane = sum(lane_avg_time_gap) / len(lane_avg_time_gap) if lane_avg_time_gap else 0
        self.update_time_in_lane()
    """
    def update_lanes(self):
        for lane in self.lanes:
            lane.sort_vehicles()  # Ensure vehicles are sorted by position
    
    def get_lane(self, lane_id):
        return self.lanes[lane_id] if lane_id < len(self.lanes) else None

    def find_ahead_and_behind(self, lane_id, vehicle_x):
        lane = self.get_lane(lane_id)
        return lane.get_vehicle_ahead_and_behind(vehicle_x) if lane else (None, None)
    """
