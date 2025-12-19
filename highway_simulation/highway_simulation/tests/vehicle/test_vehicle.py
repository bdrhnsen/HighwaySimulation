"""Tests for vehicle dynamics helpers."""

import math
import unittest

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.util.config import Config, default_config
from highway_simulation.scripts.vehicle.vehicle import IDM_PARAM, MOBIL_PARAM, Vehicle


class TestVehicle(unittest.TestCase):

    def setUp(self):
        self.config: Config = default_config
        Vehicle.set_config(self.config)
        LaneManager.set_config(self.config)
        self.vehicle = Vehicle(x=50, lane=1, speed=80, v_max=80, is_ego=True)
        self.mobil_param = MOBIL_PARAM(politeness=0.5, a_thr=0.2, b_safe=2.0)
        self.idm_param = IDM_PARAM(a_max=0.7, s0=2.0, T=1.6, b=1.7, delta=4)
        self.lane_manager = LaneManager()
    def test_initialization(self):
        self.assertEqual(self.vehicle.x, 50)
        self.assertEqual(self.vehicle.lane, 1)
        self.assertAlmostEqual(self.vehicle.speed, 80 * 10 / 36)  # Converted to m/s
        self.assertTrue(self.vehicle.is_ego)

    def test_update(self):
        vehicle = Vehicle(x=50, lane=1, speed=50.0, v_max=50.0, is_ego=True)
        dt = self.config.time_step
        vehicle_final_pos = vehicle.x + (vehicle.speed+vehicle.acc*dt) * dt+ 0.5 * vehicle.acc* dt**2
        vehicle.update()
        self.assertEqual(vehicle.x, vehicle_final_pos)


    def test_free_road_acceleration(self):
        ## TEST_CASE: vehicle has no vehicles ahead and wants to speed up from 50 to 60
        
        vehicle = Vehicle(x=50, lane= 1, speed=50.0, v_max = 60.0, idm_param= self.idm_param, is_ego =False)
        vehicle.vehicle_ahead = None
        accel_should_be = self.idm_param.a_max* (1 - (vehicle.speed/vehicle.v_max)**self.idm_param.delta)
        accel=vehicle.calculate_accel(vehicle.vehicle_ahead)
        self.assertEqual(accel, accel_should_be)

    def test_vehicle_ahead_acceleration(self):
        ## TEST_CASE: vehicle has a vehicle 50m ahead going same speed as the vehicle
        
        vehicle = Vehicle(x=50, lane= 1, speed=50.0, v_max = 50.0, idm_param= self.idm_param, is_ego =False)
        vehicle2 = Vehicle(x=100, lane= 1, speed=50.0, v_max = 50.0, idm_param= self.idm_param, is_ego =False)
        vehicle.vehicle_ahead = vehicle2

        no_vehicle_term = self.idm_param.a_max* (1 - (vehicle.speed/vehicle.v_max)**self.idm_param.delta)
        vehicle_term = self.idm_param.s0 + max(0, vehicle.speed*self.idm_param.T + vehicle.speed*(vehicle.speed-vehicle.vehicle_ahead.speed)/(2*math.sqrt(self.idm_param.a_max*self.idm_param.b)))
        accel_should_be = no_vehicle_term -self.idm_param.a_max* (vehicle_term/(vehicle.vehicle_ahead.x - vehicle.x- vehicle.length))**2
        accel=vehicle.calculate_accel(vehicle.vehicle_ahead)
        self.assertEqual(accel, accel_should_be)
    
    def test_vehicle_ahead_slow_acceleration(self):
        ## TEST_CASE: vehicle has a vehicle 50m ahead going slower
        
        vehicle = Vehicle(x=50, lane= 1, speed=50.0, v_max = 50.0, idm_param= self.idm_param, is_ego =False)
        vehicle2 = Vehicle(x=100, lane= 1, speed=40.0, v_max = 40.0, idm_param= self.idm_param, is_ego =False)
        vehicle.vehicle_ahead = vehicle2

        no_vehicle_term = self.idm_param.a_max* (1 - (vehicle.speed/vehicle.v_max)**self.idm_param.delta)
        vehicle_term = self.idm_param.s0 + max(0, vehicle.speed*self.idm_param.T + vehicle.speed*(vehicle.speed-vehicle.vehicle_ahead.speed)/(2*math.sqrt(self.idm_param.a_max*self.idm_param.b)))
        accel_should_be = no_vehicle_term -self.idm_param.a_max* (vehicle_term/(vehicle.vehicle_ahead.x - vehicle.x- vehicle.length))**2
        accel=vehicle.calculate_accel(vehicle.vehicle_ahead)
        self.assertEqual(accel, accel_should_be)

    def test_extremely_slow_vehicle_ahead(self):
        vehicle = Vehicle(x=50, lane= 1, speed=50.0, v_max = 50.0, idm_param= self.idm_param, is_ego =False)
        vehicle2 = Vehicle(x=60, lane= 1, speed=1.0, v_max = 1.0, idm_param= self.idm_param, is_ego =False)
        vehicle.vehicle_ahead = vehicle2

        no_vehicle_term = self.idm_param.a_max* (1 - (vehicle.speed/vehicle.v_max)**self.idm_param.delta)
        vehicle_term = self.idm_param.s0 + max(0, vehicle.speed*self.idm_param.T + vehicle.speed*(vehicle.speed-vehicle.vehicle_ahead.speed)/(2*math.sqrt(self.idm_param.a_max*self.idm_param.b)))
        accel_should_be = no_vehicle_term -self.idm_param.a_max* (vehicle_term/(vehicle.vehicle_ahead.x - vehicle.x- vehicle.length))**2
        # This is a fail safe for extremely negative accelerations 
        accel_should_be = max(-self.idm_param.b,accel_should_be)
        accel=vehicle.calculate_accel(vehicle.vehicle_ahead)
        self.assertEqual(accel, accel_should_be)

   
if __name__ == "__main__":
    unittest.main()
