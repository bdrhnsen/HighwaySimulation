"""Rendering utilities for the highway simulation."""

from __future__ import annotations

import numpy as np
import pygame

from highway_simulation.scripts.laneManager import LaneManager
from highway_simulation.scripts.util.config import Config

class HighwayPlotter:
    config = Config

    @classmethod
    def set_config(cls, config: Config) -> None:
        cls.config = config

    def __init__(self, lane_manager: LaneManager) -> None:
        self.lane_manager = lane_manager
        self.screen = None
        self.meter_to_pixel = 7.5
        self.lane_width = self.config.lane_width * self.meter_to_pixel
        self.road_length = self.config.road_length
        self.vehicle_height = self.config.vehicle_height * self.meter_to_pixel
        self.vehicle_width = self.config.vehicle_width * self.meter_to_pixel
        self.num_lanes = self.config.num_lanes
        self.marking_offset = 0

    def draw(self, screen) -> None:
        """Draw all vehicles in the simulation with proper padding and vehicle height considered."""
        lane_y_padding = 0  # Padding between lanes

        for lane_index, lane in enumerate(self.lane_manager.lanes):
            for vehicle in lane.vehicles:
                x = vehicle.relative_x * self.meter_to_pixel  # stationary with respect to ego
                y = (
                    vehicle.y * self.meter_to_pixel
                    + lane_y_padding * lane_index
                    + (self.lane_width // 2 - self.vehicle_height // 2)
                )
                
                # Create a surface for the vehicle
                vehicle_surface = pygame.Surface(
                    (self.vehicle_width, self.vehicle_height), pygame.SRCALPHA
                )
                pygame.draw.rect(vehicle_surface, vehicle.color, vehicle_surface.get_rect(), border_radius=5)
                
                # Rotate the surface according to the vehicle's heading angle
                rotated_surface = pygame.transform.rotate(
                    vehicle_surface, -np.degrees(vehicle.theta)
                )
                rotated_rect = rotated_surface.get_rect(
                    center=(x + self.vehicle_width / 2, y + self.vehicle_height / 2)
                )
                
                # Draw the rotated vehicle
                screen.blit(rotated_surface, rotated_rect.topleft)

    def draw_trajectory(self, screen) -> None:

        if self.lane_manager.ego_vehicle.trajectory:
            for point in self.lane_manager.ego_vehicle.trajectory.trajectory:
                x = self.lane_manager.ego_vehicle.relative_x * self.meter_to_pixel
                y = point.pos.y * self.meter_to_pixel
                pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 3)

        
        
    def draw_lane_markings(self, screen) -> None:
        """Draw dashed lane markings between lanes with a minimalistic style."""
        dash_length = 15  # Length of each dash
        dash_gap = 15     # Gap between dashes
        line_width = 2    # Line width for dashed markings
        
        ego_speed = self.lane_manager.ego_vehicle.speed / 3 * self.config.time_step
        self.marking_offset = (self.marking_offset + ego_speed) % (dash_length + dash_gap)

        for lane_index in range(1, self.num_lanes):
            y = lane_index * self.lane_width  # Center line position between lanes
            start_x = -self.marking_offset
            for x in range(int(start_x), int(self.road_length), dash_length + dash_gap):
                pygame.draw.line(screen, self.config.colors["WHITE"], (x, y), (x + dash_length, y), line_width)

    def draw_x_axis(self, screen) -> None:
        """Draw the x-axis below all lanes."""
        x_axis_y = self.config.screen_height - 50  # Position of the x-axis at the bottom of the screen
        tick_interval = 50  # Distance between ticks on the x-axis

        # Draw the horizontal line for the x-axis
        pygame.draw.line(screen, (0, 0, 0), (0, x_axis_y), (self.road_length, x_axis_y), 2)

        # Draw tick marks and labels at regular intervals
        for x in range(0, int(self.road_length), tick_interval):
            # Draw the tick mark
            pygame.draw.line(screen, (0, 0, 0), (x, x_axis_y - 5), (x, x_axis_y + 5), 1)
            
            # Render the label
            font = pygame.font.SysFont(None, 24)
            label = font.render(str(x), True, (0, 0, 0))
            screen.blit(label, (x, x_axis_y + 10))

    def draw_lane_statistics(self, screen, font) -> None:
        """Draw lane statistics as text boxes."""
        lane_stats = self.lane_manager.calculate_lane_statistics()
        for i, (vehicle_count, avg_speed, avg_time_gap, how_far_to_ego) in enumerate(lane_stats):
            text_surface = font.render(f'Lane {i+1}: {vehicle_count} vehicles, Avg speed: {avg_speed:.1f}, Avg time gap: {avg_time_gap:.1f}, How far is ego: {how_far_to_ego} ', True, (0, 0, 0))
            pygame.draw.rect(screen, (255, 255, 255), (10, i * self.lane_width + 10, 400, 40))  # Background for text
            screen.blit(text_surface, (20, i * self.lane_width + 20))
    def draw_ego_info(self, screen, font) -> None:
        """Draw information about the ego vehicle in a HUD."""
        ego = self.lane_manager.ego_vehicle
        lane_changes = self.lane_manager.ego_lane_changes
        
        # Gather relevant information
        ego_info_text = (
            f"Position: {ego.x:.1f} | Speed: {ego.speed * 3.6:.1f} km/h | Lane: {ego.lane + 1} | "
            f"Lane Changes: {lane_changes}"
        )
        
        # Render the text on a semi-transparent background at the top of the screen
        text_surface = font.render(ego_info_text, True, (200, 0, 0))
        background_rect = pygame.Surface((self.config.screen_width, 25))
        background_rect.set_alpha(180)  # Set transparency
        background_rect.fill((255, 255, 255))  # White background
        dynamic_screen_height = (self.num_lanes + 1) * self.lane_width
        screen.blit(background_rect, (0, dynamic_screen_height - 25))  # Draw background at top of screen
        screen.blit(text_surface, (10, dynamic_screen_height - 25))    # Draw text slightly offset from left edge

    def render(self) -> None:
        # Dynamically set the screen height based on number of lanes
        dynamic_screen_height = (self.num_lanes + 1) * self.lane_width

        # Initialize Pygame and screen if not already done
        if self.screen is None:
            pygame.init()
            self.font = pygame.font.SysFont("Arial", 10)
            self.screen = pygame.display.set_mode(
                (self.config.screen_width, dynamic_screen_height), 
                pygame.SRCALPHA, 
                display=0
            )
        pygame.display.set_caption("Highway Simulation")
        self.clock = pygame.time.Clock()
        self.screen.fill(self.config.colors["GRAY"])
        self.draw_lane_markings(self.screen)
        #self.draw_x_axis(self.screen)
        self.draw(self.screen)

        #self.draw_lane_statistics(self.screen,self.font)
        self.draw_ego_info(self.screen, self.font)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
