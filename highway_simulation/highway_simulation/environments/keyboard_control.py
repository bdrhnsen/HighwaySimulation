"""Keyboard control wrapper for manual interaction."""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import pygame

from highway_simulation.scripts.util.action import Action


class KeyboardControlWrapper(gym.Wrapper):
    """Gymnasium wrapper to control actions via keyboard."""

    def __init__(self, env: gym.Env, keyboard_enabled: bool = False) -> None:
        super().__init__(env)
        self.keyboard_enabled = keyboard_enabled
        self.keys: Dict[int, int] = {}
        if self.keyboard_enabled:
            pygame.init()
            self.keys = {
                pygame.K_UP: Action.ACCELERATE.value,
                pygame.K_DOWN: Action.DECELERATE.value,
                pygame.K_LEFT: Action.CHANGE_LANE_LEFT.value,
                pygame.K_RIGHT: Action.CHANGE_LANE_RIGHT.value,
                pygame.K_SPACE: Action.EMERGENCY_BRAKE.value,
            }

    def step(self, action):
        if self.keyboard_enabled:
            action = self.get_keyboard_action()
        return self.env.step(action)

    def get_keyboard_action(self) -> int:
        """Get the action based on keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.keys:
                    return self.keys[event.key]
        return Action.NO_ACTION.value  # Default action if no key is pressed

    def close(self) -> None:
        self.env.close()
        if self.keyboard_enabled:
            pygame.quit()
