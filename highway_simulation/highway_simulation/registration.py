"""Gymnasium environment registration helpers."""

from gymnasium.envs.registration import register, registry

ENV_ID = "highway_env"
ENV_ENTRY_POINT = "highway_simulation.environments.relative_to_ego_highway_env:HighwayEnv"


def register_envs() -> None:
    """Register Gymnasium environments if they are not already registered."""
    if ENV_ID in registry:
        return
    register(id=ENV_ID, entry_point=ENV_ENTRY_POINT)
