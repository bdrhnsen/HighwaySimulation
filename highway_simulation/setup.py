"""Package setup configuration."""

from setuptools import find_packages, setup

setup(
    name="highway_simulation",
    version="0.1.1",
    author="Bedirhan Sen",
    author_email="bdrhnsen@gmail.com",
    description="A Python package for highway simulation and reinforcement learning",
    packages=find_packages(),
    install_requires=[
        "gymnasium==0.29.1",
        "matplotlib==3.9.2",
        "pygame==2.6.1",
        "stable_baselines3==2.3.2",
        "torch==2.5.1",
        "scipy==1.15.3"
        "tabulate==0.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "gymnasium.envs": [
            "highway_env=highway_simulation.environments.relative_to_ego_highway_env:HighwayEnv",
        ],
    },
    python_requires=">=3.6",
)
