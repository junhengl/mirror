"""
Real-time Simulation Pipeline

Main entry point for the integrated simulation with:
- MuJoCo physics simulation (1kHz)
- PD torque control with FSM (1kHz)
- IK-based retargeting (500Hz)
- ZED body tracking (30Hz)

Usage:
    sudo /path/to/.venv/bin/python -m real_time_sim.main
"""

from .config import PipelineConfig, DEFAULT_CONFIG
from .shared_state import SharedState, RobotState
from .simulation import MuJoCoSimulation
from .nodes import BodyTrackingNode, RetargetingNode, ControllerNode

__all__ = [
    'PipelineConfig',
    'DEFAULT_CONFIG',
    'SharedState',
    'RobotState',
    'MuJoCoSimulation',
    'BodyTrackingNode',
    'RetargetingNode',
    'ControllerNode',
]
