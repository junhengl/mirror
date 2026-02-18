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

# Lazy imports — MuJoCo and controller may not be available in all environments
try:
    from .simulation import MuJoCoSimulation
except ImportError:
    MuJoCoSimulation = None

try:
    from .nodes import BodyTrackingNode, RetargetingNode, ControllerNode
except ImportError:
    # Fall back to individual imports if controller pulls in unavailable deps
    try:
        from .nodes.body_tracking_node import BodyTrackingNode
        from .nodes.retargeting_node import RetargetingNode
    except ImportError:
        BodyTrackingNode = None
        RetargetingNode = None
    ControllerNode = None

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
