"""
Configuration parameters for real-time simulation pipeline.

Defines rates, gains, and shared parameters across all nodes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimConfig:
    """MuJoCo simulation configuration."""
    model_path: str = "westwood_robots/TH02-A7-torque.xml"  # Model with actuators
    sim_dt: float = 0.001  # 1kHz simulation timestep
    render_fps: float = 60.0  # Visualization framerate
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    base_height: float = 1.3  # Robot hanging height (meters) - must clear feet
    

@dataclass
class ControlConfig:
    """Controller configuration."""
    control_rate: float = 1000.0  # 1kHz control loop
    control_dt: float = 0.001
    
    # PD gains per joint group [kp, kd]
    # Lower Kp and HIGHER Kd for stability when hanging (damping is critical!)
    leg_gains: np.ndarray = field(default_factory=lambda: np.array([50.0, 1.0]))  # [kp, kd]
    arm_gains: np.ndarray = field(default_factory=lambda: np.array([70.0, 1.0]))
    head_gains: np.ndarray = field(default_factory=lambda: np.array([10.0, 1.0]))
    
    # Torque limits (Nm)
    leg_torque_limit: float = 100.0
    arm_torque_limit: float = 30.0
    head_torque_limit: float = 10.0
    
    # Joint indices (0-based, after floating base)
    leg_r_indices: List[int] = field(default_factory=lambda: list(range(0, 6)))
    leg_l_indices: List[int] = field(default_factory=lambda: list(range(6, 12)))
    arm_r_indices: List[int] = field(default_factory=lambda: list(range(12, 19)))
    arm_l_indices: List[int] = field(default_factory=lambda: list(range(19, 26)))
    head_indices: List[int] = field(default_factory=lambda: list(range(26, 28)))


@dataclass
class RetargetingConfig:
    """Retargeting node configuration."""
    retarget_rate: float = 500.0  # 500Hz retargeting
    retarget_dt: float = 0.002
    num_ik_iterations: int = 1
    
    # Arm scale factor (human -> robot)
    arm_scale: float = 0.9
    
    # IK weights
    position_weight: float = 100.0
    regularization_weight: float = 0.1


@dataclass
class TrackingConfig:
    """Body tracking node configuration."""
    tracking_rate: float = 30.0  # 30Hz ZED tracking
    tracking_dt: float = 0.033
    
    # Confidence thresholds
    min_confidence: float = 30.0
    
    # Filter parameters
    filter_alpha: float = 0.4
    jump_threshold: float = 0.12  # meters


@dataclass 
class FSMConfig:
    """Finite State Machine configuration."""
    # State transition times (seconds)
    init_duration: float = 2.0  # Time to hold init pose
    blend_duration: float = 2.0  # Longer transition blend time for smooth tracking start
    
    # Safety limits (very permissive for hanging robot - physics provides natural damping)
    max_joint_velocity: float = 500.0  # rad/s - high because hanging robot is naturally damped
    max_tracking_error: float = 0.5  # meters


@dataclass
class PipelineConfig:
    """Master configuration for entire pipeline."""
    sim: SimConfig = field(default_factory=SimConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    retarget: RetargetingConfig = field(default_factory=RetargetingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    
    # Verbose/debug flags
    verbose: bool = False
    log_timing: bool = True


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
