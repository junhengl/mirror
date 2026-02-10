"""
Thread-safe shared state management for inter-node communication.

Provides lock-protected data structures for passing information between:
- Body tracking node (30 Hz) -> Retargeting node
- Retargeting node (500 Hz) -> Controller node  
- Controller node (1 kHz) <-> Simulation
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


class RobotState(Enum):
    """FSM states for robot control."""
    INIT = auto()           # Initial startup, moving to default pose
    IDLE = auto()           # Holding default pose, waiting for tracking
    TRACKING = auto()       # Active tracking and retargeting
    SAFETY_STOP = auto()    # Safety triggered, holding position
    SHUTDOWN = auto()       # Shutting down


@dataclass
class ArmTrackingData:
    """Container for arm tracking data from ZED (thread-safe copy)."""
    timestamp: float = 0.0
    valid: bool = False
    # Positions in local body frame (relative to COM)
    body_com: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    left_shoulder: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    left_elbow: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    left_wrist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    left_confidence: float = 0.0
    right_shoulder: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    right_elbow: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    right_wrist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    right_confidence: float = 0.0
    
    def copy(self) -> 'ArmTrackingData':
        """Create a deep copy."""
        return ArmTrackingData(
            timestamp=self.timestamp,
            valid=self.valid,
            body_com=self.body_com.copy(),
            left_shoulder=self.left_shoulder.copy(),
            left_elbow=self.left_elbow.copy(),
            left_wrist=self.left_wrist.copy(),
            left_confidence=self.left_confidence,
            right_shoulder=self.right_shoulder.copy(),
            right_elbow=self.right_elbow.copy(),
            right_wrist=self.right_wrist.copy(),
            right_confidence=self.right_confidence,
        )


@dataclass
class RetargetingOutput:
    """Output from retargeting node: desired joint positions."""
    timestamp: float = 0.0
    valid: bool = False
    q_des: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float64))  # Desired joint angles
    dq_des: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float64))  # Desired joint velocities
    
    # Task space targets for visualization (desired)
    hand_l_des: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hand_r_des: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elbow_l_des: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elbow_r_des: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    
    # Task space actual positions (from FK)
    hand_l_act: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hand_r_act: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elbow_l_act: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elbow_r_act: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    
    # Hand orientation matrices (3x3 rotation, Z-axis = elbow→hand direction)
    hand_l_orient_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    hand_r_orient_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    
    def copy(self) -> 'RetargetingOutput':
        """Create a deep copy."""
        return RetargetingOutput(
            timestamp=self.timestamp,
            valid=self.valid,
            q_des=self.q_des.copy(),
            dq_des=self.dq_des.copy(),
            hand_l_des=self.hand_l_des.copy(),
            hand_r_des=self.hand_r_des.copy(),
            elbow_l_des=self.elbow_l_des.copy(),
            elbow_r_des=self.elbow_r_des.copy(),
            hand_l_act=self.hand_l_act.copy(),
            hand_r_act=self.hand_r_act.copy(),
            elbow_l_act=self.elbow_l_act.copy(),
            elbow_r_act=self.elbow_r_act.copy(),
            hand_l_orient_mat=self.hand_l_orient_mat.copy(),
            hand_r_orient_mat=self.hand_r_orient_mat.copy(),
        )


@dataclass
class RobotFeedback:
    """Robot state feedback from simulation."""
    timestamp: float = 0.0
    # Base pose
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0], dtype=np.float64))
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # [ang_vel, lin_vel]
    # Joint state
    q: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float64))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float64))
    # COM state
    com_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    com_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    
    def copy(self) -> 'RobotFeedback':
        """Create a deep copy."""
        return RobotFeedback(
            timestamp=self.timestamp,
            base_pos=self.base_pos.copy(),
            base_quat=self.base_quat.copy(),
            base_vel=self.base_vel.copy(),
            q=self.q.copy(),
            dq=self.dq.copy(),
            com_pos=self.com_pos.copy(),
            com_vel=self.com_vel.copy(),
        )


class SharedState:
    """
    Thread-safe shared state container for all inter-node communication.
    
    Usage:
        state = SharedState()
        
        # Writer (tracking node):
        state.set_tracking_data(arm_data)
        
        # Reader (retargeting node):
        arm_data = state.get_tracking_data()
    """
    
    def __init__(self):
        # Locks for each data type
        self._tracking_lock = threading.Lock()
        self._retarget_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._fsm_lock = threading.Lock()
        self._command_lock = threading.Lock()
        
        # Data containers
        self._tracking_data = ArmTrackingData()
        self._retarget_output = RetargetingOutput()
        self._robot_feedback = RobotFeedback()
        self._fsm_state = RobotState.INIT
        
        # Control command (torques)
        self._torque_command = np.zeros(28, dtype=np.float64)
        self._command_timestamp = 0.0
        
        # Shutdown flag
        self._shutdown = threading.Event()
        
        # Timing statistics
        self._timing_stats = {
            'tracking_hz': 0.0,
            'retarget_hz': 0.0,
            'control_hz': 0.0,
            'sim_hz': 0.0,
        }
        self._timing_lock = threading.Lock()
        # Per-loop duration storage (seconds)
        self._loop_durations = {}
        self._loop_lock = threading.Lock()
        
    # --- Tracking Data ---
    def set_tracking_data(self, data: ArmTrackingData):
        """Set tracking data (called by tracking node)."""
        with self._tracking_lock:
            self._tracking_data = data.copy()
            
    def get_tracking_data(self) -> ArmTrackingData:
        """Get tracking data (called by retargeting node)."""
        with self._tracking_lock:
            return self._tracking_data.copy()
    
    # --- Retargeting Output ---
    def set_retarget_output(self, output: RetargetingOutput):
        """Set retargeting output (called by retargeting node)."""
        with self._retarget_lock:
            self._retarget_output = output.copy()
            
    def get_retarget_output(self) -> RetargetingOutput:
        """Get retargeting output (called by controller node)."""
        with self._retarget_lock:
            return self._retarget_output.copy()
    
    # --- Robot Feedback ---
    def set_robot_feedback(self, feedback: RobotFeedback):
        """Set robot feedback (called by simulation)."""
        with self._feedback_lock:
            self._robot_feedback = feedback.copy()
            
    def get_robot_feedback(self) -> RobotFeedback:
        """Get robot feedback (called by controller/retargeting)."""
        with self._feedback_lock:
            return self._robot_feedback.copy()
    
    # --- FSM State ---
    def set_fsm_state(self, state: RobotState):
        """Set FSM state."""
        with self._fsm_lock:
            self._fsm_state = state
            
    def get_fsm_state(self) -> RobotState:
        """Get FSM state."""
        with self._fsm_lock:
            return self._fsm_state
    
    # --- Torque Command ---
    def set_torque_command(self, torques: np.ndarray, timestamp: float):
        """Set torque command (called by controller)."""
        with self._command_lock:
            self._torque_command = torques.copy()
            self._command_timestamp = timestamp
            
    def get_torque_command(self) -> tuple:
        """Get torque command (called by simulation). Returns (torques, timestamp)."""
        with self._command_lock:
            return self._torque_command.copy(), self._command_timestamp
    
    # --- Timing Stats ---
    def update_timing(self, node_name: str, hz: float):
        """Update timing statistics for a node."""
        key = f'{node_name}_hz'
        with self._timing_lock:
            if key in self._timing_stats:
                self._timing_stats[key] = hz
                
    def get_timing_stats(self) -> dict:
        """Get all timing statistics."""
        with self._timing_lock:
            return self._timing_stats.copy()

    # --- Loop durations ---
    def set_loop_duration(self, node_name: str, duration_s: float):
        """Set last loop duration (in seconds) for a node."""
        key = f'{node_name}_loop_s'
        with self._loop_lock:
            self._loop_durations[key] = float(duration_s)

    def get_loop_durations(self) -> dict:
        """Get last loop durations for all nodes."""
        with self._loop_lock:
            return self._loop_durations.copy()
    
    # --- Shutdown ---
    def request_shutdown(self):
        """Request all nodes to shutdown."""
        self._shutdown.set()
        
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown.is_set()
    
    def wait_for_shutdown(self, timeout: float = None) -> bool:
        """Wait for shutdown signal. Returns True if shutdown requested."""
        return self._shutdown.wait(timeout=timeout)
