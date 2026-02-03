"""
Finite State Machine for robot control.

States:
- INIT: Moving to default standing pose
- IDLE: Holding default pose, waiting for valid tracking
- TRACKING: Active tracking and retargeting
- SAFETY_STOP: Safety triggered, holding last position
- SHUTDOWN: Clean shutdown
"""

import time
import numpy as np
from enum import Enum, auto
from typing import Optional

from ..shared_state import SharedState, RobotState, ArmTrackingData, RetargetingOutput
from ..config import FSMConfig, PipelineConfig


class FSMController:
    """
    Finite State Machine controller for robot behavior.
    
    Manages transitions between states based on:
    - Tracking validity
    - Safety constraints
    - Time-based transitions
    """
    
    def __init__(self, config: PipelineConfig, shared_state: SharedState):
        self.config = config
        self.fsm_config = config.fsm
        self.shared = shared_state
        
        # State timing
        self.state_entry_time = time.time()
        self.current_state = RobotState.INIT
        
        # Default pose (standing with arms slightly forward)
        self.default_q = np.array([
            # Right leg
            0.0, 0.0, -0.1, 0.2, -0.1, 0.0,
            # Left leg  
            0.0, 0.0, -0.1, 0.2, -0.1, 0.0,
            # Right arm (safe pose avoiding singularity)
            0.0, 0.8, -0.5, 0.78, 0.1, 0.4, 0.0,
            # Left arm
            0.0, -0.8, -0.5, -0.78, 0.1, -0.4, 0.0,
            # Head
            0.0, 0.0
        ], dtype=np.float64)
        
        # Safety hold position
        self.safety_hold_q: Optional[np.ndarray] = None
        
        # Blending
        self.blend_start_q: Optional[np.ndarray] = None
        self.blend_target_q: Optional[np.ndarray] = None
        self.blend_start_time: float = 0.0
        
        # Tracking timeout
        self.last_valid_tracking_time = 0.0
        self.tracking_timeout = 1.0  # seconds
        
    def get_state(self) -> RobotState:
        """Get current FSM state."""
        return self.current_state
    
    def set_state(self, new_state: RobotState):
        """Transition to a new state."""
        if new_state != self.current_state:
            print(f"[FSM] Transition: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_entry_time = time.time()
            self.shared.set_fsm_state(new_state)
            
            # Setup for new state
            if new_state == RobotState.SAFETY_STOP:
                feedback = self.shared.get_robot_feedback()
                self.safety_hold_q = feedback.q.copy()
    
    def time_in_state(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_entry_time
    
    def check_safety(self, feedback, retarget_output: RetargetingOutput) -> bool:
        """
        Check safety constraints.
        
        Returns True if safe, False if safety violation detected.
        """
        # Check joint velocity limits
        max_vel = np.max(np.abs(feedback.dq))
        if max_vel > self.fsm_config.max_joint_velocity:
            max_idx = np.argmax(np.abs(feedback.dq))
            print(f"[FSM] Safety: Joint {max_idx} velocity {feedback.dq[max_idx]:.2f} rad/s exceeded limit {self.fsm_config.max_joint_velocity}")
            return False
        
        # Could add more checks:
        # - COM outside support polygon
        # - Tracking error too large
        # - Joint limits
        
        return True
    
    def update(self) -> np.ndarray:
        """
        Update FSM and return desired joint positions.
        
        Should be called at control rate (1kHz).
        
        Returns:
            q_des: Desired joint positions (28,)
        """
        # Get current data
        feedback = self.shared.get_robot_feedback()
        tracking = self.shared.get_tracking_data()
        retarget = self.shared.get_retarget_output()
        
        # Update tracking validity
        if tracking.valid:
            self.last_valid_tracking_time = time.time()
        tracking_valid = (time.time() - self.last_valid_tracking_time) < self.tracking_timeout
        
        # State machine logic
        if self.current_state == RobotState.INIT:
            return self._handle_init(feedback)
            
        elif self.current_state == RobotState.IDLE:
            return self._handle_idle(feedback, tracking_valid, retarget)
            
        elif self.current_state == RobotState.TRACKING:
            return self._handle_tracking(feedback, tracking_valid, retarget)
            
        elif self.current_state == RobotState.SAFETY_STOP:
            return self._handle_safety_stop(feedback)
            
        elif self.current_state == RobotState.SHUTDOWN:
            return feedback.q.copy()  # Hold current position
            
        return self.default_q.copy()
    
    def _handle_init(self, feedback) -> np.ndarray:
        """
        INIT state: Move to default pose.
        
        Blends from current position to default pose over init_duration.
        """
        t = self.time_in_state()
        
        if self.blend_start_q is None:
            self.blend_start_q = feedback.q.copy()
            self.blend_target_q = self.default_q.copy()
            
        # Blend factor (0 to 1)
        alpha = min(t / self.fsm_config.init_duration, 1.0)
        # Smooth step for blending
        alpha = alpha * alpha * (3 - 2 * alpha)  # Hermite interpolation
        
        q_des = (1 - alpha) * self.blend_start_q + alpha * self.blend_target_q
        
        # Check if init complete
        if t >= self.fsm_config.init_duration:
            self.blend_start_q = None
            self.set_state(RobotState.IDLE)
            
        return q_des
    
    def _handle_idle(self, feedback, tracking_valid: bool, retarget: RetargetingOutput) -> np.ndarray:
        """
        IDLE state: Hold default pose, wait for tracking.
        """
        # Check for valid tracking to start
        if tracking_valid and retarget.valid:
            # Start blending to tracking
            self.blend_start_q = feedback.q.copy()
            self.blend_start_time = time.time()
            self.set_state(RobotState.TRACKING)
            
        return self.default_q.copy()
    
    def _handle_tracking(self, feedback, tracking_valid: bool, retarget: RetargetingOutput) -> np.ndarray:
        """
        TRACKING state: Follow retargeting output.
        """
        # Check safety
        if not self.check_safety(feedback, retarget):
            self.set_state(RobotState.SAFETY_STOP)
            return feedback.q.copy()
        
        # Check for tracking loss
        if not tracking_valid:
            print("[FSM] Tracking lost, returning to IDLE")
            self.blend_start_q = feedback.q.copy()
            self.set_state(RobotState.IDLE)
            return self._blend_to_default(feedback)
        
        # Blend at start of tracking
        t_since_start = time.time() - self.blend_start_time
        if t_since_start < self.fsm_config.blend_duration and self.blend_start_q is not None:
            alpha = min(t_since_start / self.fsm_config.blend_duration, 1.0)
            alpha = alpha * alpha * (3 - 2 * alpha)
            
            # Blend from start position to retarget output
            q_des = (1 - alpha) * self.blend_start_q + alpha * retarget.q_des
            return q_des
        
        # Normal tracking
        if retarget.valid:
            return retarget.q_des.copy()
        else:
            return feedback.q.copy()
    
    def _handle_safety_stop(self, feedback) -> np.ndarray:
        """
        SAFETY_STOP state: Hold last safe position.
        """
        if self.safety_hold_q is not None:
            return self.safety_hold_q.copy()
        return feedback.q.copy()
    
    def _blend_to_default(self, feedback) -> np.ndarray:
        """Helper to blend current position toward default."""
        if self.blend_start_q is None:
            return self.default_q.copy()
            
        t = self.time_in_state()
        alpha = min(t / self.fsm_config.blend_duration, 1.0)
        alpha = alpha * alpha * (3 - 2 * alpha)
        
        return (1 - alpha) * self.blend_start_q + alpha * self.default_q
    
    def request_shutdown(self):
        """Request transition to shutdown state."""
        self.set_state(RobotState.SHUTDOWN)
