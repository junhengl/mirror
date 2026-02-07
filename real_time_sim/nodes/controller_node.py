"""
Controller Node (1 kHz)

PD control with FSM-based state management.
Computes torques to track desired joint positions.
"""

import numpy as np
import time
import threading
from typing import Optional

from ..shared_state import SharedState, RobotState
from ..config import ControlConfig, PipelineConfig
from ..control.pd_controller import PDController
from ..control.fsm import FSMController


class ControllerNode:
    """
    Controller node running at 1kHz.
    
    Uses FSM to determine behavior and PD control for torque computation.
    """
    
    def __init__(self, config: PipelineConfig, shared_state: SharedState):
        self.config = config
        self.ctrl_config = config.control
        self.shared = shared_state
        
        # Initialize controllers
        self.pd_controller = PDController(config)
        self.fsm = FSMController(config, shared_state)
        
        # Thread state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Timing
        self.last_stats_time = time.time()
        self.iteration_count = 0
        
        # Diagnostics
        self.last_tracking_error = 0.0
        
        print("[Controller] Initialized (1kHz)")
    
    def start(self):
        """Start controller thread."""
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        print("[Controller] Started controller node")
    
    def stop(self):
        """Stop controller thread."""
        self.running = False
        self.fsm.request_shutdown()
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[Controller] Stopped")
    
    def _control_loop(self):
        """Main control loop at 1kHz."""
        target_dt = self.ctrl_config.control_dt
        
        while self.running and not self.shared.is_shutdown_requested():
            loop_start = time.perf_counter()
            
            # Get robot feedback
            feedback = self.shared.get_robot_feedback()
            
            # Update FSM and get desired joint positions
            q_des = self.fsm.update()
            
            # Get desired velocities from retargeting output (if in tracking mode)
            retarget = self.shared.get_retarget_output()
            if retarget.valid and self.fsm.get_state() == RobotState.TRACKING:
                dq_des = retarget.dq_des
            else:
                dq_des = np.zeros(28, dtype=np.float64)
            
            # Compute PD torques
            torques = self.pd_controller.compute_torque(
                feedback.q,
                feedback.dq,
                q_des,
                dq_des
            )

            # # test torque commands:
            # torques = np.array([
            #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right leg
            #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left leg
            #     0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,  # right arm
            #     0, -0.0, 0.0, -0.0, 0.0, -0.0, 10.0,  # left arm
            #     0.0, 0.0  # head
            # ])

            torques[18] = 0.0  # Disable right wrist for testing
            torques[25] = 0.0  # Disable left wrist for testing
            # torques[14] *= -1.0


            # Publish torque command
            self.shared.set_torque_command(torques, time.time())
            
            # Update markers for visualization (if in tracking mode)
            if self.fsm.get_state() == RobotState.TRACKING and retarget.valid:
                # Compute actual end-effector positions from feedback
                # (Simplified: use desired positions for now)
                pass
            
            # Track diagnostics
            self.last_tracking_error, _, _ = self.pd_controller.get_tracking_error(
                feedback.q, q_des
            )
            
            # Update timing stats
            self.iteration_count += 1
            if time.time() - self.last_stats_time >= 2.0:
                hz = self.iteration_count / (time.time() - self.last_stats_time)
                self.shared.update_timing('control', hz)
                self.iteration_count = 0
                self.last_stats_time = time.time()
            
            # Sleep to maintain rate (use sleep instead of busy-wait to save CPU)
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    
    def get_tracking_error(self) -> float:
        """Get current tracking error (RMS, radians)."""
        return self.last_tracking_error
    
    def get_fsm_state(self) -> RobotState:
        """Get current FSM state."""
        return self.fsm.get_state()
