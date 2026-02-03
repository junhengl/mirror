"""
Retargeting Node (500 Hz)

IK-based retargeting from body tracking to robot joint angles.
Runs in separate thread, reads tracking data and publishes desired poses.
"""

import numpy as np
import time
import threading
import sys
import os
from typing import Optional, Dict

# Add KinDynLib_single to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'KinDynLib_single'))

from ..shared_state import SharedState, ArmTrackingData, RetargetingOutput, RobotState
from ..config import RetargetingConfig, PipelineConfig


class RetargetingNode:
    """
    IK-based retargeting node.
    
    Converts body tracking data to desired robot joint positions.
    Runs at 500Hz, reads tracking data (30Hz) with zero-order hold.
    """
    
    def __init__(self, config: PipelineConfig, shared_state: SharedState):
        self.config = config
        self.retarget_config = config.retarget
        self.shared = shared_state
        
        # Import robot dynamics
        import robot_const as themis_const
        sys.modules['robot_const'] = themis_const
        from robot_dynamics import Robot
        from dynamics_lib import Xtrans
        
        self.themis_const = themis_const
        self.Xtrans = Xtrans
        
        # Initialize robot model
        print("[Retargeting] Initializing robot model...")
        self.robot = Robot()
        
        # Robot dimensions
        self.hand_r_offset = themis_const.hand_r
        self.hand_l_offset = themis_const.hand_l
        self.elbow_r_offset = np.array([0.0, -0.16, 0.0], dtype=np.float64)
        self.elbow_l_offset = np.array([0.0, 0.16, 0.0], dtype=np.float64)
        
        # Transform matrices
        self.Xhand_l = Xtrans(self.hand_l_offset)
        self.Xhand_r = Xtrans(self.hand_r_offset)
        self.Xelbow_l = Xtrans(self.elbow_l_offset)
        self.Xelbow_r = Xtrans(self.elbow_r_offset)
        
        # Link indices
        self.hand_r_link = 22
        self.hand_l_link = 29
        self.elbow_r_link = 20
        self.elbow_l_link = 27
        
        # Current IK state
        self.q = np.zeros(themis_const.DOF, dtype=np.float64)
        self.dq = np.zeros(themis_const.DOF, dtype=np.float64)
        
        # COM desired
        self.com_des = np.zeros(6, dtype=np.float64)
        
        # Last valid tracking data
        self.last_valid_tracking: Optional[ArmTrackingData] = None
        
        # Output smoothing
        self.filtered_q = self.q.copy()
        self.filter_alpha = 0.1
        self.max_joint_delta = 5  # rad per step at 500Hz
        
        # Base offset for visualization (from config)
        self.base_height = config.sim.base_height
        
        # Thread state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Timing
        self.last_stats_time = time.time()
        self.iteration_count = 0
        
        print("[Retargeting] Node initialized")
    
    def start(self):
        """Start retargeting thread."""
        self.running = True
        self.thread = threading.Thread(target=self._retargeting_loop, daemon=True)
        self.thread.start()
        print("[Retargeting] Started retargeting node (500Hz)")
    
    def stop(self):
        """Stop retargeting thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[Retargeting] Stopped")
    
    def _retargeting_loop(self):
        """Main retargeting loop at 500Hz."""
        target_dt = self.retarget_config.retarget_dt
        
        while self.running and not self.shared.is_shutdown_requested():
            loop_start = time.perf_counter()
            
            # Always run retargeting when we have tracking data
            # (don't wait for FSM to be in TRACKING - that creates circular dependency)
            tracking = self.shared.get_tracking_data()
            
            # Get robot feedback for warm start
            feedback = self.shared.get_robot_feedback()
            
            # Run IK if we have any tracking data
            if tracking.valid or self.last_valid_tracking is not None:
                output = self._solve_ik(tracking, feedback)
                
                # Publish output
                self.shared.set_retarget_output(output)
            
            # Update timing stats
            self.iteration_count += 1
            if time.time() - self.last_stats_time >= 2.0:
                hz = self.iteration_count / (time.time() - self.last_stats_time)
                self.shared.update_timing('retarget', hz)
                self.iteration_count = 0
                self.last_stats_time = time.time()
            
            # Sleep to maintain rate
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    
    def _transform_zed_to_robot(self, tracking: ArmTrackingData) -> Dict[str, np.ndarray]:
        """Transform ZED coordinates to robot frame."""
        
        def zed_to_robot(pos_zed):
            # ZED: X=right, Y=up, Z=backward
            # Robot: X=forward, Y=left, Z=up
            return np.array([pos_zed[2], -pos_zed[0], pos_zed[1]], dtype=np.float64)
        
        # Convert positions
        left_elbow_robot = zed_to_robot(tracking.left_elbow)
        left_wrist_robot = zed_to_robot(tracking.left_wrist)
        right_elbow_robot = zed_to_robot(tracking.right_elbow)
        right_wrist_robot = zed_to_robot(tracking.right_wrist)
        
        # Scale to robot dimensions
        scale = self.retarget_config.arm_scale
        
        # Build desired task-space vectors [orientation(3), position(3)]
        zero_orient = np.zeros(3, dtype=np.float64)
        
        return {
            'x_hand_l_des': np.concatenate([zero_orient, left_wrist_robot * scale]),
            'x_hand_r_des': np.concatenate([zero_orient, right_wrist_robot * scale]),
            'x_elbow_l_des': np.concatenate([zero_orient, left_elbow_robot * scale]),
            'x_elbow_r_des': np.concatenate([zero_orient, right_elbow_robot * scale])
        }
    
    def _solve_ik(self, tracking: ArmTrackingData, feedback) -> RetargetingOutput:
        """Solve IK for arm tracking."""
        output = RetargetingOutput()
        output.timestamp = time.time()
        
        # Check tracking validity
        if not tracking.valid:
            if self.last_valid_tracking is None:
                # Debug: print why we're failing
                if self.iteration_count % 500 == 0:  # Every ~1 second at 500Hz
                    print(f"[Retargeting] No valid tracking data yet (tracking.valid={tracking.valid})")
                output.valid = False
                return output
            tracking = self.last_valid_tracking
        
        # Check confidence
        min_conf = self.config.tracking.min_confidence
        if tracking.left_confidence < min_conf and tracking.right_confidence < min_conf:
            if self.last_valid_tracking is None:
                # Debug: print confidence values
                if self.iteration_count % 500 == 0:
                    print(f"[Retargeting] Low confidence: L={tracking.left_confidence:.1f}, R={tracking.right_confidence:.1f} (min={min_conf})")
                output.valid = False
                return output
            tracking = self.last_valid_tracking
        
        # Save valid tracking
        self.last_valid_tracking = tracking
        
        # Transform to robot frame
        desired = self._transform_zed_to_robot(tracking)
        
        # Warm start from feedback (add floating base)
        q_current = np.zeros(self.themis_const.DOF, dtype=np.float64)
        q_current[6:6+28] = feedback.q
        dq_current = np.zeros(self.themis_const.DOF, dtype=np.float64)
        dq_current[6:6+28] = feedback.dq
        
        # IK iterations
        for _ in range(self.retarget_config.num_ik_iterations):
            self.robot.update(q_current, dq_current)
            
            # Compute FK
            x_hand_r, _ = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
            x_hand_l, _ = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
            x_elbow_r, _ = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
            x_elbow_l, _ = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
            
            # Compute Jacobians
            J_hand_r = self.robot.compute_body_jacobian(self.hand_r_link, self.Xhand_r)
            J_hand_l = self.robot.compute_body_jacobian(self.hand_l_link, self.Xhand_l)
            J_elbow_r = self.robot.compute_body_jacobian(self.elbow_r_link, self.Xelbow_r)
            J_elbow_l = self.robot.compute_body_jacobian(self.elbow_l_link, self.Xelbow_l)
            
            # Solve IK step (QP solver)
            q_des, dq_des = self.robot.update_task_space_command_qp(
                desired['x_elbow_l_des'], desired['x_elbow_r_des'],
                x_elbow_l, x_elbow_r,
                desired['x_hand_l_des'], desired['x_hand_r_des'],
                x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                self.com_des
            )
            
            q_current = q_des.copy()
            dq_current = dq_des.copy()
        
        # Wrap angles
        q_current[7:] = np.arctan2(np.sin(q_current[7:]), np.cos(q_current[7:]))
        
        # Apply smoothing filter
        q_delta = q_current[6:6+28] - self.filtered_q[6:6+28]
        q_delta = np.arctan2(np.sin(q_delta), np.cos(q_delta))
        q_delta = np.clip(q_delta, -self.max_joint_delta, self.max_joint_delta)
        self.filtered_q[6:6+28] = self.filtered_q[6:6+28] + self.filter_alpha * q_delta
        self.filtered_q[6:6+28] = np.arctan2(
            np.sin(self.filtered_q[6:6+28]),
            np.cos(self.filtered_q[6:6+28])
        )
        
        # Update internal state
        self.q = q_current
        self.dq = dq_current
        
        # Compute final FK for actual positions after IK
        self.robot.update(self.filtered_q, dq_current)
        x_hand_r_act, _ = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
        x_hand_l_act, _ = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
        x_elbow_r_act, _ = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
        x_elbow_l_act, _ = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
        
        # Build output
        output.valid = True
        output.q_des = self.filtered_q[6:6+28].copy()
        output.dq_des = dq_current[6:6+28].copy()
        
        # Task space targets for visualization (with base height offset)
        base_offset = np.array([0.0, 0.0, self.base_height])
        output.hand_l_des = desired['x_hand_l_des'][3:6] + base_offset
        output.hand_r_des = desired['x_hand_r_des'][3:6] + base_offset
        output.elbow_l_des = desired['x_elbow_l_des'][3:6] + base_offset
        output.elbow_r_des = desired['x_elbow_r_des'][3:6] + base_offset
        
        # Actual FK positions (with base offset)
        output.hand_l_act = x_hand_l_act[3:6] + base_offset
        output.hand_r_act = x_hand_r_act[3:6] + base_offset
        output.elbow_l_act = x_elbow_l_act[3:6] + base_offset
        output.elbow_r_act = x_elbow_r_act[3:6] + base_offset
        
        return output
