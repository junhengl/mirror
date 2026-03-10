"""
Retargeting Node (500 Hz)

IK-based retargeting from body tracking to robot joint angles.
"""

import numpy as np
import time
import threading
import sys
import os
from typing import Optional, Dict

# Add KinDynLib to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'KinDynLib'))

from ..shared_state import SharedState, ArmTrackingData, RetargetingOutput, RobotState
from ..config import RetargetingConfig, PipelineConfig
from ..joint_mapping import JointMapping


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
        
        # Joint mapping (MuJoCo ↔ KinDynLib)
        self.joint_mapping = JointMapping(config.joint_mapping)
        
        # Initialize robot model
        print("[Retargeting] Initializing robot model...")
        self.robot = Robot()
        
        # Robot dimensions
        self.hand_r_offset = np.array([0.08, 0.0, 0.0], dtype=np.float64)
        self.hand_l_offset = np.array([0.08, 0.0, 0.0], dtype=np.float64)
        self.elbow_r_offset = np.array([0.0, -0.0, 0.0], dtype=np.float64)
        self.elbow_l_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Transform matrices
        self.Xhand_l = Xtrans(self.hand_l_offset)
        self.Xhand_r = Xtrans(self.hand_r_offset)
        self.Xelbow_l = Xtrans(self.elbow_l_offset)
        self.Xelbow_r = Xtrans(self.elbow_r_offset)
        
        # Link indices
        self.hand_r_link = 23
        self.hand_l_link = 30
        self.elbow_r_link = 21
        self.elbow_l_link = 28
        
        # Current IK state
        self.q = np.zeros(themis_const.DOF, dtype=np.float64)
        self.dq = np.zeros(themis_const.DOF, dtype=np.float64)
        
        # COM desired
        self.com_des = np.zeros(6, dtype=np.float64)
        
        # Reference pose tracking (for regularization toward favorable configuration)
        self.q_ref = self.retarget_config.q_ref
        if self.q_ref is None:
            # Default favorable pose: bent standing with head fixed, arms relaxed
            # [base_6DOF, right_leg_6DOF, left_leg_6DOF, right_arm_7DOF, left_arm_7DOF, head_2DOF]
            self.q_ref = np.array([
                0.0, 0.0, 1.3,                    # base: xy center, z at hanging height
                0.0, 0.0, 0.0,                    # base: rpy neutral
                # Right leg (6 DOF): bent standing configuration
                0.0, -0.8, 1.2, -0.4, 0.0, 0.0,
                # Left leg (6 DOF): bent standing configuration
                0.0, -0.8, 1.2, -0.4, 0.0, 0.0,
                # Right arm (7 DOF): relaxed
                0.0, -1.75, 1.57, 1.57, 0.0, -0.5, 0.0,
                # Left arm (7 DOF): relaxed
                0.0, -1.75, 1.57, 1.57, 0.0, -0.5, 0.0,
                # Head (2 DOF): fixed at zero
                0.0, 0.0,
            ], dtype=np.float64)
        self.w_ref = self.retarget_config.w_ref
        
        # Last valid tracking data
        self.last_valid_tracking: Optional[ArmTrackingData] = None
        
        # Output smoothing
        self.filtered_q = self.q.copy()
        self.filter_alpha = 0.2
        self.max_joint_delta = 50  # rad per step at 500Hz
        
        # Base offset for visualization (from config)
        self.base_height = config.sim.base_height
        
        # Thread state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Timing
        self.last_stats_time = time.time()
        self.iteration_count = 0
        
        print(f"[Retargeting] Node initialized (reference pose tracking weight: {self.w_ref})")
    
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
            t_tracking_read = time.time()
            
            # Get robot feedback for warm start
            feedback = self.shared.get_robot_feedback()
            
            # Run IK if we have any tracking data
            if tracking.valid or self.last_valid_tracking is not None:
                output = self._solve_ik(tracking, feedback)
                
                # Propagate capture timestamp for end-to-end latency tracking
                output.source_capture_ts = tracking.timestamp
                
                # Publish output
                self.shared.set_retarget_output(output)
                
                # Publish latency metrics
                if tracking.timestamp > 0:
                    self.shared.set_loop_duration(
                        'lat_tracking_data_age', t_tracking_read - tracking.timestamp)
                self.shared.set_loop_duration(
                    'lat_retarget_total', time.perf_counter() - loop_start)
            
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
        # NOTE: ZED camera labels keypoints from its perspective (left/right of image)
        # User faces camera, so ZED "left" is user's right, ZED "right" is user's left
        left_elbow_robot = zed_to_robot(tracking.left_elbow)      # camera left = user right
        left_wrist_robot = zed_to_robot(tracking.left_wrist)
        right_elbow_robot = zed_to_robot(tracking.right_elbow)    # camera right = user left
        right_wrist_robot = zed_to_robot(tracking.right_wrist)
        
        # Scale to robot dimensions
        scale = self.retarget_config.arm_scale
        
        # Compute hand orientations from elbow→hand direction vectors
        # (relative to body frame, where neck = identity rotation)
        # 
        # Build rotation matrix:
        #   Z-axis = arm direction (elbow → hand)
        #   X-axis = perpendicular to arm (via cross product with world up)
        #   Y-axis = perpendicular to both X and Z
        # Then convert to RPY using robot's ZYX Euler convention.
        #
        def direction_to_rotation(elbow, hand):
            """Build a 3x3 rotation matrix whose Z-axis points along
            the elbow→hand direction. Convert to RPY matching robot FK convention."""
            d = hand - elbow
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            
            z_axis = d / norm  # Z-axis points along arm direction
            
            # Choose a "world up" hint to avoid singularity when arm is vertical
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(z_axis, up)) > 0.95:
                # Arm nearly vertical — use world-forward as fallback hint
                up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            
            # X-axis: perpendicular to arm, in plane of (up × arm)
            x_axis = np.cross(up, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-8:
                # Degenerate — use different hint
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                x_axis = np.cross(up, z_axis)
                x_norm = np.linalg.norm(x_axis)
            x_axis /= x_norm
            
            # Y-axis: complete the right-handed frame
            y_axis = np.cross(z_axis, x_axis)
            # y_axis is already unit (cross of two orthonormal vectors)
            
            # R columns = [x_axis, y_axis, z_axis]
            R = np.column_stack([x_axis, y_axis, z_axis])
            
            # Extract RPY using robot's ZYX convention (rpy_from_rot_zyx):
            #   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
            sy = -R[2, 0]
            cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            pitch = np.arctan2(sy, cy)
            
            if cy > 1e-6:
                yaw  = np.arctan2(R[1, 0], R[0, 0])
                roll = np.arctan2(R[2, 1], R[2, 2])
            else:
                # Gimbal lock: pitch ≈ ±π/2, fix roll=0 and absorb into yaw
                yaw  = np.arctan2(-R[0, 1], R[1, 1])
                roll = 0.0
            
            rpy = np.array([roll, pitch, yaw], dtype=np.float64)
            return R, rpy
        
        # Orientations (using scaled positions for consistency)
        # Camera right → robot left, camera left → robot right
        hand_l_mat, hand_l_rpy = direction_to_rotation(
            right_elbow_robot * scale, right_wrist_robot * scale)
        hand_r_mat, hand_r_rpy = direction_to_rotation(
            left_elbow_robot * scale, left_wrist_robot * scale)
        
        # print(f"[Retargeting] Transformed ZED to robot frame: hand_l_rpy={np.degrees(hand_l_rpy)}, hand_r_rpy={np.degrees(hand_r_rpy)}")
        
        hand_l_rpy *= 0
        hand_r_rpy *= 0
        # hand_l_rpy[0] = 1.78
        # hand_r_rpy[0] = 1.78
        
        # Build desired task-space vectors [orientation(3), position(3)]
        return {
            'x_hand_l_des': np.concatenate([hand_l_rpy, right_wrist_robot * scale]),    # camera right → robot left
            'x_hand_r_des': np.concatenate([hand_r_rpy, left_wrist_robot * scale]),     # camera left → robot right
            'x_elbow_l_des': np.concatenate([np.zeros(3, dtype=np.float64), right_elbow_robot * scale]),   # camera right → robot left
            'x_elbow_r_des': np.concatenate([np.zeros(3, dtype=np.float64), left_elbow_robot * scale]),    # camera left → robot right
            # Rotation matrices for visualization (3x3)
            'hand_l_orient_mat': hand_l_mat,
            'hand_r_orient_mat': hand_r_mat,
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
        # Map feedback from MuJoCo to KinDynLib convention before IK
        q_kin = self.joint_mapping.forward_q(feedback.q)
        dq_kin = self.joint_mapping.forward_dq(feedback.dq)
        q_current = np.zeros(self.themis_const.DOF, dtype=np.float64)
        q_current[6:6+28] = q_kin
        dq_current = np.zeros(self.themis_const.DOF, dtype=np.float64)
        dq_current[6:6+28] = dq_kin
        
        # IK iterations
        for _ in range(self.retarget_config.num_ik_iterations):
            self.robot.update(q_current, dq_current)
            
            # Compute FK and Jacobians for hands and elbows
            x_hand_r = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
            x_hand_l = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
            x_elbow_r = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
            x_elbow_l = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
            
            # Compute Jacobians
            J_hand_r = self.robot.compute_body_jacobian(self.hand_r_link, self.Xhand_r)
            J_hand_l = self.robot.compute_body_jacobian(self.hand_l_link, self.Xhand_l)
            J_elbow_r = self.robot.compute_body_jacobian(self.elbow_r_link, self.Xelbow_r)
            J_elbow_l = self.robot.compute_body_jacobian(self.elbow_l_link, self.Xelbow_l)
            
            # timer for IK solve
            ik_start = time.perf_counter()

            # Solve IK step (QP solver: distributed OSQP)
            # q_des, dq_des = self.robot.update_task_space_command_qp_distributed(
            #     desired['x_elbow_l_des'], desired['x_elbow_r_des'],
            #     x_elbow_l, x_elbow_r,
            #     desired['x_hand_l_des'], desired['x_hand_r_des'],
            #     x_hand_l, x_hand_r,
            #     J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
            #     self.com_des
            # )

            # Solve IK step with lyapunov regulation
            q_des, dq_des = self.robot.update_task_space_command_qp_gpu_batch_distributed_alpha_lyapunov(
                        desired['x_elbow_l_des'], desired['x_elbow_r_des'], x_elbow_l, x_elbow_r,
                        desired['x_hand_l_des'], desired['x_hand_r_des'], x_hand_l, x_hand_r,
                        J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                        self.com_des,
                        n_batch=1024, max_iter=50,
                        pos_threshold=0.005,
                        n_alpha=8,
                        eta=0.0005,
                        eps_q=0.01,
                        eps_V=0.001,
                        dq_max=0.5
            )

            # q_des *= 0.0  # DEBUG: test zero pose

            ik_time = time.perf_counter() - ik_start
            self.shared.set_loop_duration('lat_ik_solve', ik_time)
            # print(f"[Retargeting] IK solve time: {ik_time*1000:.2f} ms")

            # Check for NaN/Inf after IK solve
            if not np.all(np.isfinite(q_des)) or not np.all(np.isfinite(dq_des)):
                print(f"WARNING: IK returned NaN/Inf! Holding previous pose.")
                break  # Exit IK iteration loop, use previous q_current

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
        
        # Final check before output
        if not np.all(np.isfinite(self.filtered_q)):
            print("WARNING: filtered_q contains NaN/Inf! Resetting to zero pose.")
            self.filtered_q = np.zeros_like(self.filtered_q)
            output.valid = False
            return output
        
        # # Update internal state
        self.q = q_current
        self.dq = dq_current
        
        # Compute final FK for actual positions after IK
        self.robot.update(self.filtered_q, dq_current)
        x_hand_r_act = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
        x_hand_l_act = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
        x_elbow_r_act = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
        x_elbow_l_act = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
        
        # Build output
        output.valid = True
        output.q_des = self.filtered_q[6:6+28].copy()
        output.dq_des = dq_current[6:6+28].copy()

        # Override head and leg q_des with fixed positions (not controlled by IK)
        # Layout [28]: right_leg(0-5), left_leg(6-11), right_arm(12-18), left_arm(19-25), head(26-27)
        # Legs: bent standing configuration
        leg_q_fixed = np.array([0.0, 0.0, -0.5, 1.0, -0.5, 0.0], dtype=np.float64)
        output.q_des[0:6] = leg_q_fixed    # right leg
        output.q_des[6:12] = leg_q_fixed   # left leg
        output.dq_des[0:12] = 0.0          # zero velocity for legs
        # Head: fixed at zero
        output.q_des[26:28] = 0.0
        output.dq_des[26:28] = 0.0
        # print all the desired joint angles for debugging
        # print(f"Desired joint angles (degrees): {np.degrees(output.q_des)}  (shoulder_pitch_r={np.degrees(output.q_des[12]):.1f}, shoulder_roll_r={np.degrees(output.q_des[13]):.1f}, shoulder_yaw_r={np.degrees(output.q_des[14]):.1f}, elbow_pitch_r={np.degrees(output.q_des[15]):.1f}, elbow_yaw_r={np.degrees(output.q_des[16]):.1f}, wrist_pitch_r={np.degrees(output.q_des[17]):.1f}, wrist_yaw_r={np.degrees(output.q_des[18]):.1f}, shoulder_pitch_l={np.degrees(output.q_des[19]):.1f}, shoulder_roll_l={np.degrees(output.q_des[20]):.1f}, shoulder_yaw_l={np.degrees(output.q_des[21]):.1f}, elbow_pitch_l={np.degrees(output.q_des[22]):.1f}, elbow_yaw_l={np.degrees(output.q_des[23]):.1f}, wrist_pitch_l={np.degrees(output.q_des[24]):.1f}, wrist_yaw_l={np.degrees(output.q_des[25]):.1f})")
        
        # Task space targets for visualization (with base height offset)
        base_offset = np.array([0.0, 0.0, self.base_height])
        output.hand_l_des = desired['x_hand_l_des'][3:6] + base_offset
        output.hand_r_des = desired['x_hand_r_des'][3:6] + base_offset
        output.elbow_l_des = desired['x_elbow_l_des'][3:6] + base_offset
        output.elbow_r_des = desired['x_elbow_r_des'][3:6] + base_offset
        
        # Hand orientation matrices for arrow visualization
        output.hand_l_orient_mat = desired['hand_l_orient_mat']
        output.hand_r_orient_mat = desired['hand_r_orient_mat']
        
        # Actual FK positions (with base offset)
        output.hand_l_act = x_hand_l_act[3:6] + base_offset
        output.hand_r_act = x_hand_r_act[3:6] + base_offset
        output.elbow_l_act = x_elbow_l_act[3:6] + base_offset
        output.elbow_r_act = x_elbow_r_act[3:6] + base_offset
        
        return output


