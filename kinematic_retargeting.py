"""
Real-time Kinematic Retargeting Pipeline

Integrates ZED body tracking with inverse kinematics for robot retargeting.
Uses arm position data from ZED (elbow 14,15 and wrist 16,17) as desired 
targets for IK-based retargeting.

Two visualization windows:
1. ZED camera view with skeleton overlay
2. MuJoCo robot visualization with retargeted pose

Note: Requires sudo for ZED motion sensors:
    sudo /home/junhengl/body_tracking/.venv/bin/python kinematic_retargeting.py
"""

# sudo /home/junhengl/body_tracking/.venv/bin/python kinematic_retargeting.py

import numpy as np
import time
import threading
import queue
import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Dict

# Add KinDynLib_single to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'KinDynLib_single'))

# Import kinematic libraries
import robot_const as themis_const
sys.modules['robot_const'] = themis_const
from robot_dynamics import Robot
from dynamics_lib import Xtrans

# MuJoCo visualization
import mujoco
import mujoco.viewer

# ZED imports
try:
    import pyzed.sl as sl
    import cv2
    import cv_viewer.tracking_viewer as cv_viewer
    ZED_AVAILABLE = True
except ImportError:
    print("Warning: ZED SDK not available. Running in simulation mode.")
    ZED_AVAILABLE = False
    cv2 = None
    cv_viewer = None


@dataclass
class ArmTrackingData:
    """Container for arm tracking data from ZED.
    
    Positions are in meters, in local body frame (relative to COM keypoint index 2).
    ZED coordinate system: X=right, Y=up, Z=backward
    """
    timestamp: float
    valid: bool
    # Body COM (keypoint index 2 = SPINE_2)
    body_com: np.ndarray  # [x, y, z] in ZED world frame
    # Left arm (keypoint indices: shoulder=12, elbow=14, wrist=16)
    # All positions relative to body_com
    left_shoulder: np.ndarray  # [x, y, z] in local body frame
    left_elbow: np.ndarray
    left_wrist: np.ndarray
    left_confidence: float  # min confidence of arm keypoints
    # Right arm (keypoint indices: shoulder=13, elbow=15, wrist=17)
    right_shoulder: np.ndarray
    right_elbow: np.ndarray
    right_wrist: np.ndarray
    right_confidence: float


class PositionFilter:
    """
    Low-pass filter with jump rejection for position data.
    Holds previous valid value on data loss or large jumps.
    """
    
    def __init__(self, alpha: float = 0.3, jump_threshold: float = 0.15):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
            jump_threshold: Maximum allowed position change per frame (meters).
        """
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self.last_valid: Optional[np.ndarray] = None
        self.filtered: Optional[np.ndarray] = None
        
    def update(self, value: np.ndarray, valid: bool = True) -> np.ndarray:
        """
        Update filter with new value.
        
        Args:
            value: New position value [x, y, z]
            valid: Whether the value is valid (tracked)
            
        Returns:
            Filtered position value
        """
        # Handle invalid data - return last valid
        if not valid or np.any(np.isnan(value)):
            if self.last_valid is not None:
                return self.last_valid.copy()
            else:
                return value.copy()
        
        # First valid value
        if self.filtered is None:
            self.filtered = value.copy()
            self.last_valid = value.copy()
            return self.filtered.copy()
        
        # Check for jump
        jump = np.linalg.norm(value - self.filtered)
        if jump > self.jump_threshold:
            # Large jump detected - but slowly move toward new value to avoid getting stuck
            # Use a much smaller alpha to gradually accept the new position
            self.filtered = 0.05 * value + 0.95 * self.filtered
            self.last_valid = self.filtered.copy()
            return self.filtered.copy()
        
        # Apply low-pass filter: filtered = alpha * new + (1-alpha) * old
        self.filtered = self.alpha * value + (1.0 - self.alpha) * self.filtered
        self.last_valid = self.filtered.copy()
        
        return self.filtered.copy()
    
    def reset(self):
        """Reset filter state."""
        self.last_valid = None
        self.filtered = None


class ZEDTracker:
    """ZED body tracking for arm positions.
    
    Extracts arm keypoints in local body frame (relative to COM at keypoint 2).
    """
    
    def __init__(self, data_queue: queue.Queue):
        self.data_queue = data_queue
        self.running = False
        self.thread = None
        
        # Position filters for smoothing (one per keypoint)
        # Filter parameters: alpha=0.4 (responsive), jump_threshold=0.12m
        self.filters = {
            'left_shoulder': PositionFilter(alpha=0.4, jump_threshold=0.12),
            'left_elbow': PositionFilter(alpha=0.4, jump_threshold=0.12),
            'left_wrist': PositionFilter(alpha=0.4, jump_threshold=0.12),
            'right_shoulder': PositionFilter(alpha=0.4, jump_threshold=0.12),
            'right_elbow': PositionFilter(alpha=0.4, jump_threshold=0.12),
            'right_wrist': PositionFilter(alpha=0.4, jump_threshold=0.12),
        }
        self.zed = None
        
    def start(self):
        """Start tracking thread."""
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        print("[ZED] Started body tracking thread")
        
    def stop(self):
        """Stop tracking thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.zed:
            self.zed.disable_body_tracking()
            self.zed.disable_positional_tracking()
            self.zed.close()
        print("[ZED] Stopped")
        
    def _tracking_loop(self):
        """Main tracking loop."""
        if not ZED_AVAILABLE:
            self._dummy_tracking_loop()
            return
            
        # Initialize ZED
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Failed to open camera: {status}")
            self._dummy_tracking_loop()
            return
            
        # Enable positional tracking
        tracking_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(tracking_params)
        
        # Enable body tracking
        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.body_format = sl.BODY_FORMAT.BODY_18
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        
        status = self.zed.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Failed to enable body tracking: {status}")
            self.zed.close()
            self._dummy_tracking_loop()
            return
            
        print("[ZED] Body tracking enabled")
        
        body_runtime = sl.BodyTrackingRuntimeParameters()
        body_runtime.detection_confidence_threshold = 40
        
        # Display setup
        camera_info = self.zed.get_camera_information()
        display_resolution = sl.Resolution(
            min(camera_info.camera_configuration.resolution.width, 1280),
            min(camera_info.camera_configuration.resolution.height, 720)
        )
        image_scale = [
            display_resolution.width / camera_info.camera_configuration.resolution.width,
            display_resolution.height / camera_info.camera_configuration.resolution.height
        ]
        
        bodies = sl.Bodies()
        image = sl.Mat()
        
        # Main loop
        while self.running:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Get camera image
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                
                # Get bodies
                self.zed.retrieve_bodies(bodies, body_runtime)
                
                # Display with skeleton overlay
                image_ocv = image.get_data()
                cv_viewer.render_2D(
                    image_ocv, image_scale, bodies.body_list, True, sl.BODY_FORMAT.BODY_18
                )
                cv2.imshow("ZED Body Tracking", image_ocv)
                
                # Extract arm data from first detected body
                if len(bodies.body_list) > 0:
                    body = bodies.body_list[0]
                    
                    # Get keypoints and confidences
                    kp = body.keypoint
                    conf = body.keypoint_confidence
                    
                    # ZED BODY_18 indices:
                    # COM (NECK) = 1
                    # LEFT_SHOULDER=2, LEFT_ELBOW=3, LEFT_WRIST=4
                    # RIGHT_SHOULDER=5, RIGHT_ELBOW=6, RIGHT_WRIST=7
                    
                    # Get body COM (keypoint 1 = NECK) in world frame
                    body_com = np.array([kp[1][0], kp[1][1], kp[1][2]], dtype=np.float32) 
                    # Offset from NECK to approximate torso COM
                    # ZED coords: X=right, Y=up, Z=backward
                    # Neck is above torso COM, so shift down in Y (negative)
                    body_com_offset = np.array([0.0, -0.15, 0.0], dtype=np.float32)
                    body_com += body_com_offset
                    
                    # Get raw positions in world frame
                    left_shoulder_world = np.array([kp[2][0], kp[2][1], kp[2][2]], dtype=np.float32)
                    left_elbow_world = np.array([kp[3][0], kp[3][1], kp[3][2]], dtype=np.float32)
                    left_wrist_world = np.array([kp[4][0], kp[4][1], kp[4][2]], dtype=np.float32)
                    right_shoulder_world = np.array([kp[5][0], kp[5][1], kp[5][2]], dtype=np.float32)
                    right_elbow_world = np.array([kp[6][0], kp[6][1], kp[6][2]], dtype=np.float32)
                    right_wrist_world = np.array([kp[7][0], kp[7][1], kp[7][2]], dtype=np.float32)
                    
                    # Convert to local body frame (relative to COM)
                    left_shoulder_local = left_shoulder_world - body_com
                    left_elbow_local = left_elbow_world - body_com
                    left_wrist_local = left_wrist_world - body_com
                    right_shoulder_local = right_shoulder_world - body_com
                    right_elbow_local = right_elbow_world - body_com
                    right_wrist_local = right_wrist_world - body_com
                    
                    # Check confidences (BODY_18 indices)
                    left_valid = min(conf[1], conf[2], conf[3], conf[4]) > 30
                    right_valid = min(conf[1], conf[5], conf[6], conf[7]) > 30
                    
                    # Apply filters (in local body frame)
                    left_shoulder_filtered = self.filters['left_shoulder'].update(left_shoulder_local, left_valid)
                    left_elbow_filtered = self.filters['left_elbow'].update(left_elbow_local, left_valid)
                    left_wrist_filtered = self.filters['left_wrist'].update(left_wrist_local, left_valid)
                    right_shoulder_filtered = self.filters['right_shoulder'].update(right_shoulder_local, right_valid)
                    right_elbow_filtered = self.filters['right_elbow'].update(right_elbow_local, right_valid)
                    right_wrist_filtered = self.filters['right_wrist'].update(right_wrist_local, right_valid)
                    
                    arm_data = ArmTrackingData(
                        timestamp=time.time(),
                        valid=True,
                        body_com=body_com,
                        left_shoulder=left_shoulder_filtered,
                        left_elbow=left_elbow_filtered,
                        left_wrist=left_wrist_filtered,
                        left_confidence=min(conf[2], conf[3], conf[4]),
                        right_shoulder=right_shoulder_filtered,
                        right_elbow=right_elbow_filtered,
                        right_wrist=right_wrist_filtered,
                        right_confidence=min(conf[5], conf[6], conf[7])
                    )
                else:
                    arm_data = ArmTrackingData(
                        timestamp=time.time(),
                        valid=False,
                        body_com=np.zeros(3, dtype=np.float32),
                        left_shoulder=np.zeros(3, dtype=np.float32),
                        left_elbow=np.zeros(3, dtype=np.float32),
                        left_wrist=np.zeros(3, dtype=np.float32),
                        left_confidence=0.0,
                        right_shoulder=np.zeros(3, dtype=np.float32),
                        right_elbow=np.zeros(3, dtype=np.float32),
                        right_wrist=np.zeros(3, dtype=np.float32),
                        right_confidence=0.0
                    )
                
                # Send to queue
                try:
                    self.data_queue.put_nowait(arm_data)
                except queue.Full:
                    pass
                
                # Handle CV window events
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.running = False
                    
        image.free(sl.MEM.CPU)
        cv2.destroyAllWindows()
        
    def _dummy_tracking_loop(self):
        """Dummy tracking for testing without camera."""
        print("[ZED] Running dummy tracking mode")
        t_start = time.time()
        
        while self.running:
            t = time.time() - t_start
            
            # Simulate arm movement with sinusoids (in local body frame, meters)
            arm_data = ArmTrackingData(
                timestamp=time.time(),
                valid=True,
                body_com=np.array([0.0, 1.0, 2.0], dtype=np.float32),  # Simulated COM in world
                # Left arm positions relative to COM
                left_shoulder=np.array([0.0, 0.2, 0.0], dtype=np.float32),
                left_elbow=np.array([-0.15 + 0.2*np.sin(t), 0.35 + 0.1*np.sin(t*0.7), 0.1*np.cos(t)], dtype=np.float32),
                left_wrist=np.array([-0.20 + 0.3*np.sin(t), 0.50 + 0.2*np.sin(t*0.7), 0.2*np.cos(t)], dtype=np.float32),
                left_confidence=90.0,
                # Right arm positions relative to COM
                right_shoulder=np.array([0.0, -0.2, 0.0], dtype=np.float32),
                right_elbow=np.array([-0.15 + 0.2*np.sin(t + np.pi), -0.35 + 0.1*np.sin(t*0.7 + np.pi), 0.1*np.cos(t + np.pi)], dtype=np.float32),
                right_wrist=np.array([-0.20 + 0.3*np.sin(t + np.pi), -0.50 + 0.2*np.sin(t*0.7 + np.pi), 0.2*np.cos(t + np.pi)], dtype=np.float32),
                right_confidence=90.0
            )
            
            try:
                self.data_queue.put_nowait(arm_data)
            except queue.Full:
                pass
                
            time.sleep(0.033)  # ~30 Hz


class KinematicRetargeter:
    """Real-time IK-based retargeting using KinDynLib."""
    
    def __init__(self):
        """Initialize the robot model and IK solver."""
        print("[IK] Initializing Themis robot model...")
        
        self.robot = Robot()
        
        # Initial joint configuration (from test_themis_fk.py)
        self.joint_offset = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # floating base
            0.0, 0.0, -0.1, 0.1, -0.1, 0.0,  # right leg
            0.0, 0.0, -0.1, 0.1, -0.1, 0.0,  # left leg
            0, 1.57, -1.57, 0.78, 0.0, 0.78, 0,  # right arm
            0, -1.57, -1.57, -0.78, 0.0, -0.78, 0,  # left arm
            0.0, 0.0  # head
        ], dtype=np.float32)
        
        # Current state
        self.q = np.zeros(themis_const.DOF, dtype=np.float32) + self.joint_offset*0
        self.dq = np.zeros(themis_const.DOF, dtype=np.float32)
        
        # Hand/elbow offsets for FK
        self.hand_r_offset = themis_const.hand_r
        self.hand_l_offset = themis_const.hand_l
        self.elbow_r_offset = np.array([0.0, -0.16, 0.0], dtype=np.float32)
        self.elbow_l_offset = np.array([0.0, 0.16, 0.0], dtype=np.float32)
        
        # Transform matrices
        self.Xhand_l = Xtrans(self.hand_l_offset)
        self.Xhand_r = Xtrans(self.hand_r_offset)
        self.Xelbow_l = Xtrans(self.elbow_l_offset)
        self.Xelbow_r = Xtrans(self.elbow_r_offset)
        
        # Link indices for FK/IK
        self.hand_r_link = 22
        self.hand_l_link = 29
        self.elbow_r_link = 20
        self.elbow_l_link = 27
        
        # COM desired (maintain balance)
        self.com_des = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Initialize robot state
        self.robot.update(self.q, self.dq)
        
        # Last valid arm data for hold on tracking loss
        self.last_valid_arm_data: Optional[ArmTrackingData] = None
        
        # Previous timestep state for error comparison
        self.prev_q = self.q.copy()
        self.prev_dq = self.dq.copy()
        self.prev_tracking_error = float('inf')  # Start with infinite error
        
        # Previous desired positions for fallback threshold check
        self.prev_desired_positions = None  # Will be set on first IK call
        self.desired_change_threshold = 0.01  # Only fallback if desired moved less than 2cm
        
        # Output smoothing filter
        self.filtered_q = self.q.copy()  # Filtered joint angles for output
        self.filter_alpha = .5  # Low-pass filter coefficient (0-1, lower = smoother)
        self.max_joint_delta = 10  # Maximum allowed joint change per step (radians)
        
        # IK initialization mode: 'warm_start', 'random', or 'default_pose'
        self.init_mode = 'warm_start'  # Options: 'warm_start', 'random', 'default_pose'
        
        # Joint limits for random initialization (arm joints only)
        self.arm_r_min = np.array([-3.14, -1.7, -3.14, 0, -3.14, 0, -3.14], dtype=np.float32)
        self.arm_r_max = np.array([3.14, 1.7, 3.14, 2.5, 3.14, 2.0, 3.14], dtype=np.float32)
        self.arm_l_min = np.array([-3.14, -1.7, -3.14, -2.5, -3.14, -2.0, -3.14], dtype=np.float32)
        self.arm_l_max = np.array([3.14, 1.7, 3.14, 0, 3.14, 0, 3.14], dtype=np.float32)
        
        # Default pose for 'default_pose' mode
        # Arms slightly forward and out to avoid shoulder gimbal lock singularity
        # Joint order: SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW_PITCH, ELBOW_YAW, WRIST_PITCH, WRIST_YAW
        # Avoid: SHOULDER_ROLL = ±90° combined with SHOULDER_YAW = ±90° (gimbal lock)
        self.default_pose = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # floating base
            0.0, 0.0, -0.1, 0.1, -0.1, 0.0,  # right leg
            0.0, 0.0, -0.1, 0.1, -0.1, 0.0,  # left leg
            0, 1.57, -1.57, 0.78, 0.1, 0.78, 0.1, # right arm
            0, -1.57, -1.57, -0.78, 0.1, -0.78, -0.1,  # left arm
            0.0, 0.0  # head
        ], dtype=np.float32)
        
        print("[IK] Robot model initialized")
        print(f"[IK] Initialization mode: {self.init_mode}")
        
    def transform_zed_to_robot(self, arm_data: ArmTrackingData) -> Dict[str, np.ndarray]:
        """
        Transform ZED arm positions to robot-frame desired positions.
        
        Input: Arm positions already in local body frame (relative to COM, keypoint 2).
        
        ZED coordinate system (RIGHT_HANDED_Y_UP):
            X: right (person's left from camera view), Y: up, Z: backward (towards camera)
            When facing camera: your right arm is on camera's left (negative X)
            
        Robot coordinate system (from test_themis_fk.py):
            X: forward, Y: left, Z: up
            Example from test: x_hand_l_des = [0.0, 0.0, 0.0, 0.0, 0.40, -0.20] (y=+0.40 for left hand)
            Example from test: x_hand_r_des = [0.0, 0.0, 0.0, 0.0, -0.40, -0.20] (y=-0.40 for right hand)
            
        Returns desired positions for hands and elbows in robot frame.
        """
        # Transformation: ZED -> Robot
        # When person faces camera:
        # - ZED X positive = person's left = robot's right (from robot's perspective) = -Y robot
        # - ZED Y positive = up = +Z robot
        # - ZED Z positive = toward camera = backward = -X robot
        #
        # Robot X = ZED Z (forward is positive ZED Z, i.e. toward camera)
        # Robot Y = -ZED X (robot left is negative ZED X)  
        # Robot Z = ZED Y (up is same)
        
        def zed_to_robot(pos_zed):
            return np.array([pos_zed[2], -pos_zed[0], pos_zed[1]], dtype=np.float32)
            # return np.array([pos_zed[0], pos_zed[1], pos_zed[2]], dtype=np.float32)
        
        # Data is already in local body frame (relative to COM)
        # Convert to robot frame directly
        left_elbow_robot = zed_to_robot(arm_data.left_elbow)
        left_wrist_robot = zed_to_robot(arm_data.left_wrist)
        right_elbow_robot = zed_to_robot(arm_data.right_elbow)
        right_wrist_robot = zed_to_robot(arm_data.right_wrist)

        # #overwrite:
        # left_elbow_robot = np.array([0.1, 0.3, 0.0], dtype=np.float32)
        # left_wrist_robot = np.array([0.3, 0.3, 0.0], dtype=np.float32)
        # right_elbow_robot = np.array([0.1, -0.3, 0.0], dtype=np.float32)
        # right_wrist_robot = np.array([0.3, -0.3, 0.0], dtype=np.float32)

        #overwrite by sinusoid commands:
        # t = time.time()
        # left_elbow_robot = np.array([0.1 + 0.05*np.sin(t), 0.3 + 0.05*np.sin(t*0.7), 0.0], dtype=np.float32)
        # left_wrist_robot = np.array([0.3 + 0.1*np.sin(t), 0.3 + 0.1 *np.sin(t*0.7), 0.0], dtype=np.float32)
        # right_elbow_robot = np.array([0.1 + 0.05*np.sin(t + np.pi), -0.3 + 0.05*np.sin(t*0.7 + np.pi), 0.0], dtype=np.float32)
        # right_wrist_robot = np.array([0.3 + 0.1*np.sin(t + np.pi), -0.3 + 0.1 *np.sin(t*0.7 + np.pi), 0.0], dtype=np.float32)
        
        # Scale to robot dimensions (human arm ~60cm, robot arm ~40cm)
        scale = 0.9
        
        # Build desired task-space vectors [orientation(3), position(3)]
        # Orientation is set to zero (we only track position)
        zero_orient = np.zeros(3, dtype=np.float32)
        
        return {
            'x_hand_l_des': np.concatenate([zero_orient, left_wrist_robot * scale]),
            'x_hand_r_des': np.concatenate([zero_orient, right_wrist_robot * scale]),
            'x_elbow_l_des': np.concatenate([zero_orient, left_elbow_robot * scale]),
            'x_elbow_r_des': np.concatenate([zero_orient, right_elbow_robot * scale])
        }
        
    def solve_ik(self, arm_data: ArmTrackingData, num_iterations: int = 5, verbose: bool = False) -> tuple:
        """
        Solve IK to get joint angles from arm position targets.
        
        Uses current joint state (self.q) as starting point for IK iterations.
        Stores result in self.q for use as starting point in next timestep.
        
        Args:
            arm_data: Arm position data from ZED
            num_iterations: Number of IK iterations (fewer for real-time)
            verbose: Print debug info for body-frame data and IK solutions
            
        Returns:
            Tuple of (joint_angles, desired_positions, actual_positions)
            - joint_angles: 28-element array of actuated joint angles
            - desired_positions: Dict with elbow/hand desired positions in robot frame
            - actual_positions: Dict with elbow/hand actual positions from FK
        """
        MIN_CONFIDENCE = 30.0
        
        # Default return values (current pose, no markers)
        empty_positions = {
            'elbow_l': np.zeros(3), 'elbow_r': np.zeros(3),
            'hand_l': np.zeros(3), 'hand_r': np.zeros(3)
        }
        
        # Check if tracking is valid
        if not arm_data.valid:
            if self.last_valid_arm_data is not None:
                arm_data = self.last_valid_arm_data
            else:
                return self.q[6:].copy(), empty_positions, empty_positions
                
        # Check arm confidence
        if arm_data.left_confidence < MIN_CONFIDENCE or arm_data.right_confidence < MIN_CONFIDENCE:
            if self.last_valid_arm_data is not None:
                arm_data = self.last_valid_arm_data
            else:
                return self.q[6:].copy(), empty_positions, empty_positions
                
        # Save valid arm data
        self.last_valid_arm_data = arm_data
        
        # Transform to robot frame
        desired = self.transform_zed_to_robot(arm_data)
        
        # Print body-frame positions if verbose
        if verbose:
            print("\n--- ZED Body-Frame Tracking Data (meters, local to COM) ---")
            print(f"  ZED coords: X=right, Y=up, Z=backward (toward camera)")
            print(f"  L Elbow (ZED): [{arm_data.left_elbow[0]:+.3f}, {arm_data.left_elbow[1]:+.3f}, {arm_data.left_elbow[2]:+.3f}]")
            print(f"  L Wrist (ZED): [{arm_data.left_wrist[0]:+.3f}, {arm_data.left_wrist[1]:+.3f}, {arm_data.left_wrist[2]:+.3f}]")
            print(f"  R Elbow (ZED): [{arm_data.right_elbow[0]:+.3f}, {arm_data.right_elbow[1]:+.3f}, {arm_data.right_elbow[2]:+.3f}]")
            print(f"  R Wrist (ZED): [{arm_data.right_wrist[0]:+.3f}, {arm_data.right_wrist[1]:+.3f}, {arm_data.right_wrist[2]:+.3f}]")
            print(f"  Conf: L={arm_data.left_confidence:.0f}%, R={arm_data.right_confidence:.0f}%")
            print("--- Robot-Frame Desired (X=fwd, Y=left, Z=up, scaled) ---")
            print(f"  L Elbow des: [{desired['x_elbow_l_des'][3]:+.3f}, {desired['x_elbow_l_des'][4]:+.3f}, {desired['x_elbow_l_des'][5]:+.3f}]")
            print(f"  L Wrist des: [{desired['x_hand_l_des'][3]:+.3f}, {desired['x_hand_l_des'][4]:+.3f}, {desired['x_hand_l_des'][5]:+.3f}]")
            print(f"  R Elbow des: [{desired['x_elbow_r_des'][3]:+.3f}, {desired['x_elbow_r_des'][4]:+.3f}, {desired['x_elbow_r_des'][5]:+.3f}]")
            print(f"  R Wrist des: [{desired['x_hand_r_des'][3]:+.3f}, {desired['x_hand_r_des'][4]:+.3f}, {desired['x_hand_r_des'][5]:+.3f}]")
        
        # Initialize q based on mode: 'warm_start', 'random', or 'default_pose'
        if self.init_mode == 'warm_start':
            # Warm start: use previous timestep's solution
            q_current = self.q.copy()
            dq_current = self.dq.copy()
        elif self.init_mode == 'default_pose':
            # Default pose: use predefined arm configuration
            q_current = self.default_pose.copy()
            dq_current = np.zeros(themis_const.DOF, dtype=np.float32)
        else:  # 'random'
            # Random initialization: randomize arm joints only
            q_current = np.zeros(themis_const.DOF, dtype=np.float32)
            # Randomize right arm (indices 18-24, which is q[18:25])
            q_current[18:25] = np.random.uniform(self.arm_r_min, self.arm_r_max).astype(np.float32)
            # Randomize left arm (indices 25-31, which is q[25:32])
            q_current[25:32] = np.random.uniform(self.arm_l_min, self.arm_l_max).astype(np.float32)
            dq_current = np.zeros(themis_const.DOF, dtype=np.float32)
        
        # Iterative IK
        for _ in range(num_iterations):
            self.robot.update(q_current, dq_current)
            
            # Compute FK for current configuration
            x_hand_r, _ = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
            x_hand_l, _ = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
            x_elbow_r, _ = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
            x_elbow_l, _ = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
            
            # Compute Jacobians
            J_hand_r = self.robot.compute_body_jacobian(self.hand_r_link, self.Xhand_r)
            J_hand_l = self.robot.compute_body_jacobian(self.hand_l_link, self.Xhand_l)
            J_elbow_r = self.robot.compute_body_jacobian(self.elbow_r_link, self.Xelbow_r)
            J_elbow_l = self.robot.compute_body_jacobian(self.elbow_l_link, self.Xelbow_l)
            
            # Solve IK step
            # q_des, dq_des = self.robot.update_task_space_command_with_constraints( # analytical solver
            #     desired['x_elbow_l_des'], desired['x_elbow_r_des'],
            #     x_elbow_l, x_elbow_r,
            #     desired['x_hand_l_des'], desired['x_hand_r_des'],
            #     x_hand_l, x_hand_r,
            #     J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
            #     self.com_des
            # ) 
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
        
        # Store current state for next timestep's IK starting point
        # Wrap joint angles to [-pi, pi] range
        q_current[7:] = np.arctan2(np.sin(q_current[7:]), np.cos(q_current[7:]))
        self.q = q_current
        self.dq = dq_current
        
        # Apply smoothing filter to reduce jitter
        # Low-pass exponential filter with rate limiting
        q_delta = self.q[7:] - self.filtered_q[7:]
        # Handle angle wrapping in delta calculation
        q_delta = np.arctan2(np.sin(q_delta), np.cos(q_delta))
        # Rate limit: clamp maximum change per step
        q_delta = np.clip(q_delta, -self.max_joint_delta, self.max_joint_delta)
        # Apply low-pass filter
        self.filtered_q[7:] = self.filtered_q[7:] + self.filter_alpha * q_delta
        # Wrap filtered output
        self.filtered_q[7:] = np.arctan2(np.sin(self.filtered_q[7:]), np.cos(self.filtered_q[7:]))
        
        # Compute final FK for actual positions (after IK)
        self.robot.update(self.q, self.dq)
        x_hand_r_final, _ = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
        x_hand_l_final, _ = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
        x_elbow_r_final, _ = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
        x_elbow_l_final, _ = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
        
        # Compute current tracking error (sum of squared position errors)
        error_hand_r = np.linalg.norm(desired['x_hand_r_des'][3:6] - x_hand_r_final[3:6])**2
        error_hand_l = np.linalg.norm(desired['x_hand_l_des'][3:6] - x_hand_l_final[3:6])**2
        error_elbow_r = np.linalg.norm(desired['x_elbow_r_des'][3:6] - x_elbow_r_final[3:6])**2
        error_elbow_l = np.linalg.norm(desired['x_elbow_l_des'][3:6] - x_elbow_l_final[3:6])**2
        current_tracking_error = error_hand_r + error_hand_l + error_elbow_r + error_elbow_l
        
        # Check if desired positions have changed significantly
        desired_changed_significantly = True  # Default: don't fallback
        if self.prev_desired_positions is not None:
            # Compute max change in desired positions
            change_hand_r = np.linalg.norm(desired['x_hand_r_des'][3:6] - self.prev_desired_positions['x_hand_r_des'][3:6])
            change_hand_l = np.linalg.norm(desired['x_hand_l_des'][3:6] - self.prev_desired_positions['x_hand_l_des'][3:6])
            change_elbow_r = np.linalg.norm(desired['x_elbow_r_des'][3:6] - self.prev_desired_positions['x_elbow_r_des'][3:6])
            change_elbow_l = np.linalg.norm(desired['x_elbow_l_des'][3:6] - self.prev_desired_positions['x_elbow_l_des'][3:6])
            max_desired_change = max(change_hand_r, change_hand_l, change_elbow_r, change_elbow_l)
            desired_changed_significantly = max_desired_change > self.desired_change_threshold
        
        # Only fallback if: error is worse AND desired hasn't moved much
        # If desired moved significantly, always accept new solution (target moved, so error increase is expected)
        should_fallback = (current_tracking_error > self.prev_tracking_error) and (not desired_changed_significantly)
        
        if should_fallback:
            # Keep self.q as the new solution for next iteration's starting point (avoid local minima)
            # But use fallback (prev_q) for the output command
            
            # Reset filtered_q toward prev_q for smooth fallback output
            self.filtered_q[7:] = self.prev_q[7:].copy()
            
            # Recompute FK with FALLBACK state for visualization (shows where robot will actually be)
            self.robot.update(self.prev_q, self.prev_dq)
            x_hand_r_final, _ = self.robot.compute_forward_kinematics(self.hand_r_link, self.hand_r_offset)
            x_hand_l_final, _ = self.robot.compute_forward_kinematics(self.hand_l_link, self.hand_l_offset)
            x_elbow_r_final, _ = self.robot.compute_forward_kinematics(self.elbow_r_link, self.elbow_r_offset)
            x_elbow_l_final, _ = self.robot.compute_forward_kinematics(self.elbow_l_link, self.elbow_l_offset)
            
            # Don't update prev_q/prev_dq - keep previous good solution as reference
            # Don't update prev_tracking_error - keep the better error threshold
            
            if verbose:
                print(f"  [Fallback] Current error {np.sqrt(current_tracking_error):.4f} > prev {np.sqrt(self.prev_tracking_error):.4f} (IK continues from new, output uses prev)")
        else:
            # Update previous state for next timestep
            self.prev_q = self.q.copy()
            self.prev_dq = self.dq.copy()
            self.prev_tracking_error = current_tracking_error
        
        # Always update previous desired positions for next timestep comparison
        self.prev_desired_positions = {
            'x_hand_r_des': desired['x_hand_r_des'].copy(),
            'x_hand_l_des': desired['x_hand_l_des'].copy(),
            'x_elbow_r_des': desired['x_elbow_r_des'].copy(),
            'x_elbow_l_des': desired['x_elbow_l_des'].copy(),
        }
        
        # Extract position parts (last 3 elements of 6-vector)
        # Add robot base height offset for visualization in MuJoCo
        base_offset = np.array([0.0, 0.0, 1.0])  # Robot is at z=1.0
        
        desired_positions = {
            'elbow_l': desired['x_elbow_l_des'][3:6] + base_offset,
            'elbow_r': desired['x_elbow_r_des'][3:6] + base_offset,
            'hand_l': desired['x_hand_l_des'][3:6] + base_offset,
            'hand_r': desired['x_hand_r_des'][3:6] + base_offset,
        }
        
        actual_positions = {
            'elbow_l': x_elbow_l_final[3:6] + base_offset,
            'elbow_r': x_elbow_r_final[3:6] + base_offset,
            'hand_l': x_hand_l_final[3:6] + base_offset,
            'hand_r': x_hand_r_final[3:6] + base_offset,
        }
        
        # Print IK solution if verbose
        if verbose:
            arm_joints = self.filtered_q[6:]  # Filtered actuated joints
            # Right arm joints: indices 12-18
            # Left arm joints: indices 19-25
            print("--- IK Solution (arm joints, rad) ---")
            print(f"  R Arm: [{arm_joints[12]:.3f}, {arm_joints[13]:.3f}, {arm_joints[14]:.3f}, {arm_joints[15]:.3f}, {arm_joints[16]:.3f}, {arm_joints[17]:.3f}, {arm_joints[18]:.3f}]")
            print(f"  L Arm: [{arm_joints[19]:.3f}, {arm_joints[20]:.3f}, {arm_joints[21]:.3f}, {arm_joints[22]:.3f}, {arm_joints[23]:.3f}, {arm_joints[24]:.3f}, {arm_joints[25]:.3f}]")
            print("--- Actual FK Positions (meters) ---")
            print(f"  L Elbow act: [{x_elbow_l_final[3]:+.3f}, {x_elbow_l_final[4]:+.3f}, {x_elbow_l_final[5]:+.3f}]")
            print(f"  L Hand act:  [{x_hand_l_final[3]:+.3f}, {x_hand_l_final[4]:+.3f}, {x_hand_l_final[5]:+.3f}]")
            print(f"  R Elbow act: [{x_elbow_r_final[3]:+.3f}, {x_elbow_r_final[4]:+.3f}, {x_elbow_r_final[5]:+.3f}]")
            print(f"  R Hand act:  [{x_hand_r_final[3]:+.3f}, {x_hand_r_final[4]:+.3f}, {x_hand_r_final[5]:+.3f}]")
        
        # Return filtered joint angles and positions for visualization
        return self.filtered_q[6:].copy(), desired_positions, actual_positions


class MuJoCoVisualizer:
    """MuJoCo-based robot KINEMATICS visualization (no dynamics)."""
    
    def __init__(self, model_path: str):
        """Initialize MuJoCo model and viewer."""
        print(f"[MuJoCo] Loading model from {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Marker positions for visualization (desired and actual)
        # Format: [x, y, z] in world frame
        self.markers = {
            'elbow_l_des': np.zeros(3),
            'elbow_r_des': np.zeros(3),
            'hand_l_des': np.zeros(3),
            'hand_r_des': np.zeros(3),
            'elbow_l_act': np.zeros(3),
            'elbow_r_act': np.zeros(3),
            'hand_l_act': np.zeros(3),
            'hand_r_act': np.zeros(3),
        }
        
        # Joint name to index mapping
        self.joint_map = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_map[name] = i
                
        print(f"[MuJoCo] Loaded {len(self.joint_map)} joints")
        print(f"[MuJoCo] Using kinematics-only mode (no dynamics simulation)")
        
    def start_viewer(self):
        """Start the viewer with custom rendering callback."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Configure camera
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 90
            
    def update_markers(self, desired: Dict[str, np.ndarray], actual: Dict[str, np.ndarray]):
        """
        Update marker positions for visualization.
        
        Args:
            desired: Dict with keys 'elbow_l', 'elbow_r', 'hand_l', 'hand_r' - desired positions
            actual: Dict with keys 'elbow_l', 'elbow_r', 'hand_l', 'hand_r' - actual FK positions
        """
        self.markers['elbow_l_des'] = desired.get('elbow_l', np.zeros(3))
        self.markers['elbow_r_des'] = desired.get('elbow_r', np.zeros(3))
        self.markers['hand_l_des'] = desired.get('hand_l', np.zeros(3))
        self.markers['hand_r_des'] = desired.get('hand_r', np.zeros(3))
        self.markers['elbow_l_act'] = actual.get('elbow_l', np.zeros(3))
        self.markers['elbow_r_act'] = actual.get('elbow_r', np.zeros(3))
        self.markers['hand_l_act'] = actual.get('hand_l', np.zeros(3))
        self.markers['hand_r_act'] = actual.get('hand_r', np.zeros(3))
        
    def _render_markers(self):
        """Render marker spheres for desired and actual positions.
        
        Color coding:
        - Elbows desired: Orange (bright)
        - Elbows actual: Cyan (bright)
        - Hands desired: Red (bright)
        - Hands actual: Green (bright)
        """
        if self.viewer is None:
            return
            
        # Get the scene from viewer
        with self.viewer.lock():
            ngeom = 0
            
            # Elbow desired markers (Orange)
            for key in ['elbow_l_des', 'elbow_r_des']:
                pos = self.markers[key]
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.035, 0.035, 0.035]  # 3.5cm radius
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = [1.0, 0.6, 0.0, 1.0]  # Bright orange for elbow desired
                    ngeom += 1
            
            # Elbow actual markers (Cyan)
            for key in ['elbow_l_act', 'elbow_r_act']:
                pos = self.markers[key]
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.035, 0.035, 0.035]  # 3.5cm radius
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = [0.0, 1.0, 1.0, 1.0]  # Bright cyan for elbow actual
                    ngeom += 1
            
            # Hand desired markers (Red)
            for key in ['hand_l_des', 'hand_r_des']:
                pos = self.markers[key]
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.04, 0.04, 0.04]  # 4cm radius (slightly larger for hands)
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = [1.0, 0.2, 0.2, 1.0]  # Bright red for hand desired
                    ngeom += 1
            
            # Hand actual markers (Green)
            for key in ['hand_l_act', 'hand_r_act']:
                pos = self.markers[key]
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.04, 0.04, 0.04]  # 4cm radius (slightly larger for hands)
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = [0.2, 1.0, 0.2, 1.0]  # Bright green for hand actual
                    ngeom += 1
                    
            self.viewer.user_scn.ngeom = ngeom
            
    def set_joint_positions(self, joint_angles: np.ndarray):
        """
        Set joint positions from array.
        
        Args:
            joint_angles: 28-element array of joint angles
        """
        # MuJoCo qpos layout: [freejoint(7), actuated_joints(28)]
        # freejoint: [x, y, z, qw, qx, qy, qz]
        # actuated joints order matches robot_const and XML:
        # - Right leg (6): HIP_YAW_R, HIP_ROLL_R, HIP_PITCH_R, KNEE_PITCH_R, ANKLE_PITCH_R, ANKLE_ROLL_R
        # - Left leg (6): HIP_YAW_L, HIP_ROLL_L, HIP_PITCH_L, KNEE_PITCH_L, ANKLE_PITCH_L, ANKLE_ROLL_L
        # - Right arm (7): SHOULDER_PITCH_R, SHOULDER_ROLL_R, SHOULDER_YAW_R, ELBOW_PITCH_R, ELBOW_YAW_R, WRIST_PITCH_R, WRIST_YAW_R
        # - Left arm (7): SHOULDER_PITCH_L, SHOULDER_ROLL_L, SHOULDER_YAW_L, ELBOW_PITCH_L, ELBOW_YAW_L, WRIST_PITCH_L, WRIST_YAW_L
        # - Head (2): HEAD_YAW, HEAD_PITCH
        
        # Directly set qpos[7:] (after freejoint)
        self.data.qpos[7:7+28] = joint_angles
        
        # Use kinematics only (no dynamics) - just compute positions from joint angles
        mujoco.mj_kinematics(self.model, self.data)
        
    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """Set base position and orientation."""
        # Free joint is first 7 elements of qpos (position + quaternion)
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = quaternion
        
    def sync(self):
        """Sync viewer with simulation state and render markers."""
        if self.viewer is not None:
            self._render_markers()
            self.viewer.sync()
            
    def is_running(self) -> bool:
        """Check if viewer is still open."""
        return self.viewer is not None and self.viewer.is_running()


class IntegratedRetargetingPipeline:
    """Main pipeline integrating tracking, IK, and visualization."""
    
    def __init__(self):
        """Initialize all components."""
        print("="*60)
        print("INTEGRATED KINEMATIC RETARGETING PIPELINE")
        print("="*60)
        
        # Data queue for tracking data
        self.tracking_queue = queue.Queue(maxsize=5)
        
        # Initialize components
        self.tracker = ZEDTracker(self.tracking_queue)
        self.retargeter = KinematicRetargeter()
        
        # Find model path
        model_path = os.path.join(os.path.dirname(__file__), 'westwood_robots', 'TH02-A7-v2.xml')
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), 'themis', 'TH02-A7-v2.xml')
        self.visualizer = MuJoCoVisualizer(model_path)
        
        # Latest tracking data
        self.latest_arm_data: Optional[ArmTrackingData] = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Debug printing control
        self.verbose_ik = False  # Print body-frame data and IK solutions
        self.print_interval = 10  # Print every N frames
        
    def run(self):
        """Main loop with detailed timing."""
        print("\n[Pipeline] Starting...")
        
        # Start tracking
        self.tracker.start()
        
        # Start MuJoCo viewer
        self.visualizer.start_viewer()
        
        # Set initial base pose (floating above ground)
        base_pos = np.array([0.0, 0.0, 1.0])
        base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.visualizer.set_base_pose(base_pos, base_quat)
        
        print("[Pipeline] Running main loop (press 'q' in ZED window or close MuJoCo to exit)")
        print("[Pipeline] Verbose mode: printing body-frame data and IK solutions")
        
        # Timing accumulators
        ik_time_ms = 0.0
        viz_time_ms = 0.0
        loop_time_ms = 0.0
        
        try:
            while self.visualizer.is_running():
                loop_start = time.perf_counter()
                
                # Get latest tracking data (drain queue)
                queue_start = time.perf_counter()
                while not self.tracking_queue.empty():
                    try:
                        self.latest_arm_data = self.tracking_queue.get_nowait()
                    except queue.Empty:
                        break
                queue_time_ms = (time.perf_counter() - queue_start) * 1000
                
                # Run IK if we have tracking data
                if self.latest_arm_data is not None:
                    ik_start = time.perf_counter()
                    
                    # Print verbose info every N frames
                    verbose = self.verbose_ik and (self.frame_count % self.print_interval == 0)
                    joint_angles, desired_pos, actual_pos = self.retargeter.solve_ik(
                        self.latest_arm_data, 
                        num_iterations=1,
                        verbose=verbose
                    )
                    ik_time_ms = (time.perf_counter() - ik_start) * 1000
                    
                    # Update visualization with joint angles and marker positions
                    viz_start = time.perf_counter()
                    self.visualizer.set_joint_positions(joint_angles)
                    self.visualizer.update_markers(desired_pos, actual_pos)
                    viz_time_ms = (time.perf_counter() - viz_start) * 1000
                
                # Sync viewer
                sync_start = time.perf_counter()
                self.visualizer.sync()
                sync_time_ms = (time.perf_counter() - sync_start) * 1000
                
                # Total loop time
                loop_time_ms = (time.perf_counter() - loop_start) * 1000
                
                # FPS and timing report
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 2.0:
                    fps = self.frame_count / (time.time() - self.last_fps_time)
                    print(f"\n[Timing] Loop: {loop_time_ms:.1f}ms | IK: {ik_time_ms:.2f}ms | Viz: {viz_time_ms:.2f}ms | Sync: {sync_time_ms:.2f}ms | FPS: {fps:.1f}")
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                    
                # Small sleep to prevent CPU spinning
                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.01:  # Target ~100Hz max
                    time.sleep(0.01 - elapsed)
                    
        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted")
        finally:
            self.tracker.stop()
            print("[Pipeline] Shutdown complete")


def main():
    pipeline = IntegratedRetargetingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
