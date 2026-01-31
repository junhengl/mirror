"""
Integrated ZED body tracking with MuJoCo robot visualization.

This pipeline runs ZED body tracking in parallel with the robot visualizer,
passing body tracking data in real-time. Currently displays sinwave animations
while receiving tracking data (retargeting to be added later).

Note: This script requires sudo permissions for ZED motion sensors:
    sudo /path/to/python integrated_tracking_visualization.py
"""

## sudo /home/junhengl/body_tracking/.venv/bin/python integrated_tracking_visualization.py

import numpy as np
import time
import threading
import queue
import os
from dataclasses import dataclass
from typing import List, Optional
import sys

# MuJoCo visualizer
from mujoco_visualizer import ThemisSimulator

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
class BodyTrackingData:
    """Container for body tracking data from ZED camera."""
    timestamp: float
    num_bodies: int
    keypoints: List[np.ndarray]  # List of keypoint positions for each detected body
    confidences: List[np.ndarray]  # List of confidence values for each body


class PerformanceMonitor:
    """Monitor and report performance metrics."""
    
    def __init__(self, name: str, report_interval: float = 2.0):
        """
        Initialize performance monitor.
        
        Args:
            name: Name of the module being monitored
            report_interval: How often to print reports (seconds)
        """
        self.name = name
        self.report_interval = report_interval
        self.frame_times = []
        self.last_report_time = time.time()
        self.total_frames = 0
        self.last_frame_time = None
        
    def tick(self):
        """Record a frame/update tick."""
        current_time = time.time()
        self.total_frames += 1
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames (last 100)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
        
        # Print report periodically
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
    
    def report(self):
        """Print performance report."""
        if len(self.frame_times) == 0:
            return
            
        avg_frame_time = np.mean(self.frame_times)
        min_frame_time = np.min(self.frame_times)
        max_frame_time = np.max(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"[{self.name}] FPS: {fps:.1f} | "
              f"Frame time: {avg_frame_time*1000:.1f}ms (min: {min_frame_time*1000:.1f}ms, max: {max_frame_time*1000:.1f}ms) | "
              f"Total frames: {self.total_frames}")
    
    def get_stats(self):
        """Get current statistics."""
        if len(self.frame_times) == 0:
            return {"fps": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        
        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            "fps": fps,
            "avg_ms": avg_frame_time * 1000,
            "min_ms": np.min(self.frame_times) * 1000,
            "max_ms": np.max(self.frame_times) * 1000,
            "total_frames": self.total_frames
        }
    
    
class ZEDBodyTracker:
    """Manages ZED camera and body tracking in a separate thread."""
    
    def __init__(self, data_queue: queue.Queue, resolution="HD1080"):
        """
        Initialize ZED body tracker.
        
        Args:
            data_queue: Queue to send tracking data to visualization thread
            resolution: Camera resolution (HD1080, HD720, etc.)
        """
        self.data_queue = data_queue
        self.resolution = resolution
        self.running = False
        self.thread = None
        self.zed = None
        self.perf_monitor = PerformanceMonitor("ZED Tracker", report_interval=2.0)
        
    def start(self):
        """Start the body tracking thread."""
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        print("[ZED Tracker] Started body tracking thread")
        
    def stop(self):
        """Stop the body tracking thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.zed:
            self.zed.disable_body_tracking()
            self.zed.disable_positional_tracking()
            self.zed.close()
        print("[ZED Tracker] Stopped")
        
    def _tracking_loop(self):
        """Main tracking loop running in separate thread."""
        if not ZED_AVAILABLE:
            print("[ZED Tracker] ZED SDK not available, running dummy tracker")
            self._dummy_tracking_loop()
            return
            
        # Initialize ZED camera
        self.zed = sl.Camera()
        
        # Configure initialization parameters
        init_params = sl.InitParameters()
        if self.resolution == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif self.resolution == "HD720":
            init_params.camera_resolution = sl.RESOLUTION.HD720
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        # Open camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED Tracker] Error opening camera: {err}")
            self.running = False
            return
            
        print("[ZED Tracker] Camera opened successfully")
        
        # Enable positional tracking
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        
        # Enable body tracking
        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking = True
        body_param.enable_body_fitting = False
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_param.body_format = sl.BODY_FORMAT.BODY_38
        
        err = self.zed.enable_body_tracking(body_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED Tracker] Error enabling body tracking: {err}")
            print("[ZED Tracker] Falling back to dummy mode")
            self.zed.close()
            self._dummy_tracking_loop()
            return
            
        print("[ZED Tracker] Body tracking enabled")
        
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 40
        
        # Get camera info for display resolution
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
        
        # Main tracking loop
        while self.running:
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve camera image
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                
                # Retrieve bodies
                self.zed.retrieve_bodies(bodies, body_runtime_param)
                
                # Display camera feed with skeleton overlay
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(
                    image_left_ocv,
                    image_scale,
                    bodies.body_list,
                    True,
                    sl.BODY_FORMAT.BODY_38
                )
                cv2.imshow("ZED Body Tracking", image_left_ocv)
                
                # Extract keypoints from detected bodies
                keypoints_list = []
                confidences_list = []
                
                for body in bodies.body_list:
                    # Get keypoints (BODY_38 format has 38 keypoints)
                    kp = np.zeros((38, 3))
                    conf = np.zeros(38)
                    
                    for i in range(38):
                        kp[i] = [
                            body.keypoint[i][0],
                            body.keypoint[i][1],
                            body.keypoint[i][2]
                        ]
                        conf[i] = body.keypoint_confidence[i]
                    
                    keypoints_list.append(kp)
                    confidences_list.append(conf)
                
                # Create tracking data
                tracking_data = BodyTrackingData(
                    timestamp=time.time(),
                    num_bodies=len(bodies.body_list),
                    keypoints=keypoints_list,
                    confidences=confidences_list
                )
                
                # Send to queue (non-blocking, drop if full)
                try:
                    self.data_queue.put_nowait(tracking_data)
                except queue.Full:
                    pass  # Drop frame if queue is full
                
                # Performance monitoring
                self.perf_monitor.tick()
                
                # Handle OpenCV window events
                key = cv2.waitKey(1)
                if key == 113:  # 'q' key
                    print("[ZED Tracker] User requested quit")
                    self.running = False
                    break
                    
            time.sleep(0.001)  # ~100 Hz
        
        # Cleanup
        image.free(sl.MEM.CPU)
        cv2.destroyAllWindows()
    
    def _dummy_tracking_loop(self):
        """Dummy tracking loop for testing without ZED camera."""
        print("[ZED Tracker] Running in dummy mode (no camera)")
        while self.running:
            # Send empty tracking data
            tracking_data = BodyTrackingData(
                timestamp=time.time(),
                num_bodies=0,
                keypoints=[],
                confidences=[]
            )
            try:
                self.data_queue.put_nowait(tracking_data)
            except queue.Full:
                pass
            time.sleep(0.033)  # ~30 Hz


class IntegratedVisualization:
    """Integrates body tracking with robot visualization."""
    
    def __init__(self):
        """Initialize the integrated system."""
        # Create data queue for tracking data
        self.tracking_queue = queue.Queue(maxsize=10)
        
        # Initialize ZED tracker
        self.tracker = ZEDBodyTracker(self.tracking_queue)
        
        # Initialize robot visualizer
        print("[Visualizer] Initializing robot...")
        self.sim = ThemisSimulator(headless=False)
        
        # Latest tracking data
        self.latest_tracking: Optional[BodyTrackingData] = None
        
        # Smoothing filter state
        self.filtered_joints: dict = {}  # Last filtered joint values
        self.filter_alpha = 0.4  # Exponential smoothing factor (0-1, lower = smoother)
        self.max_jump_threshold = 1.0  # Max allowed change per frame (radians) - increased for responsiveness
        self.jump_recovery_rate = 0.1  # How fast to recover towards new value after jump detection
        
        # Last valid arm angles (used when tracking is lost to prevent jumping to zero)
        self.last_valid_right_arm: dict = None
        self.last_valid_left_arm: dict = None
        
        # Performance monitors
        self.retarget_monitor = PerformanceMonitor("Retargeting", report_interval=2.0)
        self.viz_monitor = PerformanceMonitor("Visualization", report_interval=2.0)
        self.latency_samples = []
        
        # Detailed timing measurements (in microseconds)
        self.retarget_times = []
        self.robot_update_times = []
        self.mujoco_forward_times = []
        
        # Configure camera
        if self.sim.viewer is not None:
            self.sim.viewer.cam.lookat[:] = [0.0, 0.0, 1.5]
            self.sim.viewer.cam.distance = 3.0
            self.sim.viewer.cam.elevation = -20
            self.sim.viewer.cam.azimuth = 90
    
    def start(self):
        """Start the integrated system."""
        print("[System] Starting body tracking...")
        self.tracker.start()
        
        print("[System] Starting visualization loop...")
        self._visualization_loop()
    
    def _rotation_matrix_from_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that rotates vec1 to align with vec2.
        
        Args:
            vec1: Source vector (will be normalized)
            vec2: Target vector (will be normalized)
            
        Returns:
            3x3 rotation matrix
        """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        
        # Handle parallel vectors
        dot = np.dot(a, b)
        if dot > 0.9999:
            return np.eye(3)
        if dot < -0.9999:
            # 180 degree rotation around any perpendicular axis
            ortho = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, ortho)
            axis = axis / np.linalg.norm(axis)
            return self._axis_angle_to_rotation(axis, np.pi)
        
        # Rodrigues' rotation formula
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = dot
        
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-10))
        return R
    
    def _axis_angle_to_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to rotation matrix."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    
    def _rotation_to_euler_zyx(self, R: np.ndarray) -> tuple:
        """
        Extract ZYX Euler angles (yaw, pitch, roll) from rotation matrix.
        
        Returns:
            (yaw, pitch, roll) in radians
        """
        # Handle gimbal lock
        if abs(R[2, 0]) >= 1.0 - 1e-6:
            # Gimbal lock
            yaw = 0.0
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return yaw, pitch, roll
    
    def _compute_arm_rotation(self, shoulder: np.ndarray, elbow: np.ndarray, 
                              wrist: np.ndarray, is_left: bool) -> dict:
        """
        Compute shoulder, elbow, and wrist joint angles from keypoint positions
        using rotation transforms.
        
        Coordinate frame convention (ZED camera):
        - X: right (positive to the right of camera)
        - Y: down (positive downward)
        - Z: forward (positive away from camera)
        
        Robot arm neutral pose:
        - Arms hanging down along Y axis (down)
        
        Args:
            shoulder: 3D position of shoulder
            elbow: 3D position of elbow
            wrist: 3D position of wrist
            is_left: True for left arm, False for right arm
            
        Returns:
            Dictionary with shoulder_yaw, shoulder_pitch, shoulder_roll,
            elbow_yaw, elbow_pitch, wrist_yaw, wrist_pitch
        """
        result = {
            'shoulder_yaw': 0.0,
            'shoulder_pitch': 0.0,
            'shoulder_roll': 0.0,
            'elbow_yaw': 0.0,
            'elbow_pitch': 0.0,
            'wrist_yaw': 0.0,
            'wrist_pitch': 0.0,
        }
        
        # Compute upper arm and forearm vectors
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        upper_arm_len = np.linalg.norm(upper_arm)
        forearm_len = np.linalg.norm(forearm)
        
        if upper_arm_len < 0.01 or forearm_len < 0.01:
            return result
        
        # Normalize vectors
        upper_arm_dir = upper_arm / upper_arm_len
        forearm_dir = forearm / forearm_len
        
        # Reference direction: arm pointing down (neutral pose)
        ref_down = np.array([0.0, 1.0, 0.0])  # Y-down in ZED frame
        
        # ===== SHOULDER ANGLES =====
        # Use a more robust angle computation that handles edge cases better
        
        # The upper arm direction in ZED frame:
        # X: right (positive), Y: down (positive), Z: forward (positive)
        ux, uy, uz = upper_arm_dir[0], upper_arm_dir[1], upper_arm_dir[2]
        
        # Shoulder pitch: forward/backward angle from vertical
        # When arm is down (uy > 0), pitch is ~0
        # When arm swings forward (uz > 0, uy decreases), pitch increases
        # When arm swings backward (uz < 0), pitch is negative
        # Use atan2 with proper quadrant handling
        
        # Compute pitch as the angle in the sagittal plane (YZ plane)
        # Reference: arm pointing down (0, 1, 0)
        # The pitch is the angle between the arm's YZ projection and the Y-axis
        yz_magnitude = np.sqrt(uy*uy + uz*uz)
        if yz_magnitude > 0.05:  # Sufficient YZ component for stable computation
            # Angle from Y-down axis, positive when forward (Z+)
            shoulder_pitch = np.arctan2(-uz, uy)  # Negate Z to match robot convention
        else:
            # Arm is mostly in X direction (pointing sideways)
            # Pitch is poorly defined here, use small value based on Z component
            shoulder_pitch = -np.sign(uz) * np.pi / 4 if abs(uz) > 0.01 else 0.0
        
        # Shoulder roll: rotation about Z-axis (arm raise sideways)
        # Project upper arm onto XY plane
        # For right arm: raising outward (X-) should be negative roll
        # For left arm: raising outward (X+) should be positive roll
        xy_proj = np.array([upper_arm_dir[0], upper_arm_dir[1], 0])
        xy_len = np.linalg.norm(xy_proj)
        if xy_len > 0.01:
            # Angle from Y-down to projected vector
            if is_left:
                shoulder_roll = np.arctan2(upper_arm_dir[0], upper_arm_dir[1])
            else:
                shoulder_roll = -np.arctan2(upper_arm_dir[0], upper_arm_dir[1])
        else:
            shoulder_roll = 0.0
        
        # Shoulder yaw: twist around upper arm axis
        # Use forearm to determine twist
        # Project forearm perpendicular to upper arm
        forearm_perp = forearm_dir - np.dot(forearm_dir, upper_arm_dir) * upper_arm_dir
        forearm_perp_len = np.linalg.norm(forearm_perp)
        
        if forearm_perp_len > 0.1:
            forearm_perp = forearm_perp / forearm_perp_len
            
            # Reference perpendicular (based on where forearm would be with no twist)
            # When arm is down, forearm bends forward (Z+)
            ref_perp = np.cross(upper_arm_dir, np.array([1.0, 0.0, 0.0]))
            ref_perp_len = np.linalg.norm(ref_perp)
            
            if ref_perp_len > 0.1:
                ref_perp = ref_perp / ref_perp_len
                # Compute yaw angle
                cos_yaw = np.clip(np.dot(forearm_perp, ref_perp), -1.0, 1.0)
                sin_yaw = np.dot(np.cross(ref_perp, forearm_perp), upper_arm_dir)
                shoulder_yaw = np.arctan2(sin_yaw, cos_yaw)
            else:
                shoulder_yaw = 0.0
        else:
            shoulder_yaw = 0.0
        
        # ===== ELBOW ANGLES =====
        # Elbow pitch: angle between upper arm and forearm
        dot_product = np.clip(np.dot(upper_arm_dir, forearm_dir), -1.0, 1.0)
        elbow_angle = np.arccos(dot_product)  # 0 = straight, pi = fully bent back
        
        # Convert to robot convention: 0 = straight, positive = bent
        # When arm is straight, angle is ~pi (180 deg) between vectors pointing same way
        # We want 0 for straight arm
        # elbow_pitch = np.pi - elbow_angle
        elbow_pitch = - elbow_angle
        
        # Elbow yaw: twist of forearm around its axis
        # This is harder to determine from just positions, approximate with wrist orientation
        # For now, derive from the perpendicular component
        elbow_yaw = 0.0  # Would need wrist orientation keypoints for accurate computation
        
        # ===== WRIST ANGLES =====
        # Without hand keypoints, we can't accurately determine wrist angles
        # Keep at zero
        wrist_pitch = 0.0
        wrist_yaw = 0.0
        
        # Store results
        result['shoulder_pitch'] = shoulder_pitch
        result['shoulder_roll'] = shoulder_roll
        result['shoulder_yaw'] = shoulder_yaw * 0.5  # Reduce yaw sensitivity
        result['elbow_pitch'] = elbow_pitch
        result['elbow_yaw'] = elbow_yaw
        result['wrist_pitch'] = wrist_pitch
        result['wrist_yaw'] = wrist_yaw
        
        return result
    
    def _filter_joint_angles(self, raw_angles: dict) -> dict:
        """
        Apply smoothing filter to joint angles to reduce noise and discrete jumps.
        
        Uses exponential smoothing with jump rejection:
        - If the new value jumps too far from the previous, gradually move towards it
        - Otherwise, apply exponential smoothing: filtered = alpha * new + (1-alpha) * old
        
        Args:
            raw_angles: Dictionary of raw joint angles from retargeting
            
        Returns:
            Dictionary of filtered joint angles
        """
        filtered = {}
        
        # Joints to apply filtering to (arm joints only)
        arm_joints = [
            "SHOULDER_PITCH_R", "SHOULDER_ROLL_R", "SHOULDER_YAW_R",
            "ELBOW_PITCH_R", "ELBOW_YAW_R", "WRIST_PITCH_R", "WRIST_YAW_R",
            "SHOULDER_PITCH_L", "SHOULDER_ROLL_L", "SHOULDER_YAW_L",
            "ELBOW_PITCH_L", "ELBOW_YAW_L", "WRIST_PITCH_L", "WRIST_YAW_L",
        ]
        
        # Joints that need extra smoothing (more susceptible to noise)
        noisy_joints = ["SHOULDER_PITCH_R", "SHOULDER_PITCH_L"]
        
        for joint_name, raw_value in raw_angles.items():
            if joint_name not in arm_joints:
                # Pass through non-arm joints without filtering
                filtered[joint_name] = raw_value
                continue
            
            # NaN/inf protection - reset to zero if invalid
            if not np.isfinite(raw_value):
                raw_value = 0.0
            
            # Initialize filter state if needed
            if joint_name not in self.filtered_joints:
                self.filtered_joints[joint_name] = raw_value
                filtered[joint_name] = raw_value
                continue
            
            prev_value = self.filtered_joints[joint_name]
            
            # NaN/inf protection for previous value
            if not np.isfinite(prev_value):
                prev_value = 0.0
                self.filtered_joints[joint_name] = 0.0
            
            # Use stronger smoothing for noisy joints
            if joint_name in noisy_joints:
                alpha = 0.2  # Slower response, more smoothing
                jump_threshold = 0.8  # Tighter jump detection
                recovery_rate = 0.05  # Slower recovery
            else:
                alpha = self.filter_alpha
                jump_threshold = self.max_jump_threshold
                recovery_rate = self.jump_recovery_rate
            
            # Check for discrete jump
            jump = abs(raw_value - prev_value)
            
            if jump > jump_threshold:
                # Large jump detected - gradually move towards new value (don't get stuck)
                filtered[joint_name] = (recovery_rate * raw_value + 
                                       (1 - recovery_rate) * prev_value)
            else:
                # Apply exponential smoothing
                filtered[joint_name] = (alpha * raw_value + 
                                       (1 - alpha) * prev_value)
            
            # Final NaN check
            if not np.isfinite(filtered[joint_name]):
                filtered[joint_name] = 0.0
            
            # Update filter state
            self.filtered_joints[joint_name] = filtered[joint_name]
        
        return filtered
    
    def _retarget_to_robot(self, keypoints: np.ndarray, confidences: np.ndarray) -> dict:
        """
        Retarget human body keypoints to robot joint angles using rotation transforms.
        
        Uses 3D position vectors to compute rotation matrices and extract joint angles
        for shoulder, elbow, and wrist joints.
        
        Args:
            keypoints: 38x3 array of body keypoints (ZED BODY_38 format)
            confidences: 38 confidence values for each keypoint
            
        Returns:
            Dictionary of joint angles for the robot
            
        ZED BODY_38 keypoint indices:
            LEFT_SHOULDER = 12, RIGHT_SHOULDER = 13
            LEFT_ELBOW = 14, RIGHT_ELBOW = 15  
            LEFT_WRIST = 16, RIGHT_WRIST = 17
            LEFT_CLAVICLE = 10, RIGHT_CLAVICLE = 11
        """
        # Default pose (zero angles)
        joint_angles = {
            "HIP_YAW_R": 0.0, "HIP_ROLL_R": 0.0, "HIP_PITCH_R": 0.0,
            "KNEE_PITCH_R": 0.0, "ANKLE_PITCH_R": 0.0, "ANKLE_ROLL_R": 0.0,
            "HIP_YAW_L": 0.0, "HIP_ROLL_L": 0.0, "HIP_PITCH_L": 0.0,
            "KNEE_PITCH_L": 0.0, "ANKLE_PITCH_L": 0.0, "ANKLE_ROLL_L": 0.0,
            "SHOULDER_YAW_R": 0.0, "SHOULDER_ROLL_R": 0.0,
            "SHOULDER_PITCH_R": 0.0,
            "ELBOW_YAW_R": 0.0, "ELBOW_PITCH_R": 0.0,
            "WRIST_YAW_R": 0.0, "WRIST_PITCH_R": 0.0,
            "SHOULDER_YAW_L": 0.0, "SHOULDER_ROLL_L": 0.0,
            "SHOULDER_PITCH_L": 0.0,
            "ELBOW_YAW_L": 0.0, "ELBOW_PITCH_L": 0.0,
            "WRIST_YAW_L": 0.0, "WRIST_PITCH_L": 0.0,
            "HEAD_YAW": 0.0, "HEAD_PITCH": 0.0,
        }
        
        # Check if we have valid keypoints
        if keypoints.shape[0] < 18:
            return joint_angles
        
        # Minimum confidence threshold
        MIN_CONF = 30
        
        # ===== RIGHT ARM (indices: shoulder=13, elbow=15, wrist=17) =====
        if (confidences[13] > MIN_CONF and confidences[15] > MIN_CONF and 
            confidences[17] > MIN_CONF):
            
            right_shoulder = keypoints[13]
            right_elbow = keypoints[15]
            right_wrist = keypoints[17]
            
            right_arm = self._compute_arm_rotation(
                right_shoulder, right_elbow, right_wrist, is_left=False
            )
            
            # Save as last valid arm angles
            self.last_valid_right_arm = right_arm
            
            # Map to robot joints with appropriate limits
            # joint_angles["SHOULDER_PITCH_R"] = np.clip(right_arm['shoulder_pitch'], -1.8, 1.8)
            # joint_angles["SHOULDER_ROLL_R"] = np.clip(right_arm['shoulder_roll'], -1.5, 0.3)
            # joint_angles["SHOULDER_YAW_R"] = np.clip(right_arm['shoulder_yaw'], -1.5, 1.5)
            # joint_angles["ELBOW_PITCH_R"] = np.clip(right_arm['elbow_pitch'], -2.5, 0.0)
            # joint_angles["ELBOW_YAW_R"] = np.clip(right_arm['elbow_yaw'], -1.5, 1.5)
            # joint_angles["WRIST_PITCH_R"] = np.clip(right_arm['wrist_pitch'], -1.0, 1.0)
            # joint_angles["WRIST_YAW_R"] = np.clip(right_arm['wrist_yaw'], -1.0, 1.0)

            joint_angles["SHOULDER_PITCH_R"] = np.clip(right_arm['shoulder_pitch'], -3.14, 3.14)
            joint_angles["SHOULDER_ROLL_R"] = np.clip(right_arm['shoulder_roll'] - np.pi/2, -1.78, 1.78)  # -90 deg offset
            joint_angles["SHOULDER_YAW_R"] = np.clip(right_arm['shoulder_yaw'], -0, 0)
            joint_angles["ELBOW_PITCH_R"] = np.clip(right_arm['elbow_pitch'], -2.5, 0.0)  # Limited to -2.0 to prevent geometry issues
            joint_angles["ELBOW_YAW_R"] = np.clip(right_arm['elbow_yaw'], -0, 0)
            joint_angles["WRIST_PITCH_R"] = np.clip(right_arm['wrist_pitch'], -0, 0)
            joint_angles["WRIST_YAW_R"] = np.clip(right_arm['wrist_yaw'], -0, 0)
        elif self.last_valid_right_arm is not None:
            # Use last valid arm angles when tracking is lost
            right_arm = self.last_valid_right_arm
            joint_angles["SHOULDER_PITCH_R"] = np.clip(right_arm['shoulder_pitch'], -3.14, 3.14)
            joint_angles["SHOULDER_ROLL_R"] = np.clip(right_arm['shoulder_roll'] - np.pi/2, -1.78, 1.78)
            joint_angles["SHOULDER_YAW_R"] = np.clip(right_arm['shoulder_yaw'], -0, 0)
            joint_angles["ELBOW_PITCH_R"] = np.clip(right_arm['elbow_pitch'], -2.5, 0.0)
            joint_angles["ELBOW_YAW_R"] = np.clip(right_arm['elbow_yaw'], -0, 0)
            joint_angles["WRIST_PITCH_R"] = np.clip(right_arm['wrist_pitch'], -0, 0)
            joint_angles["WRIST_YAW_R"] = np.clip(right_arm['wrist_yaw'], -0, 0)
        
        # ===== LEFT ARM (indices: shoulder=12, elbow=14, wrist=16) =====
        if (confidences[12] > MIN_CONF and confidences[14] > MIN_CONF and 
            confidences[16] > MIN_CONF):
            
            left_shoulder = keypoints[12]
            left_elbow = keypoints[14]
            left_wrist = keypoints[16]
            
            left_arm = self._compute_arm_rotation(
                left_shoulder, left_elbow, left_wrist, is_left=True
            )
            
            # Save as last valid arm angles
            self.last_valid_left_arm = left_arm
            
            # Map to robot joints with appropriate limits
            # joint_angles["SHOULDER_PITCH_L"] = np.clip(left_arm['shoulder_pitch'], -1.8, 1.8)
            # joint_angles["SHOULDER_ROLL_L"] = np.clip(left_arm['shoulder_roll'], -0.3, 1.5)
            # joint_angles["SHOULDER_YAW_L"] = np.clip(left_arm['shoulder_yaw'], -1.5, 1.5)
            # joint_angles["ELBOW_PITCH_L"] = np.clip(left_arm['elbow_pitch'], -2.5, 0.0)
            # joint_angles["ELBOW_YAW_L"] = np.clip(left_arm['elbow_yaw'], -1.5, 1.5)
            # joint_angles["WRIST_PITCH_L"] = np.clip(left_arm['wrist_pitch'], -1.0, 1.0)
            # joint_angles["WRIST_YAW_L"] = np.clip(left_arm['wrist_yaw'], -1.0, 1.0)

            joint_angles["SHOULDER_PITCH_L"] = np.clip(left_arm['shoulder_pitch'], -0, 0)
            joint_angles["SHOULDER_ROLL_L"] = np.clip(-left_arm['shoulder_roll'] + np.pi/2, -1.78, 1.78)  # inverted + 90 deg offset
            joint_angles["SHOULDER_YAW_L"] = np.clip(left_arm['shoulder_yaw'], -0, 0)
            joint_angles["ELBOW_PITCH_L"] = np.clip(left_arm['elbow_pitch'], -2.5, 0.0)  # Limited to -2.0 to prevent geometry issues
            joint_angles["ELBOW_YAW_L"] = np.clip(left_arm['elbow_yaw'], -0, 0)
            joint_angles["WRIST_PITCH_L"] = np.clip(left_arm['wrist_pitch'], -0, 0)
            joint_angles["WRIST_YAW_L"] = np.clip(left_arm['wrist_yaw'], -0, 0)
        elif self.last_valid_left_arm is not None:
            # Use last valid arm angles when tracking is lost
            left_arm = self.last_valid_left_arm
            joint_angles["SHOULDER_PITCH_L"] = np.clip(left_arm['shoulder_pitch'], -0, 0)
            joint_angles["SHOULDER_ROLL_L"] = np.clip(-left_arm['shoulder_roll'] + np.pi/2, -1.78, 1.78)
            joint_angles["SHOULDER_YAW_L"] = np.clip(left_arm['shoulder_yaw'], -0, 0)
            joint_angles["ELBOW_PITCH_L"] = np.clip(left_arm['elbow_pitch'], -2.5, 0.0)
            joint_angles["ELBOW_YAW_L"] = np.clip(left_arm['elbow_yaw'], -0, 0)
            joint_angles["WRIST_PITCH_L"] = np.clip(left_arm['wrist_pitch'], -0, 0)
            joint_angles["WRIST_YAW_L"] = np.clip(left_arm['wrist_yaw'], -0, 0)
        
        return joint_angles
    
    def _visualization_loop(self):
        """Main visualization loop."""
        base_height = 1.5
        start_time = time.time()
        last_perf_report = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Poll tracking data (non-blocking) - drain queue to get latest frame
                try:
                    while not self.tracking_queue.empty():
                        self.latest_tracking = self.tracking_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Calculate actual queue latency (time since frame was captured)
                queue_latency_ms = 0.0
                if self.latest_tracking is not None:
                    queue_latency_ms = (time.time() - self.latest_tracking.timestamp) * 1000
                
                # Use tracking data if available, otherwise use sinwave animations
                if (self.latest_tracking is not None and 
                    self.latest_tracking.num_bodies > 0 and 
                    len(self.latest_tracking.keypoints) > 0):
                    
                    # Measure end-to-end latency
                    latency = time.time() - self.latest_tracking.timestamp
                    self.latency_samples.append(latency * 1000)  # Convert to ms
                    if len(self.latency_samples) > 100:
                        self.latency_samples.pop(0)
                    
                    # Retarget body to robot
                    retarget_start = time.perf_counter()
                    keypoints = self.latest_tracking.keypoints[0]
                    confidences = self.latest_tracking.confidences[0]
                    raw_pose = self._retarget_to_robot(keypoints, confidences)
                    
                    # Apply smoothing filter
                    animated_pose = self._filter_joint_angles(raw_pose)
                    
                    retarget_time = (time.perf_counter() - retarget_start) * 1000000  # microseconds
                    self.retarget_times.append(retarget_time)
                    if len(self.retarget_times) > 100:
                        self.retarget_times.pop(0)
                    self.retarget_monitor.tick()
                    
                    # Print commanded joint angles and latency for debugging
                    print(f"\r[Latency:{queue_latency_ms:4.0f}ms] "
                          f"R: pitch={animated_pose['SHOULDER_PITCH_R']:+.2f} roll={animated_pose['SHOULDER_ROLL_R']:+.2f} "
                          f"elbow={animated_pose['ELBOW_PITCH_R']:+.2f} | "
                          f"L: pitch={animated_pose['SHOULDER_PITCH_L']:+.2f} roll={animated_pose['SHOULDER_ROLL_L']:+.2f} "
                          f"elbow={animated_pose['ELBOW_PITCH_L']:+.2f}    ", end="")
                else:
                    # Fallback to sinwave animations when no body detected
                    t = time.time() - start_time
                    
                    # Create sinusoidal arm motions
                    shoulder_pitch_r = 0.5 * np.sin(2 * np.pi * 0.5 * t)
                    elbow_pitch_r = 0.3 + 0.5 * np.sin(2 * np.pi * 0.5 * t + np.pi/4)
                    
                    shoulder_pitch_l = 0.6 * np.sin(2 * np.pi * 0.3 * t)
                    shoulder_roll_l = 0.6 * np.cos(2 * np.pi * 0.3 * t)
                    
                    wrist_yaw_r = 0.4 * np.sin(2 * np.pi * 0.7 * t)
                    wrist_yaw_l = 0.4 * np.sin(2 * np.pi * 0.7 * t + np.pi)
                    
                    head_yaw = 0.5 * np.sin(2 * np.pi * 0.2 * t)
                    head_pitch = 0.2 * np.cos(2 * np.pi * 0.15 * t)
                    
                    # Animated pose
                    animated_pose = {
                        "HIP_YAW_R": 0.0, "HIP_ROLL_R": 0.0, "HIP_PITCH_R": 0.0,
                        "KNEE_PITCH_R": 0.0, "ANKLE_PITCH_R": 0.0, "ANKLE_ROLL_R": 0.0,
                        "HIP_YAW_L": 0.0, "HIP_ROLL_L": 0.0, "HIP_PITCH_L": 0.0,
                        "KNEE_PITCH_L": 0.0, "ANKLE_PITCH_L": 0.0, "ANKLE_ROLL_L": 0.0,
                        "SHOULDER_YAW_R": 0.0, "SHOULDER_ROLL_R": 0.3,
                        "SHOULDER_PITCH_R": shoulder_pitch_r,
                        "ELBOW_YAW_R": 0.0, "ELBOW_PITCH_R": elbow_pitch_r,
                        "WRIST_YAW_R": wrist_yaw_r, "WRIST_PITCH_R": 0.0,
                        "SHOULDER_YAW_L": 0.0, "SHOULDER_ROLL_L": -0.3,
                        "SHOULDER_PITCH_L": shoulder_pitch_l,
                        "ELBOW_YAW_L": 0.0, "ELBOW_PITCH_L": 0.3,
                        "WRIST_YAW_L": wrist_yaw_l, "WRIST_PITCH_L": 0.0,
                        "HEAD_YAW": head_yaw, "HEAD_PITCH": head_pitch,
                    }
                
                # Update robot
                base_pos = np.array([0.0, 0.0, base_height])
                base_rot = np.array([0.0, 0.0, 0.0, 1.0])
                
                # Measure robot update time
                robot_update_start = time.perf_counter()
                self.sim.set_base_pose(base_pos, base_rot)
                self.sim.set_joint_positions(animated_pose)
                robot_update_time = (time.perf_counter() - robot_update_start) * 1000000  # microseconds
                self.robot_update_times.append(robot_update_time)
                if len(self.robot_update_times) > 100:
                    self.robot_update_times.pop(0)
                
                # Measure MuJoCo forward kinematics time
                import mujoco
                mujoco_start = time.perf_counter()
                mujoco.mj_forward(self.sim.model, self.sim.data)
                mujoco_time = (time.perf_counter() - mujoco_start) * 1000000  # microseconds
                self.mujoco_forward_times.append(mujoco_time)
                if len(self.mujoco_forward_times) > 100:
                    self.mujoco_forward_times.pop(0)
                
                # Performance monitoring
                self.viz_monitor.tick()
                
                # Print comprehensive performance report every 5 seconds
                if time.time() - last_perf_report >= 5.0:
                    self._print_performance_report()
                    last_perf_report = time.time()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[System] Shutting down...")
            self._print_performance_report()  # Final report
            self.stop()
    
    def _print_performance_report(self):
        """Print comprehensive performance report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 70)
        
        # ZED Tracker stats
        tracker_stats = self.tracker.perf_monitor.get_stats()
        print(f"ZED Body Tracking:")
        print(f"  FPS: {tracker_stats['fps']:.1f}")
        print(f"  Frame Time: {tracker_stats['avg_ms']:.2f}ms (min: {tracker_stats['min_ms']:.2f}ms, max: {tracker_stats['max_ms']:.2f}ms)")
        print(f"  Total Frames: {tracker_stats['total_frames']}")
        
        # Detailed computation times
        print(f"\nDetailed Computation Times:")
        
        if len(self.retarget_times) > 0:
            avg_retarget = np.mean(self.retarget_times)
            min_retarget = np.min(self.retarget_times)
            max_retarget = np.max(self.retarget_times)
            print(f"  Retargeting: {avg_retarget:.1f}µs ({avg_retarget/1000:.3f}ms) | "
                  f"min: {min_retarget:.1f}µs, max: {max_retarget:.1f}µs")
        
        if len(self.robot_update_times) > 0:
            avg_robot = np.mean(self.robot_update_times)
            min_robot = np.min(self.robot_update_times)
            max_robot = np.max(self.robot_update_times)
            print(f"  Robot Update: {avg_robot:.1f}µs ({avg_robot/1000:.3f}ms) | "
                  f"min: {min_robot:.1f}µs, max: {max_robot:.1f}µs")
        
        if len(self.mujoco_forward_times) > 0:
            avg_mujoco = np.mean(self.mujoco_forward_times)
            min_mujoco = np.min(self.mujoco_forward_times)
            max_mujoco = np.max(self.mujoco_forward_times)
            print(f"  MuJoCo Forward: {avg_mujoco:.1f}µs ({avg_mujoco/1000:.3f}ms) | "
                  f"min: {min_mujoco:.1f}µs, max: {max_mujoco:.1f}µs")
        
        # Retargeting stats (frame rate)
        retarget_stats = self.retarget_monitor.get_stats()
        print(f"\nRetargeting (Frame Rate):")
        print(f"  FPS: {retarget_stats['fps']:.1f}")
        print(f"  Total Frames: {retarget_stats['total_frames']}")
        
        # Visualization stats
        viz_stats = self.viz_monitor.get_stats()
        print(f"\nVisualization Loop (Frame Rate):")
        print(f"  FPS: {viz_stats['fps']:.1f}")
        print(f"  Loop Time: {viz_stats['avg_ms']:.2f}ms (min: {viz_stats['min_ms']:.2f}ms, max: {viz_stats['max_ms']:.2f}ms)")
        print(f"  Total Frames: {viz_stats['total_frames']}")
        
        # Total computation breakdown
        if len(self.retarget_times) > 0 and len(self.robot_update_times) > 0 and len(self.mujoco_forward_times) > 0:
            total_compute = (np.mean(self.retarget_times) + 
                           np.mean(self.robot_update_times) + 
                           np.mean(self.mujoco_forward_times)) / 1000  # Convert to ms
            print(f"\nTotal Computation Time per Frame: {total_compute:.3f}ms")
            print(f"  Breakdown: Retarget {np.mean(self.retarget_times)/1000:.3f}ms + "
                  f"Update {np.mean(self.robot_update_times)/1000:.3f}ms + "
                  f"MuJoCo {np.mean(self.mujoco_forward_times)/1000:.3f}ms")
        
        # End-to-end latency
        if len(self.latency_samples) > 0:
            avg_latency = np.mean(self.latency_samples)
            min_latency = np.min(self.latency_samples)
            max_latency = np.max(self.latency_samples)
            print(f"\nEnd-to-End Latency (Camera → Robot):")
            print(f"  Average: {avg_latency:.2f}ms")
            print(f"  Min: {min_latency:.2f}ms")
            print(f"  Max: {max_latency:.2f}ms")
        
        # Queue status
        queue_size = self.tracking_queue.qsize()
        print(f"\nQueue Status:")
        print(f"  Current Size: {queue_size}/10")
        print(f"  Dropped Frames: {'Yes (queue full)' if queue_size == 10 else 'No'}")
        
        print("=" * 70 + "\n")
    
    def stop(self):
        """Stop the integrated system."""
        self.tracker.stop()
        self.sim.close()
        print("[System] Shutdown complete")


def main():
    """Run the integrated tracking and visualization system."""
    print("=" * 60)
    print("Integrated ZED Body Tracking + Robot Visualization")
    print("=" * 60)
    
    # Check for sudo permissions
    if ZED_AVAILABLE and os.geteuid() != 0:
        print("\n⚠️  WARNING: Not running with sudo permissions!")
        print("   ZED motion sensors require root access for body tracking.")
        print("   Please run with:")
        print(f"   sudo {sys.executable} {sys.argv[0]}")
        print("\n   Continuing in dummy mode (no real tracking)...")
        print("=" * 60)
        time.sleep(2)
    
    print("\nPipeline:")
    print("  1. ZED camera captures body tracking data")
    print("  2. Tracking data sent to visualization thread via queue")
    print("  3. Visualizer receives data (retargeting not yet implemented)")
    print("  4. Robot displays sinwave animations for now")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()
    
    system = IntegratedVisualization()
    system.start()


if __name__ == "__main__":
    main()
