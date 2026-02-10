"""
Body Tracking Node (30 Hz)

ZED body tracking for arm positions.
Runs in separate thread, publishes to shared state.
"""

import numpy as np
import time
import threading
from typing import Optional

from ..shared_state import SharedState, ArmTrackingData
from ..config import TrackingConfig, PipelineConfig


class PositionFilter:
    """Low-pass filter with jump rejection for position data."""
    
    def __init__(self, alpha: float = 0.3, jump_threshold: float = 0.15):
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self.last_valid: Optional[np.ndarray] = None
        self.filtered: Optional[np.ndarray] = None
        
    def update(self, value: np.ndarray, valid: bool = True) -> np.ndarray:
        if not valid or np.any(np.isnan(value)):
            if self.last_valid is not None:
                return self.last_valid.copy()
            # No previous valid data yet — return safe zeros instead of NaN
            return np.zeros_like(value)
        
        if self.filtered is None:
            self.filtered = value.copy()
            self.last_valid = value.copy()
            return self.filtered.copy()
        
        jump = np.linalg.norm(value - self.filtered)
        if jump > self.jump_threshold:
            self.filtered = 0.05 * value + 0.95 * self.filtered
            self.last_valid = self.filtered.copy()
            return self.filtered.copy()
        
        self.filtered = self.alpha * value + (1.0 - self.alpha) * self.filtered
        self.last_valid = self.filtered.copy()
        return self.filtered.copy()
    
    def reset(self):
        self.last_valid = None
        self.filtered = None


class BodyTrackingNode:
    """
    Body tracking node using ZED camera.
    
    Extracts arm keypoints and publishes to shared state at ~30Hz.
    """
    
    def __init__(self, config: PipelineConfig, shared_state: SharedState):
        self.config = config
        self.track_config = config.tracking
        self.shared = shared_state
        
        # ZED imports
        self.zed_available = False
        self.sl = None
        self.cv2 = None
        self.cv_viewer = None
        
        try:
            import pyzed.sl as sl
            import cv2
            import cv_viewer.tracking_viewer as cv_viewer
            self.sl = sl
            self.cv2 = cv2
            self.cv_viewer = cv_viewer
            self.zed_available = True
        except ImportError:
            print("[Tracking] Warning: ZED SDK not available, using dummy mode")
        
        # Position filters
        self.filters = {
            'left_shoulder': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
            'left_elbow': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
            'left_wrist': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
            'right_shoulder': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
            'right_elbow': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
            'right_wrist': PositionFilter(
                alpha=self.track_config.filter_alpha,
                jump_threshold=self.track_config.jump_threshold
            ),
        }
        
        # Thread state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.zed = None
        
        # Timing
        self.last_stats_time = time.time()
        self.frame_count = 0
        
    def start(self):
        """Start tracking thread."""
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        print("[Tracking] Started body tracking node")
        
    def stop(self):
        """Stop tracking thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.zed:
            self.zed.disable_body_tracking()
            self.zed.disable_positional_tracking()
            self.zed.close()
        print("[Tracking] Stopped")
    
    def _tracking_loop(self):
        """Main tracking loop."""
        if not self.zed_available:
            self._dummy_tracking_loop()
            return
        
        sl = self.sl
        
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
            print(f"[Tracking] Failed to open camera: {status}")
            self._dummy_tracking_loop()
            return
        
        # Enable positional tracking
        tracking_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(tracking_params)
        
        # Enable body tracking
        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.body_format = sl.BODY_FORMAT.BODY_38
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        
        status = self.zed.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[Tracking] Failed to enable body tracking: {status}")
            self.zed.close()
            self._dummy_tracking_loop()
            return
        
        print("[Tracking] ZED body tracking enabled")
        
        body_runtime = sl.BodyTrackingRuntimeParameters()
        body_runtime.detection_confidence_threshold = int(self.track_config.min_confidence)
        
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
        
        while self.running and not self.shared.is_shutdown_requested():
            loop_start = time.time()
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Get camera image
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                
                # Get bodies
                self.zed.retrieve_bodies(bodies, body_runtime)
                
                # Display with skeleton overlay
                image_ocv = image.get_data()
                self.cv_viewer.render_2D(
                    image_ocv, image_scale, bodies.body_list, True, sl.BODY_FORMAT.BODY_38
                )
                self.cv2.imshow("ZED Body Tracking", image_ocv)
                
                # Extract arm data
                arm_data = self._extract_arm_data(bodies)
                
                # Publish to shared state
                self.shared.set_tracking_data(arm_data)
                
                # Update timing stats
                self.frame_count += 1
                if time.time() - self.last_stats_time >= 2.0:
                    hz = self.frame_count / (time.time() - self.last_stats_time)
                    self.shared.update_timing('tracking', hz)
                    # Debug: print arm data validity and confidence
                    print(f"[Tracking] valid={arm_data.valid}, L_conf={arm_data.left_confidence:.1f}, R_conf={arm_data.right_confidence:.1f}")
                    self.frame_count = 0
                    self.last_stats_time = time.time()
                
                # Handle CV window events
                key = self.cv2.waitKey(1)
                if key == ord('q'):
                    self.shared.request_shutdown()
                # record loop duration
                loop_dur = time.time() - loop_start
                # publish last loop duration (seconds)
                try:
                    self.shared.set_loop_duration('tracking', loop_dur)
                except Exception:
                    pass

        image.free(sl.MEM.CPU)
        self.cv2.destroyAllWindows()
    
    def _extract_arm_data(self, bodies) -> ArmTrackingData:
        """Extract arm tracking data from ZED bodies."""
        sl = self.sl
        
        if len(bodies.body_list) == 0:
            return ArmTrackingData(timestamp=time.time(), valid=False)
        
        body = bodies.body_list[0]
        kp = body.keypoint
        conf = body.keypoint_confidence
        
        # BODY_38 indices: NECK=3, L_ELBOW=14, L_WRIST=34
        # R_ELBOW=15, R_WRIST=35
        # See https://www.stereolabs.com/docs/body-tracking for full skeleton
        
        # Helper: safely extract keypoint position (NaN → zeros)
        def _safe_kp(idx):
            p = np.array([kp[idx][0], kp[idx][1], kp[idx][2]], dtype=np.float64)
            if np.any(np.isnan(p)):
                return np.zeros(3, dtype=np.float64), False
            return p, True
        
        # Body COM (NECK) with offset
        neck_pos, neck_ok = _safe_kp(3)
        body_com = neck_pos + np.array([0.0, -0.15, 0.0]) if neck_ok else np.zeros(3, dtype=np.float64)
        
        if not neck_ok:
            # Can't compute body frame without neck
            return ArmTrackingData(timestamp=time.time(), valid=False)
        
        # Get world positions (safe extraction)
        left_elbow_world, le_ok = _safe_kp(14)
        left_wrist_world, lw_ok = _safe_kp(34)
        right_elbow_world, re_ok = _safe_kp(15)
        right_wrist_world, rw_ok = _safe_kp(35)
        
        # Convert to local body frame
        left_elbow_local = left_elbow_world - body_com
        left_wrist_local = left_wrist_world - body_com
        right_elbow_local = right_elbow_world - body_com
        right_wrist_local = right_wrist_world - body_com
        
        # Check confidences AND position validity
        min_conf = self.track_config.min_confidence
        left_valid = le_ok and lw_ok and min(conf[3], conf[14], conf[34]) > min_conf
        right_valid = re_ok and rw_ok and min(conf[3], conf[15], conf[35]) > min_conf
        
        # Apply filters (update all filters even if not all are used)
        # Shoulder placeholders (not used in BODY_38 tracking)
        left_shoulder_filtered = self.filters['left_shoulder'].update(np.zeros(3, dtype=np.float64), False)
        right_shoulder_filtered = self.filters['right_shoulder'].update(np.zeros(3, dtype=np.float64), False)
        # Actual tracking
        left_elbow_filtered = self.filters['left_elbow'].update(left_elbow_local, left_valid)
        left_wrist_filtered = self.filters['left_wrist'].update(left_wrist_local, left_valid)
        right_elbow_filtered = self.filters['right_elbow'].update(right_elbow_local, right_valid)
        right_wrist_filtered = self.filters['right_wrist'].update(right_wrist_local, right_valid)
        
        # Safe confidence extraction (NaN confidence → 0)
        def _safe_conf(idx):
            c = float(conf[idx])
            return c if not np.isnan(c) else 0.0
        
        return ArmTrackingData(
            timestamp=time.time(),
            valid=left_valid or right_valid,
            body_com=body_com,
            left_shoulder=left_shoulder_filtered,
            left_elbow=left_elbow_filtered,
            left_wrist=left_wrist_filtered,
            left_confidence=min(_safe_conf(14), _safe_conf(34)),
            right_shoulder=right_shoulder_filtered,
            right_elbow=right_elbow_filtered,
            right_wrist=right_wrist_filtered,
            right_confidence=min(_safe_conf(15), _safe_conf(35)),
        )
    
    def _dummy_tracking_loop(self):
        """Dummy tracking for testing without camera."""
        print("[Tracking] Running dummy tracking mode")
        t_start = time.time()
        
        while self.running and not self.shared.is_shutdown_requested():
            loop_start = time.time()
            t = time.time() - t_start
            
            # Simulate arm movement in standard XYZ frame (X forward, Y left, Z up)
            arm_data = ArmTrackingData(
                timestamp=time.time(),
                valid=True,
                body_com=np.array([2.0, 0.0, 1.0], dtype=np.float64),  # Transformed from [0,1,2]
                left_shoulder=np.array([0.0, 0.0, 0.2], dtype=np.float64),  # Transformed from [0,0.2,0]
                left_elbow=np.array([
                    -0.35 - 0.1*np.sin(t*0.7),  # Y -> Z
                    -0.15 + 0.15*np.sin(t),  # -X -> Y
                    0.1*np.cos(t)  # Z -> X
                ], dtype=np.float64),
                left_wrist=np.array([
                    -0.40 - 0.15*np.sin(t*0.7),  # Y -> Z
                    -0.25 + 0.25*np.sin(t),  # -X -> Y
                    0.15*np.cos(t)  # Z -> X
                ], dtype=np.float64),
                left_confidence=90.0,
                right_shoulder=np.array([0.0, 0.0, -0.2], dtype=np.float64),  # Transformed from [0,-0.2,0]
                right_elbow=np.array([
                    0.35 + 0.1*np.sin(t*0.7),  # -Y -> -Z
                    0.15 - 0.15*np.sin(t),  # -X -> -Y
                    0.1*np.cos(t)  # Z -> X
                ], dtype=np.float64),
                right_wrist=np.array([
                    0.40 + 0.15*np.sin(t*0.7),  # -Y -> -Z
                    0.25 - 0.25*np.sin(t),  # -X -> -Y
                    0.15*np.cos(t)  # Z -> X
                ], dtype=np.float64),
                right_confidence=90.0,
            )
            
            self.shared.set_tracking_data(arm_data)
            
            # Update timing stats
            self.frame_count += 1
            if time.time() - self.last_stats_time >= 2.0:
                hz = self.frame_count / (time.time() - self.last_stats_time)
                self.shared.update_timing('tracking', hz)
                self.frame_count = 0
                self.last_stats_time = time.time()
            
            # Sleep to ~30Hz
            time.sleep(self.track_config.tracking_dt)
            # record loop duration
            loop_dur = time.time() - loop_start
            try:
                # use 'tracking_dummy' to differentiate from real camera loop
                self.shared.set_loop_duration('tracking_dummy', loop_dur)
            except Exception:
                pass
