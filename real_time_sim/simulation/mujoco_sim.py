"""
MuJoCo Physics Simulation.

Real-time physics simulation with torque control interface.
Runs at 1kHz (configurable) with optional visualization.
"""

import numpy as np
import time
import threading
from typing import Optional, Callable, Dict

import mujoco
import mujoco.viewer

from ..shared_state import SharedState, RobotFeedback
from ..config import SimConfig, PipelineConfig


class MuJoCoSimulation:
    """
    MuJoCo-based physics simulation with real-time execution.
    
    Features:
    - 1kHz physics simulation
    - Torque control interface
    - State feedback (joint positions, velocities, COM)
    - Optional visualization with marker rendering
    """
    
    def __init__(self, config: PipelineConfig, shared_state: SharedState, model_path: str = None):
        self.config = config
        self.sim_config = config.sim
        self.shared = shared_state
        
        # Load model
        if model_path is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, self.sim_config.model_path)
        
        print(f"[Simulation] Loading model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation timestep
        self.model.opt.timestep = self.sim_config.sim_dt
        
        # Viewer (optional)
        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.render_enabled = True
        
        # Markers for visualization
        self.markers: Dict[str, np.ndarray] = {
            'hand_l_des': np.zeros(3),
            'hand_r_des': np.zeros(3),
            'elbow_l_des': np.zeros(3),
            'elbow_r_des': np.zeros(3),
            'hand_l_act': np.zeros(3),
            'hand_r_act': np.zeros(3),
            'elbow_l_act': np.zeros(3),
            'elbow_r_act': np.zeros(3),
            'hand_l_orient_mat': np.eye(3),
            'hand_r_orient_mat': np.eye(3),
        }
        self.markers_lock = threading.Lock()
        
        # Timing
        self.sim_time = 0.0
        self.wall_time_start = 0.0
        self.step_count = 0
        
        # Running state
        self.running = False
        self.paused = False
        
        # Base height for the welded base (no freejoint).
        # The base body is welded to the world; we set its position via model.body_pos.
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'BASE_LINK')
        self.set_base_height(self.sim_config.base_height)
        
        # Build joint mapping
        self._build_joint_map()
        
        # Initialize robot pose
        self._initialize_robot()
        
        print(f"[Simulation] Initialized with dt={self.sim_config.sim_dt}s ({1/self.sim_config.sim_dt:.0f}Hz)")
    
    def set_base_height(self, height: float):
        """Set the height of the welded base body."""
        self.model.body_pos[self._base_body_id] = [0.0, 0.0, height]
        # Forward kinematics so xpos etc. are updated
        mujoco.mj_forward(self.model, self.data)
    
    def _build_joint_map(self):
        """Build mapping from joint names to indices."""
        self.joint_map = {}
        self.actuator_map = {}
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_map[name] = i
                
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_map[name] = i
                
        print(f"[Simulation] {len(self.joint_map)} joints, {len(self.actuator_map)} actuators")
    
    def _initialize_robot(self):
        """Initialize robot to default pose.
        
        With the freejoint removed, qpos is just the 28 hinge joints
        and qvel is also 28. No base DOFs in qpos/qvel.
        """
        # qpos layout (welded base): [joints(28)] — no base DOFs
        default_joints = np.array([
            # Right leg
            0.0, 0.0, -0.1, 0.2, -0.1, 0.0,
            # Left leg
            0.0, 0.0, -0.1, 0.2, -0.1, 0.0,
            # Right arm
            0.0, 0.8, -0.5, 0.78, 0.1, 0.4, 0.0,
            # Left arm
            0.0, -0.8, -0.5, -0.78, 0.1, -0.4, 0.0,
            # Head
            0.0, 0.0
        ], dtype=np.float64)
        
        self.data.qpos[0:28] = default_joints
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
    def set_torques(self, torques: np.ndarray):
        """
        Set joint torques for control.
        
        Args:
            torques: Array of 28 joint torques (Nm)
        """
        # ctrl array maps to actuators
        if len(torques) == self.model.nu:
            self.data.ctrl[:] = torques
        else:
            # Assume direct mapping to first nu actuators
            n = min(len(torques), self.model.nu)
            self.data.ctrl[:n] = torques[:n]
    
    def get_feedback(self) -> RobotFeedback:
        """
        Get current robot state feedback.
        
        Returns:
            RobotFeedback with positions, velocities, and COM state
        """
        feedback = RobotFeedback()
        feedback.timestamp = time.time()
        
        # Base pose (from body xpos/xquat since base is welded)
        feedback.base_pos = self.data.xpos[self._base_body_id].copy()
        feedback.base_quat = self.data.xquat[self._base_body_id].copy()
        
        # Base velocity is zero (welded base)
        feedback.base_vel = np.zeros(6, dtype=np.float64)
        
        # Joint positions and velocities
        # Welded base: qpos/qvel are just the 28 hinge joints, no base prefix
        feedback.q = self.data.qpos[0:28].copy()
        feedback.dq = self.data.qvel[0:28].copy()
        
        # COM position and velocity
        # MuJoCo stores subtree COM in model coordinates
        feedback.com_pos = self.data.subtree_com[0].copy()  # Root body COM
        feedback.com_vel = np.zeros(3)  # Would need to compute from momentum
        
        return feedback
    
    def get_body_positions(self) -> dict:
        """
        Get end-effector positions from MuJoCo body xpos.
        
        Returns dict with hand_l, hand_r, elbow_l, elbow_r positions.
        """
        positions = {}
        
        # Body names to look up
        body_names = {
            'hand_l': 'LOWERWRIST_L',
            'hand_r': 'LOWERWRIST_R',
            'elbow_l': 'ELBOW_L',
            'elbow_r': 'ELBOW_R',
        }
        
        for key, body_name in body_names.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                positions[key] = self.data.xpos[body_id].copy()
            else:
                positions[key] = np.zeros(3)
        
        return positions
    
    def update_markers(self, desired: dict, actual: dict):
        """Update marker positions for visualization."""
        with self.markers_lock:
            for key in ['hand_l', 'hand_r', 'elbow_l', 'elbow_r']:
                if key in desired:
                    self.markers[f'{key}_des'] = desired[key].copy()
                if key in actual:
                    self.markers[f'{key}_act'] = actual[key].copy()
            # Orientation matrices for hand arrows
            for key in ['hand_l_orient_mat', 'hand_r_orient_mat']:
                if key in desired:
                    self.markers[key] = desired[key].copy()
    
    def _render_markers(self):
        """Render marker spheres and CBF safety sphere in viewer."""
        if self.viewer is None:
            return
            
        with self.markers_lock:
            markers = self.markers.copy()
        
        with self.viewer.lock():
            ngeom = 0
            
            # CBF safety sphere at torso (semi-transparent blue)
            # Get torso COM position from robot body
            torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
            if torso_body_id >= 0 and ngeom < self.viewer.user_scn.maxgeom:
                torso_pos = self.data.xpos[torso_body_id].copy()
                torso_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float32)
                torso_pos += torso_offset
                r_torso = 0.15  # Must match r_torso in robot_dynamics.py CBF
                safety_margin = 0.02  # Must match safety_margin in robot_dynamics.py CBF
                cbf_radius = r_torso + safety_margin
                
                g = self.viewer.user_scn.geoms[ngeom]
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [cbf_radius, cbf_radius, cbf_radius]
                g.pos[:] = torso_pos
                g.mat[:] = np.eye(3)
                g.rgba[:] = [0.0, 0.5, 1.0, 0.35]  # Light blue, semi-transparent
                ngeom += 1
            
            # CBF safety sphere at head (semi-transparent magenta)
            # Head is offset from COM by [-0.1, 0, 0.3]
            if torso_body_id >= 0 and ngeom < self.viewer.user_scn.maxgeom:
                torso_pos = self.data.xpos[torso_body_id].copy()
                head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float32)
                head_pos = torso_pos + head_offset
                r_head = 0.13  # Must match r_head in robot_dynamics.py CBF
                safety_margin = 0.02  # Must match safety_margin in robot_dynamics.py CBF
                cbf_radius_head = r_head + safety_margin
                
                g = self.viewer.user_scn.geoms[ngeom]
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [cbf_radius_head, cbf_radius_head, cbf_radius_head]
                g.pos[:] = head_pos
                g.mat[:] = np.eye(3)
                g.rgba[:] = [1.0, 0.0, 1.0, 0.25]  # Magenta, semi-transparent
                ngeom += 1
            
            # CBF safety sphere at crotch (semi-transparent green)
            # Crotch is offset from COM by [-0.1, 0, -0.3]
            if torso_body_id >= 0 and ngeom < self.viewer.user_scn.maxgeom:
                torso_pos = self.data.xpos[torso_body_id].copy()
                crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float32)
                crotch_pos = torso_pos + crotch_offset
                r_crotch = 0.19  # Must match r_crotch in robot_dynamics.py CBF
                safety_margin = 0.02  # Must match safety_margin in robot_dynamics.py CBF
                cbf_radius_crotch = r_crotch + safety_margin
                
                g = self.viewer.user_scn.geoms[ngeom]
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [cbf_radius_crotch, cbf_radius_crotch, cbf_radius_crotch]
                g.pos[:] = crotch_pos
                g.mat[:] = np.eye(3)
                g.rgba[:] = [0.0, 1.0, 0.5, 0.25]  # Cyan-green, semi-transparent
                ngeom += 1
            
            # Desired markers (increased contrast: LEFT brighter, RIGHT slightly cooler)
            # Actual markers (distinct but slightly muted relative to desired)
            
            # ── Render desired HAND markers as orientation arrows ────────
            arrow_length = 0.15   # total arrow length (m)
            arrow_radius = 0.012  # shaft radius
            cone_radius = 0.03   # arrowhead radius
            cone_length = 0.05   # arrowhead length
            shaft_length = arrow_length - cone_length
            
            hand_arrow_defs = {
                'hand_l_des': ('hand_l_orient_mat', [1.00, 0.08, 0.18, 1.00]),
                'hand_r_des': ('hand_r_orient_mat', [0.08, 0.55, 1.00, 1.00]),
            }
            
            for key, (mat_key, color) in hand_arrow_defs.items():
                pos = markers.get(key, np.zeros(3))
                orient_mat = markers.get(mat_key, np.eye(3))
                if np.any(pos != 0) and ngeom + 1 < self.viewer.user_scn.maxgeom:
                    # Z-axis of orient_mat is the arm/arrow direction
                    arm_dir = orient_mat[:, 2]
                    
                    # MuJoCo cylinder Z-axis = length axis, so we can use the matrix directly
                    # just need to ensure the matrix is properly formatted
                    
                    # Shaft (cylinder): starts at hand pos, extends along arm_dir
                    shaft_center = pos + arm_dir * (shaft_length * 0.5)
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    g.size[:] = [arrow_radius, shaft_length * 0.5, 0]
                    g.pos[:] = shaft_center
                    g.mat[:] = orient_mat
                    g.rgba[:] = color
                    ngeom += 1
                    
                    # Arrowhead (wider cylinder): at tip of shaft
                    cone_base = pos + arm_dir * shaft_length
                    cone_center = cone_base + arm_dir * (cone_length * 0.5)
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    g.size[:] = [cone_radius, cone_length * 0.5, 0]
                    g.pos[:] = cone_center
                    g.mat[:] = orient_mat
                    g.rgba[:] = color
                    ngeom += 1
            
            # ── Render desired ELBOW markers as spheres ──────────────────
            elbow_colors_des = {
                'elbow_l_des': [1.00, 0.18, 0.28, 1.00],
                'elbow_r_des': [0.18, 0.72, 1.00, 1.00],
            }
            for key, color in elbow_colors_des.items():
                pos = markers.get(key, np.zeros(3))
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.05, 0.05, 0.05]
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = color
                    ngeom += 1
            
            # Render actual markers (slightly smaller spheres)
            colors_act = {
                'hand_l_act': [0.85, 0.06, 0.12, 0.90],
                'hand_r_act': [0.06, 0.40, 0.72, 0.90],
                'elbow_l_act': [0.90, 0.12, 0.18, 0.90],
                'elbow_r_act': [0.14, 0.58, 0.82, 0.90],
            }
            for key, color in colors_act.items():
                pos = markers.get(key, np.zeros(3))
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.045, 0.045, 0.045]  # 4.5cm spheres
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = color
                    ngeom += 1
            
            self.viewer.user_scn.ngeom = ngeom
    
    def step(self):
        """
        Advance simulation by one timestep.
        
        Gets torque commands from shared state, steps physics,
        and publishes feedback.
        """
        # Get torque command from controller
        torques, cmd_time = self.shared.get_torque_command()
        self.set_torques(torques)
        
        # Step simulation (welded base = no dynamic coupling through floating base)
        mujoco.mj_step(self.model, self.data)
        
        self.sim_time += self.sim_config.sim_dt
        self.step_count += 1
        
        # Publish feedback
        feedback = self.get_feedback()
        self.shared.set_robot_feedback(feedback)
    
    def start_viewer(self):
        """Start visualization window."""
        if self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer is not None:
                # Configure camera
                self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
                self.viewer.cam.distance = 3.0
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 90
                print("[Simulation] Viewer started")
    
    def sync_viewer(self):
        """Sync viewer with simulation state."""
        if self.viewer is not None and self.viewer.is_running():
            self._render_markers()
            self.viewer.sync()
    
    def is_viewer_running(self) -> bool:
        """Check if viewer window is still open."""
        if self.viewer is None:
            return True  # No viewer, consider "running"
        return self.viewer.is_running()
    
    def run_realtime(self, duration: float = None):
        """
        Run simulation in real-time.
        
        This is a blocking call that runs the simulation loop,
        synchronizing sim time with wall time.
        
        Args:
            duration: Optional max duration in seconds
        """
        self.running = True
        self.wall_time_start = time.time()
        self.sim_time = 0.0
        self.step_count = 0
        
        last_render_time = 0.0
        render_interval = 1.0 / self.sim_config.render_fps
        
        # Timing statistics
        last_stats_time = time.time()
        stats_step_count = 0
        
        print("[Simulation] Starting real-time loop...")
        
        try:
            while self.running:
                # Check shutdown
                if self.shared.is_shutdown_requested():
                    break
                
                # Check viewer
                if not self.is_viewer_running():
                    self.shared.request_shutdown()
                    break
                
                # Check duration
                if duration is not None and self.sim_time >= duration:
                    break
                
                # Compute target sim time based on wall time
                wall_elapsed = time.time() - self.wall_time_start
                
                # Step simulation to catch up with wall time
                while self.sim_time < wall_elapsed and self.running:
                    if self.paused:
                        self.wall_time_start = time.time() - self.sim_time
                        break
                    
                    self.step()
                    stats_step_count += 1
                
                # Render at specified framerate
                if time.time() - last_render_time >= render_interval:
                    self.sync_viewer()
                    last_render_time = time.time()
                
                # Print timing stats periodically
                if time.time() - last_stats_time >= 2.0:
                    actual_hz = stats_step_count / (time.time() - last_stats_time)
                    self.shared.update_timing('sim', actual_hz)
                    stats_step_count = 0
                    last_stats_time = time.time()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.0001)
                
        except KeyboardInterrupt:
            print("[Simulation] Interrupted")
        finally:
            self.running = False
            print(f"[Simulation] Stopped after {self.step_count} steps ({self.sim_time:.2f}s sim time)")
    
    def stop(self):
        """Stop simulation."""
        self.running = False


class SimulationThread(threading.Thread):
    """Thread wrapper for running simulation in background."""
    
    def __init__(self, sim: MuJoCoSimulation, duration: float = None):
        super().__init__(daemon=True)
        self.sim = sim
        self.duration = duration
        
    def run(self):
        self.sim.run_realtime(self.duration)
