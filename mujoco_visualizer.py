"""
MuJoCo Simulation Visualizer for Themis Humanoid Robot

This module provides a real-time visualization API for the Themis humanoid robot
using MuJoCo simulator. It allows updating the robot's base position/rotation and
joint positions in real-time.
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import time


class ThemisSimulator:
    """Real-time MuJoCo simulator for Themis humanoid robot."""
    
    def __init__(self, model_path: Optional[str] = None, headless: bool = False):
        """
        Initialize the Themis simulator.
        
        Args:
            model_path: Path to MJCF/URDF file. If None, uses default Themis MJCF.
            headless: If True, run without viewer. Use render() method to get frames.
        """
        if model_path is None:
            # Try MJCF first, fallback to URDF
            mjcf_path = Path(__file__).parent / "themis" / "TH02-A7.xml"
            if mjcf_path.exists():
                model_path = mjcf_path
            else:
                model_path = Path(__file__).parent / "themis" / "urdf" / "TH02-A7.urdf"
        
        self.model_path = Path(model_path)
        self.headless = headless
        
        # Load model directly (no conversion needed for MJCF)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        
        # Store initial state
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        
        # Viewer and threading
        self.viewer = None
        self._viewer_thread = None
        self._viewer_running = False
        self._lock = threading.Lock()
        
        if not headless:
            self._start_viewer()
    
    def _start_viewer(self):
        """Start the MuJoCo viewer in a separate thread."""
        self._viewer_running = True
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False
        )
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()
    
    def _viewer_loop(self):
        """Main viewer loop running in separate thread."""
        while self._viewer_running and self.viewer.is_running():
            # Sync viewer with current data state
            with self._lock:
                self.viewer.sync()
            time.sleep(0.01)  # Slower update rate to avoid conflicts
    
    def set_base_pose(self, position: np.ndarray, rotation: np.ndarray):
        """
        Set the base link position and rotation.
        
        Args:
            position: Base position [x, y, z]
            rotation: Base rotation as quaternion [x, y, z, w] or rotation matrix (3x3)
        """
        with self._lock:
            position = np.array(position, dtype=np.float64)
            
            # Handle rotation
            if isinstance(rotation, np.ndarray):
                if rotation.shape == (3, 3):
                    # Convert rotation matrix to quaternion
                    rotation = self._rotation_matrix_to_quaternion(rotation)
                else:
                    rotation = np.array(rotation, dtype=np.float64)
                    # Ensure quaternion is [x, y, z, w] format
                    if len(rotation) == 4:
                        rotation = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])
            
            # Update base position
            self.data.qpos[0:3] = position
            
            # Update base rotation (quaternion: [x, y, z, w])
            self.data.qpos[3:7] = rotation
            
            mujoco.mj_forward(self.model, self.data)
    
    def set_joint_positions(self, joint_positions: dict):
        """
        Set joint positions for the robot.
        
        Args:
            joint_positions: Dictionary mapping joint names to position values (in radians)
                           Example: {"HIP_YAW_R": 0.0, "HIP_PITCH_R": -0.5, ...}
        """
        # Joints that need to be negated due to reversed axis orientations in URDF
        negated_joints = {
            "HIP_ROLL_R", "HIP_ROLL_L",
            "HIP_PITCH_R", "KNEE_PITCH_R", "ANKLE_PITCH_R",
            "HIP_PITCH_L", "KNEE_PITCH_L", "ANKLE_PITCH_L",
            "SHOULDER_YAW_L",
            "SHOULDER_PITCH_R", "SHOULDER_PITCH_L",
            "ELBOW_PITCH_R",
            "ELBOW_YAW_R", "ELBOW_YAW_L",
            "WRIST_PITCH_R"
        }
        
        with self._lock:
            for joint_name, position in joint_positions.items():
                try:
                    joint_id = mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_name
                    )
                    if joint_id >= 0:
                        # Find the index in qpos (accounting for base 7 DOF: 3 position + 4 quaternion)
                        qpos_addr = self.model.jnt_qposadr[joint_id]
                        # Negate left leg pitch joints to match right leg convention
                        if joint_name in negated_joints:
                            self.data.qpos[qpos_addr] = -position
                        else:
                            self.data.qpos[qpos_addr] = position
                except Exception as e:
                    print(f"Warning: Could not set joint {joint_name}: {e}")
            
            mujoco.mj_forward(self.model, self.data)
    
    def get_joint_positions(self) -> dict:
        """
        Get current joint positions.
        
        Returns:
            Dictionary mapping joint names to their current positions
        """
        # Joints that need to be negated due to reversed axis orientations in URDF
        negated_joints = {
            "HIP_ROLL_R", "HIP_ROLL_L",
            "HIP_PITCH_R", "KNEE_PITCH_R", "ANKLE_PITCH_R",
            "HIP_PITCH_L", "KNEE_PITCH_L", "ANKLE_PITCH_L",
            "SHOULDER_YAW_L",
            "SHOULDER_PITCH_R", "SHOULDER_PITCH_L",
            "ELBOW_PITCH_R",
            "ELBOW_YAW_R", "ELBOW_YAW_L",
            "WRIST_PITCH_R"
        }
        
        with self._lock:
            joint_positions = {}
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    qpos_addr = self.model.jnt_qposadr[i]
                    qpos_size = self.model.jnt_qposadr[i + 1] - qpos_addr if i + 1 < self.model.njnt else 1
                    if qpos_size == 1:
                        value = float(self.data.qpos[qpos_addr])
                        # Un-negate left leg pitch joints for consistent API
                        if joint_name in negated_joints:
                            joint_positions[joint_name] = -value
                        else:
                            joint_positions[joint_name] = value
                    else:
                        joint_positions[joint_name] = self.data.qpos[qpos_addr:qpos_addr + qpos_size].copy()
            return joint_positions
    
    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose of a body.
        
        Args:
            body_name: Name of the body
            
        Returns:
            Tuple of (position, rotation_matrix)
        """
        with self._lock:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Body '{body_name}' not found in model")
            
            position = self.data.xpos[body_id].copy()
            rotation = self.data.xmat[body_id].reshape(3, 3).copy()
            return position, rotation
    
    def reset(self):
        """Reset the simulation to initial state."""
        with self._lock:
            self.data.qpos = self._initial_qpos.copy()
            self.data.qvel = self._initial_qvel.copy()
            mujoco.mj_forward(self.model, self.data)
    
    def step(self, dt: Optional[float] = None):
        """
        Step the simulation forward.
        
        Args:
            dt: Time step duration. If None, uses model default.
        """
        with self._lock:
            if dt is not None:
                self.model.opt.timestep = dt
            mujoco.mj_step(self.model, self.data)
    
    def render(self) -> np.ndarray:
        """
        Get a rendered frame of the simulation.
        
        Returns:
            Image array (height, width, 3)
        """
        with self._lock:
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            renderer.close()
            return pixels
    
    def close(self):
        """Close the simulator and viewer."""
        self._viewer_running = False
        if self.viewer is not None:
            self.viewer.close()
    
    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion [x, y, z, w].
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [x, y, z, w]
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])


def example_usage():
    """Example of how to use the Themis simulator."""
    print("Initializing Themis simulator...")
    sim = ThemisSimulator(headless=False)
    
    print("Robot joint names:")
    joint_pos = sim.get_joint_positions()
    for joint_name in sorted(joint_pos.keys()):
        if joint_name != "free_joint":  # Skip the free joint
            print(f"  {joint_name}")
    
    print("\nSetting robot pose...")
    # Set base position and rotation (identity rotation)
    base_pos = np.array([0.0, 0.0, 1.0])
    base_rot = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    sim.set_base_pose(base_pos, base_rot)
    
    # Set some joint positions
    joint_targets = {
        "HIP_YAW_R": 0.0,
        "HIP_PITCH_R": -0.5,
        "HIP_ABAD_R": 0.0,
    }
    sim.set_joint_positions(joint_targets)
    
    print("Simulator running... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sim.close()


if __name__ == "__main__":
    example_usage()
