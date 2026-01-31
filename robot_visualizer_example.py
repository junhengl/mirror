"""
Example integration of ZED body tracking with MuJoCo visualization.

This script demonstrates how to use the ThemisSimulator to visualize
the robot pose based on input from the ZED body tracking.
"""

#### sudo /home/junhengl/body_tracking/.venv/bin/python integrated_tracking_visualization.py

import numpy as np
from mujoco_visualizer import ThemisSimulator
import time


class RobotVisualizer:
    """Integrates body tracking with robot visualization."""
    
    def __init__(self):
        """Initialize the visualizer and simulator."""
        self.sim = ThemisSimulator(headless=False)
        self._print_available_joints()
    
    def _print_available_joints(self):
        """Print all available joints in the robot."""
        print("\nAvailable joints in Themis robot:")
        joint_pos = self.sim.get_joint_positions()
        joint_names = sorted([j for j in joint_pos.keys() if j != "free_joint"])
        for joint_name in joint_names:
            print(f"  - {joint_name}")
    
    def update_robot_pose(
        self,
        base_position: np.ndarray,
        base_rotation: np.ndarray,
        joint_positions: dict
    ):
        """
        Update the robot pose in the simulator.
        
        Args:
            base_position: Base link position [x, y, z]
            base_rotation: Base link rotation as quaternion [x, y, z, w]
            joint_positions: Dictionary of joint names to positions (radians)
        """
        # Update base pose
        self.sim.set_base_pose(base_position, base_rotation)
        
        # Update joint positions
        self.sim.set_joint_positions(joint_positions)
    
    def step(self):
        """Step the simulation (forward kinematics only, no dynamics)."""
        # Just update forward kinematics, don't run physics
        import mujoco
        mujoco.mj_forward(self.sim.model, self.sim.data)
    
    def close(self):
        """Close the visualizer."""
        self.sim.close()


def main():
    """Example of continuous pose updates."""
    print("Initializing Robot Visualizer...")
    viz = RobotVisualizer()
    
    # Set camera to focus on robot
    if viz.sim.viewer is not None:
        viz.sim.viewer.cam.lookat[:] = [0.0, 0.0, 1.5]  # Look at robot center
        viz.sim.viewer.cam.distance = 3.0  # Distance from target
        viz.sim.viewer.cam.elevation = -20  # Camera elevation angle
        viz.sim.viewer.cam.azimuth = 90  # Camera azimuth angle
    
    print("\nShowing robot with animated arm motions... (Press Ctrl+C to stop)")
    
    # Robot floating in the air
    base_height = 1.5
    
    # Animation start time
    start_time = time.time()
    
    try:
        while True:
            # Calculate animation time
            t = time.time() - start_time
            
            # Create sinusoidal arm motions
            # Right arm: wave motion (shoulder pitch and elbow)
            shoulder_pitch_r = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz wave
            elbow_pitch_r = 0.3 + 0.5 * np.sin(2 * np.pi * 0.5 * t + np.pi/4)  # Elbow follows
            
            # Left arm: circular motion (shoulder pitch and roll)
            shoulder_pitch_l = 0.6 * np.sin(2 * np.pi * 0.3 * t)  # 0.3 Hz slower
            shoulder_roll_l = 0.6 * np.cos(2 * np.pi * 0.3 * t)
            
            # Both wrists: gentle rotation
            wrist_yaw_r = 0.4 * np.sin(2 * np.pi * 0.7 * t)
            wrist_yaw_l = 0.4 * np.sin(2 * np.pi * 0.7 * t + np.pi)  # Out of phase
            
            # Head: gentle looking around
            head_yaw = 0.5 * np.sin(2 * np.pi * 0.2 * t)
            head_pitch = 0.2 * np.cos(2 * np.pi * 0.15 * t)
            
            # Zero pose with animated arms and head
            animated_pose = {
                "HIP_YAW_R": 0.0,
                "HIP_ROLL_R": 0.0,
                "HIP_PITCH_R": 0.0,
                "KNEE_PITCH_R": 0.0,
                "ANKLE_PITCH_R": 0.0,
                "ANKLE_ROLL_R": 0.0,
                
                "HIP_YAW_L": 0.0,
                "HIP_ROLL_L": 0.0,
                "HIP_PITCH_L": 0.0,
                "KNEE_PITCH_L": 0.0,
                "ANKLE_PITCH_L": 0.0,
                "ANKLE_ROLL_L": 0.0,
                
                "SHOULDER_YAW_R": 0.3,
                "SHOULDER_ROLL_R": 0.0,  # Arms slightly out
                "SHOULDER_PITCH_R": 0.0,
                "ELBOW_YAW_R": 0.0,
                "ELBOW_PITCH_R": 0.0,
                "WRIST_YAW_R": wrist_yaw_r*0,
                "WRIST_PITCH_R": 0.0,
                
                "SHOULDER_YAW_L": 0.0,
                "SHOULDER_ROLL_L": -0.3,  # Arms slightly out
                "SHOULDER_PITCH_L": shoulder_pitch_l*0,
                "ELBOW_YAW_L": 0.0,
                "ELBOW_PITCH_L": 0.3,
                "WRIST_YAW_L": wrist_yaw_l*0,
                "WRIST_PITCH_L": 0.0,
                
                "HEAD_YAW": head_yaw,
                "HEAD_PITCH": head_pitch,
            }
            
            # Keep robot at fixed height with zero rotation
            base_pos = np.array([0.0, 0.0, base_height])
            base_rot = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
            
            # Update robot pose with animated joints
            viz.update_robot_pose(base_pos, base_rot, animated_pose)
            viz.step()
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        viz.close()


if __name__ == "__main__":
    main()
