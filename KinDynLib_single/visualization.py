"""
Visualization module for robot pose animation using MuJoCo

This module provides functions to visualize robot configurations and animate
sequences of configurations (e.g., IK iteration solutions).
"""

import numpy as np
import time


def animate_ik_iterations(q_trajectory, hand_positions_trajectory=None, 
                          xml_path=None, frame_duration=0.5, loop=True):
    """
    Animate a sequence of robot configurations (e.g., from IK iterations)
    
    Args:
        q_trajectory: list of (DOF,) configuration vectors, one per iteration
        hand_positions_trajectory: list of dicts with hand position info per iteration
                                   Each dict has 'right', 'left', 'right_des', 'left_des' keys
        xml_path: str, path to robot XML file (default: Themis TH02-A7.xml)
        frame_duration: float, time to display each frame in seconds
        loop: bool, whether to loop the animation
    """
    # Import mujoco only when visualization is needed
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco not installed. Skipping visualization.")
        print("Install with: pip install mujoco")
        return
    
    # Import euler_to_quat for quaternion conversion
    from kinwbc import euler_to_quat
    
    if xml_path is None:
        xml_path = "/home/junhengl/g1_ctrl_py/g1_ctrl/westwood_robots/TH02-A7.xml"
    
    num_frames = len(q_trajectory)
    print()
    print("=" * 60)
    print("IK ITERATION ANIMATION")
    print("=" * 60)
    print(f"Loading robot model from: {xml_path}")
    print(f"Number of frames: {num_frames}")
    print(f"Frame duration: {frame_duration}s")
    print(f"Loop animation: {loop}")
    print()
    
    try:
        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        print("Controls:")
        print("  - Animation will cycle through IK iterations")
        print("  - Close the viewer window to exit")
        print("=" * 60)
        
        # Launch interactive viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame_idx = 0
            last_frame_time = time.time()
            
            while viewer.is_running():
                current_time = time.time()
                
                # Check if it's time to advance to next frame
                if current_time - last_frame_time >= frame_duration:
                    frame_idx += 1
                    if frame_idx >= num_frames:
                        if loop:
                            frame_idx = 0
                        else:
                            frame_idx = num_frames - 1
                    last_frame_time = current_time
                
                # Get current configuration
                q_config = q_trajectory[frame_idx]
                
                # Set robot configuration
                # Base position
                data.qpos[0:3] = q_config[0:3]
                
                # Base orientation - convert from euler angles to quaternion
                roll, pitch, yaw = q_config[3], q_config[4], q_config[5]
                quat_wxyz = euler_to_quat(np.array([roll, pitch, yaw], dtype=np.float32))
                data.qpos[3:7] = quat_wxyz
                
                # Joint positions (28 actuated joints)
                data.qpos[7:] = q_config[6:]
                
                # Update kinematics
                mujoco.mj_forward(model, data)
                
                # Add hand position markers
                if hand_positions_trajectory is not None and frame_idx < len(hand_positions_trajectory):
                    hand_positions = hand_positions_trajectory[frame_idx]
                    _draw_hand_markers(viewer, mujoco, hand_positions)
                
                viewer.sync()
                time.sleep(0.02)  # 50 Hz update rate
        
        print("Animation viewer closed.")
        
    except Exception as e:
        print(f"Error during animation: {e}")
        import traceback
        traceback.print_exc()


def _draw_hand_markers(viewer, mujoco, hand_positions):
    """
    Draw hand and elbow position markers as spheres
    
    Args:
        viewer: MuJoCo viewer instance
        mujoco: mujoco module
        hand_positions: dict with 'right', 'left', 'right_des', 'left_des', 
                        'elbow_right', 'elbow_left', 'elbow_right_des', 'elbow_left_des' keys
    """
    if hand_positions is None:
        return
    
    geom_idx = 0
    
    # Right hand actual - solid red sphere
    if 'right' in hand_positions:
        right_pos = hand_positions['right']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]),  # size (radius)
            right_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([1.0, 0.2, 0.2, 0.9])  # rgba (solid red)
        )
        geom_idx += 1
    
    # Left hand actual - solid blue sphere
    if 'left' in hand_positions:
        left_pos = hand_positions['left']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]),  # size (radius)
            left_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([0.2, 0.2, 1.0, 0.9])  # rgba (solid blue)
        )
        geom_idx += 1
    
    # Right hand desired - transparent red sphere
    if 'right_des' in hand_positions:
        right_des_pos = hand_positions['right_des']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.035, 0, 0]),  # size (slightly larger radius)
            right_des_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([1.0, 0.4, 0.4, 0.3])  # rgba (transparent red)
        )
        geom_idx += 1
    
    # Left hand desired - transparent blue sphere
    if 'left_des' in hand_positions:
        left_des_pos = hand_positions['left_des']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.035, 0, 0]),  # size (slightly larger radius)
            left_des_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([0.4, 0.4, 1.0, 0.3])  # rgba (transparent blue)
        )
        geom_idx += 1
    
    # Right elbow actual - solid orange sphere
    if 'elbow_right' in hand_positions:
        elbow_right_pos = hand_positions['elbow_right']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0, 0]),  # size (slightly smaller radius)
            elbow_right_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([1.0, 0.6, 0.0, 0.9])  # rgba (solid orange)
        )
        geom_idx += 1
    
    # Left elbow actual - solid cyan sphere
    if 'elbow_left' in hand_positions:
        elbow_left_pos = hand_positions['elbow_left']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0, 0]),  # size (slightly smaller radius)
            elbow_left_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([0.0, 0.8, 0.8, 0.9])  # rgba (solid cyan)
        )
        geom_idx += 1
    
    # Right elbow desired - transparent orange sphere
    if 'elbow_right_des' in hand_positions:
        elbow_right_des_pos = hand_positions['elbow_right_des']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]),  # size (slightly larger than actual)
            elbow_right_des_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([1.0, 0.7, 0.3, 0.3])  # rgba (transparent orange)
        )
        geom_idx += 1
    
    # Left elbow desired - transparent cyan sphere
    if 'elbow_left_des' in hand_positions:
        elbow_left_des_pos = hand_positions['elbow_left_des']
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]),  # size (slightly larger than actual)
            elbow_left_des_pos.astype(np.float64),  # position
            np.eye(3).flatten().astype(np.float64),  # rotation matrix
            np.array([0.3, 0.9, 0.9, 0.3])  # rgba (transparent cyan)
        )
        geom_idx += 1
    
    viewer.user_scn.ngeom = geom_idx


def visualize_single_pose(q_config, hand_positions=None, xml_path=None):
    """
    Visualize a single robot configuration (static pose)
    
    Args:
        q_config: (DOF,) configuration vector [base_pos(3), base_rot(3), joint_pos(28)]
        hand_positions: dict with 'left', 'right', 'left_des', 'right_des' keys
        xml_path: str, path to robot XML file
    """
    # Import mujoco only when visualization is needed
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco not installed. Skipping visualization.")
        print("Install with: pip install mujoco")
        return
    
    from kinwbc import euler_to_quat
    
    if xml_path is None:
        xml_path = "/home/junhengl/g1_ctrl_py/g1_ctrl/westwood_robots/TH02-A7.xml"
    
    print()
    print("=" * 60)
    print("STATIC POSE VISUALIZATION")
    print("=" * 60)
    print(f"Loading robot model from: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Set robot configuration
        data.qpos[0:3] = q_config[0:3]
        roll, pitch, yaw = q_config[3], q_config[4], q_config[5]
        quat_wxyz = euler_to_quat(np.array([roll, pitch, yaw], dtype=np.float32))
        data.qpos[3:7] = quat_wxyz
        data.qpos[7:] = q_config[6:]
        
        mujoco.mj_forward(model, data)
        
        print("Viewer launched. Close window to exit.")
        print("=" * 60)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if hand_positions is not None:
                    _draw_hand_markers(viewer, mujoco, hand_positions)
                viewer.sync()
                time.sleep(0.05)
        
        print("Viewer closed.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
