#!/usr/bin/env python3
"""
Test script for KinDynLib_single - single-threaded numpy version
Mimics par_test_seq.py but for single robot input (no parallelization)
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from kinwbc import run_kinWBC, SolverInput, Trajectory, quat_to_euler, euler_to_quat
from robot_const import DOF, N_LINKS


def create_step_input(step_idx, prev_base_pos, prev_base_quat, prev_joint_pos,
                      target_step_length, target_step_height, target_rpy):
    """Create SolverInput for the next step based on previous state
    
    Args:
        step_idx: Current step index
        prev_base_pos: Previous base position [x, y, z]
        prev_base_quat: Previous base quaternion [w, x, y, z]
        prev_joint_pos: Previous joint positions
        target_step_length: Desired step length [x, y, z]
        target_step_height: Desired step height
        target_rpy: Target orientation [roll, pitch, yaw]
    """
    inp = SolverInput()
    
    # Use previous final state as initial state
    inp.base0_pos = prev_base_pos.copy()
    inp.base0_quat = prev_base_quat.copy()
    inp.joint0_pos = prev_joint_pos.copy()
    
    # Step parameters
    inp.target_step_length = target_step_length.copy()
    inp.target_step_height = target_step_height
    inp.target_rpy = target_rpy.copy()
    
    if step_idx == 0:
        # Left foot (anchor) stays at current position
        inp.anchor_foot_wpos = np.array([
            prev_base_pos[0],
            -0.1,  # Left side (y)
            0.0
        ], dtype=np.float32)
        # Right foot (swing) moves forward by step_length
        inp.swing_foot_wpos = np.array([
            prev_base_pos[0],
            0.1,   # Right side (y)
            0.0
        ], dtype=np.float32)
        inp.target_step_length[0] = target_step_length[0] / 2
    else:
        # Alternate feet (0 = right foot swing, 1 = left foot swing)
        inp.foot = step_idx % 2
        inp.step = step_idx
        
        # Calculate foot positions relative to current base position
        if inp.foot == 0:  # Right foot swing
            # Left foot (anchor) stays at current position
            inp.anchor_foot_wpos = np.array([
                prev_base_pos[0] + target_step_length[0] / 2,
                prev_base_pos[1] + target_step_length[1] / 2 - 0.1,  # Left side (y)
                0.0
            ], dtype=np.float32)
            # Right foot (swing) moves forward by step_length
            inp.swing_foot_wpos = np.array([
                prev_base_pos[0] - target_step_length[0] / 2,
                prev_base_pos[1] - target_step_length[1] / 2 + 0.1,   # Right side (y)
                0.0
            ], dtype=np.float32)
        else:  # Left foot swing
            # Right foot (anchor) stays at current position
            inp.anchor_foot_wpos = np.array([
                prev_base_pos[0] + target_step_length[0] / 2,
                prev_base_pos[1] + target_step_length[1] / 2 + 0.1,   # Right side (y)
                0.0
            ], dtype=np.float32)
            # Left foot (swing) moves forward by step_length
            inp.swing_foot_wpos = np.array([
                prev_base_pos[0] - target_step_length[0] / 2,
                prev_base_pos[1] - target_step_length[1] / 2 - 0.1,  # Left side (y)
                0.0
            ], dtype=np.float32)
    
    return inp


def main():
    print("=" * 60)
    print("KinDynLib Single-Threaded Test")
    print("=" * 60)
    
    # Initialize MuJoCo
    model_path = "../unitree_robots/g1/g1_29dof.xml"
    print(f"Loading MuJoCo model from: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"Model loaded successfully!")
        print(f"  nq (positions): {model.nq}")
        print(f"  nv (velocities): {model.nv}")
        print(f"  nu (actuators): {model.nu}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Walking parameters
    num_steps = 1  # Generate only 1 step for debugging
    T = 20  # timesteps per step
    dt = 0.001
    
    target_step_length = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target_step_height = 0.1
    target_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Initial state
    init_base_pos = np.array([0.0, 0.0, 0.72], dtype=np.float32)
    init_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    init_joint_pos = np.zeros(29, dtype=np.float32)
    init_joint_pos[0] = -0.1
    init_joint_pos[3] = 0.2
    init_joint_pos[4] = -0.1
    init_joint_pos[6] = -0.1
    init_joint_pos[9] = 0.2
    init_joint_pos[10] = -0.1
    init_joint_pos[15] = 0.2
    init_joint_pos[16] = 0.2
    init_joint_pos[18] = 1.1
    init_joint_pos[22] = 0.2
    init_joint_pos[23] = -0.2
    init_joint_pos[25] = 1.1
    
    # Storage for all trajectories
    all_trajectories = []
    
    # Generate trajectories for each step
    print(f"\nGenerating {num_steps} walking steps...")
    for step_idx in range(num_steps):
        print(f"\n--- Step {step_idx + 1}/{num_steps} ---")
        
        # Get previous state
        if step_idx == 0:
            prev_base_pos = init_base_pos
            prev_base_quat = init_base_quat
            prev_joint_pos = init_joint_pos
        else:
            prev_traj = all_trajectories[-1]
            last_frame = prev_traj.get_frame(T - 1)
            prev_base_pos = last_frame[:3]
            prev_base_quat = last_frame[3:7]
            prev_joint_pos = last_frame[7:]
        
        # Create input for this step
        inp = create_step_input(step_idx, prev_base_pos, prev_base_quat, prev_joint_pos,
                               target_step_length, target_step_height, target_rpy)
        
        # Generate trajectory
        traj = Trajectory(T, dt)
        start_time = time.time()
        
        for t in range(T):
            run_kinWBC(inp, traj, t, T)
        
        elapsed = time.time() - start_time
        print(f"  Trajectory generation time: {elapsed*1000:.2f} ms ({elapsed*1000/T:.2f} ms per frame)")
        
        all_trajectories.append(traj)
    
    print("\n" + "=" * 60)
    print(f"Total steps generated: {num_steps}")
    print(f"Total frames: {num_steps * T}")
    print("=" * 60)
    
    # Print all trajectory states
    print("\n" + "=" * 60)
    print("TRAJECTORY STATES")
    print("=" * 60)
    
    for step_idx in range(num_steps):
        print(f"\n--- Step {step_idx + 1} ---")
        traj = all_trajectories[step_idx]
        
        for t in [0, T//2, T-1]:  # Print first, middle, and last frame
            frame = traj.get_frame(t)
            print(f"\n  Frame t={t}:")
            print(f"    Base position: [{frame[0]:.4f}, {frame[1]:.4f}, {frame[2]:.4f}]")
            print(f"    Base quat (w,x,y,z): [{frame[3]:.4f}, {frame[4]:.4f}, {frame[5]:.4f}, {frame[6]:.4f}]")
            print(f"    Joint positions (29 DOF):")
            
            # Print joints in groups for readability
            joints = frame[7:]
            print(f"      Left leg  [0-5]:   {joints[0:6]}")
            print(f"      Right leg [6-11]:  {joints[6:12]}")
            print(f"      Waist     [12-14]: {joints[12:15]}")
            print(f"      Head      [15-17]: {joints[15:18]}")
            print(f"      Left arm  [18-22]: {joints[18:23]}")
            print(f"      Right arm [23-28]: {joints[23:29]}")
    
    print("\n" + "=" * 60)
    
    # Visualize in MuJoCo
    print("\nLaunching MuJoCo viewer...")
    print("Controls:")
    print("  - Space: pause/resume")
    print("  - ESC: exit")
    print("\nInitializing MuJoCo data with first frame...")
    
    # Initialize data with first frame
    first_frame = all_trajectories[0].get_frame(0)
    data.qpos[0:3] = first_frame[:3]
    quat = first_frame[3:7]  # [w, x, y, z]
    data.qpos[3:7] = [quat[1], quat[2], quat[3], quat[0]]  # MuJoCo uses [x, y, z, w]
    data.qpos[7:] = first_frame[7:]
    mujoco.mj_forward(model, data)
    
    print(f"Initial qpos set: base=[{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        total_frames = num_steps * T
        
        print("\nViewer launched! Playing trajectory...")
        print("(Slower playback for better visualization)")
        
        while viewer.is_running():
            step_idx = frame_idx // T
            t = frame_idx % T
            
            if step_idx < num_steps:
                frame = all_trajectories[step_idx].get_frame(t)
                
                # Update MuJoCo data
                # Base position
                data.qpos[0:3] = frame[:3]
                
                # Base orientation (convert to MuJoCo format)
                quat = frame[3:7]  # [w, x, y, z]
                data.qpos[3:7] = [quat[1], quat[2], quat[3], quat[0]]  # MuJoCo uses [x, y, z, w]
                
                # Joint positions
                data.qpos[7:] = frame[7:]
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
            
            # Advance frame
            frame_idx = (frame_idx + 1) % total_frames
            
            # Sync viewer
            viewer.sync()
            time.sleep(dt * 10)  # Slower playback (10x slower)


if __name__ == "__main__":
    main()
