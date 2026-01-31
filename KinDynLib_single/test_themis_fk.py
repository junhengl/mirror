import numpy as np
import sys
import time

# Test forward kinematics for Themis robot
# Import Themis robot constants by temporarily renaming them
import robot_const as themis_const

# Replace robot_const module with themis constants
sys.modules['robot_const'] = themis_const

from robot_dynamics import Robot
from math_operations import euler_to_rotation_matrix
from dynamics_lib import Xtrans
from kinwbc import euler_to_quat

def test_themis_fk():
    """Test forward kinematics for Themis robot hands at zero configuration"""
    
    print("="*60)
    print("THEMIS ROBOT FORWARD KINEMATICS TEST")
    print("="*60)
    print(f"DOF: {themis_const.DOF}")
    print(f"Number of links: {themis_const.N_LINKS}")
    print()
    
    # Create Themis robot instance (it will read from robot_const_themis module)
    robot = Robot()
    
    # Initialize state: all joints at zero, COM at zero
    base_pos = np.zeros(3, dtype=np.float32)  # [0, 0, 0]
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity quaternion (w, x, y, z)
    joint_pos = np.zeros(28, dtype=np.float32)  # All actuated joints at zero
    
    print("Initial Configuration:")
    print(f"  Base position: {base_pos}")
    print(f"  Base quaternion (wxyz): {base_quat}")
    print(f"  Joint positions: all zeros (28 joints)")
    print()
    
    # Set robot state
    # Convert quaternion to rotation matrix for robot.update
    R_base = euler_to_rotation_matrix(0.0, 0.0, 0.0)  # Identity rotation
    
    # Construct full q vector: [base_pos (3), base_rot (3), joint_pos (28)]
    q = np.zeros(themis_const.DOF, dtype=np.float32)
    q[0:3] = base_pos  # base position
    # q[3:6] are base rotations, already zeros
    # q[6:] are joint positions, already zeros
    dq = np.zeros(themis_const.DOF, dtype=np.float32)

    ## joint offset to avoid singularities at zero position
    joint_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, -0.1, 0.1,-0.1, 0.0,
                             0.0, 0.0, -0.1, 0.1,-0.1, 0.0,
                             0, 1.57, -1.57, 0.78, 0.0, 0.78, 0,
                             0, -1.57, -1.57, -0.78, 0.0, -0.78, 0,
                             0.0, 0.0], dtype=np.float32)
    q += joint_offset
    
    # Update robot with zero configuration
    robot.update(q, dq)
    
    print("Computing Forward Kinematics for Hand Positions:")
    print()
    
    # Hand offsets (at wrist center)
    hand_r_offset = themis_const.hand_r  # [0, 0, 0]
    hand_l_offset = themis_const.hand_l  # [0, 0, 0]

    # elbow offsets 
    elbow_r_offset = np.array([0.0, -0.16, 0.0], dtype=np.float32)
    elbow_l_offset = np.array([0.0, 0.16, 0.0], dtype=np.float32)
    
    # Compute FK hands with timing
    print("=" * 60)
    print("PERFORMANCE MEASUREMENTS")
    print("=" * 60)
    
    t_start = time.perf_counter()
    Xhand_l = Xtrans(hand_l_offset)
    Xhand_r = Xtrans(hand_r_offset)
    Xelbow_l = Xtrans(elbow_l_offset)
    Xelbow_r = Xtrans(elbow_r_offset)
    t_xtrans = time.perf_counter() - t_start
    print(f"Xtrans computation time: {t_xtrans*1000:.4f} ms")

    t_start = time.perf_counter()
    x_hand_r, _ = robot.compute_forward_kinematics(23, hand_r_offset)
    x_elbow_r, _ = robot.compute_forward_kinematics(20, elbow_r_offset)
    t_fk_right = time.perf_counter() - t_start
    print(f"Right hand FK time: {t_fk_right*1000:.4f} ms")
    
    print(f"Right Hand (link index 23):")
    print(f"  Position: {x_hand_r}")
    
    t_start = time.perf_counter()
    x_hand_l, _ = robot.compute_forward_kinematics(30, hand_l_offset)
    x_elbow_l, _ = robot.compute_forward_kinematics(20, elbow_l_offset)
    t_fk_left = time.perf_counter() - t_start
    print(f"Left hand FK time: {t_fk_left*1000:.4f} ms")
    
    print(f"Left Hand (link index 30):")
    print(f"  Position: {x_hand_l}")
    
    t_start = time.perf_counter()
    J_hand_r = robot.compute_body_jacobian(23, Xhand_r)
    J_elbow_r = robot.compute_body_jacobian(20, Xelbow_r)
    t_jac_right = time.perf_counter() - t_start
    print(f"Right hand Jacobian time: {t_jac_right*1000:.4f} ms")
    
    t_start = time.perf_counter()
    J_hand_l = robot.compute_body_jacobian(30, Xhand_l)
    J_elbow_l = robot.compute_body_jacobian(20, Xelbow_l)
    t_jac_left = time.perf_counter() - t_start
    print(f"Left hand Jacobian time: {t_jac_left*1000:.4f} ms")
    
    print()
    print(f"Total FK computation time: {(t_fk_right + t_fk_left)*1000:.4f} ms")
    print(f"Total Jacobian computation time: {(t_jac_left + t_jac_right)*1000:.4f} ms")
    print(f"Total kinematics time: {(t_xtrans + t_fk_right + t_fk_left + t_jac_left + t_jac_right)*1000:.4f} ms")
    print()
    print("=" * 60)
    print()

    # task definitions:
    x_hand_l_des = np.array([0.0, 0.0, 0.0,  0.0, 0.40, -0.20], dtype=np.float32)
    x_hand_r_des = np.array([0.0, 0.0, 0.0,  0.0, -0.40, -0.20], dtype=np.float32)
    x_elbow_l_des = np.array([0.0, 0.0, 0.0,  -0.20, 0.20, 0.0], dtype=np.float32)
    x_elbow_r_des = np.array([0.0, 0.0, 0.0,  -0.20, -0.20, 0.0], dtype=np.float32)
    com_des = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) #pos rpy

    # Iterative IK solve
    num_ik_iterations = 50  # Modifiable iteration number
    print(f"Solving Inverse Kinematics with {num_ik_iterations} iterations...")
    
    t_start = time.perf_counter()
    q_current = q.copy()
    dq_current = dq.copy()
    
    # Store all solutions for animation
    q_trajectory = [q_current.copy()]  # Include initial configuration
    hand_positions_trajectory = []
    
    # Track best iteration based on MSRE
    best_iter = 0
    best_msre = float('inf')
    best_q = q_current.copy()
    best_dq = dq_current.copy()
    msre_history = []
    
    for ik_iter in range(num_ik_iterations):
        # Update robot state with current configuration
        robot.update(q_current, dq_current)
        
        # Compute FK for hands at current configuration
        x_hand_r, _ = robot.compute_forward_kinematics(23, hand_r_offset)
        x_hand_l, _ = robot.compute_forward_kinematics(30, hand_l_offset)
        x_elbow_r, _ = robot.compute_forward_kinematics(20, elbow_r_offset)
        x_elbow_l, _ = robot.compute_forward_kinematics(27, elbow_l_offset)
        
        # Store hand and elbow positions for this iteration
        hand_positions_trajectory.append({
            'right': x_hand_r[3:6].copy(),
            'left': x_hand_l[3:6].copy(),
            'right_des': x_hand_r_des[3:6].copy(),
            'left_des': x_hand_l_des[3:6].copy(),
            'elbow_right': x_elbow_r[3:6].copy(),
            'elbow_left': x_elbow_l[3:6].copy(),
            'elbow_right_des': x_elbow_r_des[3:6].copy(),
            'elbow_left_des': x_elbow_l_des[3:6].copy()
        })
        
        # Compute Jacobians at current configuration
        J_hand_r = robot.compute_body_jacobian(23, Xhand_r)
        J_hand_l = robot.compute_body_jacobian(30, Xhand_l)
        J_elbow_r = robot.compute_body_jacobian(20, Xelbow_r)
        J_elbow_l = robot.compute_body_jacobian(27, Xelbow_l)
        
        # Solve IK with updated hand positions and Jacobians
        q_des, dq_des = robot.update_task_space_command_with_constraints(
            x_elbow_l_des, x_elbow_r_des,
            x_elbow_l, x_elbow_r,
            x_hand_l_des, x_hand_r_des,
            x_hand_l, x_hand_r,
            J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
            com_des
        )
        
        # Store this iteration's solution
        q_trajectory.append(q_des.copy())
        
        # Compute error for convergence check (MSRE for hands and elbows)
        error_hand_r = np.linalg.norm(x_hand_r_des[3:6] - x_hand_r[3:6])**2
        error_hand_l = np.linalg.norm(x_hand_l_des[3:6] - x_hand_l[3:6])**2
        error_elbow_r = np.linalg.norm(x_elbow_r_des[3:6] - x_elbow_r[3:6])**2
        error_elbow_l = np.linalg.norm(x_elbow_l_des[3:6] - x_elbow_l[3:6])**2
        
        # Combined MSRE (mean squared root error)
        msre = np.sqrt((error_hand_r + error_hand_l + error_elbow_r + error_elbow_l) / 4.0)
        msre_history.append(msre)
        
        # Track best iteration
        if msre < best_msre:
            best_msre = msre
            best_iter = ik_iter
            best_q = q_des.copy()
            best_dq = dq_des.copy()
        
        print(f"  Iter {ik_iter+1}: Hand err (R={np.sqrt(error_hand_r):.6f}, L={np.sqrt(error_hand_l):.6f}), "
              f"Elbow err (R={np.sqrt(error_elbow_r):.6f}, L={np.sqrt(error_elbow_l):.6f}), MSRE={msre:.6f}")
        
        # Update current configuration for next iteration
        q_current = q_des.copy()
        dq_current = dq_des.copy()
    
    # Add final hand and elbow positions after last iteration
    robot.update(q_des, dq_des)
    x_hand_r_fk, _ = robot.compute_forward_kinematics(23, hand_r_offset)
    x_hand_l_fk, _ = robot.compute_forward_kinematics(30, hand_l_offset)
    x_elbow_r_fk, _ = robot.compute_forward_kinematics(20, elbow_r_offset)
    x_elbow_l_fk, _ = robot.compute_forward_kinematics(27, elbow_l_offset)
    hand_positions_trajectory.append({
        'right': x_hand_r_fk[3:6].copy(),
        'left': x_hand_l_fk[3:6].copy(),
        'right_des': x_hand_r_des[3:6].copy(),
        'left_des': x_hand_l_des[3:6].copy(),
        'elbow_right': x_elbow_r_fk[3:6].copy(),
        'elbow_left': x_elbow_l_fk[3:6].copy(),
        'elbow_right_des': x_elbow_r_des[3:6].copy(),
        'elbow_left_des': x_elbow_l_des[3:6].copy()
    })
    
    t_ik = time.perf_counter() - t_start

    # Print best iteration info
    print()
    print(f"Best iteration: {best_iter + 1} with MSRE = {best_msre:.6f}")
    
    # Use best solution instead of last iteration
    q_des = best_q
    dq_des = best_dq

    # check results using best solution
    robot.update(q_des, dq_des)
    x_hand_r_fk, _ = robot.compute_forward_kinematics(23, hand_r_offset)
    x_hand_l_fk, _ = robot.compute_forward_kinematics(30, hand_l_offset)
    x_elbow_r_fk, _ = robot.compute_forward_kinematics(20, elbow_r_offset)
    x_elbow_l_fk, _ = robot.compute_forward_kinematics(27, elbow_l_offset)
    print("IK Forward Kinematics Check (Best Iteration):")
    print(f"  Right Hand FK: {x_hand_r_fk[3:6]}  (desired: {x_hand_r_des[3:6]})")
    print(f"  Left Hand FK:  {x_hand_l_fk[3:6]}  (desired: {x_hand_l_des[3:6]})")
    print(f"  Right Elbow FK: {x_elbow_r_fk[3:6]}  (desired: {x_elbow_r_des[3:6]})")
    print(f"  Left Elbow FK:  {x_elbow_l_fk[3:6]}  (desired: {x_elbow_l_des[3:6]})")
    
    #print results
    print()
    print("Inverse Kinematics Result at Zero Configuration:")
    print(f"  Desired Joint Positions: {q_des[6:]}")
    print(f"  IK solver time: {t_ik*1000:.4f} ms")
    print()
    print("=" * 60)
    print("TOTAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"FK + Jacobian: {(t_xtrans + t_fk_right + t_fk_left + t_jac_left + t_jac_right)*1000:.4f} ms")
    print(f"IK solver:     {t_ik*1000:.4f} ms")
    print(f"Total time:    {(t_xtrans + t_fk_right + t_fk_left + t_jac_left + t_jac_right + t_ik)*1000:.4f} ms")
    print("=" * 60)
    
    # Return the IK solution, hand positions, and trajectories for visualization
    x_elbow_r_fk, _ = robot.compute_forward_kinematics(20, elbow_r_offset)
    x_elbow_l_fk, _ = robot.compute_forward_kinematics(27, elbow_l_offset)
    hand_positions = {
        'right': x_hand_r_fk[3:6],  # actual position [x, y, z]
        'left': x_hand_l_fk[3:6],   # actual position [x, y, z]
        'right_des': x_hand_r_des[3:6],  # desired position [x, y, z]
        'left_des': x_hand_l_des[3:6],   # desired position [x, y, z]
        'elbow_right': x_elbow_r_fk[3:6],
        'elbow_left': x_elbow_l_fk[3:6],
        'elbow_right_des': x_elbow_r_des[3:6],
        'elbow_left_des': x_elbow_l_des[3:6]
    }
    
    # Return trajectories for animation
    return {
        'q_des': q_des,  # Best iteration solution
        'dq_des': dq_des,
        'hand_positions': hand_positions,
        'q_trajectory': q_trajectory,
        'hand_positions_trajectory': hand_positions_trajectory,
        'best_iter': best_iter,
        'best_msre': best_msre,
        'msre_history': msre_history
    }


def translate_joint_angles(q_config, invert_joints=None, verbose=False):
    """
    Translate joint angles by inverting specified joints
    
    Args:
        q_config: (DOF,) configuration vector [base_pos(3), base_rot(3), joint_pos(28)]
        invert_joints: list of joint indices (0-27) to invert, or None for default arm inversions
        verbose: bool, whether to print translation info
    
    Returns:
        q_translated: (DOF,) translated configuration
    """
    q_translated = q_config.copy()
    
    if invert_joints is None:
        
        invert_joints = [
            12,  # SHOULDER_ROLL_R
            14,
            16,
            19  # SHOULDER_ROLL_L
        ]
    
    # Invert specified joint angles
    for joint_idx in invert_joints:
        if 0 <= joint_idx < 28:  # Valid joint index
            q_translated[6 + joint_idx] *= -1.0
    
    if verbose:
        print()
        print("=" * 60)
        print("JOINT ANGLE TRANSLATION")
        print("=" * 60)
        print(f"Inverting joints: {invert_joints}")
        print(f"Original arm joints (first 14): {q_config[6:20]}")
        print(f"Translated arm joints (first 14): {q_translated[6:20]}")
        print("=" * 60)
    
    return q_translated


def visualize_robot_pose(q_config, duration=5.0, xml_path=None, hand_positions=None):
    """
    Visualize the robot configuration in MuJoCo
    
    Args:
        q_config: (DOF,) configuration vector [base_pos(3), base_rot(3), joint_pos(28)]
        duration: float, visualization duration in seconds
        xml_path: str, path to robot XML file (default: Themis TH02-A7.xml)
        hand_positions: dict with 'left' and 'right' keys, each containing (3,) position arrays
    """
    # Import mujoco only when visualization is needed
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco not installed. Skipping visualization.")
        print("Install with: pip install mujoco")
        return
    
    if xml_path is None:
        xml_path = "/home/junhengl/g1_ctrl_py/g1_ctrl/westwood_robots/TH02-A7.xml"
    
    print()
    print("=" * 60)
    print("MUJOCO VISUALIZATION")
    print("=" * 60)
    print(f"Loading robot model from: {xml_path}")
    
    try:
        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Set robot configuration
        # MuJoCo expects: [qpos_base(7=xyz+quat), joint_positions(28)]
        # Our format: [base_pos(3), base_rot(3), joint_pos(28)]
        
        # Base position
        data.qpos[0:3] = q_config[0:3]
        
        # Base orientation - convert from euler angles to quaternion
        roll, pitch, yaw = q_config[3], q_config[4], q_config[5]
        quat_wxyz = euler_to_quat(np.array([roll, pitch, yaw], dtype=np.float32))
        # MuJoCo uses wxyz format
        data.qpos[3:7] = quat_wxyz
        
        # Joint positions (28 actuated joints)
        data.qpos[7:] = q_config[6:]
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        print(f"Robot configuration set successfully")
        print(f"Base position: {data.qpos[0:3]}")
        print(f"Base quaternion (wxyz): {data.qpos[3:7]}")
        print(f"Joint positions (first 10): {data.qpos[7:17]}")
        if hand_positions is not None:
            print(f"Right hand target: {hand_positions.get('right', 'N/A')}")
            print(f"Left hand target: {hand_positions.get('left', 'N/A')}")
        print()
        print(f"Launching viewer (will stay open until you close the window)...")
        print("=" * 60)
        
        # Launch interactive viewer (kinematic only, no dynamics)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Wait indefinitely until user closes the viewer
            while viewer.is_running():
                # Keep kinematics updated without running dynamics
                # No mj_step - just maintain the pose
                
                # Add hand position markers as spheres
                if hand_positions is not None:
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
                    
                    viewer.user_scn.ngeom = geom_idx
                    
                viewer.sync()
                time.sleep(0.05)  # 20 Hz update rate
        
        print("Viewer closed.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Make sure the XML path is correct and mujoco is properly installed.")


if __name__ == "__main__":
    result = test_themis_fk()
    
    # Visualize the IK solution
    if result is not None:
        # Extract results
        q_des = result['q_des']
        dq_des = result['dq_des']
        hand_positions = result['hand_positions']
        q_trajectory = result['q_trajectory']
        hand_positions_trajectory = result['hand_positions_trajectory']
        best_iter = result['best_iter']
        best_msre = result['best_msre']
        
        # Apply joint angle translation to best configuration
        q_best_translated = translate_joint_angles(q_des, invert_joints=[12, 14, 16, 19])
        
        # Get hand positions for best iteration
        best_hand_positions = hand_positions_trajectory[best_iter]
        
        print()
        print(f"Best iteration: {best_iter + 1} with MSRE = {best_msre:.6f}")
        print(f"Actual hand positions: Right at {hand_positions['right']}, Left at {hand_positions['left']}")
        print(f"Desired hand positions: Right at {hand_positions['right_des']}, Left at {hand_positions['left_des']}")
        print(f"Actual elbow positions: Right at {hand_positions['elbow_right']}, Left at {hand_positions['elbow_left']}")
        print(f"Desired elbow positions: Right at {hand_positions['elbow_right_des']}, Left at {hand_positions['elbow_left_des']}")
        
        # Import and use the visualization module
        from visualization import visualize_single_pose
        
        print()
        print("Launching best iteration visualization...")
        visualize_single_pose(q_best_translated, hand_positions=hand_positions)
