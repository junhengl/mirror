import numpy as np
import time
from robot_const import *
from robot_dynamics import Robot
from dynamics_lib import Xtrans


class SolverInput:
    """Input structure for trajectory solver"""
    def __init__(self):
        self.base0_pos = np.zeros(3, dtype=np.float32)
        self.base0_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz
        self.joint0_pos = np.zeros(29, dtype=np.float32)
        self.target_step_length = np.zeros(3, dtype=np.float32)
        self.target_rpy = np.zeros(3, dtype=np.float32)
        self.target_step_height = 0.0
        self.swing_foot_wpos = np.zeros(3, dtype=np.float32)
        self.anchor_foot_wpos = np.zeros(3, dtype=np.float32)
        self.foot = 0  # 0 or 1
        self.step = 0  # current step index


class Trajectory:
    """Trajectory storage with per-frame link poses"""
    def __init__(self, T, dt=0.001):
        self.T = T
        self.dt = dt
        # Main trajectory: (T, DOF+1) stores [x, y, z, qw, qx, qy, qz, joints...]
        self.data = np.zeros((T, DOF + 1), dtype=np.float32)
        # Link poses: list of (DOF-5, 6) arrays, one per timestep
        self.linkPose = [np.zeros((DOF-5, 6), dtype=np.float32) for _ in range(T)]
    
    def get_frame(self, t):
        """Get frame at timestep t"""
        return self.data[t]
    
    def set_frame(self, t, frame):
        """Set frame at timestep t"""
        self.data[t] = frame


def quat_to_euler(quat):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)
    Args:
        quat: (4,) - quaternion [w, x, y, z]
    Returns:
        euler: (3,) - [roll, pitch, yaw]
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


def euler_to_quat(euler):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)
    Args:
        euler: (3,) - [roll, pitch, yaw]
    Returns:
        quat: (4,) - [w, x, y, z]
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z], dtype=np.float32)


def run_kinWBC(solver_input: SolverInput, traj: Trajectory, t: int, T: int):
    """Run kinematic whole body control for one timestep
    Args:
        solver_input: SolverInput object with initial conditions
        traj: Trajectory object to fill
        t: current timestep
        T: total timesteps
    """
    # Initialize robot
    robot = Robot()
    
    # Get current and previous frames
    frame = traj.get_frame(t)
    frame_prev = traj.get_frame(t-1) if t > 0 else None
    
    # Initialize q and dq
    q = np.zeros(DOF, dtype=np.float32)
    dq = np.zeros(DOF, dtype=np.float32)
    
    if t == 0:
        # Initialize from input
        q[:3] = solver_input.base0_pos
        euler = quat_to_euler(solver_input.base0_quat)
        q[3:6] = euler
        q[6:] = solver_input.joint0_pos
    else:
        # Update from previous frame
        q[:3] = frame_prev[:3]  # base position
        # Convert quaternion to Euler angles
        quat_prev = frame_prev[3:7]  # [qw, qx, qy, qz]
        euler = quat_to_euler(quat_prev)
        q[3:6] = euler  # base orientation as Euler
        q[6:] = frame_prev[7:]  # joint positions
    
    # Update robot state
    robot.update(q, dq)
    
    # Compute contact Jacobians
    contact_l_offset = contact_l
    contact_r_offset = contact_r
    hand_l_offset = hand_l
    hand_r_offset = hand_r
    
    Xend_l = Xtrans(contact_l_offset)
    Xend_r = Xtrans(contact_r_offset)
    Xhand_l = Xtrans(hand_l_offset)
    Xhand_r = Xtrans(hand_r_offset)
    
    J_l = robot.compute_body_jacobian(10, Xend_l)
    J_r = robot.compute_body_jacobian(16, Xend_r)
    J_hand_l = robot.compute_body_jacobian(26, Xhand_l)
    J_hand_r = robot.compute_body_jacobian(33, Xhand_r)
    
    # Compute forward kinematics
    x_foot_l, _ = robot.compute_forward_kinematics(10, contact_l_offset)
    x_foot_r, _ = robot.compute_forward_kinematics(16, contact_r_offset)
    x_hand_l, _ = robot.compute_forward_kinematics(26, hand_l_offset)
    x_hand_r, _ = robot.compute_forward_kinematics(33, hand_r_offset)
    
    # Desired trajectories
    com_des = np.zeros(6, dtype=np.float32)  # pos, rpy
    com_des[0] = solver_input.base0_pos[0] + solver_input.target_step_length[0] * t / T
    com_des[2] = 0.75  # desired COM height
    
    x_foot_l_des = np.zeros(6, dtype=np.float32)
    x_foot_r_des = np.zeros(6, dtype=np.float32)
    x_hand_l_des = np.zeros(6, dtype=np.float32)
    x_hand_r_des = np.zeros(6, dtype=np.float32)
    
    foot_contact = int(solver_input.foot % 2)
    if foot_contact == 0:
        # Right foot swing
        x_foot_l_des[3:] = solver_input.swing_foot_wpos + \
                          solver_input.target_step_length * 2 * t / T
        x_foot_l_des[5] = solver_input.target_step_height * np.sin(np.pi * t / T)
        x_foot_r_des[3:] = solver_input.anchor_foot_wpos
        
        # Hand trajectories
        x_hand_r_des[3] = com_des[0] - solver_input.target_step_length[0] / 4 + \
                         solver_input.target_step_length[0] * t / T / 2
        x_hand_r_des[4] = com_des[1] - 0.2
        x_hand_r_des[5] = com_des[2] - 0.1
        
        x_hand_l_des[3] = com_des[0] + solver_input.target_step_length[0] / 4 - \
                         solver_input.target_step_length[0] * t / T / 2
        x_hand_l_des[4] = com_des[1] + 0.2
        x_hand_l_des[5] = com_des[2] - 0.1
    else:
        # Left foot swing
        x_foot_r_des[3:] = solver_input.swing_foot_wpos + \
                          solver_input.target_step_length * 2 * t / T
        x_foot_r_des[5] = solver_input.target_step_height * np.sin(np.pi * t / T)
        x_foot_l_des[3:] = solver_input.anchor_foot_wpos
        
        x_hand_l_des[3] = com_des[0] - solver_input.target_step_length[0] / 4 + \
                         solver_input.target_step_length[0] * t / T / 2
        x_hand_l_des[4] = com_des[1] + 0.2
        x_hand_l_des[5] = com_des[2] - 0.1
        
        x_hand_r_des[3] = com_des[0] + solver_input.target_step_length[0] / 4 - \
                         solver_input.target_step_length[0] * t / T / 2
        x_hand_r_des[4] = com_des[1] - 0.2
        x_hand_r_des[5] = com_des[2] - 0.1
    
    # Solve IK using weighted IK solver
    q_des, dq_des = robot.update_task_space_command_with_constraints(
        x_foot_l_des, x_foot_r_des,
        x_foot_l, x_foot_r,
        x_hand_l_des, x_hand_r_des,
        x_hand_l, x_hand_r,
        J_l, J_r, J_hand_l, J_hand_r,
        com_des
    )
    
    # Store result in trajectory
    frame[3:7] = euler_to_quat(q_des[3:6])  # convert back to quat
    frame[:3] = q_des[:3]
    frame[7:] = q_des[6:]
    traj.set_frame(t, frame)
    
    return traj


def run_kinWBC_batch(solver_inputs, T=20, dt=0.001):
    """Run kinWBC for a batch of inputs sequentially
    Args:
        solver_inputs: list of SolverInput objects
        T: number of timesteps
        dt: time step
    Returns:
        trajectories: list of Trajectory objects
    """
    trajectories = []
    
    for solver_input in solver_inputs:
        traj = Trajectory(T, dt)
        for t in range(T):
            traj = run_kinWBC(solver_input, traj, t, T)
        trajectories.append(traj)
    
    return trajectories
