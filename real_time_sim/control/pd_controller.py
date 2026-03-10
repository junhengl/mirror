"""
PD Controller for joint-level torque control.

Computes torques to track desired joint positions:
    tau = Kp * (q_des - q) + Kd * (dq_des - dq)
"""

import numpy as np
from typing import Tuple

from ..config import ControlConfig, PipelineConfig


class PDController:
    """
    Joint-level PD controller with per-joint gains.
    
    Supports different gains for different joint groups:
    - Legs: Higher gains for stance stability
    - Arms: Lower gains for compliant motion
    - Head: Soft gains
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ctrl_config = config.control
        
        # Build gain arrays (28 joints)
        self.kp = np.zeros(28, dtype=np.float64)
        self.kd = np.zeros(28, dtype=np.float64)
        self.torque_limits = np.zeros(28, dtype=np.float64)
        
        # Right leg (0-5): hip roll/pitch/yaw (0-2), knee pitch (3), ankle pitch/roll (4-5)
        self.kp[0:4] = self.ctrl_config.leg_gains[0]
        self.kd[0:4] = self.ctrl_config.leg_gains[1]
        self.kp[4:6] = self.ctrl_config.ankle_gains[0]  # Reduced gains for ankle
        self.kd[4:6] = self.ctrl_config.ankle_gains[1]
        self.torque_limits[0:6] = self.ctrl_config.leg_torque_limit
        
        # Left leg (6-11): hip roll/pitch/yaw (6-8), knee pitch (9), ankle pitch/roll (10-11)
        self.kp[6:10] = self.ctrl_config.leg_gains[0]
        self.kd[6:10] = self.ctrl_config.leg_gains[1]
        self.kp[10:12] = self.ctrl_config.ankle_gains[0]  # Reduced gains for ankle
        self.kd[10:12] = self.ctrl_config.ankle_gains[1]
        self.torque_limits[6:12] = self.ctrl_config.leg_torque_limit
        
        # Right arm (12-18)
        self.kp[12:19] = self.ctrl_config.arm_gains[0]
        self.kd[12:19] = self.ctrl_config.arm_gains[1]
        self.torque_limits[12:19] = self.ctrl_config.arm_torque_limit
        
        # Left arm (19-25)
        self.kp[19:26] = self.ctrl_config.arm_gains[0]
        self.kd[19:26] = self.ctrl_config.arm_gains[1]
        self.torque_limits[19:26] = self.ctrl_config.arm_torque_limit
        
        # Head (26-27)
        self.kp[26:28] = self.ctrl_config.head_gains[0]
        self.kd[26:28] = self.ctrl_config.head_gains[1]
        self.torque_limits[26:28] = self.ctrl_config.head_torque_limit
        
        print(f"[PD Controller] Initialized with gains:")
        print(f"  Legs (hip/knee): Kp={self.ctrl_config.leg_gains[0]}, Kd={self.ctrl_config.leg_gains[1]}")
        print(f"  Ankles: Kp={self.ctrl_config.ankle_gains[0]}, Kd={self.ctrl_config.ankle_gains[1]}")
        print(f"  Arms: Kp={self.ctrl_config.arm_gains[0]}, Kd={self.ctrl_config.arm_gains[1]}")
        print(f"  Head: Kp={self.ctrl_config.head_gains[0]}, Kd={self.ctrl_config.head_gains[1]}")
    
    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute PD control torques.
        
        Args:
            q: Current joint positions (28,)
            dq: Current joint velocities (28,)
            q_des: Desired joint positions (28,)
            dq_des: Desired joint velocities (28,), defaults to zero
            
        Returns:
            tau: Joint torques (28,)
        """
        if dq_des is None:
            dq_des = np.zeros(28, dtype=np.float64)
        
        # Position error
        q_error = q_des - q

        # print(f"q_des: {q_des}")
        # print(f"q: {q}")
        
        # Handle angle wrapping for error (keep error in [-pi, pi])
        # q_error = np.arctan2(np.sin(q_error), np.cos(q_error))
        
        # Velocity error
        dq_error = dq_des - dq
        
        # PD control law
        tau = self.kp * q_error + self.kd * dq_error
        
        # Apply torque limits
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        
        return tau
    
    def set_arm_gains(self, kp: float, kd: float):
        """Update arm gains dynamically."""
        self.kp[12:19] = kp
        self.kp[19:26] = kp
        self.kd[12:19] = kd
        self.kd[19:26] = kd
        
    def set_leg_gains(self, kp: float, kd: float):
        """Update leg gains dynamically."""
        self.kp[0:12] = kp
        self.kd[0:12] = kd
        
    def get_tracking_error(
        self,
        q: np.ndarray,
        q_des: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Get tracking error statistics.
        
        Returns:
            (total_error, arm_error, leg_error) in radians (RMS)
        """
        q_error = q_des - q
        q_error = np.arctan2(np.sin(q_error), np.cos(q_error))
        
        total_error = np.sqrt(np.mean(q_error**2))
        arm_error = np.sqrt(np.mean(q_error[12:26]**2))
        leg_error = np.sqrt(np.mean(q_error[0:12]**2))
        
        return total_error, arm_error, leg_error
