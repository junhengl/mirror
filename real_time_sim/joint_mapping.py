"""
Joint mapping between MuJoCo simulation convention and KinDynLib kinematics convention.

The robot kinematics tree (KinDynLib) may define joint directions and zero-pose
offsets differently from the MuJoCo URDF/XML. This module provides a linear
transform to convert between the two:

    Forward  (MuJoCo → KinDynLib):  q_kin  = sign * q_muj + offset
    Reverse  (KinDynLib → MuJoCo):  q_muj  = sign * (q_kin - offset)

where `sign` is a diagonal vector of +1 / -1 (joint direction flips) and
`offset` accounts for the difference in zero pose.

Sign also applies to velocities and torques:
    dq_kin  = sign * dq_muj
    tau_muj = sign * tau_kin     (torques flip the same way as position)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class JointMappingConfig:
    """
    Configuration for the MuJoCo ↔ KinDynLib joint mapping.

    sign:   (28,) array of +1.0 or -1.0  — joint direction flips
    offset: (28,) array of floats        — zero-pose offset  (radians)

    The arrays are ordered as the 28 actuated joints:
        [right_leg(6), left_leg(6), right_arm(7), left_arm(7), head(2)]
    """
    sign: np.ndarray = field(default_factory=lambda: np.ones(28, dtype=np.float64))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float64))


class JointMapping:
    """
    Stateless linear mapping  q_kin = sign * q_muj + offset.

    Usage
    -----
    mapping = JointMapping(config)

    # In controller loop:
    q_kin   = mapping.forward_q(feedback.q)      # MuJoCo → KinDynLib
    dq_kin  = mapping.forward_dq(feedback.dq)    # velocities (no offset)
    tau_muj = mapping.reverse_torque(tau_kin)     # KinDynLib → MuJoCo

    # If you need to map a desired q from KinDynLib back to MuJoCo:
    q_muj   = mapping.reverse_q(q_des_kin)
    """

    def __init__(self, config: JointMappingConfig):
        self.sign = config.sign.copy()
        self.offset = config.offset.copy()
        assert self.sign.shape == (28,), f"sign must be (28,), got {self.sign.shape}"
        assert self.offset.shape == (28,), f"offset must be (28,), got {self.offset.shape}"

    # ---- Forward: MuJoCo → KinDynLib ----

    def forward_q(self, q_muj: np.ndarray) -> np.ndarray:
        """Map joint positions from MuJoCo to KinDynLib convention."""
        return self.sign * q_muj + self.offset

    def forward_dq(self, dq_muj: np.ndarray) -> np.ndarray:
        """Map joint velocities from MuJoCo to KinDynLib convention."""
        return self.sign * dq_muj

    # ---- Reverse: KinDynLib → MuJoCo ----

    def reverse_q(self, q_kin: np.ndarray) -> np.ndarray:
        """Map joint positions from KinDynLib to MuJoCo convention."""
        return self.sign * (q_kin - self.offset)

    def reverse_dq(self, dq_kin: np.ndarray) -> np.ndarray:
        """Map joint velocities from KinDynLib to MuJoCo convention."""
        return self.sign * dq_kin

    def reverse_torque(self, tau_kin: np.ndarray) -> np.ndarray:
        """Map torques from KinDynLib to MuJoCo convention.

        Torque direction flips with joint direction:  tau_muj = sign * tau_kin
        """
        return self.sign * tau_kin

    def forward_torque(self, tau_muj: np.ndarray) -> np.ndarray:
        """Map torques from MuJoCo to KinDynLib convention."""
        return self.sign * tau_muj
