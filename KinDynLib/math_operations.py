import numpy as np

def apply_transform(X, v):
    """Apply spatial transform X to spatial vector v
    Args:
        X: (6, 6) spatial transform matrix
        v: (6,) spatial vector
    Returns:
        result: (6,) transformed vector
    """
    return X @ v

def apply_transpose_transform(X, f):
    """Apply transpose of spatial transform X to force f
    Args:
        X: (6, 6)
        f: (6,)
    Returns:
        result: (6,)
    """
    return X.T @ f

def AtBA(A, B):
    """Compute A^T * B * A
    Args:
        A: (6, 6)
        B: (6, 6)
    Returns:
        result: (6, 6)
    """
    temp = B @ A
    return A.T @ temp

def compute_spatial_rotm(rotation, theta):
    """Compute spatial rotation matrix
    Args:
        rotation: str - 'Rx', 'Ry', or 'Rz'
        theta: scalar
    Returns:
        R: (6, 6) spatial rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.zeros((6, 6), dtype=np.float32)
    
    if rotation == "Rx":
        R[0, 0] = 1.0
        R[1, 1] = c; R[1, 2] = -s
        R[2, 1] = s; R[2, 2] = c
        R[3, 3] = 1.0
        R[4, 4] = c; R[4, 5] = -s
        R[5, 4] = s; R[5, 5] = c
    elif rotation == "Ry":
        R[0, 0] = c; R[0, 2] = s
        R[1, 1] = 1.0
        R[2, 0] = -s; R[2, 2] = c
        R[3, 3] = c; R[3, 5] = s
        R[4, 4] = 1.0
        R[5, 3] = -s; R[5, 5] = c
    elif rotation == "Rz":
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        R[2, 2] = 1.0
        R[3, 3] = c; R[3, 4] = -s
        R[4, 3] = s; R[4, 4] = c
        R[5, 5] = 1.0
    else:
        R = np.eye(6, dtype=np.float32)
    
    return R

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix (ZYX convention)
    Args:
        roll, pitch, yaw: scalars
    Returns:
        R: (3, 3)
    """
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)
    
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    
    return R

def euler_rate_mapping_matrix(roll, pitch, yaw):
    """Compute Euler rate to angular velocity mapping matrix
    Args:
        roll, pitch, yaw: scalars
    Returns:
        Tinv: (3, 3)
    """
    sr = np.sin(roll); cr = np.cos(roll)
    st = np.sin(pitch); ct = np.cos(pitch)
    
    if np.abs(ct) < 1e-6:
        raise RuntimeError("Pitch angle too close to +/- 90 degrees (gimbal lock).")
    
    Tinv = np.zeros((3, 3), dtype=np.float32)
    Tinv[0, 0] = 1.0
    Tinv[0, 1] = sr * st / ct
    Tinv[0, 2] = cr * st / ct
    Tinv[1, 1] = cr
    Tinv[1, 2] = -sr
    Tinv[2, 1] = sr / ct
    Tinv[2, 2] = cr / ct
    
    return Tinv

def matrix_log_rotm(R):
    """Compute matrix logarithm of rotation matrix (orientation error)
    Args:
        R: (3, 3)
    Returns:
        error_R: (3,)
    """
    tmp = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = np.arccos(np.clip(tmp, -1.0, 1.0)) + 1e-5
    error_R = np.array([R[2, 1], R[0, 2], R[1, 0]], dtype=np.float32)
    error_R = error_R * (theta / (2 * np.sin(theta)))
    
    return error_R

def wrap_pi(a):
    """Wrap angle to (-pi, pi]
    Args:
        a: scalar
    Returns:
        wrapped: scalar
    """
    return np.remainder(a + np.pi, 2.0 * np.pi) - np.pi

def rpy_from_rot_zyx(R):
    """Extract roll, pitch, yaw from rotation matrix (ZYX convention)
    Args:
        R: (3, 3)
    Returns:
        roll, pitch, yaw: scalars
    """
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    pitch = np.arctan2(sy, cy)
    
    if cy > 1e-8:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    
    roll = wrap_pi(roll)
    pitch = wrap_pi(pitch)
    yaw = wrap_pi(yaw)
    
    return roll, pitch, yaw
