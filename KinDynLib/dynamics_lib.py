import numpy as np

def crm(v):
    """Compute cross product matrix for spatial motion vector
    Args:
        v: (6,) spatial vector
    Returns:
        m: (6, 6) cross product matrix
    """
    m = np.zeros((6, 6), dtype=np.float32)
    m[0, 1] = -v[2]; m[0, 2] = v[1]
    m[1, 0] = v[2]; m[1, 2] = -v[0]
    m[2, 0] = -v[1]; m[2, 1] = v[0]
    
    m[3, 1] = -v[5]; m[3, 2] = v[4]
    m[4, 0] = v[5]; m[4, 2] = -v[3]
    m[5, 0] = -v[4]; m[5, 1] = v[3]
    
    m[3, 4] = -v[2]; m[3, 5] = v[1]
    m[4, 3] = v[2]; m[4, 5] = -v[0]
    m[5, 3] = -v[1]; m[5, 4] = v[0]
    return m

def crf(v):
    """Compute cross product matrix for spatial force vector
    Args:
        v: (6,)
    Returns:
        m: (6, 6)
    """
    return -crm(v).T

def McI(mass, com, I_com):
    """Compute spatial inertia matrix
    Args:
        mass: scalar
        com: (3,) - center of mass
        I_com: (3, 3) - inertia at CoM
    Returns:
        result: (6, 6) spatial inertia matrix
    """
    result = np.zeros((6, 6), dtype=np.float32)
    
    # Skew-symmetric matrix for com
    C = np.zeros((3, 3), dtype=np.float32)
    C[0, 1] = -com[2]; C[0, 2] = com[1]
    C[1, 0] = com[2]; C[1, 2] = -com[0]
    C[2, 0] = -com[1]; C[2, 1] = com[0]
    
    # Top-left 3x3: I + m*C*C^T
    CCt = C @ C.T
    result[:3, :3] = I_com + mass * CCt
    
    # Top-right and bottom-left
    result[:3, 3:] = mass * C
    result[3:, :3] = mass * C.T
    
    # Bottom-right 3x3: m*I
    result[3:, 3:] = mass * np.eye(3, dtype=np.float32)
    
    return result

def Xtrans(r):
    """Compute spatial translation transform
    Args:
        r: (3,) translation vector
    Returns:
        X: (6, 6)
    """
    X = np.eye(6, dtype=np.float32)
    X[3, 1] = r[2]; X[3, 2] = -r[1]
    X[4, 0] = -r[2]; X[4, 2] = r[0]
    X[5, 0] = r[1]; X[5, 1] = -r[0]
    return X

def identity6():
    """Create 6x6 identity matrix
    Returns:
        I: (6, 6)
    """
    return np.eye(6, dtype=np.float32)

def rotx(q):
    """Rotation about X-axis (spatial)
    Args:
        q: scalar angle
    Returns:
        X: (6, 6)
    """
    X = np.eye(6, dtype=np.float32)
    c = np.cos(q); s = np.sin(q)
    X[1, 1] = c; X[1, 2] = s
    X[2, 1] = -s; X[2, 2] = c
    X[4, 4] = c; X[4, 5] = s
    X[5, 4] = -s; X[5, 5] = c
    return X

def roty(q):
    """Rotation about Y-axis (spatial)
    Args:
        q: scalar angle
    Returns:
        X: (6, 6)
    """
    X = np.eye(6, dtype=np.float32)
    c = np.cos(q); s = np.sin(q)
    X[0, 0] = c; X[0, 2] = -s
    X[2, 0] = s; X[2, 2] = c
    X[3, 3] = c; X[3, 5] = -s
    X[5, 3] = s; X[5, 5] = c
    return X

def rotz(q):
    """Rotation about Z-axis (spatial)
    Args:
        q: scalar angle
    Returns:
        X: (6, 6)
    """
    X = np.eye(6, dtype=np.float32)
    c = np.cos(q); s = np.sin(q)
    X[0, 0] = c; X[0, 1] = s
    X[1, 0] = -s; X[1, 1] = c
    X[3, 3] = c; X[3, 4] = s
    X[4, 3] = -s; X[4, 4] = c
    return X

def jcalc(joint_type, q):
    """Calculate joint transform and motion subspace
    Args:
        joint_type: str - 'Rx', 'Ry', 'Rz', 'Px', 'Py', 'Pz', 'fixed'
        q: scalar joint angle/position
    Returns:
        XJ: (6, 6) joint transform
        S: (6,) motion subspace vector
    """
    XJ = identity6()
    S = np.zeros(6, dtype=np.float32)
    
    if joint_type == "Rx":
        XJ = rotx(q)
        S[0] = 1.0
    elif joint_type == "Ry":
        XJ = roty(q)
        S[1] = 1.0
    elif joint_type in ["Rz", "R"]:
        XJ = rotz(q)
        S[2] = 1.0
    elif joint_type == "Px":
        r = np.array([q, 0.0, 0.0], dtype=np.float32)
        XJ = Xtrans(r)
        S[3] = 1.0
    elif joint_type == "Py":
        r = np.array([0.0, q, 0.0], dtype=np.float32)
        XJ = Xtrans(r)
        S[4] = 1.0
    elif joint_type in ["Pz", "P"]:
        r = np.array([0.0, 0.0, q], dtype=np.float32)
        XJ = Xtrans(r)
        S[5] = 1.0
    elif joint_type in ["fixed", "Fixed"]:
        pass  # XJ is identity, S is zeros
    
    return XJ, S

def joint_transform(joint_type, q):
    """Calculate joint transform as SE(3) using homogeneous coordinates
    Args:
        joint_type: str
        q: scalar
    Returns:
        Xj: (4, 4) homogeneous transform
    """
    Xj = np.eye(4, dtype=np.float32)
    
    if joint_type == "Rx":
        c = np.cos(q); s = np.sin(q)
        Xj[1, 1] = c; Xj[1, 2] = -s
        Xj[2, 1] = s; Xj[2, 2] = c
    elif joint_type == "Ry":
        c = np.cos(q); s = np.sin(q)
        Xj[0, 0] = c; Xj[0, 2] = s
        Xj[2, 0] = -s; Xj[2, 2] = c
    elif joint_type == "Rz":
        c = np.cos(q); s = np.sin(q)
        Xj[0, 0] = c; Xj[0, 1] = -s
        Xj[1, 0] = s; Xj[1, 1] = c
    elif joint_type == "Px":
        Xj[0, 3] = q
    elif joint_type == "Py":
        Xj[1, 3] = q
    elif joint_type == "Pz":
        Xj[2, 3] = q
    
    return Xj

def spatial_to_isometry(X):
    """Convert spatial transform to SE(3) homogeneous transform
    Args:
        X: (6, 6) spatial transform
    Returns:
        T: (4, 4) homogeneous transform
    """
    T = np.eye(4, dtype=np.float32)
    
    # Extract rotation (transpose of top-left 3x3)
    R = X[:3, :3].T
    T[:3, :3] = R
    
    # Extract translation from bottom-left 3x3
    skew_rR = -X[3:, :3]
    skew_r = R.T @ skew_rR
    
    # Recover translation from skew matrix
    r = np.array([-skew_r[1, 2], skew_r[0, 2], -skew_r[0, 1]], dtype=np.float32)
    T[:3, 3] = r
    
    return T
