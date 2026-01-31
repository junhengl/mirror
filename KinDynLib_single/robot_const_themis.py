import numpy as np

# Themis TH02-A7 Robot Constants
# Parsed from TH02-A7.xml

# Degrees of freedom
# Floating base: 6 DOF (Px, Py, Pz, Rx, Ry, Rz)
# Right leg: 6 joints (hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
# Left leg: 6 joints (hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
# Right arm: 7 joints (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw)
# Left arm: 7 joints (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw)
# Head: 2 joints (head_yaw, head_pitch)
# Total: 6 + 28 = 34 DOF

DOF = 34
NUM_LINKS = 34
GRAVITY = -9.81
WBC_dt = 0.002  # From XML timestep
N_LINKS = 29  # Number of actual rigid bodies (excluding floating base virtual links)

# QP matrix constants (for WBC if needed)
n_ddq = DOF
n_f = 12  # Contact forces (4 contact points per foot * 3 components)
n_tau = DOF
nV = n_ddq + n_f + n_tau
nC_dyn = DOF
nC_fc = 10
nC_lf = 16
nC_h = 12
nC = nC_dyn + nC_fc + nC_lf + nC_h

# Physical constants
F_max = 1000.0
F_min = 0.0
mu = 1.0  # Friction coefficient from foot contact geoms
lt = 0.17  # Foot length (approx from contact points)
lh = 0.12  # Foot width (approx from contact points)
w = 0.03  # Contact sphere size
alpha = 300.0
beta = 10.0

# Joint names (for reference)
JOINT_NAMES = [
    # Floating base (virtual)
    "root_x", "root_y", "root_z", "root_roll", "root_pitch", "root_yaw",
    # Right leg (6)
    "HIP_YAW_R", "HIP_ROLL_R", "HIP_PITCH_R", "KNEE_PITCH_R", "ANKLE_PITCH_R", "ANKLE_ROLL_R",
    # Left leg (6)
    "HIP_YAW_L", "HIP_ROLL_L", "HIP_PITCH_L", "KNEE_PITCH_L", "ANKLE_PITCH_L", "ANKLE_ROLL_L",
    # Right arm (7)
    "SHOULDER_PITCH_R", "SHOULDER_ROLL_R", "SHOULDER_YAW_R", "ELBOW_PITCH_R", "ELBOW_YAW_R", "WRIST_PITCH_R", "WRIST_YAW_R",
    # Left arm (7)
    "SHOULDER_PITCH_L", "SHOULDER_ROLL_L", "SHOULDER_YAW_L", "ELBOW_PITCH_L", "ELBOW_YAW_L", "WRIST_PITCH_L", "WRIST_YAW_L",
    # Head (2)
    "HEAD_YAW", "HEAD_PITCH"
]

# Masses (29 links - excluding floating base virtual links)
# Order: BASE, right leg (6), left leg (6), right arm (8), left arm (8)
mass = np.array([
    # BASE_LINK (computed from remaining mass if needed, placeholder for now)
    10.0,  # Base torso mass (will need to be adjusted)
    # Right leg
    0.83264,   # HIP_R
    2.78794,   # HIP_ABAD_R
    3.75384,   # FEMUR_R
    1.27716,   # TIBIA_R
    0.006946,  # ANKLE_R
    0.41628,   # FOOT_R
    # Left leg
    0.83264,   # HIP_L
    2.818,     # HIP_ABAD_L
    3.75393,   # FEMUR_L
    1.27716,   # TIBIA_L
    0.006946,  # ANKLE_L
    0.41628,   # FOOT_L
    # Right arm
    0.43611,   # UPPERSHOULDER_R
    0.35798,   # LOWERSHOULDER_R
    0.51544,   # UPPERARM_R
    0.37853,   # ELBOW_R
    0.47694,   # FOREARM_R
    0.37004,   # UPPERWRIST_R
    0.42,      # LOWERWRIST_R
    # Left arm
    0.43611,   # UPPERSHOULDER_L
    0.35798,   # LOWERSHOULDER_L
    0.51544,   # UPPERARM_L
    0.37854,   # ELBOW_L
    0.47694,   # FOREARM_L
    0.37004,   # UPPERWRIST_L
    0.42,      # LOWERWRIST_L
    # Head
    0.34044,   # NECK
    0.67367,   # HEAD
], dtype=np.float32)

# Parent->child offsets r (29 x 3)
# These are the pos attributes from the XML body tags
r = np.array([
    # BASE_LINK (origin)
    [0.0, 0.0, 0.0],
    # Right leg chain
    [-0.18337794, -0.0625, -0.28837794],  # HIP_R from BASE
    [0.141, 0.0, 0.0],                     # HIP_ABAD_R from HIP_R
    [0.029, 0.0, 0.0],                     # FEMUR_R from HIP_ABAD_R
    [0.375, 0.0, 0.0],                     # TIBIA_R from FEMUR_R
    [0.375, 0.0, 0.0],                     # ANKLE_R from TIBIA_R
    [0.0, 0.0, 0.0],                       # FOOT_R from ANKLE_R
    # Left leg chain
    [-0.18337794, 0.0625, -0.28837794],   # HIP_L from BASE
    [0.141, 0.0, 0.0],                     # HIP_ABAD_L from HIP_L
    [-0.029, 0.0, 0.0],                    # FEMUR_L from HIP_ABAD_L
    [0.375, 0.0, 0.0],                     # TIBIA_L from FEMUR_L
    [0.375, 0.0, 0.0],                     # ANKLE_L from TIBIA_L
    [0.0, 0.0, 0.0],                       # FOOT_L from ANKLE_L
    # Right arm chain
    [-0.065084608, -0.1567, 0.113474882], # UPPERSHOULDER_R from BASE
    [0.0, 0.0, 0.0],                       # LOWERSHOULDER_R from UPPERSHOULDER_R
    [0.0, -0.06, 0.0],                       # UPPERARM_R from LOWERSHOULDER_R
    [0.0, -0.16, 0.0],                # ELBOW_R from UPPERARM_R
    [0.0, -0.08, 0.0],     # FOREARM_R from ELBOW_R
    [0.0, -0.08, 0.0],                       # UPPERWRIST_R from FOREARM_R
    [0.0, -0.08, 0.0],                       # LOWERWRIST_R from UPPERWRIST_R
    # Left arm chain
    [-0.065084608, 0.1567, 0.113474882],  # UPPERSHOULDER_L from BASE
    [0.0, 0.0, 0.0],                       # LOWERSHOULDER_L from UPPERSHOULDER_L
    [0.0, 0.06, 0.0],                       # UPPERARM_L from LOWERSHOULDER_L
    [0.0, 0.16, 0.0],                 # ELBOW_L from UPPERARM_L
    [0.0, 0.08, 0.0],             # FOREARM_L from ELBOW_L
    [0.0, 0.08, 0.0],                       # UPPERWRIST_L from FOREARM_L
    [0.0, 0.08, 0.0],                       # LOWERWRIST_L from UPPERWRIST_L
    # Head
    [-0.069373655, 0.0, 0.230835692],     # NECK from BASE
    [0.0, 0.0, 0.0],                       # HEAD from NECK
], dtype=np.float32)

# Link CoMs (29 x 3) - from inertial pos in XML
com = np.array([
    # BASE_LINK (estimate, needs adjustment)
    [0.0, 0.0, 0.0],
    # Right leg
    [0.00398, -0.00006, 0.00636],   # HIP_R
    [-0.002, 0.00147, 0.00404],     # HIP_ABAD_R
    [0.06111, 0.00174, 0.02002],    # FEMUR_R
    [0.09678, 0.00694, -0.00006],   # TIBIA_R
    [0.0, 0.0, 0.0],                # ANKLE_R
    [0.0337, 0.00005, 0.04684],     # FOOT_R
    # Left leg
    [0.00398, -0.00006, 0.00636],   # HIP_L
    [0.00198, 0.00146, 0.00395],    # HIP_ABAD_L
    [0.06112, 0.00177, -0.02002],   # FEMUR_L
    [0.09678, 0.00694, -0.00006],   # TIBIA_L
    [0.0, 0.0, 0.0],                # ANKLE_L
    [0.0337, 0.00005, 0.04684],     # FOOT_L
    # Right arm
    [0.00158, 0.00121, -0.00674],   # UPPERSHOULDER_R
    [0.0535, -0.00006, 0.00002],    # LOWERSHOULDER_R
    [-0.00152, -0.0104, 0.1973],    # UPPERARM_R
    [0.05387, -0.01441, 0.00001],   # ELBOW_R
    [-0.00164, 0.00001, -0.01407],  # FOREARM_R
    [0.05463, -0.00013, 0.00009],   # UPPERWRIST_R
    [0.0, 0.00000242, 0.00010824],  # LOWERWRIST_R
    # Left arm
    [0.00158, 0.00123, 0.00673],    # UPPERSHOULDER_L
    [0.0535, 0.00006, -0.00002],    # LOWERSHOULDER_L
    [-0.00152, 0.01042, 0.1973],    # UPPERARM_L
    [0.05387, 0.01441, -0.00001],   # ELBOW_L
    [0.00164, 0.00002, -0.01407],   # FOREARM_L
    [0.05463, 0.00013, 0.00007],    # UPPERWRIST_L
    [0.0, -0.00000237, 0.00010824], # LOWERWRIST_L
    # Head
    [0.0, 0.00017, -0.00489],       # NECK
    [0.00094, 0.0631, -0.00001],    # HEAD
], dtype=np.float32)

# Inertia tensors at CoM (29 x 3 x 3)
# Storing diagonal elements for diagonalized inertias, or full 3x3 for non-diagonal
# For simplicity, I'll store the diagonal elements and note which need full inertia
I_diag = np.array([
    # BASE_LINK (placeholder)
    [0.1, 0.1, 0.1],
    # Right leg
    [0.0019, 0.00124, 0.00094],           # HIP_R (diag)
    [0.01442, 0.00791, 0.0072],           # HIP_ABAD_R (approx diag)
    [0.01172, 0.07026, 0.07594],          # FEMUR_R (approx diag)
    [0.00091, 0.02323, 0.02318],          # TIBIA_R (approx diag)
    [1e-06, 1e-06, 1e-06],                # ANKLE_R (diag)
    [0.002594, 0.002998, 0.000722],       # FOOT_R (approx diag)
    # Left leg
    [0.0019, 0.00124, 0.00094],           # HIP_L (diag)
    [0.01442, 0.00791, 0.0072],           # HIP_ABAD_L (approx diag)
    [0.01172, 0.07027, 0.07595],          # FEMUR_L (approx diag)
    [0.00091, 0.02323, 0.02318],          # TIBIA_L (approx diag)
    [1e-06, 1e-06, 1e-06],                # ANKLE_L (diag)
    [0.002594, 0.002998, 0.000722],       # FOOT_L (approx diag)
    # Right arm
    [0.000385927, 0.000304132, 0.000234707],  # UPPERSHOULDER_R (approx diag)
    [0.000261, 0.001315, 0.001242],           # LOWERSHOULDER_R (approx diag)
    [0.021671894, 0.021508856, 0.000365708], # UPPERARM_R (approx diag)
    [0.000368755, 0.001430606, 0.00143177],  # ELBOW_R (approx diag)
    [0.000684321, 0.000600813, 0.000265071], # FOREARM_R (approx diag)
    [0.000277159, 0.001412655, 0.001337476], # UPPERWRIST_R (approx diag)
    [1e-06, 1e-06, 1e-06],                    # LOWERWRIST_R (diag)
    # Left arm
    [0.000385927, 0.000304104, 0.000234735],  # UPPERSHOULDER_L (approx diag)
    [0.000261, 0.001315, 0.001242],           # LOWERSHOULDER_L (approx diag)
    [0.021671007, 0.021507692, 0.000365985],  # UPPERARM_L (approx diag)
    [0.000368777, 0.00143067, 0.001431848],   # ELBOW_L (approx diag)
    [0.00068432, 0.00060081, 0.000265072],    # FOREARM_L (approx diag)
    [0.000277159, 0.001412655, 0.001337475],  # UPPERWRIST_L (approx diag)
    [1e-06, 1e-06, 1e-06],                     # LOWERWRIST_L (diag)
    # Head
    [0.000199, 0.000274, 0.000167],           # NECK (approx diag)
    [0.004633, 0.003171, 0.004426],           # HEAD (approx diag)
], dtype=np.float32)

# Link rotation offsets (if any non-zero rotations in Xtree)
# From quat attributes in body tags - most are identity or 90-degree rotations
LRot = np.zeros(29, dtype=np.float32)

LROT_type = ["None"] * 29

# Joint limits (all joints have range="-3.14 3.14" in XML)
smallNum = -1e5
bigNum = 1e5

q_min = np.array([
    # Floating base (virtual, no limits)
    smallNum, smallNum, smallNum, smallNum, smallNum, smallNum,
    # Right leg
    -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
    # Left leg
    -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
    # Right arm
    -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
    # Left arm
    -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,
    # Head
    -3.14, -3.14
], dtype=np.float32)

q_max = np.array([
    # Floating base (virtual, no limits)
    bigNum, bigNum, bigNum, bigNum, bigNum, bigNum,
    # Right leg
    3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
    # Left leg
    3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
    # Right arm
    3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
    # Left arm
    3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
    # Head
    3.14, 3.14
], dtype=np.float32)

# Torque limits (would need actuator info, using conservative estimates)
tau_min = np.array([
    # Floating base (no actuation)
    0, 0, 0, 0, 0, 0,
    # Right leg (larger torques for hip/knee)
    -100, -100, -150, -150, -80, -80,
    # Left leg
    -100, -100, -150, -150, -80, -80,
    # Right arm (smaller torques)
    -50, -50, -50, -50, -30, -30, -30,
    # Left arm
    -50, -50, -50, -50, -30, -30, -30,
    # Head (small torques)
    -20, -20
], dtype=np.float32)

tau_max = np.array([
    # Floating base (no actuation)
    0, 0, 0, 0, 0, 0,
    # Right leg
    100, 100, 150, 150, 80, 80,
    # Left leg
    100, 100, 150, 150, 80, 80,
    # Right arm
    50, 50, 50, 50, 30, 30, 30,
    # Left arm
    50, 50, 50, 50, 30, 30, 30,
    # Head
    20, 20
], dtype=np.float32)

# Contact offsets (foot contact points)
# Right foot has 4 contact spheres at specific positions
contact_r = np.array([0.035, 0.0, 0.08], dtype=np.float32)  # Approximate center
# Left foot has 4 contact spheres at specific positions
contact_l = np.array([0.035, 0.0, 0.08], dtype=np.float32)  # Approximate center

# Hand/wrist end effector offsets
hand_r = np.array([0.0, -0.08, 0.0], dtype=np.float32)  # At wrist
hand_l = np.array([0.0, 0.08, 0.0], dtype=np.float32)  # At wrist

# Link indices (for reference)
# Useful for forward kinematics calculations
LINK_BASE = 0
LINK_HIP_R = 1
LINK_HIP_ABAD_R = 2
LINK_FEMUR_R = 3
LINK_TIBIA_R = 4
LINK_ANKLE_R = 5
LINK_FOOT_R = 6
LINK_HIP_L = 7
LINK_HIP_ABAD_L = 8
LINK_FEMUR_L = 9
LINK_TIBIA_L = 10
LINK_ANKLE_L = 11
LINK_FOOT_L = 12
LINK_UPPERSHOULDER_R = 13
LINK_LOWERSHOULDER_R = 14
LINK_UPPERARM_R = 15
LINK_ELBOW_R = 16
LINK_FOREARM_R = 17
LINK_UPPERWRIST_R = 18
LINK_LOWERWRIST_R = 19
LINK_UPPERSHOULDER_L = 20
LINK_LOWERSHOULDER_L = 21
LINK_UPPERARM_L = 22
LINK_ELBOW_L = 23
LINK_FOREARM_L = 24
LINK_UPPERWRIST_L = 25
LINK_LOWERWRIST_L = 26
LINK_NECK = 27
LINK_HEAD = 28
