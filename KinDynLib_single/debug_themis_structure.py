import numpy as np

# Themis robot structure analysis

# From robot_dynamics.py:
parent = [-1,  0,  1,  2,  3,  4, 
           5,  6,  7,  8,  9, 10,
           5, 12, 13, 14, 15, 16,
           5, 18, 19, 20, 21, 22, 23,
           5, 25, 26, 27, 28, 29, 30,
           5, 32]

joint_types = ["Px", "Py", "Pz", "Rx", "Ry", "Rz",
               "Rz", "Rx", "Ry", "Ry", "Ry", "Rx",
               "Rz", "Rx", "Ry", "Ry", "Ry", "Rx",
               "Ry", "Rx", "Ry", "Rx", "Ry", "Rx", "Ry",
               "Ry", "Rx", "Ry", "Rx", "Ry", "Rx", "Ry",
               "Rz", "Ry"]

# Joint names from robot_const.py
JOINT_NAMES = [
    # Floating base (virtual) - indices 0-5
    "root_x", "root_y", "root_z", "root_roll", "root_pitch", "root_yaw",
    # Right leg (6) - indices 6-11
    "HIP_YAW_R", "HIP_ROLL_R", "HIP_PITCH_R", "KNEE_PITCH_R", "ANKLE_PITCH_R", "ANKLE_ROLL_R",
    # Left leg (6) - indices 12-17
    "HIP_YAW_L", "HIP_ROLL_L", "HIP_PITCH_L", "KNEE_PITCH_L", "ANKLE_PITCH_L", "ANKLE_ROLL_R",
    # Right arm (7) - indices 18-24
    "SHOULDER_PITCH_R", "SHOULDER_ROLL_R", "SHOULDER_YAW_R", "ELBOW_PITCH_R", "ELBOW_YAW_R", "WRIST_PITCH_R", "WRIST_YAW_R",
    # Left arm (7) - indices 25-31
    "SHOULDER_PITCH_L", "SHOULDER_ROLL_L", "SHOULDER_YAW_L", "ELBOW_PITCH_L", "ELBOW_YAW_L", "WRIST_PITCH_L", "WRIST_YAW_L",
    # Head (2) - indices 32-33
    "HEAD_YAW", "HEAD_PITCH"
]

# From XML structure:
# BASE (index 5 in links, connected to floating base)
#   -> Right leg chain (indices 6-11)
#   -> Left leg chain (indices 12-17)
#   -> Right arm chain (indices 18-24)
#   -> Left arm chain (indices 25-31)
#   -> Head chain (indices 32-33)

print("Total DOF:", len(parent))
print("Total joints:", len(joint_types))
print()

# Trace right arm chain from base to wrist
print("RIGHT ARM CHAIN:")
print("Base (5) -> SHOULDER_PITCH_R (18) -> SHOULDER_ROLL_R (19) -> SHOULDER_YAW_R (20)")
print("  -> ELBOW_PITCH_R (21) -> ELBOW_YAW_R (22) -> WRIST_PITCH_R (23) -> WRIST_YAW_R (24)")
print()

# Count: From XML
# BASE -> UPPERSHOULDER_R -> LOWERSHOULDER_R -> UPPERARM_R -> ELBOW_R -> FOREARM_R -> UPPERWRIST_R -> LOWERWRIST_R
# That's 8 links in right arm (7 joints)

print("LEFT ARM CHAIN:")
print("Base (5) -> SHOULDER_PITCH_L (25) -> SHOULDER_ROLL_L (26) -> SHOULDER_YAW_L (27)")
print("  -> ELBOW_PITCH_L (28) -> ELBOW_YAW_L (29) -> WRIST_PITCH_L (30) -> WRIST_YAW_L (31)")
print()

# So right wrist should be at index 24 (WRIST_YAW_R)
# And left wrist should be at index 31 (WRIST_YAW_L)

print("Expected wrist indices:")
print("  Right wrist (WRIST_YAW_R): 24")
print("  Left wrist (WRIST_YAW_L): 31")
print()

# But the test is using indices 23 and 30
print("Test is using:")
print("  Right hand: index 23 (should be WRIST_PITCH_R)")
print("  Left hand: index 30 (should be WRIST_PITCH_L)")
