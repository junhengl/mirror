# Themis Humanoid Robot - MuJoCo Simulation

Real-time visualization API for the Themis humanoid robot using MuJoCo simulator.

## Setup

The URDF has been converted to MuJoCo's native XML (MJCF) format for better compatibility:
- Original URDF: `themis/urdf/TH02-A7.urdf`  
- MuJoCo XML: `themis/TH02-A7.xml`

## Quick Start

### Basic Usage

```python
from mujoco_visualizer import ThemisSimulator
import numpy as np

# Initialize simulator (opens interactive viewer)
sim = ThemisSimulator(headless=False)

# Set base position and rotation
base_pos = np.array([0.0, 0.0, 1.0])  # x, y, z
base_rot = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion [x, y, z, w]
sim.set_base_pose(base_pos, base_rot)

# Set joint positions (in radians)
joint_positions = {
    "HIP_PITCH_R": -0.5,
    "KNEE_PITCH_R": 1.0,
    "ANKLE_PITCH_R": -0.5,
    # ... more joints
}
sim.set_joint_positions(joint_positions)

# Close when done
sim.close()
```

### Available Joints

The robot has 28 controllable joints:

**Legs (12 joints)**
- `HIP_YAW_L/R` - Hip yaw (rotation)
- `HIP_ROLL_L/R` - Hip abduction/adduction  
- `HIP_PITCH_L/R` - Hip flexion/extension
- `KNEE_PITCH_L/R` - Knee flexion/extension
- `ANKLE_PITCH_L/R` - Ankle dorsiflexion/plantarflexion
- `ANKLE_ROLL_L/R` - Ankle inversion/eversion

**Arms (12 joints)**
- `SHOULDER_YAW_L/R` - Shoulder rotation
- `SHOULDER_PITCH_L/R` - Shoulder flexion/extension
- `SHOULDER_ROLL_L/R` - Shoulder abduction/adduction
- `ELBOW_YAW_L/R` - Elbow rotation
- `ELBOW_PITCH_L/R` - Elbow flexion/extension
- `WRIST_YAW_L/R` - Wrist rotation
- `WRIST_PITCH_L/R` - Wrist flexion/extension

**Head (2 joints)**
- `HEAD_YAW` - Head rotation
- `HEAD_PITCH` - Head tilt

**Base (7 DOF free joint)**
- `root_joint` - 3D position + quaternion rotation (automatically managed)

## API Reference

### ThemisSimulator

#### `__init__(model_path=None, headless=False)`
Initialize the simulator.
- `model_path`: Path to MJCF file (default: uses included Themis model)
- `headless`: If True, runs without GUI (use `render()` for images)

#### `set_base_pose(position, rotation)`
Set the robot's base position and orientation.
- `position`: np.array([x, y, z])
- `rotation`: np.array([x, y, z, w]) quaternion OR 3x3 rotation matrix

#### `set_joint_positions(joint_positions)`
Update joint angles.
- `joint_positions`: dict mapping joint names to angles (radians)

#### `get_joint_positions()`
Returns dict of all current joint positions.

#### `get_body_pose(body_name)`
Get position and rotation of a specific body.
Returns: `(position, rotation_matrix)`

#### `step(dt=None)`
Step the simulation forward by one timestep.

#### `reset()`
Reset to initial state.

#### `render()`
Get rendered image (for headless mode).
Returns: np.array image (height, width, 3)

#### `close()`
Close the simulator and viewer.

## Examples

### Example 1: Static Pose
```python
from mujoco_visualizer import ThemisSimulator
import numpy as np
import time

sim = ThemisSimulator()

# T-pose
joint_pos = {
    "SHOULDER_ROLL_L": 1.57,  # 90 degrees
    "SHOULDER_ROLL_R": -1.57,
    "ELBOW_PITCH_L": 0.0,
    "ELBOW_PITCH_R": 0.0,
}
sim.set_joint_positions(joint_pos)

# Keep running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    sim.close()
```

### Example 2: Animated Motion
See `robot_visualizer_example.py` for a complete example with sine wave animations.

```bash
python robot_visualizer_example.py
```

### Example 3: Headless Rendering
```python
sim = ThemisSimulator(headless=True)

# Set pose
sim.set_base_pose([0, 0, 1], [0, 0, 0, 1])
sim.set_joint_positions({"HIP_PITCH_R": -0.5})

# Get rendered frame
frame = sim.render()  # Returns (720, 1280, 3) numpy array

sim.close()
```

## Integration with ZED Body Tracking

To connect body tracking data to the robot visualization:

1. Get body keypoints from ZED camera
2. Calculate joint angles using inverse kinematics or retargeting
3. Update robot pose in real-time:

```python
from mujoco_visualizer import ThemisSimulator
# ... ZED tracking code ...

sim = ThemisSimulator()

while tracking:
    # Get tracking data
    bodies = get_zed_bodies()
    
    # Retarget to robot (you'll implement this)
    joint_angles = retarget_to_robot(bodies)
    
    # Update visualization
    sim.set_joint_positions(joint_angles)
    sim.step()
```

## Troubleshooting

### "No module named 'mujoco'"
```bash
pip install mujoco
```

### Model fails to load
The MJCF file should be automatically used. If you need to regenerate it:
```bash
python convert_urdf_to_mjcf.py
```

### Viewer doesn't open
Make sure you have a display available. For headless environments, use `headless=True`.

## Files

- `mujoco_visualizer.py` - Main simulator class
- `robot_visualizer_example.py` - Example with animations
- `convert_urdf_to_mjcf.py` - URDF to MJCF converter
- `themis/TH02-A7.xml` - MuJoCo model file
- `themis/meshes/` - STL mesh files
