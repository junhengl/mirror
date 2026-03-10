# MIRROR: Visual Motion Imitation via Real-time Retargeting and Teleoperation with Parallel Differential Inverse Kinematics

Complete real-time simulation pipeline for humanoid robot body tracking with parallel differential IK and ZED camera body tracking.

![Title](assets/title.gif)

## Quick Start

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mirror
```

### Run simulation

```bash
cd mirror
.venv/bin/python real_time_sim/main.py
```

Optional flags:
- `--no-camera` - Disable ZED camera (use dummy tracking data)
- `--headless` - Run without viewer visualization
- `--model themis` - Use Themis model instead of default Westwood

### Configuration

Edit [real_time_sim/config.py](real_time_sim/config.py) to customize:
- Simulation timestep and rates
- PD controller gains per joint
- Joint limits and safety thresholds
- Camera and tracking parameters

## Hardware Demo

![Demo](assets/demo.gif)

## Project Structure

```
real_time_sim/                 # Main pipeline
├── main.py                    # Entry point - launches all nodes
├── config.py                  # Pipeline configuration
├── shared_state.py            # Thread-safe inter-node communication
├── joint_mapping.py           # KinDynLib ↔ MuJoCo coordinate conversion
├── simulation/
│   └── mujoco_sim.py          # MuJoCo physics engine (1 kHz)
└── nodes/
    ├── body_tracking_node.py  # ZED tracking + filtering (30 Hz)
    ├── retargeting_node.py    # IK-based body retargeting (500 Hz)
    └── controller_node.py     # PD torque control (1 kHz)

KinDynLib/                     # Robot dynamics library
├── robot_dynamics.py          # Forward/inverse kinematics
├── robot_const.py             # Robot joint/link constants
└── dynamics_lib.py            # Spatial transform math

westwood_robots/               # MuJoCo models
└── TH02-A7-torque.xml         # Humanoid with actuators
```

## Components

### 1. Body Tracking Node (30 Hz)
- Captures human pose from ZED camera
- Extracts arm keypoints (shoulders, elbows, wrists, hands)
- Applies position filtering and jump rejection
- Publishes to shared state

**Config parameters:** [TrackingConfig](real_time_sim/config.py#L50)

### 2. Retargeting Node (200-500 Hz)
- Converts human arm positions → robot joint angles
- Uses inverse kinematics with KinDynLib
- Applies joint limits and collision avoidance (optional)
- Outputs desired joint positions for controller

**References:** [RetargetingConfig](real_time_sim/config.py#L70), [LimbRetargeting](real_time_sim/nodes/retargeting_node.py#L300)

### 3. Controller Node (1 kHz)
- PD torque control with FSM state management
- States: INIT → IDLE → TRACKING → SAFETY_STOP
- Torque limits and safety thresholds
- Sends torques to MuJoCo simulator

**Gains:** [ControlConfig](real_time_sim/config.py#L25)

### 4. MuJoCo Simulator (1 kHz)
- Fixed-base humanoid physics simulation
- Torque-controlled joints
- Center-of-mass tracking
- Real-time visualization (optional)

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `numpy` | Array math, linear algebra | Yes |
| `mujoco` | Physics simulation & visualization | Yes |
| `scipy` | Spatial transforms, sparse matrices | Yes |
| `opencv-python` | Camera frame display | Yes |
| `osqp` | QP solver for dynamics | Optional |
| `torch` | GPU-accelerated QP solver | Optional |
| `pyzed` | ZED camera body tracking (requires [ZED SDK](https://www.stereolabs.com/developers/release)) | Optional (camera only) |

All required packages are installed automatically via [environment.yml](environment.yml). The ZED SDK must be installed separately if using camera tracking — the pipeline runs without it using `--no-camera`.

## TODOs
- Hardware interface documentation for THEMIS
- Gripper integration in sim
- Migration to Unitree G1 ecosystem