# THEMIS ROS2 Interface — Safe Arm Control

## Why the Old Approach Broke the Robot

The previous `shm_udp_server.py` bridge **directly accessed POSIX shared memory** on the robot:

```python
# OLD — DANGEROUS: bypasses WBC, causes robot to collapse
import Startup.memory_manager as MM
MM.connect()
MM.RIGHT_ARM_JOINT_COMMAND.set(command)   # ← fights with WBC for SHM ownership
```

This competed with the robot's **Whole-Body Controller (WBC)** for ownership of the shared memory segments. The WBC lost its ability to read/write joint commands, the balance controller stopped working, and the robot collapsed.

## The New Approach: Official `wbc_api`

The THEMIS Developer Manual (§3.7) provides an **official API** that cooperates with the WBC pipeline:

```python
# NEW — SAFE: goes through WBC
from Play.Others import wbc as wbc_api

# Read (safe — same data, proper synchronization)
q, dq, u, temp, volt = wbc_api.get_joint_states(chain)  # chain: +2=right_arm, -2=left_arm
a, w, R = wbc_api.get_imu_states()

# Write (safe — WBC stays in the loop)
wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
```

We wrap this in ROS2 topics so you can send commands from any machine on the network.

## Architecture

```
┌──────────── Desktop PC ─────────────────────┐
│                                              │
│  test_arm_ros2.py  or  your tracking code    │
│    publishes:  /themis/arm_cmd/{right,left}  │
│    subscribes: /themis/joint_state/*          │
│                /themis/imu                    │
│                                              │
└──────── ROS2 DDS (192.168.0.0/24) ──────────┘
                       │
┌──────────── Robot PC (themisxxx) ────────────┐
│                                              │
│  themis_ros2_robot_node.py                   │
│    wbc_api.get_joint_states()  → publish     │
│    subscribe → wbc_api.set_joint_states()    │
│                                              │
│  WBC pipeline stays fully operational ✓      │
│  Balance controller keeps running ✓          │
│  No shared memory conflicts ✓                │
│                                              │
└──────────────────────────────────────────────┘
```

## Files

| File | Runs on | Description |
|------|---------|-------------|
| `themis_ros2_robot_node.py` | **Robot PC** | Publishes joint state, subscribes to arm commands via official `wbc_api` |
| `themis_ros2_desktop_client.py` | Desktop | Drop-in replacement for `ThemisUDPClient` — same API, uses ROS2 |
| `test_arm_ros2.py` | Desktop | Test script: read state, sinusoidal arm sweep |

## ROS2 Topics

### Published by robot node (robot → desktop)

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/themis/joint_state/right_arm` | `sensor_msgs/JointState` | 100 Hz | Right arm q, dq, torque |
| `/themis/joint_state/left_arm` | `sensor_msgs/JointState` | 100 Hz | Left arm q, dq, torque |
| `/themis/joint_state/right_leg` | `sensor_msgs/JointState` | 100 Hz | Right leg q, dq, torque |
| `/themis/joint_state/left_leg` | `sensor_msgs/JointState` | 100 Hz | Left leg q, dq, torque |
| `/themis/joint_state/head` | `sensor_msgs/JointState` | 100 Hz | Head q, dq, torque |
| `/themis/imu` | `sensor_msgs/Imu` | 100 Hz | Accel, gyro, orientation |

### Subscribed by robot node (desktop → robot)

| Topic | Type | Description |
|-------|------|-------------|
| `/themis/arm_cmd/right` | `sensor_msgs/JointState` | Right arm command (q, dq, u, kp, kd) |
| `/themis/arm_cmd/left` | `sensor_msgs/JointState` | Left arm command (q, dq, u, kp, kd) |

## Quick Start

### Prerequisites

Both machines need ROS2 Humble and must be on the same subnet:

```bash
# Robot PC (already has ROS2 Humble pre-installed)
export ROS_DOMAIN_ID=0

# Desktop PC
sudo apt install ros-humble-desktop   # if not installed
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
```

### Step 1: Boot the robot (on Robot PC)

```bash
ssh themis@192.168.0.11   # password: themis
cd /home/themis/THEMIS/THEMIS
bash Play/bootup.sh
```

### Step 2: Start the robot-side ROS2 node (on Robot PC)

```bash
# Copy the node to the robot first:
scp hw_interface/ros2/themis_ros2_robot_node.py themis@192.168.0.11:~/

# SSH into robot and run:
ssh themis@192.168.0.11
cd /home/themis/THEMIS/THEMIS
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
python3 ~/themis_ros2_robot_node.py
```

### Step 3: Run the test (on Desktop)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

# Read-only mode first (SAFE — no commands sent):
python3 hw_interface/ros2/test_arm_ros2.py --read-only

# If state reads look correct, run the sinusoidal test:
python3 hw_interface/ros2/test_arm_ros2.py
```

### Step 4: Verify with ROS2 CLI tools

```bash
# List topics (should see /themis/*)
ros2 topic list

# Echo arm state
ros2 topic echo /themis/joint_state/right_arm

# Check publish rate
ros2 topic hz /themis/joint_state/right_arm
```

## Using the Client in Your Code

The `ThemisROS2Client` is a drop-in replacement for `ThemisUDPClient`:

```python
from hw_interface.ros2.themis_ros2_desktop_client import ThemisROS2Client

client = ThemisROS2Client()
client.connect()

# Wait for first valid state
fb = client.wait_for_state(timeout=5.0)
print(f"Right arm: {fb.right_arm_q}")
print(f"IMU accel: {fb.imu_accel}")

# Send arm commands (goes through wbc_api on robot)
client.send_arm_command('right', q=desired_q, kp=np.full(7, 10.0), kd=np.full(7, 1.0))

# Cleanup
client.shutdown()
```

## Chain Index Reference (from THEMIS docs)

| Chain | Index | DOF |
|-------|-------|-----|
| Head | 0 | 2 |
| Right Leg | +1 | 6 |
| Left Leg | -1 | 6 |
| Right Arm | +2 | 7 |
| Left Arm | -2 | 7 |
| Right Hand | +3 | — |
| Left Hand | -3 | — |

## Troubleshooting

### "No valid state" on desktop
1. Check both machines are on the same subnet: `ping 192.168.0.11`
2. Check `ROS_DOMAIN_ID` matches: `echo $ROS_DOMAIN_ID`
3. Check the robot node is running: `ros2 node list` (should show `/themis_robot_node`)
4. Check firewall: ROS2 DDS uses UDP multicast on ports 7400+

### Robot still collapses
- Make sure `shm_udp_server.py` is **NOT** running
- Only `themis_ros2_robot_node.py` should be accessing the robot APIs
- Verify the robot node uses `wbc_api` not `memory_manager`

### High latency
- Use Ethernet, not WiFi
- The QoS is set to BEST_EFFORT for lowest latency
- Check with: `ros2 topic delay /themis/joint_state/right_arm`
