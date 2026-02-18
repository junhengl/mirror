# Themis Hardware Interface — Body Tracking Pipeline

This directory contains everything needed to run the motion tracking + retargeting pipeline against the **real Themis robot** (instead of MuJoCo simulation).

## Architecture

```
┌──────────────────────────── Desktop PC (your machine) ─────────────────────────┐
│                                                                                 │
│  [ZED Camera] → BodyTrackingNode (30 Hz) → RetargetingNode (500 Hz IK)         │
│                                                    │                            │
│                                              HardwareSenderNode (100 Hz)        │
│                                                    │ UDP                        │
│                                                    ▼                            │
└───────────────────────────── Ethernet cable ────────────────────────────────────┘
                                                     │
┌──────────────────────────── Robot PC ──────────────────────────────────────────┐
│                                                                                │
│  shm_udp_server.py ─── POSIX Shared Memory ─── WBC / BEAR Actuators           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Where it runs | Description |
|------|:-------------:|-------------|
| `shm_udp_server.py` | **Robot PC** | UDP ↔ shared-memory bridge; exposes joint state and accepts commands |
| `themis_udp_client.py` | Desktop | Python client API for the UDP bridge |
| `test_arm_simple.py` | Desktop | **(1) Simple test** — sends sinusoidal arm motions, prints feedback |
| `integrated_hw_pipeline.py` | Desktop | **(2) Full pipeline** — ZED tracking + IK retargeting → robot |

## Quick Start

### 1. On the Robot PC

Make sure the Themis AOS stack is booted up (shared memory initialized, actuator threads running):

```bash
cd /home/themis/AOS_hw/THEMIS
bash Play/bootup.sh        # or your usual boot procedure
```

Then start the UDP bridge:

```bash
cd /path/to/body_tracking_arm_mod_hw
python3 hw_interface/shm_udp_server.py --host 0.0.0.0 --port 9870
```

> Set `AOS_PATH` environment variable if AOS_hw is not at `/home/themis/AOS_hw/THEMIS`:
> ```bash
> export AOS_PATH=/your/path/to/AOS_hw/THEMIS
> ```

### 2. On the Desktop — Simple Arm Test

**First, configure ethernet** (one-time, see [Network Setup](#network-setup) below).

```bash
# Dry-run (no robot needed, prints what would be sent):
python3 hw_interface/test_arm_simple.py --dry-run

# Real robot:
python3 hw_interface/test_arm_simple.py --robot-ip 192.168.0.11 --port 9870
```

**Options:**
- `--direct` — bypass WBC, write directly to JOINT_COMMAND (use with caution)
- `--rate 50` — change command loop rate (default: 100 Hz)

### 3. On the Desktop — Full Integrated Pipeline

```bash
# With ZED camera (needs sudo for USB permissions):
sudo python3 hw_interface/integrated_hw_pipeline.py \
    --robot-ip 192.168.0.11 --port 9870

# Without camera (dummy tracking for testing):
sudo python3 hw_interface/integrated_hw_pipeline.py \
    --robot-ip 192.168.0.11 --no-camera

# Full dry-run (no robot, no camera):
python3 hw_interface/integrated_hw_pipeline.py --dry-run
```

**Options:**
- `--direct` — bypass WBC, send directly to JOINT_COMMAND
- `--blend-time 5.0` — how long to blend from idle to tracked pose (default: 3 s)
- `--hang-height 1.3` — robot hanging height for IK (default: 1.3 m)
- `--rate 100` — hardware sender rate (default: 100 Hz)

## Command Paths

There are two ways to send arm commands to the robot:

### Via MANIPULATION_REFERENCE (default, `--no-direct`)
- Commands go to `MANIPULATION_REFERENCE` shared memory
- The robot's WBC (whole-body controller) reads these and computes motor commands
- **Safer** — WBC handles dynamics, limits, and coordination with other limbs
- This is the recommended path

### Via JOINT_COMMAND (with `--direct`)
- Commands go directly to `*_ARM_JOINT_COMMAND` shared memory
- Bear actuators execute PD control directly: `τ = u + kp*(q_goal - q) + kd*(dq_goal - dq)`
- **More responsive** but bypasses WBC safety checks
- Use for testing or when WBC is not running

## Joint Convention

The Themis arm has 7 DOF per arm:

| Local Index | Joint | Description |
|:-----------:|-------|-------------|
| 0 | `shoulder_pitch` | Sagittal plane shoulder rotation |
| 1 | `shoulder_roll` | Frontal plane shoulder rotation |
| 2 | `shoulder_yaw` | Transverse plane shoulder rotation |
| 3 | `elbow_pitch` | Elbow flexion/extension |
| 4 | `elbow_yaw` | Forearm rotation |
| 5 | `wrist_pitch` | Wrist flexion/extension |
| 6 | `wrist_yaw` | Wrist rotation |

**Global joint IDs** (in AOS shared memory):
- Right arm: 13–19
- Left arm: 20–26

**Nominal IDLE pose:**
- Right: `[-0.20, +1.40, +1.57, +0.40, 0.00, 0.00, -1.50]`
- Left:  `[-0.20, -1.40, -1.57, -0.40, 0.00, 0.00, +1.50]`

## Network Protocol

The UDP bridge uses a simple binary protocol (see `shm_udp_server.py` header for full spec):

| Msg Type | Direction | Payload |
|:--------:|:---------:|---------|
| `0x01` | Desktop → Robot | State request (empty) |
| `0x02` | Robot → Desktop | State response (712 B: joint pos/vel/torq + motor temp/volt + base + IMU) |
| `0x10` | Desktop → Robot | Arm joint command (281 B) |
| `0x11` | Desktop → Robot | Manipulation reference (256 B) |
| `0x20` | Desktop → Robot | Heartbeat (empty) |
| `0xFE` | Robot → Desktop | ACK (1 B status) |

## Network Setup

The robot's main computer uses a static IP following the pattern `192.168.0.xx1` (see Themis documentation Table 6.1). Your desktop needs to be on the same `192.168.0.0/24` subnet via the Ethernet cable.

**Desktop ethernet setup (one-time):**

```bash
# Find your ethernet interface name (usually enp*, eth*, etc.)
ip link show | grep 'state UP'

# Assign a static IP on the same subnet (pick any unused .0.x address)
sudo ip addr add 192.168.0.100/24 dev enp130s0

# Verify connectivity
ping 192.168.0.11
```

To make this persistent across reboots, add it via NetworkManager or `/etc/netplan/`.

**Default credentials** (from Themis documentation):

| Host | IP Pattern | Username | Password |
|------|-----------|----------|----------|
| Main Computer | `192.168.0.xx1` | `themis` | `themis` |
| Head Jetson | `192.168.0.xx2` | `nvidia` | `nvidia` |
| Front Jetson | `192.168.0.xx3` | `nvidia` | `nvidia` |

## Safety Notes

1. **Always start with `--dry-run`** to verify the pipeline before connecting to the real robot.
2. The integrated pipeline **blends** from idle to tracked pose over 3 seconds (configurable).
3. On shutdown, arms automatically return to IDLE pose.
4. Per-step joint deltas are clamped to 0.05 rad (≈ 2.9°) at 100 Hz to prevent jerky motion.
5. The `--direct` flag bypasses WBC — use only when you know what you're doing.
