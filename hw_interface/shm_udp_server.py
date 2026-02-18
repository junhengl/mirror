#!/usr/bin/env python3
"""
Themis Shared Memory ↔ UDP Bridge  (runs on the ROBOT's onboard PC)

This server exposes selected POSIX shared-memory segments over UDP so that
a remote desktop (the one running body tracking + retargeting) can:

  • READ  joint state feedback   (robot → desktop)
  • WRITE arm joint commands      (desktop → robot)

Protocol
--------
Every UDP datagram is a simple binary frame:

    [1 B msg_type] [payload …]

msg_type values
    0x01  STATE_REQUEST       desktop → robot   (empty payload)
    0x02  STATE_RESPONSE      robot → desktop   (packed feedback)
    0x10  ARM_JOINT_COMMAND   desktop → robot   (packed arm command)
    0x11  MANIP_REFERENCE     desktop → robot   (packed manipulation ref)
    0x20  HEARTBEAT           desktop → robot   (empty, keeps connection alive)
    0xFE  ACK                 robot → desktop   (1 B status)

Feedback payload layout (STATE_RESPONSE, 0x02)
    right_arm_q        (7 × f64)  =  56 B
    right_arm_dq       (7 × f64)  =  56 B
    right_arm_torque   (7 × f64)  =  56 B
    left_arm_q         (7 × f64)  =  56 B
    left_arm_dq        (7 × f64)  =  56 B
    left_arm_torque    (7 × f64)  =  56 B
    right_arm_temp     (7 × f64)  =  56 B   (motor temperatures °C)
    right_arm_volt     (7 × f64)  =  56 B   (motor voltages V)
    left_arm_temp      (7 × f64)  =  56 B
    left_arm_volt      (7 × f64)  =  56 B
    base_position      (3 × f64)  =  24 B
    base_rot_matrix    (9 × f64)  =  72 B
    imu_accel          (3 × f64)  =  24 B
    imu_gyro           (3 × f64)  =  24 B
    timestamp          (1 × f64)  =   8 B
                                   ------
                                    712 B total

Arm joint command payload (ARM_JOINT_COMMAND, 0x10)
    side               (1 × u8)      1 B   (+2 right, 0xFE = -2 left)
    goal_positions     (7 × f64)  =  56 B
    goal_velocities    (7 × f64)  =  56 B
    goal_torques       (7 × f64)  =  56 B
    kp                 (7 × f64)  =  56 B
    kd                 (7 × f64)  =  56 B
                                   ------
                                    281 B total

Manipulation reference payload (MANIP_REFERENCE, 0x11)
    right_arm_pose     (7 × f64)  =  56 B
    right_arm_rate     (7 × f64)  =  56 B
    right_mode         (1 × f64)  =   8 B
    right_phase        (1 × f64)  =   8 B
    left_arm_pose      (7 × f64)  =  56 B
    left_arm_rate      (7 × f64)  =  56 B
    left_mode          (1 × f64)  =   8 B
    left_phase         (1 × f64)  =   8 B
                                   ------
                                    256 B total

Usage (on the robot PC):
    cd /home/themis/AOS_hw/THEMIS
    python3 /path/to/hw_interface/shm_udp_server.py --port 9870

    # Or set AOS_PATH if not at default location:
    AOS_PATH=/home/themis/AOS_hw/THEMIS python3 shm_udp_server.py

Make sure the AOS memory_manager has already been initialised (bootup.sh).

Network:
    Robot main computer IP: 192.168.0.xx1 (see Themis documentation)
    Desktop should be on same 192.168.0.0/24 subnet via Ethernet.
"""

import argparse
import socket
import struct
import time
import sys
import os
import numpy as np

# ── Make AOS importable ──────────────────────────────────────────────
# Adjust this path if AOS_hw lives elsewhere on the robot.
AOS_PATH = os.environ.get("AOS_PATH", "/home/themis/THEMIS/THEMIS")
if AOS_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(AOS_PATH))   # parent of THEMIS
    sys.path.insert(0, AOS_PATH)                     # THEMIS itself

import Startup.memory_manager as MM

# ── Message types ────────────────────────────────────────────────────
MSG_STATE_REQUEST     = 0x01
MSG_STATE_RESPONSE    = 0x02
MSG_ARM_JOINT_CMD     = 0x10
MSG_MANIP_REF         = 0x11
MSG_HEARTBEAT         = 0x20
MSG_ACK               = 0xFE

SIDE_RIGHT = 2    # +2
SIDE_LEFT  = 0xFE # -2  (stored as unsigned byte)


def pack_state_response() -> bytes:
    """Read shared memory and pack a STATE_RESPONSE datagram."""
    ra = MM.RIGHT_ARM_JOINT_STATE.get()
    la = MM.LEFT_ARM_JOINT_STATE.get()
    ra_bear = MM.RIGHT_ARM_BEAR_STATE.get()
    la_bear = MM.LEFT_ARM_BEAR_STATE.get()
    bs = MM.BASE_STATE.get()
    ss = MM.SENSE_STATE.get()

    buf = struct.pack('B', MSG_STATE_RESPONSE)
    buf += ra['joint_positions'].astype(np.float64).tobytes()
    buf += ra['joint_velocities'].astype(np.float64).tobytes()
    buf += ra['joint_torques'].astype(np.float64).tobytes()
    buf += la['joint_positions'].astype(np.float64).tobytes()
    buf += la['joint_velocities'].astype(np.float64).tobytes()
    buf += la['joint_torques'].astype(np.float64).tobytes()
    buf += ra_bear['bear_temperatures'].astype(np.float64).tobytes()
    buf += ra_bear['bear_voltages'].astype(np.float64).tobytes()
    buf += la_bear['bear_temperatures'].astype(np.float64).tobytes()
    buf += la_bear['bear_voltages'].astype(np.float64).tobytes()
    buf += bs['base_position'].astype(np.float64).tobytes()
    buf += bs['base_rotation_matrix'].astype(np.float64).tobytes()
    buf += ss['imu_acceleration'].astype(np.float64).tobytes()
    buf += ss['imu_angular_rate'].astype(np.float64).tobytes()
    buf += np.array([time.time()], dtype=np.float64).tobytes()
    return buf


def handle_arm_joint_cmd(payload: bytes):
    """Unpack ARM_JOINT_COMMAND and write to shared memory."""
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    # data layout: goal_pos(7), goal_vel(7), goal_torq(7), kp(7), kd(7) = 35
    assert data.size == 35, f"ARM_JOINT_CMD payload size mismatch: {data.size}"
    q   = data[0:7]
    dq  = data[7:14]
    u   = data[14:21]
    kp  = data[21:28]
    kd  = data[28:35]

    command = {
        'goal_joint_positions':    q,
        'goal_joint_velocities':   dq,
        'goal_joint_torques':      u,
        'bear_enable_statuses':    np.ones(7),
        'bear_operating_modes':    np.full(7, 3.0),   # force mode
        'bear_proportional_gains': kp,
        'bear_derivative_gains':   kd,
    }
    if side == SIDE_RIGHT:
        MM.RIGHT_ARM_JOINT_COMMAND.set(command)
    elif side == SIDE_LEFT:
        MM.LEFT_ARM_JOINT_COMMAND.set(command)
    else:
        print(f"[bridge] Unknown arm side byte: 0x{side:02X}")


def handle_manip_ref(payload: bytes):
    """Unpack MANIP_REFERENCE and write to shared memory."""
    data = np.frombuffer(payload, dtype=np.float64)
    # layout: r_pose(7), r_rate(7), r_mode(1), r_phase(1),
    #         l_pose(7), l_rate(7), l_mode(1), l_phase(1)  = 32
    assert data.size == 32, f"MANIP_REF payload size mismatch: {data.size}"
    ref = {
        'right_arm_pose':              data[0:7],
        'right_arm_pose_rate':         data[7:14],
        'right_manipulation_mode':     data[14:15],
        'right_manipulation_phase':    data[15:16],
        'left_arm_pose':               data[16:23],
        'left_arm_pose_rate':          data[23:30],
        'left_manipulation_mode':      data[30:31],
        'left_manipulation_phase':     data[31:32],
    }
    MM.MANIPULATION_REFERENCE.set(ref)


def run_server(host: str, port: int):
    """Main server loop."""
    # Connect to existing shared memory
    print("[bridge] Connecting to THEMIS shared memory …")
    MM.connect()
    print("[bridge] Shared memory connected.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)   # 500 ms for graceful shutdown check
    print(f"[bridge] UDP server listening on {host}:{port}")

    running = True
    last_heartbeat = time.time()
    pkt_count = 0

    try:
        while running:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                # Check heartbeat timeout (10 s)
                if time.time() - last_heartbeat > 30.0:
                    print("[bridge] No heartbeat for 30 s — still waiting …")
                    last_heartbeat = time.time()
                continue

            if len(data) < 1:
                continue

            msg_type = data[0]
            payload  = data[1:]

            if msg_type == MSG_STATE_REQUEST:
                resp = pack_state_response()
                sock.sendto(resp, addr)

            elif msg_type == MSG_ARM_JOINT_CMD:
                handle_arm_joint_cmd(payload)
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)

            elif msg_type == MSG_MANIP_REF:
                handle_manip_ref(payload)
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)

            elif msg_type == MSG_HEARTBEAT:
                last_heartbeat = time.time()
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)

            else:
                print(f"[bridge] Unknown msg_type 0x{msg_type:02X} from {addr}")

            pkt_count += 1
            if pkt_count % 500 == 0:
                print(f"[bridge] Processed {pkt_count} packets")

    except KeyboardInterrupt:
        print("\n[bridge] Shutting down …")
    finally:
        sock.close()
        print("[bridge] Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Themis SHM ↔ UDP bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9870, help="UDP port")
    args = parser.parse_args()
    run_server(args.host, args.port)
