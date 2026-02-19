#!/usr/bin/env python3
"""
Themis WBC API ↔ UDP Bridge  (runs on the ROBOT's onboard PC)

██  SAFE VERSION — uses the official wbc_api  ██

This replaces shm_udp_server.py which directly accessed POSIX shared
memory via memory_manager (MM).  That approach COMPETED with the WBC
for ownership of the shared-memory segments, causing the robot to lose
its balance controller and collapse.

This server uses the OFFICIAL APIs from the THEMIS Developer Manual:

  READS  (§3.7.2 — wbc_api):
    q, dq, u, temp, volt = wbc_api.get_joint_states(chain)
    accel, gyro, R       = wbc_api.get_imu_states()

  WRITES (§3.7.2 — wbc_api):
    wbc_api.set_joint_states(chain, u, q, dq, kp, kd)

These go THROUGH the whole-body controller, so the balance controller
stays alive and the robot doesn't collapse.

The UDP wire protocol is IDENTICAL to shm_udp_server.py so the desktop-
side ThemisUDPClient works without changes.

Usage (on the robot PC):
    # 1. Boot AOS first:
    cd /home/themis/THEMIS/THEMIS
    bash Play/bootup.sh

    # 2. Run this bridge:
    cd /home/themis/THEMIS/THEMIS
    python3 ~/wbc_udp_server.py --port 9870

    # Or with explicit AOS path:
    AOS_PATH=/home/themis/THEMIS/THEMIS python3 wbc_udp_server.py
"""

import argparse
import socket
import struct
import time
import sys
import os
import traceback
import numpy as np

# ── Make AOS importable ──────────────────────────────────────────────
AOS_PATH = os.environ.get("AOS_PATH", "/home/themis/THEMIS/THEMIS")
if AOS_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(AOS_PATH))   # parent dir
    sys.path.insert(0, AOS_PATH)                     # THEMIS itself

# ── Import the OFFICIAL APIs (from THEMIS Developer Manual §3.7) ─────
from Play.Others import wbc as wbc_api

# ── Chain indices (from documentation) ───────────────────────────────
CHAIN_HEAD      =  0
CHAIN_RIGHT_LEG =  1   # +1
CHAIN_LEFT_LEG  = -1
CHAIN_RIGHT_ARM =  2   # +2
CHAIN_LEFT_ARM  = -2
CHAIN_RIGHT_HAND =  3  # +3  (DXL hand motors)
CHAIN_LEFT_HAND  = -3

# ── Message types (same wire format as shm_udp_server.py) ───────────
MSG_STATE_REQUEST     = 0x01
MSG_STATE_RESPONSE    = 0x02
MSG_HAND_STATE_REQUEST  = 0x03
MSG_HAND_STATE_RESPONSE = 0x04
MSG_ARM_JOINT_CMD     = 0x10
MSG_HAND_JOINT_CMD    = 0x12
MSG_HEAD_JOINT_CMD    = 0x13
MSG_MANIP_REF         = 0x11
MSG_HEARTBEAT         = 0x20
MSG_ACK               = 0xFE

SIDE_RIGHT = 2      # +2
SIDE_LEFT  = 0xFE   # -2 stored as unsigned byte
SIDE_RIGHT_HAND = 3     # +3
SIDE_LEFT_HAND  = 0xFD  # -3 stored as unsigned byte
SIDE_HEAD = 0           # chain 0


def pack_state_response() -> bytes:
    """
    Read joint states via the OFFICIAL wbc_api and pack a
    STATE_RESPONSE datagram.

    Wire format (identical to shm_udp_server.py for compatibility):
        right_arm_q(7), right_arm_dq(7), right_arm_torque(7),
        left_arm_q(7),  left_arm_dq(7),  left_arm_torque(7),
        right_arm_temp(7), right_arm_volt(7),
        left_arm_temp(7),  left_arm_volt(7),
        base_position(3), base_rot_matrix(9),
        imu_accel(3), imu_gyro(3),
        timestamp(1)
        = 89 doubles = 712 bytes
    """
    # ── Read arm states via official API ─────────────────────────────
    # wbc_api.get_joint_states(chain) → (q, dq, u)
    # Note: the real API returns 3 values, NOT 5.
    # temp/volt are not available through wbc_api.
    try:
        ra_q, ra_dq, ra_u = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
        ra_temp = np.zeros(7)
        ra_volt = np.zeros(7)
    except Exception:
        ra_q = ra_dq = ra_u = ra_temp = ra_volt = np.zeros(7)

    try:
        la_q, la_dq, la_u = wbc_api.get_joint_states(CHAIN_LEFT_ARM)
        la_temp = np.zeros(7)
        la_volt = np.zeros(7)
    except Exception:
        la_q = la_dq = la_u = la_temp = la_volt = np.zeros(7)

    # ── Read IMU via official API ────────────────────────────────────
    # wbc_api.get_imu_states() → (accel, angular_rate, rotation_matrix)
    try:
        imu_accel, imu_gyro, imu_R = wbc_api.get_imu_states()
    except Exception:
        imu_accel = np.zeros(3)
        imu_gyro  = np.zeros(3)
        imu_R     = np.eye(3)

    # Base position — not directly available from wbc_api,
    # but we can estimate from the rotation matrix or set to zero.
    # The old server read MM.BASE_STATE['base_position'].
    # We'll use zeros since wbc_api doesn't expose base_position directly.
    base_pos = np.zeros(3)

    # ── Pack into the same wire format ───────────────────────────────
    buf = struct.pack('B', MSG_STATE_RESPONSE)
    buf += np.asarray(ra_q,      dtype=np.float64).tobytes()
    buf += np.asarray(ra_dq,     dtype=np.float64).tobytes()
    buf += np.asarray(ra_u,      dtype=np.float64).tobytes()
    buf += np.asarray(la_q,      dtype=np.float64).tobytes()
    buf += np.asarray(la_dq,     dtype=np.float64).tobytes()
    buf += np.asarray(la_u,      dtype=np.float64).tobytes()
    buf += np.asarray(ra_temp,   dtype=np.float64).tobytes()
    buf += np.asarray(ra_volt,   dtype=np.float64).tobytes()
    buf += np.asarray(la_temp,   dtype=np.float64).tobytes()
    buf += np.asarray(la_volt,   dtype=np.float64).tobytes()
    buf += np.asarray(base_pos,  dtype=np.float64).tobytes()
    buf += np.asarray(imu_R,     dtype=np.float64).ravel().tobytes()
    buf += np.asarray(imu_accel, dtype=np.float64).tobytes()
    buf += np.asarray(imu_gyro,  dtype=np.float64).tobytes()
    buf += np.array([time.time()], dtype=np.float64).tobytes()
    return buf


def handle_arm_joint_cmd(payload: bytes):
    """
    Unpack ARM_JOINT_COMMAND and apply via the OFFICIAL API:

        wbc_api.set_joint_states(chain, u, q, dq, kp, kd)

    This goes THROUGH the WBC pipeline — the balance controller
    stays operational.
    """
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 35:
        print(f"[wbc-bridge] ARM_JOINT_CMD payload mismatch: got {data.size}, expected 35")
        return

    q   = data[0:7].copy()
    dq  = data[7:14].copy()
    u   = data[14:21].copy()
    kp  = data[21:28].copy()
    kd  = data[28:35].copy()

    if side == SIDE_RIGHT:
        chain = CHAIN_RIGHT_ARM
    elif side == SIDE_LEFT:
        chain = CHAIN_LEFT_ARM
    else:
        print(f"[wbc-bridge] Unknown arm side byte: 0x{side:02X}")
        return

    try:
        # Official API: set_joint_states(chain, u, q, dq, kp, kd)
        wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
    except Exception as e:
        print(f"[wbc-bridge] set_joint_states failed: {e}")


def handle_manip_ref(payload: bytes):
    """
    Unpack MANIP_REFERENCE.

    NOTE: The official wbc_api does not directly expose a
    set_manipulation_reference() method. The manipulation reference
    is typically set by the locomotion top_level.py.

    For arm control, use ARM_JOINT_CMD (0x10) with
    wbc_api.set_joint_states() instead — this is the documented
    safe path.

    If you absolutely need to set manipulation references, the old
    shm_udp_server.py can be used, but BE AWARE it accesses shared
    memory directly.
    """
    data = np.frombuffer(payload, dtype=np.float64)
    if data.size != 32:
        print(f"[wbc-bridge] MANIP_REF payload mismatch: got {data.size}, expected 32")
        return

    # Extract arm poses and apply as joint commands via wbc_api
    r_pose = data[0:7].copy()
    r_rate = data[7:14].copy()
    # r_mode = data[14]   # unused for now
    # r_phase = data[15]  # unused for now
    l_pose = data[16:23].copy()
    l_rate = data[23:30].copy()
    # l_mode = data[30]
    # l_phase = data[31]

    # Use set_joint_states as the safe alternative
    # Default soft gains for manipulation reference
    kp = np.full(7, 10.0)
    kd = np.full(7,  1.0)
    u  = np.zeros(7)

    try:
        wbc_api.set_joint_states(CHAIN_RIGHT_ARM, u, r_pose, r_rate, kp, kd)
    except Exception as e:
        print(f"[wbc-bridge] set_joint_states (right, manip_ref) failed: {e}")

    try:
        wbc_api.set_joint_states(CHAIN_LEFT_ARM, u, l_pose, l_rate, kp, kd)
    except Exception as e:
        print(f"[wbc-bridge] set_joint_states (left, manip_ref) failed: {e}")


def pack_hand_state_response() -> bytes:
    """
    Read hand (DXL) joint states via wbc_api and pack a
    HAND_STATE_RESPONSE datagram.

    Wire format:
        right_hand_q(7), right_hand_dq(7), right_hand_u(7),
        left_hand_q(7),  left_hand_dq(7),  left_hand_u(7),
        timestamp(1)
        = 43 doubles = 344 bytes
    """
    try:
        rh_q, rh_dq, rh_u = wbc_api.get_joint_states(CHAIN_RIGHT_HAND)
    except Exception:
        rh_q = rh_dq = rh_u = np.zeros(7)

    try:
        lh_q, lh_dq, lh_u = wbc_api.get_joint_states(CHAIN_LEFT_HAND)
    except Exception:
        lh_q = lh_dq = lh_u = np.zeros(7)

    buf = struct.pack('B', MSG_HAND_STATE_RESPONSE)
    buf += np.asarray(rh_q,  dtype=np.float64).tobytes()
    buf += np.asarray(rh_dq, dtype=np.float64).tobytes()
    buf += np.asarray(rh_u,  dtype=np.float64).tobytes()
    buf += np.asarray(lh_q,  dtype=np.float64).tobytes()
    buf += np.asarray(lh_dq, dtype=np.float64).tobytes()
    buf += np.asarray(lh_u,  dtype=np.float64).tobytes()
    buf += np.array([time.time()], dtype=np.float64).tobytes()
    return buf


def handle_hand_joint_cmd(payload: bytes):
    """
    Unpack HAND_JOINT_COMMAND and apply via official API:

        wbc_api.set_joint_states(±3, u, q, dq, kp, kd)

    Each hand has 7 DXL motors:
      [0:1] = finger 1 flex (2 DOF: prox, dist)
      [2:3] = finger 2 flex (2 DOF: prox, dist)
      [4:5] = finger 3 flex (2 DOF: prox, dist)
      [6]   = finger split
    """
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 35:
        print(f"[wbc-bridge] HAND_JOINT_CMD payload mismatch: got {data.size}, expected 35")
        return

    q   = data[0:7].copy()
    dq  = data[7:14].copy()
    u   = data[14:21].copy()
    kp  = data[21:28].copy()
    kd  = data[28:35].copy()

    if side == SIDE_RIGHT_HAND:
        chain = CHAIN_RIGHT_HAND
    elif side == SIDE_LEFT_HAND:
        chain = CHAIN_LEFT_HAND
    else:
        print(f"[wbc-bridge] Unknown hand side byte: 0x{side:02X}")
        return

    try:
        wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
    except Exception as e:
        print(f"[wbc-bridge] set_joint_states (hand) failed: {e}")


def handle_head_joint_cmd(payload: bytes):
    """
    Unpack HEAD_JOINT_COMMAND and apply via official API:

        wbc_api.set_joint_states(0, u, q, dq, kp, kd)

    Head is chain 0 with 2 motors.
    Wire format: side_byte(1) + [q(2), dq(2), u(2), kp(2), kd(2)] as float64
                 = 1 + 10*8 = 81 bytes payload
    """
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 10:
        print(f"[wbc-bridge] HEAD_JOINT_CMD payload mismatch: got {data.size}, expected 10")
        return

    q   = data[0:2].copy()
    dq  = data[2:4].copy()
    u   = data[4:6].copy()
    kp  = data[6:8].copy()
    kd  = data[8:10].copy()

    if side != SIDE_HEAD:
        print(f"[wbc-bridge] Unknown head side byte: 0x{side:02X} (expected 0x00)")
        return

    try:
        wbc_api.set_joint_states(CHAIN_HEAD, u, q, dq, kp, kd)
    except Exception as e:
        print(f"[wbc-bridge] set_joint_states (head) failed: {e}")


def run_server(host: str, port: int, log_enabled: bool = False):
    """Main server loop."""
    print("=" * 60)
    print("  THEMIS WBC-API UDP Bridge  (SAFE — official API)")
    print("=" * 60)
    print(f"  AOS_PATH: {AOS_PATH}")
    print(f"  Using:    wbc_api.get_joint_states() / set_joint_states()")
    print(f"  NOT using: memory_manager (shared memory)")
    if log_enabled:
        print(f"  LOGGING:  ENABLED — will save server_log_*.npz on exit")
    print("=" * 60)

    # Verify wbc_api is accessible
    print("\n[wbc-bridge] Verifying wbc_api access …")
    try:
        ra_q, ra_dq, ra_u = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
        print(f"[wbc-bridge] ✓ Right arm q = {np.degrees(np.asarray(ra_q)).round(1)}")
    except Exception as e:
        print(f"[wbc-bridge] ✗ wbc_api.get_joint_states() failed: {e}")
        print("[wbc-bridge]   Make sure AOS is booted (bash Play/bootup.sh)")
        traceback.print_exc()
        sys.exit(1)

    try:
        imu_a, imu_w, imu_R = wbc_api.get_imu_states()
        print(f"[wbc-bridge] ✓ IMU accel = {np.asarray(imu_a).round(3)}")
    except Exception as e:
        print(f"[wbc-bridge] ✗ wbc_api.get_imu_states() failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("[wbc-bridge] API verification passed ✓\n")

    # ── Server-side logging (optional) ───────────────────────────────
    LOG_MAX = 300_000  # ~5 min at 1 kHz
    if log_enabled:
        log_recv_t   = np.zeros(LOG_MAX)        # time each CMD arrived
        log_set_t    = np.zeros(LOG_MAX)        # time set_joint_states returned
        log_q_r      = np.zeros((LOG_MAX, 7))   # commanded q right
        log_q_l      = np.zeros((LOG_MAX, 7))   # commanded q left
        log_fb_q_r   = np.zeros((LOG_MAX, 7))   # actual q right (read after set)
        log_fb_q_l   = np.zeros((LOG_MAX, 7))   # actual q left
        log_idx = 0

    # Wrap handle_arm_joint_cmd to add logging
    _orig_handle_arm = handle_arm_joint_cmd

    def handle_arm_joint_cmd_logged(payload: bytes):
        nonlocal log_idx
        t_recv = time.perf_counter()

        side = payload[0]
        data = np.frombuffer(payload[1:], dtype=np.float64)
        if data.size != 35:
            return
        q  = data[0:7].copy()
        dq = data[7:14].copy()
        u  = data[14:21].copy()
        kp = data[21:28].copy()
        kd = data[28:35].copy()

        if side == SIDE_RIGHT:
            chain = CHAIN_RIGHT_ARM
        elif side == SIDE_LEFT:
            chain = CHAIN_LEFT_ARM
        else:
            return

        try:
            wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
        except Exception as e:
            print(f"[wbc-bridge] set_joint_states failed: {e}")

        t_done = time.perf_counter()

        if log_idx < LOG_MAX:
            log_recv_t[log_idx] = t_recv
            log_set_t[log_idx]  = t_done
            if side == SIDE_RIGHT:
                log_q_r[log_idx] = q
                # Read actual state immediately after write
                try:
                    fb_q, _, _ = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
                    log_fb_q_r[log_idx] = fb_q
                except Exception:
                    pass
            else:
                log_q_l[log_idx] = q
                try:
                    fb_q, _, _ = wbc_api.get_joint_states(CHAIN_LEFT_ARM)
                    log_fb_q_l[log_idx] = fb_q
                except Exception:
                    pass
            log_idx += 1

    # ── Start UDP server ─────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"[wbc-bridge] UDP server listening on {host}:{port}")
    print("[wbc-bridge] Waiting for desktop connection …\n")

    last_heartbeat = time.time()
    pkt_count = 0
    cmd_count = 0
    last_stats = time.time()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                if time.time() - last_heartbeat > 30.0:
                    print("[wbc-bridge] No heartbeat for 30 s — still waiting …")
                    last_heartbeat = time.time()
                continue

            if len(data) < 1:
                continue

            msg_type = data[0]
            payload  = data[1:]

            if msg_type == MSG_STATE_REQUEST:
                resp = pack_state_response()
                sock.sendto(resp, addr)

            elif msg_type == MSG_HAND_STATE_REQUEST:
                resp = pack_hand_state_response()
                sock.sendto(resp, addr)

            elif msg_type == MSG_HAND_JOINT_CMD:
                handle_hand_joint_cmd(payload)
                # Fire-and-forget — no ACK (same as arm cmds)
                cmd_count += 1

            elif msg_type == MSG_HEAD_JOINT_CMD:
                handle_head_joint_cmd(payload)
                cmd_count += 1

            elif msg_type == MSG_ARM_JOINT_CMD:
                if log_enabled:
                    handle_arm_joint_cmd_logged(payload)
                else:
                    handle_arm_joint_cmd(payload)
                # NO ACK for arm cmds — fire-and-forget pattern.
                # Sending 2000 ACKs/s wastes server time and clogs
                # the client socket (interferes with get_state reads).
                cmd_count += 1

            elif msg_type == MSG_MANIP_REF:
                handle_manip_ref(payload)
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)
                cmd_count += 1

            elif msg_type == MSG_HEARTBEAT:
                last_heartbeat = time.time()
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)

            else:
                print(f"[wbc-bridge] Unknown msg_type 0x{msg_type:02X} from {addr}")

            pkt_count += 1

            # Periodic stats
            if time.time() - last_stats >= 5.0:
                elapsed = time.time() - last_stats
                pps = pkt_count / elapsed
                cps = cmd_count / elapsed
                print(f"[wbc-bridge] {pps:.0f} pkt/s | {cps:.0f} cmd/s | total {pkt_count} pkts")
                pkt_count = 0
                cmd_count = 0
                last_stats = time.time()

    except KeyboardInterrupt:
        print("\n[wbc-bridge] Shutting down …")
    finally:
        sock.close()
        if log_enabled and log_idx > 0:
            from datetime import datetime as _dt
            ts = _dt.now().strftime('%Y%m%d_%H%M%S')
            fname = f'server_log_{ts}.npz'
            np.savez_compressed(
                fname,
                recv_t=log_recv_t[:log_idx],
                set_t=log_set_t[:log_idx],
                q_r=log_q_r[:log_idx],
                q_l=log_q_l[:log_idx],
                fb_q_r=log_fb_q_r[:log_idx],
                fb_q_l=log_fb_q_l[:log_idx],
            )
            print(f"[wbc-bridge] Server log saved → {fname}  ({log_idx} entries)")
        print("[wbc-bridge] Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Themis WBC-API ↔ UDP bridge (safe — uses official API)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9870, help="UDP port")
    parser.add_argument("--log", action="store_true",
                        help="Enable server-side logging of every CMD received "
                             "and actual joint state after each set_joint_states(). "
                             "Saves server_log_*.npz on Ctrl-C.")
    args = parser.parse_args()
    run_server(args.host, args.port, log_enabled=args.log)
