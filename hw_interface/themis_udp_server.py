#!/usr/bin/env python3
"""
Themis Unified UDP Server  (runs on the ROBOT's onboard PC)

Replaces both  shm_udp_server.py  (direct SHM, no WBC)
           and  wbc_udp_server.py  (WBC only).

Auto-detects whether the Whole-Body Controller (WBC) is running and
switches between two operating modes:

  WBC mode  (recommended when AOS is fully booted):
    • Reads state via wbc_api.get_joint_states()
    • Arms: writes to MANIPULATION_REFERENCE  (POSE + SWING mode)
    • Disables the onboard manipulation thread to prevent conflicts
    • Hands/head: wbc_api.set_joint_states()
    • Base lean: lm_api.set_base_orientation()

  Direct mode  (AOS partially booted / no WBC):
    • Reads state from MM.*_JOINT_STATE shared memory
    • Arms: writes to MM.*_ARM_JOINT_COMMAND directly
    • Hands: writes to MM.*_HAND_JOINT_COMMAND directly
    • Head: writes to MM.HEAD_JOINT_COMMAND (if available)
    • Base lean: not available (silently ignored)

Both desktop pipelines work unchanged:
  • integrated_hw_wbc.py   sends MANIP_REF   → routed correctly in both modes
  • integrated_hw_pipeline.py sends ARM_JOINT_CMD → routed correctly in both modes

Protocol (identical wire format — ThemisUDPClient works unchanged)
---------------------------------------------------------------------
    0x01  STATE_REQUEST       desktop → robot   (empty)
    0x02  STATE_RESPONSE      robot → desktop   (712 B feedback)
    0x03  HAND_STATE_REQUEST  desktop → robot   (empty)
    0x04  HAND_STATE_RESPONSE robot → desktop   (344 B hand feedback)
    0x10  ARM_JOINT_COMMAND   desktop → robot   (281 B arm cmd)
    0x11  MANIP_REFERENCE     desktop → robot   (256 B manip ref)
    0x12  HAND_JOINT_CMD      desktop → robot   (281 B hand cmd)
    0x13  HEAD_JOINT_CMD      desktop → robot   (81 B head cmd)
    0x14  BASE_ORIENT         desktop → robot   (24 B rpy)
    0x20  HEARTBEAT           desktop → robot   (empty)
    0x30  MODE_QUERY          desktop → robot   (empty)
    0x31  MODE_RESPONSE       robot → desktop   (1 B mode: 0=direct, 1=wbc)
    0xFE  ACK                 robot → desktop   (1 B status)

Usage (on the robot PC):
    cd /home/themis/THEMIS/THEMIS

    # Auto-detect mode (recommended):
    python3 ~/themis_udp_server.py

    # Force a specific mode:
    python3 ~/themis_udp_server.py --mode wbc
    python3 ~/themis_udp_server.py --mode direct

    # With server-side logging:
    python3 ~/themis_udp_server.py --log
"""

import argparse
import socket
import struct
import subprocess
import time
import sys
import os
import traceback
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# AOS imports (soft — auto-detect what's available)
# ═══════════════════════════════════════════════════════════════════════

AOS_PATH = os.environ.get("AOS_PATH", "/home/themis/THEMIS/THEMIS")
if AOS_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(AOS_PATH))   # parent dir
    sys.path.insert(0, AOS_PATH)                     # THEMIS itself

# ── WBC API (optional) ──────────────────────────────────────────────
wbc_api = None
try:
    from Play.Others import wbc as wbc_api
    print("[server] ✓ wbc_api imported")
except ImportError:
    print("[server] ⚠ wbc_api not available — direct SHM only")

# ── Locomotion API (optional — for base orientation) ─────────────────
lm_api = None
if wbc_api is not None:
    for _name in ['locomotion_manager', 'locomotion', 'lm',
                   'locomotion_control', 'walking']:
        try:
            _mod = __import__('Play.Others', fromlist=[_name])
            lm_api = getattr(_mod, _name)
            print(f"[server] ✓ Locomotion API: Play.Others.{_name}")
            break
        except (ImportError, AttributeError):
            continue
    if lm_api is None and hasattr(wbc_api, 'set_base_orientation'):
        lm_api = wbc_api
        print("[server] ✓ Locomotion API found on wbc_api")
    if lm_api is None:
        print("[server] ⚠ Locomotion API not found — base orient ignored")

# ── Shared memory (always needed) ───────────────────────────────────
import Startup.memory_manager as MM


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Chain indices  (from AOS documentation)
CHAIN_HEAD       =  0
CHAIN_RIGHT_LEG  =  1
CHAIN_LEFT_LEG   = -1
CHAIN_RIGHT_ARM  =  2
CHAIN_LEFT_ARM   = -2
CHAIN_RIGHT_HAND =  3
CHAIN_LEFT_HAND  = -3

# Message types
MSG_STATE_REQUEST       = 0x01
MSG_STATE_RESPONSE      = 0x02
MSG_HAND_STATE_REQUEST  = 0x03
MSG_HAND_STATE_RESPONSE = 0x04
MSG_ARM_JOINT_CMD       = 0x10
MSG_MANIP_REF           = 0x11
MSG_HAND_JOINT_CMD      = 0x12
MSG_HEAD_JOINT_CMD      = 0x13
MSG_BASE_ORIENT         = 0x14
MSG_HEARTBEAT           = 0x20
MSG_MODE_QUERY          = 0x30
MSG_MODE_RESPONSE       = 0x31
MSG_ACK                 = 0xFE

SIDE_RIGHT      = 2
SIDE_LEFT       = 0xFE
SIDE_RIGHT_HAND = 3
SIDE_LEFT_HAND  = 0xFD
SIDE_HEAD       = 0

# Operating modes
MODE_DIRECT = 0    # Direct shared-memory (no WBC)
MODE_WBC    = 1    # Through WBC pipeline

# Manipulation reference constants  (from manipulation_macros.py)
MANIP_MODE_POSE   = 100.0   # POSE mode
MANIP_PHASE_SWING = 0.0     # SWING phase — arm free to move

# Default arm poses  (from manipulation_macros.py)
IDLE_R = np.array([-0.20, +1.40, +1.57, +0.40, 0.0, 0.0, -1.50])
IDLE_L = np.array([-0.20, -1.40, -1.57, -0.40, 0.0, 0.0, +1.50])

# Default PD gains for direct-mode arm commands
DEFAULT_ARM_KP = np.full(7, 80.0)
DEFAULT_ARM_KD = np.full(7,  3.0)


# ═══════════════════════════════════════════════════════════════════════
# Global state
# ═══════════════════════════════════════════════════════════════════════

_server_mode = MODE_DIRECT

# Track per-arm pose for ARM_JOINT_CMD → MANIPULATION_REFERENCE conversion
_last_arm_r = IDLE_R.copy()
_last_arm_l = IDLE_L.copy()

# Whether the manipulation thread was running before we disabled it
_manip_thread_was_running = False
_command_thread_was_running = False


# ═══════════════════════════════════════════════════════════════════════
# Screen-based thread control  (WBC mode only)
# ═══════════════════════════════════════════════════════════════════════

def _kill_screen_window(session: str, window: str):
    """Send Ctrl-C to a screen window to stop the process inside it.

    Equivalent to:
        for i in $(seq 1 5); do
            screen -S <session> -p <window> -X stuff '^C'
            sleep 0.1
        done
    """
    for _ in range(5):
        try:
            subprocess.run(
                ['screen', '-S', session, '-p', window, '-X', 'stuff', '\x03'],
                timeout=2, capture_output=True,
            )
        except Exception:
            pass
        time.sleep(0.1)


def _restart_screen_window(session: str, window: str, command: str):
    """Restart a command inside an existing screen window."""
    try:
        subprocess.run(
            ['screen', '-S', session, '-p', window, '-X', 'stuff',
             f'{command}\n'],
            timeout=2, capture_output=True,
        )
    except Exception as e:
        print(f"[server] ⚠ Could not restart {window}: {e}")


def kill_conflicting_threads():
    """Kill the manipulation and command screen windows.

    These threads continuously write to MANIPULATION_REFERENCE and
    ARM_JOINT_COMMAND respectively, which conflicts with our external
    arm commands via MANIPULATION_REFERENCE.
    """
    global _manip_thread_was_running, _command_thread_was_running

    print("[server] Killing 'manipulation' screen window …")
    _kill_screen_window('themis', 'manipulation')
    print("[server] ✓ manipulation killed")

    print("[server] Killing 'command' screen window …")
    _kill_screen_window('themis', 'command')
    print("[server] ✓ command killed")

    _manip_thread_was_running = True
    _command_thread_was_running = True

    # Also disable via shared memory as a safety net
    try:
        MM.THREAD_COMMAND.set({'manipulation': np.array([0.0])}, opt='update')
    except Exception:
        pass

    time.sleep(0.5)  # let processes stop

def disable_manipulation_thread():
    """Disable the onboard manipulation manager via shared memory.

    The manipulation manager is a compiled binary that continuously
    writes to MANIPULATION_REFERENCE.  We set THREAD_COMMAND.manipulation
    to 0 so it stops, allowing our external commands through.

    NOTE: kill_conflicting_threads() also kills it via screen ^C,
    this is an additional safety net.
    """
    try:
        MM.THREAD_COMMAND.set({'manipulation': np.array([0.0])}, opt='update')
        print("[server] ✓ Manipulation thread disabled (SHM)")
        return True
    except Exception as e:
        print(f"[server] ⚠ Could not disable manipulation thread: {e}")
        traceback.print_exc()
        return False


def enable_manipulation_thread():
    """Re-enable the manipulation thread (cleanup on exit)."""
    try:
        MM.THREAD_COMMAND.set({'manipulation': np.array([1.0])}, opt='update')
        print("[server] ✓ Manipulation thread re-enabled (SHM)")
    except Exception as e:
        print(f"[server] ⚠ Could not re-enable manipulation thread: {e}")


# ═══════════════════════════════════════════════════════════════════════
# State response packing  (dual-mode)
# ═══════════════════════════════════════════════════════════════════════

def pack_state_response() -> bytes:
    """Read joint states and pack STATE_RESPONSE (712 B)."""
    if _server_mode == MODE_WBC:
        return _pack_state_wbc()
    else:
        return _pack_state_direct()


def _pack_state_wbc() -> bytes:
    """Read state via wbc_api."""
    try:
        ra_q, ra_dq, ra_u = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
    except Exception:
        ra_q = ra_dq = ra_u = np.zeros(7)
    try:
        la_q, la_dq, la_u = wbc_api.get_joint_states(CHAIN_LEFT_ARM)
    except Exception:
        la_q = la_dq = la_u = np.zeros(7)

    ra_temp = ra_volt = la_temp = la_volt = np.zeros(7)

    try:
        imu_accel, imu_gyro, imu_R = wbc_api.get_imu_states()
    except Exception:
        imu_accel = np.zeros(3)
        imu_gyro  = np.zeros(3)
        imu_R     = np.eye(3)

    base_pos = np.zeros(3)

    return _pack_state_buf(ra_q, ra_dq, ra_u, la_q, la_dq, la_u,
                           ra_temp, ra_volt, la_temp, la_volt,
                           base_pos, imu_R, imu_accel, imu_gyro)


def _pack_state_direct() -> bytes:
    """Read state from shared memory."""
    try:
        ra = MM.RIGHT_ARM_JOINT_STATE.get()
        ra_q  = ra['joint_positions']
        ra_dq = ra['joint_velocities']
        ra_u  = ra['joint_torques']
    except Exception:
        ra_q = ra_dq = ra_u = np.zeros(7)

    try:
        la = MM.LEFT_ARM_JOINT_STATE.get()
        la_q  = la['joint_positions']
        la_dq = la['joint_velocities']
        la_u  = la['joint_torques']
    except Exception:
        la_q = la_dq = la_u = np.zeros(7)

    try:
        ra_bear = MM.RIGHT_ARM_BEAR_STATE.get()
        la_bear = MM.LEFT_ARM_BEAR_STATE.get()
        ra_temp = ra_bear['bear_temperatures']
        ra_volt = ra_bear['bear_voltages']
        la_temp = la_bear['bear_temperatures']
        la_volt = la_bear['bear_voltages']
    except Exception:
        ra_temp = ra_volt = la_temp = la_volt = np.zeros(7)

    try:
        bs = MM.BASE_STATE.get()
        base_pos = bs['base_position']
        imu_R    = bs['base_rotation_matrix']
    except Exception:
        base_pos = np.zeros(3)
        imu_R    = np.eye(3)

    try:
        ss = MM.SENSE_STATE.get()
        imu_accel = ss['imu_acceleration']
        imu_gyro  = ss['imu_angular_rate']
    except Exception:
        imu_accel = np.zeros(3)
        imu_gyro  = np.zeros(3)

    return _pack_state_buf(ra_q, ra_dq, ra_u, la_q, la_dq, la_u,
                           ra_temp, ra_volt, la_temp, la_volt,
                           base_pos, imu_R, imu_accel, imu_gyro)


def _pack_state_buf(ra_q, ra_dq, ra_u, la_q, la_dq, la_u,
                    ra_temp, ra_volt, la_temp, la_volt,
                    base_pos, imu_R, imu_accel, imu_gyro) -> bytes:
    """Pack state into the common wire format (712 B)."""
    buf = struct.pack('B', MSG_STATE_RESPONSE)
    for arr in [ra_q, ra_dq, ra_u, la_q, la_dq, la_u,
                ra_temp, ra_volt, la_temp, la_volt]:
        buf += np.asarray(arr, dtype=np.float64).ravel()[:7].tobytes()
    buf += np.asarray(base_pos, dtype=np.float64).ravel()[:3].tobytes()
    buf += np.asarray(imu_R,    dtype=np.float64).ravel()[:9].tobytes()
    buf += np.asarray(imu_accel, dtype=np.float64).ravel()[:3].tobytes()
    buf += np.asarray(imu_gyro,  dtype=np.float64).ravel()[:3].tobytes()
    buf += np.array([time.time()], dtype=np.float64).tobytes()
    return buf


# ═══════════════════════════════════════════════════════════════════════
# Arm joint command handling  (dual-mode)
# ═══════════════════════════════════════════════════════════════════════

def handle_arm_joint_cmd(payload: bytes):
    """Route arm command based on server mode.

    WBC mode:   Extract q and write to MANIPULATION_REFERENCE (POSE+SWING).
                The manipulation thread must be disabled first.
                The WBC reads this reference through its normal pipeline
                — no conflict with its own arm control loop.

    Direct mode: Write to MM.*_ARM_JOINT_COMMAND (q, dq, u, kp, kd).
    """
    global _last_arm_r, _last_arm_l

    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 35:
        print(f"[server] ARM_JOINT_CMD payload mismatch: {data.size}")
        return

    q   = data[0:7].copy()
    dq  = data[7:14].copy()
    u   = data[14:21].copy()
    kp  = data[21:28].copy()
    kd  = data[28:35].copy()

    if _server_mode == MODE_WBC:
        # WBC mode: route to MANIPULATION_REFERENCE
        if side == SIDE_RIGHT:
            _last_arm_r = q.copy()
            chain = CHAIN_RIGHT_ARM
        elif side == SIDE_LEFT:
            _last_arm_l = q.copy()
            chain = CHAIN_LEFT_ARM
        else:
            print(f"[server] Unknown arm side: 0x{side:02X}")
            return
        _write_manip_ref(_last_arm_r, _last_arm_l)
        # Also apply via set_joint_states for more responsive control
        try:
            wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
        except Exception as e:
            print(f"[server] set_joint_states (arm) failed: {e}")

    else:
        # Direct mode: write to JOINT_COMMAND shared memory
        command = {
            'goal_joint_positions':    q,
            'goal_joint_velocities':   dq,
            'goal_joint_torques':      u,
            'bear_enable_statuses':    np.ones(7),
            'bear_operating_modes':    np.full(7, 3.0),  # force/PD mode
            'bear_proportional_gains': kp,
            'bear_derivative_gains':   kd,
        }
        try:
            if side == SIDE_RIGHT:
                MM.RIGHT_ARM_JOINT_COMMAND.set(command)
            elif side == SIDE_LEFT:
                MM.LEFT_ARM_JOINT_COMMAND.set(command)
            else:
                print(f"[server] Unknown arm side: 0x{side:02X}")
        except Exception as e:
            print(f"[server] ARM_JOINT_COMMAND write failed: {e}")


def handle_manip_ref(payload: bytes):
    """Write manipulation reference — works in both modes.

    WBC mode:   Write to MANIPULATION_REFERENCE (WBC reads it).
    Direct mode: Extract arm poses and write to JOINT_COMMAND
                 with default PD gains (200/2), since nothing
                 reads MANIPULATION_REFERENCE without WBC.
    """
    global _last_arm_r, _last_arm_l

    data = np.frombuffer(payload, dtype=np.float64)
    if data.size != 32:
        print(f"[server] MANIP_REF payload mismatch: {data.size}")
        return

    # Track per-arm state
    _last_arm_r = data[0:7].copy()
    _last_arm_l = data[16:23].copy()

    print(f"[server] MANIP_REF: {data[30:32]}")
    
    if _server_mode == MODE_WBC:
        # WBC mode: write full reference to shared memory
        ref = {
            'right_arm_pose':           data[0:7].copy(),
            'right_arm_pose_rate':      data[7:14].copy(),
            'right_manipulation_mode':  data[14:15].copy(),
            'right_manipulation_phase': data[15:16].copy(),
            'left_arm_pose':            data[16:23].copy(),
            'left_arm_pose_rate':       data[23:30].copy(),
            'left_manipulation_mode':   data[30:31].copy(),
            'left_manipulation_phase':  data[31:32].copy(),
        }
        
        try:
            MM.MANIPULATION_REFERENCE.set(ref, opt='update')
        except Exception as e:
            print(f"[server] MANIPULATION_REFERENCE.set() failed: {e}")
        # Also apply via set_joint_states for more responsive control
        q_r  = data[0:7].copy()
        dq_r = data[7:14].copy()
        q_l  = data[16:23].copy()
        dq_l = data[23:30].copy()
        _zeros7 = np.zeros(7, dtype=np.float64)
        try:
            wbc_api.set_joint_states(CHAIN_RIGHT_ARM, _zeros7, q_r, dq_r,
                                    DEFAULT_ARM_KP, DEFAULT_ARM_KD)
            wbc_api.set_joint_states(CHAIN_LEFT_ARM,  _zeros7, q_l, dq_l,
                                    DEFAULT_ARM_KP, DEFAULT_ARM_KD)
        except Exception as e:
            print(f"[server] set_joint_states (arm) failed: {e}")

    else:
        # Direct mode: extract arm poses → JOINT_COMMAND
        q_r   = data[0:7].copy()
        dq_r  = data[7:14].copy()     # pose rate as desired velocity
        q_l   = data[16:23].copy()
        dq_l  = data[23:30].copy()

        for side_byte, q, dq in [(SIDE_RIGHT, q_r, dq_r),
                                  (SIDE_LEFT,  q_l, dq_l)]:
            command = {
                'goal_joint_positions':    q,
                'goal_joint_velocities':   dq,
                'goal_joint_torques':      np.zeros(7, dtype=np.float64),
                'bear_enable_statuses':    np.ones(7),
                'bear_operating_modes':    np.full(7, 3.0),
                'bear_proportional_gains': DEFAULT_ARM_KP.copy(),
                'bear_derivative_gains':   DEFAULT_ARM_KD.copy(),
            }
            try:
                if side_byte == SIDE_RIGHT:
                    MM.RIGHT_ARM_JOINT_COMMAND.set(command)
                else:
                    MM.LEFT_ARM_JOINT_COMMAND.set(command)
            except Exception as e:
                print(f"[server] ARM_JOINT_COMMAND write failed: {e}")


def _write_manip_ref(q_r, q_l):
    """Write both arms to MANIPULATION_REFERENCE with POSE + SWING."""
    ref = {
        'right_arm_pose':           np.asarray(q_r, dtype=np.float64),
        'right_arm_pose_rate':      np.zeros(7, dtype=np.float64),
        'right_manipulation_mode':  np.array([MANIP_MODE_POSE]),
        'right_manipulation_phase': np.array([MANIP_PHASE_SWING]),
        'left_arm_pose':            np.asarray(q_l, dtype=np.float64),
        'left_arm_pose_rate':       np.zeros(7, dtype=np.float64),
        'left_manipulation_mode':   np.array([MANIP_MODE_POSE]),
        'left_manipulation_phase':  np.array([MANIP_PHASE_SWING]),
    }
    try:
        MM.MANIPULATION_REFERENCE.set(ref, opt='update')
    except Exception as e:
        print(f"[server] MANIPULATION_REFERENCE.set() failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Hand state response  (dual-mode)
# ═══════════════════════════════════════════════════════════════════════

def pack_hand_state_response() -> bytes:
    """Read hand (DXL) joint states."""
    if _server_mode == MODE_WBC:
        try:
            rh_q, rh_dq, rh_u = wbc_api.get_joint_states(CHAIN_RIGHT_HAND)
        except Exception:
            rh_q = rh_dq = rh_u = np.zeros(7)
        try:
            lh_q, lh_dq, lh_u = wbc_api.get_joint_states(CHAIN_LEFT_HAND)
        except Exception:
            lh_q = lh_dq = lh_u = np.zeros(7)
    else:
        try:
            rh = MM.RIGHT_HAND_JOINT_STATE.get()
            rh_q  = rh['joint_positions']
            rh_dq = rh['joint_velocities']
            rh_u  = rh['joint_torques']
        except Exception:
            rh_q = rh_dq = rh_u = np.zeros(7)
        try:
            lh = MM.LEFT_HAND_JOINT_STATE.get()
            lh_q  = lh['joint_positions']
            lh_dq = lh['joint_velocities']
            lh_u  = lh['joint_torques']
        except Exception:
            lh_q = lh_dq = lh_u = np.zeros(7)

    buf = struct.pack('B', MSG_HAND_STATE_RESPONSE)
    for arr in [rh_q, rh_dq, rh_u, lh_q, lh_dq, lh_u]:
        buf += np.asarray(arr, dtype=np.float64).ravel()[:7].tobytes()
    buf += np.array([time.time()], dtype=np.float64).tobytes()
    return buf


# ═══════════════════════════════════════════════════════════════════════
# Hand / head / base command handling  (dual-mode)
# ═══════════════════════════════════════════════════════════════════════

def handle_hand_joint_cmd(payload: bytes):
    """Send hand command via wbc_api (WBC) or MM (direct)."""
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 35:
        print(f"[server] HAND_JOINT_CMD payload mismatch: {data.size}")
        return

    q   = data[0:7].copy()
    dq  = data[7:14].copy()
    u   = data[14:21].copy()
    kp  = data[21:28].copy()
    kd  = data[28:35].copy()

    if _server_mode == MODE_WBC:
        if side == SIDE_RIGHT_HAND:
            chain = CHAIN_RIGHT_HAND
        elif side == SIDE_LEFT_HAND:
            chain = CHAIN_LEFT_HAND
        else:
            print(f"[server] Unknown hand side: 0x{side:02X}")
            return
        try:
            wbc_api.set_joint_states(chain, u, q, dq, kp, kd)
        except Exception as e:
            print(f"[server] set_joint_states (hand) failed: {e}")

    else:
        # Direct mode: write to hand JOINT_COMMAND shared memory
        command = {
            'goal_joint_positions':    q,
            'goal_joint_velocities':   dq,
            'goal_joint_torques':      u,
            'dxl_enable_statuses':     np.ones(7),
            'dxl_operating_modes':     np.full(7, 5.0),
            'dxl_proportional_gains':  kp,
            'dxl_derivative_gains':    kd,
        }
        try:
            if side == SIDE_RIGHT_HAND:
                MM.RIGHT_HAND_JOINT_COMMAND.set(command)
            elif side == SIDE_LEFT_HAND:
                MM.LEFT_HAND_JOINT_COMMAND.set(command)
            else:
                print(f"[server] Unknown hand side: 0x{side:02X}")
        except Exception as e:
            print(f"[server] HAND_JOINT_COMMAND write failed: {e}")


def handle_head_joint_cmd(payload: bytes):
    """Send head command via wbc_api (WBC) or MM (direct)."""
    side = payload[0]
    data = np.frombuffer(payload[1:], dtype=np.float64)
    if data.size != 10:
        print(f"[server] HEAD_JOINT_CMD payload mismatch: {data.size}")
        return

    q   = data[0:2].copy()
    dq  = data[2:4].copy()
    u   = data[4:6].copy()
    kp  = data[6:8].copy()
    kd  = data[8:10].copy()

    if side != SIDE_HEAD:
        print(f"[server] Unknown head side: 0x{side:02X} (expected 0x00)")
        return

    if _server_mode == MODE_WBC:
        try:
            wbc_api.set_joint_states(CHAIN_HEAD, u, q, dq, kp, kd)
        except Exception as e:
            print(f"[server] set_joint_states (head) failed: {e}")
    else:
        # Direct mode: try shared memory (may not exist in all AOS versions)
        try:
            command = {
                'goal_joint_positions':    q,
                'goal_joint_velocities':   dq,
                'goal_joint_torques':      u,
                'dxl_enable_statuses':     np.ones(2),
                'dxl_operating_modes':     np.full(2, 5.0),
                'dxl_proportional_gains':  kp,
                'dxl_derivative_gains':    kd,
            }
            MM.HEAD_JOINT_COMMAND.set(command)
        except Exception:
            pass  # HEAD_JOINT_COMMAND may not exist


def handle_base_orient(payload: bytes):
    """Send base orientation via locomotion API (WBC mode only)."""
    if lm_api is None:
        return  # silently skip — no WBC or no locomotion API

    data = np.frombuffer(payload, dtype=np.float64)
    if data.size != 3:
        print(f"[server] BASE_ORIENT payload mismatch: {data.size}")
        return

    try:
        lm_api.set_base_orientation(float(data[0]), float(data[1]), float(data[2]))
    except Exception as e:
        print(f"[server] set_base_orientation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Main server loop
# ═══════════════════════════════════════════════════════════════════════

def run_server(host: str, port: int, mode: str = 'auto',
               log_enabled: bool = False):
    """Main server loop with auto-detection and dual-mode routing."""
    global _server_mode, _last_arm_r, _last_arm_l

    # ── Detect mode ──────────────────────────────────────────────────
    if mode == 'wbc':
        if wbc_api is None:
            print("[server] ERROR: --mode wbc requested but wbc_api not available")
            sys.exit(1)
        _server_mode = MODE_WBC
    elif mode == 'direct':
        _server_mode = MODE_DIRECT
    else:  # auto
        _server_mode = MODE_WBC if wbc_api is not None else MODE_DIRECT

    mode_str = "WBC" if _server_mode == MODE_WBC else "DIRECT"

    # ── Banner ───────────────────────────────────────────────────────
    print("=" * 62)
    print("  THEMIS Unified UDP Server")
    print("=" * 62)
    print(f"  Mode:      {mode_str}  (detected: {'auto' if mode == 'auto' else 'forced'})")
    print(f"  AOS path:  {AOS_PATH}")
    if _server_mode == MODE_WBC:
        print(f"  Arms:      MM.MANIPULATION_REFERENCE  (POSE + SWING)")
        print(f"             Manipulation thread disabled to avoid conflict")
        print(f"  Hands:     wbc_api.set_joint_states()")
        print(f"  Base lean: {'lm_api ✓' if lm_api else 'NOT available'}")
    else:
        print(f"  Arms:      MM.*_ARM_JOINT_COMMAND  (direct)")
        print(f"  Hands:     MM.*_HAND_JOINT_COMMAND  (direct)")
        print(f"  Base lean: NOT available (no WBC)")
    if log_enabled:
        print(f"  Logging:   ENABLED")
    print("=" * 62)

    # ── Connect shared memory ────────────────────────────────────────
    print("\n[server] Connecting to shared memory …")
    try:
        MM.connect()
        print("[server] ✓ Shared memory connected")
    except Exception as e:
        print(f"[server] ✗ MM.connect() failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Verify API access ────────────────────────────────────────────
    if _server_mode == MODE_WBC:
        print("[server] Verifying wbc_api …")
        try:
            ra_q, ra_dq, ra_u = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
            print(f"[server] ✓ Right arm q = {np.degrees(np.asarray(ra_q)).round(1)}")
        except Exception as e:
            print(f"[server] ✗ wbc_api failed: {e}")
            if mode != 'wbc':  # only fall back in auto mode
                print("[server]   Falling back to DIRECT mode")
                _server_mode = MODE_DIRECT
                mode_str = "DIRECT"
            else:
                traceback.print_exc()
                sys.exit(1)

    if _server_mode == MODE_DIRECT:
        print("[server] Verifying direct shared memory …")
        try:
            ra = MM.RIGHT_ARM_JOINT_STATE.get()
            print(f"[server] ✓ Right arm q = {np.degrees(ra['joint_positions']).round(1)}")
        except Exception as e:
            print(f"[server] ✗ MM read failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ── Initialize arm pose tracking ─────────────────────────────────
    try:
        if _server_mode == MODE_WBC:
            ra_q, _, _ = wbc_api.get_joint_states(CHAIN_RIGHT_ARM)
            la_q, _, _ = wbc_api.get_joint_states(CHAIN_LEFT_ARM)
        else:
            ra = MM.RIGHT_ARM_JOINT_STATE.get()
            la = MM.LEFT_ARM_JOINT_STATE.get()
            ra_q = ra['joint_positions']
            la_q = la['joint_positions']
        _last_arm_r = np.asarray(ra_q, dtype=np.float64).copy()
        _last_arm_l = np.asarray(la_q, dtype=np.float64).copy()
        print(f"[server] ✓ Initial arm poses captured")
    except Exception as e:
        print(f"[server] ⚠ Could not read initial arm poses: {e}")

    # ── Kill conflicting threads (WBC mode only) ─────────────────────
    if _server_mode == MODE_WBC:
        kill_conflicting_threads()
        disable_manipulation_thread()   # SHM safety net

        # Seed MANIPULATION_REFERENCE with current arm poses
        _write_manip_ref(_last_arm_r, _last_arm_l)
        print("[server] ✓ MANIPULATION_REFERENCE initialized with current pose")

    print(f"\n[server] Running in {mode_str} mode ✓\n")

    # ── Server-side logging (optional) ───────────────────────────────
    LOG_MAX = 300_000   # ~5 min at 1 kHz
    if log_enabled:
        log_recv_t = np.zeros(LOG_MAX)
        log_q_r    = np.zeros((LOG_MAX, 7))
        log_q_l    = np.zeros((LOG_MAX, 7))
        log_idx = 0

    # ── UDP socket ───────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"[server] UDP listening on {host}:{port}")
    print("[server] Waiting for desktop connection …\n")

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
                    print("[server] No heartbeat for 30 s — still waiting …")
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

            elif msg_type == MSG_ARM_JOINT_CMD:
                handle_arm_joint_cmd(payload)
                cmd_count += 1
                if log_enabled and log_idx < LOG_MAX:
                    log_recv_t[log_idx] = time.perf_counter()
                    log_q_r[log_idx] = _last_arm_r
                    log_q_l[log_idx] = _last_arm_l
                    log_idx += 1

            elif msg_type == MSG_MANIP_REF:
                handle_manip_ref(payload)
                cmd_count += 1
                if log_enabled and log_idx < LOG_MAX:
                    log_recv_t[log_idx] = time.perf_counter()
                    log_q_r[log_idx] = _last_arm_r
                    log_q_l[log_idx] = _last_arm_l
                    log_idx += 1

            elif msg_type == MSG_HAND_JOINT_CMD:
                handle_hand_joint_cmd(payload)
                cmd_count += 1

            elif msg_type == MSG_HEAD_JOINT_CMD:
                handle_head_joint_cmd(payload)
                cmd_count += 1

            elif msg_type == MSG_BASE_ORIENT:
                handle_base_orient(payload)
                cmd_count += 1

            elif msg_type == MSG_MODE_QUERY:
                resp = struct.pack('BB', MSG_MODE_RESPONSE, _server_mode)
                sock.sendto(resp, addr)

            elif msg_type == MSG_HEARTBEAT:
                last_heartbeat = time.time()
                sock.sendto(struct.pack('BB', MSG_ACK, 0x00), addr)

            else:
                print(f"[server] Unknown msg_type 0x{msg_type:02X} from {addr}")

            pkt_count += 1

            # Periodic stats
            if time.time() - last_stats >= 5.0:
                elapsed = time.time() - last_stats
                pps = pkt_count / elapsed
                cps = cmd_count / elapsed
                print(f"[server] {pps:.0f} pkt/s | {cps:.0f} cmd/s | "
                      f"mode={mode_str}")
                pkt_count = 0
                cmd_count = 0
                last_stats = time.time()

    except KeyboardInterrupt:
        print("\n[server] Shutting down …")
    finally:
        sock.close()

        # Re-enable threads if we disabled them
        if _server_mode == MODE_WBC:
            enable_manipulation_thread()
            # Note: manipulation and command screen processes were killed.
            # They need to be manually restarted if needed.
            print("[server] ⚠ 'manipulation' and 'command' screen windows were killed.")
            print("[server]   Restart them manually if needed (re-enter the screen).")

        # Save log
        if log_enabled and log_idx > 0:
            from datetime import datetime as _dt
            ts = _dt.now().strftime('%Y%m%d_%H%M%S')
            fname = f'server_log_{ts}.npz'
            np.savez_compressed(fname,
                                recv_t=log_recv_t[:log_idx],
                                q_r=log_q_r[:log_idx],
                                q_l=log_q_l[:log_idx])
            print(f"[server] Log saved → {fname}  ({log_idx} entries)")

        print("[server] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Themis Unified UDP Server (auto-detects WBC)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9870,
                        help="UDP port (default: 9870)")
    parser.add_argument("--mode", choices=['auto', 'wbc', 'direct'],
                        default='auto',
                        help="Operating mode: auto-detect (default), "
                             "force WBC, or force direct SHM")
    parser.add_argument("--log", action="store_true",
                        help="Enable server-side command logging "
                             "(saves server_log_*.npz on Ctrl-C)")
    args = parser.parse_args()
    run_server(args.host, args.port, mode=args.mode, log_enabled=args.log)
