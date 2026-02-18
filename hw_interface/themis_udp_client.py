#!/usr/bin/env python3
"""
Themis UDP Client — Desktop-side communication layer.

Provides a clean Python API for the desktop to talk to the robot over
the UDP bridge (shm_udp_server.py running on the robot's PC).

Usage:
    client = ThemisUDPClient(robot_ip="192.168.1.100", port=9870)
    client.connect()

    # Read feedback
    fb = client.get_state()
    print(fb.right_arm_q)

    # Send arm commands
    client.send_arm_command(side='right', q=desired_q, kp=200*np.ones(7), kd=2*np.ones(7))

    # Send manipulation reference (via WBC)
    client.send_manip_reference(right_arm_pose=q_r, left_arm_pose=q_l, mode=100, phase=1)
"""

import socket
import struct
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Message types (must match shm_udp_server.py) ────────────────────
MSG_STATE_REQUEST     = 0x01
MSG_STATE_RESPONSE    = 0x02
MSG_ARM_JOINT_CMD     = 0x10
MSG_MANIP_REF         = 0x11
MSG_HEARTBEAT         = 0x20
MSG_ACK               = 0xFE

SIDE_RIGHT = 2      # +2 in the AOS convention
SIDE_LEFT  = 0xFE   # -2 stored as unsigned byte


@dataclass
class ThemisStateFeedback:
    """Parsed robot state feedback from the UDP bridge."""
    timestamp: float = 0.0
    valid: bool = False

    # Arm joint states (7 DOF each)
    # Order: shoulder_pitch, shoulder_roll, shoulder_yaw,
    #        elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw
    right_arm_q:      np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    right_arm_dq:     np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    right_arm_torque: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_q:       np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_dq:      np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_torque:  np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    # Base state
    base_position:    np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_rot_matrix:  np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    # Motor diagnostics (from BEAR_STATE)
    right_arm_temp:   np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    right_arm_volt:   np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_temp:    np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_volt:    np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    # IMU
    imu_accel:        np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    imu_gyro:         np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


class ThemisUDPClient:
    """Desktop-side UDP client for Themis robot communication."""

    # ── Arm joint names for pretty-printing ──────────────────────────
    ARM_JOINT_NAMES = [
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
        "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw",
    ]

    # ── Default PD gains (from bear_macros.py) ───────────────────────
    # Shoulder pitch has gear ratio 9, rest have gear ratio 20
    DEFAULT_KP = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    DEFAULT_KD = np.array([  2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0])

    # ── Nominal arm poses (from manipulation_macros.py) ──────────────
    IDLE_POSE_R  = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
    IDLE_POSE_L  = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

    def __init__(self, robot_ip: str = "192.168.0.11", port: int = 9870,
                 timeout: float = 0.5):
        self.robot_ip = robot_ip
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

        # Cached state
        self._state_lock = threading.Lock()
        self._last_state = ThemisStateFeedback()

        # Heartbeat thread
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False

    # ─────────────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────────────
    def connect(self):
        """Open UDP socket and start heartbeat."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        self._running = True
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()
        print(f"[ThemisClient] Connected to {self.robot_ip}:{self.port}")

    def disconnect(self):
        """Close socket and stop heartbeat."""
        self._running = False
        if self._hb_thread:
            self._hb_thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
            self.sock = None
        print("[ThemisClient] Disconnected")

    def _heartbeat_loop(self):
        """Send heartbeat every 2 seconds."""
        while self._running:
            try:
                self._send(struct.pack('B', MSG_HEARTBEAT))
            except Exception:
                pass
            time.sleep(2.0)

    # ─────────────────────────────────────────────────────────────────
    # Low-level send / receive
    # ─────────────────────────────────────────────────────────────────
    def _send(self, data: bytes):
        if self.sock is None:
            raise RuntimeError("Not connected")
        self.sock.sendto(data, (self.robot_ip, self.port))

    def _recv(self, bufsize: int = 4096) -> bytes:
        if self.sock is None:
            raise RuntimeError("Not connected")
        data, _ = self.sock.recvfrom(bufsize)
        return data

    # ─────────────────────────────────────────────────────────────────
    # State feedback
    # ─────────────────────────────────────────────────────────────────
    def get_state(self) -> ThemisStateFeedback:
        """Request and receive robot state feedback (blocking)."""
        self._send(struct.pack('B', MSG_STATE_REQUEST))
        try:
            data = self._recv()
        except socket.timeout:
            fb = ThemisStateFeedback()
            fb.valid = False
            return fb

        if len(data) < 2 or data[0] != MSG_STATE_RESPONSE:
            fb = ThemisStateFeedback()
            fb.valid = False
            return fb

        return self._parse_state_response(data[1:])

    def _parse_state_response(self, payload: bytes) -> ThemisStateFeedback:
        """Parse STATE_RESPONSE payload into ThemisStateFeedback."""
        fb = ThemisStateFeedback()
        arr = np.frombuffer(payload, dtype=np.float64)

        # Expected: 7*6 + 7*6 + 7*2 + 7*2 + 3+9 + 3+3 + 1 = 89 doubles
        if arr.size < 89:
            fb.valid = False
            return fb

        i = 0
        fb.right_arm_q      = arr[i:i+7].copy(); i += 7
        fb.right_arm_dq     = arr[i:i+7].copy(); i += 7
        fb.right_arm_torque = arr[i:i+7].copy(); i += 7
        fb.left_arm_q       = arr[i:i+7].copy(); i += 7
        fb.left_arm_dq      = arr[i:i+7].copy(); i += 7
        fb.left_arm_torque  = arr[i:i+7].copy(); i += 7
        fb.right_arm_temp   = arr[i:i+7].copy(); i += 7
        fb.right_arm_volt   = arr[i:i+7].copy(); i += 7
        fb.left_arm_temp    = arr[i:i+7].copy(); i += 7
        fb.left_arm_volt    = arr[i:i+7].copy(); i += 7
        fb.base_position    = arr[i:i+3].copy(); i += 3
        fb.base_rot_matrix  = arr[i:i+9].copy().reshape(3, 3); i += 9
        fb.imu_accel        = arr[i:i+3].copy(); i += 3
        fb.imu_gyro         = arr[i:i+3].copy(); i += 3
        fb.timestamp        = float(arr[i]); i += 1
        fb.valid = True
        return fb

    # ─────────────────────────────────────────────────────────────────
    # Arm joint commands (direct to actuators via JOINT_COMMAND)
    # ─────────────────────────────────────────────────────────────────
    def send_arm_command(self, side: str,
                         q: np.ndarray,
                         dq: Optional[np.ndarray] = None,
                         u: Optional[np.ndarray] = None,
                         kp: Optional[np.ndarray] = None,
                         kd: Optional[np.ndarray] = None) -> bool:
        """
        Send arm joint command (goes to JOINT_COMMAND via force-mode PD).

        Parameters
        ----------
        side : 'right' or 'left'
        q    : (7,) goal joint positions [rad]
        dq   : (7,) goal joint velocities [rad/s], default zeros
        u    : (7,) feedforward torques [Nm], default zeros
        kp   : (7,) proportional gains, default [200]*7
        kd   : (7,) derivative gains, default [2]*7

        Returns
        -------
        True if ACK received, False on timeout.
        """
        q  = np.asarray(q, dtype=np.float64).ravel()
        dq = np.zeros(7, dtype=np.float64) if dq is None else np.asarray(dq, dtype=np.float64).ravel()
        u  = np.zeros(7, dtype=np.float64) if u is None else np.asarray(u, dtype=np.float64).ravel()
        kp = self.DEFAULT_KP.copy() if kp is None else np.asarray(kp, dtype=np.float64).ravel()
        kd = self.DEFAULT_KD.copy() if kd is None else np.asarray(kd, dtype=np.float64).ravel()

        side_byte = SIDE_RIGHT if side == 'right' else SIDE_LEFT
        payload = struct.pack('B', side_byte) + q.tobytes() + dq.tobytes() + u.tobytes() + kp.tobytes() + kd.tobytes()
        self._send(struct.pack('B', MSG_ARM_JOINT_CMD) + payload)

        try:
            resp = self._recv()
            return len(resp) >= 2 and resp[0] == MSG_ACK
        except socket.timeout:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Manipulation reference (via WBC)
    # ─────────────────────────────────────────────────────────────────
    def send_manip_reference(self,
                              right_arm_pose: np.ndarray,
                              left_arm_pose: np.ndarray,
                              right_arm_rate: Optional[np.ndarray] = None,
                              left_arm_rate: Optional[np.ndarray] = None,
                              right_mode: float = 100.0,   # POSE
                              right_phase: float = 1.0,    # STANDBY
                              left_mode: float = 100.0,
                              left_phase: float = 1.0) -> bool:
        """
        Send manipulation reference (goes to MANIPULATION_REFERENCE for WBC).

        Parameters
        ----------
        right_arm_pose : (7,) desired joint angles for right arm [rad]
        left_arm_pose  : (7,) desired joint angles for left arm [rad]
        right_arm_rate : (7,) desired joint velocities (default zeros)
        left_arm_rate  : (7,) desired joint velocities (default zeros)
        right_mode     : manipulation mode (100=POSE)
        right_phase    : manipulation phase (1=STANDBY)
        left_mode      : manipulation mode
        left_phase     : manipulation phase

        Returns
        -------
        True if ACK received.
        """
        rp = np.asarray(right_arm_pose, dtype=np.float64).ravel()
        lp = np.asarray(left_arm_pose, dtype=np.float64).ravel()
        rr = np.zeros(7, dtype=np.float64) if right_arm_rate is None else np.asarray(right_arm_rate, dtype=np.float64).ravel()
        lr = np.zeros(7, dtype=np.float64) if left_arm_rate is None else np.asarray(left_arm_rate, dtype=np.float64).ravel()

        buf = np.concatenate([
            rp, rr, [right_mode, right_phase],
            lp, lr, [left_mode,  left_phase],
        ]).astype(np.float64)

        self._send(struct.pack('B', MSG_MANIP_REF) + buf.tobytes())

        try:
            resp = self._recv()
            return len(resp) >= 2 and resp[0] == MSG_ACK
        except socket.timeout:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────
    def print_state(self, fb: Optional[ThemisStateFeedback] = None):
        """Pretty-print the current robot state."""
        if fb is None:
            fb = self.get_state()
        if not fb.valid:
            print("[ThemisClient] No valid state feedback")
            return

        print(f"\n{'='*70}")
        print(f"  Themis Robot State  (t = {fb.timestamp:.3f})")
        print(f"{'='*70}")

        print(f"\n  Base position: [{fb.base_position[0]:+.3f}, {fb.base_position[1]:+.3f}, {fb.base_position[2]:+.3f}]")
        print(f"  IMU accel:     [{fb.imu_accel[0]:+.3f}, {fb.imu_accel[1]:+.3f}, {fb.imu_accel[2]:+.3f}]")
        print(f"  IMU gyro:      [{fb.imu_gyro[0]:+.3f}, {fb.imu_gyro[1]:+.3f}, {fb.imu_gyro[2]:+.3f}]")

        for label, q, dq, tau, temp, volt in [
            ("Right Arm", fb.right_arm_q, fb.right_arm_dq, fb.right_arm_torque, fb.right_arm_temp, fb.right_arm_volt),
            ("Left Arm",  fb.left_arm_q,  fb.left_arm_dq,  fb.left_arm_torque,  fb.left_arm_temp,  fb.left_arm_volt),
        ]:
            print(f"\n  {label}:")
            print(f"    {'Joint':<18s} {'Pos [deg]':>10s} {'Vel [rad/s]':>12s} {'Torque [Nm]':>12s} {'Temp [°C]':>10s} {'Volt [V]':>9s}")
            print(f"    {'─'*18} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*9}")
            for j in range(7):
                print(f"    {self.ARM_JOINT_NAMES[j]:<18s} {np.degrees(q[j]):>+10.2f} {dq[j]:>+12.3f} {tau[j]:>+12.3f} {temp[j]:>10.1f} {volt[j]:>9.2f}")
        print()
