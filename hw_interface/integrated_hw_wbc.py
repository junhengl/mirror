#!/usr/bin/env python3
"""
Integrated Body Tracking → Retargeting → Hardware Pipeline  (WBC-safe)

Uses the SAME safe command pattern as test_arm_wbc.py:
  • 1 kHz fire-and-forget UDP arm commands
  • wbc_api.set_joint_states() on the robot (WBC stays alive)
  • Separate-socket background feedback thread (never blocks cmd loop)
  • Spin-wait timing for <1 ms jitter

Architecture:
  ┌──────────── Desktop ────────────────────────────────────────────┐
  │                                                                 │
  │  [ZED Camera] ──30 Hz──▶ BodyTrackingNode                      │
  │                              │                                  │
  │                       ArmTrackingData                           │
  │                              │                                  │
  │                              ▼                                  │
  │                        RetargetingNode  ──500 Hz IK──▶ q_des    │
  │                              │                                  │
  │                       RetargetingOutput                         │
  │                              │                                  │
  │                              ▼                                  │
  │  ╔══════════════════════════════════════════════════════════╗    │
  │  ║  1 kHz Command Loop  (this script — NEVER blocks)       ║    │
  │  ║    • Reads latest q_des from SharedState (lock-free)    ║    │
  │  ║    • Applies joint mapping + safety limits              ║    │
  │  ║    • Sends fire-and-forget UDP (same as test_arm_wbc)   ║    │
  │  ╚══════════════════════════════════════════════════════════╝    │
  │                              │                                  │
  │                     fire-and-forget UDP                         │
  └──────────────────────────────│──────────────────────────────────┘
                                 │
  ┌──────────── Robot PC ────────│──────────────────────────────────┐
  │  wbc_udp_server.py  ◀───────┘                                  │
  │    wbc_api.set_joint_states(chain, u, q, dq, kp, kd)           │
  │    WBC stays in control ✓                                       │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  # 1. Robot PC — start the SAFE server:
  ssh themis@192.168.0.11
  cd /home/themis/THEMIS/THEMIS
  python3 ~/wbc_udp_server.py --port 9870

  # 2. Desktop — run with ZED camera:
  sudo .venv/bin/python hw_interface/integrated_hw_wbc.py

  # 3. Desktop — dry-run (no robot, dummy tracking):
  .venv/bin/python hw_interface/integrated_hw_wbc.py --dry-run

  # 4. Desktop — no camera, dummy tracking to real robot:
  .venv/bin/python hw_interface/integrated_hw_wbc.py --no-camera
"""

import os
import sys
import time
import signal
import struct
import socket
import argparse
import threading
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from real_time_sim.config import PipelineConfig
from real_time_sim.shared_state import (
    SharedState, RobotState, RobotFeedback, RetargetingOutput,
    ArmTrackingData, HandTrackingData,
)
from real_time_sim.nodes.body_tracking_node import BodyTrackingNode
from real_time_sim.nodes.retargeting_node import RetargetingNode
from real_time_sim.joint_mapping import JointMapping

from hw_interface.themis_udp_client import ThemisUDPClient, ThemisStateFeedback
from hw_interface.hw_visualizer import HardwareVisualizer


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
IDLE_R_HW = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L_HW = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

KP_SOFT = np.full(7, 20.0)
KD_SOFT = np.full(7,  2.0)

MSG_ARM_JOINT_CMD  = 0x10
MSG_HAND_JOINT_CMD = 0x12
MSG_HEAD_JOINT_CMD = 0x13
MSG_STATE_REQUEST  = 0x01
MSG_STATE_RESPONSE = 0x02
SIDE_RIGHT = 2
SIDE_LEFT  = 0xFE
SIDE_RIGHT_HAND = 3
SIDE_LEFT_HAND  = 0xFD
SIDE_HEAD = 0

# ── Recorded hand poses (from test_hand_open_close.py) ──────────────
#  Motor order: [f1_prox, f1_dist, f2_prox, f2_dist, f3_prox, f3_dist, split]
LEFT_FIST  = np.array([+1.5156, +0.8099, +1.5125, +0.8391, +0.0690, +0.6366, -2.6047])
LEFT_OPEN  = np.array([+1.0937, +0.4065, +1.1075, +0.4817, -0.3421, +0.4541, -2.6108])
RIGHT_FIST = np.array([+1.5723, +0.9327, +1.5493, +1.0293, -0.0782, +0.9342, -2.5203])
RIGHT_OPEN = np.array([+1.1735, +0.6872, +1.1827, +0.0353, -0.6703, +0.6489, -2.4743])

KP_HAND = np.full(7, 5.0)
KD_HAND = np.full(7, 0.5)

# ── Head: 2 motors, chain 0, hold at zero ────────────────────────────
HEAD_ZERO = np.zeros(2, dtype=np.float64)
KP_HEAD = np.full(2, 10.0)
KD_HEAD = np.full(2,  1.0)

# ── Hand smoothing ───────────────────────────────────────────────────
MAX_HAND_VEL     = 2.0     # rad/s per motor — rate limit on hand cmds

# Shutdown flag
_shutdown = False


def _sig(s, f):
    global _shutdown
    _shutdown = True
    print("\n[Main] Shutdown requested …")

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# ═══════════════════════════════════════════════════════════════════════
# Low-level helpers (identical to test_arm_wbc.py)
# ═══════════════════════════════════════════════════════════════════════

def _send_both_arms_ff(client, q_r, q_l, kp, kd):
    """Fire-and-forget both arms — never blocks."""
    dq = np.zeros(7, dtype=np.float64)
    u  = np.zeros(7, dtype=np.float64)
    for side_byte, q in [(SIDE_RIGHT, q_r), (SIDE_LEFT, q_l)]:
        q_  = np.asarray(q, dtype=np.float64).ravel()
        kp_ = np.asarray(kp, dtype=np.float64).ravel()
        kd_ = np.asarray(kd, dtype=np.float64).ravel()
        pkt = (struct.pack('B', MSG_ARM_JOINT_CMD)
               + struct.pack('B', side_byte)
               + q_.tobytes() + dq.tobytes() + u.tobytes()
               + kp_.tobytes() + kd_.tobytes())
        client._send(pkt)


def _send_both_hands_ff(client, q_r, q_l, kp, kd):
    """Fire-and-forget both hand commands — never blocks."""
    dq = np.zeros(7, dtype=np.float64)
    u  = np.zeros(7, dtype=np.float64)
    for side_byte, q in [(SIDE_RIGHT_HAND, q_r), (SIDE_LEFT_HAND, q_l)]:
        q_  = np.asarray(q, dtype=np.float64).ravel()
        kp_ = np.asarray(kp, dtype=np.float64).ravel()
        kd_ = np.asarray(kd, dtype=np.float64).ravel()
        pkt = (struct.pack('B', MSG_HAND_JOINT_CMD)
               + struct.pack('B', side_byte)
               + q_.tobytes() + dq.tobytes() + u.tobytes()
               + kp_.tobytes() + kd_.tobytes())
        client._send(pkt)


def _send_head_ff(client, q, kp, kd):
    """Fire-and-forget head command (chain 0, 2 motors) — never blocks."""
    q_  = np.asarray(q,  dtype=np.float64).ravel()
    dq  = np.zeros(2, dtype=np.float64)
    u   = np.zeros(2, dtype=np.float64)
    kp_ = np.asarray(kp, dtype=np.float64).ravel()
    kd_ = np.asarray(kd, dtype=np.float64).ravel()
    pkt = (struct.pack('B', MSG_HEAD_JOINT_CMD)
           + struct.pack('B', SIDE_HEAD)
           + q_.tobytes() + dq.tobytes() + u.tobytes()
           + kp_.tobytes() + kd_.tobytes())
    client._send(pkt)


def _spin_wait(target):
    """High-resolution busy-wait."""
    now = time.perf_counter()
    rem = target - now
    if rem <= 0:
        return
    if rem > 0.0005:
        time.sleep(rem - 0.0005)
    while time.perf_counter() < target:
        pass


def lerp(a, b, t):
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) * a + t * b


# ═══════════════════════════════════════════════════════════════════════
# Background feedback thread (separate UDP socket, never blocks cmd loop)
# ═══════════════════════════════════════════════════════════════════════

class FeedbackThread:
    """Reads robot state at ~20 Hz on its own UDP socket."""

    def __init__(self, robot_ip, port, client):
        self.robot_ip = robot_ip
        self.port = port
        self.client = client  # for _parse_state_response
        self._lock = threading.Lock()
        self._fb = ThemisStateFeedback()
        self._stop = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)

    def get(self):
        with self._lock:
            return self._fb

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        try:
            while not self._stop and not _shutdown:
                try:
                    sock.sendto(struct.pack('B', MSG_STATE_REQUEST),
                                (self.robot_ip, self.port))
                    data, _ = sock.recvfrom(4096)
                    if len(data) > 1 and data[0] == MSG_STATE_RESPONSE:
                        fb = self.client._parse_state_response(data[1:])
                        with self._lock:
                            self._fb = fb
                except socket.timeout:
                    pass
                except Exception:
                    pass
                time.sleep(0.05)  # ~20 Hz
        finally:
            sock.close()


# ═══════════════════════════════════════════════════════════════════════
# Dry-run mock client
# ═══════════════════════════════════════════════════════════════════════

class DryRunClient:
    robot_ip = "127.0.0.1"
    port = 9870

    def connect(self):
        print("[DryRun] Connected (no real robot)")

    def disconnect(self):
        print("[DryRun] Disconnected")

    def _send(self, data):
        pass

    def get_state(self):
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = IDLE_R_HW.copy()
        fb.left_arm_q  = IDLE_L_HW.copy()
        fb.right_arm_dq = np.zeros(7)
        fb.left_arm_dq  = np.zeros(7)
        fb.right_arm_torque = np.zeros(7)
        fb.left_arm_torque  = np.zeros(7)
        fb.base_position = np.zeros(3)
        fb.imu_accel = np.zeros(3)
        fb.imu_gyro  = np.zeros(3)
        fb.timestamp = time.time()
        return fb

    def _parse_state_response(self, payload):
        return self.get_state()

    def print_state(self, fb=None):
        if fb is None:
            fb = self.get_state()
        print(f"[DryRun] R: {np.degrees(fb.right_arm_q).round(1)}")
        print(f"[DryRun] L: {np.degrees(fb.left_arm_q).round(1)}")


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(client, shared, config, joint_mapping, fb_thread,
                 rate_hz=1000.0, blend_time=3.0, max_delta_per_s=2.0):
    """
    1 kHz command loop that reads retargeting output and sends to robot.

    Safety features (same as test_arm_wbc.py):
      • Ramp to IDLE on start
      • Smooth blend from IDLE to tracking
      • Per-step velocity clamp
      • Ramp back to IDLE on exit
    """
    global _shutdown
    dt = 1.0 / rate_hz
    max_delta = max_delta_per_s * dt  # rad per tick

    # ── Phase 1: Ramp to IDLE ────────────────────────────────────────
    print(f"\n[Phase 1] Ramping to IDLE over 3.0 s …")
    fb = fb_thread.get()
    if fb.valid:
        start_r = fb.right_arm_q.copy()
        start_l = fb.left_arm_q.copy()
    else:
        fb = client.get_state()
        start_r = fb.right_arm_q.copy() if fb.valid else IDLE_R_HW.copy()
        start_l = fb.left_arm_q.copy()  if fb.valid else IDLE_L_HW.copy()

    RAMP = 3.0
    t0 = time.perf_counter()
    tick = t0
    while not _shutdown:
        now = time.perf_counter()
        if now - t0 >= RAMP:
            break
        a = (now - t0) / RAMP
        _send_both_arms_ff(client, lerp(start_r, IDLE_R_HW, a),
                           lerp(start_l, IDLE_L_HW, a), KP_SOFT, KD_SOFT)
        # Ramp hands to FIST (from current → fist, same alpha)
        _send_both_hands_ff(client, lerp(RIGHT_OPEN, RIGHT_FIST, a),
                            lerp(LEFT_OPEN, LEFT_FIST, a), KP_HAND, KD_HAND)
        # Head to zero
        _send_head_ff(client, HEAD_ZERO, KP_HEAD, KD_HEAD)
        tick += dt
        _spin_wait(tick)

    # Hold IDLE 0.5 s
    t0 = time.perf_counter(); tick = t0
    while time.perf_counter() - t0 < 0.5 and not _shutdown:
        _send_both_arms_ff(client, IDLE_R_HW, IDLE_L_HW, KP_SOFT, KD_SOFT)
        _send_both_hands_ff(client, RIGHT_FIST, LEFT_FIST, KP_HAND, KD_HAND)
        _send_head_ff(client, HEAD_ZERO, KP_HEAD, KD_HEAD)
        tick += dt; _spin_wait(tick)

    if _shutdown:
        return

    print("[Phase 1] At IDLE — waiting for valid tracking …")

    # ── Phase 2: Tracking loop ───────────────────────────────────────
    last_cmd_r = IDLE_R_HW.copy()
    last_cmd_l = IDLE_L_HW.copy()
    last_hand_r = RIGHT_FIST.copy()
    last_hand_l = LEFT_FIST.copy()
    max_hand_delta = MAX_HAND_VEL * dt  # per-tick hand motor limit
    blend_started = False
    blend_t0 = 0.0
    blend_start_r = IDLE_R_HW.copy()
    blend_start_l = IDLE_L_HW.copy()

    tick = time.perf_counter()
    last_print = time.perf_counter()
    step = 0
    tracking_count = 0

    print("[Phase 2] Tracking active — 1 kHz command loop running\n")

    while not _shutdown:
        now = time.perf_counter()
        step += 1

        # Read latest retargeting output (non-blocking, from SharedState)
        retarget = shared.get_retarget_output()

        if retarget.valid:
            tracking_count += 1

            # Convert IK output (KinDynLib) → hardware convention
            # q_des is 28 joints: [rleg(6), lleg(6), rarm(7), larm(7), head(2)]
            q_kin_full = np.zeros(28, dtype=np.float64)
            q_kin_full[:] = retarget.q_des
            q_hw_full = joint_mapping.reverse_q(q_kin_full)

            target_r = q_hw_full[12:19].copy()
            target_l = q_hw_full[19:26].copy()

            # ── Blend from IDLE to tracking (first valid frame) ──────
            if not blend_started:
                blend_started = True
                blend_t0 = now
                blend_start_r = last_cmd_r.copy()
                blend_start_l = last_cmd_l.copy()
                print("[Phase 2] First valid IK — blending to tracked pose …")

            if now - blend_t0 < blend_time:
                alpha = (now - blend_t0) / blend_time
                alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))  # smooth ease-in-out
                target_r = lerp(blend_start_r, target_r, alpha)
                target_l = lerp(blend_start_l, target_l, alpha)

            # ── Per-step velocity clamp (safety) ─────────────────────
            delta_r = np.clip(target_r - last_cmd_r, -max_delta, max_delta)
            delta_l = np.clip(target_l - last_cmd_l, -max_delta, max_delta)
            cmd_r = last_cmd_r + delta_r
            cmd_l = last_cmd_l + delta_l

        else:
            # No valid tracking — hold last commanded pose
            cmd_r = last_cmd_r
            cmd_l = last_cmd_l

        # Send arms
        _send_both_arms_ff(client, cmd_r, cmd_l, KP_SOFT, KD_SOFT)
        last_cmd_r = cmd_r.copy()
        last_cmd_l = cmd_l.copy()

        # ── Hand commands (direct lerp + velocity clamp) ──────────────
        hand_data = shared.get_hand_tracking_data()
        if hand_data.valid:
            # NOTE: ZED camera is mirrored (camera-left = user-right)
            # So tracking's "left_open_close" → command RIGHT hand
            # and tracking's "right_open_close" → command LEFT hand
            hand_target_r = lerp(RIGHT_FIST, RIGHT_OPEN, hand_data.left_open_close)
            hand_target_l = lerp(LEFT_FIST, LEFT_OPEN, hand_data.right_open_close)
        else:
            hand_target_r = last_hand_r
            hand_target_l = last_hand_l

        # Per-motor velocity clamp (prevents sudden jumps)
        hand_delta_r = np.clip(hand_target_r - last_hand_r, -max_hand_delta, max_hand_delta)
        hand_delta_l = np.clip(hand_target_l - last_hand_l, -max_hand_delta, max_hand_delta)
        hand_cmd_r = last_hand_r + hand_delta_r
        hand_cmd_l = last_hand_l + hand_delta_l

        _send_both_hands_ff(client, hand_cmd_r, hand_cmd_l, KP_HAND, KD_HAND)
        last_hand_r = hand_cmd_r.copy()
        last_hand_l = hand_cmd_l.copy()

        # ── Head: hold at zero ───────────────────────────────────────
        _send_head_ff(client, HEAD_ZERO, KP_HEAD, KD_HEAD)

        # Also publish robot feedback to SharedState for IK warm-start
        # (use background thread's latest — no blocking)
        hw_fb = fb_thread.get()
        if hw_fb.valid:
            robot_fb = RobotFeedback()
            robot_fb.timestamp = hw_fb.timestamp
            q_hw = np.zeros(28, dtype=np.float64)
            dq_hw = np.zeros(28, dtype=np.float64)
            q_hw[12:19] = hw_fb.right_arm_q
            q_hw[19:26] = hw_fb.left_arm_q
            dq_hw[12:19] = hw_fb.right_arm_dq
            dq_hw[19:26] = hw_fb.left_arm_dq
            robot_fb.q = q_hw
            robot_fb.dq = dq_hw
            robot_fb.base_pos = hw_fb.base_position.copy() if hasattr(hw_fb, 'base_position') else np.zeros(3)
            shared.set_robot_feedback(robot_fb)

        # Status print every 2 s
        if now - last_print >= 2.0:
            last_print = now
            stats = shared.get_timing_stats()
            fb_now = fb_thread.get()
            print(f"[Status] Tracking: {stats.get('tracking_hz', 0):.0f} Hz | "
                  f"Retarget: {stats.get('retarget_hz', 0):.0f} Hz | "
                  f"Cmd loop: {step/(now - tick + step*dt):.0f} Hz | "
                  f"IK valid: {retarget.valid} | "
                  f"frames: {tracking_count}")
            if fb_now.valid and retarget.valid:
                err_r = np.degrees(np.linalg.norm(fb_now.right_arm_q - cmd_r))
                err_l = np.degrees(np.linalg.norm(fb_now.left_arm_q  - cmd_l))
                print(f"         Tracking error: R={err_r:.1f}°  L={err_l:.1f}°")
                print(f"         R cmd: {np.degrees(cmd_r).round(1)}")
                print(f"         R fb:  {np.degrees(fb_now.right_arm_q).round(1)}")
            if hand_data.valid:
                print(f"         Hands: R oc={hand_data.right_open_close:.2f} | L oc={hand_data.left_open_close:.2f}")

        tick += dt
        _spin_wait(tick)

    # ── Phase 3: Ramp back to IDLE ───────────────────────────────────
    print(f"\n[Phase 3] Returning to IDLE over 3.0 s …")
    cur_r = last_cmd_r.copy()
    cur_l = last_cmd_l.copy()
    cur_hand_r = last_hand_r.copy()
    cur_hand_l = last_hand_l.copy()
    t0 = time.perf_counter()
    tick = t0
    _shutdown = False  # allow ramp to complete
    for _ in range(int(RAMP * rate_hz)):
        now = time.perf_counter()
        a = min((now - t0) / RAMP, 1.0)
        _send_both_arms_ff(client, lerp(cur_r, IDLE_R_HW, a),
                           lerp(cur_l, IDLE_L_HW, a), KP_SOFT, KD_SOFT)
        _send_both_hands_ff(client, lerp(cur_hand_r, RIGHT_FIST, a),
                            lerp(cur_hand_l, LEFT_FIST, a), KP_HAND, KD_HAND)
        _send_head_ff(client, HEAD_ZERO, KP_HEAD, KD_HEAD)
        tick += dt
        _spin_wait(tick)

    # Hold IDLE 0.5 s
    t0 = time.perf_counter(); tick = t0
    while time.perf_counter() - t0 < 0.5:
        _send_both_arms_ff(client, IDLE_R_HW, IDLE_L_HW, KP_SOFT, KD_SOFT)
        _send_both_hands_ff(client, RIGHT_FIST, LEFT_FIST, KP_HAND, KD_HAND)
        _send_head_ff(client, HEAD_ZERO, KP_HEAD, KD_HEAD)
        tick += dt; _spin_wait(tick)

    print("[Phase 3] Arms at IDLE, hands at FIST, head at zero ✓")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Body tracking + retargeting → Themis HW (WBC-safe, 1 kHz)")
    parser.add_argument("--robot-ip", default="192.168.0.11")
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--rate", type=float, default=1000.0,
                        help="Command loop rate in Hz (default: 1000)")
    parser.add_argument("--blend-time", type=float, default=3.0,
                        help="Seconds to blend from IDLE to tracking (default: 3)")
    parser.add_argument("--max-vel", type=float, default=2.0,
                        help="Max joint velocity in rad/s (default: 2.0)")
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true",
                        help="Use dummy tracking (no ZED camera)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No real robot (mock UDP client)")
    parser.add_argument("--hang-height", type=float, default=1.3,
                        help="Robot height for IK (default: 1.3 m)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable MuJoCo visualization window")
    args = parser.parse_args()

    # Gains
    global KP_SOFT, KD_SOFT
    KP_SOFT = np.full(7, args.kp)
    KD_SOFT = np.full(7, args.kd)

    # ── Banner ───────────────────────────────────────────────────────
    print("=" * 72)
    print("  THEMIS HW — Body Tracking + Retargeting (WBC-safe, 1 kHz)")
    print("=" * 72)
    print(f"  Robot:       {args.robot_ip}:{args.port}")
    print(f"  Cmd rate:    {args.rate:.0f} Hz")
    print(f"  Arm gains:   kp={args.kp:.1f}  kd={args.kd:.1f}")
    print(f"  Hand gains:  kp=5.0  kd=0.5")
    print(f"  Head gains:  kp=10.0  kd=1.0")
    print(f"  Blend:       {args.blend_time:.1f} s")
    print(f"  Max vel:     {args.max_vel:.1f} rad/s")
    print(f"  Camera:      {'dummy' if args.no_camera else 'ZED'}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Visualizer:  {'OFF' if args.no_viz else 'ON'}")
    print("=" * 72)

    # ── Config ───────────────────────────────────────────────────────
    config = PipelineConfig()
    config.sim.base_height = args.hang_height

    # ── Shared state ─────────────────────────────────────────────────
    shared = SharedState()

    # ── UDP client ───────────────────────────────────────────────────
    if args.dry_run:
        client = DryRunClient()
    else:
        client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port)
    client.connect()

    # ── Joint mapping ────────────────────────────────────────────────
    joint_mapping = JointMapping(config.joint_mapping)

    # ── Background feedback thread ───────────────────────────────────
    fb_thread = FeedbackThread(args.robot_ip if not args.dry_run else "127.0.0.1",
                               args.port, client)
    if not args.dry_run:
        fb_thread.start()
    else:
        # For dry-run, provide a mock feedback thread
        fb_thread._fb = client.get_state()

    # ── Seed initial feedback into SharedState ───────────────────────
    print("[Main] Reading initial robot state …")
    time.sleep(0.3)
    fb = fb_thread.get() if not args.dry_run else client.get_state()
    if fb.valid:
        robot_fb = RobotFeedback()
        robot_fb.timestamp = fb.timestamp
        q_hw = np.zeros(28, dtype=np.float64)
        q_hw[12:19] = fb.right_arm_q
        q_hw[19:26] = fb.left_arm_q
        robot_fb.q = q_hw
        shared.set_robot_feedback(robot_fb)
        print(f"[Main] Initial R arm: {np.degrees(fb.right_arm_q).round(1)}")
        print(f"[Main] Initial L arm: {np.degrees(fb.left_arm_q).round(1)}")
    else:
        print("[Main] WARNING: No robot state — using defaults")

    # ── Start tracking + retargeting nodes ───────────────────────────
    print("[Main] Initializing tracking + retargeting …\n")
    retargeter = RetargetingNode(config, shared)
    tracker = BodyTrackingNode(config, shared)

    # ── Visualizer (separate thread, ~30 Hz) ─────────────────────────
    viz = None
    if not args.no_viz:
        viz = HardwareVisualizer(shared, config)

    try:
        retargeter.start()
        tracker.start()

        # Launch viewer after nodes are up (needs a valid SharedState)
        if viz is not None:
            viz.start()

        print("[Main] All nodes started — entering 1 kHz command loop\n")

        run_pipeline(
            client, shared, config, joint_mapping, fb_thread,
            rate_hz=args.rate,
            blend_time=args.blend_time,
            max_delta_per_s=args.max_vel,
        )

    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[Main] Shutting down …")
        shared.request_shutdown()
        if viz is not None:
            viz.stop()
        tracker.stop()
        retargeter.stop()
        fb_thread.stop()
        client.disconnect()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
