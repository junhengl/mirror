#!/usr/bin/env python3
"""
THEMIS Arm Test via ROS2  (runs on the DESKTOP)

This test script uses ROS2 topics instead of raw UDP+shared-memory.
It communicates with the robot through the official wbc_api via the
robot-side ROS2 node (themis_ros2_robot_node.py).

Architecture:
  ┌──────────── Desktop PC ─────────────────────┐
  │                                              │
  │  This script (test_arm_ros2.py)              │
  │    publishes:  /themis/arm_cmd/{right,left}  │
  │    subscribes: /themis/joint_state/*          │
  │                /themis/imu                    │
  │                                              │
  └──── ROS2 DDS (same ROS_DOMAIN_ID) ──────────┘
                       │
  ┌──────────── Robot PC ───────────────────────┐
  │                                              │
  │  themis_ros2_robot_node.py                   │
  │    uses: wbc_api.get_joint_states(chain)     │
  │          wbc_api.set_joint_states(chain,…)   │
  │    (safe, goes through WBC — no collapse)    │
  │                                              │
  └──────────────────────────────────────────────┘

Why this doesn't break the robot:
  The old shm_udp_server.py directly accessed POSIX shared memory
  (MM.RIGHT_ARM_JOINT_COMMAND.set(...)) which competed with the robot's
  WBC loop for ownership of the shared memory segments, causing the
  robot to lose its balance controller and collapse.

  This new approach uses wbc_api.set_joint_states() which is the
  OFFICIAL API that cooperates with the WBC pipeline. Commands flow
  through the whole-body controller → actuators, so the balance
  controller stays alive.

Demo motions:
  1. Read current state (read-only, no commands)
  2. Gentle sinusoidal sweep on shoulder_pitch + elbow_pitch
  3. Return to initial pose

Usage:
  # On the robot PC (first):
  cd /home/themis/THEMIS/THEMIS
  python3 /path/to/hw_interface/ros2/themis_ros2_robot_node.py

  # On the desktop (second):
  export ROS_DOMAIN_ID=0
  python3 hw_interface/ros2/test_arm_ros2.py

  # Read-only mode (just prints state, no commands):
  python3 hw_interface/ros2/test_arm_ros2.py --read-only

  # Dry-run (no robot needed):
  python3 hw_interface/ros2/test_arm_ros2.py --dry-run
"""

import argparse
import time
import sys
import os
import signal
import numpy as np
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)


# ── Arm joint ordering (same as documentation) ──────────────────────
ARM_JOINT_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw",
]

# ── Nominal poses (from THEMIS manipulation_macros.py) ───────────────
IDLE_R = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

# ── Default PD gains (soft for safety) ──────────────────────────────
KP_SOFT = np.full(7, 10.0)
KD_SOFT = np.full(7,  1.0)


def lerp_pose(q0, q1, alpha):
    """Linear interpolation between two joint poses."""
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * q0 + alpha * q1


# ── Dry-run mock ─────────────────────────────────────────────────────
class DryRunClient:
    """Fake client for testing without a robot."""

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    DEFAULT_KP = KP_SOFT
    DEFAULT_KD = KD_SOFT

    def connect(self):
        print("[DryRun] Connected (no real robot)")

    def disconnect(self):
        print("[DryRun] Disconnected")

    def shutdown(self):
        self.disconnect()

    def get_state(self):
        from hw_interface.ros2.themis_ros2_desktop_client import ThemisStateFeedback
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = IDLE_R.copy()
        fb.left_arm_q  = IDLE_L.copy()
        fb.timestamp = time.time()
        return fb

    def wait_for_state(self, timeout=5.0):
        return self.get_state()

    def send_arm_command(self, side, q, dq=None, u=None, kp=None, kd=None):
        return True

    def print_state(self, fb=None):
        if fb is None:
            fb = self.get_state()
        print(f"[DryRun] R_arm_q = {np.degrees(fb.right_arm_q).round(1)}")
        print(f"[DryRun] L_arm_q = {np.degrees(fb.left_arm_q).round(1)}")


# ── Graceful shutdown ────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[Test] Shutdown requested …")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_read_only(client, duration: float = 10.0, rate_hz: float = 2.0):
    """
    Read-only mode: just prints joint state, sends NO commands.

    Use this to verify that ROS2 communication works without touching
    the robot's actuators.
    """
    global _shutdown_requested
    dt = 1.0 / rate_hz

    print("\n" + "=" * 72)
    print("  THEMIS ARM READ-ONLY TEST (ROS2)")
    print("  No commands will be sent — read only")
    print("=" * 72)

    t0 = time.time()
    while (time.time() - t0) < duration and not _shutdown_requested:
        fb = client.get_state()
        if fb.valid:
            client.print_state(fb)
        else:
            print(f"  [{time.time()-t0:.1f}s] Waiting for valid state …")
        time.sleep(dt)

    print("\n[ReadOnly] Done.")


def run_test(client, rate_hz: float = 100.0):
    """
    Main test sequence — sinusoidal arm sweep.

    Uses wbc_api.set_joint_states() on the robot side, which is the
    official API that cooperates with WBC. The robot should NOT collapse.
    """
    global _shutdown_requested
    dt = 1.0 / rate_hz

    print("\n" + "=" * 72)
    print("  THEMIS ARM JOINT-SPACE TEST (ROS2 + Official wbc_api)")
    print("=" * 72)
    print(f"  Command path:  ROS2 → wbc_api.set_joint_states()")
    print(f"  Loop rate:     {rate_hz:.0f} Hz")
    print(f"  PD gains:      kp={KP_SOFT[0]:.1f}, kd={KD_SOFT[0]:.1f}")
    print("=" * 72)

    # ── Phase 0: Read current state ──────────────────────────────────
    print("\n[Phase 0] Reading initial joint state …")
    fb = client.wait_for_state(timeout=5.0)
    if not fb.valid:
        print("[WARNING] No valid state — using IDLE as start")
        start_q_r = IDLE_R.copy()
        start_q_l = IDLE_L.copy()
    else:
        start_q_r = fb.right_arm_q.copy()
        start_q_l = fb.left_arm_q.copy()
        client.print_state(fb)

    # ── Phase 1: Ramp to IDLE ────────────────────────────────────────
    RAMP_TIME = 3.0
    print(f"\n[Phase 1] Ramping to IDLE pose over {RAMP_TIME:.1f} s …")
    t0 = time.time()
    while (time.time() - t0) < RAMP_TIME and not _shutdown_requested:
        alpha = (time.time() - t0) / RAMP_TIME
        q_r = lerp_pose(start_q_r, IDLE_R, alpha)
        q_l = lerp_pose(start_q_l, IDLE_L, alpha)
        client.send_arm_command('right', q=q_r, kp=KP_SOFT, kd=KD_SOFT)
        client.send_arm_command('left',  q=q_l, kp=KP_SOFT, kd=KD_SOFT)
        time.sleep(dt)

    if _shutdown_requested:
        return

    # Hold IDLE briefly
    for _ in range(50):
        client.send_arm_command('right', q=IDLE_R, kp=KP_SOFT, kd=KD_SOFT)
        client.send_arm_command('left',  q=IDLE_L, kp=KP_SOFT, kd=KD_SOFT)
        time.sleep(dt)

    # Print state at IDLE
    fb = client.get_state()
    if fb.valid:
        print("[Phase 1] Reached IDLE.  Current state:")
        client.print_state(fb)

    # ── Phase 2: Sinusoidal sweep ────────────────────────────────────
    SWEEP_TIME = 10.0
    AMPLITUDE  = 0.15    # rad (~8.6°) — gentle
    FREQ       = 0.3     # Hz — slow and smooth

    print(f"\n[Phase 2] Sinusoidal sweep:")
    print(f"  amp = {np.degrees(AMPLITUDE):.1f}°, freq = {FREQ:.1f} Hz, duration = {SWEEP_TIME:.0f} s")
    print(f"  Modulating: shoulder_pitch (idx 0) + elbow_pitch (idx 3)")

    t0 = time.time()
    step = 0
    while (time.time() - t0) < SWEEP_TIME and not _shutdown_requested:
        t = time.time() - t0
        wave = AMPLITUDE * np.sin(2.0 * np.pi * FREQ * t)

        q_r = IDLE_R.copy()
        q_l = IDLE_L.copy()
        q_r[0] += wave
        q_r[3] += wave * 0.5
        q_l[0] += wave
        q_l[3] -= wave * 0.5

        client.send_arm_command('right', q=q_r, kp=KP_SOFT, kd=KD_SOFT)
        client.send_arm_command('left',  q=q_l, kp=KP_SOFT, kd=KD_SOFT)

        step += 1
        if step % int(rate_hz) == 0:
            fb = client.get_state()
            if fb.valid:
                err_r = np.linalg.norm(fb.right_arm_q - q_r)
                err_l = np.linalg.norm(fb.left_arm_q  - q_l)
                print(f"  t={t:5.1f}s  |  "
                      f"cmd_R[0]={np.degrees(q_r[0]):+6.1f}° "
                      f"fb_R[0]={np.degrees(fb.right_arm_q[0]):+6.1f}° "
                      f"err_R={np.degrees(err_r):5.2f}° | "
                      f"τ_R={fb.right_arm_torque.round(2)}")

        time.sleep(dt)

    if _shutdown_requested:
        return

    # ── Phase 3: Return to IDLE ──────────────────────────────────────
    print(f"\n[Phase 3] Returning to IDLE over {RAMP_TIME:.1f} s …")
    fb = client.get_state()
    cur_r = fb.right_arm_q.copy() if fb.valid else IDLE_R.copy()
    cur_l = fb.left_arm_q.copy()  if fb.valid else IDLE_L.copy()

    t0 = time.time()
    while (time.time() - t0) < RAMP_TIME and not _shutdown_requested:
        alpha = (time.time() - t0) / RAMP_TIME
        q_r = lerp_pose(cur_r, IDLE_R, alpha)
        q_l = lerp_pose(cur_l, IDLE_L, alpha)
        client.send_arm_command('right', q=q_r, kp=KP_SOFT, kd=KD_SOFT)
        client.send_arm_command('left',  q=q_l, kp=KP_SOFT, kd=KD_SOFT)
        time.sleep(dt)

    # Hold IDLE
    for _ in range(50):
        client.send_arm_command('right', q=IDLE_R, kp=KP_SOFT, kd=KD_SOFT)
        client.send_arm_command('left',  q=IDLE_L, kp=KP_SOFT, kd=KD_SOFT)
        time.sleep(dt)

    # Final state
    fb = client.get_state()
    if fb.valid:
        print("\n[Done] Final joint state:")
        client.print_state(fb)

    print("\n[Test] Complete ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Themis arm test via ROS2 (using official wbc_api)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Command loop rate in Hz (default: 100)")
    parser.add_argument("--read-only", action="store_true",
                        help="Only read state — send NO commands")
    parser.add_argument("--read-duration", type=float, default=10.0,
                        help="Duration for read-only mode (default: 10s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without a real robot (mock client)")
    args = parser.parse_args()

    if args.dry_run:
        client = DryRunClient()
        client.connect()
    else:
        from hw_interface.ros2.themis_ros2_desktop_client import ThemisROS2Client
        client = ThemisROS2Client()
        client.connect()
        # Give the subscriber time to discover the robot node
        print("[Main] Waiting for ROS2 discovery …")
        time.sleep(1.0)

    try:
        if args.read_only:
            run_read_only(client, duration=args.read_duration)
        else:
            run_test(client, rate_hz=args.rate)
    finally:
        if args.dry_run:
            client.disconnect()
        else:
            client.shutdown()


if __name__ == "__main__":
    main()
