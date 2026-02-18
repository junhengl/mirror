#!/usr/bin/env python3
"""
Simple Arm Joint-Space Test  (runs on the DESKTOP)

Sends arm joint position commands to the Themis robot and prints joint
feedback in a loop.  This is a bare-bones validation of the full pipeline:

    [Desktop]  ──UDP──▶  [Robot bridge]  ──SHM──▶  [WBC / Actuators]
                                              ▼
    [Desktop]  ◀──UDP──  [Robot bridge]  ◀──SHM──  [Joint State]

The script walks through several demo motions:
    1.  Move to IDLE pose
    2.  Simple sinusoidal sweep on each arm joint
    3.  Return to IDLE

Usage:
    sudo ip addr add 192.168.0.100/24 dev enp130s0
    python3 hw_interface/test_arm_simple.py --robot-ip 192.168.1.100 --port 9870

    # Dry-run (no robot, just prints what would be sent):
    
    python3 hw_interface/test_arm_simple.py --dry-run
"""

import argparse
import time
import sys
import os
import signal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hw_interface.themis_udp_client import ThemisUDPClient, ThemisStateFeedback


# ── Robot arm joint ordering ─────────────────────────────────────────
# Index:  0               1               2              3
# Name:   shoulder_pitch  shoulder_roll   shoulder_yaw   elbow_pitch
# Index:  4               5               6
# Name:   elbow_yaw       wrist_pitch     wrist_yaw

ARM_JOINT_NAMES = ThemisUDPClient.ARM_JOINT_NAMES

# ── Nominal poses (from manipulation_macros.py) ──────────────────────
IDLE_R  = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L  = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

# ── PD gains ─────────────────────────────────────────────────────────
KP_DEFAULT = np.full(7, 10.0)
KD_DEFAULT = np.full(7,   1.0)

# Softer gains for the test (lower stiffness = safer)
KP_SOFT = np.full(7, 10.0)
KD_SOFT = np.full(7,  1.0)


def lerp_pose(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation between two joint poses, clamped to [0, 1]."""
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * q0 + alpha * q1


class DryRunClient:
    """Fake client that just prints commands for offline testing."""

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    DEFAULT_KP = KP_DEFAULT
    DEFAULT_KD = KD_DEFAULT

    def connect(self):
        print("[DryRun] Connected (no real robot)")

    def disconnect(self):
        print("[DryRun] Disconnected")

    def get_state(self) -> ThemisStateFeedback:
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = IDLE_R.copy()
        fb.left_arm_q  = IDLE_L.copy()
        fb.timestamp   = time.time()
        return fb

    def send_arm_command(self, side, q, dq=None, u=None, kp=None, kd=None):
        return True

    def send_manip_reference(self, right_arm_pose, left_arm_pose, **kw):
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


def run_test(client, use_manip_ref: bool = True, rate_hz: float = 100.0):
    """
    Main test sequence.

    Parameters
    ----------
    client : ThemisUDPClient or DryRunClient
    use_manip_ref : if True, commands go via MANIPULATION_REFERENCE (WBC path);
                    if False, commands go directly to JOINT_COMMAND.
    rate_hz : command loop frequency
    """
    global _shutdown_requested
    dt = 1.0 / rate_hz

    print("\n" + "=" * 70)
    print("  THEMIS ARM JOINT-SPACE TEST")
    print("=" * 70)
    mode_str = "MANIPULATION_REFERENCE (WBC)" if use_manip_ref else "JOINT_COMMAND (direct)"
    print(f"  Command path:  {mode_str}")
    print(f"  Loop rate:     {rate_hz:.0f} Hz")
    print("=" * 70)

    def send(q_r, q_l, kp=KP_SOFT, kd=KD_SOFT):
        """Send joint command via the chosen path."""
        if use_manip_ref:
            client.send_manip_reference(
                right_arm_pose=q_r, left_arm_pose=q_l,
                right_mode=100.0, right_phase=1.0,
                left_mode=100.0,  left_phase=1.0,
            )
        else:
            client.send_arm_command('right', q=q_r, kp=kp, kd=kd)
            client.send_arm_command('left',  q=q_l, kp=kp, kd=kd)

    # ── Phase 1: Read current state ──────────────────────────────────
    print("\n[Phase 0] Reading initial joint state …")
    fb = client.get_state()
    if not fb.valid:
        print("[WARNING] Could not read robot state — using IDLE pose as start")
        start_q_r = IDLE_R.copy()
        start_q_l = IDLE_L.copy()
    else:
        start_q_r = fb.right_arm_q.copy()
        start_q_l = fb.left_arm_q.copy()
        client.print_state(fb)

    # ── Phase 1: Move to IDLE ────────────────────────────────────────
    RAMP_TIME = 3.0  # seconds
    print(f"\n[Phase 1] Ramping to IDLE pose over {RAMP_TIME:.1f} s …")
    t0 = time.time()
    while (time.time() - t0) < RAMP_TIME and not _shutdown_requested:
        alpha = (time.time() - t0) / RAMP_TIME
        q_r = lerp_pose(start_q_r, IDLE_R, alpha)
        q_l = lerp_pose(start_q_l, IDLE_L, alpha)
        send(q_r, q_l)
        time.sleep(dt)

    if _shutdown_requested:
        return

    send(IDLE_R, IDLE_L)
    time.sleep(0.5)

    # Print state at IDLE
    fb = client.get_state()
    if fb.valid:
        print("[Phase 1] Reached IDLE.  Current joint state:")
        client.print_state(fb)

    # ── Phase 2: Sinusoidal sweep ────────────────────────────────────
    SWEEP_TIME = 10.0   # seconds per sweep
    AMPLITUDE  = 0.20   # radians (≈ 11.5°) — small & safe
    FREQ       = 0.5    # Hz

    print(f"\n[Phase 2] Sinusoidal sweep: amp={np.degrees(AMPLITUDE):.1f}°, freq={FREQ:.1f} Hz, duration={SWEEP_TIME:.0f} s")
    print("          Modulating shoulder_pitch (idx 0) and elbow_pitch (idx 3)")

    t0 = time.time()
    step = 0
    while (time.time() - t0) < SWEEP_TIME and not _shutdown_requested:
        t = time.time() - t0
        wave = AMPLITUDE * np.sin(2.0 * np.pi * FREQ * t)

        q_r = IDLE_R.copy()
        q_l = IDLE_L.copy()

        # Modulate shoulder pitch and elbow pitch symmetrically
        q_r[0] += wave        # shoulder_pitch_R
        q_r[3] += wave * 0.5  # elbow_pitch_R  (half amplitude)
        q_l[0] += wave        # shoulder_pitch_L
        q_l[3] -= wave * 0.5  # elbow_pitch_L  (mirrored)

        send(q_r, q_l)

        # Print feedback periodically
        step += 1
        if step % int(rate_hz) == 0:
            fb = client.get_state()
            if fb.valid:
                err_r = np.linalg.norm(fb.right_arm_q - q_r)
                err_l = np.linalg.norm(fb.left_arm_q  - q_l)
                print(f"  t={t:5.1f}s  |  cmd_R[0]={np.degrees(q_r[0]):+6.1f}°  "
                      f"fb_R[0]={np.degrees(fb.right_arm_q[0]):+6.1f}°  "
                      f"err_R={np.degrees(err_r):5.2f}°  |  "
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
        send(q_r, q_l)
        time.sleep(dt)

    send(IDLE_R, IDLE_L)
    time.sleep(0.5)

    # Final state
    fb = client.get_state()
    if fb.valid:
        print("\n[Done] Final joint state:")
        client.print_state(fb)

    print("\n[Test] Complete.")


def main():
    parser = argparse.ArgumentParser(description="Themis arm joint-space test")
    parser.add_argument("--robot-ip", default="192.168.0.11",
                        help="Robot PC IP address (default: 192.168.0.11)")
    parser.add_argument("--port", type=int, default=9870,
                        help="UDP bridge port (default: 9870)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Command loop rate in Hz (default: 100)")
    parser.add_argument("--direct", action="store_true",
                        help="Use JOINT_COMMAND path instead of MANIPULATION_REFERENCE (WBC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without a real robot (prints commands)")
    args = parser.parse_args()

    if args.dry_run:
        client = DryRunClient()
    else:
        client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port)

    client.connect()

    try:
        run_test(client, use_manip_ref=not args.direct, rate_hz=args.rate)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
