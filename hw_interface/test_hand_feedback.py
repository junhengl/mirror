#!/usr/bin/env python3
"""
Test script: Read hand (DXL) motor feedback from THEMIS robot.

Continuously reads hand joint states at ~10 Hz and displays:
  • Current joint positions (radians + degrees)
  • Current joint velocities
  • Current joint torques
  • Running min / max seen for each joint

This lets you manually move the fingers and see the full range of
motion for each DXL motor before writing the control code.

Usage:
  # 1. Robot PC — start the server (updated with hand support):
  ssh themis@192.168.0.11
  cd /home/themis/THEMIS/THEMIS
  python3 ~/wbc_udp_server.py --port 9870

  # 2. Desktop — run this script:
  .venv/bin/python hw_interface/test_hand_feedback.py

  # Options:
  .venv/bin/python hw_interface/test_hand_feedback.py --robot-ip 192.168.0.11
  .venv/bin/python hw_interface/test_hand_feedback.py --rate 20   # faster updates
  .venv/bin/python hw_interface/test_hand_feedback.py --left-only  # only show left hand
  .venv/bin/python hw_interface/test_hand_feedback.py --right-only # only show right hand
  .venv/bin/python hw_interface/test_hand_feedback.py --csv        # CSV log mode

Hand motor layout (7 DXL motors per hand):
  [0]   finger1_prox    — finger 1 proximal flex
  [1]   finger1_dist    — finger 1 distal flex
  [2]   finger2_prox    — finger 2 proximal flex
  [3]   finger2_dist    — finger 2 distal flex
  [4]   finger3_prox    — finger 3 proximal flex
  [5]   finger3_dist    — finger 3 distal flex
  [6]   finger_split    — controls 3-finger splay

Press Ctrl-C to stop.  Press 'r' + Enter to reset min/max tracking.
"""

import os
import sys
import time
import signal
import argparse
import threading
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from hw_interface.themis_udp_client import ThemisUDPClient, ThemisHandFeedback

# ── Shutdown ─────────────────────────────────────────────────────────
_shutdown = False

def _sig(s, f):
    global _shutdown
    _shutdown = True

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

# ── Joint names ──────────────────────────────────────────────────────
HAND_JOINT_NAMES = [
    "finger1_prox ",
    "finger1_dist ",
    "finger2_prox ",
    "finger2_dist ",
    "finger3_prox ",
    "finger3_dist ",
    "finger_split ",
]


def clear_screen():
    """Clear terminal (ANSI escape)."""
    print("\033[2J\033[H", end="")


def print_hand_table(label, q, dq, torque, q_min, q_max):
    """Print a formatted table for one hand."""
    print(f"\n  {label}:")
    print(f"    {'Joint':<16s} {'Pos [rad]':>10s} {'Pos [deg]':>10s} "
          f"{'Vel [r/s]':>10s} {'Torque':>8s} "
          f"{'Min [rad]':>10s} {'Min [deg]':>10s} "
          f"{'Max [rad]':>10s} {'Max [deg]':>10s}")
    print(f"    {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*8} "
          f"{'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for j in range(7):
        print(f"    {HAND_JOINT_NAMES[j]:<16s} "
              f"{q[j]:>+10.4f} {np.degrees(q[j]):>+10.2f} "
              f"{dq[j]:>+10.4f} {torque[j]:>+8.3f} "
              f"{q_min[j]:>+10.4f} {np.degrees(q_min[j]):>+10.2f} "
              f"{q_max[j]:>+10.4f} {np.degrees(q_max[j]):>+10.2f}")


def run(args):
    global _shutdown

    client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port,
                             timeout=0.5)
    client.connect()

    # Min/max trackers
    r_min = np.full(7, +np.inf)
    r_max = np.full(7, -np.inf)
    l_min = np.full(7, +np.inf)
    l_max = np.full(7, -np.inf)

    # Input thread for 'r' to reset min/max
    reset_event = threading.Event()

    def _input_loop():
        while not _shutdown:
            try:
                line = input()
                if line.strip().lower() == 'r':
                    reset_event.set()
            except EOFError:
                break

    input_thread = threading.Thread(target=_input_loop, daemon=True)
    input_thread.start()

    dt = 1.0 / args.rate
    read_count = 0
    fail_count = 0
    t_start = time.time()

    # CSV header
    if args.csv:
        hdr = "time"
        for side in ["R", "L"]:
            for j in range(7):
                hdr += f",{side}_q{j},{side}_dq{j},{side}_tau{j}"
        print(hdr)

    print("=" * 80)
    print("  THEMIS Hand (DXL) Feedback Monitor")
    print("=" * 80)
    print(f"  Robot:  {args.robot_ip}:{args.port}")
    print(f"  Rate:   {args.rate:.0f} Hz")
    print(f"  Mode:   {'CSV' if args.csv else 'Terminal'}")
    print("=" * 80)
    print("  Press 'r' + Enter to reset min/max  |  Ctrl-C to quit\n")

    while not _shutdown:
        # Reset min/max if requested
        if reset_event.is_set():
            r_min = np.full(7, +np.inf)
            r_max = np.full(7, -np.inf)
            l_min = np.full(7, +np.inf)
            l_max = np.full(7, -np.inf)
            reset_event.clear()
            if not args.csv:
                print("\n  *** Min/Max RESET ***\n")

        # Read hand state
        fb = client.get_hand_state()

        if not fb.valid:
            fail_count += 1
            if fail_count % 10 == 1:
                print(f"  [WARNING] No hand state response (fail #{fail_count})")
            time.sleep(dt)
            continue

        read_count += 1

        # Update min/max
        if not args.left_only:
            r_min = np.minimum(r_min, fb.right_hand_q)
            r_max = np.maximum(r_max, fb.right_hand_q)
        if not args.right_only:
            l_min = np.minimum(l_min, fb.left_hand_q)
            l_max = np.maximum(l_max, fb.left_hand_q)

        # CSV mode
        if args.csv:
            t = time.time() - t_start
            row = f"{t:.3f}"
            for q, dq, tau in [
                (fb.right_hand_q, fb.right_hand_dq, fb.right_hand_torque),
                (fb.left_hand_q,  fb.left_hand_dq,  fb.left_hand_torque),
            ]:
                for j in range(7):
                    row += f",{q[j]:.6f},{dq[j]:.6f},{tau[j]:.6f}"
            print(row)
            time.sleep(dt)
            continue

        # Terminal display mode
        clear_screen()
        elapsed = time.time() - t_start
        hz = read_count / elapsed if elapsed > 0 else 0

        print("=" * 100)
        print(f"  THEMIS Hand (DXL) Feedback   |   {hz:.1f} Hz   |   "
              f"reads: {read_count}   fails: {fail_count}   |   "
              f"Press 'r'+Enter to reset min/max")
        print("=" * 100)

        if not args.left_only:
            print_hand_table("RIGHT HAND",
                             fb.right_hand_q, fb.right_hand_dq,
                             fb.right_hand_torque, r_min, r_max)

        if not args.right_only:
            print_hand_table("LEFT HAND",
                             fb.left_hand_q, fb.left_hand_dq,
                             fb.left_hand_torque, l_min, l_max)

        # Also show arm state for context
        if args.show_arms:
            arm_fb = client.get_state()
            if arm_fb.valid:
                print(f"\n  ARM CONTEXT:")
                print(f"    R arm q: {np.degrees(arm_fb.right_arm_q).round(1)}")
                print(f"    L arm q: {np.degrees(arm_fb.left_arm_q).round(1)}")

        print(f"\n  Range (deg) summary:")
        if not args.left_only:
            r_range = np.degrees(r_max - r_min)
            print(f"    R hand range: {r_range.round(1)}")
        if not args.right_only:
            l_range = np.degrees(l_max - l_min)
            print(f"    L hand range: {l_range.round(1)}")

        time.sleep(dt)

    print("\n\n  Final min/max summary:")
    if not args.left_only:
        print(f"\n  RIGHT HAND:")
        for j in range(7):
            print(f"    {HAND_JOINT_NAMES[j]}:  "
                  f"min={r_min[j]:+.4f} ({np.degrees(r_min[j]):+.1f}°)  "
                  f"max={r_max[j]:+.4f} ({np.degrees(r_max[j]):+.1f}°)  "
                  f"range={np.degrees(r_max[j]-r_min[j]):.1f}°")
    if not args.right_only:
        print(f"\n  LEFT HAND:")
        for j in range(7):
            print(f"    {HAND_JOINT_NAMES[j]}:  "
                  f"min={l_min[j]:+.4f} ({np.degrees(l_min[j]):+.1f}°)  "
                  f"max={l_max[j]:+.4f} ({np.degrees(l_max[j]):+.1f}°)  "
                  f"range={np.degrees(l_max[j]-l_min[j]):.1f}°")

    print()
    client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Read THEMIS hand (DXL) motor feedback and track joint ranges")
    parser.add_argument("--robot-ip", default="192.168.0.11",
                        help="Robot IP (default: 192.168.0.11)")
    parser.add_argument("--port", type=int, default=9870,
                        help="UDP port (default: 9870)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Read rate in Hz (default: 10)")
    parser.add_argument("--left-only", action="store_true",
                        help="Only show left hand")
    parser.add_argument("--right-only", action="store_true",
                        help="Only show right hand")
    parser.add_argument("--show-arms", action="store_true",
                        help="Also show arm joint positions for context")
    parser.add_argument("--csv", action="store_true",
                        help="Output CSV format (for logging/plotting)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
