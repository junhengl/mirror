#!/usr/bin/env python3
"""
Test script: Left hand open / close on THEMIS robot.

Reads the current hand position (should be near fist — the default),
then smoothly cycles between FIST (closed) and OPEN poses so you can
verify motor directions, speeds, and gains before integrating with the
body-tracking pipeline.

Uses the same safe patterns as test_arm_wbc.py:
  • 1 kHz fire-and-forget UDP sends
  • Spin-wait timing for <1 ms jitter
  • Smooth ramp to start pose on startup
  • Ramp back to fist on Ctrl-C

Hand motor layout (left hand, 7 DXL motors):
  [0]  finger1_prox   — finger 1 proximal flex
  [1]  finger1_dist   — finger 1 distal flex
  [2]  finger2_prox   — finger 2 proximal flex
  [3]  finger2_dist   — finger 2 distal flex
  [4]  finger3_prox   — finger 3 proximal flex
  [5]  finger3_dist   — finger 3 distal flex
  [6]  finger_split   — 3-finger splay

Measured LEFT HAND ranges:
  finger1_prox:  -0.015 → +2.146  (  -1° → +123°)
  finger1_dist:  +0.296 → +1.959  ( +17° → +112°)
  finger2_prox:  -0.502 → +2.072  ( -29° → +119°)
  finger2_dist:  -2.059 → +0.276  (-118° →  +16°)
  finger3_prox:  +0.129 → +1.954  (  +7° → +112°)
  finger3_dist:  -0.727 → +0.288  ( -42° →  +17°)
  finger_split:  +0.874 → +1.999  ( +50° → +115°)

Usage:
  # 1. Robot PC — start the server:
  ssh themis@192.168.0.11
  cd /home/themis/THEMIS/THEMIS
  python3 ~/wbc_udp_server.py --port 9870

  # 2. Desktop — run this test:
  .venv/bin/python hw_interface/test_hand_open_close.py

  # Options:
  --period 4.0      # seconds per open-close cycle (default 4)
  --kp 5.0          # proportional gain (default 5)
  --kd 0.5          # derivative gain (default 0.5)
  --amplitude 0.8   # fraction of full range to use (default 0.8)
  --dry-run         # no robot connection
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

from hw_interface.themis_udp_client import (
    ThemisUDPClient, ThemisHandFeedback,
    MSG_HAND_JOINT_CMD, MSG_HAND_STATE_REQUEST, MSG_HAND_STATE_RESPONSE,
    MSG_STATE_REQUEST, SIDE_LEFT_HAND, SIDE_RIGHT_HAND,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants — Recorded hand poses from finger_test.txt
# ═══════════════════════════════════════════════════════════════════════

# Motor order: [f1_prox, f1_dist, f2_prox, f2_dist, f3_prox, f3_dist, split]
#                 0         1        2        3        4        5       6

# ── LEFT HAND (recorded) ─────────────────────────────────────────────
LEFT_FIST = np.array([
    +1.5156,  # finger1_prox
    +0.8099,  # finger1_dist
    +1.5125,  # finger2_prox
    +0.8391,  # finger2_dist
    +0.0690,  # finger3_prox
    +0.6366,  # finger3_dist
    -2.6047,  # finger_split
], dtype=np.float64)

LEFT_OPEN = np.array([
    +1.0937,  # finger1_prox
    +0.4065,  # finger1_dist
    +1.1075,  # finger2_prox
    +0.4817,  # finger2_dist
    -0.3421,  # finger3_prox
    +0.4541,  # finger3_dist
    -2.6108,  # finger_split
], dtype=np.float64)

# ── RIGHT HAND (recorded) ────────────────────────────────────────────
RIGHT_FIST = np.array([
    +1.5723,  # finger1_prox
    +0.9327,  # finger1_dist
    +1.5493,  # finger2_prox
    +1.0293,  # finger2_dist
    -0.0782,  # finger3_prox
    +0.9342,  # finger3_dist
    -2.5203,  # finger_split
], dtype=np.float64)

RIGHT_OPEN = np.array([
    +1.1735,  # finger1_prox
    +0.6872,  # finger1_dist
    +1.1827,  # finger2_prox
    +0.0353,  # finger2_dist
    -0.6703,  # finger3_prox
    +0.6489,  # finger3_dist
    -2.4743,  # finger_split
], dtype=np.float64)

JOINT_NAMES = [
    "f1_prox", "f1_dist", "f2_prox", "f2_dist",
    "f3_prox", "f3_dist", "split",
]

# ═══════════════════════════════════════════════════════════════════════
# Shutdown
# ═══════════════════════════════════════════════════════════════════════

_shutdown = False

def _sig(s, f):
    global _shutdown
    _shutdown = True
    print("\n[Main] Shutdown requested …")

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _send_hand_ff(client, side_byte, q, kp, kd):
    """Fire-and-forget hand command — never blocks."""
    q_  = np.asarray(q, dtype=np.float64).ravel()
    dq  = np.zeros(7, dtype=np.float64)
    u   = np.zeros(7, dtype=np.float64)
    kp_ = np.asarray(kp, dtype=np.float64).ravel()
    kd_ = np.asarray(kd, dtype=np.float64).ravel()
    pkt = (struct.pack('B', MSG_HAND_JOINT_CMD)
           + struct.pack('B', side_byte)
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
# Background feedback thread (for hand state)
# ═══════════════════════════════════════════════════════════════════════

class HandFeedbackThread:
    """Reads hand state at ~20 Hz on its own UDP socket."""

    def __init__(self, robot_ip, port, client):
        self.robot_ip = robot_ip
        self.port = port
        self.client = client
        self._lock = threading.Lock()
        self._fb = ThemisHandFeedback()
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
                    sock.sendto(struct.pack('B', MSG_HAND_STATE_REQUEST),
                                (self.robot_ip, self.port))
                    data, _ = sock.recvfrom(4096)
                    if len(data) > 1 and data[0] == MSG_HAND_STATE_RESPONSE:
                        fb = self.client._parse_hand_state_response(data[1:])
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
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(args):
    global _shutdown

    kp = np.full(7, args.kp, dtype=np.float64)
    kd = np.full(7, args.kd, dtype=np.float64)
    rate_hz = 1000.0
    dt = 1.0 / rate_hz

    # Select poses and side byte based on --side
    if args.side == 'left':
        fist_target = LEFT_FIST.copy()
        open_target = LEFT_OPEN.copy()
        side_byte = SIDE_LEFT_HAND
        side_label = "LEFT"
    else:
        fist_target = RIGHT_FIST.copy()
        open_target = RIGHT_OPEN.copy()
        side_byte = SIDE_RIGHT_HAND
        side_label = "RIGHT"

    print("=" * 70)
    print(f"  THEMIS — {side_label} Hand Open/Close Test")
    print("=" * 70)
    print(f"  Robot:     {args.robot_ip}:{args.port}")
    print(f"  Side:      {side_label}")
    print(f"  Gains:     kp={args.kp:.1f}  kd={args.kd:.1f}")
    print(f"  Period:    {args.period:.1f} s")
    print(f"  Dry-run:   {args.dry_run}")
    print(f"  Rate:      {rate_hz:.0f} Hz")
    print("=" * 70)
    print(f"\n  FIST target (deg): {np.degrees(fist_target).round(1)}")
    print(f"  OPEN target (deg): {np.degrees(open_target).round(1)}")
    print()

    # ── Connect ──────────────────────────────────────────────────────
    client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port)
    client.connect()

    # Start feedback thread
    fb_thread = HandFeedbackThread(args.robot_ip, args.port, client)
    if not args.dry_run:
        fb_thread.start()
        time.sleep(0.3)  # let it get first reading

    # ── Read initial hand position ───────────────────────────────────
    if args.dry_run:
        start_q = fist_target.copy()
        print(f"  [DryRun] Starting from FIST pose")
    else:
        fb = client.get_hand_state()
        if fb.valid:
            start_q = (fb.left_hand_q.copy() if args.side == 'left'
                       else fb.right_hand_q.copy())
            print(f"  Initial {side_label} hand: {np.degrees(start_q).round(1)}")
        else:
            print("  WARNING: no hand feedback — starting from FIST pose")
            start_q = fist_target.copy()

    # ── Phase 1: Ramp to FIST (safe start) ───────────────────────────
    RAMP = 2.0
    print(f"\n[Phase 1] Ramping to FIST pose over {RAMP:.1f} s …")
    t0 = time.perf_counter()
    tick = t0
    while not _shutdown:
        now = time.perf_counter()
        if now - t0 >= RAMP:
            break
        a = (now - t0) / RAMP
        cmd = lerp(start_q, fist_target, a)
        if not args.dry_run:
            _send_hand_ff(client, side_byte, cmd, kp, kd)
        tick += dt
        _spin_wait(tick)

    # Hold FIST 0.5 s
    t0 = time.perf_counter(); tick = t0
    while time.perf_counter() - t0 < 0.5 and not _shutdown:
        if not args.dry_run:
            _send_hand_ff(client, side_byte, fist_target, kp, kd)
        tick += dt; _spin_wait(tick)

    if _shutdown:
        print("[Phase 1] Interrupted during ramp")
        client.disconnect()
        return

    print("[Phase 1] At FIST ✓")

    # ── Phase 2: Cycle open ↔ close ──────────────────────────────────
    print(f"\n[Phase 2] Cycling open ↔ close (period={args.period:.1f} s)")
    print("         Ctrl-C to stop\n")

    t0 = time.perf_counter()
    tick = t0
    last_print = t0
    step = 0
    last_cmd = fist_target.copy()

    while not _shutdown:
        now = time.perf_counter()
        step += 1

        # Smooth sinusoidal interpolation: 0 = fist, 1 = open
        phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * (now - t0) / args.period))
        cmd = lerp(fist_target, open_target, phase)
        last_cmd = cmd.copy()

        if not args.dry_run:
            _send_hand_ff(client, side_byte, cmd, kp, kd)

        # Status every 1 s
        if now - last_print >= 1.0:
            last_print = now
            fb = fb_thread.get()
            state = "OPENING" if phase > 0.5 else "CLOSING"
            print(f"  [{state:>7s} {phase*100:5.1f}%]  "
                  f"cmd: {np.degrees(cmd).round(1)}")
            if fb.valid:
                fb_q = (fb.left_hand_q if args.side == 'left'
                        else fb.right_hand_q)
                err = np.degrees(np.abs(fb_q - cmd))
                print(f"  {'':>16s}  fb:  {np.degrees(fb_q).round(1)}")
                print(f"  {'':>16s}  err: {err.round(1)}")

        tick += dt
        _spin_wait(tick)

    # ── Phase 3: Ramp back to FIST ───────────────────────────────────
    print(f"\n[Phase 3] Returning to FIST over {RAMP:.1f} s …")
    cur = last_cmd.copy()
    t0 = time.perf_counter()
    tick = t0
    _shutdown = False  # allow ramp to complete
    for _ in range(int(RAMP * rate_hz)):
        now = time.perf_counter()
        a = min((now - t0) / RAMP, 1.0)
        cmd = lerp(cur, fist_target, a)
        if not args.dry_run:
            _send_hand_ff(client, side_byte, cmd, kp, kd)
        tick += dt
        _spin_wait(tick)

    # Hold FIST 0.5 s
    t0 = time.perf_counter(); tick = t0
    while time.perf_counter() - t0 < 0.5:
        if not args.dry_run:
            _send_hand_ff(client, side_byte, fist_target, kp, kd)
        tick += dt; _spin_wait(tick)

    print("[Phase 3] At FIST ✓")

    fb_thread.stop()
    client.disconnect()
    print("[Main] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS hand open/close test (DXL motors)")
    parser.add_argument("--robot-ip", default="192.168.0.11")
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--side", choices=["left", "right"], default="left",
                        help="Which hand to test (default: left)")
    parser.add_argument("--period", type=float, default=4.0,
                        help="Seconds per open-close cycle (default 4)")
    parser.add_argument("--kp", type=float, default=5.0,
                        help="Proportional gain (default 5)")
    parser.add_argument("--kd", type=float, default=0.5,
                        help="Derivative gain (default 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No robot connection")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
