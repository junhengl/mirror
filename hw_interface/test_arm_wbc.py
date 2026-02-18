#!/usr/bin/env python3
"""
THEMIS Arm Test — Safe Version  (runs on the DESKTOP, no ROS2 needed)

This test communicates with wbc_udp_server.py on the robot, which uses
the OFFICIAL wbc_api instead of raw shared memory.

WHY THE OLD APPROACH BROKE THE ROBOT:
  The old shm_udp_server.py did:
    MM.connect()
    MM.RIGHT_ARM_JOINT_COMMAND.set(command)   ← raw shared memory write

  This COMPETED with the WBC for shared-memory ownership. The WBC lost
  control, the balance controller died, and the robot collapsed.

THE FIX — wbc_udp_server.py does:
    wbc_api.set_joint_states(chain, u, q, dq, kp, kd)   ← official API

  This goes THROUGH the WBC pipeline. Balance stays alive. Robot
  doesn't collapse.

Architecture:
  ┌─── Desktop ────────────────────────────┐
  │  this script (test_arm_wbc.py)         │
  │    uses ThemisUDPClient (pure Python)  │
  │    no ROS2 needed                      │
  └───── UDP over Ethernet ────────────────┘
                    │
  ┌─── Robot PC ───────────────────────────┐
  │  wbc_udp_server.py  (NEW — safe)       │
  │    wbc_api.get_joint_states()  → read  │
  │    wbc_api.set_joint_states()  → write │
  │    WBC stays in control ✓              │
  └────────────────────────────────────────┘

Usage:
  # 1. On Robot PC — start the SAFE server:
  ssh themis@192.168.0.11
  cd /home/themis/THEMIS/THEMIS
  python3 ~/wbc_udp_server.py --port 9870

  # 2. On Desktop — read-only first (SAFE, no commands):
  python3 hw_interface/test_arm_wbc.py --read-only

  # 3. If state looks correct, run the motion test:
  python3 hw_interface/test_arm_wbc.py

  # Dry-run (no robot needed):
  python3 hw_interface/test_arm_wbc.py --dry-run
"""

import argparse
import time
import sys
import os
import signal
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hw_interface.themis_udp_client import ThemisUDPClient, ThemisStateFeedback


# ── Arm joint names ─────────────────────────────────────────────────
ARM_JOINT_NAMES = ThemisUDPClient.ARM_JOINT_NAMES

# ── Nominal poses (from manipulation_macros.py) ─────────────────────
IDLE_R = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

# ── PD gains ─────────────────────────────────────────────────────────
# Soft gains — the WBC handles the actual motor PD loop,
# these are reference tracking gains sent to wbc_api.set_joint_states()
KP_SOFT = np.full(7, 30.0)
KD_SOFT = np.full(7,  2.0)


def lerp_pose(q0, q1, alpha):
    """Linear interpolation between two joint poses."""
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * q0 + alpha * q1


class DryRunClient:
    """Fake client for testing without a robot."""
    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    DEFAULT_KP = KP_SOFT
    DEFAULT_KD = KD_SOFT

    def connect(self):
        print("[DryRun] Connected (no real robot)")

    def disconnect(self):
        print("[DryRun] Disconnected")

    def get_state(self):
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = IDLE_R.copy()
        fb.left_arm_q  = IDLE_L.copy()
        fb.timestamp = time.time()
        return fb

    def send_arm_command(self, side, q, dq=None, u=None, kp=None, kd=None):
        return True

    def _send(self, data):
        """No-op for fire-and-forget in dry-run mode."""
        pass

    def send_manip_reference(self, *a, **kw):
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


def run_read_only(client, duration=10.0, rate_hz=2.0):
    """
    Read-only mode: prints joint state, sends NO commands.

    Use this first to verify communication without touching actuators.
    """
    global _shutdown_requested
    dt = 1.0 / rate_hz

    print("\n" + "=" * 72)
    print("  THEMIS ARM READ-ONLY TEST")
    print("  ██ NO commands will be sent — read only ██")
    print("=" * 72)

    t0 = time.time()
    read_count = 0
    fail_count = 0

    while (time.time() - t0) < duration and not _shutdown_requested:
        fb = client.get_state()
        elapsed = time.time() - t0

        if fb.valid:
            read_count += 1
            if read_count == 1:
                print(f"\n  ✓ First valid state received at t={elapsed:.1f}s")
            client.print_state(fb)
        else:
            fail_count += 1
            print(f"  [{elapsed:.1f}s] No valid state (attempt {fail_count}) — "
                  f"is wbc_udp_server.py running on the robot?")

        time.sleep(dt)

    print(f"\n[ReadOnly] Done. {read_count} successful reads, {fail_count} failures.")


def _send_both_arms_fire_and_forget(client, q_r, q_l, kp, kd):
    """
    Send arm commands for both sides WITHOUT waiting for ACK.

    At 1 kHz, waiting for a UDP round-trip (~0.2–1 ms) per side would
    consume the entire time budget.  Fire-and-forget keeps the loop
    deterministic — the server will still apply every packet it receives.
    """
    import struct
    SIDE_RIGHT = 2
    SIDE_LEFT  = 0xFE
    MSG_ARM_JOINT_CMD = 0x10
    dq = np.zeros(7, dtype=np.float64)
    u  = np.zeros(7, dtype=np.float64)

    for side_byte, q in [(SIDE_RIGHT, q_r), (SIDE_LEFT, q_l)]:
        q_  = np.asarray(q, dtype=np.float64).ravel()
        kp_ = np.asarray(kp, dtype=np.float64).ravel()
        kd_ = np.asarray(kd, dtype=np.float64).ravel()
        payload = (struct.pack('B', MSG_ARM_JOINT_CMD)
                   + struct.pack('B', side_byte)
                   + q_.tobytes() + dq.tobytes() + u.tobytes()
                   + kp_.tobytes() + kd_.tobytes())
        client._send(payload)


def _spin_wait_until(target_time):
    """
    High-resolution busy-wait.

    time.sleep() on Linux has ~1–4 ms jitter, which is too coarse for
    1 kHz.  We sleep for the bulk of the wait, then spin on perf_counter
    for the final 0.5 ms.
    """
    now = time.perf_counter()
    remaining = target_time - now
    if remaining <= 0:
        return
    # Sleep for most of the wait (leave 0.5 ms margin for spin)
    if remaining > 0.0005:
        time.sleep(remaining - 0.0005)
    # Spin for the last bit
    while time.perf_counter() < target_time:
        pass


def run_test(client, rate_hz=1000.0):
    """
    Main test sequence — sinusoidal arm sweep via wbc_api.

    Commands go: Desktop → UDP → wbc_udp_server.py → wbc_api.set_joint_states()
    This is the official API path, so WBC stays in control.

    KEY DESIGN RULE: The 1 kHz command loop NEVER blocks.
      • fire-and-forget UDP sends (no ACK wait)
      • spin-wait for precise timing
      • state feedback is read on a SEPARATE THREAD via a separate UDP
        socket so the command stream is never interrupted
    """
    global _shutdown_requested
    dt = 1.0 / rate_hz

    print("\n" + "=" * 72)
    print("  THEMIS ARM JOINT-SPACE TEST  (via official wbc_api)")
    print("=" * 72)
    print(f"  Server:    wbc_udp_server.py (uses wbc_api, NOT memory_manager)")
    print(f"  Loop rate: {rate_hz:.0f} Hz")
    print(f"  PD gains:  kp={KP_SOFT[0]:.1f}, kd={KD_SOFT[0]:.1f}")
    print(f"  Safety:    WBC stays in control — robot should NOT collapse")
    print("=" * 72)

    # ── Separate feedback socket (so cmd socket is never blocked) ────
    import socket as _socket, struct as _struct, threading as _threading

    fb_lock = _threading.Lock()
    fb_latest = {'fb': ThemisStateFeedback(), 'stop': False}

    def _feedback_reader():
        """Background thread: polls state on a SEPARATE UDP socket."""
        fb_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        fb_sock.settimeout(0.3)
        try:
            while not fb_latest['stop'] and not _shutdown_requested:
                try:
                    fb_sock.sendto(
                        _struct.pack('B', 0x01),  # MSG_STATE_REQUEST
                        (client.robot_ip, client.port),
                    )
                    data, _ = fb_sock.recvfrom(4096)
                    if len(data) > 1 and data[0] == 0x02:  # MSG_STATE_RESPONSE
                        fb = client._parse_state_response(data[1:])
                        with fb_lock:
                            fb_latest['fb'] = fb
                except _socket.timeout:
                    pass
                except Exception:
                    pass
                time.sleep(0.05)  # ~20 Hz feedback polling
        finally:
            fb_sock.close()

    fb_thread = _threading.Thread(target=_feedback_reader, daemon=True)
    fb_thread.start()

    def _get_latest_fb():
        with fb_lock:
            return fb_latest['fb']

    # ── Phase 0: Read current state ──────────────────────────────────
    print("\n[Phase 0] Reading initial joint state …")
    time.sleep(0.3)  # let feedback thread get first sample
    fb = _get_latest_fb()
    if not fb.valid:
        # Fall back to blocking read on main socket (only at startup)
        fb = client.get_state()

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
    t_start = time.perf_counter()
    next_tick = t_start
    while not _shutdown_requested:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed >= RAMP_TIME:
            break
        alpha = elapsed / RAMP_TIME
        q_r = lerp_pose(start_q_r, IDLE_R, alpha)
        q_l = lerp_pose(start_q_l, IDLE_L, alpha)
        _send_both_arms_fire_and_forget(client, q_r, q_l, KP_SOFT, KD_SOFT)
        next_tick += dt
        _spin_wait_until(next_tick)

    if _shutdown_requested:
        fb_latest['stop'] = True
        return

    # Hold IDLE briefly (0.5 s)
    t_start = time.perf_counter()
    next_tick = t_start
    while time.perf_counter() - t_start < 0.5:
        _send_both_arms_fire_and_forget(client, IDLE_R, IDLE_L, KP_SOFT, KD_SOFT)
        next_tick += dt
        _spin_wait_until(next_tick)

    fb = _get_latest_fb()
    if fb.valid:
        print("[Phase 1] Reached IDLE:")
        client.print_state(fb)

    # ── Phase 2: Sinusoidal sweep ────────────────────────────────────
    SWEEP_TIME = 10.0
    AMPLITUDE  = 0.5    # rad (~28.6°)
    FREQ       = 0.3    # Hz — slow

    print(f"\n[Phase 2] Sinusoidal sweep:")
    print(f"  amp = {np.degrees(AMPLITUDE):.1f}°, freq = {FREQ:.1f} Hz, duration = {SWEEP_TIME:.0f} s")
    print(f"  Modulating: shoulder_pitch (idx 0), shoulder_roll (idx 1), elbow_pitch (idx 3)")

    t_start = time.perf_counter()
    next_tick = t_start
    last_print = t_start
    step = 0
    while not _shutdown_requested:
        now = time.perf_counter()
        t = now - t_start
        if t >= SWEEP_TIME:
            break

        wave = AMPLITUDE * np.sin(2.0 * np.pi * FREQ * t)

        q_r = IDLE_R.copy()
        q_l = IDLE_L.copy()
        
        # Shoulder pitch modulation (±0.5 rad from IDLE)
        q_r[0] += wave
        q_l[0] += wave
        
        # Shoulder roll modulation (±20° around ±50°)
        #   Right: [30, 70]° → center 50°, amplitude 20°
        #   Left:  [-30, -70]° → center -50°, amplitude 20°
        sr_center_r = np.radians(50.0)
        sr_amp = np.radians(30.0)
        q_r[1] = sr_center_r - sr_amp * np.sin(2.0 * np.pi * FREQ * t)
        q_l[1] = -sr_center_r + sr_amp * np.sin(2.0 * np.pi * FREQ * t)
        
        # Elbow pitch modulation (±0.5 rad from IDLE, scaled 1.0)
        q_r[3] += wave * 1.0
        q_l[3] -= wave * 1.0


        _send_both_arms_fire_and_forget(client, q_r, q_l, KP_SOFT, KD_SOFT)

        # Print feedback once per second — NEVER BLOCKS the loop.
        # Reads the latest snapshot from the background thread.
        step += 1
        if now - last_print >= 1.0:
            last_print = now
            fb = _get_latest_fb()
            if fb.valid:
                err_r = np.linalg.norm(fb.right_arm_q - q_r)
                actual_hz = step / t if t > 0 else 0
                print(f"  t={t:5.1f}s  |  "
                      f"cmd_R[0]={np.degrees(q_r[0]):+6.1f}° "
                      f"fb_R[0]={np.degrees(fb.right_arm_q[0]):+6.1f}° "
                      f"err_R={np.degrees(err_r):5.2f}° | "
                      f"loop={actual_hz:.0f} Hz | "
                      f"τ_R={fb.right_arm_torque.round(2)}")

        next_tick += dt
        _spin_wait_until(next_tick)

    if _shutdown_requested:
        fb_latest['stop'] = True
        return

    # ── Phase 3: Return to IDLE ──────────────────────────────────────
    print(f"\n[Phase 3] Returning to IDLE over {RAMP_TIME:.1f} s …")
    fb = _get_latest_fb()
    cur_r = fb.right_arm_q.copy() if fb.valid else IDLE_R.copy()
    cur_l = fb.left_arm_q.copy()  if fb.valid else IDLE_L.copy()

    t_start = time.perf_counter()
    next_tick = t_start
    while not _shutdown_requested:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed >= RAMP_TIME:
            break
        alpha = elapsed / RAMP_TIME
        q_r = lerp_pose(cur_r, IDLE_R, alpha)
        q_l = lerp_pose(cur_l, IDLE_L, alpha)
        _send_both_arms_fire_and_forget(client, q_r, q_l, KP_SOFT, KD_SOFT)
        next_tick += dt
        _spin_wait_until(next_tick)

    # Hold IDLE (0.5 s)
    t_start = time.perf_counter()
    next_tick = t_start
    while time.perf_counter() - t_start < 0.5:
        _send_both_arms_fire_and_forget(client, IDLE_R, IDLE_L, KP_SOFT, KD_SOFT)
        next_tick += dt
        _spin_wait_until(next_tick)

    fb_latest['stop'] = True
    fb = _get_latest_fb()
    if fb.valid:
        print("\n[Done] Final state:")
        client.print_state(fb)

    print("\n[Test] Complete ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Themis arm test via wbc_api (safe — no robot collapse)")
    parser.add_argument("--robot-ip", default="192.168.0.11",
                        help="Robot PC IP (default: 192.168.0.11)")
    parser.add_argument("--port", type=int, default=9870,
                        help="UDP port (default: 9870)")
    parser.add_argument("--rate", type=float, default=1000.0,
                        help="Command loop rate Hz (default: 1000)")
    parser.add_argument("--read-only", action="store_true",
                        help="Only read state — send NO commands (safe)")
    parser.add_argument("--read-duration", type=float, default=10.0,
                        help="Duration for read-only mode (default: 10s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No real robot (mock client)")
    args = parser.parse_args()

    if args.dry_run:
        client = DryRunClient()
    else:
        client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port)

    client.connect()

    try:
        if args.read_only:
            run_read_only(client, duration=args.read_duration)
        else:
            run_test(client, rate_hz=args.rate)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
