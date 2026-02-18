#!/usr/bin/env python3
"""
THEMIS Arm Diagnostics — Log & Plot commanded vs actual joint trajectories.

Runs a sinusoidal sweep while logging EVERY command sent and periodically
sampling the robot's actual joint state.  After the test, generates plots
showing:

  1. Commanded vs actual joint position over time
  2. Tracking error over time
  3. Command timing jitter (actual loop dt vs target dt)
  4. Joint velocity (numerical derivative of feedback)

This helps diagnose whether jerkiness is caused by:
  • Desktop send-side jitter (timing plot)
  • Network packet loss / reordering (gaps in feedback)
  • Robot-side WBC not tracking smoothly (cmd vs actual plot)
  • Gain tuning issues (tracking error magnitude)

Usage:
  python3 hw_interface/test_arm_diagnostics.py --robot-ip 192.168.0.11
  python3 hw_interface/test_arm_diagnostics.py --dry-run   # offline test

Outputs:
  arm_diagnostics_YYYYMMDD_HHMMSS.npz   — raw logged data
  arm_diagnostics_YYYYMMDD_HHMMSS.png   — multi-panel plot
"""

import argparse
import time
import sys
import os
import signal
import struct
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hw_interface.themis_udp_client import ThemisUDPClient, ThemisStateFeedback

# ── Constants ────────────────────────────────────────────────────────
ARM_JOINT_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw",
]
IDLE_R = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

_shutdown = False
def _sig(s, f):
    global _shutdown; _shutdown = True; print("\n[Diag] Stopping …")
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


def _send_ff(client, q_r, q_l, kp, kd):
    """Fire-and-forget both arms."""
    SIDE_R, SIDE_L, MSG = 2, 0xFE, 0x10
    dq = np.zeros(7, dtype=np.float64)
    u  = np.zeros(7, dtype=np.float64)
    for sb, q in [(SIDE_R, q_r), (SIDE_L, q_l)]:
        q_  = np.asarray(q, dtype=np.float64).ravel()
        kp_ = np.asarray(kp, dtype=np.float64).ravel()
        kd_ = np.asarray(kd, dtype=np.float64).ravel()
        pkt = (struct.pack('B', MSG) + struct.pack('B', sb)
               + q_.tobytes() + dq.tobytes() + u.tobytes()
               + kp_.tobytes() + kd_.tobytes())
        client._send(pkt)


def _spin_until(t):
    now = time.perf_counter()
    rem = t - now
    if rem <= 0: return
    if rem > 0.0005: time.sleep(rem - 0.0005)
    while time.perf_counter() < t: pass


class DryRunClient:
    """Simulates a robot with a simple first-order lag for testing."""
    _q_r = IDLE_R.copy()
    _q_l = IDLE_L.copy()
    _target_r = IDLE_R.copy()
    _target_l = IDLE_L.copy()

    def connect(self): print("[DryRun] Connected")
    def disconnect(self): print("[DryRun] Disconnected")
    def _send(self, data):
        # Simulate first-order tracking with some lag
        alpha = 0.02  # lag factor
        self._q_r += alpha * (self._target_r - self._q_r)
        self._q_l += alpha * (self._target_l - self._q_l)
    def get_state(self):
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = self._q_r.copy() + np.random.normal(0, 0.001, 7)
        fb.left_arm_q  = self._q_l.copy() + np.random.normal(0, 0.001, 7)
        fb.right_arm_dq = np.zeros(7)
        fb.left_arm_dq  = np.zeros(7)
        fb.right_arm_torque = np.random.normal(0, 0.1, 7)
        fb.left_arm_torque  = np.random.normal(0, 0.1, 7)
        fb.timestamp = time.time()
        return fb
    def set_target(self, q_r, q_l):
        self._target_r = q_r.copy()
        self._target_l = q_l.copy()


def run_diagnostics(client, rate_hz, kp, kd, amplitude, freq, sweep_time,
                    fb_rate_hz, is_dry_run=False):
    """Run sweep and log everything."""
    global _shutdown
    dt = 1.0 / rate_hz
    fb_interval = 1.0 / fb_rate_hz

    # Pre-allocate log arrays (oversize, trim later)
    max_cmd  = int(rate_hz * (sweep_time + 10)) + 1000
    max_fb   = int(fb_rate_hz * (sweep_time + 10)) + 1000

    cmd_times  = np.zeros(max_cmd)
    cmd_r      = np.zeros((max_cmd, 7))
    cmd_l      = np.zeros((max_cmd, 7))
    loop_dts   = np.zeros(max_cmd)

    fb_times   = np.zeros(max_fb)
    fb_r_q     = np.zeros((max_fb, 7))
    fb_l_q     = np.zeros((max_fb, 7))
    fb_r_dq    = np.zeros((max_fb, 7))
    fb_l_dq    = np.zeros((max_fb, 7))
    fb_r_tau   = np.zeros((max_fb, 7))
    fb_l_tau   = np.zeros((max_fb, 7))
    fb_valid   = np.zeros(max_fb, dtype=bool)

    ci = 0  # command index
    fi = 0  # feedback index

    print(f"\n{'='*72}")
    print(f"  THEMIS ARM DIAGNOSTICS")
    print(f"{'='*72}")
    print(f"  Command rate:  {rate_hz:.0f} Hz")
    print(f"  Feedback rate: {fb_rate_hz:.0f} Hz")
    print(f"  Sweep:         amp={np.degrees(amplitude):.1f}°  freq={freq:.2f} Hz  dur={sweep_time:.0f}s")
    print(f"  Gains:         kp={kp[0]:.1f}  kd={kd[0]:.2f}")
    print(f"  Joints logged: all 7 per arm")
    print(f"{'='*72}\n")

    # ── Phase 1: Ramp to IDLE (3 s) ─────────────────────────────────
    print("[Phase 1] Ramping to IDLE …")
    fb = client.get_state()
    start_r = fb.right_arm_q.copy() if fb.valid else IDLE_R.copy()
    start_l = fb.left_arm_q.copy()  if fb.valid else IDLE_L.copy()

    RAMP = 3.0
    t0 = time.perf_counter()
    tick = t0
    while not _shutdown:
        now = time.perf_counter()
        if now - t0 >= RAMP: break
        a = (now - t0) / RAMP
        q_r = (1-a)*start_r + a*IDLE_R
        q_l = (1-a)*start_l + a*IDLE_L
        if is_dry_run: client.set_target(q_r, q_l)
        _send_ff(client, q_r, q_l, kp, kd)
        tick += dt
        _spin_until(tick)

    # Hold IDLE 1 s
    t0 = time.perf_counter(); tick = t0
    while time.perf_counter() - t0 < 1.0:
        if is_dry_run: client.set_target(IDLE_R, IDLE_L)
        _send_ff(client, IDLE_R, IDLE_L, kp, kd)
        tick += dt; _spin_until(tick)

    if _shutdown: return None

    # ── Phase 2: Sweep + log ────────────────────────────────────────
    print("[Phase 2] Sinusoidal sweep — logging …")
    t0 = time.perf_counter()
    tick = t0
    last_fb_time = t0
    prev_tick = t0

    while not _shutdown:
        now = time.perf_counter()
        t = now - t0
        if t >= sweep_time: break

        wave = amplitude * np.sin(2.0 * np.pi * freq * t)
        q_r = IDLE_R.copy()
        q_l = IDLE_L.copy()
        q_r[0] += wave
        q_r[3] += wave * 0.5
        q_l[0] += wave
        q_l[3] -= wave * 0.5

        if is_dry_run: client.set_target(q_r, q_l)
        _send_ff(client, q_r, q_l, kp, kd)

        # Log command
        if ci < max_cmd:
            cmd_times[ci] = t
            cmd_r[ci] = q_r
            cmd_l[ci] = q_l
            loop_dts[ci] = now - prev_tick
            ci += 1
        prev_tick = now

        # Sample feedback at fb_rate_hz
        if now - last_fb_time >= fb_interval:
            last_fb_time = now
            fb = client.get_state()
            if fi < max_fb:
                fb_times[fi] = t
                fb_valid[fi] = fb.valid
                if fb.valid:
                    fb_r_q[fi]   = fb.right_arm_q
                    fb_l_q[fi]   = fb.left_arm_q
                    fb_r_dq[fi]  = fb.right_arm_dq
                    fb_l_dq[fi]  = fb.left_arm_dq
                    fb_r_tau[fi] = fb.right_arm_torque
                    fb_l_tau[fi] = fb.left_arm_torque
                fi += 1

        tick += dt
        _spin_until(tick)

    # ── Phase 3: Return to IDLE ──────────────────────────────────────
    print("[Phase 3] Returning to IDLE …")
    fb = client.get_state()
    cur_r = fb.right_arm_q.copy() if fb.valid else IDLE_R.copy()
    cur_l = fb.left_arm_q.copy()  if fb.valid else IDLE_L.copy()
    t0 = time.perf_counter(); tick = t0
    while not _shutdown:
        now = time.perf_counter()
        if now - t0 >= RAMP: break
        a = (now - t0) / RAMP
        q_r = (1-a)*cur_r + a*IDLE_R
        q_l = (1-a)*cur_l + a*IDLE_L
        if is_dry_run: client.set_target(q_r, q_l)
        _send_ff(client, q_r, q_l, kp, kd)
        tick += dt; _spin_until(tick)

    # Trim
    cmd_times = cmd_times[:ci]; cmd_r = cmd_r[:ci]; cmd_l = cmd_l[:ci]
    loop_dts = loop_dts[:ci]
    fb_times = fb_times[:fi]; fb_r_q = fb_r_q[:fi]; fb_l_q = fb_l_q[:fi]
    fb_r_dq = fb_r_dq[:fi]; fb_l_dq = fb_l_dq[:fi]
    fb_r_tau = fb_r_tau[:fi]; fb_l_tau = fb_l_tau[:fi]
    fb_valid = fb_valid[:fi]

    print(f"\n[Log] {ci} commands logged, {fi} feedback samples ({np.sum(fb_valid)} valid)")
    actual_hz = ci / sweep_time if sweep_time > 0 else 0
    print(f"[Log] Actual command rate: {actual_hz:.0f} Hz")

    return {
        'cmd_times': cmd_times, 'cmd_r': cmd_r, 'cmd_l': cmd_l,
        'loop_dts': loop_dts,
        'fb_times': fb_times, 'fb_r_q': fb_r_q, 'fb_l_q': fb_l_q,
        'fb_r_dq': fb_r_dq, 'fb_l_dq': fb_l_dq,
        'fb_r_tau': fb_r_tau, 'fb_l_tau': fb_l_tau,
        'fb_valid': fb_valid,
        'rate_hz': rate_hz, 'fb_rate_hz': fb_rate_hz,
        'amplitude': amplitude, 'freq': freq, 'sweep_time': sweep_time,
        'kp': kp, 'kd': kd,
    }


def save_and_plot(data, save_dir='.'):
    """Save raw data and generate diagnostic plots."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    npz_path = os.path.join(save_dir, f'arm_diagnostics_{ts}.npz')
    png_path = os.path.join(save_dir, f'arm_diagnostics_{ts}.png')

    np.savez_compressed(npz_path, **data)
    print(f"[Save] Data → {npz_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed — skipping plots")
        print("       Install with: pip install matplotlib")
        return npz_path, None

    cmd_t = data['cmd_times']
    fb_t  = data['fb_times']
    fb_ok = data['fb_valid']

    # Only use valid feedback
    fb_t_v  = fb_t[fb_ok]
    fb_r_v  = data['fb_r_q'][fb_ok]
    fb_l_v  = data['fb_l_q'][fb_ok]
    fb_rdq  = data['fb_r_dq'][fb_ok]
    fb_rtau = data['fb_r_tau'][fb_ok]

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=False)
    fig.suptitle(f"THEMIS Arm Diagnostics — {data['rate_hz']:.0f} Hz cmd, "
                 f"kp={data['kp'][0]:.0f} kd={data['kd'][0]:.1f}, "
                 f"amp={np.degrees(data['amplitude']):.1f}° freq={data['freq']:.2f} Hz",
                 fontsize=14, fontweight='bold')

    # ── Row 0: Commanded vs Actual — shoulder_pitch (joint 0) ────────
    ax = axes[0, 0]
    ax.plot(cmd_t, np.degrees(data['cmd_r'][:, 0]), 'b-', lw=0.5, alpha=0.7, label='cmd R[0]')
    if len(fb_t_v) > 0:
        ax.plot(fb_t_v, np.degrees(fb_r_v[:, 0]), 'r.-', lw=0.8, ms=2, label='actual R[0]')
    ax.set_ylabel('shoulder_pitch [deg]')
    ax.set_title('Right Arm — Joint 0 (shoulder_pitch)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cmd_t, np.degrees(data['cmd_r'][:, 3]), 'b-', lw=0.5, alpha=0.7, label='cmd R[3]')
    if len(fb_t_v) > 0:
        ax.plot(fb_t_v, np.degrees(fb_r_v[:, 3]), 'r.-', lw=0.8, ms=2, label='actual R[3]')
    ax.set_ylabel('elbow_pitch [deg]')
    ax.set_title('Right Arm — Joint 3 (elbow_pitch)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Tracking error ────────────────────────────────────────
    if len(fb_t_v) > 0:
        # Interpolate cmd at feedback times
        from scipy import interpolate
        try:
            cmd_r0_interp = interpolate.interp1d(cmd_t, data['cmd_r'][:, 0],
                                                  kind='linear', fill_value='extrapolate')(fb_t_v)
            cmd_r3_interp = interpolate.interp1d(cmd_t, data['cmd_r'][:, 3],
                                                  kind='linear', fill_value='extrapolate')(fb_t_v)
            err_0 = np.degrees(fb_r_v[:, 0] - cmd_r0_interp)
            err_3 = np.degrees(fb_r_v[:, 3] - cmd_r3_interp)

            ax = axes[1, 0]
            ax.plot(fb_t_v, err_0, 'r-', lw=0.8, label='error R[0]')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel('Tracking error [deg]')
            ax.set_title(f'Tracking Error — shoulder_pitch  (RMS={np.sqrt(np.mean(err_0**2)):.2f}°)')
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(fb_t_v, err_3, 'r-', lw=0.8, label='error R[3]')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel('Tracking error [deg]')
            ax.set_title(f'Tracking Error — elbow_pitch  (RMS={np.sqrt(np.mean(err_3**2)):.2f}°)')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'scipy interpolation failed:\n{e}',
                           transform=axes[1, 0].transAxes, ha='center')
            axes[1, 1].text(0.5, 0.5, 'N/A', transform=axes[1, 1].transAxes, ha='center')
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid feedback', transform=axes[1, 0].transAxes, ha='center')
        axes[1, 1].text(0.5, 0.5, 'No valid feedback', transform=axes[1, 1].transAxes, ha='center')

    # ── Row 2: Loop timing jitter ────────────────────────────────────
    dts_ms = data['loop_dts'][1:] * 1000.0  # skip first (invalid)
    target_dt_ms = 1000.0 / data['rate_hz']

    ax = axes[2, 0]
    ax.plot(cmd_t[1:], dts_ms, 'g-', lw=0.3, alpha=0.6)
    ax.axhline(target_dt_ms, color='k', ls='--', lw=1, label=f'target={target_dt_ms:.2f} ms')
    ax.set_ylabel('Loop dt [ms]')
    ax.set_title(f'Command Loop Timing  (mean={np.mean(dts_ms):.3f} ms, '
                 f'std={np.std(dts_ms):.3f} ms, max={np.max(dts_ms):.2f} ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.hist(dts_ms, bins=100, color='green', alpha=0.7, edgecolor='none')
    ax.axvline(target_dt_ms, color='k', ls='--', lw=1.5, label=f'target')
    ax.set_xlabel('Loop dt [ms]')
    ax.set_ylabel('Count')
    ax.set_title(f'Loop dt Distribution  (>{2*target_dt_ms:.1f}ms: '
                 f'{np.sum(dts_ms > 2*target_dt_ms)} overruns)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Feedback velocity & torque ────────────────────────────
    if len(fb_t_v) > 1:
        # Numerical velocity from position (to compare with reported dq)
        num_vel = np.diff(fb_r_v[:, 0]) / np.diff(fb_t_v)

        ax = axes[3, 0]
        ax.plot(fb_t_v[1:], np.degrees(num_vel), 'b-', lw=0.8, alpha=0.7, label='num d/dt q[0]')
        ax.plot(fb_t_v, np.degrees(fb_rdq[:, 0]), 'r-', lw=0.8, alpha=0.7, label='reported dq[0]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [deg/s]')
        ax.set_title('Right Arm Joint 0 — Velocity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3, 1]
        ax.plot(fb_t_v, fb_rtau[:, 0], 'b-', lw=0.8, alpha=0.7, label='τ R[0]')
        ax.plot(fb_t_v, fb_rtau[:, 3], 'r-', lw=0.8, alpha=0.7, label='τ R[3]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [Nm]')
        ax.set_title('Right Arm — Joint Torques')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[3, 0].text(0.5, 0.5, 'Insufficient feedback', transform=axes[3, 0].transAxes, ha='center')
        axes[3, 1].text(0.5, 0.5, 'Insufficient feedback', transform=axes[3, 1].transAxes, ha='center')

    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"[Plot] Figure → {png_path}")

    # Also try to show interactively
    try:
        matplotlib.use('TkAgg')
        plt.show()
    except Exception:
        pass

    return npz_path, png_path


def main():
    parser = argparse.ArgumentParser(description="THEMIS arm diagnostics — log & plot")
    parser.add_argument("--robot-ip", default="192.168.0.11")
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--rate", type=float, default=1000.0,
                        help="Command send rate in Hz (default: 1000)")
    parser.add_argument("--fb-rate", type=float, default=50.0,
                        help="Feedback sampling rate in Hz (default: 50)")
    parser.add_argument("--kp", type=float, default=30.0, help="P gain (default: 30)")
    parser.add_argument("--kd", type=float, default=2.0, help="D gain (default: 2)")
    parser.add_argument("--amplitude", type=float, default=0.5,
                        help="Sweep amplitude in rad (default: 0.5)")
    parser.add_argument("--freq", type=float, default=0.3,
                        help="Sweep frequency in Hz (default: 0.3)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Sweep duration in seconds (default: 10)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: cwd)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulated robot (no real hardware)")
    args = parser.parse_args()

    kp = np.full(7, args.kp)
    kd = np.full(7, args.kd)

    if args.dry_run:
        client = DryRunClient()
    else:
        client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port)

    client.connect()

    try:
        data = run_diagnostics(
            client, rate_hz=args.rate, kp=kp, kd=kd,
            amplitude=args.amplitude, freq=args.freq,
            sweep_time=args.duration, fb_rate_hz=args.fb_rate,
            is_dry_run=args.dry_run,
        )
        if data is not None:
            save_and_plot(data, save_dir=args.output_dir)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
