#!/usr/bin/env python3
"""
Plot server-side log from wbc_udp_server.py --log

Shows:
  1. Commanded vs actual q per joint (server-side)
  2. set_joint_states() latency per call
  3. Inter-arrival time of commands (reveals packet bunching/drops)

Usage:
  # After running test, scp the log from robot:
  scp themis@192.168.0.11:~/server_log_*.npz .
  
  # Then plot (use venv python if available):
  .venv/bin/python hw_interface/plot_server_log.py server_log_20250101_120000.npz
  
  # Or with system python (if matplotlib installed):
  python3 hw_interface/plot_server_log.py server_log_20250101_120000.npz
"""

import sys
import os

# Try to ensure we're using a python with matplotlib+scipy
try:
    import matplotlib
    import scipy
except ImportError as e:
    # Try to find venv python
    venv_path = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')
    if os.path.exists(venv_path):
        print(f"[Warning] matplotlib/scipy not in current python. Re-running with {venv_path}...", file=sys.stderr)
        import subprocess
        subprocess.run([venv_path] + sys.argv)
        sys.exit(0)
    else:
        print(f"Error: matplotlib or scipy not found.", file=sys.stderr)
        print(f"  Install with: pip install matplotlib scipy", file=sys.stderr)
        sys.exit(1)

# Set Agg backend BEFORE importing pyplot (no display needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
import numpy as np

def plot_server_log(npz_path):
    data = np.load(npz_path)
    recv_t = data['recv_t']
    set_t  = data['set_t']
    q_r    = data['q_r']
    q_l    = data['q_l']
    fb_q_r = data['fb_q_r']
    fb_q_l = data['fb_q_l']

    n = len(recv_t)
    if n == 0:
        print("No entries in log.")
        return

    # Time relative to start
    t0 = recv_t[0]
    t = recv_t - t0

    # set_joint_states() call latency
    latency_us = (set_t - recv_t) * 1e6

    # Inter-arrival times
    dt_ms = np.diff(recv_t) * 1000.0

    joint_names = ["sh_pitch", "sh_roll", "sh_yaw", "el_pitch",
                   "el_yaw", "wr_pitch", "wr_yaw"]

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle(f"Server-Side Log — {npz_path}\n{n} cmds over {t[-1]:.1f}s "
                 f"({n/t[-1]:.0f} cmd/s avg)", fontsize=13, fontweight='bold')

    # ── Cmd vs actual for joints 0 and 3 (right arm) ────────────────
    for col, ji in enumerate([0, 3]):
        ax = axes[0, col]
        # Only plot where q_r is non-zero (right arm cmds)
        mask_r = np.any(q_r != 0, axis=1)
        if mask_r.any():
            ax.plot(t[mask_r], np.degrees(q_r[mask_r, ji]), 'b-', lw=0.5, alpha=0.7, label='cmd')
            ax.plot(t[mask_r], np.degrees(fb_q_r[mask_r, ji]), 'r-', lw=0.8, alpha=0.7, label='actual')
        ax.set_ylabel(f'{joint_names[ji]} [deg]')
        ax.set_title(f'Right Arm — {joint_names[ji]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Tracking error ───────────────────────────────────────────────
    for col, ji in enumerate([0, 3]):
        ax = axes[1, col]
        mask_r = np.any(q_r != 0, axis=1) & np.any(fb_q_r != 0, axis=1)
        if mask_r.any():
            err = np.degrees(fb_q_r[mask_r, ji] - q_r[mask_r, ji])
            ax.plot(t[mask_r], err, 'r-', lw=0.5)
            rms = np.sqrt(np.mean(err**2))
            ax.set_title(f'Tracking Error {joint_names[ji]}  (RMS={rms:.2f}°)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('Error [deg]')
        ax.grid(True, alpha=0.3)

    # ── set_joint_states() latency ───────────────────────────────────
    ax = axes[2, 0]
    ax.plot(t, latency_us, 'g-', lw=0.3, alpha=0.6)
    ax.set_ylabel('set_joint_states latency [µs]')
    ax.set_title(f'API Call Latency  (mean={np.mean(latency_us):.0f} µs, '
                 f'max={np.max(latency_us):.0f} µs, p99={np.percentile(latency_us, 99):.0f} µs)')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.hist(latency_us, bins=100, color='green', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Latency [µs]')
    ax.set_ylabel('Count')
    ax.set_title('API Call Latency Distribution')
    ax.grid(True, alpha=0.3)

    # ── Command inter-arrival timing ─────────────────────────────────
    ax = axes[3, 0]
    ax.plot(t[1:], dt_ms, 'b-', lw=0.3, alpha=0.6)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Inter-arrival [ms]')
    ax.set_title(f'CMD Inter-Arrival  (mean={np.mean(dt_ms):.3f} ms, '
                 f'std={np.std(dt_ms):.3f} ms, max={np.max(dt_ms):.2f} ms)')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.hist(dt_ms, bins=100, color='blue', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Inter-arrival [ms]')
    ax.set_ylabel('Count')
    pct_over2 = 100.0 * np.sum(dt_ms > 2.0) / len(dt_ms) if len(dt_ms) > 0 else 0
    ax.set_title(f'Inter-Arrival Distribution  ({pct_over2:.1f}% > 2 ms)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = npz_path.replace('.npz', '.png')
    fig.savefig(out, dpi=150)
    print(f"[Plot] Saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <server_log_*.npz>")
        sys.exit(1)
    plot_server_log(sys.argv[1])
