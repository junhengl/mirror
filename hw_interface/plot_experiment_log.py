#!/usr/bin/env python3
"""
Plot experiment log data saved by ExperimentLogger.

Usage:
    python hw_interface/plot_experiment_log.py experiment_log_20260220_163012.npz
    python hw_interface/plot_experiment_log.py  # auto-picks latest
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_log(path: str) -> dict:
    """Load .npz and return as a plain dict."""
    d = dict(np.load(path, allow_pickle=True))
    # Decode latency key names
    if 'latency_keys' in d:
        d['latency_keys'] = [s if isinstance(s, str) else s.decode() for s in d['latency_keys']]
    return d


def plot_latencies(d, ax=None):
    """Plot latency breakdown over time."""
    t = d['t_wall'] - d['t_wall'][0]
    lat = d['latencies'] * 1000  # → ms
    keys = d['latency_keys']

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    # Stack the important ones
    important = [
        ('lat_zed_grab', 'ZED grab'),
        ('lat_zed_retrieve', 'Body detect'),
        ('lat_display', 'Display'),
        ('lat_tracking_extract', 'Extract'),
        ('lat_tracking_data_age', 'Data age→retarget'),
        ('lat_ik_solve', 'IK solve'),
        ('lat_retarget_output_age', 'Output age→HW'),
    ]
    for key_name, label in important:
        if key_name in keys:
            idx = keys.index(key_name)
            ax.plot(t, lat[:, idx], label=label, alpha=0.8, linewidth=0.8)

    # End-to-end as bold
    if 'lat_total_capture_to_cmd' in keys:
        idx = keys.index('lat_total_capture_to_cmd')
        ax.plot(t, lat[:, idx], label='END-TO-END', color='black',
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Pipeline Latency Breakdown')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return ax


def plot_task_space(d, side='right'):
    """Plot hand and elbow desired vs actual (3D components)."""
    t = d['t_wall'] - d['t_wall'][0]
    valid = d['retarget_valid']
    s = 'r' if side == 'right' else 'l'
    labels_xyz = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    fig.suptitle(f'{side.capitalize()} Arm — Task Space (des vs act)')

    for j in range(3):
        # Hand
        ax = axes[0, j]
        ax.plot(t, d[f'hand_{s}_des'][:, j], label='desired', alpha=0.8)
        ax.plot(t, d[f'hand_{s}_act'][:, j], label='actual', alpha=0.8)
        ax.set_ylabel(f'Hand {labels_xyz[j]} (m)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Elbow
        ax = axes[1, j]
        ax.plot(t, d[f'elbow_{s}_des'][:, j], label='desired', alpha=0.8)
        ax.plot(t, d[f'elbow_{s}_act'][:, j], label='actual', alpha=0.8)
        ax.set_ylabel(f'Elbow {labels_xyz[j]} (m)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_joint_tracking(d, side='right'):
    """Plot desired vs feedback joint angles."""
    t = d['t_wall'] - d['t_wall'][0]
    s = 'r' if side == 'right' else 'l'
    joint_names = ['sh_pitch', 'sh_roll', 'sh_yaw',
                   'elb_pitch', 'elb_yaw', 'wr_pitch', 'wr_yaw']

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{side.capitalize()} Arm — Joint Tracking (deg)')
    axes_flat = axes.flatten()

    for j in range(7):
        ax = axes_flat[j]
        q_des = np.degrees(d[f'q_des_{s}'][:, j])
        q_fb  = np.degrees(d[f'q_fb_{s}'][:, j])
        ax.plot(t, q_des, label='cmd', alpha=0.8, linewidth=0.8)
        ax.plot(t, q_fb,  label='fb',  alpha=0.8, linewidth=0.8)
        ax.set_ylabel(f'{joint_names[j]} (°)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide the 8th subplot
    axes_flat[7].set_visible(False)
    axes_flat[-2].set_xlabel('Time (s)')
    axes_flat[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Find latest experiment log
        logs = sorted(glob.glob('experiment_log_*.npz'))
        if not logs:
            print("No experiment_log_*.npz found.  Pass a path as argument.")
            sys.exit(1)
        path = logs[-1]
        print(f"Using latest: {path}")

    d = load_log(path)
    n = len(d['t_wall'])
    dur = d['t_wall'][-1] - d['t_wall'][0] if n > 1 else 0
    print(f"Loaded {n} samples, {dur:.1f}s")

    # Latency plot
    fig_lat, ax_lat = plt.subplots(figsize=(14, 5))
    plot_latencies(d, ax=ax_lat)
    lat_file = os.path.splitext(path)[0] + '_latency.png'
    fig_lat.savefig(lat_file, dpi=100, bbox_inches='tight')
    print(f"Saved: {lat_file}")
    plt.close(fig_lat)

    # Task-space (both arms)
    fig_r = plot_task_space(d, side='right')
    fig_r.savefig(os.path.splitext(path)[0] + '_task_right.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.splitext(path)[0] + '_task_right.png'}")
    plt.close(fig_r)
    
    fig_l = plot_task_space(d, side='left')
    fig_l.savefig(os.path.splitext(path)[0] + '_task_left.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.splitext(path)[0] + '_task_left.png'}")
    plt.close(fig_l)

    # Joint tracking (both arms)
    fig_j_r = plot_joint_tracking(d, side='right')
    fig_j_r.savefig(os.path.splitext(path)[0] + '_joint_right.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.splitext(path)[0] + '_joint_right.png'}")
    plt.close(fig_j_r)
    
    fig_j_l = plot_joint_tracking(d, side='left')
    fig_j_l.savefig(os.path.splitext(path)[0] + '_joint_left.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.splitext(path)[0] + '_joint_left.png'}")
    plt.close(fig_j_l)


if __name__ == '__main__':
    main()
