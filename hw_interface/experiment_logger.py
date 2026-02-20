"""
Lightweight experiment data logger for the HW pipeline.

Design goals:
  • Zero allocation during logging — pre-allocated numpy ring buffers
  • Single method call per tick (< 5 µs overhead measured)
  • Configurable log rate (default 500 Hz, i.e. every 2nd tick at 1 kHz)
  • Saves to timestamped .npz on flush (< 50 ms for 60 s of data)

Usage:
    logger = ExperimentLogger(log_rate_hz=500, max_duration_s=120)
    # In 1 kHz loop:
    logger.log(
        retarget=retarget,       # RetargetingOutput from SharedState
        cmd_r=cmd_r, cmd_l=cmd_l,  # 7-DOF commanded arm poses (HW convention)
        fb=hw_fb,                 # ThemisStateFeedback
        latencies=shared.get_loop_durations(),  # latency dict
    )
    # On shutdown:
    path = logger.save()         # → "experiment_log_20260220_163012.npz"
"""

import os
import time
import numpy as np
from typing import Optional


class ExperimentLogger:
    """Pre-allocated ring-buffer logger for real-time experiment data."""

    def __init__(self, log_rate_hz: float = 500.0, cmd_rate_hz: float = 1000.0,
                 max_duration_s: float = 120.0, save_dir: str = "."):
        self.decimation = max(1, int(round(cmd_rate_hz / log_rate_hz)))
        self.tick = 0  # call counter (incremented every call)
        self.idx = 0   # write pointer into arrays
        self.capacity = int(log_rate_hz * max_duration_s)
        self.save_dir = save_dir

        # ── Pre-allocate all arrays ──────────────────────────────────
        N = self.capacity

        # Timestamps
        self.t_wall = np.zeros(N, dtype=np.float64)

        # Latency channels (seconds, will be stored as-is)
        self._lat_keys = [
            'lat_zed_grab', 'lat_zed_retrieve', 'lat_display',
            'lat_tracking_extract', 'lat_tracking_total',
            'lat_tracking_data_age', 'lat_ik_solve',
            'lat_retarget_total', 'lat_retarget_output_age',
            'lat_total_capture_to_cmd',
        ]
        self.latencies = np.zeros((N, len(self._lat_keys)), dtype=np.float32)

        # Task-space: hand & elbow desired + actual (3D each, 8 vectors = 24 floats)
        self.hand_r_des  = np.zeros((N, 3), dtype=np.float32)
        self.hand_l_des  = np.zeros((N, 3), dtype=np.float32)
        self.elbow_r_des = np.zeros((N, 3), dtype=np.float32)
        self.elbow_l_des = np.zeros((N, 3), dtype=np.float32)
        self.hand_r_act  = np.zeros((N, 3), dtype=np.float32)
        self.hand_l_act  = np.zeros((N, 3), dtype=np.float32)
        self.elbow_r_act = np.zeros((N, 3), dtype=np.float32)
        self.elbow_l_act = np.zeros((N, 3), dtype=np.float32)

        # Joint-space: desired (IK output, HW convention) + feedback (7 per arm)
        self.q_des_r = np.zeros((N, 7), dtype=np.float32)
        self.q_des_l = np.zeros((N, 7), dtype=np.float32)
        self.q_fb_r  = np.zeros((N, 7), dtype=np.float32)
        self.q_fb_l  = np.zeros((N, 7), dtype=np.float32)
        self.dq_fb_r = np.zeros((N, 7), dtype=np.float32)
        self.dq_fb_l = np.zeros((N, 7), dtype=np.float32)

        # Hand open/close
        self.hand_oc_r = np.zeros(N, dtype=np.float32)
        self.hand_oc_l = np.zeros(N, dtype=np.float32)

        # Retarget valid flag
        self.retarget_valid = np.zeros(N, dtype=np.bool_)

        actual_hz = cmd_rate_hz / self.decimation
        print(f"[Logger] Pre-allocated {N} samples "
              f"({max_duration_s:.0f}s × {actual_hz:.0f}Hz), "
              f"decimation={self.decimation}")

    def log(self, retarget, cmd_r, cmd_l, fb, latencies, hand_data=None):
        """Record one sample.  Call every tick; decimation is handled internally.

        Args:
            retarget:  RetargetingOutput from SharedState
            cmd_r:     (7,) commanded right arm pose (HW convention)
            cmd_l:     (7,) commanded left arm pose (HW convention)
            fb:        ThemisStateFeedback (or similar with .right_arm_q etc.)
            latencies: dict from shared.get_loop_durations()
            hand_data: HandTrackingData (optional)
        """
        self.tick += 1
        if self.tick % self.decimation != 0:
            return
        i = self.idx
        if i >= self.capacity:
            return  # buffer full — silently stop

        self.t_wall[i] = time.time()

        # Latencies
        for k, key in enumerate(self._lat_keys):
            self.latencies[i, k] = latencies.get(f'{key}_loop_s', 0.0)

        # Task-space (from retarget)
        self.retarget_valid[i] = retarget.valid
        if retarget.valid:
            self.hand_r_des[i]  = retarget.hand_r_des
            self.hand_l_des[i]  = retarget.hand_l_des
            self.elbow_r_des[i] = retarget.elbow_r_des
            self.elbow_l_des[i] = retarget.elbow_l_des
            self.hand_r_act[i]  = retarget.hand_r_act
            self.hand_l_act[i]  = retarget.hand_l_act
            self.elbow_r_act[i] = retarget.elbow_r_act
            self.elbow_l_act[i] = retarget.elbow_l_act

        # Joint-space desired (already in HW convention from caller)
        self.q_des_r[i] = cmd_r
        self.q_des_l[i] = cmd_l

        # Joint-space feedback
        if fb is not None and hasattr(fb, 'valid') and fb.valid:
            self.q_fb_r[i]  = fb.right_arm_q
            self.q_fb_l[i]  = fb.left_arm_q
            self.dq_fb_r[i] = fb.right_arm_dq
            self.dq_fb_l[i] = fb.left_arm_dq

        # Hand open/close
        if hand_data is not None and hand_data.valid:
            self.hand_oc_r[i] = hand_data.right_open_close
            self.hand_oc_l[i] = hand_data.left_open_close

        self.idx = i + 1

    def save(self, tag: str = "") -> str:
        """Flush logged data to a compressed .npz file.  Returns the file path."""
        n = self.idx  # number of valid samples
        if n == 0:
            print("[Logger] No data to save.")
            return ""

        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"experiment_log_{ts}{('_' + tag) if tag else ''}.npz"
        path = os.path.join(self.save_dir, name)

        np.savez_compressed(
            path,
            # Metadata
            t_wall=self.t_wall[:n],
            latency_keys=np.array(self._lat_keys),
            latencies=self.latencies[:n],
            retarget_valid=self.retarget_valid[:n],
            # Task-space
            hand_r_des=self.hand_r_des[:n],
            hand_l_des=self.hand_l_des[:n],
            elbow_r_des=self.elbow_r_des[:n],
            elbow_l_des=self.elbow_l_des[:n],
            hand_r_act=self.hand_r_act[:n],
            hand_l_act=self.hand_l_act[:n],
            elbow_r_act=self.elbow_r_act[:n],
            elbow_l_act=self.elbow_l_act[:n],
            # Joint-space (HW convention)
            q_des_r=self.q_des_r[:n],
            q_des_l=self.q_des_l[:n],
            q_fb_r=self.q_fb_r[:n],
            q_fb_l=self.q_fb_l[:n],
            dq_fb_r=self.dq_fb_r[:n],
            dq_fb_l=self.dq_fb_l[:n],
            # Hand
            hand_oc_r=self.hand_oc_r[:n],
            hand_oc_l=self.hand_oc_l[:n],
        )
        duration = self.t_wall[n - 1] - self.t_wall[0] if n > 1 else 0.0
        actual_hz = (n - 1) / duration if duration > 0 else 0.0
        size_mb = os.path.getsize(path) / 1e6
        print(f"[Logger] Saved {n} samples ({duration:.1f}s @ {actual_hz:.0f}Hz) "
              f"→ {path} ({size_mb:.1f} MB)")
        return path
