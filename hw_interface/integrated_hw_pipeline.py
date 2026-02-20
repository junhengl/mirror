#!/usr/bin/env python3
"""
Integrated Body Tracking → Retargeting → Hardware Pipeline

Runs on the DESKTOP PC.  Architecture:

    ┌─────────────────────────────────────────────── Desktop PC ──────────────────────────────────────────┐
    │                                                                                                     │
    │  [ZED Camera]                                                                                       │
    │       │  30 Hz                                                                                      │
    │       ▼                                                                                             │
    │  BodyTrackingNode  ─── ArmTrackingData ──▶  RetargetingNode  ─── q_des (28) ──▶  HardwareSender    │
    │                                                   ▲                                    │            │
    │                                                   │                              UDP   │            │
    │                                              RobotFeedback                             │            │
    │                                                   │                                    ▼            │
    └───────────────────────────────────────────────────│────────────────────────── Ethernet ──┘           │
                                                        │                                                  
    ┌─────────────────────────────────────────────── Robot PC ────────────────────────────────────────────┐
    │                                                                                                     │
    │  shm_udp_server.py  ◀──UDP──  joint cmds                                                           │
    │       │                                                                                             │
    │       ▼                                                                                             │
    │  Shared Memory  ─▶  WBC / Actuator Threads  ─▶  BEAR Motors                                        │
    │       │                                                                                             │
    │       ▼                                                                                             │
    │  Joint State    ──UDP──▶  Desktop (feedback)                                                        │
    │                                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────┘

This script replaces real_time_sim/main.py for hardware deployment.
It reuses:
  - BodyTrackingNode  (unchanged — ZED camera on desktop)
  - RetargetingNode   (unchanged — IK solver)
  - SharedState       (unchanged — inter-thread communication)
but replaces the MuJoCo simulation and PD controller with the UDP
hardware interface.

Usage:
    sudo python3 hw_interface/integrated_hw_pipeline.py \\
        --robot-ip 192.168.1.100 --port 9870

    # With dummy tracking (no ZED camera):
    sudo python3 hw_interface/integrated_hw_pipeline.py \\
        --robot-ip 192.168.1.100 --no-camera

    # Dry-run (no robot, no camera):
    python3 hw_interface/integrated_hw_pipeline.py --dry-run
"""

import os
import sys
import time
import signal
import argparse
import threading
import numpy as np
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Import submodules directly to avoid pulling in MuJoCo via __init__.py
from real_time_sim.config import PipelineConfig, RetargetingConfig, TrackingConfig
from real_time_sim.shared_state import SharedState, RobotState, RobotFeedback, RetargetingOutput
from real_time_sim.nodes.body_tracking_node import BodyTrackingNode
from real_time_sim.nodes.retargeting_node import RetargetingNode
from real_time_sim.joint_mapping import JointMapping

from hw_interface.themis_udp_client import ThemisUDPClient, ThemisStateFeedback


# ═══════════════════════════════════════════════════════════════════════
# Joint convention mapping:  KinDynLib IK output  →  Themis hardware
# ═══════════════════════════════════════════════════════════════════════
#
# The retargeting node outputs q_des as 28 joints in KinDynLib order:
#   [right_leg(6), left_leg(6), right_arm(7), left_arm(7), head(2)]
#
# Right arm = indices 12..18,  Left arm = indices 19..25
# Within each arm: [shoulder_pitch, shoulder_roll, shoulder_yaw,
#                   elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw]
#
# In the simulation, a JointMapping converts KinDynLib ↔ MuJoCo.
# For hardware, we need KinDynLib → Themis HW.
#
# The existing _create_default_joint_mapping() in config.py defines:
#   sign[15] = -1  (elbow_pitch_R)
#   sign[16] = -1  (forearm_roll_R / elbow_yaw_R)
#   sign[17] = -1  (forearm_pitch_R / wrist_pitch_R)
#   sign[18] = -1  (wrist_roll_R / wrist_yaw_R)
#   sign[21] = -1  (upperarm_yaw_L / shoulder_yaw_L)
#   offset[13] = -pi/2  (shoulder_roll_R)
#   offset[14] = +pi/2  (shoulder_yaw_R)
#   offset[15] = +pi/2  (elbow_pitch_R)
#   offset[20] = +pi/2  (shoulder_roll_L)
#   offset[21] = -pi/2  (shoulder_yaw_L)
#   offset[22] = +pi/2  (elbow_pitch_L)
#
# These map KinDynLib ← MuJoCo.  Since MuJoCo was built from the same
# URDF as the hardware, the MuJoCo convention IS the hardware convention.
# So KinDynLib → HW is the *reverse* mapping:
#   q_hw = sign * (q_kin - offset)
#
# We reuse JointMapping.reverse_q() for this.


# ═══════════════════════════════════════════════════════════════════════
# Nominal arm poses (hardware convention, from manipulation_macros.py)
# ═══════════════════════════════════════════════════════════════════════
IDLE_R_HW = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
IDLE_L_HW = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])


class HardwareSenderNode:
    """
    Reads retargeting output from SharedState and sends arm joint
    commands to the Themis robot over UDP.

    Also reads robot state feedback and writes it back to SharedState
    so the retargeting node can use it for IK warm-starting.

    Runs at ~100 Hz (limited by network round-trip).
    """

    def __init__(self, config: PipelineConfig, shared: SharedState,
                 client: ThemisUDPClient, use_manip_ref: bool = True):
        self.config = config
        self.shared = shared
        self.client = client
        self.use_manip_ref = use_manip_ref

        # Joint mapping (KinDynLib ↔ MuJoCo/HW)
        self.joint_mapping = JointMapping(config.joint_mapping)

        # Command rate
        self.command_rate = 100.0   # Hz
        self.command_dt   = 1.0 / self.command_rate

        # Safety: ramp from current pose to tracking over this duration
        self.blend_duration = 3.0   # seconds
        self.blend_start_time: Optional[float] = None
        self.blend_start_q_r: Optional[np.ndarray] = None
        self.blend_start_q_l: Optional[np.ndarray] = None

        # State
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.tracking_active = False

        # Timing
        self.last_stats_time = time.time()
        self.loop_count = 0

        # PD gains for arm control
        self.kp = np.full(7, 200.0)
        self.kd = np.full(7,   2.0)

        # Last commanded poses (for safety fallback)
        self.last_cmd_r = IDLE_R_HW.copy()
        self.last_cmd_l = IDLE_L_HW.copy()

    def start(self):
        """Start the hardware sender thread."""
        self.running = True
        self.thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.thread.start()
        print("[HWSender] Started hardware sender node (100 Hz)")

    def stop(self):
        """Stop the hardware sender thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[HWSender] Stopped")

    def _sender_loop(self):
        """Main loop: read retarget output, send to robot, read feedback."""
        while self.running and not self.shared.is_shutdown_requested():
            loop_start = time.perf_counter()

            try:
                # 1. Read robot feedback and publish to shared state
                _t_fb = time.perf_counter()
                self._read_feedback()
                self.shared.set_loop_duration(
                    'lat_udp_feedback_rtt', time.perf_counter() - _t_fb)

                # 2. Read retargeting output and send to robot
                self._send_commands()

            except Exception as e:
                print(f"[HWSender] Error: {e}")

            # Timing stats
            self.loop_count += 1
            if time.time() - self.last_stats_time >= 2.0:
                hz = self.loop_count / (time.time() - self.last_stats_time)
                self.shared.update_timing('control', hz)
                self.loop_count = 0
                self.last_stats_time = time.time()

            # Rate limiting
            elapsed = time.perf_counter() - loop_start
            if elapsed < self.command_dt:
                time.sleep(self.command_dt - elapsed)

    def _read_feedback(self):
        """Read robot state from UDP and write to SharedState."""
        fb = self.client.get_state()
        if not fb.valid:
            return

        # Build RobotFeedback for the retargeting node
        # The retargeting node expects 28 joints in KinDynLib convention,
        # so we need to convert HW → KinDynLib via forward_q().
        robot_fb = RobotFeedback()
        robot_fb.timestamp = fb.timestamp

        # Assemble 28-joint vector in MuJoCo/HW order:
        #   [right_leg(6), left_leg(6), right_arm(7), left_arm(7), head(2)]
        q_hw  = np.zeros(28, dtype=np.float64)
        dq_hw = np.zeros(28, dtype=np.float64)

        # Fill arm joints (legs and head stay at zero — not tracked)
        q_hw[12:19] = fb.right_arm_q
        q_hw[19:26] = fb.left_arm_q
        dq_hw[12:19] = fb.right_arm_dq
        dq_hw[19:26] = fb.left_arm_dq

        # Convert to KinDynLib convention for the retargeting node
        # forward_q: q_kin = sign * q_hw + offset
        q_kin  = self.joint_mapping.forward_q(q_hw)
        dq_kin = self.joint_mapping.forward_dq(dq_hw)

        robot_fb.q  = q_hw    # SharedState stores in MuJoCo/HW convention
        robot_fb.dq = dq_hw
        robot_fb.base_pos = fb.base_position.copy()

        self.shared.set_robot_feedback(robot_fb)

    def _send_commands(self):
        """Read retarget output and send to robot."""
        retarget = self.shared.get_retarget_output()
        
        # Track end-to-end pipeline latency
        if retarget.valid and retarget.source_capture_ts > 0:
            _t_now = time.time()
            self.shared.set_loop_duration(
                'lat_retarget_output_age', _t_now - retarget.timestamp)
            self.shared.set_loop_duration(
                'lat_total_capture_to_cmd', _t_now - retarget.source_capture_ts)
        
        fsm_state = self.shared.get_fsm_state()

        if not retarget.valid:
            # No valid tracking — hold last commanded pose
            self._send_arm_poses(self.last_cmd_r, self.last_cmd_l)
            return

        # Extract arm joints from the 28-element IK output (KinDynLib convention)
        # q_des layout: [right_leg(6), left_leg(6), right_arm(7), left_arm(7), head(2)]
        q_kin = np.zeros(28, dtype=np.float64)
        q_kin[:] = retarget.q_des

        # Convert KinDynLib → HW (MuJoCo) convention
        # reverse_q: q_hw = sign * (q_kin - offset)
        q_hw = self.joint_mapping.reverse_q(q_kin)

        # Extract arm-only (7 DOF each)
        q_r_hw = q_hw[12:19].copy()
        q_l_hw = q_hw[19:26].copy()

        # ── Blend logic ──────────────────────────────────────────────
        if not self.tracking_active:
            # First valid tracking frame — start blending
            self.tracking_active = True
            self.blend_start_time = time.time()
            self.blend_start_q_r = self.last_cmd_r.copy()
            self.blend_start_q_l = self.last_cmd_l.copy()
            print("[HWSender] Tracking active — blending to tracked pose …")

        if self.blend_start_time is not None:
            alpha = (time.time() - self.blend_start_time) / self.blend_duration
            if alpha >= 1.0:
                self.blend_start_time = None  # Blend complete
                print("[HWSender] Blend complete — full tracking mode")
            else:
                alpha = np.clip(alpha, 0.0, 1.0)
                # Smooth blend (ease-in-out via cosine)
                alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
                q_r_hw = (1.0 - alpha) * self.blend_start_q_r + alpha * q_r_hw
                q_l_hw = (1.0 - alpha) * self.blend_start_q_l + alpha * q_l_hw

        # ── Safety clamp ─────────────────────────────────────────────
        # Limit per-step change (max ~1 rad/s at 100 Hz = 0.01 rad/step)
        MAX_DELTA = 0.05  # rad per step (≈ 2.9° at 100 Hz = ~5 rad/s max)
        delta_r = np.clip(q_r_hw - self.last_cmd_r, -MAX_DELTA, MAX_DELTA)
        delta_l = np.clip(q_l_hw - self.last_cmd_l, -MAX_DELTA, MAX_DELTA)
        q_r_hw = self.last_cmd_r + delta_r
        q_l_hw = self.last_cmd_l + delta_l

        # Send
        self._send_arm_poses(q_r_hw, q_l_hw)

        # Save for next iteration
        self.last_cmd_r = q_r_hw.copy()
        self.last_cmd_l = q_l_hw.copy()

    def _send_arm_poses(self, q_r: np.ndarray, q_l: np.ndarray):
        """Send arm poses to robot via the configured command path."""
        if self.use_manip_ref:
            self.client.send_manip_reference(
                right_arm_pose=q_r, left_arm_pose=q_l,
                right_mode=100.0, right_phase=1.0,   # POSE / STANDBY
                left_mode=100.0,  left_phase=1.0,
            )
        else:
            self.client.send_arm_command('right', q=q_r, kp=self.kp, kd=self.kd)
            self.client.send_arm_command('left',  q=q_l, kp=self.kp, kd=self.kd)


class DummyClient:
    """Minimal mock client for dry-run mode."""
    def connect(self):
        print("[DryRun] Client connected (no real robot)")
    def disconnect(self):
        print("[DryRun] Client disconnected")
    def get_state(self):
        fb = ThemisStateFeedback()
        fb.valid = True
        fb.right_arm_q = IDLE_R_HW.copy()
        fb.left_arm_q  = IDLE_L_HW.copy()
        fb.timestamp = time.time()
        return fb
    def send_arm_command(self, *a, **kw): return True
    def send_manip_reference(self, *a, **kw): return True
    def print_state(self, fb=None): pass


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def print_banner():
    print("=" * 72)
    print("  THEMIS HARDWARE — Body Tracking & Retargeting Pipeline")
    print("=" * 72)
    print("  Components:")
    print("    • ZED Body Tracking       (30 Hz, desktop)")
    print("    • IK-based Retargeting    (500 Hz, desktop)")
    print("    • Hardware Sender         (100 Hz, desktop → robot via UDP)")
    print("    • Robot WBC + Actuators   (on robot PC)")
    print("=" * 72)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Integrated body tracking + retargeting → Themis hardware")
    parser.add_argument("--robot-ip", default="192.168.0.11",
                        help="Robot PC IP address (default: 192.168.0.11)")
    parser.add_argument("--port", type=int, default=9870,
                        help="UDP bridge port (default: 9870)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Use dummy tracking (no ZED camera)")
    parser.add_argument("--direct", action="store_true",
                        help="Send via JOINT_COMMAND (bypass WBC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No real robot (mock UDP client)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Hardware sender rate in Hz (default: 100)")
    parser.add_argument("--blend-time", type=float, default=3.0,
                        help="Seconds to blend from idle to tracked pose (default: 3)")
    parser.add_argument("--hang-height", type=float, default=1.3,
                        help="Robot hanging height in meters for IK (default: 1.3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()

    print_banner()

    # ── Configuration ────────────────────────────────────────────────
    config = PipelineConfig()
    config.verbose = args.verbose
    config.sim.base_height = args.hang_height

    # Use softer PD gains for retargeting when on real hardware
    # (the robot's own WBC handles the final PD loop)
    config.retarget.retarget_rate = 500.0
    config.retarget.retarget_dt = 1.0 / config.retarget.retarget_rate

    # ── Shared state ─────────────────────────────────────────────────
    shared = SharedState()

    # ── UDP client ───────────────────────────────────────────────────
    if args.dry_run:
        client = DummyClient()
    else:
        client = ThemisUDPClient(robot_ip=args.robot_ip, port=args.port,
                                  timeout=0.1)
    client.connect()

    # ── Initialize robot feedback with current hardware state ────────
    print("[Main] Reading initial robot state from hardware …")
    fb = client.get_state()
    if fb.valid:
        print("[Main] Got initial state from robot")
        initial_fb = RobotFeedback()
        initial_fb.timestamp = fb.timestamp
        q_hw = np.zeros(28, dtype=np.float64)
        q_hw[12:19] = fb.right_arm_q
        q_hw[19:26] = fb.left_arm_q
        initial_fb.q = q_hw
        initial_fb.base_pos = fb.base_position
        shared.set_robot_feedback(initial_fb)
    else:
        print("[Main] WARNING: Could not read robot state — using defaults")

    # ── Create nodes ─────────────────────────────────────────────────
    print("[Main] Initializing nodes …")

    hw_sender = HardwareSenderNode(config, shared, client,
                                    use_manip_ref=not args.direct)
    hw_sender.command_rate = args.rate
    hw_sender.command_dt = 1.0 / args.rate
    hw_sender.blend_duration = args.blend_time

    retargeter = RetargetingNode(config, shared)
    tracker = BodyTrackingNode(config, shared)

    # ── Signal handler ───────────────────────────────────────────────
    def signal_handler(sig, frame):
        print("\n[Main] Shutdown requested …")
        shared.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── Start nodes (downstream → upstream) ──────────────────────────
    try:
        print("\n[Main] Starting nodes …")

        # 1. Hardware sender (needs retarget output)
        hw_sender.start()

        # 2. Retargeting (needs tracking data)
        retargeter.start()

        # 3. Body tracking (upstream)
        tracker.start()

        print("\n[Main] All nodes started.  Pipeline running.")
        print("[Main] Press Ctrl+C to exit\n")

        # ── Status printing loop ─────────────────────────────────────
        last_print = time.time()
        while not shared.is_shutdown_requested():
            time.sleep(0.5)

            if time.time() - last_print >= 3.0:
                last_print = time.time()
                stats = shared.get_timing_stats()
                retarget_out = shared.get_retarget_output()
                hw_fb = client.get_state() if not args.dry_run else ThemisStateFeedback(valid=False)

                print(f"[Status] Tracking: {stats.get('tracking_hz', 0):.0f} Hz | "
                      f"Retarget: {stats.get('retarget_hz', 0):.0f} Hz | "
                      f"HW Sender: {stats.get('control_hz', 0):.0f} Hz | "
                      f"IK valid: {retarget_out.valid}")

                # Latency breakdown
                lat = shared.get_loop_durations()
                zed_grab = lat.get('lat_zed_grab_loop_s', 0) * 1000
                zed_retr = lat.get('lat_zed_retrieve_loop_s', 0) * 1000
                disp     = lat.get('lat_display_loop_s', 0) * 1000
                trk_ext  = lat.get('lat_tracking_extract_loop_s', 0) * 1000
                trk_tot  = lat.get('lat_tracking_total_loop_s', 0) * 1000
                data_age = lat.get('lat_tracking_data_age_loop_s', 0) * 1000
                ik_solve = lat.get('lat_ik_solve_loop_s', 0) * 1000
                rt_tot   = lat.get('lat_retarget_total_loop_s', 0) * 1000
                rt_age   = lat.get('lat_retarget_output_age_loop_s', 0) * 1000
                udp_rtt  = lat.get('lat_udp_feedback_rtt_loop_s', 0) * 1000
                e2e      = lat.get('lat_total_capture_to_cmd_loop_s', 0) * 1000
                if trk_tot > 0 or data_age > 0 or ik_solve > 0:
                    print(f"[Latency] ZED grab: {zed_grab:.1f}ms | "
                          f"body detect: {zed_retr:.1f}ms | "
                          f"display: {disp:.1f}ms | "
                          f"extract: {trk_ext:.1f}ms | "
                          f"tracking total: {trk_tot:.1f}ms")
                    print(f"          data age\u2192retarget: {data_age:.1f}ms | "
                          f"IK solve: {ik_solve:.1f}ms | "
                          f"retarget loop: {rt_tot:.1f}ms | "
                          f"output age\u2192hw: {rt_age:.1f}ms")
                    print(f"          UDP RTT: {udp_rtt:.1f}ms | "
                          f"END-TO-END (capture\u2192command): {e2e:.1f}ms")

                if hw_fb.valid and retarget_out.valid:
                    # Show tracking error (desired vs actual)
                    q_kin = retarget_out.q_des
                    joint_mapping = JointMapping(config.joint_mapping)
                    q_hw_des = joint_mapping.reverse_q(q_kin)
                    err_r = np.linalg.norm(hw_fb.right_arm_q - q_hw_des[12:19])
                    err_l = np.linalg.norm(hw_fb.left_arm_q  - q_hw_des[19:26])
                    print(f"         Arm tracking error:  R={np.degrees(err_r):.1f}°  L={np.degrees(err_l):.1f}°")
                    print(f"         R_arm cmd: {np.degrees(q_hw_des[12:19]).round(1)}")
                    print(f"         R_arm fb:  {np.degrees(hw_fb.right_arm_q).round(1)}")

    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback; traceback.print_exc()

    finally:
        # ── Graceful shutdown ────────────────────────────────────────
        print("\n[Main] Shutting down …")
        shared.request_shutdown()

        tracker.stop()
        retargeter.stop()
        hw_sender.stop()

        # Return arms to IDLE before disconnecting
        if not args.dry_run:
            print("[Main] Returning arms to IDLE pose …")
            RAMP_STEPS = 200   # ~2 s at 100 Hz
            cur_r = hw_sender.last_cmd_r.copy()
            cur_l = hw_sender.last_cmd_l.copy()
            for i in range(RAMP_STEPS):
                alpha = (i + 1) / RAMP_STEPS
                q_r = (1.0 - alpha) * cur_r + alpha * IDLE_R_HW
                q_l = (1.0 - alpha) * cur_l + alpha * IDLE_L_HW
                if hw_sender.use_manip_ref:
                    client.send_manip_reference(
                        right_arm_pose=q_r, left_arm_pose=q_l,
                        right_mode=100.0, right_phase=1.0,
                        left_mode=100.0,  left_phase=1.0)
                else:
                    client.send_arm_command('right', q=q_r)
                    client.send_arm_command('left',  q=q_l)
                time.sleep(0.01)
            print("[Main] Arms returned to IDLE.")

        client.disconnect()
        print("[Main] Pipeline stopped.")


if __name__ == "__main__":
    main()
