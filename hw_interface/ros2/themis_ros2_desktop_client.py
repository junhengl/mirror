#!/usr/bin/env python3
"""
THEMIS Desktop-Side ROS2 Client  (runs on the DESKTOP / development PC)

A clean Python API that wraps ROS2 publishers and subscribers so that
callers can:
  • Read the latest robot joint state (from the robot node's topics)
  • Send arm joint commands (published to the robot node's subscribers)

This replaces ThemisUDPClient (themis_udp_client.py) and the UDP bridge
entirely.  Communication is handled by ROS2 DDS, which means:
  - No custom UDP protocol
  - No shared-memory access on the robot
  - Works across subnets with proper DDS discovery

Prerequisites:
  • ROS2 Humble installed on the desktop  (apt install ros-humble-desktop)
  • export ROS_DOMAIN_ID=0   (same as robot PC)
  • Both PCs on same subnet (192.168.0.0/24 recommended)

Usage:
    from hw_interface.ros2.themis_ros2_desktop_client import ThemisROS2Client

    client = ThemisROS2Client()
    client.wait_for_state(timeout=5.0)

    # Read feedback
    fb = client.get_state()
    print(fb.right_arm_q)

    # Send command
    client.send_arm_command('right', q=desired_q)

    # Cleanup
    client.shutdown()
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu


# ── Feedback dataclass (same fields as the old ThemisStateFeedback) ──
@dataclass
class ThemisStateFeedback:
    """Parsed robot state from ROS2 topics."""
    timestamp: float = 0.0
    valid: bool = False

    # Arm joints (7 DOF each)
    right_arm_q:      np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    right_arm_dq:     np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    right_arm_torque: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_q:       np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_dq:      np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    left_arm_torque:  np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    # Leg joints (6 DOF each)
    right_leg_q:      np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_leg_dq:     np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_leg_torque: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_leg_q:       np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_leg_dq:      np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_leg_torque:  np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    # Head joints (2 DOF)
    head_q:           np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    head_dq:          np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    head_torque:      np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    # IMU
    imu_accel:        np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    imu_gyro:         np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    imu_orientation:  np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.], dtype=np.float64))


ARM_JOINT_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw",
]


class _ThemisListenerNode(Node):
    """
    Internal ROS2 node that subscribes to robot state topics and
    publishes arm commands.
    """

    def __init__(self):
        super().__init__('themis_desktop_client')

        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────
        self._lock = threading.Lock()
        self._state = ThemisStateFeedback()
        self._got_right_arm = False
        self._got_left_arm  = False

        self.create_subscription(
            JointState, '/themis/joint_state/right_arm',
            self._right_arm_cb, qos_fast)
        self.create_subscription(
            JointState, '/themis/joint_state/left_arm',
            self._left_arm_cb, qos_fast)
        self.create_subscription(
            JointState, '/themis/joint_state/right_leg',
            self._right_leg_cb, qos_fast)
        self.create_subscription(
            JointState, '/themis/joint_state/left_leg',
            self._left_leg_cb, qos_fast)
        self.create_subscription(
            JointState, '/themis/joint_state/head',
            self._head_cb, qos_fast)
        self.create_subscription(
            Imu, '/themis/imu',
            self._imu_cb, qos_fast)

        # ── Publishers ───────────────────────────────────────────────
        self.pub_right_arm_cmd = self.create_publisher(
            JointState, '/themis/arm_cmd/right', qos_fast)
        self.pub_left_arm_cmd = self.create_publisher(
            JointState, '/themis/arm_cmd/left', qos_fast)

    # ── Subscriber callbacks ─────────────────────────────────────────
    def _right_arm_cb(self, msg: JointState):
        with self._lock:
            self._state.right_arm_q      = np.array(msg.position, dtype=np.float64)
            self._state.right_arm_dq     = np.array(msg.velocity, dtype=np.float64)
            self._state.right_arm_torque = np.array(msg.effort, dtype=np.float64)
            self._state.timestamp = time.time()
            self._got_right_arm = True
            if self._got_left_arm:
                self._state.valid = True

    def _left_arm_cb(self, msg: JointState):
        with self._lock:
            self._state.left_arm_q      = np.array(msg.position, dtype=np.float64)
            self._state.left_arm_dq     = np.array(msg.velocity, dtype=np.float64)
            self._state.left_arm_torque = np.array(msg.effort, dtype=np.float64)
            self._state.timestamp = time.time()
            self._got_left_arm = True
            if self._got_right_arm:
                self._state.valid = True

    def _right_leg_cb(self, msg: JointState):
        with self._lock:
            self._state.right_leg_q      = np.array(msg.position, dtype=np.float64)
            self._state.right_leg_dq     = np.array(msg.velocity, dtype=np.float64)
            self._state.right_leg_torque = np.array(msg.effort, dtype=np.float64)

    def _left_leg_cb(self, msg: JointState):
        with self._lock:
            self._state.left_leg_q      = np.array(msg.position, dtype=np.float64)
            self._state.left_leg_dq     = np.array(msg.velocity, dtype=np.float64)
            self._state.left_leg_torque = np.array(msg.effort, dtype=np.float64)

    def _head_cb(self, msg: JointState):
        with self._lock:
            self._state.head_q      = np.array(msg.position, dtype=np.float64)
            self._state.head_dq     = np.array(msg.velocity, dtype=np.float64)
            self._state.head_torque = np.array(msg.effort, dtype=np.float64)

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._state.imu_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ], dtype=np.float64)
            self._state.imu_gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ], dtype=np.float64)
            self._state.imu_orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ], dtype=np.float64)

    def get_state_snapshot(self) -> ThemisStateFeedback:
        """Thread-safe copy of the latest state."""
        with self._lock:
            fb = ThemisStateFeedback(
                timestamp       = self._state.timestamp,
                valid           = self._state.valid,
                right_arm_q     = self._state.right_arm_q.copy(),
                right_arm_dq    = self._state.right_arm_dq.copy(),
                right_arm_torque= self._state.right_arm_torque.copy(),
                left_arm_q      = self._state.left_arm_q.copy(),
                left_arm_dq     = self._state.left_arm_dq.copy(),
                left_arm_torque = self._state.left_arm_torque.copy(),
                right_leg_q     = self._state.right_leg_q.copy(),
                right_leg_dq    = self._state.right_leg_dq.copy(),
                right_leg_torque= self._state.right_leg_torque.copy(),
                left_leg_q      = self._state.left_leg_q.copy(),
                left_leg_dq     = self._state.left_leg_dq.copy(),
                left_leg_torque = self._state.left_leg_torque.copy(),
                head_q          = self._state.head_q.copy(),
                head_dq         = self._state.head_dq.copy(),
                head_torque     = self._state.head_torque.copy(),
                imu_accel       = self._state.imu_accel.copy(),
                imu_gyro        = self._state.imu_gyro.copy(),
                imu_orientation = self._state.imu_orientation.copy(),
            )
        return fb

    def publish_arm_cmd(self, side: str,
                        q: np.ndarray,
                        dq: Optional[np.ndarray] = None,
                        u: Optional[np.ndarray] = None,
                        kp: Optional[np.ndarray] = None,
                        kd: Optional[np.ndarray] = None):
        """Publish arm command to the robot node."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = q.tolist()
        msg.velocity = (dq if dq is not None else np.zeros(7)).tolist()
        msg.effort   = (u  if u  is not None else np.zeros(7)).tolist()

        # Pack gains into name field
        names = []
        if kp is not None:
            names.append('kp=' + ','.join(f'{v:.2f}' for v in kp))
        if kd is not None:
            names.append('kd=' + ','.join(f'{v:.2f}' for v in kd))
        msg.name = names

        if side == 'right':
            self.pub_right_arm_cmd.publish(msg)
        elif side == 'left':
            self.pub_left_arm_cmd.publish(msg)


class ThemisROS2Client:
    """
    High-level desktop-side client.

    Drop-in replacement for ThemisUDPClient — same get_state() /
    send_arm_command() API, but uses ROS2 instead of raw UDP.
    """

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    DEFAULT_KP = np.full(7, 10.0)
    DEFAULT_KD = np.full(7,  1.0)

    # Nominal arm poses (hardware convention)
    IDLE_POSE_R = np.array([-0.20, +1.40, +1.57, +0.40,  0.00,  0.00, -1.50])
    IDLE_POSE_L = np.array([-0.20, -1.40, -1.57, -0.40,  0.00,  0.00, +1.50])

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self._node = _ThemisListenerNode()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._running = False

    def connect(self):
        """Start the ROS2 spin thread (analogous to opening the UDP socket)."""
        self._running = True
        self._spin_thread.start()
        print("[ThemisROS2Client] Connected — listening on /themis/* topics")

    def _spin(self):
        """Background thread spinning the ROS2 node."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)

    def disconnect(self):
        """Stop the spin thread."""
        self._running = False
        self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
        print("[ThemisROS2Client] Disconnected")

    def shutdown(self):
        """Full cleanup."""
        self.disconnect()
        if rclpy.ok():
            rclpy.shutdown()

    # ─────────────────────────────────────────────────────────────────
    # State reading
    # ─────────────────────────────────────────────────────────────────
    def get_state(self) -> ThemisStateFeedback:
        """Return the latest robot state (non-blocking)."""
        return self._node.get_state_snapshot()

    def wait_for_state(self, timeout: float = 5.0) -> ThemisStateFeedback:
        """Block until valid state is received or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            fb = self.get_state()
            if fb.valid:
                return fb
            time.sleep(0.05)
        print("[ThemisROS2Client] WARNING: Timed out waiting for robot state")
        return self.get_state()

    # ─────────────────────────────────────────────────────────────────
    # Arm commands
    # ─────────────────────────────────────────────────────────────────
    def send_arm_command(self, side: str,
                         q: np.ndarray,
                         dq: Optional[np.ndarray] = None,
                         u: Optional[np.ndarray] = None,
                         kp: Optional[np.ndarray] = None,
                         kd: Optional[np.ndarray] = None) -> bool:
        """
        Send arm joint command to the robot.

        Parameters
        ----------
        side : 'right' or 'left'
        q    : (7,) goal joint positions [rad]
        dq   : (7,) goal joint velocities [rad/s], default zeros
        u    : (7,) feedforward torques [Nm], default zeros
        kp   : (7,) proportional gains
        kd   : (7,) derivative gains
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        self._node.publish_arm_cmd(side, q, dq, u, kp, kd)
        return True

    # ─────────────────────────────────────────────────────────────────
    # Pretty-print
    # ─────────────────────────────────────────────────────────────────
    def print_state(self, fb: Optional[ThemisStateFeedback] = None):
        """Pretty-print the current robot state."""
        if fb is None:
            fb = self.get_state()
        if not fb.valid:
            print("[ThemisROS2Client] No valid state feedback yet")
            return

        print(f"\n{'='*72}")
        print(f"  Themis Robot State  (t = {fb.timestamp:.3f})")
        print(f"{'='*72}")

        print(f"\n  IMU accel: [{fb.imu_accel[0]:+.3f}, {fb.imu_accel[1]:+.3f}, {fb.imu_accel[2]:+.3f}]")
        print(f"  IMU gyro:  [{fb.imu_gyro[0]:+.3f}, {fb.imu_gyro[1]:+.3f}, {fb.imu_gyro[2]:+.3f}]")

        for label, q, dq, tau in [
            ("Right Arm", fb.right_arm_q, fb.right_arm_dq, fb.right_arm_torque),
            ("Left Arm",  fb.left_arm_q,  fb.left_arm_dq,  fb.left_arm_torque),
        ]:
            print(f"\n  {label}:")
            print(f"    {'Joint':<18s} {'Pos [deg]':>10s} {'Vel [rad/s]':>12s} {'Torque [Nm]':>12s}")
            print(f"    {'─'*18} {'─'*10} {'─'*12} {'─'*12}")
            for j in range(7):
                print(f"    {ARM_JOINT_NAMES[j]:<18s} {np.degrees(q[j]):>+10.2f} {dq[j]:>+12.3f} {tau[j]:>+12.3f}")

        for label, q, dq, tau in [
            ("Right Leg", fb.right_leg_q, fb.right_leg_dq, fb.right_leg_torque),
            ("Left Leg",  fb.left_leg_q,  fb.left_leg_dq,  fb.left_leg_torque),
        ]:
            if np.any(q != 0):
                print(f"\n  {label}:")
                for j in range(len(q)):
                    print(f"    joint_{j}: pos={np.degrees(q[j]):+.2f}° vel={dq[j]:+.3f} τ={tau[j]:+.3f}")

        if np.any(fb.head_q != 0):
            print(f"\n  Head:")
            for j, name in enumerate(['pitch', 'yaw']):
                print(f"    {name}: pos={np.degrees(fb.head_q[j]):+.2f}° vel={fb.head_dq[j]:+.3f} τ={fb.head_torque[j]:+.3f}")

        print()
