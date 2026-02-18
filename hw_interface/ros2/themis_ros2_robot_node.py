#!/usr/bin/env python3
"""
THEMIS Robot-Side ROS2 Node  (runs on the ROBOT main PC)

Uses the OFFICIAL Westwood Robotics WBC API to safely read/write joint
states WITHOUT bypassing the robot's control loops.

This replaces the shared-memory approach (shm_udp_server.py) which was
directly writing to POSIX shared memory and breaking the WBC pipeline,
causing the robot to collapse.

The official API path documented in the THEMIS Developer Manual:
  • wbc_api.get_joint_states(chain)   → safe read  of q, dq, u, temp, volt
  • wbc_api.set_joint_states(chain, …) → safe write through WBC pipeline
  • wbc_api.get_imu_states()           → IMU data

Published topics (from robot → desktop):
  /themis/joint_state/right_arm   sensor_msgs/JointState  @ 100 Hz
  /themis/joint_state/left_arm    sensor_msgs/JointState  @ 100 Hz
  /themis/imu                     sensor_msgs/Imu         @ 100 Hz

Subscribed topics (from desktop → robot):
  /themis/arm_cmd/right           sensor_msgs/JointState  (cmd)
  /themis/arm_cmd/left            sensor_msgs/JointState  (cmd)

Usage (on robot PC):
  # Make sure AOS is booted:
  cd /home/themis/THEMIS/THEMIS
  bash Play/bootup.sh

  # Then run this node:
  cd /home/themis/THEMIS/THEMIS
  python3 /path/to/themis_ros2_robot_node.py

  # Or if you want to specify the AOS path:
  AOS_PATH=/home/themis/THEMIS/THEMIS python3 themis_ros2_robot_node.py

ROS2 DDS Setup (for cross-machine communication):
  Both machines must be on the same ROS_DOMAIN_ID (default 0) and
  reachable on the same subnet (192.168.0.0/24).
  Set on both machines:
    export ROS_DOMAIN_ID=0
"""

import os
import sys
import time
import numpy as np

# ── Make AOS importable ──────────────────────────────────────────────
AOS_PATH = os.environ.get("AOS_PATH", "/home/themis/THEMIS/THEMIS")
if AOS_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(AOS_PATH))
    sys.path.insert(0, AOS_PATH)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion

# ── Import the OFFICIAL Westwood WBC API ─────────────────────────────
# As documented in THEMIS Developer Manual §3.7 (Whole-Body Control API)
from Play.Others import wbc as wbc_api

# Robot model API for joint state conversions (motor ↔ joint)
from Library.ROBOT_MODEL import model as rm_api


# ── Chain indices (from the documentation) ───────────────────────────
CHAIN_HEAD      =  0
CHAIN_RIGHT_LEG =  1   # +1
CHAIN_LEFT_LEG  = -1
CHAIN_RIGHT_ARM =  2   # +2
CHAIN_LEFT_ARM  = -2
CHAIN_RIGHT_HAND =  3  # +3
CHAIN_LEFT_HAND  = -3

# ── Arm joint names ─────────────────────────────────────────────────
ARM_JOINT_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw",
]


class ThemisROS2RobotNode(Node):
    """
    Robot-side ROS2 node.

    Publishes joint state feedback using the OFFICIAL wbc_api,
    and subscribes to arm commands, applying them through the same
    safe WBC interface.
    """

    def __init__(self):
        super().__init__('themis_robot_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('enable_arm_control', True)
        self.declare_parameter('kp_default', [10.0] * 7)
        self.declare_parameter('kd_default', [1.0] * 7)

        publish_rate = self.get_parameter('publish_rate').value
        self.enable_arm_control = self.get_parameter('enable_arm_control').value
        self.kp_default = np.array(
            self.get_parameter('kp_default').value, dtype=np.float64)
        self.kd_default = np.array(
            self.get_parameter('kd_default').value, dtype=np.float64)

        # ── QoS for low-latency real-time feedback ───────────────────
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers: Joint state feedback ─────────────────────────
        self.pub_right_arm = self.create_publisher(
            JointState, '/themis/joint_state/right_arm', qos_fast)
        self.pub_left_arm = self.create_publisher(
            JointState, '/themis/joint_state/left_arm', qos_fast)
        self.pub_right_leg = self.create_publisher(
            JointState, '/themis/joint_state/right_leg', qos_fast)
        self.pub_left_leg = self.create_publisher(
            JointState, '/themis/joint_state/left_leg', qos_fast)
        self.pub_head = self.create_publisher(
            JointState, '/themis/joint_state/head', qos_fast)
        self.pub_imu = self.create_publisher(
            Imu, '/themis/imu', qos_fast)

        # ── Subscribers: Arm joint commands from desktop ─────────────
        self.sub_right_arm_cmd = self.create_subscription(
            JointState, '/themis/arm_cmd/right',
            self._right_arm_cmd_cb, qos_fast)
        self.sub_left_arm_cmd = self.create_subscription(
            JointState, '/themis/arm_cmd/left',
            self._left_arm_cmd_cb, qos_fast)

        # ── Latest received commands ─────────────────────────────────
        self._last_right_cmd = None  # (q, dq, u, kp, kd)
        self._last_left_cmd  = None
        self._last_right_cmd_time = 0.0
        self._last_left_cmd_time  = 0.0
        self._cmd_timeout = 0.5  # seconds — stop sending if no cmd

        # ── Timer for publishing state at fixed rate ─────────────────
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f"THEMIS Robot Node started — publishing at {publish_rate:.0f} Hz")
        self.get_logger().info(
            f"  Arm control: {'ENABLED' if self.enable_arm_control else 'DISABLED (read-only)'}")

    # ─────────────────────────────────────────────────────────────────
    # Timer callback — runs at publish_rate Hz
    # ─────────────────────────────────────────────────────────────────
    def _timer_cb(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        # ── 1. Read & publish joint states (official API) ────────────
        self._publish_arm_state(CHAIN_RIGHT_ARM, self.pub_right_arm,
                                'right_arm', stamp)
        self._publish_arm_state(CHAIN_LEFT_ARM, self.pub_left_arm,
                                'left_arm', stamp)
        self._publish_leg_state(CHAIN_RIGHT_LEG, self.pub_right_leg,
                                'right_leg', stamp)
        self._publish_leg_state(CHAIN_LEFT_LEG, self.pub_left_leg,
                                'left_leg', stamp)
        self._publish_head_state(stamp)
        self._publish_imu(stamp)

        # ── 2. Apply arm commands if enabled and fresh ───────────────
        if self.enable_arm_control:
            self._apply_arm_commands()

    # ─────────────────────────────────────────────────────────────────
    # State publishing (using official wbc_api.get_joint_states)
    # ─────────────────────────────────────────────────────────────────
    def _publish_arm_state(self, chain, publisher, prefix, stamp):
        """Read arm joint states via official API and publish."""
        try:
            # Official API: returns (q, dq, u)
            # Note: temp and volt are NOT returned by wbc_api
            q, dq, u = wbc_api.get_joint_states(chain)
        except Exception as e:
            self.get_logger().warn(
                f"Failed to read {prefix} joint state: {e}", throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = prefix
        msg.name = [f"{prefix}_{jn}" for jn in ARM_JOINT_NAMES]
        msg.position = q.tolist()
        msg.velocity = dq.tolist()
        msg.effort = u.tolist()
        publisher.publish(msg)

    def _publish_leg_state(self, chain, publisher, prefix, stamp):
        """Read leg joint states via official API and publish."""
        try:
            q, dq, u = wbc_api.get_joint_states(chain)
        except Exception as e:
            self.get_logger().warn(
                f"Failed to read {prefix} joint state: {e}", throttle_duration_sec=2.0)
            return

        leg_names = [f"{prefix}_j{i}" for i in range(len(q))]
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = prefix
        msg.name = leg_names
        msg.position = q.tolist()
        msg.velocity = dq.tolist()
        msg.effort = u.tolist()
        publisher.publish(msg)

    def _publish_head_state(self, stamp):
        """Read head joint states via official API and publish."""
        try:
            q, dq, u = wbc_api.get_joint_states(CHAIN_HEAD)
        except Exception as e:
            self.get_logger().warn(
                f"Failed to read head joint state: {e}", throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = 'head'
        msg.name = ['head_pitch', 'head_yaw']
        msg.position = q.tolist()
        msg.velocity = dq.tolist()
        msg.effort = u.tolist()
        self.pub_head.publish(msg)

    def _publish_imu(self, stamp):
        """Read IMU data via official API and publish."""
        try:
            accel, angular_vel, rot_matrix = wbc_api.get_imu_states()
        except Exception as e:
            self.get_logger().warn(
                f"Failed to read IMU: {e}", throttle_duration_sec=2.0)
            return

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'

        # Linear acceleration
        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])

        # Angular velocity
        msg.angular_velocity.x = float(angular_vel[0])
        msg.angular_velocity.y = float(angular_vel[1])
        msg.angular_velocity.z = float(angular_vel[2])

        # Convert rotation matrix to quaternion
        quat = self._rot_to_quat(rot_matrix)
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]

        self.pub_imu.publish(msg)

    @staticmethod
    def _rot_to_quat(R):
        """Convert 3×3 rotation matrix to quaternion [x, y, z, w]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    # ─────────────────────────────────────────────────────────────────
    # Command callbacks (from desktop → robot)
    # ─────────────────────────────────────────────────────────────────
    def _right_arm_cmd_cb(self, msg: JointState):
        """Receive right arm command from desktop."""
        q  = np.array(msg.position, dtype=np.float64)
        dq = np.array(msg.velocity, dtype=np.float64) if msg.velocity else np.zeros(7)
        u  = np.array(msg.effort, dtype=np.float64) if msg.effort else np.zeros(7)

        # kp/kd are packed into the JointState name field as
        # "kp=10.0,10.0,...;kd=1.0,1.0,..." if provided, else use defaults
        kp, kd = self._parse_gains(msg.name)

        self._last_right_cmd = (q, dq, u, kp, kd)
        self._last_right_cmd_time = time.time()

    def _left_arm_cmd_cb(self, msg: JointState):
        """Receive left arm command from desktop."""
        q  = np.array(msg.position, dtype=np.float64)
        dq = np.array(msg.velocity, dtype=np.float64) if msg.velocity else np.zeros(7)
        u  = np.array(msg.effort, dtype=np.float64) if msg.effort else np.zeros(7)

        kp, kd = self._parse_gains(msg.name)

        self._last_left_cmd = (q, dq, u, kp, kd)
        self._last_left_cmd_time = time.time()

    def _parse_gains(self, names):
        """Parse kp/kd from JointState name field, or return defaults."""
        kp = self.kp_default.copy()
        kd = self.kd_default.copy()
        if names and len(names) > 0:
            for n in names:
                if n.startswith('kp='):
                    try:
                        kp = np.array([float(x) for x in n[3:].split(',')],
                                      dtype=np.float64)
                    except ValueError:
                        pass
                elif n.startswith('kd='):
                    try:
                        kd = np.array([float(x) for x in n[3:].split(',')],
                                      dtype=np.float64)
                    except ValueError:
                        pass
        return kp, kd

    # ─────────────────────────────────────────────────────────────────
    # Apply received commands via official wbc_api.set_joint_states
    # ─────────────────────────────────────────────────────────────────
    def _apply_arm_commands(self):
        """
        Apply the latest arm commands using the OFFICIAL API:

            wbc_api.set_joint_states(chain, u, q, dq, kp, kd)

        This goes through the whole-body controller, so it cooperates
        with the robot's balance controller instead of fighting it.
        """
        now = time.time()

        # Right arm
        if (self._last_right_cmd is not None and
                now - self._last_right_cmd_time < self._cmd_timeout):
            q, dq, u, kp, kd = self._last_right_cmd
            try:
                wbc_api.set_joint_states(CHAIN_RIGHT_ARM, u, q, dq, kp, kd)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to set right arm: {e}", throttle_duration_sec=2.0)

        # Left arm
        if (self._last_left_cmd is not None and
                now - self._last_left_cmd_time < self._cmd_timeout):
            q, dq, u, kp, kd = self._last_left_cmd
            try:
                wbc_api.set_joint_states(CHAIN_LEFT_ARM, u, q, dq, kp, kd)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to set left arm: {e}", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = ThemisROS2RobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down …")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
