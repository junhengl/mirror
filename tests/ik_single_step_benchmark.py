#!/usr/bin/env python3
"""
IK Solver Benchmark — Single-Step Near-Target Comparison
=========================================================

Compares all IK solver methods on **single-step** performance with
targets that are close to the initial configuration, simulating the
real-time retargeting scenario where the robot tracks a smoothly
moving target at 500 Hz.

Baseline:
  (0) Multi-step SQP (backtracking line search) — converged solution
      Runs N outer iterations to provide the "best achievable" reference.

Single-step methods:
  (1) Single QP (ProxQP, 34-DOF)
  (2) Distributed QP (ProxQP, 2×13-DOF)
  (3) Batched QP (GPU ADMM, 34-DOF, N parallel)
  (4) Batched Distributed QP (GPU ADMM, 2×13-DOF, N parallel)
  (5) Batched α-Continuation (GPU ADMM, max-feasible-α)

Metrics:
  - Position tracking error per end-effector (mm)
  - Solve time (ms)
  - CBF / joint-limit violations
  - Gap to SQP baseline (how close a single step gets to the converged solution)
  - Aggregate statistics (mean, std, percentiles)

Test design:
  - N_CONFIGS random initial configurations
  - N_TARGETS near targets per configuration (small joint perturbation)
  - Single-step methods: 1 IK iteration only
  - SQP baseline: multi-step until convergence (default 50 outer iters)
  - Warm-start disabled between trials for fair comparison

Usage:
    cd /home/junhengl/body_tracking_arm_mod
    python -m tests.ik_single_step_benchmark [--n-configs 50] [--n-targets 10]
"""

import sys
import os
import io
import time
import argparse
import contextlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'KinDynLib_single'))

import robot_const as rc
from robot_dynamics import Robot
from dynamics_lib import Xtrans

DOF = rc.DOF
q_min = rc.q_min
q_max = rc.q_max


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Result from a single (config, target) IK trial."""
    hand_l_error: float = 0.0       # L2 position error (m)
    hand_r_error: float = 0.0
    elbow_l_error: float = 0.0
    elbow_r_error: float = 0.0
    total_error: float = 0.0        # sum of 4 EE errors (m)
    solve_time_ms: float = 0.0      # Wall-clock time (ms)
    cbf_violation_count: int = 0
    cbf_violation_max: float = 0.0
    joint_limit_violations: int = 0
    initial_error: float = 0.0      # error BEFORE IK (for reduction %)
    error_reduction: float = 0.0    # 1 - final/initial


@dataclass
class MethodStats:
    """Aggregated statistics for one solver method."""
    name: str = ""
    trials: List[TrialResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trials)

    def mean(self, attr: str) -> float:
        vals = [getattr(t, attr) for t in self.trials]
        return float(np.mean(vals)) if vals else 0.0

    def std(self, attr: str) -> float:
        vals = [getattr(t, attr) for t in self.trials]
        return float(np.std(vals)) if vals else 0.0

    def percentile(self, attr: str, p: float) -> float:
        vals = [getattr(t, attr) for t in self.trials]
        return float(np.percentile(vals, p)) if vals else 0.0

    def total_cbf_violations(self) -> int:
        return sum(t.cbf_violation_count for t in self.trials)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (shared with ik_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Link indices and offsets
HAND_R_LINK = 23
HAND_L_LINK = 30
ELBOW_R_LINK = 21
ELBOW_L_LINK = 28

HAND_R_OFFSET = np.array([0.08, 0.0, 0.0], dtype=np.float64)
HAND_L_OFFSET = np.array([0.08, 0.0, 0.0], dtype=np.float64)
ELBOW_R_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ELBOW_L_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

XHAND_R = Xtrans(HAND_R_OFFSET)
XHAND_L = Xtrans(HAND_L_OFFSET)
XELBOW_R = Xtrans(ELBOW_R_OFFSET)
XELBOW_L = Xtrans(ELBOW_L_OFFSET)

# CBF parameters
COM_OFFSET = np.array([-0.1, 0.0, 0.0], dtype=np.float64)
HEAD_OFFSET = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
CROTCH_OFFSET = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
R_TORSO, R_HEAD, R_CROTCH = 0.13, 0.11, 0.16
SAFETY_MARGIN = 0.02


def compute_fk_and_jacobians(robot: Robot):
    """Compute FK positions and Jacobians for all tracked links."""
    x_hand_r = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
    x_hand_l = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
    x_elbow_r = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)
    x_elbow_l = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)

    J_hand_r = robot.compute_body_jacobian(HAND_R_LINK, XHAND_R)
    J_hand_l = robot.compute_body_jacobian(HAND_L_LINK, XHAND_L)
    J_elbow_r = robot.compute_body_jacobian(ELBOW_R_LINK, XELBOW_R)
    J_elbow_l = robot.compute_body_jacobian(ELBOW_L_LINK, XELBOW_L)

    return (x_hand_l, x_hand_r, x_elbow_l, x_elbow_r,
            J_hand_l, J_hand_r, J_elbow_l, J_elbow_r)


def _ee_positions_cbf_safe(base_pos: np.ndarray,
                           ee_rpypos_list: List[np.ndarray]) -> bool:
    """Return True if ALL EE positions are outside every CBF sphere."""
    com_center = base_pos.astype(np.float64) + COM_OFFSET
    head_center = base_pos.astype(np.float64) + HEAD_OFFSET
    crotch_center = base_pos.astype(np.float64) + CROTCH_OFFSET

    spheres = [
        (com_center, R_TORSO + SAFETY_MARGIN),
        (head_center, R_HEAD + SAFETY_MARGIN),
        (crotch_center, R_CROTCH + SAFETY_MARGIN),
    ]
    for x_rpypos in ee_rpypos_list:
        pos = x_rpypos[3:].astype(np.float64)
        for center, radius in spheres:
            if np.sum((pos - center) ** 2) < radius ** 2:
                return False
    return True


def generate_random_config(rng: np.random.Generator,
                           robot: Optional[Robot] = None,
                           max_retries: int = 50) -> np.ndarray:
    """Generate a random feasible robot configuration."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    for _ in range(max_retries):
        q = np.zeros(DOF, dtype=np.float32)
        q[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        q[3:6] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(6, DOF):
            lo = max(q_min[i], -2.5)
            hi = min(q_max[i], 2.5)
            q[i] = rng.uniform(lo, hi)

        if robot is None:
            return q

        robot.update(q, dq_zero)
        x_hl = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
        x_hr = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
        x_el = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)
        x_er = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)
        if _ee_positions_cbf_safe(q[:3], [x_hl, x_hr, x_el, x_er]):
            return q
    return q


def generate_near_target(robot: Robot, q_init: np.ndarray,
                         rng: np.random.Generator,
                         perturbation_range: float = 0.05,
                         max_retries: int = 50
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a NEAR task-space target by small joint perturbation.

    Simulates the real-time scenario: the target at the next control step
    is very close to the current configuration, as the human moves smoothly.

    Args:
        perturbation_range: max per-joint perturbation (rad). Default 0.05
            corresponds to ~3° per joint — roughly what a 500 Hz loop sees
            for moderate human arm speeds.
    """
    dq_zero = np.zeros(DOF, dtype=np.float32)
    base_pos = q_init[:3]

    for _ in range(max_retries):
        q_target = q_init.copy()
        # Perturb arm joints only (indices 18-31 for right/left arms)
        for i in range(18, 32):
            lo = max(q_min[i], q_init[i] - perturbation_range)
            hi = min(q_max[i], q_init[i] + perturbation_range)
            q_target[i] = rng.uniform(lo, hi)

        robot.update(q_target, dq_zero)
        x_hand_l_des = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
        x_hand_r_des = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
        x_elbow_l_des = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)
        x_elbow_r_des = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)

        if _ee_positions_cbf_safe(base_pos,
                                  [x_hand_l_des, x_hand_r_des,
                                   x_elbow_l_des, x_elbow_r_des]):
            return x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des

    return x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des


def check_cbf_violations(q: np.ndarray, robot: Robot) -> Tuple[int, float]:
    """Check CBF constraint violations after IK solution."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    robot.update(q, dq_zero)

    x_hand_r = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
    x_hand_l = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
    x_elbow_r = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)
    x_elbow_l = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)

    base_pos = q[:3].astype(np.float64)
    com_center = base_pos + COM_OFFSET
    head_center = base_pos + HEAD_OFFSET
    crotch_center = base_pos + CROTCH_OFFSET

    ee_positions = [
        x_elbow_l[3:].astype(np.float64),
        x_elbow_r[3:].astype(np.float64),
        x_hand_l[3:].astype(np.float64),
        x_hand_r[3:].astype(np.float64),
    ]
    sphere_centers = [com_center, head_center, crotch_center]
    sphere_radii = [
        R_TORSO + SAFETY_MARGIN,
        R_HEAD + SAFETY_MARGIN,
        R_CROTCH + SAFETY_MARGIN,
    ]

    n_viol = 0
    max_viol = 0.0
    for ee_pos in ee_positions:
        for center, radius in zip(sphere_centers, sphere_radii):
            dist_sq = np.sum((ee_pos - center) ** 2)
            h = dist_sq - radius ** 2
            if h < 0:
                n_viol += 1
                max_viol = max(max_viol, -h)

    return n_viol, max_viol


def check_joint_limit_violations(q: np.ndarray) -> int:
    """Count how many joints violate their limits."""
    count = 0
    for i in range(6, DOF):
        if q[i] < q_min[i] - 1e-4 or q[i] > q_max[i] + 1e-4:
            count += 1
    return count


def compute_tracking_errors(robot: Robot, q_result: np.ndarray,
                            x_hand_l_des, x_hand_r_des,
                            x_elbow_l_des, x_elbow_r_des):
    """Compute position tracking errors (m) for each end effector."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    robot.update(q_result, dq_zero)

    x_hl = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
    x_hr = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
    x_el = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)
    x_er = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)

    return (
        np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]),
        np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]),
        np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]),
        np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]),
    )


def compute_total_error(robot: Robot, q: np.ndarray,
                        x_hl_des, x_hr_des, x_el_des, x_er_des) -> float:
    """Sum of 4 EE position errors (m)."""
    e = compute_tracking_errors(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)
    return sum(e)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver wrappers — single step only
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_qp(robot: Robot, q_init: np.ndarray,
                  x_hl_des, x_hr_des, x_el_des, x_er_des,
                  com_des: np.ndarray) -> TrialResult:
    """Method 1: Single ProxQP (34-DOF)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)

    robot._osqp_solver = None
    robot._osqp_prev_dq = np.zeros(DOF, dtype=np.float64)

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

    t0 = time.perf_counter()
    q_des, dq_des = robot.update_task_space_command_qp(
        x_el_des, x_er_des, x_el, x_er,
        x_hl_des, x_hr_des, x_hl, x_hr,
        J_el, J_er, J_hl, J_hr, com_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q_des, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(q_des, robot)
    n_jl = check_joint_limit_violations(q_des)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


def run_distributed_qp(robot: Robot, q_init: np.ndarray,
                       x_hl_des, x_hr_des, x_el_des, x_er_des,
                       com_des: np.ndarray) -> TrialResult:
    """Method 2: Distributed ProxQP (2×13-DOF)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

    t0 = time.perf_counter()
    q_des, dq_des = robot.update_task_space_command_qp_distributed(
        x_el_des, x_er_des, x_el, x_er,
        x_hl_des, x_hr_des, x_hl, x_hr,
        J_el, J_er, J_hl, J_hr, com_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q_des, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(q_des, robot)
    n_jl = check_joint_limit_violations(q_des)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


def run_batched_qp(robot: Robot, q_init: np.ndarray,
                   x_hl_des, x_hr_des, x_el_des, x_er_des,
                   com_des: np.ndarray, n_batch: int = 128) -> TrialResult:
    """Method 3: Batched GPU QP (34-DOF, N parallel ADMM)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)
    robot._gpu_qp_solver = None

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

    t0 = time.perf_counter()
    q_des, dq_des = robot.update_task_space_command_qp_gpu_batch(
        x_el_des, x_er_des, x_el, x_er,
        x_hl_des, x_hr_des, x_hl, x_hr,
        J_el, J_er, J_hl, J_hr, com_des,
        n_batch=n_batch, max_iter=50, pos_threshold=0.0)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q_des, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(q_des, robot)
    n_jl = check_joint_limit_violations(q_des)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


def run_batched_distributed_qp(robot: Robot, q_init: np.ndarray,
                                x_hl_des, x_hr_des, x_el_des, x_er_des,
                                com_des: np.ndarray, n_batch: int = 128,
                                max_iter: int = 50) -> TrialResult:
    """Method 4: Batched Distributed GPU QP (2×13-DOF fused, N parallel)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)
    robot._gpu_qp_solver_right = None

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

    t0 = time.perf_counter()
    q_des, dq_des = robot.update_task_space_command_qp_gpu_batch_distributed(
        x_el_des, x_er_des, x_el, x_er,
        x_hl_des, x_hr_des, x_hl, x_hr,
        J_el, J_er, J_hl, J_hr, com_des,
        n_batch=n_batch, max_iter=max_iter,
        pos_threshold=0.0)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q_des, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(q_des, robot)
    n_jl = check_joint_limit_violations(q_des)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


def run_batched_distributed_alpha_qp(robot: Robot, q_init: np.ndarray,
                                     x_hl_des, x_hr_des, x_el_des, x_er_des,
                                     com_des: np.ndarray, n_batch: int = 4096,
                                     n_alpha: int = 8) -> TrialResult:
    """Method 5: Max-feasible-α Batched Distributed GPU QP."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)
    robot._gpu_qp_solver_right = None

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

    t0 = time.perf_counter()
    q_des, dq_des = robot.update_task_space_command_qp_gpu_batch_distributed_alpha(
        x_el_des, x_er_des, x_el, x_er,
        x_hl_des, x_hr_des, x_hl, x_hr,
        J_el, J_er, J_hl, J_hr, com_des,
        n_batch=n_batch, max_iter=50,
        pos_threshold=0.0, n_alpha=n_alpha)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q_des, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(q_des, robot)
    n_jl = check_joint_limit_violations(q_des)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SQP baseline (multi-step, converged solution)
# ═══════════════════════════════════════════════════════════════════════════════

def _sqp_tracking_cost(robot: Robot, q: np.ndarray,
                       x_hl_des, x_hr_des, x_el_des, x_er_des) -> float:
    """Sum of squared EE position errors (scalar cost for line search)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    robot.update(q, dq_zero)
    x_hl = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
    x_hr = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
    x_el = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)
    x_er = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)
    return float(
        np.sum((x_hl[3:] - x_hl_des[3:])**2)
        + np.sum((x_hr[3:] - x_hr_des[3:])**2)
        + np.sum((x_el[3:] - x_el_des[3:])**2)
        + np.sum((x_er[3:] - x_er_des[3:])**2))


def run_sqp_baseline(robot: Robot, q_init: np.ndarray,
                     x_hl_des, x_hr_des, x_el_des, x_er_des,
                     com_des: np.ndarray,
                     max_outer_iters: int = 50,
                     convergence_tol: float = 1e-6,
                     ls_alpha_min: float = 0.01,
                     ls_beta: float = 0.5,
                     ls_max_trials: int = 10) -> TrialResult:
    """Method 0: Multi-step SQP baseline (backtracking line search).

    Runs multiple outer iterations to provide the best achievable solution
    as a reference for the single-step methods.
    """
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)

    robot._osqp_solver = None
    robot._osqp_prev_dq = np.zeros(DOF, dtype=np.float64)

    initial_err = compute_total_error(robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)

    best_cost = float('inf')
    best_q = q.copy()

    t0 = time.perf_counter()

    for k in range(max_outer_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        q_des_full, dq_sol = robot.update_task_space_command_qp(
            x_el_des, x_er_des, x_el, x_er,
            x_hl_des, x_hr_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des)

        dq_step = (q_des_full - q).astype(np.float64)
        if np.linalg.norm(dq_step) < convergence_tol:
            break

        cost_current = _sqp_tracking_cost(
            robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)
        if cost_current < best_cost:
            best_cost = cost_current
            best_q = q.copy()

        # Backtracking line search
        alpha = 1.0
        best_alpha = alpha
        best_ls_cost = float('inf')
        for _ in range(ls_max_trials):
            q_trial = q + (alpha * dq_step).astype(np.float32)
            for i in range(6, DOF):
                q_trial[i] = np.clip(q_trial[i], q_min[i], q_max[i])
            cost_trial = _sqp_tracking_cost(
                robot, q_trial, x_hl_des, x_hr_des, x_el_des, x_er_des)
            if cost_trial < best_ls_cost:
                best_ls_cost = cost_trial
                best_alpha = alpha
            if cost_trial < cost_current:
                break
            alpha *= ls_beta
            if alpha < ls_alpha_min:
                alpha = ls_alpha_min
                break

        q = q + (best_alpha * dq_step).astype(np.float32)
        for i in range(6, DOF):
            q[i] = np.clip(q[i], q_min[i], q_max[i])
        robot.update(q, dq_zero)

        if np.linalg.norm(best_alpha * dq_step) < convergence_tol:
            break

    # Final check
    cost_final = _sqp_tracking_cost(
        robot, q, x_hl_des, x_hr_des, x_el_des, x_er_des)
    if cost_final < best_cost:
        best_q = q.copy()

    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, best_q, x_hl_des, x_hr_des, x_el_des, x_er_des)
    final_err = hl_e + hr_e + el_e + er_e
    n_cbf, max_cbf = check_cbf_violations(best_q, robot)
    n_jl = check_joint_limit_violations(best_q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        total_error=final_err,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        initial_error=initial_err,
        error_reduction=1.0 - final_err / initial_err if initial_err > 1e-12 else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_configs: int = 50, n_targets: int = 10,
                  n_batch: int = 4096, target_perturbation: float = 0.05,
                  sqp_iters: int = 50, seed: int = 42):
    """Run single-step IK benchmark with near targets.

    Args:
        n_configs: number of random initial configurations
        n_targets: number of near targets per configuration
        n_batch: GPU batch size
        target_perturbation: per-joint perturbation range (rad) for target
            generation. Default 0.05 rad ≈ 3° — realistic for 500 Hz tracking.
        sqp_iters: max outer iterations for SQP baseline (default 50)
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    robot = Robot()
    com_des = np.zeros(6, dtype=np.float32)

    # ── Warm-up solvers ──────────────────────────────────────────────────────
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q_warmup = generate_random_config(rng, robot)
    robot.update(q_warmup, dq_zero)
    (x_hl, x_hr, x_el, x_er,
     J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
    with contextlib.redirect_stderr(io.StringIO()), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            robot.update_task_space_command_qp(
                x_el, x_er, x_el, x_er, x_hl, x_hr, x_hl, x_hr,
                J_el, J_er, J_hl, J_hr, com_des)
        except Exception:
            pass
        try:
            robot.update_task_space_command_qp_distributed(
                x_el, x_er, x_el, x_er, x_hl, x_hr, x_hl, x_hr,
                J_el, J_er, J_hl, J_hr, com_des)
        except Exception:
            pass
    rng = np.random.default_rng(seed)

    methods = {
        '0_sqp_baseline': MethodStats(name=f'SQP Baseline ({sqp_iters}-step)'),
        '1_single_qp': MethodStats(name='Single QP (ProxQP 34-DOF)'),
        '2_distributed_qp': MethodStats(name='Distributed QP (ProxQP 2×13-DOF)'),
        '3_batched_qp': MethodStats(name=f'Batched QP (GPU {n_batch}×34-DOF)'),
        '4_batched_distributed': MethodStats(name=f'Batched Distributed (GPU {n_batch}×2×13-DOF)'),
        '5a_alpha_B8': MethodStats(name='α-Continuation (B=8)'),
        '5b_alpha_B64': MethodStats(name='α-Continuation (B=64)'),
        '5c_alpha_B512': MethodStats(name='α-Continuation (B=512)'),
        '5d_alpha_B2048': MethodStats(name='α-Continuation (B=2048)'),
    }

    solvers = {
        '0_sqp_baseline': lambda q, hl, hr, el, er: run_sqp_baseline(
            robot, q, hl, hr, el, er, com_des, max_outer_iters=sqp_iters),
        '1_single_qp': lambda q, hl, hr, el, er: run_single_qp(
            robot, q, hl, hr, el, er, com_des),
        '2_distributed_qp': lambda q, hl, hr, el, er: run_distributed_qp(
            robot, q, hl, hr, el, er, com_des),
        '3_batched_qp': lambda q, hl, hr, el, er: run_batched_qp(
            robot, q, hl, hr, el, er, com_des, n_batch),
        '4_batched_distributed': lambda q, hl, hr, el, er: run_batched_distributed_qp(
            robot, q, hl, hr, el, er, com_des, n_batch),
        '5a_alpha_B8': lambda q, hl, hr, el, er: run_batched_distributed_alpha_qp(
            robot, q, hl, hr, el, er, com_des, n_batch=8),
        '5b_alpha_B64': lambda q, hl, hr, el, er: run_batched_distributed_alpha_qp(
            robot, q, hl, hr, el, er, com_des, n_batch=64),
        '5c_alpha_B512': lambda q, hl, hr, el, er: run_batched_distributed_alpha_qp(
            robot, q, hl, hr, el, er, com_des, n_batch=512),
        '5d_alpha_B2048': lambda q, hl, hr, el, er: run_batched_distributed_alpha_qp(
            robot, q, hl, hr, el, er, com_des, n_batch=2048),
    }

    total_trials = n_configs * n_targets
    print("=" * 110)
    print("  SINGLE-STEP IK BENCHMARK  (1 iteration, near targets)")
    print(f"  Baseline: SQP with {sqp_iters} outer iterations (converged solution)")
    print(f"  {n_configs} configs × {n_targets} targets = {total_trials} trials per method")
    print(f"  Target perturbation: {target_perturbation:.3f} rad ({np.degrees(target_perturbation):.1f}°)")
    print(f"  Batch size: {n_batch} | Seed: {seed}")
    print("=" * 110)

    trial_idx = 0
    skipped_methods = set()

    for cfg_i in range(n_configs):
        q_init = generate_random_config(rng, robot)

        for tgt_j in range(n_targets):
            trial_idx += 1

            (x_hl_des, x_hr_des,
             x_el_des, x_er_des) = generate_near_target(
                robot, q_init, rng,
                perturbation_range=target_perturbation)

            for method_key, solver_fn in solvers.items():
                if method_key in skipped_methods:
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = solver_fn(q_init, x_hl_des, x_hr_des,
                                           x_el_des, x_er_des)
                    methods[method_key].trials.append(result)
                except Exception as e:
                    skipped_methods.add(method_key)
                    print(f"  ⚠ Skipping {methods[method_key].name}: {e}")

            if trial_idx % 50 == 0 or trial_idx == total_trials:
                print(f"  [{trial_idx}/{total_trials}] trials complete")

    print_results(methods)
    return methods


def print_results(methods: Dict[str, MethodStats]):
    """Print formatted single-step benchmark results."""
    print(f"\n{'=' * 120}")
    print(f"{'SINGLE-STEP IK RESULTS':^120}")
    print(f"{'=' * 120}")

    # ── Main results table ───────────────────────────────────────────────────
    hdr = "{:<50} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>6}"
    print(hdr.format(
        'Method', 'Hand L', 'Hand R', 'Elbow L', 'Elbow R', 'Total',
        'Time', 'CBF'))
    print(hdr.format(
        '', '(mm)', '(mm)', '(mm)', '(mm)', '(mm)', '(ms)', 'Viol'))
    print("-" * 120)

    row = "{:<50} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.3f} {:>6d}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            print(f"  {m.name:<50} {'(skipped)':>70}")
            continue
        print(row.format(
            m.name,
            m.mean('hand_l_error') * 1000,
            m.mean('hand_r_error') * 1000,
            m.mean('elbow_l_error') * 1000,
            m.mean('elbow_r_error') * 1000,
            m.mean('total_error') * 1000,
            m.mean('solve_time_ms'),
            m.total_cbf_violations()))

    # ── Std dev ──────────────────────────────────────────────────────────────
    print()
    print("Standard deviations:")
    print("-" * 120)
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(row.format(
            f"  ±{m.name[:47]}",
            m.std('hand_l_error') * 1000,
            m.std('hand_r_error') * 1000,
            m.std('elbow_l_error') * 1000,
            m.std('elbow_r_error') * 1000,
            m.std('total_error') * 1000,
            m.std('solve_time_ms'),
            0))

    # ── Percentiles ──────────────────────────────────────────────────────────
    print()
    print("Percentiles (total EE error, mm):")
    pct_hdr = "{:<50} {:>9} {:>9} {:>9} {:>9} {:>9}"
    print(pct_hdr.format('Method', 'P50', 'P90', 'P95', 'P99', 'Max'))
    print("-" * 100)
    pct_row = "{:<50} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(pct_row.format(
            m.name,
            m.percentile('total_error', 50) * 1000,
            m.percentile('total_error', 90) * 1000,
            m.percentile('total_error', 95) * 1000,
            m.percentile('total_error', 99) * 1000,
            m.percentile('total_error', 100) * 1000))

    # ── Error reduction ──────────────────────────────────────────────────────
    print()
    print("Error reduction (initial → final):")
    red_hdr = "{:<50} {:>12} {:>12} {:>12}"
    print(red_hdr.format('Method', 'Mean Red.%', 'Init (mm)', 'Final (mm)'))
    print("-" * 90)
    red_row = "{:<50} {:>11.1f}% {:>12.3f} {:>12.3f}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(red_row.format(
            m.name,
            m.mean('error_reduction') * 100,
            m.mean('initial_error') * 1000,
            m.mean('total_error') * 1000))

    # ── Timing breakdown ─────────────────────────────────────────────────────
    print()
    print("Timing (ms):")
    time_hdr = "{:<50} {:>9} {:>9} {:>9} {:>9} {:>9}"
    print(time_hdr.format('Method', 'Mean', 'Std', 'P50', 'P95', 'Max'))
    print("-" * 100)
    time_row = "{:<50} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(time_row.format(
            m.name,
            m.mean('solve_time_ms'),
            m.std('solve_time_ms'),
            m.percentile('solve_time_ms', 50),
            m.percentile('solve_time_ms', 95),
            m.percentile('solve_time_ms', 100)))

    # ── Joint limit violations ───────────────────────────────────────────────
    print()
    print("Joint limit violations (total across all trials):")
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        total_jl = sum(t.joint_limit_violations for t in m.trials)
        print(f"  {m.name}: {total_jl}")

    # ── Pairwise comparison vs SQP baseline ──────────────────────────────────
    baseline_key = '0_sqp_baseline'
    if baseline_key in methods and methods[baseline_key].n > 0:
        baseline = methods[baseline_key]
        bl_errors = np.array([t.total_error for t in baseline.trials])
        bl_times = np.array([t.solve_time_ms for t in baseline.trials])

        print(f"\n{'=' * 120}")
        print(f"{'PAIRWISE COMPARISON vs SQP BASELINE (converged)':^120}")
        print(f"{'=' * 120}")
        cmp_fmt = "{:<50} {:>10} {:>12} {:>12} {:>10} {:>12}"
        print(cmp_fmt.format('Method', 'Error(mm)', 'Gap to SQP',
                             '% of SQP', 'Time(ms)', 'Speedup'))
        print("-" * 120)

        for key in sorted(methods.keys()):
            m = methods[key]
            if m.n == 0 or key == baseline_key:
                continue
            m_errors = np.array([t.total_error for t in m.trials])
            m_times = np.array([t.solve_time_ms for t in m.trials])

            avg_e = np.mean(m_errors) * 1000
            avg_bl = np.mean(bl_errors) * 1000
            gap = avg_e - avg_bl
            # % of SQP error: 100% = same as SQP, >100% = worse
            pct_of_sqp = avg_e / avg_bl * 100 if avg_bl > 0 else float('inf')
            speedup = np.mean(bl_times) / np.mean(m_times) if np.mean(m_times) > 0 else float('inf')

            print(cmp_fmt.format(
                m.name[:50],
                f"{avg_e:.3f}",
                f"{gap:+.3f} mm",
                f"{pct_of_sqp:.1f}%",
                f"{np.mean(m_times):.3f}",
                f"{speedup:.1f}x"))

    print(f"\n{'=' * 120}")
    return methods


def save_results_json(methods: Dict[str, MethodStats], filepath: str):
    """Save benchmark results to JSON."""
    import json
    data = {'benchmark': 'single_step_near_target', 'methods': {}}
    for key, m in methods.items():
        data['methods'][key] = {
            'name': m.name,
            'n_trials': m.n,
            'hand_l_error_mm_mean': m.mean('hand_l_error') * 1000,
            'hand_l_error_mm_std': m.std('hand_l_error') * 1000,
            'hand_r_error_mm_mean': m.mean('hand_r_error') * 1000,
            'hand_r_error_mm_std': m.std('hand_r_error') * 1000,
            'elbow_l_error_mm_mean': m.mean('elbow_l_error') * 1000,
            'elbow_l_error_mm_std': m.std('elbow_l_error') * 1000,
            'elbow_r_error_mm_mean': m.mean('elbow_r_error') * 1000,
            'elbow_r_error_mm_std': m.std('elbow_r_error') * 1000,
            'total_error_mm_mean': m.mean('total_error') * 1000,
            'total_error_mm_std': m.std('total_error') * 1000,
            'solve_time_ms_mean': m.mean('solve_time_ms'),
            'solve_time_ms_std': m.std('solve_time_ms'),
            'error_reduction_mean': m.mean('error_reduction'),
            'total_cbf_violations': m.total_cbf_violations(),
            'total_joint_limit_violations': sum(
                t.joint_limit_violations for t in m.trials),
        }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single-Step IK Benchmark (near targets)')
    parser.add_argument('--n-configs', type=int, default=50,
                        help='Number of random initial configurations (default: 50)')
    parser.add_argument('--n-targets', type=int, default=10,
                        help='Number of near targets per config (default: 10)')
    parser.add_argument('--n-batch', type=int, default=8,
                        help='Batch size for GPU solvers (default: 4096)')
    parser.add_argument('--target-perturbation', type=float, default=0.01,
                        help='Per-joint perturbation range in rad (default: 0.05 ≈ 3°)')
    parser.add_argument('--sqp-iters', type=int, default=50,
                        help='Max outer iterations for SQP baseline (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    if args.quick:
        args.n_configs = 5
        args.n_targets = 3

    methods = run_benchmark(
        n_configs=args.n_configs,
        n_targets=args.n_targets,
        n_batch=args.n_batch,
        target_perturbation=args.target_perturbation,
        sqp_iters=args.sqp_iters,
        seed=args.seed,
    )

    if args.save:
        save_results_json(methods, args.save)
