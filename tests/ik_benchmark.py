#!/usr/bin/env python3
"""
IK Solver Benchmark — Multi-step Convergence Comparison
========================================================

Compares:
  (1) Single QP IK                     (ProxQP, 34-DOF)
  (2) Distributed Single QP IK         (ProxQP, 2×13-DOF)
  (3) Batched QP IK                    (GPU ADMM, 34-DOF, N parallel)
  (4) Batched Distributed QP IK        (GPU ADMM, 2×13-DOF fused, N parallel)
  (5) Batched α-Continuation IK        (GPU ADMM, max-feasible-α)

Primary metric: Multi-step IK convergence
  Run each method for K IK iterations on far-away targets.
  At each differential IK step, re-linearize (FK + Jacobian) at the new q,
  solve QP for Δq, then update q ← q + Δq.  After K steps, measure final
  tracking error.  Methods with better exploration (basin escape, α
  continuation) should converge to lower final error because they can
  escape poor local minima during the iterative re-linearization process.

Secondary metrics:
  - Per-step timing
  - CBF / joint limit violations
  - Convergence trajectory (error vs step)

Test conditions:
  - N_CONFIGS random initial robot configurations
  - N_TARGETS random feasible task-space targets per configuration
  - Targets generated with LARGE joint perturbation (0.5 rad) to create
    targets that require multiple IK steps
  - Warm-start disabled between trials for fair comparison

Usage:
    cd /home/junhengl/body_tracking
    python -m tests.ik_benchmark [--n-configs 20] [--n-targets 5] [--n-batch 4096]
"""

import sys
import os
import time
import argparse
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
    solve_time_ms: float = 0.0      # Wall-clock time (ms)
    cbf_violation_count: int = 0    # Number of CBF constraints violated
    cbf_violation_max: float = 0.0  # Worst CBF violation magnitude
    joint_limit_violations: int = 0 # Number of joint limit violations
    qp_cost: float = 0.0           # Raw QP objective value
    converged: bool = True
    error_trajectory: List[float] = field(default_factory=list)
    # Per-step total position error (sum of 4 EE norms) at each IK step


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

    def total_cbf_violations(self) -> int:
        return sum(t.cbf_violation_count for t in self.trials)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

# Link indices and offsets (matching retargeting_node.py)
HAND_R_LINK = 22
HAND_L_LINK = 29
ELBOW_R_LINK = 21
ELBOW_L_LINK = 28

HAND_R_OFFSET = np.array([0.0, -0.08, 0.0], dtype=np.float64)
HAND_L_OFFSET = np.array([0.0, 0.08, 0.0], dtype=np.float64)
ELBOW_R_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ELBOW_L_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

XHAND_R = Xtrans(HAND_R_OFFSET)
XHAND_L = Xtrans(HAND_L_OFFSET)
XELBOW_R = Xtrans(ELBOW_R_OFFSET)
XELBOW_L = Xtrans(ELBOW_L_OFFSET)

# CBF parameters (matching robot_dynamics.py)
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


def generate_random_config(rng: np.random.Generator,
                          robot: Optional[Robot] = None,
                          max_retries: int = 50) -> np.ndarray:
    """Generate a random feasible robot configuration.

    If *robot* is provided, rejects configs where any arm EE is inside
    a CBF exclusion sphere.
    """
    dq_zero = np.zeros(DOF, dtype=np.float32)
    for _attempt in range(max_retries):
        q = np.zeros(DOF, dtype=np.float32)
        # Base: fixed hanging position
        q[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        q[3:6] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Joints: random within limits (arm joints only)
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

    return q  # fallback


def _ee_positions_cbf_safe(base_pos: np.ndarray,
                          ee_rpypos_list: List[np.ndarray]) -> bool:
    """Return True if ALL end-effector positions are outside every CBF sphere."""
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


def generate_reachable_target(robot: Robot, q_init: np.ndarray,
                              rng: np.random.Generator,
                              perturbation_range: float = 0.15,
                              max_retries: int = 50
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate reachable task-space targets by FK at a perturbed config.

    Strategy: perturb q_init's arm joints, compute FK → those become the
    desired EE targets. Rejects targets where any EE is inside a CBF
    exclusion sphere (torso / head / crotch).
    """
    dq_zero = np.zeros(DOF, dtype=np.float32)
    base_pos = q_init[:3]

    for _attempt in range(max_retries):
        q_target = q_init.copy()
        # Perturb arm joints only (indices 18-31)
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

    # Fallback: return last sample even if not perfectly safe
    return x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des


def check_cbf_violations(q: np.ndarray, robot: Robot) -> Tuple[int, float]:
    """Check CBF constraint violations after IK solution.

    Returns (n_violations, max_violation) where violation = penetration depth².
    """
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

    # FK returns (6,) = [roll, pitch, yaw, x, y, z]; positions are [3:]
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


# ═══════════════════════════════════════════════════════════════════════════════
# Solver wrappers (each resets state, runs N IK iterations, returns result)
# ═══════════════════════════════════════════════════════════════════════════════

def _sqp_tracking_cost(robot: Robot, q: np.ndarray,
                       x_hand_l_des, x_hand_r_des,
                       x_elbow_l_des, x_elbow_r_des) -> float:
    """Compute the sum of squared position tracking errors (scalar cost for line search)."""
    dq_zero = np.zeros(DOF, dtype=np.float32)
    robot.update(q, dq_zero)
    x_hl = robot.compute_forward_kinematics(HAND_L_LINK, HAND_L_OFFSET)
    x_hr = robot.compute_forward_kinematics(HAND_R_LINK, HAND_R_OFFSET)
    x_el = robot.compute_forward_kinematics(ELBOW_L_LINK, ELBOW_L_OFFSET)
    x_er = robot.compute_forward_kinematics(ELBOW_R_LINK, ELBOW_R_OFFSET)
    cost = (np.sum((x_hl[3:] - x_hand_l_des[3:])**2)
            + np.sum((x_hr[3:] - x_hand_r_des[3:])**2)
            + np.sum((x_el[3:] - x_elbow_l_des[3:])**2)
            + np.sum((x_er[3:] - x_elbow_r_des[3:])**2))
    return float(cost)


def run_global_ik_sqp(robot: Robot, q_init: np.ndarray,
                      x_hand_l_des, x_hand_r_des,
                      x_elbow_l_des, x_elbow_r_des,
                      com_des: np.ndarray,
                      max_outer_iters: int = 50,
                      convergence_tol: float = 1e-6,
                      ls_alpha_min: float = 0.01,
                      ls_beta: float = 0.5,
                      ls_max_trials: int = 10) -> TrialResult:
    """Method 0: Global IK via Sequential QP (SQP with backtracking line search).

    This solves the full nonlinear IK problem by repeatedly linearizing and
    solving a local QP, then updating the configuration with a step size
    chosen by backtracking line search:

        for k = 1 … K:
            1. Linearize at q_k  (compute FK, Jacobians)
            2. Solve QP for δq   (same formulation as Single QP)
            3. Line search: find α ∈ (0, 1] such that
                  cost(q_k + α δq) < cost(q_k)
               using backtracking: α ← β·α until improvement or α < α_min
            4. Update: q_{k+1} = q_k + α δq
            5. Check convergence: ||α δq|| < tol

    This is the standard baseline for nonlinear IK treated as a generic
    optimization problem — no exploitation of the kinematic structure
    beyond what the QP linearization provides.
    """
    dq_zero = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq_zero)

    # Clear warm-start state
    robot._osqp_solver = None
    robot._osqp_prev_dq = np.zeros(DOF, dtype=np.float64)

    # Track best result across iterations
    best_overall_cost = float('inf')
    best_overall_q = q.copy()

    trajectory = []

    t0 = time.perf_counter()

    for k in range(max_outer_iters):
        # 1. Linearize: compute FK and Jacobians at current q
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        # Record error at this linearization point
        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        # 2. Solve QP for δq (reuse the existing single-QP solver)
        q_des_full, dq_sol = robot.update_task_space_command_qp(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des)
        # dq_sol = q_des_full - q  (the QP step)
        dq_step = (q_des_full - q).astype(np.float64)

        if np.linalg.norm(dq_step) < convergence_tol:
            break  # converged

        # 3. Backtracking line search over α ∈ (0, 1]
        cost_current = _sqp_tracking_cost(
            robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)

        # Update best overall if current is better
        if cost_current < best_overall_cost:
            best_overall_cost = cost_current
            best_overall_q = q.copy()

        alpha = 1.0
        best_alpha = alpha
        best_cost = float('inf')

        for _ in range(ls_max_trials):
            q_trial = q + (alpha * dq_step).astype(np.float32)
            # Clip to joint limits
            for i in range(6, DOF):
                q_trial[i] = np.clip(q_trial[i], q_min[i], q_max[i])
            cost_trial = _sqp_tracking_cost(
                robot, q_trial, x_hand_l_des, x_hand_r_des,
                x_elbow_l_des, x_elbow_r_des)
            if cost_trial < best_cost:
                best_cost = cost_trial
                best_alpha = alpha
            if cost_trial < cost_current:
                break  # sufficient decrease
            alpha *= ls_beta
            if alpha < ls_alpha_min:
                alpha = ls_alpha_min
                break

        # 4. Update q with best step
        alpha = best_alpha
        q = q + (alpha * dq_step).astype(np.float32)
        for i in range(6, DOF):
            q[i] = np.clip(q[i], q_min[i], q_max[i])
        robot.update(q, dq_zero)

        # 5. Check convergence
        if np.linalg.norm(alpha * dq_step) < convergence_tol:
            break

    # Check final iteration
    cost_final = _sqp_tracking_cost(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    if cost_final < best_overall_cost:
        best_overall_q = q.copy()

    t1 = time.perf_counter()

    # Use the best result across all iterations
    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, best_overall_q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(best_overall_q, robot)
    n_jl = check_joint_limit_violations(best_overall_q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


def run_single_qp(robot: Robot, q_init: np.ndarray,
                  x_hand_l_des, x_hand_r_des,
                  x_elbow_l_des, x_elbow_r_des,
                  com_des: np.ndarray, n_ik_iters: int = 1) -> TrialResult:
    """Method 1: Single ProxQP (34-DOF, OSQP fallback)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Clear any warm-start state
    robot._osqp_solver = None
    robot._osqp_prev_dq = np.zeros(DOF, dtype=np.float64)

    trajectory = []

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        # Record error at this linearization point
        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        q_des, dq_des = robot.update_task_space_command_qp(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des)
        q = q_des.copy()
        robot.update(q, dq_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(q, robot)
    n_jl = check_joint_limit_violations(q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


def run_distributed_qp(robot: Robot, q_init: np.ndarray,
                       x_hand_l_des, x_hand_r_des,
                       x_elbow_l_des, x_elbow_r_des,
                       com_des: np.ndarray, n_ik_iters: int = 1) -> TrialResult:
    """Method 2: Distributed ProxQP (2×13-DOF)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    trajectory = []

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        q_des, dq_des = robot.update_task_space_command_qp_distributed(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des)
        q = q_des.copy()
        robot.update(q, dq_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(q, robot)
    n_jl = check_joint_limit_violations(q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


def run_batched_qp(robot: Robot, q_init: np.ndarray,
                   x_hand_l_des, x_hand_r_des,
                   x_elbow_l_des, x_elbow_r_des,
                   com_des: np.ndarray, n_ik_iters: int = 1,
                   n_batch: int = 128
                   ) -> TrialResult:
    """Method 3: Batched GPU QP (34-DOF, N parallel ADMM)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Reset GPU solver cache
    robot._gpu_qp_solver = None

    trajectory = []

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        q_des, dq_des = robot.update_task_space_command_qp_gpu_batch(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des,
            n_batch=n_batch, max_iter=50,
            pos_threshold=0.0)  # Disable ratchet for fair comparison
        q = q_des.copy()
        robot.update(q, dq_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(q, robot)
    n_jl = check_joint_limit_violations(q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


def run_batched_distributed_qp(robot: Robot, q_init: np.ndarray,
                                x_hand_l_des, x_hand_r_des,
                                x_elbow_l_des, x_elbow_r_des,
                                com_des: np.ndarray, n_ik_iters: int = 1,
                                n_batch: int = 128,
                                max_iter: int = 50,
                                q_ref: np.ndarray = None,
                                w_ref: float = 0.0
                                ) -> TrialResult:
    """Method 4: Batched Distributed GPU QP (2×13-DOF fused, N parallel ADMM)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Reset GPU solver cache
    robot._gpu_qp_solver_right = None

    trajectory = []

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        q_des, dq_des = robot.update_task_space_command_qp_gpu_batch_distributed(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des,
            n_batch=n_batch, max_iter=max_iter,
            pos_threshold=0.0,  # Disable ratchet for fair comparison
            q_ref=q_ref, w_ref=w_ref)
        q = q_des.copy()
        robot.update(q, dq_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(q, robot)
    n_jl = check_joint_limit_violations(q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


def run_batched_distributed_alpha_qp(robot: Robot, q_init: np.ndarray,
                                     x_hand_l_des, x_hand_r_des,
                                     x_elbow_l_des, x_elbow_r_des,
                                     com_des: np.ndarray, n_ik_iters: int = 1,
                                     n_batch: int = 4096,
                                     n_alpha: int = 8
                                     ) -> TrialResult:
    """Method 5: Max-feasible-α Batched Distributed GPU QP.
    
    Explores intermediate targets along line from current to desired.
    Selects largest α whose solution is feasible and makes progress.
    """
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Reset GPU solver cache
    robot._gpu_qp_solver_right = None

    trajectory = []

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)

        err = (np.linalg.norm(x_hl[3:] - x_hand_l_des[3:]) +
               np.linalg.norm(x_hr[3:] - x_hand_r_des[3:]) +
               np.linalg.norm(x_el[3:] - x_elbow_l_des[3:]) +
               np.linalg.norm(x_er[3:] - x_elbow_r_des[3:]))
        trajectory.append(float(err))

        q_des, dq_des = robot.update_task_space_command_qp_gpu_batch_distributed_alpha(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des,
            n_batch=n_batch, max_iter=50,
            pos_threshold=0.0,  # Disable ratchet for fair comparison
            n_alpha=n_alpha)
        q = q_des.copy()
        robot.update(q, dq_des)
    t1 = time.perf_counter()

    hl_e, hr_e, el_e, er_e = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    n_cbf, max_cbf = check_cbf_violations(q, robot)
    n_jl = check_joint_limit_violations(q)

    return TrialResult(
        hand_l_error=hl_e, hand_r_error=hr_e,
        elbow_l_error=el_e, elbow_r_error=er_e,
        solve_time_ms=(t1 - t0) * 1000,
        cbf_violation_count=n_cbf, cbf_violation_max=max_cbf,
        joint_limit_violations=n_jl,
        error_trajectory=trajectory)


# ═══════════════════════════════════════════════════════════════════════════════
# Basin-escape analysis
#
# Design rationale:
#   The GPU batched solver selects the best instance using linearized QP cost.
#   Because the QP is convex, the unperturbed instance (δq=0) almost always
#   has the lowest linearized cost, making n_batch irrelevant for a single call.
#
#   To properly test basin escape, we instead run B independent IK solves,
#   each starting from a PERTURBED initial configuration.  Each perturbed
#   start produces a different linearization (different Jacobians, FK, CBF
#   geometry), so the GPU batched solver at each start explores a genuinely
#   different region.  We then pick the best by TRUE FK tracking error.
#
#   Comparison:
#     - single-shot: 1 distributed ProxQP from q_init  → 1 solution
#     - "batched":   B ProxQP from perturbed q_init's   → best of B by FK error
#   If best-of-B consistently beats single-shot, it proves that exploring
#   diverse linearization points (the principle behind the GPU batched solver)
#   helps escape poor basins.
# ═══════════════════════════════════════════════════════════════════════════════

def _run_perturbed_basin_trials(
        robot: Robot, q_init: np.ndarray,
        x_hand_l_des, x_hand_r_des,
        x_elbow_l_des, x_elbow_r_des,
        com_des: np.ndarray,
        solver_fn,
        n_candidates: int,
        n_ik_iters: int,
        perturb_sigma: float,
        rng: np.random.Generator,
        solver_kwargs: Dict = None,
        reset_cache_attrs: List[str] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run B independent IK solves from perturbed initial configs.

    For each candidate b in [0, B):
      1. q_b = q_init + δq_b  (δq_b ~ N(0, σ²) on arm joints, 0 for base)
      2. Reset solver caches (avoid warm-start leakage between candidates)
      3. Run n_ik_iters of IK from q_b
      4. Compute TRUE FK tracking error

    Returns:
        best_cost:   float — lowest FK tracking cost among all candidates
        all_costs:   (B,) array of FK costs
        q_solutions: (B, DOF) array of final q for each candidate
    """
    import io, contextlib
    if solver_kwargs is None:
        solver_kwargs = {}
    if reset_cache_attrs is None:
        reset_cache_attrs = []
    dq_zero = np.zeros(DOF, dtype=np.float32)
    all_costs = []
    q_solutions = []

    for b in range(n_candidates):
        # Perturb arm joints (indices 18-31), keep base + legs fixed
        q_start = q_init.copy()
        if b > 0:  # candidate 0 = unperturbed (exact same start as baseline)
            for i in range(18, 32):
                q_start[i] += rng.normal(0, perturb_sigma)
                q_start[i] = np.clip(q_start[i], q_min[i], q_max[i])

        # Reset solver caches so each candidate starts fresh
        for attr in reset_cache_attrs:
            setattr(robot, attr, None)

        q = q_start.copy()
        robot.update(q, dq_zero)

        # Suppress repeated solver fallback warnings during batch trials
        with contextlib.redirect_stderr(io.StringIO()), \
             contextlib.redirect_stdout(io.StringIO()):
            for _ in range(n_ik_iters):
                (x_hl, x_hr, x_el, x_er,
                 J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
                q_des, dq_des = solver_fn(
                    x_elbow_l_des, x_elbow_r_des, x_el, x_er,
                    x_hand_l_des, x_hand_r_des, x_hl, x_hr,
                    J_el, J_er, J_hl, J_hr, com_des,
                    **solver_kwargs)
                q = q_des.copy()
                robot.update(q, dq_des)

        hl_e, hr_e, _, _ = compute_tracking_errors(
            robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
        all_costs.append(hl_e + hr_e)
        q_solutions.append(q.copy())

    all_costs = np.array(all_costs)
    q_solutions = np.array(q_solutions)
    best_cost = float(all_costs.min())

    return best_cost, all_costs, q_solutions


def _basin_stats(single_cost: float, best_cost: float,
                 all_costs: np.ndarray,
                 q_solutions: np.ndarray) -> Dict:
    """Compute basin-escape statistics from raw arrays."""
    escape = bool(best_cost < single_cost)
    improvement = float((single_cost - best_cost) / single_cost if single_cost > 0 else 0.0)
    solution_spread = float(np.mean(np.std(q_solutions[:, 18:32], axis=0)))
    return {
        'single_cost': single_cost,
        'best_cost': best_cost,
        'all_costs_mean': float(all_costs.mean()),
        'all_costs_std': float(all_costs.std()),
        'all_costs_min': float(all_costs.min()),
        'all_costs_max': float(all_costs.max()),
        'escaped': escape,
        'improvement_frac': improvement,
        'solution_spread_rad': solution_spread,
        'n_unique_basins': int(np.sum(
            np.ptp(q_solutions[:, 18:32], axis=0) > 0.05)),
    }


def run_basin_escape_test(robot: Robot, q_init: np.ndarray,
                          x_hand_l_des, x_hand_r_des,
                          x_elbow_l_des, x_elbow_r_des,
                          com_des: np.ndarray,
                          n_ik_iters: int = 1,
                          n_candidates: int = 128,
                          n_batch: int = 128,
                          perturb_sigma: float = 0.1,
                          rng: np.random.Generator = None,
                          ) -> Dict:
    """Test basin-escape using GPU batched solvers from perturbed starts.

    Compares a single-shot distributed ProxQP solve from q_init against
    best-of-B solves from perturbed initial configs using the GPU batched
    solvers, evaluated by TRUE FK tracking error.

    Tests both GPU Batched QP (34-DOF) and GPU Batched Distributed QP
    (2×13-DOF) independently.

    Returns dict with:
        - single_cost: tracking cost from unperturbed single-shot
        - batched_full / batched_distributed: sub-dicts with escape stats
    """
    if rng is None:
        rng = np.random.default_rng()

    import io, contextlib
    dq = np.zeros(DOF, dtype=np.float32)

    # ── Baseline: single-shot distributed ProxQP (unperturbed) ──────────────
    q = q_init.copy()
    robot.update(q, dq)
    with contextlib.redirect_stderr(io.StringIO()), \
         contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_ik_iters):
            (x_hl, x_hr, x_el, x_er,
             J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
            q_des, dq_des = robot.update_task_space_command_qp_distributed_proxqp(
                x_elbow_l_des, x_elbow_r_des, x_el, x_er,
                x_hand_l_des, x_hand_r_des, x_hl, x_hr,
                J_el, J_er, J_hl, J_hr, com_des)
            q = q_des.copy()
            robot.update(q, dq_des)

    hl_e, hr_e, _, _ = compute_tracking_errors(
        robot, q, x_hand_l_des, x_hand_r_des, x_elbow_l_des, x_elbow_r_des)
    single_cost = float(hl_e + hr_e)

    # NOTE: max_iter=20 to match deployment conditions
    gpu_kwargs = dict(n_batch=n_batch, max_iter=20,
                      pos_threshold=0.0)

    solvers_to_test = [
        ('batched_full',
         robot.update_task_space_command_qp_gpu_batch,
         gpu_kwargs, ['_gpu_qp_solver']),
        ('batched_distributed',
         robot.update_task_space_command_qp_gpu_batch_distributed,
         gpu_kwargs, ['_gpu_qp_solver_right']),
    ]

    results = {'single_cost': single_cost}

    for key, solver_fn, kwargs, cache_attrs in solvers_to_test:
        try:
            best, costs, qs = _run_perturbed_basin_trials(
                robot, q_init, x_hand_l_des, x_hand_r_des,
                x_elbow_l_des, x_elbow_r_des, com_des,
                solver_fn=solver_fn,
                n_candidates=n_candidates,
                n_ik_iters=n_ik_iters,
                perturb_sigma=perturb_sigma,
                rng=rng,
                solver_kwargs=kwargs,
                reset_cache_attrs=cache_attrs)
            results[key] = _basin_stats(single_cost, best, costs, qs)
        except Exception:
            results[key] = None  # GPU unavailable

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_configs: int = 20, n_targets: int = 5,
                  n_ik_iters: int = 50, n_batch: int = 4096,
                  target_perturbation: float = 0.5,
                  seed: int = 42):
    """Run the multi-step IK convergence benchmark.

    Each trial:
      1. Sample a random initial config q_init
      2. Generate a FAR target (large joint perturbation → FK)
      3. Run each method for n_ik_iters differential IK steps
         (re-linearize, solve QP, update q each step)
      4. Record final tracking error and convergence trajectory

    Far targets are essential: with close targets (small perturbation),
    one step suffices and all methods converge equally.  With far targets,
    the iterative re-linearization exposes differences in how methods
    handle constraints and local minima.

    Args:
        n_configs: number of random initial configurations
        n_targets: number of far targets per configuration
        n_ik_iters: IK steps per trial (default 50)
        n_batch: GPU batch size (default 4096)
        target_perturbation: joint perturbation range (rad) for target
            generation (default 0.5 — creates challenging far targets)
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    robot = Robot()

    com_des = np.zeros(6, dtype=np.float32)

    # ── Warm-up solvers (triggers one-time import warnings) ──────────────────
    import io, contextlib
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
            robot.update_task_space_command_qp_distributed_proxqp(
                x_el, x_er, x_el, x_er, x_hl, x_hr, x_hl, x_hr,
                J_el, J_er, J_hl, J_hr, com_des)
        except Exception:
            pass
    # Reset rng to undo the warmup config draw
    rng = np.random.default_rng(seed)

    methods = {
        '0_global_ik_sqp': MethodStats(name='Global IK (SQP, line search)'),
        '1_single_qp': MethodStats(name='Single QP (ProxQP 34-DOF)'),
        '2_distributed_qp': MethodStats(name='Distributed QP (ProxQP 2×13-DOF)'),
        '3_batched_qp': MethodStats(name=f'Batched QP (GPU {n_batch}×34-DOF)'),
        '4_batched_distributed': MethodStats(name=f'Batched Distributed (GPU {n_batch}×2×13-DOF)'),
        '5_batched_alpha': MethodStats(name=f'Batched α-Continuation (GPU {n_batch}×n_alpha)'),
    }

    solvers = {
        '0_global_ik_sqp': lambda r, q, hl, hr, el, er, c: run_global_ik_sqp(
            r, q, hl, hr, el, er, c),
        '1_single_qp': lambda r, q, hl, hr, el, er, c: run_single_qp(
            r, q, hl, hr, el, er, c, n_ik_iters),
        '2_distributed_qp': lambda r, q, hl, hr, el, er, c: run_distributed_qp(
            r, q, hl, hr, el, er, c, n_ik_iters),
        '3_batched_qp': lambda r, q, hl, hr, el, er, c: run_batched_qp(
            r, q, hl, hr, el, er, c, n_ik_iters, n_batch),
        '4_batched_distributed': lambda r, q, hl, hr, el, er, c: run_batched_distributed_qp(
            r, q, hl, hr, el, er, c, n_ik_iters, n_batch, max_iter=50),
        '5_batched_alpha': lambda r, q, hl, hr, el, er, c: run_batched_distributed_alpha_qp(
            r, q, hl, hr, el, er, c, n_ik_iters, n_batch),
    }

    # ── Multi-step convergence benchmark ─────────────────────────────────────
    print("=" * 100)
    print("  MULTI-STEP IK CONVERGENCE BENCHMARK")
    print(f"  {n_configs} configs × {n_targets} targets = {n_configs * n_targets} trials per method")
    print(f"  {n_ik_iters} IK iterations per trial | Batch size: {n_batch}")
    print(f"  Target perturbation: {target_perturbation:.2f} rad (far targets)")
    print(f"  Seed: {seed}")
    print("=" * 100)

    total_trials = n_configs * n_targets
    trial_idx = 0
    skipped_methods = set()

    for cfg_i in range(n_configs):
        q_init = generate_random_config(rng, robot)

        for tgt_j in range(n_targets):
            trial_idx += 1
            # Generate FAR target (large perturbation)
            (x_hl_des, x_hr_des,
             x_el_des, x_er_des) = generate_reachable_target(
                robot, q_init, rng,
                perturbation_range=target_perturbation)

            for method_key, solver_fn in solvers.items():
                if method_key in skipped_methods:
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = solver_fn(robot, q_init, x_hl_des, x_hr_des,
                                           x_el_des, x_er_des, com_des)
                    methods[method_key].trials.append(result)
                except Exception as e:
                    skipped_methods.add(method_key)
                    print(f"  ⚠ Skipping {methods[method_key].name}: {e}")

            if trial_idx % 10 == 0 or trial_idx == total_trials:
                print(f"  [{trial_idx}/{total_trials}] trials complete")

    # ── Print results ────────────────────────────────────────────────────────
    print_results(methods, n_ik_iters)

    return methods


def print_results(methods: Dict[str, MethodStats], n_ik_iters: int):
    """Print formatted multi-step convergence benchmark results."""
    print(f"\n{'=' * 110}")
    print(f"{'MULTI-STEP IK CONVERGENCE RESULTS  (' + str(n_ik_iters) + ' IK steps per trial)':^110}")
    print(f"{'=' * 110}")

    # ── Final tracking error table ───────────────────────────────────────────
    header_fmt = "{:<45} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}"
    print(header_fmt.format(
        'Method', 'Hand L', 'Hand R', 'Elbow L', 'Elbow R', 'Time', 'CBF'))
    print(header_fmt.format(
        '', '(mm)', '(mm)', '(mm)', '(mm)', '(ms)', 'Viol.'))
    print("-" * 110)

    row_fmt = "{:<45} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>8.2f} {:>8d}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            print(f"  {m.name:<45} {'(skipped — solver unavailable)':>65}")
            continue
        print(row_fmt.format(
            m.name,
            m.mean('hand_l_error') * 1000,
            m.mean('hand_r_error') * 1000,
            m.mean('elbow_l_error') * 1000,
            m.mean('elbow_r_error') * 1000,
            m.mean('solve_time_ms'),
            m.total_cbf_violations()))

    # ── Std dev ──────────────────────────────────────────────────────────────
    print()
    print("Standard deviations:")
    print("-" * 110)
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(row_fmt.format(
            f"  ±{m.name[:41]}",
            m.std('hand_l_error') * 1000,
            m.std('hand_r_error') * 1000,
            m.std('elbow_l_error') * 1000,
            m.std('elbow_r_error') * 1000,
            m.std('solve_time_ms'),
            0))

    # ── Joint limit violations ───────────────────────────────────────────────
    print()
    print("Joint limit violations (total across all trials):")
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        total_jl = sum(t.joint_limit_violations for t in m.trials)
        print(f"  {m.name}: {total_jl}")

    # ── Convergence trajectory summary ───────────────────────────────────────
    # Show mean error at selected IK steps across all trials
    print(f"\n{'=' * 110}")
    print(f"{'CONVERGENCE TRAJECTORY  (mean total EE position error in mm)':^110}")
    print(f"{'=' * 110}")

    # Select which steps to display (evenly spaced + final)
    if n_ik_iters <= 10:
        display_steps = list(range(n_ik_iters))
    else:
        display_steps = sorted(set(
            [0, 1, 2, 5] +
            list(range(0, n_ik_iters, max(1, n_ik_iters // 8))) +
            [n_ik_iters - 1]))
        display_steps = [s for s in display_steps if s < n_ik_iters]

    # Header
    step_labels = [f"Step {s}" for s in display_steps]
    hdr = "{:<35}" + " {:>9}" * len(display_steps)
    print(hdr.format("Method", *step_labels))
    print("-" * (35 + 10 * len(display_steps)))

    row = "{:<35}" + " {:>9.1f}" * len(display_steps)
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        # Only include trials that have trajectory data
        trajs = [t.error_trajectory for t in m.trials if t.error_trajectory]
        if not trajs:
            # SQP and other methods without trajectory — show just final
            final_err = (m.mean('hand_l_error') + m.mean('hand_r_error') +
                         m.mean('elbow_l_error') + m.mean('elbow_r_error')) * 1000
            vals = [final_err] * len(display_steps)
            print(row.format(m.name[:35], *vals) + "  (no trajectory)")
            continue

        mean_traj = np.mean(trajs, axis=0) * 1000  # convert to mm
        vals = [float(mean_traj[s]) if s < len(mean_traj) else float('nan')
                for s in display_steps]
        print(row.format(m.name[:35], *vals))

    # ── Pairwise comparison vs Single QP baseline ────────────────────────────
    baseline_key = '1_single_qp'
    if baseline_key in methods and methods[baseline_key].n > 0:
        baseline = methods[baseline_key]
        bl_final = np.array([
            t.hand_l_error + t.hand_r_error + t.elbow_l_error + t.elbow_r_error
            for t in baseline.trials])

        print(f"\n{'=' * 110}")
        print(f"{'PAIRWISE COMPARISON vs Single QP':^110}")
        print(f"{'=' * 110}")
        cmp_fmt = "{:<45}  {:>10}  {:>12}  {:>12}  {:>10}"
        print(cmp_fmt.format("Method", "Final (mm)",
                             "Δ vs Single", "% improve", "Wins"))
        print("-" * 110)

        for key in sorted(methods.keys()):
            m = methods[key]
            if m.n == 0 or key == baseline_key:
                continue
            m_final = np.array([
                t.hand_l_error + t.hand_r_error + t.elbow_l_error + t.elbow_r_error
                for t in m.trials])

            avg_m = np.mean(m_final) * 1000
            avg_bl = np.mean(bl_final) * 1000
            delta = avg_m - avg_bl
            pct = (avg_bl - avg_m) / avg_bl * 100 if avg_bl > 0 else 0
            # Count trials where this method beats baseline
            wins = int(np.sum(m_final < bl_final))
            total = min(len(m_final), len(bl_final))
            print(cmp_fmt.format(
                m.name[:45],
                f"{avg_m:.2f}",
                f"{delta:+.2f} mm",
                f"{pct:+.1f}%",
                f"{wins}/{total}"))

    print(f"\n{'=' * 110}")

    return methods


def save_results_json(methods: Dict[str, MethodStats],
                      filepath: str):
    """Save benchmark results to JSON."""
    import json
    data = {'methods': {}}
    for key, m in methods.items():
        # Collect per-trial trajectories
        trajs = [t.error_trajectory for t in m.trials if t.error_trajectory]
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
            'solve_time_ms_mean': m.mean('solve_time_ms'),
            'solve_time_ms_std': m.std('solve_time_ms'),
            'total_cbf_violations': m.total_cbf_violations(),
            'total_joint_limit_violations': sum(
                t.joint_limit_violations for t in m.trials),
            'mean_trajectory_mm': (np.mean(trajs, axis=0) * 1000).tolist()
                if trajs else [],
        }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-step IK Convergence Benchmark')
    parser.add_argument('--n-configs', type=int, default=20,
                        help='Number of random initial configurations (default: 20)')
    parser.add_argument('--n-targets', type=int, default=5,
                        help='Number of random targets per config (default: 5)')
    parser.add_argument('--n-ik-iters', type=int, default=50,
                        help='IK iterations per trial (default: 50)')
    parser.add_argument('--n-batch', type=int, default=4096,
                        help='Batch size for GPU solvers (default: 4096)')
    parser.add_argument('--target-perturbation', type=float, default=0.5,
                        help='Joint perturbation range (rad) for target gen (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file (e.g. --save results.json)')
    args = parser.parse_args()

    if args.quick:
        args.n_configs = 3
        args.n_targets = 2
        args.n_ik_iters = 20

    methods = run_benchmark(
        n_configs=args.n_configs,
        n_targets=args.n_targets,
        n_ik_iters=args.n_ik_iters,
        n_batch=args.n_batch,
        target_perturbation=args.target_perturbation,
        seed=args.seed,
    )

    if args.save:
        save_results_json(methods, args.save)
