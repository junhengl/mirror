#!/usr/bin/env python3
"""
IK Solver Benchmark — Standardized Comparison of 4 IK Methods
=============================================================

Compares:
  (1) Single QP IK                     (ProxQP, 34-DOF)
  (2) Distributed Single QP IK         (ProxQP, 2×13-DOF)
  (3) Batched QP IK                    (GPU ADMM, 34-DOF, N parallel)
  (4) Batched Distributed QP IK        (GPU ADMM, 2×13-DOF fused, N parallel)

Metrics:
  A. Average hand tracking error norm (m)
  B. Average elbow tracking error norm (m)
  C. Average computation time (ms)
  D. CBF constraint violation count & magnitude
  E. Basin-escape metric: fraction of trials where batched methods find a
     better solution than single-shot (proves probabilistic exploration)

Test conditions:
  - N_CONFIGS random initial robot configurations
  - N_TARGETS random feasible task-space targets per configuration
  - Multiple IK iterations per (config, target) pair
  - Warm-start disabled between trials for fair comparison

Usage:
    cd /home/junhengl/body_tracking
    python -m tests.ik_benchmark [--n-configs 50] [--n-targets 10] [--n-batch 128]
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

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
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
        joint_limit_violations=n_jl)


def run_distributed_qp(robot: Robot, q_init: np.ndarray,
                       x_hand_l_des, x_hand_r_des,
                       x_elbow_l_des, x_elbow_r_des,
                       com_des: np.ndarray, n_ik_iters: int = 1) -> TrialResult:
    """Method 2: Distributed ProxQP (2×13-DOF)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
        q_des, dq_des = robot.update_task_space_command_qp_distributed_proxqp(
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
        joint_limit_violations=n_jl)


def run_batched_qp(robot: Robot, q_init: np.ndarray,
                   x_hand_l_des, x_hand_r_des,
                   x_elbow_l_des, x_elbow_r_des,
                   com_des: np.ndarray, n_ik_iters: int = 1,
                   n_batch: int = 128, q_perturb_sigma: float = 0.0
                   ) -> TrialResult:
    """Method 3: Batched GPU QP (34-DOF, N parallel ADMM)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Reset GPU solver cache
    robot._gpu_qp_solver = None

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
        q_des, dq_des = robot.update_task_space_command_qp_gpu_batch(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des,
            n_batch=n_batch, max_iter=20,
            q_perturb_sigma=q_perturb_sigma,
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
        joint_limit_violations=n_jl)


def run_batched_distributed_qp(robot: Robot, q_init: np.ndarray,
                                x_hand_l_des, x_hand_r_des,
                                x_elbow_l_des, x_elbow_r_des,
                                com_des: np.ndarray, n_ik_iters: int = 1,
                                n_batch: int = 128,
                                q_perturb_sigma: float = 0.0
                                ) -> TrialResult:
    """Method 4: Batched Distributed GPU QP (2×13-DOF fused, N parallel ADMM)."""
    dq = np.zeros(DOF, dtype=np.float32)
    q = q_init.copy()
    robot.update(q, dq)

    # Reset GPU solver cache
    robot._gpu_qp_solver_right = None

    t0 = time.perf_counter()
    for _ in range(n_ik_iters):
        (x_hl, x_hr, x_el, x_er,
         J_hl, J_hr, J_el, J_er) = compute_fk_and_jacobians(robot)
        q_des, dq_des = robot.update_task_space_command_qp_gpu_batch_distributed(
            x_elbow_l_des, x_elbow_r_des, x_el, x_er,
            x_hand_l_des, x_hand_r_des, x_hl, x_hr,
            J_el, J_er, J_hl, J_hr, com_des,
            n_batch=n_batch, max_iter=20,
            q_perturb_sigma=q_perturb_sigma,
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
        joint_limit_violations=n_jl)


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

    gpu_kwargs = dict(n_batch=n_batch, max_iter=20,
                      q_perturb_sigma=0.0, pos_threshold=0.0)

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

def run_benchmark(n_configs: int = 50, n_targets: int = 100,
                  n_ik_iters: int = 1, n_batch: int = 128,
                  q_perturb_sigma: float = 0.0,
                  n_basin_configs: int = 100,
                  n_basin_runs: int = 20,
                  basin_perturb_sigma: float = 0.1,
                  seed: int = 42):
    """Run the full benchmark suite.

    Args:
        q_perturb_sigma: perturbation sigma for the main tracking benchmark
            (set 0 for apples-to-apples single-solve comparison).
        basin_perturb_sigma: perturbation sigma for basin-escape analysis.
            MUST be > 0 for batch diversity to work — this is the mechanism
            that gives each batch element a different linearization point.
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
        '1_single_qp': MethodStats(name='Single QP (ProxQP 34-DOF)'),
        '2_distributed_qp': MethodStats(name='Distributed QP (ProxQP 2×13-DOF)'),
        '3_batched_qp': MethodStats(name=f'Batched QP (GPU {n_batch}×34-DOF)'),
        '4_batched_distributed': MethodStats(name=f'Batched Distributed (GPU {n_batch}×2×13-DOF)'),
    }

    solvers = {
        '1_single_qp': lambda r, q, hl, hr, el, er, c: run_single_qp(
            r, q, hl, hr, el, er, c, n_ik_iters),
        '2_distributed_qp': lambda r, q, hl, hr, el, er, c: run_distributed_qp(
            r, q, hl, hr, el, er, c, n_ik_iters),
        '3_batched_qp': lambda r, q, hl, hr, el, er, c: run_batched_qp(
            r, q, hl, hr, el, er, c, n_ik_iters, n_batch, q_perturb_sigma),
        '4_batched_distributed': lambda r, q, hl, hr, el, er, c: run_batched_distributed_qp(
            r, q, hl, hr, el, er, c, n_ik_iters, n_batch, q_perturb_sigma),
    }

    # ── Tracking accuracy & timing benchmark ─────────────────────────────────
    print("=" * 80)
    print("  IK SOLVER BENCHMARK")
    print(f"  {n_configs} configs × {n_targets} targets = {n_configs * n_targets} trials per method")
    print(f"  {n_ik_iters} IK iterations per trial | Batch size: {n_batch}")
    print(f"  q_perturb_sigma: {q_perturb_sigma} | Seed: {seed}")
    print("=" * 80)

    total_trials = n_configs * n_targets
    trial_idx = 0
    skipped_methods = set()  # Track methods that fail (e.g. no CUDA)

    for cfg_i in range(n_configs):
        q_init = generate_random_config(rng, robot)

        for tgt_j in range(n_targets):
            trial_idx += 1
            # Generate reachable target from perturbed FK (CBF-safe)
            (x_hl_des, x_hr_des,
             x_el_des, x_er_des) = generate_reachable_target(robot, q_init, rng)

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

            if trial_idx % 50 == 0 or trial_idx == total_trials:
                print(f"  [{trial_idx}/{total_trials}] trials complete")

    # ── Basin-escape test ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  BASIN-ESCAPE ANALYSIS")
    print(f"  {n_basin_configs} configs × best-of-{n_basin_runs} perturbed starts each")
    print(f"  basin_perturb_sigma: {basin_perturb_sigma}")
    print(f"  GPU batch n_batch: {n_batch}")
    print(f"  Evaluates: does best-of-B GPU batched solve from perturbed starts beat single-shot?")
    print(f"{'=' * 80}")

    basin_results = []
    for cfg_i in range(n_basin_configs):
        q_init = generate_random_config(rng, robot)
        (x_hl_des, x_hr_des,
         x_el_des, x_er_des) = generate_reachable_target(
            robot, q_init, rng, perturbation_range=0.25)

        res = run_basin_escape_test(
            robot, q_init, x_hl_des, x_hr_des, x_el_des, x_er_des,
            com_des, n_ik_iters=n_ik_iters,
            n_candidates=n_basin_runs,
            n_batch=n_batch,
            perturb_sigma=basin_perturb_sigma,
            rng=rng)
        basin_results.append(res)
        bf = res['batched_full']
        bd = res['batched_distributed']
        bf_str = (f"bestB_full={bf['best_cost']*1000:.1f}mm ({'✓' if bf['escaped'] else '✗'})"
                  if bf else 'bestB_full=N/A (no GPU)')
        bd_str = (f"bestB_dist={bd['best_cost']*1000:.1f}mm ({'✓' if bd['escaped'] else '✗'})"
                  if bd else 'bestB_dist=N/A (no GPU)')
        print(f"  Config {cfg_i+1}/{n_basin_configs}: "
              f"single={res['single_cost']*1000:.1f}mm | {bf_str} | {bd_str}")

    # ── Print results ────────────────────────────────────────────────────────
    print_results(methods, basin_results)

    return methods, basin_results


def print_results(methods: Dict[str, MethodStats],
                  basin_results: List[Dict]):
    """Print formatted benchmark results."""
    print(f"\n{'=' * 100}")
    print(f"{'BENCHMARK RESULTS':^100}")
    print(f"{'=' * 100}")

    # ── Header ───────────────────────────────────────────────────────────────
    header_fmt = "{:<40} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}"
    print(header_fmt.format(
        'Method', 'Hand L', 'Hand R', 'Elbow L', 'Elbow R', 'Time', 'CBF'))
    print(header_fmt.format(
        '', '(mm)', '(mm)', '(mm)', '(mm)', '(ms)', 'Viol.'))
    print("-" * 100)

    # ── Per-method rows ──────────────────────────────────────────────────────
    row_fmt = "{:<40} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>8.2f} {:>8d}"
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            print(f"  {m.name:<40} {'(skipped — solver unavailable)':>60}")
            continue
        print(row_fmt.format(
            m.name,
            m.mean('hand_l_error') * 1000,
            m.mean('hand_r_error') * 1000,
            m.mean('elbow_l_error') * 1000,
            m.mean('elbow_r_error') * 1000,
            m.mean('solve_time_ms'),
            m.total_cbf_violations()))

    # ── Standard deviation ───────────────────────────────────────────────────
    print()
    print("Standard deviations:")
    print("-" * 100)
    for key in sorted(methods.keys()):
        m = methods[key]
        if m.n == 0:
            continue
        print(row_fmt.format(
            f"  ±{m.name[:36]}",
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

    # ── Basin-escape summary ────────────────────────────────────────────
    if basin_results:
        print(f"\n{'=' * 100}")
        print(f"{'BASIN-ESCAPE ANALYSIS  (best-of-B perturbed starts vs single-shot)':^100}")
        print(f"{'=' * 100}")

        single_costs = np.array([r['single_cost'] for r in basin_results])

        for label, key in [('Best-of-B GPU Batched QP (34-DOF)', 'batched_full'),
                           ('Best-of-B GPU Batched Distributed (2×13-DOF)', 'batched_distributed')]:
            # Skip if GPU was unavailable (all entries are None)
            if all(r.get(key) is None for r in basin_results):
                print(f"\n  ── {label} ──")
                print(f"    (skipped — GPU solver unavailable)")
                continue
            # Filter to configs where this solver ran
            valid = [r for r in basin_results if r.get(key) is not None]
            print(f"\n  ── {label} ──")
            escaped    = np.array([r[key]['escaped'] for r in valid])
            best_costs = np.array([r[key]['best_cost'] for r in valid])
            all_means  = np.array([r[key]['all_costs_mean'] for r in valid])
            spreads    = np.array([r[key]['solution_spread_rad'] for r in valid])
            n_basins   = np.array([r[key]['n_unique_basins'] for r in valid])
            valid_single = np.array([r['single_cost'] for r in valid])

            print(f"    Escape rate (best-of-B beats single): "
                  f"{np.mean(escaped)*100:.1f}%  ({int(np.sum(escaped))}/{len(escaped)} configs)")
            print(f"    Avg single-shot cost:    {np.mean(valid_single)*1000:.2f} mm")
            print(f"    Avg best-of-B cost:      {np.mean(best_costs)*1000:.2f} mm")
            print(f"    Avg mean-of-B cost:      {np.mean(all_means)*1000:.2f} mm")
            denom = np.mean(valid_single)
            improvement = (denom - np.mean(best_costs)) / denom * 100 if denom > 0 else 0.0
            print(f"    Avg improvement (best):  {improvement:.1f}%")
            print(f"    Avg solution spread:     {np.mean(spreads):.4f} rad")
            print(f"    Avg DOFs with spread > 0.05 rad: {np.mean(n_basins):.1f} / 14")

    print(f"\n{'=' * 100}")

    return methods, basin_results


def save_results_json(methods: Dict[str, MethodStats],
                      basin_results: List[Dict],
                      filepath: str):
    """Save benchmark results to JSON."""
    import json
    data = {
        'methods': {},
        'basin_escape': basin_results,
    }
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
            'solve_time_ms_mean': m.mean('solve_time_ms'),
            'solve_time_ms_std': m.std('solve_time_ms'),
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
    parser = argparse.ArgumentParser(description='IK Solver Benchmark')
    parser.add_argument('--n-configs', type=int, default=50,
                        help='Number of random initial configurations (default: 50)')
    parser.add_argument('--n-targets', type=int, default=10,
                        help='Number of random targets per config (default: 10)')
    parser.add_argument('--n-ik-iters', type=int, default=5,
                        help='IK iterations per trial (default: 5)')
    parser.add_argument('--n-batch', type=int, default=128,
                        help='Batch size for GPU solvers (default: 128)')
    parser.add_argument('--q-perturb-sigma', type=float, default=0.1,
                        help='Q-perturbation sigma for batched methods (default: 0.1)')
    parser.add_argument('--n-basin-configs', type=int, default=10,
                        help='Number of configs for basin-escape test (default: 10)')
    parser.add_argument('--n-basin-runs', type=int, default=20,
                        help='Independent batch runs per basin config (default: 20)')
    parser.add_argument('--basin-perturb-sigma', type=float, default=0.1,
                        help='Q-perturbation sigma for basin-escape test (default: 0.1). '
                             'MUST be > 0 for batch diversity to work.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file (e.g. --save results.json)')
    args = parser.parse_args()

    # overwriting args
    args.n_configs = 50
    args.n_targets = 10
    args.n_basin_configs = 100
    args.n_basin_runs = 20
    args.n_ik_iters = 1
    args.q_perturb_sigma = 0.05
    args.basin_perturb_sigma = 0.01
    args.n_batch = 4

    if args.quick:
        args.n_configs = 5
        args.n_targets = 3
        args.n_basin_configs = 3
        args.n_basin_runs = 5

    methods, basin_results = run_benchmark(
        n_configs=args.n_configs,
        n_targets=args.n_targets,
        n_ik_iters=args.n_ik_iters,
        n_batch=args.n_batch,
        q_perturb_sigma=args.q_perturb_sigma,
        n_basin_configs=args.n_basin_configs,
        n_basin_runs=args.n_basin_runs,
        basin_perturb_sigma=args.basin_perturb_sigma,
        seed=args.seed,
    )

    if args.save:
        save_results_json(methods, basin_results, args.save)
