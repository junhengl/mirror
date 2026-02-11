"""
Batched GPU QP Solver using ADMM in PyTorch.

Solves N parallel instances of the same QP with randomized warm starts
to improve solution quality within a fixed iteration budget. The best
solution (lowest tracking error) is returned.

Optionally perturbs the feedback configuration q across batch instances
(q-randomization) so that each instance solves a *slightly different*
linearization of the IK problem, helping escape local minima that arise
from a single linearization point.

QP formulation:
    min  0.5 * x^T H x + g^T x
    s.t. l <= C x <= u

Uses ADMM (Alternating Direction Method of Multipliers) which decomposes
the QP into:
  1. A linear system solve (shared Cholesky factorization across batches)
  2. An element-wise projection (clamp)
Both operations are highly parallelizable on GPU.

Typical usage for IK:
    solver = BatchedGPUQPSolver(n_batch=64, max_iter=40)
    dq_best = solver.solve(H, g, C, l, u, x_warm=prev_dq)
"""

import numpy as np
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _generate_lhs_samples(n_samples: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Latin Hypercube Samples (LHS).
    
    Ensures uniform marginal coverage along each dimension while maintaining
    diversity. Returns samples in [-1, 1]^n_dims, ready to be scaled by
    perturbation sigma and dampening factors.
    
    Args:
        n_samples: Number of LHS samples to generate
        n_dims: Dimensionality (DOF count, typically 13 for distributed arm or 34 for full)
        seed: Random seed for reproducibility
    
    Returns:
        samples: (n_samples, n_dims) numpy array in [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # LHS matrix: each row is a permutation ensuring uniform marginals
    samples = np.zeros((n_samples, n_dims), dtype=np.float32)
    
    for d in range(n_dims):
        # Generate n_samples evenly-spaced points in [0, 1]
        bins = np.arange(n_samples) / n_samples
        # Random offset within each bin [i/n, (i+1)/n]
        offsets = np.random.rand(n_samples) / n_samples
        samples[:, d] = bins + offsets
        # Shuffle to randomize order (standard LHS)
        np.random.shuffle(samples[:, d])
    
    # Map from [0, 1] to [-1, 1]
    samples = 2.0 * samples - 1.0
    
    return samples


class BatchedGPUQPSolver:
    """
    Batched QP solver using ADMM on GPU via PyTorch.

    Solves N_batch copies of the same QP (or slightly perturbed copies
    when q_perturb_sigma > 0) in parallel, each with a different random
    warm start. Returns the best solution (lowest QP cost among feasible
    candidates evaluated against the *original* unperturbed problem).

    ADMM (scaled form) updates per iteration:
        x  <- (H + ρ C^T C)^{-1} [-g + ρ C^T (z - u)]
        z  <- clamp(α(Cx) + (1-α)z_old + u, l, upper)   [over-relaxed]
        u  <- u + α(Cx) + (1-α)z_old - z

    Where α ∈ [1.0, 1.8] is the over-relaxation parameter for faster
    convergence, and u is the scaled dual variable.

    Q-perturbation mode (q_perturb_sigma > 0):
        For each batch b, generate δq_b ~ N(0, σ²I).
        The perturbed QP for instance b has:
          - Same H (Jacobians assumed constant for small δq)
          - g_b  = g + H_task @ δq_b  (linearized error correction)
          - lb_b = lb - δq_b (box only), ub_b = ub - δq_b (box only)
          - CBF bounds unchanged (small perturbation)
        After solving dq_b, the effective displacement is δq_b + dq_b.
        Selection uses the original QP cost at effective_dq.
    """

    def __init__(self,
                 n_batch: int = 64,
                 max_iter: int = 40,
                 rho: float = 50.0,
                 alpha: float = 1.6,
                 sigma: float = 1e-6,
                 device: Optional[str] = None,
                 dtype: str = 'float32'):
        """
        Args:
            n_batch:  Number of parallel QP instances (more = better exploration,
                      but diminishing returns past ~128).
            max_iter: ADMM iterations per solve. 30-50 is typically sufficient
                      for IK-level accuracy.
            rho:      ADMM penalty parameter. Higher values enforce feasibility
                      faster but slow down cost minimization. Range [10, 200].
            alpha:    Over-relaxation parameter in [1.0, 1.8]. Values > 1.0
                      typically speed up convergence. Default 1.6 is a good
                      general-purpose choice.
            sigma:    Tikhonov regularization added to K for numerical stability.
            device:   'cuda', 'cpu', or None (auto-detect GPU).
            dtype:    'float32' (fast, sufficient for IK) or 'float64' (robust).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BatchedGPUQPSolver. "
                              "Install with: pip install torch")

        self.n_batch = n_batch
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.sigma = sigma

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.torch_dtype = torch.float32 if dtype == 'float32' else torch.float64

        # Previous best solution for temporal warm-starting
        self._prev_best: Optional[torch.Tensor] = None
        self._prev_n: int = 0

        # Diagnostics (updated each solve)
        self.last_best_cost: float = 0.0
        self.last_best_violation: float = 0.0
        self.last_all_costs: Optional[np.ndarray] = None

    @torch.no_grad()
    def solve(self,
              H: np.ndarray,
              g: np.ndarray,
              C: np.ndarray,
              l: np.ndarray,
              u: np.ndarray,
              x_warm: Optional[np.ndarray] = None,
              q_perturb_sigma: float = 0.0,
              n_box: int = 0,
              H_task: Optional[np.ndarray] = None,
              perturb_dampen: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the batched QP and return the best solution.

        Args:
            H: (n, n) Hessian matrix (positive semi-definite, will be regularized)
            g: (n,)   Linear cost vector
            C: (m, n) Constraint matrix (stacked [I; A_cbf] for box + CBF)
            l: (m,)   Lower bounds on C @ x
            u: (m,)   Upper bounds on C @ x  (use np.inf for one-sided)
            x_warm: (n,) Optional warm start center (e.g. previous solution)
            q_perturb_sigma: float ≥ 0. When > 0, randomize the feedback q for
                each batch instance by δq ~ N(0, σ²I) to explore different
                linearization points. Default 0 = no q perturbation.
            n_box: int. Number of leading constraint rows that are box constraints
                (I @ dq ∈ [lb, ub]). Used to shift box bounds per batch when
                q_perturb_sigma > 0. The remaining rows are CBF constraints
                whose bounds are kept unchanged. Set to n (= DOF) for the
                standard [I; A_cbf] layout.
            H_task: (n, n) Optional. Task-only Hessian (= H - Wq - regularization).
                Used for the linearized g correction under q perturbation:
                g_b = g + H_task @ δq_b. If None and q_perturb_sigma > 0,
                falls back to using H itself (a reasonable approximation).
            perturb_dampen: (n,) Optional. Per-DOF scaling factors for the
                perturbation δq. Values < 1.0 dampen that DOF's perturbation
                (e.g. 0.1 for base translation). If None, the default base-DOF
                dampening is used (DOFs 0-2 ×0.1, DOFs 3-5 ×0.3). Use this
                when the problem has non-standard DOF layout (e.g. fused
                block-diagonal with multiple base DOF ranges).

        Returns:
            x_best: (n,) numpy float32 — best solution across all batch instances
        """
        n = H.shape[0]
        m = C.shape[0]
        B = self.n_batch
        rho = self.rho
        alph = self.alpha
        dt = self.torch_dtype
        dev = self.device
        do_q_perturb = (q_perturb_sigma > 0.0 and n_box > 0)

        # ── Transfer to GPU ──────────────────────────────────────────────
        H_t = torch.as_tensor(np.ascontiguousarray(H), dtype=dt, device=dev)
        g_t = torch.as_tensor(np.ascontiguousarray(g), dtype=dt, device=dev)
        C_t = torch.as_tensor(np.ascontiguousarray(C), dtype=dt, device=dev)
        l_t = torch.as_tensor(np.ascontiguousarray(l), dtype=dt, device=dev)
        u_t = torch.as_tensor(np.ascontiguousarray(u), dtype=dt, device=dev)

        # Clamp infinities so GPU arithmetic doesn't produce NaN
        l_t = l_t.clamp(min=-1e8)
        u_t = u_t.clamp(max=1e8)

        # ── Q-perturbation setup ─────────────────────────────────────────
        if do_q_perturb:
            # H_task for the linearized g correction: g_b = g + H_task @ δq_b
            if H_task is not None:
                Ht = torch.as_tensor(np.ascontiguousarray(H_task), dtype=dt, device=dev)
            else:
                Ht = H_t  # approximate

            # Generate per-batch perturbations δq: (B, n)
            delta_q = torch.zeros(B, n, dtype=dt, device=dev)
            # Instance 0: no perturbation (exact original problem)
            
            # Split remaining instances: ~25% Latin Hypercube Sampling, ~75% Gaussian
            n_lhs = max(1, int(0.25 * (B - 1)))
            n_gaussian = (B - 1) - n_lhs
            
            # LHS samples: uniform coverage of [-1, 1]^n configuration space
            if n_lhs > 0:
                lhs_samples = _generate_lhs_samples(n_lhs, n)
                delta_q[1:1+n_lhs] = q_perturb_sigma * torch.from_numpy(lhs_samples).to(dtype=dt, device=dev)
            
            # Gaussian samples: tiered exploration around warm start (if available)
            if n_gaussian > 0:
                delta_q[1+n_lhs:] = q_perturb_sigma * torch.randn(n_gaussian, n, dtype=dt, device=dev)
            
            # Apply per-DOF dampening
            if perturb_dampen is not None:
                dampen_t = torch.as_tensor(
                    np.ascontiguousarray(perturb_dampen), dtype=dt, device=dev)
                delta_q *= dampen_t  # (B, n) * (n,) broadcast
            else:
                # Default: dampen floating-base DOFs 0-5
                delta_q[:, :3] *= 0.1   # dampen base translation perturbation
                delta_q[:, 3:6] *= 0.3  # dampen base orientation perturbation

            # Per-batch g: g_b = g + H_task @ δq_b → (B, n)
            g_batch = g_t.unsqueeze(0) + delta_q @ Ht.T  # (B, n)

            # Per-batch bounds: shift box constraints by -δq, keep CBF unchanged
            # l_batch, u_batch: (B, m)
            l_batch = l_t.unsqueeze(0).expand(B, -1).clone()
            u_batch = u_t.unsqueeze(0).expand(B, -1).clone()
            l_batch[:, :n_box] -= delta_q[:, :n_box]  # box lb -= δq
            u_batch[:, :n_box] -= delta_q[:, :n_box]  # box ub -= δq
        else:
            delta_q = None

        # ── Pre-compute KKT matrix inverse (shared across all batches) ──
        # For small n (34 or 13), explicit inverse + batched matmul is faster
        # than cholesky_solve per iteration due to lower kernel-launch overhead.
        # K = H + rho * C^T C + sigma * I
        CtC = C_t.T @ C_t
        K = H_t + rho * CtC
        K = 0.5 * (K + K.T)                           # enforce symmetry
        K.diagonal().add_(self.sigma)
        K_inv = torch.linalg.inv(K)                     # (n, n)  — O(n^3), tiny

        # ── Generate randomized warm starts (B, n) ───────────────────────
        x = torch.zeros(B, n, dtype=dt, device=dev)

        # Determine the center for warm starts
        if x_warm is not None:
            x_center = torch.as_tensor(
                np.ascontiguousarray(x_warm), dtype=dt, device=dev)
        elif self._prev_best is not None and self._prev_n == n:
            x_center = self._prev_best
        else:
            x_center = None

        if x_center is not None:
            # Tiered randomization strategy:
            #   Tier 0 (1):    exact warm start (exploitation)
            #   Tier 1 (~20%): σ=0.01  (fine local search)
            #   Tier 2 (~20%): σ=0.05  (medium exploration)
            #   Tier 3 (~20%): σ=0.15  (broad exploration)
            #   Tier LHS (~20%): Latin Hypercube sampling (workspace distributed)
            #   Tier 4 (rest): σ=0.40  (wide exploration to escape local minima)
            x[0] = x_center
            idx = 1
            
            # Gaussian tiered: 20% each for fine/medium/broad
            tier_fracs = [0.20, 0.20, 0.20]
            tier_sigmas = [0.01, 0.05, 0.15]
            for frac, sig in zip(tier_fracs, tier_sigmas):
                cnt = max(1, int(B * frac))
                end = min(idx + cnt, B)
                if idx < end:
                    x[idx:end] = x_center + sig * torch.randn(
                        end - idx, n, dtype=dt, device=dev)
                idx = end
            
            # Latin Hypercube: ~20% for distributed workspace exploration
            n_lhs_warmstart = max(1, int(B * 0.20))
            end = min(idx + n_lhs_warmstart, B)
            if idx < end:
                lhs_samples = _generate_lhs_samples(end - idx, n)
                # Scale LHS to add meaningful diversity around warm start
                lhs_tensor = torch.from_numpy(lhs_samples).to(dtype=dt, device=dev)
                x[idx:end] = x_center + 0.25 * lhs_tensor  # Moderate scaling
                idx = end
            
            # Remaining: wide Gaussian exploration
            if idx < B:
                x[idx:] = x_center + 0.40 * torch.randn(
                    B - idx, n, dtype=dt, device=dev)
        else:
            # No warm start: mix Gaussian (50%) and LHS (50%) directly
            n_lhs_direct = max(1, int(B * 0.50))
            n_gaussian_direct = B - n_lhs_direct
            
            # LHS samples
            if n_lhs_direct > 0:
                lhs_samples = _generate_lhs_samples(n_lhs_direct, n)
                x[:n_lhs_direct] = 0.15 * torch.from_numpy(lhs_samples).to(dtype=dt, device=dev)
            
            # Gaussian samples
            if n_gaussian_direct > 0:
                x[n_lhs_direct:] = 0.10 * torch.randn(n_gaussian_direct, n, dtype=dt, device=dev)

        # ── Initialize ADMM dual variables ───────────────────────────────
        if do_q_perturb:
            Cx = x @ C_t.T                             # (B, m)
            z = Cx.clamp(min=l_batch, max=u_batch)
        else:
            Cx = x @ C_t.T                             # (B, m)
            z = Cx.clamp(min=l_t, max=u_t)
        u_dual = torch.zeros(B, m, dtype=dt, device=dev)

        # ── ADMM iterations ──────────────────────────────────────────────
        # Precompute combined matrices to reduce per-iteration kernel launches:
        #   x = K_inv @ (-g + ρ C^T (z - u))
        #     = K_inv @ (-g) + ρ K_inv @ C^T @ (z - u)
        #     = x_offset + (z - u) @ M
        # where x_offset = K_inv @ (-g)     [vector or (B,n) if q-perturbed]
        #       M = ρ (K_inv @ C^T)^T       [(m, n), computed once]
        M = rho * (K_inv @ C_t.T).T                     # (m, n)
        Ct = C_t.T                                      # (n, m)

        if do_q_perturb:
            # Per-batch offset: x_offset = (-g_batch) @ K_inv → (B, n)
            x_offset = (-g_batch) @ K_inv               # (B, n)
        else:
            x_offset = (-g_t) @ K_inv                   # (n,)

        for _ in range(self.max_iter):
            # x-update: x = x_offset + (z - u) @ M
            x = x_offset + (z - u_dual) @ M            # (B, n) — single batched matmul

            # z-update with over-relaxation:
            #   x_hat = α Cx + (1-α) z_old
            #   z_new = clamp(x_hat + u_dual, l, u)
            Cx = x @ Ct                                 # (B, m)
            x_hat = alph * Cx + (1.0 - alph) * z       # over-relaxed
            if do_q_perturb:
                z_new = (x_hat + u_dual).clamp(min=l_batch, max=u_batch)
            else:
                z_new = (x_hat + u_dual).clamp(min=l_t, max=u_t)

            # u-update (scaled dual ascent)
            u_dual = u_dual + x_hat - z_new
            z = z_new

        # ── Compute effective dq and select best ─────────────────────────
        if do_q_perturb:
            # Effective displacement from original q_fb: δq_b + dq_b
            effective_x = delta_q + x                   # (B, n)
        else:
            effective_x = x

        # QP cost evaluated against ORIGINAL (unperturbed) problem:
        # 0.5 * eff^T H eff + g^T eff
        Heff = effective_x @ H_t                        # (B, n)
        cost = 0.5 * (effective_x * Heff).sum(dim=1) + (effective_x * g_t).sum(dim=1)

        # Penalize constraint violations against ORIGINAL bounds
        Ceff = effective_x @ Ct                         # (B, m)
        lb_viol = (l_t - Ceff).clamp(min=0)
        ub_viol = (Ceff - u_t).clamp(min=0)
        violation = lb_viol.pow(2).sum(dim=1) + ub_viol.pow(2).sum(dim=1)
        total_cost = cost + 1e6 * violation

        best_idx = torch.argmin(total_cost)
        x_best = effective_x[best_idx]

        # Cache the raw dq (not effective) for warm-starting next call
        self._prev_best = x[best_idx].clone()
        self._prev_n = n

        # Store diagnostics
        self.last_best_cost = cost[best_idx].item()
        self.last_best_violation = violation[best_idx].item()
        self.last_all_costs = total_cost.cpu().numpy()

        return x_best.cpu().numpy().astype(np.float32)

    def reset(self):
        """Clear cached warm start (e.g. on task change)."""
        self._prev_best = None
        self._prev_n = 0

    @staticmethod
    @torch.no_grad()
    def solve_pair(solver: 'BatchedGPUQPSolver',
                   H_a: np.ndarray, g_a: np.ndarray, C_a: np.ndarray,
                   l_a: np.ndarray, u_a: np.ndarray,
                   H_b: np.ndarray, g_b: np.ndarray, C_b: np.ndarray,
                   l_b: np.ndarray, u_b: np.ndarray,
                   x_warm_a: Optional[np.ndarray] = None,
                   x_warm_b: Optional[np.ndarray] = None,
                   q_perturb_sigma: float = 0.0,
                   n_box_a: int = 0, n_box_b: int = 0,
                   H_task_a: Optional[np.ndarray] = None,
                   H_task_b: Optional[np.ndarray] = None):
        """Solve two QPs as a single fused block-diagonal problem.

        Embeds both QPs into one (n_a + n_b)-dimensional QP with
        block-diagonal structure and solves with a single GPU batch call.
        This halves kernel-launch overhead compared to sequential solving,
        achieving ~2× speedup for small problems where launch latency
        dominates.

        Constraint rows are reordered so all box constraints come first
        (required for correct q-perturbation bound shifting).

        Args:
            solver:  A single BatchedGPUQPSolver instance (reused across calls).
            H/g/C/l/u:  Per-QP matrices (problem A and B).
            x_warm_a/b:  Optional warm starts for each sub-problem.
            q_perturb_sigma:  Per-batch q perturbation (applied to both).
            n_box_a/b:  Number of leading box-constraint rows per sub-problem.
            H_task_a/b:  Task-only Hessians for g correction under perturbation.
        Returns:
            (x_best_a, x_best_b): tuple of (n_a,) and (n_b,) numpy float32.
        """
        from scipy.linalg import block_diag as blkdiag

        n_a, n_b = H_a.shape[0], H_b.shape[0]
        m_a, m_b = C_a.shape[0], C_b.shape[0]

        # ── Block-diagonal Hessian & linear term ─────────────────────────
        H_fused = blkdiag(H_a, H_b).astype(np.float64)
        g_fused = np.concatenate([g_a, g_b]).astype(np.float64)

        # ── Reorder constraints: [box_a, box_b, cbf_a, cbf_b] ───────────
        # This ensures the first n_box_fused rows are box constraints so
        # q-perturbation can shift their bounds correctly.
        n_cbf_a = m_a - n_box_a
        n_cbf_b = m_b - n_box_b
        Z_a = np.zeros  # shorthand
        Z_b = np.zeros

        C_fused = np.vstack([
            np.hstack([C_a[:n_box_a],  Z_a((n_box_a, n_b))]),   # box A
            np.hstack([Z_b((n_box_b, n_a)), C_b[:n_box_b]]),    # box B
            np.hstack([C_a[n_box_a:],  Z_a((n_cbf_a, n_b))]),   # cbf A
            np.hstack([Z_b((n_cbf_b, n_a)), C_b[n_box_b:]]),    # cbf B
        ]).astype(np.float64)

        l_fused = np.concatenate([
            l_a[:n_box_a], l_b[:n_box_b],
            l_a[n_box_a:], l_b[n_box_b:]
        ]).astype(np.float64)
        u_fused = np.concatenate([
            u_a[:n_box_a], u_b[:n_box_b],
            u_a[n_box_a:], u_b[n_box_b:]
        ]).astype(np.float64)

        n_box_fused = n_box_a + n_box_b

        # ── Warm start (concatenate) ────────────────────────────────────
        x_warm_fused = None
        if x_warm_a is not None and x_warm_b is not None:
            x_warm_fused = np.concatenate([x_warm_a, x_warm_b])
        elif x_warm_a is not None:
            x_warm_fused = np.concatenate([x_warm_a, np.zeros(n_b, dtype=np.float32)])
        elif x_warm_b is not None:
            x_warm_fused = np.concatenate([np.zeros(n_a, dtype=np.float32), x_warm_b])

        # ── Task Hessian (block-diagonal) ────────────────────────────────
        H_task_fused = None
        if H_task_a is not None and H_task_b is not None:
            H_task_fused = blkdiag(H_task_a, H_task_b).astype(np.float64)

        # ── Per-DOF perturbation dampening for both bases ────────────────
        perturb_dampen = None
        if q_perturb_sigma > 0:
            n_fused = n_a + n_b
            perturb_dampen = np.ones(n_fused, dtype=np.float64)
            # Problem A base: DOFs 0:3 translation, 3:6 orientation
            perturb_dampen[0:3] = 0.1
            perturb_dampen[3:6] = 0.3
            # Problem B base: DOFs n_a:n_a+3 translation, n_a+3:n_a+6 orientation
            perturb_dampen[n_a:n_a + 3] = 0.1
            perturb_dampen[n_a + 3:n_a + 6] = 0.3

        # ── Single fused solve ───────────────────────────────────────────
        x_fused = solver.solve(
            H_fused, g_fused, C_fused, l_fused, u_fused,
            x_warm=x_warm_fused,
            q_perturb_sigma=q_perturb_sigma,
            n_box=n_box_fused,
            H_task=H_task_fused,
            perturb_dampen=perturb_dampen
        )

        return x_fused[:n_a], x_fused[n_a:]
