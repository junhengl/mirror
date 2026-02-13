"""
Batched GPU QP Solver using ADMM in PyTorch.

Solves N parallel instances of the same QP with randomized warm starts
to improve solution quality within a fixed iteration budget. The best
solution (lowest tracking error) is returned.

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
from typing import Optional, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BatchedGPUQPSolver:
    """
    Batched QP solver using ADMM on GPU via PyTorch.

    Solves N_batch copies of the same QP in parallel, each with a
    different random warm start. Returns the best solution (lowest QP
    cost among feasible candidates).

    ADMM (scaled form) updates per iteration:
        x  <- (H + ρ C^T C)^{-1} [-g + ρ C^T (z - u)]
        z  <- clamp(α(Cx) + (1-α)z_old + u, l, upper)   [over-relaxed]
        u  <- u + α(Cx) + (1-α)z_old - z

    Where α ∈ [1.0, 1.8] is the over-relaxation parameter for faster
    convergence, and u is the scaled dual variable.
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
            n_batch:  Number of parallel QP instances.
            max_iter: ADMM iterations per solve.
            rho:      ADMM penalty parameter.
            alpha:    Over-relaxation parameter in [1.0, 1.8].
            sigma:    Tikhonov regularization for numerical stability.
            device:   'cuda', 'cpu', or None (auto-detect GPU).
            dtype:    'float32' or 'float64'.
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
              x_warm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the batched QP and return the best solution.

        Args:
            H: (n, n) Hessian matrix (positive semi-definite)
            g: (n,)   Linear cost vector
            C: (m, n) Constraint matrix
            l: (m,)   Lower bounds on C @ x
            u: (m,)   Upper bounds on C @ x
            x_warm: (n,) Optional warm start center

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

        # ── Transfer to GPU ──────────────────────────────────────────────
        H_t = torch.as_tensor(np.ascontiguousarray(H), dtype=dt, device=dev)
        g_t = torch.as_tensor(np.ascontiguousarray(g), dtype=dt, device=dev)
        C_t = torch.as_tensor(np.ascontiguousarray(C), dtype=dt, device=dev)
        l_t = torch.as_tensor(np.ascontiguousarray(l), dtype=dt, device=dev)
        u_t = torch.as_tensor(np.ascontiguousarray(u), dtype=dt, device=dev)

        l_t = l_t.clamp(min=-1e8)
        u_t = u_t.clamp(max=1e8)

        # ── Pre-compute KKT matrix inverse (shared across all batches) ──
        CtC = C_t.T @ C_t
        K = H_t + rho * CtC
        K = 0.5 * (K + K.T)
        K.diagonal().add_(self.sigma)
        K_inv = torch.linalg.inv(K)

        # ── Generate randomized warm starts (B, n) ───────────────────────
        x = torch.zeros(B, n, dtype=dt, device=dev)

        if x_warm is not None:
            x_center = torch.as_tensor(
                np.ascontiguousarray(x_warm), dtype=dt, device=dev)
        elif self._prev_best is not None and self._prev_n == n:
            x_center = self._prev_best
        else:
            x_center = None

        if x_center is not None:
            x[0] = x_center
            idx = 1
            tier_fracs = [0.25, 0.25, 0.25]
            tier_sigmas = [0.01, 0.10, 0.30]
            for frac, sig in zip(tier_fracs, tier_sigmas):
                cnt = max(1, int((B - 1) * frac))
                end = min(idx + cnt, B)
                if idx < end:
                    x[idx:end] = x_center + sig * torch.randn(
                        end - idx, n, dtype=dt, device=dev)
                idx = end
            if idx < B:
                x[idx:] = x_center + 0.60 * torch.randn(
                    B - idx, n, dtype=dt, device=dev)
        else:
            x = 0.20 * torch.randn(B, n, dtype=dt, device=dev)

        # ── Initialize ADMM dual variables ───────────────────────────────
        Cx = x @ C_t.T
        z = Cx.clamp(min=l_t, max=u_t)
        u_dual = torch.zeros(B, m, dtype=dt, device=dev)

        # ── ADMM iterations ──────────────────────────────────────────────
        M = rho * (K_inv @ C_t.T).T
        Ct = C_t.T
        x_offset = (-g_t) @ K_inv

        for _ in range(self.max_iter):
            x = x_offset + (z - u_dual) @ M
            Cx = x @ Ct
            x_hat = alph * Cx + (1.0 - alph) * z
            z_new = (x_hat + u_dual).clamp(min=l_t, max=u_t)
            u_dual = u_dual + x_hat - z_new
            z = z_new

        # ── Select best solution ─────────────────────────────────────────
        Hx = x @ H_t
        cost = 0.5 * (x * Hx).sum(dim=1) + (x * g_t).sum(dim=1)

        Cx_all = x @ Ct
        lb_viol = (l_t - Cx_all).clamp(min=0)
        ub_viol = (Cx_all - u_t).clamp(min=0)
        violation = lb_viol.pow(2).sum(dim=1) + ub_viol.pow(2).sum(dim=1)
        total_cost = cost + 1e6 * violation

        best_idx = torch.argmin(total_cost)
        x_best = x[best_idx]

        self._prev_best = x_best.clone()
        self._prev_n = n

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
                   x_warm_b: Optional[np.ndarray] = None):
        """Solve two QPs as a single fused block-diagonal problem.

        Args:
            solver:  A single BatchedGPUQPSolver instance.
            H/g/C/l/u:  Per-QP matrices (problem A and B).
            x_warm_a/b:  Optional warm starts for each sub-problem.
        Returns:
            (x_best_a, x_best_b): tuple of (n_a,) and (n_b,) numpy float32.
        """
        from scipy.linalg import block_diag as blkdiag

        n_a, n_b = H_a.shape[0], H_b.shape[0]

        H_fused = blkdiag(H_a, H_b).astype(np.float64)
        g_fused = np.concatenate([g_a, g_b]).astype(np.float64)
        C_fused = blkdiag(C_a, C_b).astype(np.float64)
        l_fused = np.concatenate([l_a, l_b]).astype(np.float64)
        u_fused = np.concatenate([u_a, u_b]).astype(np.float64)

        x_warm_fused = None
        if x_warm_a is not None and x_warm_b is not None:
            x_warm_fused = np.concatenate([x_warm_a, x_warm_b])
        elif x_warm_a is not None:
            x_warm_fused = np.concatenate([x_warm_a, np.zeros(n_b, dtype=np.float32)])
        elif x_warm_b is not None:
            x_warm_fused = np.concatenate([np.zeros(n_a, dtype=np.float32), x_warm_b])

        x_fused = solver.solve(
            H_fused, g_fused, C_fused, l_fused, u_fused,
            x_warm=x_warm_fused
        )

        return x_fused[:n_a], x_fused[n_a:]

    @staticmethod
    @torch.no_grad()
    def solve_pair_multi_g(solver: 'BatchedGPUQPSolver',
                           H_r: np.ndarray, H_l: np.ndarray,
                           g_r_list: List[np.ndarray],
                           g_l_list: List[np.ndarray],
                           C_r: np.ndarray, C_l: np.ndarray,
                           l_r: np.ndarray, u_r: np.ndarray,
                           l_l: np.ndarray, u_l: np.ndarray):
        """Solve K target-perturbed QPs in a SINGLE efficient GPU call.

        All K linearizations share the same H, C, l, u (Jacobians, weights,
        joint limits, and CBF constraints are unchanged by target perturbation).
        Only g differs per linearization (since g depends on the tracking
        error e = x_des_perturbed - x_current).

        Because H is shared, we compute a SINGLE K_inv and M matrix.
        Per-batch g just means per-batch x_offset = (-g) @ K_inv, which is
        a (B, n) tensor — the standard ADMM loop handles this naturally
        with no einsum or extra memory.

        The batch B is partitioned into K groups.  Best solution is selected
        on the **nominal** (index-0) QP cost.

        Args:
            solver:      BatchedGPUQPSolver instance.
            H_r, H_l:    (n_sub, n_sub) shared Hessians for right/left.
            g_r/l_list:  Lists of (n_sub,) — per-lin linear cost vectors.
            C_r, C_l:    (m, n_sub) shared constraint matrices.
            l/u_r/l:     (m,) shared bounds (same for all lins).
        Returns:
            (dq_right, dq_left): best (n_a,), (n_b,) numpy float32.
        """
        from scipy.linalg import block_diag as blkdiag

        n_lin = len(g_r_list)
        n_a, n_b = H_r.shape[0], H_l.shape[0]
        n = n_a + n_b
        B = solver.n_batch
        dt = solver.torch_dtype
        dev = solver.device
        rho = solver.rho
        alph = solver.alpha
        per_lin = B // n_lin

        # ── Fuse block-diagonal H, C, bounds (shared across all lins) ───
        H_fused = blkdiag(H_r, H_l).astype(np.float64)
        C_fused = blkdiag(C_r, C_l).astype(np.float64)
        l_fused = np.concatenate([l_r, l_l]).astype(np.float64)
        u_fused = np.concatenate([u_r, u_l]).astype(np.float64)

        H_t = torch.as_tensor(np.ascontiguousarray(H_fused), dtype=dt, device=dev)
        C_t = torch.as_tensor(np.ascontiguousarray(C_fused), dtype=dt, device=dev)
        l_t = torch.as_tensor(np.ascontiguousarray(l_fused), dtype=dt, device=dev).clamp(min=-1e8)
        u_t = torch.as_tensor(np.ascontiguousarray(u_fused), dtype=dt, device=dev).clamp(max=1e8)
        m = C_fused.shape[0]

        # ── Single K_inv (shared H → shared KKT) ────────────────────────
        Ct = C_t.T
        CtC = Ct @ C_t
        K = H_t + rho * CtC
        K = 0.5 * (K + K.T)
        K.diagonal().add_(solver.sigma)
        K_inv = torch.linalg.inv(K)
        M = rho * (K_inv @ C_t.T).T   # (m, n)

        # ── Per-batch x_offset from per-lin g ────────────────────────────
        x_offset = torch.empty(B, n, dtype=dt, device=dev)
        for k in range(n_lin):
            s = k * per_lin
            e = s + per_lin if k < n_lin - 1 else B
            g_fused_k = np.concatenate([g_r_list[k], g_l_list[k]]).astype(np.float64)
            g_t_k = torch.as_tensor(g_fused_k, dtype=dt, device=dev)
            x_offset[s:e] = (-g_t_k) @ K_inv          # broadcast (n,) → (per_lin, n)

        # Nominal g for final cost evaluation
        g_nom = torch.as_tensor(
            np.concatenate([g_r_list[0], g_l_list[0]]).astype(np.float64),
            dtype=dt, device=dev)

        # ── Warm starts ──────────────────────────────────────────────────
        x = torch.zeros(B, n, dtype=dt, device=dev)
        x_center = None
        if solver._prev_best is not None and solver._prev_n == n:
            x_center = solver._prev_best
        if x_center is not None:
            x[0] = x_center
            idx = 1
            for frac, sig in [(0.25, 0.01), (0.25, 0.10), (0.25, 0.30)]:
                cnt = max(1, int((B - 1) * frac))
                end_idx = min(idx + cnt, B)
                if idx < end_idx:
                    x[idx:end_idx] = x_center + sig * torch.randn(
                        end_idx - idx, n, dtype=dt, device=dev)
                    idx = end_idx
            if idx < B:
                x[idx:] = x_center + 0.60 * torch.randn(
                    B - idx, n, dtype=dt, device=dev)
        else:
            x = 0.20 * torch.randn(B, n, dtype=dt, device=dev)

        # ── ADMM variables ───────────────────────────────────────────────
        Cx = x @ Ct
        z = Cx.clamp(min=l_t, max=u_t)
        u_dual = torch.zeros(B, m, dtype=dt, device=dev)

        # ── ADMM iterations (shared M, per-batch x_offset) ──────────────
        for _ in range(solver.max_iter):
            x = x_offset + (z - u_dual) @ M
            Cx = x @ Ct
            x_hat = alph * Cx + (1.0 - alph) * z
            z_new = (x_hat + u_dual).clamp(min=l_t, max=u_t)
            u_dual = u_dual + x_hat - z_new
            z = z_new

        # ── Select best on NOMINAL cost ──────────────────────────────────
        Hx = x @ H_t
        cost = 0.5 * (x * Hx).sum(dim=1) + (x * g_nom).sum(dim=1)
        Cx_all = x @ Ct
        lb_v = (l_t - Cx_all).clamp(min=0)
        ub_v = (Cx_all - u_t).clamp(min=0)
        viol = lb_v.pow(2).sum(dim=1) + ub_v.pow(2).sum(dim=1)
        total = cost + 1e6 * viol

        best = torch.argmin(total)
        x_best = x[best]

        solver._prev_best = x_best.clone()
        solver._prev_n = n
        solver.last_all_costs = total.cpu().numpy()
        solver.last_best_cost = cost[best].item()
        solver.last_best_violation = viol[best].item()

        dq_r = x_best[:n_a].cpu().numpy().astype(np.float32)
        dq_l = x_best[n_a:].cpu().numpy().astype(np.float32)
        return dq_r, dq_l

    @staticmethod
    @torch.no_grad()
    def solve_pair_multi_g_all(solver: 'BatchedGPUQPSolver',
                               H_r: np.ndarray, H_l: np.ndarray,
                               g_r_list: List[np.ndarray],
                               g_l_list: List[np.ndarray],
                               C_r: np.ndarray, C_l: np.ndarray,
                               l_r: np.ndarray, u_r: np.ndarray,
                               l_l: np.ndarray, u_l: np.ndarray):
        """Solve K QPs (shared H, per-lin g) and return ALL K solutions.

        Identical to solve_pair_multi_g except it returns the best solution
        *per linearization* instead of overall best.  Used by the
        max-feasible-α method which needs to evaluate each α candidate
        individually.

        Args:
            solver:      BatchedGPUQPSolver instance.
            H_r, H_l:    (n_sub, n_sub) shared Hessians.
            g_r/l_list:  Lists of K (n_sub,) linear cost vectors.
            C_r, C_l:    (m, n_sub) shared constraint matrices.
            l/u_r/l:     (m,) shared bounds.
        Returns:
            list of K (dq_r, dq_l) tuples, one per linearization.
        """
        from scipy.linalg import block_diag as blkdiag

        n_lin = len(g_r_list)
        n_a, n_b = H_r.shape[0], H_l.shape[0]
        n = n_a + n_b
        B = solver.n_batch
        dt = solver.torch_dtype
        dev = solver.device
        rho = solver.rho
        alph = solver.alpha
        per_lin = B // n_lin

        # ── Fuse block-diagonal H, C, bounds (shared) ───────────────────
        H_fused = blkdiag(H_r, H_l).astype(np.float64)
        C_fused = blkdiag(C_r, C_l).astype(np.float64)
        l_fused = np.concatenate([l_r, l_l]).astype(np.float64)
        u_fused = np.concatenate([u_r, u_l]).astype(np.float64)

        H_t = torch.as_tensor(np.ascontiguousarray(H_fused), dtype=dt, device=dev)
        C_t = torch.as_tensor(np.ascontiguousarray(C_fused), dtype=dt, device=dev)
        l_t = torch.as_tensor(np.ascontiguousarray(l_fused), dtype=dt, device=dev).clamp(min=-1e8)
        u_t = torch.as_tensor(np.ascontiguousarray(u_fused), dtype=dt, device=dev).clamp(max=1e8)
        m = C_fused.shape[0]

        # ── Single K_inv ─────────────────────────────────────────────────
        Ct = C_t.T
        CtC = Ct @ C_t
        K = H_t + rho * CtC
        K = 0.5 * (K + K.T)
        K.diagonal().add_(solver.sigma)
        K_inv = torch.linalg.inv(K)
        M = rho * (K_inv @ C_t.T).T

        # ── Per-lin g → per-batch x_offset ───────────────────────────────
        # Also store per-lin g tensors for per-partition cost evaluation
        g_tensors = []
        x_offset = torch.empty(B, n, dtype=dt, device=dev)
        for k in range(n_lin):
            s = k * per_lin
            e = s + per_lin if k < n_lin - 1 else B
            g_fused_k = np.concatenate([g_r_list[k], g_l_list[k]]).astype(np.float64)
            g_t_k = torch.as_tensor(g_fused_k, dtype=dt, device=dev)
            g_tensors.append(g_t_k)
            x_offset[s:e] = (-g_t_k) @ K_inv

        # ── Warm starts ──────────────────────────────────────────────────
        x = torch.zeros(B, n, dtype=dt, device=dev)
        x_center = None
        if solver._prev_best is not None and solver._prev_n == n:
            x_center = solver._prev_best
        if x_center is not None:
            x[0] = x_center
            idx = 1
            for frac, sig in [(0.25, 0.01), (0.25, 0.10), (0.25, 0.30)]:
                cnt = max(1, int((B - 1) * frac))
                end_idx = min(idx + cnt, B)
                if idx < end_idx:
                    x[idx:end_idx] = x_center + sig * torch.randn(
                        end_idx - idx, n, dtype=dt, device=dev)
                    idx = end_idx
            if idx < B:
                x[idx:] = x_center + 0.60 * torch.randn(
                    B - idx, n, dtype=dt, device=dev)
        else:
            x = 0.20 * torch.randn(B, n, dtype=dt, device=dev)

        # ── ADMM loop ────────────────────────────────────────────────────
        Cx = x @ Ct
        z = Cx.clamp(min=l_t, max=u_t)
        u_dual = torch.zeros(B, m, dtype=dt, device=dev)

        for _ in range(solver.max_iter):
            x = x_offset + (z - u_dual) @ M
            Cx = x @ Ct
            x_hat = alph * Cx + (1.0 - alph) * z
            z_new = (x_hat + u_dual).clamp(min=l_t, max=u_t)
            u_dual = u_dual + x_hat - z_new
            z = z_new

        # ── Select best per partition ────────────────────────────────────
        results = []
        overall_best_cost = float('inf')
        overall_best_x = None

        for k in range(n_lin):
            s = k * per_lin
            e = s + per_lin if k < n_lin - 1 else B
            x_k = x[s:e]                     # (per_lin, n)
            g_k = g_tensors[k]

            Hx_k = x_k @ H_t
            cost_k = 0.5 * (x_k * Hx_k).sum(dim=1) + (x_k * g_k).sum(dim=1)
            Cx_k = x_k @ Ct
            lb_v = (l_t - Cx_k).clamp(min=0)
            ub_v = (Cx_k - u_t).clamp(min=0)
            viol_k = lb_v.pow(2).sum(dim=1) + ub_v.pow(2).sum(dim=1)
            total_k = cost_k + 1e6 * viol_k

            best_k = torch.argmin(total_k)
            x_best_k = x_k[best_k]

            dq_r_k = x_best_k[:n_a].cpu().numpy().astype(np.float32)
            dq_l_k = x_best_k[n_a:].cpu().numpy().astype(np.float32)
            results.append((dq_r_k, dq_l_k))

            # Track overall best for warm-start caching
            if total_k[best_k].item() < overall_best_cost:
                overall_best_cost = total_k[best_k].item()
                overall_best_x = x_best_k

        if overall_best_x is not None:
            solver._prev_best = overall_best_x.clone()
            solver._prev_n = n

        return results

    @staticmethod
    @torch.no_grad()
    def solve_pair_multi_lin(solver: 'BatchedGPUQPSolver',
                             H_r_list: List[np.ndarray],
                             H_l_list: List[np.ndarray],
                             g_r_list: List[np.ndarray],
                             g_l_list: List[np.ndarray],
                             C_r: np.ndarray, C_l: np.ndarray,
                             l_r_list: List[np.ndarray],
                             u_r_list: List[np.ndarray],
                             l_l_list: List[np.ndarray],
                             u_l_list: List[np.ndarray],
                             n_box_a: int, n_box_b: int):
        """Solve multiple weight-perturbed linearizations in a SINGLE GPU call.

        Each linearization may have different H and g (from perturbed tracking
        weights), but all share the same constraint matrix C (joint limits +
        CBF geometry don't depend on tracking weights).

        Since H differs per linearization, each has a different KKT matrix
        K_k = H_k + ρ C^T C.  We pre-compute K_inv for each linearization
        once (n_lin small inversions of 26×26 matrices — negligible), then
        run a single batched ADMM loop where each batch instance uses the
        K_inv/g/bounds corresponding to its linearization partition.

        The batch B is partitioned: B / n_lin instances per linearization.
        Best solution is selected on the nominal (index-0) QP cost.

        Args:
            solver:      BatchedGPUQPSolver instance.
            H_r/l_list:  Lists[ndarray] of length n_lin — per-lin Hessians.
            g_r/l_list:  Lists[ndarray] of length n_lin — per-lin linear costs.
            C_r, C_l:    (m, n) constraint matrices (shared across lins).
            l/u_r/l_list: Lists[ndarray] of length n_lin — per-lin bounds.
            n_box_a/b:   Number of leading box-constraint rows per sub-problem.

        Returns:
            (dq_right, dq_left): best (n_a,), (n_b,) numpy float32.
        """
        from scipy.linalg import block_diag as blkdiag

        n_lin = len(H_r_list)
        n_a, n_b = H_r_list[0].shape[0], H_l_list[0].shape[0]
        m_a, m_b = C_r.shape[0], C_l.shape[0]
        B = solver.n_batch
        dt = solver.torch_dtype
        dev = solver.device
        rho = solver.rho
        alph = solver.alpha

        n = n_a + n_b
        n_cbf_a, n_cbf_b = m_a - n_box_a, m_b - n_box_b

        # ── Fuse shared constraint matrix (block-diagonal) ───────────────
        Z = np.zeros
        C_fused = np.vstack([
            np.hstack([C_r[:n_box_a],  Z((n_box_a, n_b))]),
            np.hstack([Z((n_box_b, n_a)), C_l[:n_box_b]]),
            np.hstack([C_r[n_box_a:],  Z((n_cbf_a, n_b))]),
            np.hstack([Z((n_cbf_b, n_a)), C_l[n_box_b:]]),
        ]).astype(np.float64)
        m = C_fused.shape[0]

        C_t = torch.as_tensor(np.ascontiguousarray(C_fused), dtype=dt, device=dev)
        Ct = C_t.T
        CtC = Ct @ C_t  # (n, n)

        # ── Per-linearization: fuse H, compute K_inv, fuse g/bounds ──────
        per_lin = B // n_lin

        # Store per-lin precomputed ADMM constants
        M_list = []           # M_k = rho * (K_inv_k @ C^T)^T  — (m, n)
        x_offset_ranges = []  # (start, end, x_offset_k)

        g_batch = torch.empty(B, n, dtype=dt, device=dev)
        l_batch = torch.empty(B, m, dtype=dt, device=dev)
        u_batch = torch.empty(B, m, dtype=dt, device=dev)

        for k in range(n_lin):
            s = k * per_lin
            e = s + per_lin if k < n_lin - 1 else B

            # Fuse block-diagonal H for this linearization
            H_fused_k = blkdiag(H_r_list[k], H_l_list[k]).astype(np.float64)
            H_t_k = torch.as_tensor(np.ascontiguousarray(H_fused_k), dtype=dt, device=dev)

            # K_inv for this linearization
            K_k = H_t_k + rho * CtC
            K_k = 0.5 * (K_k + K_k.T)
            K_k.diagonal().add_(solver.sigma)
            K_inv_k = torch.linalg.inv(K_k)

            # Precomputed ADMM matrix for this lin
            M_k = rho * (K_inv_k @ C_t.T).T  # (m, n)
            M_list.append(M_k)

            # Fuse g, bounds
            g_fused_k = np.concatenate([g_r_list[k], g_l_list[k]]).astype(np.float64)
            l_fused_k = np.concatenate([
                l_r_list[k][:n_box_a], l_l_list[k][:n_box_b],
                l_r_list[k][n_box_a:], l_l_list[k][n_box_b:]
            ]).astype(np.float64)
            u_fused_k = np.concatenate([
                u_r_list[k][:n_box_a], u_l_list[k][:n_box_b],
                u_r_list[k][n_box_a:], u_l_list[k][n_box_b:]
            ]).astype(np.float64)

            g_t_k = torch.as_tensor(g_fused_k, dtype=dt, device=dev)
            g_batch[s:e] = g_t_k
            l_batch[s:e] = torch.as_tensor(l_fused_k, dtype=dt, device=dev).clamp(min=-1e8)
            u_batch[s:e] = torch.as_tensor(u_fused_k, dtype=dt, device=dev).clamp(max=1e8)

            x_off_k = (-g_t_k) @ K_inv_k  # (n,)
            x_offset_ranges.append((s, e, x_off_k))

        # ── Vectorize per-lin M and x_offset into (B, m, n) / (B, n) ───
        # This eliminates the Python for-loop inside the ADMM iteration,
        # replacing it with a single batched einsum.
        M_stacked = torch.stack(M_list, dim=0)  # (n_lin, m, n)
        # Expand to (B, m, n) by repeating each lin's M for its partition
        M_batch = torch.empty(B, m, n, dtype=dt, device=dev)
        x_offset_batch = torch.empty(B, n, dtype=dt, device=dev)
        for k in range(n_lin):
            s, e, x_off_k = x_offset_ranges[k]
            M_batch[s:e] = M_stacked[k]
            x_offset_batch[s:e] = x_off_k

        # Nominal H, g, l, u for final cost evaluation (index-0 lin)
        H_nom = torch.as_tensor(
            np.ascontiguousarray(blkdiag(H_r_list[0], H_l_list[0]).astype(np.float64)),
            dtype=dt, device=dev)
        g_nom = g_batch[0]
        l_nom = l_batch[0]
        u_nom = u_batch[0]

        # ── Warm starts ──────────────────────────────────────────────────
        x = torch.zeros(B, n, dtype=dt, device=dev)
        x_center = None
        if solver._prev_best is not None and solver._prev_n == n:
            x_center = solver._prev_best

        if x_center is not None:
            x[0] = x_center
            idx = 1
            for frac, sig in [(0.25, 0.01), (0.25, 0.10), (0.25, 0.30)]:
                cnt = max(1, int((B - 1) * frac))
                end_idx = min(idx + cnt, B)
                if idx < end_idx:
                    x[idx:end_idx] = x_center + sig * torch.randn(
                        end_idx - idx, n, dtype=dt, device=dev)
                    idx = end_idx
            if idx < B:
                x[idx:] = x_center + 0.60 * torch.randn(
                    B - idx, n, dtype=dt, device=dev)
        else:
            x = 0.20 * torch.randn(B, n, dtype=dt, device=dev)

        # ── ADMM variables ───────────────────────────────────────────────
        Cx = x @ Ct
        z = Cx.clamp(min=l_batch, max=u_batch)
        u_dual = torch.zeros(B, m, dtype=dt, device=dev)

        # ── ADMM iterations (fully vectorized — no Python for-loop) ──────
        # x-update: x[b] = x_offset[b] + (z[b] - u[b]) @ M[b]
        # Using einsum 'bm,bmn->bn' for batched (B, m) @ (B, m, n) matmul.
        for _ in range(solver.max_iter):
            zmu = z - u_dual                                    # (B, m)
            x = x_offset_batch + torch.einsum('bm,bmn->bn', zmu, M_batch)

            # z-update with over-relaxation
            Cx = x @ Ct
            x_hat = alph * Cx + (1.0 - alph) * z
            z_new = (x_hat + u_dual).clamp(min=l_batch, max=u_batch)

            # u-update
            u_dual = u_dual + x_hat - z_new
            z = z_new

        # ── Select best on NOMINAL cost ──────────────────────────────────
        Hx = x @ H_nom
        cost = 0.5 * (x * Hx).sum(dim=1) + (x * g_nom).sum(dim=1)
        Cx_all = x @ Ct
        lb_v = (l_nom - Cx_all).clamp(min=0)
        ub_v = (Cx_all - u_nom).clamp(min=0)
        viol = lb_v.pow(2).sum(dim=1) + ub_v.pow(2).sum(dim=1)
        total = cost + 1e6 * viol

        best = torch.argmin(total)
        x_best = x[best]

        solver._prev_best = x_best.clone()
        solver._prev_n = n
        solver.last_all_costs = total.cpu().numpy()
        solver.last_best_cost = cost[best].item()
        solver.last_best_violation = viol[best].item()

        dq_r = x_best[:n_a].cpu().numpy().astype(np.float32)
        dq_l = x_best[n_a:].cpu().numpy().astype(np.float32)
        return dq_r, dq_l
