import numpy as np
from typing import List
from robot_const import *
from dynamics_lib import *
from math_operations import *


class Link:
    """Link structure for robot"""
    def __init__(self, parent, dof_index, joint_type, inertia, Xtree):
        self.parent = parent
        self.dof_index = dof_index
        self.joint_type = joint_type
        self.inertia = inertia
        self.Xtree = Xtree


class Robot:
    """Robot dynamics and kinematics (single input, numpy)"""
    
    def __init__(self):
        self.links = []
        
        ## themis
        parent = [-1,  0,  1,  2,  3,  4, 
                   5,  6,  7,  8,  9, 10,
                   5, 12, 13, 14, 15, 16,
                   5, 18, 19, 20, 21, 22, 23,
                   5, 25, 26, 27, 28, 29, 30,
                   5, 32]
        
        joint_types = ["Px", "Py", "Pz", "Rx", "Ry", "Rz",
                       "Rz", "Rx", "Ry", "Ry", "Ry", "Rx",
                       "Rz", "Rx", "Ry", "Ry", "Ry", "Rx",
                       "Ry", "Rx", "Rz", "Ry", "Rx", "Ry", "Rx",
                       "Ry", "Rx", "Rz", "Ry", "Rx", "Ry", "Rx",
                       "Rz", "Ry"]
        
        # Use robot constants (already numpy arrays)
        self.mass = mass
        self.r = r
        self.com = com
        self.I_diag = I_diag
        
        # Float-base DOFs (no mass)
        for i in range(5):
            r_zero = np.zeros(3, dtype=np.float32)
            inertia = np.zeros((6, 6), dtype=np.float32)
            Xtree = Xtrans(r_zero)
            self.links.append(Link(parent[i], i, joint_types[i], inertia, Xtree))
        
        # Base link (pelvis)
        I_com_0 = np.diag(self.I_diag[0])
        inertia_5 = McI(self.mass[0], self.com[0], I_com_0)
        Xtree_5 = Xtrans(self.r[0])
        self.links.append(Link(parent[5], 5, joint_types[5], inertia_5, Xtree_5))
        
        # Joint links
        for i in range(6, DOF):
            I_com_i = np.diag(self.I_diag[i-5])
            inertia_i = McI(self.mass[i-5], self.com[i-5], I_com_i)
            Xtree_i = Xtrans(self.r[i-5])
            
            # Add joint rotation offset
            if LROT_type[i-5] != "None":
                JRot = compute_spatial_rotm(LROT_type[i-5], -LRot[i-5])
                Xtree_i = JRot @ Xtree_i
            
            self.links.append(Link(parent[i], i, joint_types[i], inertia_i, Xtree_i))
        
        # State vectors
        self.q = np.zeros(DOF, dtype=np.float32)
        self.dq = np.zeros(DOF, dtype=np.float32)
        
        # OSQP solver for warm starting (initialized on first call)
        self._osqp_solver = None
        self._osqp_prev_dq = np.zeros(DOF, dtype=np.float64)  # previous solution for warm start
        
        # Distributed QP warm start (13 DOFs each: 6 base + 7 arm)
        self._osqp_prev_dq_right = np.zeros(13, dtype=np.float64)
        self._osqp_prev_dq_left = np.zeros(13, dtype=np.float64)
        
        # ProxQP solver (proximal augmented Lagrangian, warm-startable)
        self._proxqp_solver = None
        self._proxqp_prev_dq = np.zeros(DOF, dtype=np.float64)
        
        # GPU batched QP solver (PyTorch ADMM, lazily initialized)
        self._gpu_qp_solver = None
        self._gpu_qp_solver_right = None  # distributed: right arm
        self._gpu_qp_solver_left = None   # distributed: left arm
        
        # Ratchet: when desired targets are nearly static, only accept IK
        # solutions that reduce tracking error vs the previous accepted solution.
        self._ik_ratchet_prev_des = None     # (12,) previous desired EE XYZ
        self._ik_ratchet_prev_cost = np.inf  # previous accepted tracking cost
        self._ik_ratchet_q_des = None        # previous accepted q_des (to hold on rejection)
        self._ik_ratchet_dq_des = None       # previous accepted dq_des
    
    def update(self, q, dq):
        """Update robot state
        Args:
            q: (DOF,)
            dq: (DOF,)
        """
        self.q = q.astype(np.float32)
        self.dq = dq.astype(np.float32)
    
    def compute_hand_C(self):
        """Compute mass matrix H and Coriolis/gravity vector C
        Returns:
            C: (DOF,)
            H: (DOF, DOF)
        """
        a_grav = np.zeros(6, dtype=np.float32)
        a_grav[5] = GRAVITY
        
        Xup = []
        S = []
        v = []
        avp = []
        fvp = []
        
        # Forward pass
        for i in range(NUM_LINKS):
            XJ, Si = jcalc(self.links[i].joint_type, self.q[self.links[i].dof_index])
            S.append(Si)
            vJ = Si * self.dq[self.links[i].dof_index]
            
            Xup_i = XJ @ self.links[i].Xtree
            Xup.append(Xup_i)
            
            if self.links[i].parent == -1:
                v.append(vJ)
                avp_i = apply_transform(Xup_i, a_grav)
                avp.append(-avp_i)
            else:
                p = self.links[i].parent
                v.append(apply_transform(Xup_i, v[p]) + vJ)
                avp.append(apply_transform(Xup_i, avp[p]) + 
                         apply_transform(crm(v[i]), vJ))
            
            fvp.append(self.links[i].inertia @ avp[i] + 
                      apply_transform(crf(v[i]), self.links[i].inertia @ v[i]))
        
        # Backward pass for C
        C = np.zeros(DOF, dtype=np.float32)
        for i in range(NUM_LINKS-1, -1, -1):
            C[self.links[i].dof_index] = np.dot(S[i], fvp[i])
            if self.links[i].parent != -1:
                p = self.links[i].parent
                fvp[p] += apply_transpose_transform(Xup[i], fvp[i])
        
        # Compute H using composite rigid body algorithm
        I_c = [link.inertia.copy() for link in self.links]
        for i in range(DOF-1, -1, -1):
            if self.links[i].parent != -1:
                p = self.links[i].parent
                I_c[p] += AtBA(Xup[i], I_c[i])
        
        H = np.zeros((DOF, DOF), dtype=np.float32)
        for i in range(DOF):
            fh = I_c[i] @ S[i]
            H[i, i] = np.dot(S[i], fh)
            
            j = i
            while self.links[j].parent != -1:
                fh = apply_transpose_transform(Xup[j], fh)
                j = self.links[j].parent
                H[i, j] = np.dot(S[j], fh)
                H[j, i] = H[i, j]
        
        return C, H
    
    def compute_body_jacobian(self, body_index, Xend):
        """Compute body Jacobian
        Args:
            body_index: int - link index
            Xend: (6, 6) - end-effector offset transform
        Returns:
            J: (6, DOF)
        """
        J = np.zeros((6, DOF), dtype=np.float32)
        X = Xend.copy()
        j = body_index
        
        while j >= 0:
            XJ, S = jcalc(self.links[j].joint_type, self.q[self.links[j].dof_index])
            col = apply_transform(X, S)
            J[:, self.links[j].dof_index] = col
            
            if self.links[j].parent >= 0:
                X = X @ XJ @ self.links[j].Xtree
            
            j = self.links[j].parent 
        
        return J
    
    def compute_forward_kinematics(self, end_effector_index, contact_offset):
        """Compute forward kinematics to end effector
        Args:
            end_effector_index: int
            contact_offset: (3,)
        Returns:
            rpypos: (6,) - [roll, pitch, yaw, x, y, z]
            J: (6, DOF) - Jacobian
        """
        X = np.eye(4, dtype=np.float32)
        
        current_parent = self.links[end_effector_index].parent + 1
        current = end_effector_index
        
        while current_parent != -1:
            link_parent = self.links[current_parent]
            link = self.links[current]
            Xj = joint_transform(link_parent.joint_type, self.q[link.dof_index])
            Xtree = spatial_to_isometry(link.Xtree)
            X = Xtree @ Xj @ X
            current_parent = link_parent.parent
            current = current_parent
        
        # Apply contact offset
        X_offset = np.eye(4, dtype=np.float32)
        X_offset[:3, 3] = contact_offset
        X = X @ X_offset
        
        # Extract position and orientation
        rpypos = np.zeros(6, dtype=np.float32)
        rpypos[3:] = X[:3, 3]
        
        R = X[:3, :3]
        roll, pitch, yaw = rpy_from_rot_zyx(R)
        rpypos[0] = roll
        rpypos[1] = pitch
        rpypos[2] = yaw
        
        # Compute Jacobian
        Xend = Xtrans(contact_offset)
        # J = self.compute_body_jacobian(end_effector_index, Xend)
        
        return rpypos
    
    def update_task_space_command_with_constraints(self, 
                                                   x_elbow_l_des, x_elbow_r_des,
                                                   x_elbow_l, x_elbow_r,
                                                   x_hand_l_des, x_hand_r_des,
                                                   x_hand_l, x_hand_r,
                                                   J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                                   com_des):
        """Solve weighted IK with joint limit constraints (numpy version)
        Args:
            All (6,) for positions, (6, DOF) for Jacobians
        Returns:
            q_des: (DOF,)
            dq_des: (DOF,)
        """
        # Weights
        We = np.eye(6, dtype=np.float32)*10 #  elbow control
        We[:3, :3] = 0  # don't care about elbow orientation
        We[3, 3] *= 100  # elbow x
        We[4, 4] *= 100  # elbow y
        We[5, 5] *= 100  # elbow z
        
        Wq = np.eye(DOF, dtype=np.float32) * 20  # tracking control
        Wq[:6, :6] = 0
        # Wq[21, 21] *= 10
        # Wq[23, 23] *= 10
        # Wq[28, 28] *= 10
        # Wq[30, 30] *= 10

        Wq_ref = np.eye(DOF, dtype=np.float32)*10   # joint error control
        Wq_ref[:6, :6] = 0
        # Wq_ref[21, 21] *= 10
        # Wq_ref[23, 23] *= 10
        # Wq_ref[28, 28] *= 10
        # Wq_ref[30, 30] *= 10
        # Use current pose as reference to minimize incremental change
        q_ref = self.q.copy()

        Wc = np.eye(6, dtype=np.float32) * 50000   # COM control
        
        Wh = np.eye(6, dtype=np.float32)*10   # hand control
        Wh[:3, :3] = 0  # don't care about hand orientation
        Wh[3, 3] *= 500   # hand x
        Wh[4, 4] *= 500  # hand y
        Wh[5, 5] *= 500  # hand z
        
        # COM Jacobian (6 x DOF)
        J_com = np.zeros((6, DOF), dtype=np.float32)
        J_com[:, :6] = np.eye(6, dtype=np.float32)
        
        # Joint limit constraints
        qj_min = q_min
        qj_max = q_max
        qj_mid = (qj_min + qj_max) / 2
        D = np.diag(1.0 / (qj_max - qj_min))
        lambda_jl = 3
        
        # Task errors
        e_q = self.q - q_ref
        e_elbow_l = x_elbow_l_des - x_elbow_l
        e_elbow_r = x_elbow_r_des - x_elbow_r
        e_hand_l = x_hand_l_des - x_hand_l
        e_hand_r = x_hand_r_des - x_hand_r
        e_com = com_des - self.q[:6]

        # print("e_hand_l:", e_hand_l)
        # print("e_hand_r:", e_hand_r) 
        
        # Build quadratic program matrix
        A = lambda_jl * (D.T @ D) + Wq
        # A += Wq + Wq_ref
        A += J_elbow_l.T @ We @ J_elbow_l
        A += J_elbow_r.T @ We @ J_elbow_r
        A += J_com.T @ Wc @ J_com
        A += J_hand_l.T @ Wh @ J_hand_l
        A += J_hand_r.T @ Wh @ J_hand_r
        
        b = -lambda_jl * (D.T @ D @ (self.q - qj_mid))
        # b += Wq_ref @ -e_q
        b += J_elbow_l.T @ We @ e_elbow_l
        b += J_elbow_r.T @ We @ e_elbow_r
        b += J_com.T @ Wc @ e_com
        b += J_hand_l.T @ Wh @ e_hand_l
        b += J_hand_r.T @ Wh @ e_hand_r
        
        # Solve
        dq_sol = np.linalg.solve(A, b)
        q_des = self.q + dq_sol
        # q_des = self.q + np.linalg.inv(A) @ b
        dq_des = np.zeros_like(self.dq)
        
        return q_des, dq_des



    def update_task_space_command_qp_distributed(self, 
                                                  x_elbow_l_des, x_elbow_r_des,
                                                  x_elbow_l, x_elbow_r,
                                                  x_hand_l_des, x_hand_r_des,
                                                  x_hand_l, x_hand_r,
                                                  J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                                  com_des):
        """Distributed IK: solve separate QP for each arm to avoid cross-arm interference.
        
        Solves two independent 13-DOF QPs (base + each arm) to decouple arm tracking.
        """
        try:
            import osqp
            from scipy import sparse
        except ImportError:
            return self.update_task_space_command_qp(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des)
        
        BASE_DOFS = np.arange(0, 6)
        RIGHT_ARM_DOFS = np.arange(18, 25)
        LEFT_ARM_DOFS = np.arange(25, 32)
        
        right_dofs = np.concatenate([BASE_DOFS, RIGHT_ARM_DOFS])
        left_dofs = np.concatenate([BASE_DOFS, LEFT_ARM_DOFS])
        
        # Weights: right arm gets primary COM control, left arm gets secondary
        # We = 0*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        # Wh = 10*np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)
        We = .01*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1*np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc_primary = np.eye(6, dtype=np.float64) * 5000
        Wc_secondary = np.eye(6, dtype=np.float64) * 5000
        Wq_diag = np.ones(DOF, dtype=np.float64) * 10
        # Wq_diag[:6] = 0
        
        # Current state
        q_fb = self.q.astype(np.float64)
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        J_com_full = np.zeros((6, DOF), dtype=np.float64)
        J_com_full[:, :6] = np.eye(6, dtype=np.float64)
        
        J_elbow_l, J_elbow_r = J_elbow_l.astype(np.float64), J_elbow_r.astype(np.float64)
        J_hand_l, J_hand_r = J_hand_l.astype(np.float64), J_hand_r.astype(np.float64)
        
        qj_min, qj_max = q_min.astype(np.float64), q_max.astype(np.float64)
        
        # CBF setup
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos, crotch_pos = q_fb[:3] + head_offset, q_fb[:3] + crotch_offset
        
        r_torso, r_head, r_crotch = 0.18, 0.11, 0.16
        rho_torso = r_torso + 0.02
        rho_head = r_head + 0.02
        rho_crotch = r_crotch + 0.02
        lambda_cbf = 0.5
        
        def _solve_arm_qp(dof_indices, J_elbow, J_hand, e_elbow, e_hand,
                          x_elbow, x_hand, prev_dq, side_name, Wc):
            n = len(dof_indices)
            J_elbow_sub = J_elbow[:, dof_indices]
            J_hand_sub = J_hand[:, dof_indices]
            J_com_sub = J_com_full[:, dof_indices]
            Wq_sub = np.diag(Wq_diag[dof_indices])
            
            P = (J_elbow_sub.T @ We @ J_elbow_sub +
                 J_hand_sub.T @ Wh @ J_hand_sub +
                 J_com_sub.T @ Wc @ J_com_sub +
                 Wq_sub)
            P = 0.5 * (P + P.T)
            P += np.eye(n) * 1e-6
            
            c = -(J_elbow_sub.T @ We @ e_elbow +
                  J_hand_sub.T @ Wh @ e_hand +
                  J_com_sub.T @ Wc @ e_com)
            
            # CBF distances
            x_elbow_pos = x_elbow[3:].astype(np.float64)
            x_hand_pos = x_hand[3:].astype(np.float64)
            
            h_values = np.array([
                np.linalg.norm(x_elbow_pos - q_fb[:3])**2 - rho_torso**2,
                np.linalg.norm(x_hand_pos - q_fb[:3])**2 - rho_torso**2,
                np.linalg.norm(x_elbow_pos - head_pos)**2 - rho_head**2,
                np.linalg.norm(x_hand_pos - head_pos)**2 - rho_head**2,
                np.linalg.norm(x_elbow_pos - crotch_pos)**2 - rho_crotch**2,
                np.linalg.norm(x_hand_pos - crotch_pos)**2 - rho_crotch**2,
            ], dtype=np.float64)
            
            J_elbow_lin = J_elbow[3:, dof_indices]
            J_hand_lin = J_hand[3:, dof_indices]
            J_com_lin = J_com_full[:3, dof_indices]
            
            d_vectors = [
                x_elbow_pos - q_fb[:3], x_hand_pos - q_fb[:3],
                x_elbow_pos - head_pos, x_hand_pos - head_pos,
                x_elbow_pos - crotch_pos, x_hand_pos - crotch_pos
            ]
            J_refs = [J_elbow_lin, J_hand_lin, J_elbow_lin, J_hand_lin, J_elbow_lin, J_hand_lin]
            
            aT_rows = [2 * d @ (J_ref - J_com_lin) for d, J_ref in zip(d_vectors, J_refs)]
            aT_cbf = np.vstack(aT_rows)
            b_cbf = -lambda_cbf * h_values
            
            lb_box = qj_min[dof_indices] - q_fb[dof_indices]
            ub_box = qj_max[dof_indices] - q_fb[dof_indices]
            
            A_combined = sparse.vstack([sparse.eye(n, format='csc'), sparse.csc_matrix(aT_cbf)])
            lb_combined = np.hstack([lb_box, b_cbf])
            ub_combined = np.hstack([ub_box, np.inf * np.ones(6)])
            
            solver = osqp.OSQP()
            import sys, os
            devnull = open(os.devnull, 'w')
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                solver.setup(sparse.csc_matrix(P), c, A_combined, lb_combined, ub_combined,
                            verbose=False, polish=True, eps_abs=1e-5, eps_rel=1e-5, max_iter=500, warm_start=True)
                solver.warm_start(x=prev_dq)
                result = solver.solve()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                devnull.close()
            
            if result.info.status in ('solved', 'solved_inaccurate'):
                return result.x
            else:
                print(f"[DistQP] {side_name}: {result.info.status}, unconstrained fallback")
                try:
                    return np.linalg.solve(P, -c)
                except:
                    return np.zeros(n)
        
        # Solve both QPs
        dq_right = _solve_arm_qp(right_dofs, J_elbow_r, J_hand_r, e_elbow_r, e_hand_r,
                                 x_elbow_r, x_hand_r, self._osqp_prev_dq_right, 'right', Wc_primary)
        self._osqp_prev_dq_right = dq_right.copy()
        
        dq_left = _solve_arm_qp(left_dofs, J_elbow_l, J_hand_l, e_elbow_l, e_hand_l,
                                x_elbow_l, x_hand_l, self._osqp_prev_dq_left, 'left', Wc_secondary)
        self._osqp_prev_dq_left = dq_left.copy()
        
        # Combine: right arm's base takes priority
        dq_sol = np.zeros(DOF, dtype=np.float32)
        dq_sol[:6] = dq_right[:6].astype(np.float32)
        dq_sol[18:25] = dq_right[6:].astype(np.float32)
        dq_sol[25:32] = dq_left[6:].astype(np.float32)
        
        q_des = self.q + dq_sol
        dq_des = dq_sol
        
        return q_des, dq_des

    # -----------------------------------------------------------------
    # Lyapunov Progress Certificate variant
    # -----------------------------------------------------------------
    def update_task_space_command_qp_gpu_batch_distributed_alpha_lyapunov(
            self,
            x_elbow_l_des, x_elbow_r_des,
            x_elbow_l, x_elbow_r,
            x_hand_l_des, x_hand_r_des,
            x_hand_l, x_hand_r,
            J_elbow_l, J_elbow_r,
            J_hand_l, J_hand_r,
            com_des,
            n_batch=4096, max_iter=50,
            pos_threshold=0.005,
            q_ref=None,
            w_ref=0.0,
            n_alpha=8,
            eta=0.0005,
            eps_q=0.01,
            eps_V=0.001,
            dq_max=0.5):
        """Max-feasible-α distributed GPU-batched IK-QP with Lyapunov
        progress certificate selection.

        Explores intermediate targets along the line from current EE
        position to desired target:

            x_d^{(k)} = (1 - α_k) * x(q) + α_k * x_d,   0 < α_1 < … < α_K = 1

        For each α_k a QP is solved (shared linearisation, different RHS).
        Candidate selection uses a Lyapunov-like objective instead of the
        simple norm-based progress check:

            V(q)   = ½ e(q)ᵀ W e(q)
            ê⁺(Δq) = e(q) − J(q) Δq
            V̂⁺(Δq) = ½ ê⁺(Δq)ᵀ W ê⁺(Δq)

        A candidate α_j is **accepted** if it is feasible AND:
            (a)  V̂⁺(Δq(α_j))  ≤  V(q) − η          [Lyapunov decrease]
            (b)  ‖Δq(α_j)‖ ≥ ε_q  when V(q) > ε_V   [escape condition]

        Among accepted candidates the **largest** α is selected, yielding
        the most aggressive (closest-to-target) continuation update that
        still certifies progress toward the real nonlinear IK objective.

        Args:
            x_{elbow,hand}_{l,r}_des: (6,) desired task-space poses [rpy, xyz]
            x_{elbow,hand}_{l,r}: (6,) current task-space poses
            J_{elbow,hand}_{l,r}: (6, DOF) task Jacobians
            com_des: (6,) desired COM/base pose
            n_batch: number of parallel QP instances (default 4096)
            max_iter: ADMM iterations per solve (default 50)
            pos_threshold: ratchet threshold (default 0.005 m)
            q_ref: (DOF,) optional reference pose
            w_ref: weight on reference pose tracking
            n_alpha: number of α levels to try (default 8)
            eta: Lyapunov decrease margin η > 0 (default 0.0005)
            eps_q: minimum step norm ε_q for escape condition (default 0.01)
            eps_V: Lyapunov threshold ε_V above which escape is required
                   (default 0.001)
            dq_max: max step norm (rad) for feasibility (default 0.5)

        Returns:
            q_des: (DOF,) desired joint configuration
            dq_des: (DOF,) joint velocity
        """
        try:
            from gpu_qp_solver import BatchedGPUQPSolver, TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
        except ImportError:
            print("Warning: gpu_qp_solver/PyTorch not available. "
                  "Falling back to distributed ProxQP.")
            return self.update_task_space_command_qp_distributed_proxqp(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des)

        # =================================================================
        # Lazily create GPU solver
        # =================================================================
        if (self._gpu_qp_solver_right is None
                or self._gpu_qp_solver_right.n_batch != n_batch
                or self._gpu_qp_solver_right.max_iter != max_iter):
            self._gpu_qp_solver_right = BatchedGPUQPSolver(
                n_batch=n_batch,
                max_iter=max_iter,
                rho=50.0,
                alpha=1.6,
                sigma=1e-6,
                device=None,
                dtype='float32'
            )

        # =================================================================
        # DOF partitioning
        # =================================================================
        BASE_DOFS = np.arange(0, 6)
        RIGHT_ARM_DOFS = np.arange(18, 25)
        LEFT_ARM_DOFS = np.arange(25, 32)
        right_dofs = np.concatenate([BASE_DOFS, RIGHT_ARM_DOFS])  # 13
        left_dofs = np.concatenate([BASE_DOFS, LEFT_ARM_DOFS])    # 13
        n_sub = 13

        # =================================================================
        # Weights
        # =================================================================
        We = 0.01 * np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1.0 * np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc = np.eye(6, dtype=np.float64) * 5000
        Wq_val = 10.0

        # =================================================================
        # Current state
        # =================================================================
        q_fb = self.q.astype(np.float64)
        J_com_full = np.zeros((6, DOF), dtype=np.float64)
        J_com_full[:, :6] = np.eye(6, dtype=np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)

        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)

        qj_min = q_min.astype(np.float64)
        qj_max = q_max.astype(np.float64)

        # CBF setup
        com_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float64)
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos = q_fb[:3] + head_offset
        crotch_pos = q_fb[:3] + crotch_offset

        r_torso, r_head, r_crotch = 0.18, 0.11, 0.16
        safety_margin = 0.02
        rho_torso = r_torso + safety_margin
        rho_head = r_head + safety_margin
        rho_crotch = r_crotch + safety_margin
        lambda_cbf = 0.5

        # Reference pose
        q_ref_r = None
        q_ref_l = None
        if q_ref is not None:
            q_ref = np.asarray(q_ref, dtype=np.float64)
            if q_ref.shape[0] == DOF:
                q_ref_r = q_ref[right_dofs]
                q_ref_l = q_ref[left_dofs]

        # =================================================================
        # Build QP helper (closure over shared state)
        # =================================================================
        def _build_arm_qp_alpha(dof_indices, J_elbow, J_hand, e_elbow, e_hand,
                                x_elbow, x_hand, Wc_arm, q_ref_arm=None,
                                w_ref_arm=0.0):
            """Build QP matrices for one arm with given tracking errors."""
            J_elbow_sub = J_elbow[:, dof_indices]
            J_hand_sub = J_hand[:, dof_indices]
            J_com_sub = J_com_full[:, dof_indices]
            Wq_sub = np.eye(n_sub, dtype=np.float64) * Wq_val

            Wq_with_ref = Wq_sub.copy()
            if w_ref_arm > 0 and q_ref_arm is not None:
                Wq_with_ref += np.eye(n_sub, dtype=np.float64) * w_ref_arm

            # Hessian (shared across all α)
            H = (J_elbow_sub.T @ We @ J_elbow_sub +
                 J_hand_sub.T @ Wh @ J_hand_sub +
                 J_com_sub.T @ Wc_arm @ J_com_sub +
                 Wq_with_ref)
            H = 0.5 * (H + H.T)
            H += np.eye(n_sub) * 1e-6

            # Linear term
            g = -(J_elbow_sub.T @ We @ e_elbow +
                  J_hand_sub.T @ Wh @ e_hand +
                  J_com_sub.T @ Wc_arm @ e_com)

            if w_ref_arm > 0 and q_ref_arm is not None:
                q_fb_arm = q_fb[dof_indices]
                q_ref_feedback = q_ref_arm.astype(np.float64)
                g += -w_ref_arm * q_ref_feedback + w_ref_arm * q_fb_arm

            # CBF constraints
            x_elbow_pos = x_elbow[3:].astype(np.float64)
            x_hand_pos = x_hand[3:].astype(np.float64)

            d_elbow_com = x_elbow_pos - q_fb[:3] - com_offset
            d_hand_com = x_hand_pos - q_fb[:3] - com_offset
            d_elbow_head = x_elbow_pos - head_pos
            d_hand_head = x_hand_pos - head_pos
            d_elbow_crotch = x_elbow_pos - crotch_pos
            d_hand_crotch = x_hand_pos - crotch_pos

            h_vals = np.array([
                np.dot(d_elbow_com, d_elbow_com) - rho_torso**2,
                np.dot(d_hand_com, d_hand_com) - rho_torso**2,
                np.dot(d_elbow_head, d_elbow_head) - rho_head**2,
                np.dot(d_hand_head, d_hand_head) - rho_head**2,
                np.dot(d_elbow_crotch, d_elbow_crotch) - rho_crotch**2,
                np.dot(d_hand_crotch, d_hand_crotch) - rho_crotch**2,
            ], dtype=np.float64)

            J_elbow_lin = J_elbow[3:, dof_indices]
            J_hand_lin = J_hand[3:, dof_indices]
            J_com_lin = J_com_full[:3, dof_indices]

            d_list = [d_elbow_com, d_hand_com, d_elbow_head,
                      d_hand_head, d_elbow_crotch, d_hand_crotch]
            J_refs = [J_elbow_lin, J_hand_lin, J_elbow_lin,
                      J_hand_lin, J_elbow_lin, J_hand_lin]

            aT_rows = [2.0 * d @ (J_ref - J_com_lin)
                       for d, J_ref in zip(d_list, J_refs)]
            aT_cbf = np.vstack(aT_rows)
            b_cbf = -lambda_cbf * h_vals

            lb_box = qj_min[dof_indices] - q_fb[dof_indices]
            ub_box = qj_max[dof_indices] - q_fb[dof_indices]

            n_cbf = 6
            C_qp = np.vstack([np.eye(n_sub, dtype=np.float64), aT_cbf])
            l_qp = np.concatenate([lb_box, b_cbf])
            u_qp = np.concatenate([ub_box, np.full(n_cbf, np.inf)])

            return H, g, C_qp, l_qp, u_qp

        # =================================================================
        # Generate α schedule: [α_1, ..., α_K] with α_K = 1
        # =================================================================
        alphas = np.linspace(1.0 / n_alpha, 1.0, n_alpha)

        # Final target errors (for Lyapunov evaluation)
        e_elbow_l_final = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r_final = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l_final = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r_final = (x_hand_r_des - x_hand_r).astype(np.float64)

        # =================================================================
        # Lyapunov value at current state: V(q) = ½ eᵀ W e
        # Weight matrix W is block-diagonal matching We (elbow) and Wh
        # (hand) for each arm.
        # =================================================================
        V_current = 0.5 * (
            e_elbow_r_final @ We @ e_elbow_r_final +
            e_hand_r_final  @ Wh @ e_hand_r_final +
            e_elbow_l_final @ We @ e_elbow_l_final +
            e_hand_l_final  @ Wh @ e_hand_l_final
        )

        # =================================================================
        # Build QP once (α=1) to get shared H, C, l, u
        # =================================================================
        H_r, g_r_full, C_r, l_r, u_r = _build_arm_qp_alpha(
            right_dofs, J_elbow_r, J_hand_r, e_elbow_r_final, e_hand_r_final,
            x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref)
        H_l, g_l_full, C_l, l_l, u_l = _build_arm_qp_alpha(
            left_dofs, J_elbow_l, J_hand_l, e_elbow_l_final, e_hand_l_final,
            x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref)

        # Store Jacobian subsets for Lyapunov predicted-error computation
        J_elbow_r_sub = J_elbow_r[:, right_dofs]
        J_hand_r_sub  = J_hand_r[:, right_dofs]
        J_elbow_l_sub = J_elbow_l[:, left_dofs]
        J_hand_l_sub  = J_hand_l[:, left_dofs]

        # =================================================================
        # Compute g for each α analytically  (g is linear in α)
        # g(α) = α * g_tracking + g_const
        # =================================================================
        _, g_r_const, _, _, _ = _build_arm_qp_alpha(
            right_dofs, J_elbow_r, J_hand_r,
            np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64),
            x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref)
        _, g_l_const, _, _, _ = _build_arm_qp_alpha(
            left_dofs, J_elbow_l, J_hand_l,
            np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64),
            x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref)

        g_r_track = g_r_full - g_r_const
        g_l_track = g_l_full - g_l_const

        g_r_list = [alpha * g_r_track + g_r_const for alpha in alphas]
        g_l_list = [alpha * g_l_track + g_l_const for alpha in alphas]

        # =================================================================
        # Solve ALL α-QPs in a SINGLE GPU call
        # =================================================================
        fused_solver = self._gpu_qp_solver_right

        all_solutions = BatchedGPUQPSolver.solve_pair_multi_g_all(
            fused_solver,
            H_r, H_l,
            g_r_list, g_l_list,
            C_r, C_l,
            l_r, u_r, l_l, u_l)

        # =================================================================
        # Lyapunov progress certificate selection
        # =================================================================
        best_k = -1

        for k in range(n_alpha - 1, -1, -1):  # largest α first
            dq_r_k, dq_l_k = all_solutions[k]

            # ------ Feasibility check 1: step limit ------ #
            dq_r_norm = np.linalg.norm(dq_r_k)
            dq_l_norm = np.linalg.norm(dq_l_k)
            if dq_r_norm > dq_max or dq_l_norm > dq_max:
                continue

            # ------ Feasibility check 2: joint limits ------ #
            q_new_r = q_fb[right_dofs] + dq_r_k
            q_new_l = q_fb[left_dofs] + dq_l_k
            if (np.any(q_new_r < qj_min[right_dofs] - 1e-4)
                    or np.any(q_new_r > qj_max[right_dofs] + 1e-4)):
                continue
            if (np.any(q_new_l < qj_min[left_dofs] - 1e-4)
                    or np.any(q_new_l > qj_max[left_dofs] + 1e-4)):
                continue

            # ------ Feasibility check 3: CBF slack ------ #
            Cx_r = C_r @ dq_r_k
            Cx_l = C_l @ dq_l_k
            cbf_viol_r = np.maximum(l_r - Cx_r, 0).sum()
            cbf_viol_l = np.maximum(l_l - Cx_l, 0).sum()
            if cbf_viol_r > 1e-3 or cbf_viol_l > 1e-3:
                continue

            # ------ Predicted next-step error (first-order) ------ #
            # ê⁺ = e − J Δq  (per EE, using final-target errors)
            e_pred_er = e_elbow_r_final - J_elbow_r_sub @ dq_r_k
            e_pred_hr = e_hand_r_final  - J_hand_r_sub  @ dq_r_k
            e_pred_el = e_elbow_l_final - J_elbow_l_sub @ dq_l_k
            e_pred_hl = e_hand_l_final  - J_hand_l_sub  @ dq_l_k

            # ------ Predicted Lyapunov value: V̂⁺ = ½ ê⁺ᵀ W ê⁺ ------ #
            V_pred = 0.5 * (
                e_pred_er @ We @ e_pred_er +
                e_pred_hr @ Wh @ e_pred_hr +
                e_pred_el @ We @ e_pred_el +
                e_pred_hl @ Wh @ e_pred_hl
            )

            # ------ Lyapunov decrease condition (eq. 7) ------ #
            eta =  0.01
            if V_pred > V_current - eta:
                continue

            # ------ Escape condition ------ #
            # When far from goal (V > ε_V), require a non-trivial step
            # ‖Δq‖ ≥ ε_q to avoid getting stuck at intermediate basins.
            dq_full_norm = np.sqrt(dq_r_norm**2 + dq_l_norm**2)
            if V_current > eps_V and dq_full_norm < eps_q:
                continue

            # All checks passed — accept this (largest feasible α)
            best_k = k
            break

        # Fallback: if no α satisfies Lyapunov certificate, use smallest α
        if best_k < 0:
            best_k = 0

        dq_right, dq_left = all_solutions[best_k]

        # =================================================================
        # Combine
        # =================================================================
        dq_sol = np.zeros(DOF, dtype=np.float32)
        dq_sol[:6] = dq_right[:6].astype(np.float32)
        dq_sol[18:25] = dq_right[6:].astype(np.float32)
        dq_sol[25:32] = dq_left[6:].astype(np.float32)

        # Ratchet
        if pos_threshold > 0:
            current_cost = (np.sum((x_elbow_l_des[3:] - x_elbow_l[3:])**2) +
                            np.sum((x_elbow_r_des[3:] - x_elbow_r[3:])**2) +
                            np.sum((x_hand_l_des[3:] - x_hand_l[3:])**2) +
                            np.sum((x_hand_r_des[3:] - x_hand_r[3:])**2))

            des_pos = np.concatenate([
                x_elbow_l_des[3:], x_elbow_r_des[3:],
                x_hand_l_des[3:], x_hand_r_des[3:]
            ]).astype(np.float64)

            target_moved = True
            if self._ik_ratchet_prev_des is not None:
                max_shift = np.max(np.abs(des_pos - self._ik_ratchet_prev_des))
                target_moved = (max_shift > pos_threshold)

            if target_moved or current_cost < self._ik_ratchet_prev_cost:
                self._ik_ratchet_prev_des = des_pos
                self._ik_ratchet_prev_cost = current_cost
            else:
                if (self._ik_ratchet_q_des is not None
                        and self._ik_ratchet_dq_des is not None):
                    return self._ik_ratchet_q_des, self._ik_ratchet_dq_des

        q_des = self.q + dq_sol
        dq_des = dq_sol

        if not np.all(np.isfinite(q_des)) or not np.all(np.isfinite(dq_des)):
            print("WARNING: Final q_des/dq_des contains NaN/Inf! "
                  "Returning current state.")
            return self.q, np.zeros(DOF, dtype=np.float32)

        self._ik_ratchet_q_des = q_des
        self._ik_ratchet_dq_des = dq_des

        return q_des, dq_des
