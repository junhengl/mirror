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
        
        # Parent indices
        ## g1
        # parent = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        #           5, 12, 13, 14, 15, 16,
        #           5, 18, 19,
        #           20, 21, 22, 23, 24, 25, 26,
        #           20, 28, 29, 30, 31, 32, 33]
        
        # joint_types = ["Px", "Py", "Pz", "Rx", "Ry", "Rz",
        #                "Ry", "Rx", "Rz", "Ry", "Ry", "Rx",
        #                "Ry", "Rx", "Rz", "Ry", "Ry", "Rx",
        #                "Rz", "Rx", "Ry",
        #                "Ry", "Rx", "Rz", "Ry", "Rx", "Ry", "Rz",
        #                "Ry", "Rx", "Rz", "Ry", "Rx", "Ry", "Rz"]

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

    def update_task_space_command_qp(self, 
                                     x_elbow_l_des, x_elbow_r_des,
                                     x_elbow_l, x_elbow_r,
                                     x_hand_l_des, x_hand_r_des,
                                     x_hand_l, x_hand_r,
                                     J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                     com_des):
        """Solve IK as a Quadratic Programming problem with hard joint limit constraints.
        
        Uses OSQP solver (fast, designed for real-time robotics).
        
        Formulation:
            Decision variable: dq = q_des - q_fb (joint displacement)
            
            Cost 1 (task space tracking): ||J * dq - e||^2_W  where e = x_des - x
            Cost 2 (joint regularization): ||dq||^2_Wq
            
            Total: min 0.5 * dq^T @ P @ dq + c^T @ dq
                   where P = J^T W J + Wq
                         c = -J^T W e
            
            Subject to: q_min <= q_fb + dq <= q_max
                       (equivalently: q_min - q_fb <= dq <= q_max - q_fb)
        
        Args:
            All (6,) for positions, (6, DOF) for Jacobians
        Returns:
            q_des: (DOF,)
            dq_des: (DOF,)
        """
        try:
            import osqp
            from scipy import sparse
        except ImportError:
            print("Warning: osqp not installed. Falling back to unconstrained solver.")
            print("Install with: pip install osqp")
            return self.update_task_space_command_with_constraints(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des
            )
        
        # =====================================================================
        # Task space weights (for ||J*dq - e||^2_W)
        # =====================================================================
        # Elbow weight matrix (6x6: orientation + position)
        # We = 10*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)  # position only
        We = .1*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1*np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        # # Hand weight matrix (6x6: orientation + position)
        # Wh = 0*np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)  # position only
        
        # COM weight matrix (6x6)
        Wc = np.eye(6, dtype=np.float64) * 5000
        
        # Joint regularization weight (for ||dq||^2_Wq)
        # Must be much smaller than task weights for good tracking
        Wq = np.eye(DOF, dtype=np.float64) * 10
        Wq[:6, :6] = 0  # no regularization on floating base (it's fixed anyway)
        
        # Nominal pose attraction weight (for ||q - q_nom||^2_Wq_nom)
        # Helps avoid joint limits by attracting to a safe nominal configuration
        Wq_nom = np.eye(DOF, dtype=np.float64) * 5
        Wq_nom[:6, :6] = 0  # no nominal attraction on floating base
        q_nom = np.array([0.0, 0.0, 0.85,  # base position (X, Y, Z)
                          0.0, 0.0, 0.0,  # base orientation (Rx, Ry, Rz)
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # torso (joint 6)
                          0.2, 0.5, 0.2, 0.5, 0.2,  0.5, 0.2,  # right arm (7 DOFs)
                          0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2,  # left arm (7 DOFs)
                          0.0, 0.0], dtype=np.float64)  # remaining joints
        
        # =====================================================================
        # Current state and task errors: e = x_des - x
        # =====================================================================
        q_fb = self.q.astype(np.float64)  # current joint configuration
        
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        # COM Jacobian (6 x DOF) - identity for floating base DOFs
        J_com = np.zeros((6, DOF), dtype=np.float64)
        J_com[:, :6] = np.eye(6, dtype=np.float64)
        
        # Convert Jacobians to float64
        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)

        # ========= Add CBF constraints =========
        # TORSO sphere (original)
        d_elbow_com_l = x_elbow_l[3:] - q_fb[:3]   
        d_elbow_com_r = x_elbow_r[3:] - q_fb[:3]     
        d_hand_com_l = x_hand_l[3:] - q_fb[:3]   
        d_hand_com_r = x_hand_r[3:] - q_fb[:3]
        
        # HEAD sphere offset from COM
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)  # Offset from COM
        head_pos = q_fb[:3] + head_offset  # Head sphere position in world frame
        
        # CROTCH sphere offset from COM
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)  # Offset from COM
        crotch_pos = q_fb[:3] + crotch_offset  # Crotch sphere position in world frame
        
        # Distance vectors from elbows/hands to head sphere
        d_elbow_head_l = x_elbow_l[3:] - head_pos
        d_elbow_head_r = x_elbow_r[3:] - head_pos
        d_hand_head_l = x_hand_l[3:] - head_pos
        d_hand_head_r = x_hand_r[3:] - head_pos
        
        # Distance vectors from elbows/hands to crotch sphere
        d_elbow_crotch_l = x_elbow_l[3:] - crotch_pos
        d_elbow_crotch_r = x_elbow_r[3:] - crotch_pos
        d_hand_crotch_l = x_hand_l[3:] - crotch_pos
        d_hand_crotch_r = x_hand_r[3:] - crotch_pos
        
        # Sphere radii
        r_torso = 0.13  # torso radius
        r_head = 0.11   # head radius
        r_crotch = 0.16  # crotch radius
        safety_margin = 0.02  # additional safety margin
        rho_torso = r_torso + safety_margin
        rho_head = r_head + safety_margin
        rho_crotch = r_crotch + safety_margin

        # Barrier functions (h >= 0 means outside sphere)
        h_elbow_l_torso = np.linalg.norm(d_elbow_com_l)**2 - rho_torso**2
        h_elbow_r_torso = np.linalg.norm(d_elbow_com_r)**2 - rho_torso**2
        h_hand_l_torso = np.linalg.norm(d_hand_com_l)**2 - rho_torso**2
        h_hand_r_torso = np.linalg.norm(d_hand_com_r)**2 - rho_torso**2
        
        h_elbow_l_head = np.linalg.norm(d_elbow_head_l)**2 - rho_head**2
        h_elbow_r_head = np.linalg.norm(d_elbow_head_r)**2 - rho_head**2
        h_hand_l_head = np.linalg.norm(d_hand_head_l)**2 - rho_head**2
        h_hand_r_head = np.linalg.norm(d_hand_head_r)**2 - rho_head**2
        
        h_elbow_l_crotch = np.linalg.norm(d_elbow_crotch_l)**2 - rho_crotch**2
        h_elbow_r_crotch = np.linalg.norm(d_elbow_crotch_r)**2 - rho_crotch**2
        h_hand_l_crotch = np.linalg.norm(d_hand_crotch_l)**2 - rho_crotch**2
        h_hand_r_crotch = np.linalg.norm(d_hand_crotch_r)**2 - rho_crotch**2

        # Note: Featherstone spatial convention: J[0:3,:]=angular, J[3:6,:]=linear
        # CBF is position-based, so use the linear (translational) rows J[3:,:]
        # J_com[:3,:] is already translational (maps Px,Py,Pz DOFs to position)
        # TORSO constraints
        aT_elbow_l_torso = 2*d_elbow_com_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_torso = 2*d_elbow_com_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_torso = 2*d_hand_com_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_torso = 2*d_hand_com_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        
        # HEAD constraints (note: head_pos = q_fb[:3] + head_offset, so d/dq[0:3] includes head motion)
        aT_elbow_l_head = 2*d_elbow_head_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_head = 2*d_elbow_head_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_head = 2*d_hand_head_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_head = 2*d_hand_head_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        
        # CROTCH constraints (note: crotch_pos = q_fb[:3] + crotch_offset, so d/dq[0:3] includes crotch motion)
        aT_elbow_l_crotch = 2*d_elbow_crotch_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_crotch = 2*d_elbow_crotch_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_crotch = 2*d_hand_crotch_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_crotch = 2*d_hand_crotch_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        
        lambda_cbf = 0.5  # CBF constraint gain

        # Constraint bounds
        b_elbow_l_torso = -lambda_cbf * h_elbow_l_torso
        b_elbow_r_torso = -lambda_cbf * h_elbow_r_torso
        b_hand_l_torso = -lambda_cbf * h_hand_l_torso
        b_hand_r_torso = -lambda_cbf * h_hand_r_torso
        
        b_elbow_l_head = -lambda_cbf * h_elbow_l_head
        b_elbow_r_head = -lambda_cbf * h_elbow_r_head
        b_hand_l_head = -lambda_cbf * h_hand_l_head
        b_hand_r_head = -lambda_cbf * h_hand_r_head
        
        b_elbow_l_crotch = -lambda_cbf * h_elbow_l_crotch
        b_elbow_r_crotch = -lambda_cbf * h_elbow_r_crotch
        b_hand_l_crotch = -lambda_cbf * h_hand_l_crotch
        b_hand_r_crotch = -lambda_cbf * h_hand_r_crotch
        
        # Concatenate ALL CBF constraints: aT @ dq >= b
        # Order: torso (4) + head (4) + crotch (4) = 12 total
        aT = np.vstack([aT_elbow_l_torso, aT_elbow_r_torso, aT_hand_l_torso, aT_hand_r_torso,
                        aT_elbow_l_head, aT_elbow_r_head, aT_hand_l_head, aT_hand_r_head,
                        aT_elbow_l_crotch, aT_elbow_r_crotch, aT_hand_l_crotch, aT_hand_r_crotch])  # (12, DOF)
        b = np.array([b_elbow_l_torso, b_elbow_r_torso, b_hand_l_torso, b_hand_r_torso,
                      b_elbow_l_head, b_elbow_r_head, b_hand_l_head, b_hand_r_head,
                      b_elbow_l_crotch, b_elbow_r_crotch, b_hand_l_crotch, b_hand_r_crotch], dtype=np.float64)  # (12,)
        # =====================================================================
        # Build QP: min 0.5 * dq^T @ P @ dq + c^T @ dq
        # 
        # Cost1: ||J*dq - e||^2_W = dq^T (J^T W J) dq - 2 e^T W J dq + e^T W e
        # Cost2: ||dq||^2_Wq = dq^T Wq dq
        # Cost3: ||q_fb + dq - q_nom||^2_Wq_nom = dq^T Wq_nom dq - 2(q_nom - q_fb)^T Wq_nom dq + ...
        #
        # P = sum(J^T W J) + Wq + Wq_nom
        # c = sum(-J^T W e) - 2*Wq_nom*(q_nom - q_fb)
        # =====================================================================
        
        # P matrix (Hessian)
        P = np.zeros((DOF, DOF), dtype=np.float64)
        P += J_elbow_l.T @ We @ J_elbow_l
        P += J_elbow_r.T @ We @ J_elbow_r
        P += J_hand_l.T @ Wh @ J_hand_l
        P += J_hand_r.T @ Wh @ J_hand_r
        P += J_com.T @ Wc @ J_com
        P += Wq  # regularization
        P += Wq_nom  # nominal pose attraction
        
        # Make P symmetric and positive definite
        P = 0.5 * (P + P.T)
        P += np.eye(DOF) * 1e-6  # numerical stability
        
        # c vector (linear term)
        c = np.zeros(DOF, dtype=np.float64)
        c -= J_elbow_l.T @ We @ e_elbow_l
        c -= J_elbow_r.T @ We @ e_elbow_r
        c -= J_hand_l.T @ Wh @ e_hand_l
        c -= J_hand_r.T @ Wh @ e_hand_r
        c -= J_com.T @ Wc @ e_com
        # Note: Wq term contributes 0 to c since we minimize ||dq||^2 (reference is 0)
        
        # Nominal pose attraction: min ||q_fb + dq - q_nom||^2_Wq_nom
        # = min dq^T Wq_nom dq - 2*(q_nom - q_fb)^T Wq_nom dq + const
        # Linear term: c += -2*Wq_nom*(q_nom - q_fb) = 2*Wq_nom*(q_fb - q_nom)
        e_nom = q_fb - q_nom  # error from nominal pose
        c += 2 * Wq_nom @ e_nom
        
        # =====================================================================
        # Inequality constraints: q_min <= q_fb + dq <= q_max
        # Rearranged: q_min - q_fb <= dq <= q_max - q_fb
        # Plus CBF constraints: aT @ dq >= b  (rearranged: b <= aT @ dq <= inf)
        # =====================================================================
        qj_min = q_min.astype(np.float64)
        qj_max = q_max.astype(np.float64)
        
        lb_box = qj_min - q_fb  # lower bound on dq (box constraints)
        ub_box = qj_max - q_fb  # upper bound on dq (box constraints)
        
        # Combine box constraints (I @ dq) and CBF constraints (aT @ dq)
        # OSQP format: l <= A @ dq <= u
        num_box_constraints = DOF
        num_cbf_constraints = aT.shape[0]  # 4 (elbow_l, elbow_r, hand_l, hand_r)
        num_total_constraints = num_box_constraints + num_cbf_constraints
        
        # Build combined constraint matrix: [I; aT]
        A_box = sparse.eye(DOF, format='csc')
        A_cbf = sparse.csc_matrix(aT)
        A_combined = sparse.vstack([A_box, A_cbf])
        
        # Build combined bounds
        lb_combined = np.hstack([lb_box, b])  # [lb_box; b]
        ub_combined = np.hstack([ub_box, np.inf * np.ones(num_cbf_constraints)])  # [ub_box; inf]
        
        # # Debug: print CBF values periodically
        # if not hasattr(self, '_cbf_debug_count'):
        #     self._cbf_debug_count = 0
        # self._cbf_debug_count += 1
        # if self._cbf_debug_count % 500 == 0:  # Every ~1 second at 500Hz
        #     print(f"[CBF] h_L={h_elbow_l:.4f}, h_R={h_elbow_r:.4f}, "
        #           f"d_L={np.linalg.norm(d_elbow_com_l):.3f}, d_R={np.linalg.norm(d_elbow_com_r):.3f}, "
        #           f"rho={rho:.3f}")
        
        # =====================================================================
        # Solve QP using OSQP
        # OSQP format: min 0.5 x'Px + q'x  s.t. l <= Ax <= u
        # Constraints: [I; aT] @ dq with bounds [lb_box; b] <= dq_constraint <= [ub_box; inf]
        # =====================================================================
        P_sparse = sparse.csc_matrix(P)
        A_sparse = A_combined  # Combined box + CBF constraints
        
        # Recreate solver each iteration (P and A change with robot configuration)
        # Use warm start from previous solution for faster convergence
        self._osqp_solver = osqp.OSQP()
        
        # Suppress all OSQP output during setup
        import sys, os
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        
        try:
            self._osqp_solver.setup(P_sparse, c, A_sparse, lb_combined, ub_combined,
                       verbose=False,
                       polish=True,
                       eps_abs=1e-5,
                       eps_rel=1e-5,
                       max_iter=500,
                       warm_start=True)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
        
        # Warm start with previous solution
        self._osqp_solver.warm_start(x=self._osqp_prev_dq)
        
        # Solve (suppress OSQP output)
        import sys, os
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            result = self._osqp_solver.solve()
        finally:
            sys.stdout = old_stdout
            devnull.close()
        
        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            dq_sol = result.x.astype(np.float32)
            self._osqp_prev_dq = result.x.copy()  # store for warm start
        else:
            # Fallback to unconstrained solution: P @ dq = -c => dq = -P^{-1} @ c
            print(f"OSQP status: {result.info.status}, using unconstrained solution")
            dq_sol = np.linalg.solve(P, -c).astype(np.float32)
            self._osqp_prev_dq = dq_sol.astype(np.float64)  # store for warm start
        
        q_des = self.q + dq_sol
        dq_des = dq_sol  # return the joint velocity
        
        return q_des, dq_des

    def update_task_space_command_qp_distributed_proxqp(self, 
                                                         x_elbow_l_des, x_elbow_r_des,
                                                         x_elbow_l, x_elbow_r,
                                                         x_hand_l_des, x_hand_r_des,
                                                         x_hand_l, x_hand_r,
                                                         J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                                         com_des):
        """Distributed IK with ProxQP: solve separate 13-DOF QPs for each arm.
        
        Decouples left/right arm tracking to avoid cross-arm interference while
        using the faster ProxQP solver. Each arm gets (base 6 + arm 7 = 13 DOF).
        """
        try:
            import proxsuite
        except ImportError:
            print("Warning: proxsuite not installed. Install with: pip install proxsuite")
            print("Falling back to distributed OSQP.")
            return self.update_task_space_command_qp_distributed(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des)
        
        BASE_DOFS = np.arange(0, 6)
        RIGHT_ARM_DOFS = np.arange(18, 25)
        LEFT_ARM_DOFS = np.arange(25, 32)
        
        right_dofs = np.concatenate([BASE_DOFS, RIGHT_ARM_DOFS])
        left_dofs = np.concatenate([BASE_DOFS, LEFT_ARM_DOFS])
        
        # Weights
        We = np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 0*np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc_primary = np.eye(6, dtype=np.float64) * 5000
        Wc_secondary = np.eye(6, dtype=np.float64) * 5000
        
        # Current state
        q_fb = self.q.astype(np.float64)
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        J_com_full = np.zeros((6, DOF), dtype=np.float64)
        J_com_full[:, :6] = np.eye(6, dtype=np.float64)
        
        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)
        
        qj_min, qj_max = q_min.astype(np.float64), q_max.astype(np.float64)
        
        # CBF setup (same as full ProxQP)
        com_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float64)
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos = q_fb[:3] + head_offset
        crotch_pos = q_fb[:3] + crotch_offset
        
        r_torso, r_head, r_crotch = 0.13, 0.11, 0.16
        rho_torso = r_torso + 0.02
        rho_head = r_head + 0.02
        rho_crotch = r_crotch + 0.02
        lambda_cbf = 0.5
        
        def _solve_arm_qp_proxqp(dof_indices, J_elbow, J_hand, e_elbow, e_hand,
                                  x_elbow, x_hand, side_name, Wc, solver_attr):
            """Solve one arm's 13-DOF QP with ProxQP."""
            n = len(dof_indices)
            J_elbow_sub = J_elbow[:, dof_indices]
            J_hand_sub = J_hand[:, dof_indices]
            J_com_sub = J_com_full[:, dof_indices]
            Wq_sub = np.eye(n, dtype=np.float64) * 10
            
            # Cost: min 0.5 dq^T H dq + g^T dq
            H = (J_elbow_sub.T @ We @ J_elbow_sub +
                 J_hand_sub.T @ Wh @ J_hand_sub +
                 J_com_sub.T @ Wc @ J_com_sub +
                 Wq_sub)
            H = 0.5 * (H + H.T)
            H += np.eye(n) * 1e-6
            
            g = -(J_elbow_sub.T @ We @ e_elbow +
                  J_hand_sub.T @ Wh @ e_hand +
                  J_com_sub.T @ Wc @ e_com)
            
            # CBF constraints (distance-based)
            x_elbow_pos = x_elbow[3:].astype(np.float64)
            x_hand_pos = x_hand[3:].astype(np.float64)
            
            d_elbow_com = x_elbow_pos - q_fb[:3] - com_offset
            d_hand_com = x_hand_pos - q_fb[:3] - com_offset
            d_elbow_head = x_elbow_pos - head_pos
            d_hand_head = x_hand_pos - head_pos
            d_elbow_crotch = x_elbow_pos - crotch_pos
            d_hand_crotch = x_hand_pos - crotch_pos
            
            h_values = np.array([
                np.linalg.norm(d_elbow_com)**2 - rho_torso**2,
                np.linalg.norm(d_hand_com)**2 - rho_torso**2,
                np.linalg.norm(d_elbow_head)**2 - rho_head**2,
                np.linalg.norm(d_hand_head)**2 - rho_head**2,
                np.linalg.norm(d_elbow_crotch)**2 - rho_crotch**2,
                np.linalg.norm(d_hand_crotch)**2 - rho_crotch**2,
            ], dtype=np.float64)
            
            J_elbow_lin = J_elbow[3:, dof_indices]
            J_hand_lin = J_hand[3:, dof_indices]
            J_com_lin = J_com_full[:3, dof_indices]
            
            d_vectors = [
                d_elbow_com, d_hand_com,
                d_elbow_head, d_hand_head,
                d_elbow_crotch, d_hand_crotch
            ]
            J_refs = [J_elbow_lin, J_hand_lin, J_elbow_lin, J_hand_lin, J_elbow_lin, J_hand_lin]
            
            aT_rows = [2 * d @ (J_ref - J_com_lin) for d, J_ref in zip(d_vectors, J_refs)]
            aT_cbf = np.vstack(aT_rows)  # (6, n)
            b_cbf = -lambda_cbf * h_values
            
            # Box + CBF constraints: l <= C dq <= u
            lb_box = qj_min[dof_indices] - q_fb[dof_indices]
            ub_box = qj_max[dof_indices] - q_fb[dof_indices]
            
            C = np.vstack([np.eye(n, dtype=np.float64), aT_cbf])
            l = np.concatenate([lb_box, b_cbf])
            u = np.concatenate([ub_box, np.full(6, 1e30)])
            
            n_ineq = C.shape[0]
            
            # Solve with ProxQP
            try:
                # Get or create solver for this arm
                if not hasattr(self, solver_attr) or getattr(self, solver_attr) is None:
                    solver = proxsuite.proxqp.dense.QP(n=n, n_eq=0, n_in=n_ineq)
                    solver.settings.eps_abs = 1e-4
                    solver.settings.eps_rel = 0.0
                    solver.settings.max_iter = 100
                    solver.settings.verbose = False
                    solver.settings.initial_guess = (
                        proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                    )
                    solver.init(H=H, g=g, A=None, b=None, C=C, l=l, u=u)
                    setattr(self, solver_attr, solver)
                else:
                    solver = getattr(self, solver_attr)
                    solver.update(H=H, g=g, A=None, b=None, C=C, l=l, u=u)
                
                solver.solve()
                dq_sol = solver.results.x.astype(np.float32)
                
            except Exception as e:
                print(f"[DistProxQP] {side_name}: {e}, unconstrained fallback")
                try:
                    dq_sol = np.linalg.solve(H, -g).astype(np.float32)
                except:
                    dq_sol = np.zeros(n, dtype=np.float32)
            
            return dq_sol
        
        # Solve both arms
        dq_right = _solve_arm_qp_proxqp(right_dofs, J_elbow_r, J_hand_r, e_elbow_r, e_hand_r,
                                         x_elbow_r, x_hand_r, 'right', Wc_primary, '_proxqp_solver_right')
        dq_left = _solve_arm_qp_proxqp(left_dofs, J_elbow_l, J_hand_l, e_elbow_l, e_hand_l,
                                        x_elbow_l, x_hand_l, 'left', Wc_secondary, '_proxqp_solver_left')
        
        # Combine: right arm's base takes priority
        dq_sol = np.zeros(DOF, dtype=np.float32)
        dq_sol[:6] = dq_right[:6].astype(np.float32)
        dq_sol[18:25] = dq_right[6:].astype(np.float32)
        dq_sol[25:32] = dq_left[6:].astype(np.float32)
        
        q_des = self.q + dq_sol
        dq_des = dq_sol
        
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
        We = .1*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
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
        
        r_torso, r_head, r_crotch = 0.13, 0.11, 0.16
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

    def update_task_space_command_qp_proxqp(self,
                                             x_elbow_l_des, x_elbow_r_des,
                                             x_elbow_l, x_elbow_r,
                                             x_hand_l_des, x_hand_r_des,
                                             x_hand_l, x_hand_r,
                                             J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                             com_des):
        """Solve IK using ProxQP (proxsuite) - proximal augmented Lagrangian method.
        
        ProxQP is designed for robotics QPs by INRIA. It handles near-infeasibility
        gracefully, supports warm-starting, and is very fast for dense problems.
        
        ProxQP format:
            min  0.5 * x^T H x + g^T x
            s.t. A_eq x = b_eq           (equality constraints)
                 l <= C x <= u           (inequality constraints)
        
        We map our problem as:
            - No equality constraints
            - C = [I; aT_cbf]  (box + CBF)
            - l = [lb_box; b_cbf], u = [ub_box; +inf]
        
        Args:
            All (6,) for positions, (6, DOF) for Jacobians
        Returns:
            q_des: (DOF,)
            dq_des: (DOF,)
        """
        try:
            import proxsuite
        except ImportError:
            print("Warning: proxsuite not installed. Install with: pip install proxsuite")
            print("Falling back to OSQP.")
            return self.update_task_space_command_qp(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des)
        
        # =====================================================================
        # Weights (same as OSQP version)
        # =====================================================================
        We = np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc = np.eye(6, dtype=np.float64) * 5000
        Wq = np.eye(DOF, dtype=np.float64) * 50
        Wq[:6, :6] = 0
        
        # =====================================================================
        # Current state and errors
        # =====================================================================
        q_fb = self.q.astype(np.float64)
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        J_com = np.zeros((6, DOF), dtype=np.float64)
        J_com[:, :6] = np.eye(6, dtype=np.float64)
        
        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)
        
        # =====================================================================
        # CBF constraints
        # =====================================================================
        com_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float64)  # COM sphere at base position
        d_elbow_com_l = x_elbow_l[3:] - q_fb[:3] - com_offset
        d_elbow_com_r = x_elbow_r[3:] - q_fb[:3] - com_offset
        d_hand_com_l = x_hand_l[3:] - q_fb[:3] - com_offset
        d_hand_com_r = x_hand_r[3:] - q_fb[:3] - com_offset
        
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos = q_fb[:3] + head_offset
        crotch_pos = q_fb[:3] + crotch_offset
        
        d_elbow_head_l = x_elbow_l[3:] - head_pos
        d_elbow_head_r = x_elbow_r[3:] - head_pos
        d_hand_head_l = x_hand_l[3:] - head_pos
        d_hand_head_r = x_hand_r[3:] - head_pos
        
        d_elbow_crotch_l = x_elbow_l[3:] - crotch_pos
        d_elbow_crotch_r = x_elbow_r[3:] - crotch_pos
        d_hand_crotch_l = x_hand_l[3:] - crotch_pos
        d_hand_crotch_r = x_hand_r[3:] - crotch_pos
        
        r_torso, r_head, r_crotch = 0.13, 0.11, 0.16
        safety_margin = 0.02
        rho_torso = r_torso + safety_margin
        rho_head = r_head + safety_margin
        rho_crotch = r_crotch + safety_margin
        
        h_elbow_l_torso = np.linalg.norm(d_elbow_com_l)**2 - rho_torso**2
        h_elbow_r_torso = np.linalg.norm(d_elbow_com_r)**2 - rho_torso**2
        h_hand_l_torso = np.linalg.norm(d_hand_com_l)**2 - rho_torso**2
        h_hand_r_torso = np.linalg.norm(d_hand_com_r)**2 - rho_torso**2
        h_elbow_l_head = np.linalg.norm(d_elbow_head_l)**2 - rho_head**2
        h_elbow_r_head = np.linalg.norm(d_elbow_head_r)**2 - rho_head**2
        h_hand_l_head = np.linalg.norm(d_hand_head_l)**2 - rho_head**2
        h_hand_r_head = np.linalg.norm(d_hand_head_r)**2 - rho_head**2
        h_elbow_l_crotch = np.linalg.norm(d_elbow_crotch_l)**2 - rho_crotch**2
        h_elbow_r_crotch = np.linalg.norm(d_elbow_crotch_r)**2 - rho_crotch**2
        h_hand_l_crotch = np.linalg.norm(d_hand_crotch_l)**2 - rho_crotch**2
        h_hand_r_crotch = np.linalg.norm(d_hand_crotch_r)**2 - rho_crotch**2
        
        # Featherstone: J[0:3,:]=angular, J[3:6,:]=linear. CBF uses linear rows.
        aT_elbow_l_torso = 2*d_elbow_com_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_torso = 2*d_elbow_com_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_torso = 2*d_hand_com_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_torso = 2*d_hand_com_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        aT_elbow_l_head = 2*d_elbow_head_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_head = 2*d_elbow_head_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_head = 2*d_hand_head_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_head = 2*d_hand_head_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        aT_elbow_l_crotch = 2*d_elbow_crotch_l.T @ (J_elbow_l[3:, :] - J_com[:3, :])
        aT_elbow_r_crotch = 2*d_elbow_crotch_r.T @ (J_elbow_r[3:, :] - J_com[:3, :])
        aT_hand_l_crotch = 2*d_hand_crotch_l.T @ (J_hand_l[3:, :] - J_com[:3, :])
        aT_hand_r_crotch = 2*d_hand_crotch_r.T @ (J_hand_r[3:, :] - J_com[:3, :])
        
        lambda_cbf = 0.5
        
        b_cbf = -lambda_cbf * np.array([
            h_elbow_l_torso, h_elbow_r_torso, h_hand_l_torso, h_hand_r_torso,
            h_elbow_l_head, h_elbow_r_head, h_hand_l_head, h_hand_r_head,
            h_elbow_l_crotch, h_elbow_r_crotch, h_hand_l_crotch, h_hand_r_crotch
        ], dtype=np.float64)
        
        aT_cbf = np.vstack([
            aT_elbow_l_torso, aT_elbow_r_torso, aT_hand_l_torso, aT_hand_r_torso,
            aT_elbow_l_head, aT_elbow_r_head, aT_hand_l_head, aT_hand_r_head,
            aT_elbow_l_crotch, aT_elbow_r_crotch, aT_hand_l_crotch, aT_hand_r_crotch
        ])  # (12, DOF)
        
        # =====================================================================
        # Build QP cost: min 0.5 * dq^T H dq + g^T dq
        # =====================================================================
        H = np.zeros((DOF, DOF), dtype=np.float64)
        H += J_elbow_l.T @ We @ J_elbow_l
        H += J_elbow_r.T @ We @ J_elbow_r
        H += J_hand_l.T @ Wh @ J_hand_l
        H += J_hand_r.T @ Wh @ J_hand_r
        H += J_com.T @ Wc @ J_com
        H += Wq
        H = 0.5 * (H + H.T)
        H += np.eye(DOF) * 1e-6
        
        g = np.zeros(DOF, dtype=np.float64)
        g -= J_elbow_l.T @ We @ e_elbow_l
        g -= J_elbow_r.T @ We @ e_elbow_r
        g -= J_hand_l.T @ Wh @ e_hand_l
        g -= J_hand_r.T @ Wh @ e_hand_r
        g -= J_com.T @ Wc @ e_com
        
        # =====================================================================
        # Inequality constraints: l <= C @ dq <= u
        # Row block 1: box constraints   I @ dq in [q_min - q, q_max - q]
        # Row block 2: CBF constraints  aT @ dq in [b_cbf, +inf]
        # =====================================================================
        qj_min = q_min.astype(np.float64)
        qj_max = q_max.astype(np.float64)
        lb_box = qj_min - q_fb
        ub_box = qj_max - q_fb
        
        n_cbf = aT_cbf.shape[0]  # 12
        
        C = np.vstack([np.eye(DOF, dtype=np.float64), aT_cbf])  # (DOF+12, DOF)
        l = np.concatenate([lb_box, b_cbf])                      # (DOF+12,)
        u = np.concatenate([ub_box, np.full(n_cbf, 1e30)])       # (DOF+12,)
        
        n_ineq = C.shape[0]
        
        # =====================================================================
        # Solve with ProxQP (dense backend)
        # =====================================================================
        try:
            if self._proxqp_solver is None or self._proxqp_solver.model.dim != DOF:
                # First call: create solver
                self._proxqp_solver = proxsuite.proxqp.dense.QP(
                    n=DOF,       # number of variables
                    n_eq=0,      # no equality constraints
                    n_in=n_ineq  # inequality constraints
                )
                # Configure for real-time performance
                self._proxqp_solver.settings.eps_abs = 1e-4
                self._proxqp_solver.settings.eps_rel = 0.0
                self._proxqp_solver.settings.max_iter = 100
                self._proxqp_solver.settings.verbose = False
                self._proxqp_solver.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
                
                # Init (no equality constraints → pass None for A, b)
                self._proxqp_solver.init(
                    H=H, g=g,
                    A=None, b=None,
                    C=C, l=l, u=u
                )
                self._proxqp_solver.solve()
            else:
                # Subsequent calls: update problem data and warm-start solve
                self._proxqp_solver.update(
                    H=H, g=g,
                    A=None, b=None,
                    C=C, l=l, u=u
                )
                self._proxqp_solver.solve()
            
            dq_sol = self._proxqp_solver.results.x.astype(np.float32)
            self._proxqp_prev_dq = self._proxqp_solver.results.x.copy()
            
        except Exception as e:
            print(f"[ProxQP] Error: {e}, falling back to unconstrained")
            try:
                dq_sol = np.linalg.solve(H, -g).astype(np.float32)
            except:
                dq_sol = np.zeros(DOF, dtype=np.float32)
        
        q_des = self.q + dq_sol
        dq_des = dq_sol
        
        return q_des, dq_des

    def update_task_space_command_qp_gpu_batch(self,
                                               x_elbow_l_des, x_elbow_r_des,
                                               x_elbow_l, x_elbow_r,
                                               x_hand_l_des, x_hand_r_des,
                                               x_hand_l, x_hand_r,
                                               J_elbow_l, J_elbow_r, J_hand_l, J_hand_r,
                                               com_des,
                                               n_batch=128, max_iter=20,
                                               pos_threshold=0.005):
        """GPU-batched IK-QP: solve N parallel QPs with randomized warm starts.
        
        Solves the same weighted IK problem N times in parallel on GPU, each
        with a different random initial guess. Returns the solution with the
        lowest tracking error. This helps escape poor convergence regions that
        a single warm start might get stuck in.
        
        Includes a ratchet mechanism: when the desired end-effector targets
        haven't moved beyond pos_threshold, only accept the new solution if
        its linearized tracking error is lower than the previous accepted
        solution. Otherwise the robot holds its current pose (dq=0).
        
        Uses ADMM (Alternating Direction Method of Multipliers) implemented
        in PyTorch for GPU-parallel batch solving.
        
        QP formulation (same as proxqp version):
            min  0.5 * dq^T H dq + g^T dq
            s.t. q_min - q <= dq <= q_max - q          (joint limits)
                 A_cbf @ dq >= b_cbf                    (CBF collision avoidance)
        
        Args:
            x_{elbow,hand}_{l,r}_des: (6,) desired task-space poses [rpy, xyz]
            x_{elbow,hand}_{l,r}: (6,) current task-space poses
            J_{elbow,hand}_{l,r}: (6, DOF) task Jacobians
            com_des: (6,) desired COM/base pose
            n_batch: number of parallel QP instances (default 128)
            max_iter: ADMM iterations per solve (default 20)
            pos_threshold: float, max position change (m) in desired EE targets
                below which the ratchet engages (default 0.005 = 5 mm).
                Set to 0 to always accept.
        Returns:
            q_des: (DOF,) desired joint configuration
            dq_des: (DOF,) joint velocity (= dq_sol)
        """
        try:
            from gpu_qp_solver import BatchedGPUQPSolver, TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
        except ImportError:
            print("Warning: gpu_qp_solver/PyTorch not available. "
                  "Falling back to ProxQP.")
            return self.update_task_space_command_qp_proxqp(
                x_elbow_l_des, x_elbow_r_des, x_elbow_l, x_elbow_r,
                x_hand_l_des, x_hand_r_des, x_hand_l, x_hand_r,
                J_elbow_l, J_elbow_r, J_hand_l, J_hand_r, com_des)
        
        # =====================================================================
        # Lazily create GPU solver (persists across calls for warm-starting)
        # =====================================================================
        if (self._gpu_qp_solver is None
                or self._gpu_qp_solver.n_batch != n_batch
                or self._gpu_qp_solver.max_iter != max_iter):
            self._gpu_qp_solver = BatchedGPUQPSolver(
                n_batch=n_batch,
                max_iter=max_iter,
                rho=50.0,       # ADMM penalty (tuned for IK weight scale)
                alpha=1.6,      # over-relaxation for faster convergence
                sigma=1e-6,
                device=None,    # auto-detect GPU
                dtype='float32'
            )
        
        # =====================================================================
        # Task-space weights
        # =====================================================================
        # We = np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        # Wh = 2*np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)
        We = .1*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1*np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc = np.eye(6, dtype=np.float64) * 5000
        Wq = np.eye(DOF, dtype=np.float64) * 10
        Wq[:6, :6] = 0  # no regularization on floating base
        
        # =====================================================================
        # Current state and task errors
        # =====================================================================
        q_fb = self.q.astype(np.float64)
        
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        # COM Jacobian (identity on floating base)
        J_com = np.zeros((6, DOF), dtype=np.float64)
        J_com[:, :6] = np.eye(6, dtype=np.float64)
        
        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)
        
        # =====================================================================
        # CBF collision-avoidance constraints  (torso / head / crotch spheres)
        # =====================================================================
        com_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float64)
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos = q_fb[:3] + head_offset
        crotch_pos = q_fb[:3] + crotch_offset
        
        r_torso, r_head, r_crotch = 0.13, 0.11, 0.16
        safety_margin = 0.02
        rho_torso = r_torso + safety_margin
        rho_head = r_head + safety_margin
        rho_crotch = r_crotch + safety_margin
        
        # Distance vectors from end-effectors to each sphere center
        d_elbow_com_l  = x_elbow_l[3:] - q_fb[:3] - com_offset
        d_elbow_com_r  = x_elbow_r[3:] - q_fb[:3] - com_offset
        d_hand_com_l   = x_hand_l[3:]  - q_fb[:3] - com_offset
        d_hand_com_r   = x_hand_r[3:]  - q_fb[:3] - com_offset
        
        d_elbow_head_l = x_elbow_l[3:] - head_pos
        d_elbow_head_r = x_elbow_r[3:] - head_pos
        d_hand_head_l  = x_hand_l[3:]  - head_pos
        d_hand_head_r  = x_hand_r[3:]  - head_pos
        
        d_elbow_crotch_l = x_elbow_l[3:] - crotch_pos
        d_elbow_crotch_r = x_elbow_r[3:] - crotch_pos
        d_hand_crotch_l  = x_hand_l[3:]  - crotch_pos
        d_hand_crotch_r  = x_hand_r[3:]  - crotch_pos
        
        # Barrier functions h >= 0 (outside sphere)
        h_vals = np.array([
            np.dot(d_elbow_com_l, d_elbow_com_l)     - rho_torso**2,
            np.dot(d_elbow_com_r, d_elbow_com_r)     - rho_torso**2,
            np.dot(d_hand_com_l, d_hand_com_l)       - rho_torso**2,
            np.dot(d_hand_com_r, d_hand_com_r)       - rho_torso**2,
            np.dot(d_elbow_head_l, d_elbow_head_l)   - rho_head**2,
            np.dot(d_elbow_head_r, d_elbow_head_r)   - rho_head**2,
            np.dot(d_hand_head_l, d_hand_head_l)     - rho_head**2,
            np.dot(d_hand_head_r, d_hand_head_r)     - rho_head**2,
            np.dot(d_elbow_crotch_l, d_elbow_crotch_l) - rho_crotch**2,
            np.dot(d_elbow_crotch_r, d_elbow_crotch_r) - rho_crotch**2,
            np.dot(d_hand_crotch_l, d_hand_crotch_l)   - rho_crotch**2,
            np.dot(d_hand_crotch_r, d_hand_crotch_r)   - rho_crotch**2,
        ], dtype=np.float64)
        
        # Featherstone: J[0:3,:]=angular, J[3:6,:]=linear. CBF uses linear rows.
        J_com_lin = J_com[:3, :]  # translational COM Jacobian
        d_list = [
            d_elbow_com_l, d_elbow_com_r, d_hand_com_l, d_hand_com_r,
            d_elbow_head_l, d_elbow_head_r, d_hand_head_l, d_hand_head_r,
            d_elbow_crotch_l, d_elbow_crotch_r, d_hand_crotch_l, d_hand_crotch_r,
        ]
        J_ee_lin_list = [
            J_elbow_l[3:, :], J_elbow_r[3:, :], J_hand_l[3:, :], J_hand_r[3:, :],
            J_elbow_l[3:, :], J_elbow_r[3:, :], J_hand_l[3:, :], J_hand_r[3:, :],
            J_elbow_l[3:, :], J_elbow_r[3:, :], J_hand_l[3:, :], J_hand_r[3:, :],
        ]
        
        aT_rows = [2.0 * d @ (J_ee - J_com_lin)
                   for d, J_ee in zip(d_list, J_ee_lin_list)]
        aT_cbf = np.vstack(aT_rows)           # (12, DOF)
        
        lambda_cbf = 0.5
        b_cbf = -lambda_cbf * h_vals            # (12,)
        
        # =====================================================================
        # Build QP cost: min 0.5 * dq^T H dq + g^T dq
        # =====================================================================
        H_qp = np.zeros((DOF, DOF), dtype=np.float64)
        H_qp += J_elbow_l.T @ We @ J_elbow_l
        H_qp += J_elbow_r.T @ We @ J_elbow_r
        H_qp += J_hand_l.T @ Wh @ J_hand_l
        H_qp += J_hand_r.T @ Wh @ J_hand_r
        H_qp += J_com.T @ Wc @ J_com
        H_qp += Wq
        H_qp = 0.5 * (H_qp + H_qp.T)           # symmetrize
        H_qp += np.eye(DOF) * 1e-6              # numerical stability
        
        g_qp = np.zeros(DOF, dtype=np.float64)
        g_qp -= J_elbow_l.T @ We @ e_elbow_l
        g_qp -= J_elbow_r.T @ We @ e_elbow_r
        g_qp -= J_hand_l.T @ Wh @ e_hand_l
        g_qp -= J_hand_r.T @ Wh @ e_hand_r
        g_qp -= J_com.T @ Wc @ e_com
        
        # =====================================================================
        # Inequality constraints: l <= C @ dq <= u
        #   Row block 1: box (joint limits)   I @ dq in [q_min-q, q_max-q]
        #   Row block 2: CBF                  aT @ dq in [b_cbf,  +inf]
        # =====================================================================
        qj_min = q_min.astype(np.float64)
        qj_max = q_max.astype(np.float64)
        lb_box = qj_min - q_fb
        ub_box = qj_max - q_fb
        
        n_cbf = aT_cbf.shape[0]   # 12
        C_qp = np.vstack([np.eye(DOF, dtype=np.float64), aT_cbf])   # (DOF+12, DOF)
        l_qp = np.concatenate([lb_box, b_cbf])                      # (DOF+12,)
        u_qp = np.concatenate([ub_box, np.full(n_cbf, np.inf)])      # (DOF+12,)
        
        # =====================================================================
        # Solve with GPU-batched ADMM
        # =====================================================================
        # Warm start from previous best (internal to solver)
        x_warm = None
        if (self._gpu_qp_solver._prev_best is not None
                and self._gpu_qp_solver._prev_n == DOF):
            x_warm = self._gpu_qp_solver._prev_best.cpu().numpy()
        
        dq_sol = self._gpu_qp_solver.solve(
            H=H_qp, g=g_qp, C=C_qp, l=l_qp, u=u_qp,
            x_warm=x_warm
        )
        
        # =================================================================
        # Ratchet: only accept when tracking improves (static targets)
        # =================================================================
        if pos_threshold > 0:
            # Current tracking errors (position only, indices 3:6)
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
                # Accept: target moved or current tracking is better
                self._ik_ratchet_prev_des = des_pos
                self._ik_ratchet_prev_cost = current_cost
            else:
                # Reject: return previously accepted pose (hold)
                if (self._ik_ratchet_q_des is not None
                        and self._ik_ratchet_dq_des is not None):
                    return self._ik_ratchet_q_des, self._ik_ratchet_dq_des
                else:
                    # First iteration, no prior solution yet; send this one
                    pass
        
        q_des = self.q + dq_sol.astype(np.float32)
        dq_des = dq_sol.astype(np.float32)
        
        # Cache as the last accepted solution for future ratchet rejections
        self._ik_ratchet_q_des = q_des
        self._ik_ratchet_dq_des = dq_des
        
        return q_des, dq_des

    def update_task_space_command_qp_gpu_batch_distributed(self,
                                                            x_elbow_l_des, x_elbow_r_des,
                                                            x_elbow_l, x_elbow_r,
                                                            x_hand_l_des, x_hand_r_des,
                                                            x_hand_l, x_hand_r,
                                                            J_elbow_l, J_elbow_r,
                                                            J_hand_l, J_hand_r,
                                                            com_des,
                                                            n_batch=128, max_iter=20,
                                                            pos_threshold=0.005,
                                                            q_ref=None,
                                                            w_ref=0.0):
        """Distributed GPU-batched IK-QP: two independent 13-DOF QPs per arm.
        
        Decouples left/right arm tracking by solving separate (base 6 + arm 7
        = 13 DOF) QPs for each arm. Uses weight perturbation: N_LIN diverse
        tracking weight configurations are solved in a single GPU batch call
        to explore different cost trade-offs and escape local minima.
        
        Includes a ratchet mechanism: when the desired end-effector targets
        haven't moved beyond pos_threshold, only accept the new solution if
        its linearized tracking error is lower than the previous accepted
        solution. Otherwise the robot holds its current pose (dq=0).
        
        Optionally tracks a reference (favorable) pose with weighted cost.
        
        This combines the benefits of:
          - Distributed solving (no cross-arm interference)
          - GPU-batched ADMM (parallel local-minima exploration)
          - Weight perturbation (diverse cost landscapes)
          - Ratchet (monotonic tracking improvement when target is static)
          - Reference pose tracking (regularization toward favorable config)
        
        Args:
            x_{elbow,hand}_{l,r}_des: (6,) desired task-space poses [rpy, xyz]
            x_{elbow,hand}_{l,r}: (6,) current task-space poses
            J_{elbow,hand}_{l,r}: (6, DOF) task Jacobians
            com_des: (6,) desired COM/base pose
            n_batch: number of parallel QP instances per arm (default 128)
            max_iter: ADMM iterations per solve (default 20)
            pos_threshold: float, max position change (m) in desired EE targets
                below which the ratchet engages (default 0.005 = 5 mm).
                Set to 0 to always accept.
            q_ref: (DOF,) optional reference pose for tracking (None = no tracking)
            w_ref: float >= 0, weight on reference pose tracking (default 0 = disabled)
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
        
        # =====================================================================
        # Lazily create a single fused GPU solver (persist across calls)
        # =====================================================================
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
        
        # =====================================================================
        # DOF partitioning
        # =====================================================================
        BASE_DOFS = np.arange(0, 6)
        RIGHT_ARM_DOFS = np.arange(18, 25)
        LEFT_ARM_DOFS = np.arange(25, 32)
        right_dofs = np.concatenate([BASE_DOFS, RIGHT_ARM_DOFS])  # 13
        left_dofs = np.concatenate([BASE_DOFS, LEFT_ARM_DOFS])    # 13
        n_sub = 13
        
        # =====================================================================
        # Weights (matching distributed ProxQP version)
        # =====================================================================
        We = .1*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1*np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc = np.eye(6, dtype=np.float64) * 5000
        Wq_val = 10.0
        
        # =====================================================================
        # Current state and errors
        # =====================================================================
        q_fb = self.q.astype(np.float64)
        e_elbow_l = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r = (x_hand_r_des - x_hand_r).astype(np.float64)
        e_com = (com_des - self.q[:6]).astype(np.float64)
        
        J_com_full = np.zeros((6, DOF), dtype=np.float64)
        J_com_full[:, :6] = np.eye(6, dtype=np.float64)
        
        J_elbow_l = J_elbow_l.astype(np.float64)
        J_elbow_r = J_elbow_r.astype(np.float64)
        J_hand_l = J_hand_l.astype(np.float64)
        J_hand_r = J_hand_r.astype(np.float64)
        
        qj_min = q_min.astype(np.float64)
        qj_max = q_max.astype(np.float64)
        
        # =====================================================================
        # CBF setup (torso / head / crotch spheres)
        # =====================================================================
        com_offset = np.array([-0.1, 0.0, 0.0], dtype=np.float64)
        head_offset = np.array([-0.1, 0.0, 0.3], dtype=np.float64)
        crotch_offset = np.array([-0.1, 0.0, -0.3], dtype=np.float64)
        head_pos = q_fb[:3] + head_offset
        crotch_pos = q_fb[:3] + crotch_offset
        
        r_torso, r_head, r_crotch = 0.15, 0.13, 0.19
        rho_torso = r_torso + 0.02
        rho_head = r_head + 0.02
        rho_crotch = r_crotch + 0.02
        lambda_cbf = 0.5
        
        def _build_arm_qp(dof_indices, J_elbow, J_hand, e_elbow, e_hand,
                          x_elbow, x_hand, Wc_arm, q_ref_arm=None, w_ref_arm=0.0,
                          We_override=None, Wh_override=None, Wq_val_override=None,
                          mu=None):
            """Build the QP matrices for one arm (13 DOF).
            
            Args:
                q_ref_arm: (13,) reference pose for this arm (base 6 + arm 7), or None
                w_ref_arm: weight on reference pose tracking cost
                We_override: (6,6) optional elbow weight matrix override
                Wh_override: (6,6) optional hand weight matrix override
                Wq_val_override: float, optional joint reg weight override
                mu: (n_sub,) optional joint velocity attractor shift.
                    Changes cost from ||dq||^2_Wq to ||dq - mu||^2_Wq.
                    Adds -Wq*mu to g (H unchanged).
            """
            We_use = We_override if We_override is not None else We
            Wh_use = Wh_override if Wh_override is not None else Wh
            Wq_use = Wq_val_override if Wq_val_override is not None else Wq_val
            
            J_elbow_sub = J_elbow[:, dof_indices]
            J_hand_sub = J_hand[:, dof_indices]
            J_com_sub = J_com_full[:, dof_indices]
            Wq_sub = np.eye(n_sub, dtype=np.float64) * Wq_use
            
            # Add reference pose regularization to Wq if enabled
            Wq_with_ref = Wq_sub.copy()
            if w_ref_arm > 0 and q_ref_arm is not None:
                Wq_with_ref += np.eye(n_sub, dtype=np.float64) * w_ref_arm
            
            # Hessian
            H = (J_elbow_sub.T @ We_use @ J_elbow_sub +
                 J_hand_sub.T @ Wh_use @ J_hand_sub +
                 J_com_sub.T @ Wc_arm @ J_com_sub +
                 Wq_with_ref)
            H = 0.5 * (H + H.T)
            H += np.eye(n_sub) * 1e-6
            
            # Check for NaN/Inf in Hessian
            if not np.all(np.isfinite(H)):
                print(f"WARNING: Hessian contains NaN/Inf! Adding stronger regularization.")
                H = np.eye(n_sub, dtype=np.float64) * 1e-3  # Fallback to identity
            
            # Linear term: standard tracking errors
            g = -(J_elbow_sub.T @ We_use @ e_elbow +
                  J_hand_sub.T @ Wh_use @ e_hand +
                  J_com_sub.T @ Wc_arm @ e_com)
            
            # Joint velocity attractor shift: ||dq - mu||^2_Wq
            # Expands to dq^T Wq dq - 2 mu^T Wq dq + const
            # so g gets -Wq @ mu (since cost = 0.5 x^T H x + g^T x)
            if mu is not None:
                g += -Wq_sub @ mu
            
            # Add reference pose tracking term to linear cost
            if w_ref_arm > 0 and q_ref_arm is not None:
                # Current position relative to reference: q - q_ref
                q_fb_arm = q_fb[dof_indices]
                q_ref_feedback = q_ref_arm.astype(np.float64)
                # Linear term for minimizing (q - q_ref)^2 is -2 * w_ref * (q_ref - q)
                # which we add as Wq_ref @ (q_ref - q_fb_arm)
                g += -w_ref_arm * q_ref_feedback + w_ref_arm * q_fb_arm
            
            # Check for NaN/Inf in gradient
            if not np.all(np.isfinite(g)):
                print(f"WARNING: Gradient g contains NaN/Inf! Zeroing it.")
                g = np.zeros(n_sub, dtype=np.float64)
            
            # ── CBF constraints ──────────────────────────────────────────
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
            aT_cbf = np.vstack(aT_rows)                # (6, n_sub)
            b_cbf = -lambda_cbf * h_vals                # (6,)
            
            # ── Combined constraints ─────────────────────────────────────
            lb_box = qj_min[dof_indices] - q_fb[dof_indices]
            ub_box = qj_max[dof_indices] - q_fb[dof_indices]
            
            n_cbf = 6
            C_qp = np.vstack([np.eye(n_sub, dtype=np.float64), aT_cbf])
            l_qp = np.concatenate([lb_box, b_cbf])
            u_qp = np.concatenate([ub_box, np.full(n_cbf, np.inf)])
            
            return H, g, C_qp, l_qp, u_qp
        
        # =====================================================================
        # Build and solve QPs (fused block-diagonal for 2× speedup)
        # =====================================================================
        # Extract reference pose for each arm if provided
        q_ref_r = None
        q_ref_l = None
        if q_ref is not None:
            q_ref = np.asarray(q_ref, dtype=np.float64)
            if q_ref.shape[0] == DOF:
                q_ref_r = q_ref[right_dofs]
                q_ref_l = q_ref[left_dofs]
        
        fused_solver = self._gpu_qp_solver_right  # reuse field for fused solver
        
        # ─── Weight-perturbed multi-lin solve ────────────────────────────
        # Each of K partitions solves a QP with differently scaled
        # tracking weights (We, Wh, Wq). This changes both H and g,
        # reshaping the cost landscape so each partition converges to a
        # different solution. We then pick the one with the lowest
        # *nominal* tracking cost.
        #
        # Weight scales are drawn from LogNormal(0, σ) so they're always
        # positive and centered around the nominal value (scale=1).
        # Independent scales for elbow, hand, and joint regularization
        # give each partition a unique cost-surface shape.
        N_WEIGHT_PERTURB = 4     # K: number of weight partitions
        WEIGHT_SIGMA = 0.7       # log-std for weight scaling (~[0.2, 5.0] range)
        MU_SIGMA = 0.05          # rad: std of joint velocity attractor shift
        
        # Partition 0 = nominal weights, mu=0 (always included)
        H_r_list, g_r_list = [], []
        H_l_list, g_l_list = [], []
        
        # Build nominal QP (partition 0) — also provides shared C, l, u
        H_r_0, g_r_0, C_r, l_r, u_r = _build_arm_qp(
            right_dofs, J_elbow_r, J_hand_r, e_elbow_r, e_hand_r,
            x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref)
        H_l_0, g_l_0, C_l, l_l, u_l = _build_arm_qp(
            left_dofs, J_elbow_l, J_hand_l, e_elbow_l, e_hand_l,
            x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref)
        H_r_list.append(H_r_0);  g_r_list.append(g_r_0)
        H_l_list.append(H_l_0);  g_l_list.append(g_l_0)
        
        n_box = n_sub  # box-constraint rows = joint limits (13 per arm)
        
        # Generate K-1 perturbed QPs (weight scaling + mu shift)
        for _ in range(N_WEIGHT_PERTURB - 1):
            # Independent log-normal scales for each weight component
            s_e = np.exp(np.random.randn() * WEIGHT_SIGMA)   # elbow scale
            s_h = np.exp(np.random.randn() * WEIGHT_SIGMA)   # hand scale
            s_q = np.exp(np.random.randn() * WEIGHT_SIGMA)   # joint reg scale
            
            We_k = We * s_e
            Wh_k = Wh * s_h
            Wq_k = Wq_val * s_q
            
            # Random joint velocity attractor shift (independent per arm)
            mu_r = np.random.randn(n_sub) * MU_SIGMA
            mu_l = np.random.randn(n_sub) * MU_SIGMA
            
            H_r_k, g_r_k, _, _, _ = _build_arm_qp(
                right_dofs, J_elbow_r, J_hand_r, e_elbow_r, e_hand_r,
                x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref,
                We_override=We_k, Wh_override=Wh_k, Wq_val_override=Wq_k,
                mu=mu_r)
            H_l_k, g_l_k, _, _, _ = _build_arm_qp(
                left_dofs, J_elbow_l, J_hand_l, e_elbow_l, e_hand_l,
                x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref,
                We_override=We_k, Wh_override=Wh_k, Wq_val_override=Wq_k,
                mu=mu_l)
            H_r_list.append(H_r_k);  g_r_list.append(g_r_k)
            H_l_list.append(H_l_k);  g_l_list.append(g_l_k)
        
        # Shared bounds (constraints don't depend on weights)
        l_r_list = [l_r] * N_WEIGHT_PERTURB
        u_r_list = [u_r] * N_WEIGHT_PERTURB
        l_l_list = [l_l] * N_WEIGHT_PERTURB
        u_l_list = [u_l] * N_WEIGHT_PERTURB
        
        dq_right, dq_left = BatchedGPUQPSolver.solve_pair_multi_lin(
            fused_solver,
            H_r_list, H_l_list,
            g_r_list, g_l_list,
            C_r, C_l,
            l_r_list, u_r_list,
            l_l_list, u_l_list,
            n_box_a=n_box, n_box_b=n_box,
        )
        
        # Check for NaN/Inf in solver output
        if (not np.all(np.isfinite(dq_right)) or not np.all(np.isfinite(dq_left))):
            print(f"WARNING: GPU QP solver returned NaN/Inf!")
            print(f"  dq_right finite: {np.isfinite(dq_right)}")
            print(f"  dq_left finite: {np.isfinite(dq_left)}")
            print("  Returning zero velocity.")
            return self.q, np.zeros(DOF, dtype=np.float32)
        
        # =====================================================================
        # Combine: right arm base takes priority, left arm contributes limb only
        # =====================================================================
        dq_sol = np.zeros(DOF, dtype=np.float32)
        dq_sol[:6] = dq_right[:6].astype(np.float32)         # base from right
        dq_sol[18:25] = dq_right[6:].astype(np.float32)      # right arm joints
        dq_sol[25:32] = dq_left[6:].astype(np.float32)       # left arm joints
        
        # =================================================================
        # Ratchet: only accept when tracking improves (static targets)
        # =================================================================
        if pos_threshold > 0:
            # Current tracking errors (position only, indices 3:6)
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
                # Accept: target moved or current tracking is better
                self._ik_ratchet_prev_des = des_pos
                self._ik_ratchet_prev_cost = current_cost
            else:
                # Reject: return previously accepted pose (hold)
                if (self._ik_ratchet_q_des is not None
                        and self._ik_ratchet_dq_des is not None):
                    return self._ik_ratchet_q_des, self._ik_ratchet_dq_des
                else:
                    # First iteration, no prior solution yet; send this one
                    pass
        
        q_des = self.q + dq_sol
        dq_des = dq_sol
        
        # Final NaN/Inf check before returning
        if not np.all(np.isfinite(q_des)) or not np.all(np.isfinite(dq_des)):
            print("WARNING: Final q_des/dq_des contains NaN/Inf! Returning current state.")
            return self.q, np.zeros(DOF, dtype=np.float32)
        
        # Cache as the last accepted solution for future ratchet rejections
        self._ik_ratchet_q_des = q_des
        self._ik_ratchet_dq_des = dq_des
        
        return q_des, dq_des

    def update_task_space_command_qp_gpu_batch_distributed_alpha(
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
            delta_progress=0.001,
            dq_max=0.5):
        """Max-feasible-α distributed GPU-batched IK-QP.

        Instead of perturbing weights or μ, this method explores intermediate
        targets along the line from current EE position to desired target:

            x_d^{(k)} = (1 - α_k) * x(q) + α_k * x_d,   0 < α_1 < ... < α_K = 1

        For each α_k, we solve an independent QP (same linearization point,
        different tracking error / RHS). Then select the **largest** α_k
        whose candidate Δq^{(k)} satisfies:

        1. **Feasibility**:
           - CBF constraints satisfied (or slack below threshold)
           - Joint limits satisfied
           - Step limit: ||Δq^{(k)}|| ≤ dq_max

        2. **Progress toward final target** (using predicted linear model):
           ê_final^{(k)} = (x_d - x(q)) - J * Δq^{(k)}
           Require: ||ê_final^{(k)}|| ≤ ||e_final^{(0)}|| - δ
           where e_final^{(0)} = x_d - x(q) is the initial error.

        This ensures we always make progress toward the true goal while
        respecting constraints, choosing the most aggressive feasible step.

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
            delta_progress: minimum progress margin δ (default 0.001 m)
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

        # =====================================================================
        # Lazily create GPU solver
        # =====================================================================
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

        # =====================================================================
        # DOF partitioning
        # =====================================================================
        BASE_DOFS = np.arange(0, 6)
        RIGHT_ARM_DOFS = np.arange(18, 25)
        LEFT_ARM_DOFS = np.arange(25, 32)
        right_dofs = np.concatenate([BASE_DOFS, RIGHT_ARM_DOFS])  # 13
        left_dofs = np.concatenate([BASE_DOFS, LEFT_ARM_DOFS])    # 13
        n_sub = 13

        # =====================================================================
        # Weights
        # =====================================================================
        We = 0.01 * np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)
        Wh = 1.0 * np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float64)
        Wc = np.eye(6, dtype=np.float64) * 5000
        Wq_val = 10.0

        # =====================================================================
        # Current state
        # =====================================================================
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

        r_torso, r_head, r_crotch = 0.15, 0.13, 0.19
        rho_torso = r_torso + 0.02
        rho_head = r_head + 0.02
        rho_crotch = r_crotch + 0.02
        lambda_cbf = 0.5

        # Reference pose
        q_ref_r = None
        q_ref_l = None
        if q_ref is not None:
            q_ref = np.asarray(q_ref, dtype=np.float64)
            if q_ref.shape[0] == DOF:
                q_ref_r = q_ref[right_dofs]
                q_ref_l = q_ref[left_dofs]

        # =====================================================================
        # Build QP helper (closure over shared state)
        # =====================================================================
        def _build_arm_qp_alpha(dof_indices, J_elbow, J_hand, e_elbow, e_hand,
                                x_elbow, x_hand, Wc_arm, q_ref_arm=None, w_ref_arm=0.0):
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

        # =====================================================================
        # Generate α schedule: [α_1, ..., α_K] with α_K = 1
        # =====================================================================
        alphas = np.linspace(1.0 / n_alpha, 1.0, n_alpha)  # e.g., [0.125, 0.25, ..., 1.0]

        # Final target errors (for progress check)
        e_elbow_l_final = (x_elbow_l_des - x_elbow_l).astype(np.float64)
        e_elbow_r_final = (x_elbow_r_des - x_elbow_r).astype(np.float64)
        e_hand_l_final = (x_hand_l_des - x_hand_l).astype(np.float64)
        e_hand_r_final = (x_hand_r_des - x_hand_r).astype(np.float64)

        # Initial error norm (for progress requirement)
        e0_norm = np.sqrt(
            np.sum(e_elbow_l_final[3:]**2) + np.sum(e_elbow_r_final[3:]**2) +
            np.sum(e_hand_l_final[3:]**2) + np.sum(e_hand_r_final[3:]**2))

        # =====================================================================
        # Build QP once (α=1) to get shared H, C, l, u
        # =====================================================================
        H_r, g_r_full, C_r, l_r, u_r = _build_arm_qp_alpha(
            right_dofs, J_elbow_r, J_hand_r, e_elbow_r_final, e_hand_r_final,
            x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref)
        H_l, g_l_full, C_l, l_l, u_l = _build_arm_qp_alpha(
            left_dofs, J_elbow_l, J_hand_l, e_elbow_l_final, e_hand_l_final,
            x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref)

        # Store Jacobian subsets for progress check
        J_elbow_r_sub = J_elbow_r[:, right_dofs]
        J_hand_r_sub = J_hand_r[:, right_dofs]
        J_elbow_l_sub = J_elbow_l[:, left_dofs]
        J_hand_l_sub = J_hand_l[:, left_dofs]

        # =====================================================================
        # Compute g for each α analytically (g is linear in α)
        # g(α) = α * g_tracking + g_const
        # where g_tracking = -(J_e^T We e_final + J_h^T Wh e_final)
        # and g_const includes COM + reference terms (independent of α)
        # =====================================================================
        # Extract constant part: build g with zero tracking errors
        _, g_r_const, _, _, _ = _build_arm_qp_alpha(
            right_dofs, J_elbow_r, J_hand_r,
            np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64),
            x_elbow_r, x_hand_r, Wc, q_ref_arm=q_ref_r, w_ref_arm=w_ref)
        _, g_l_const, _, _, _ = _build_arm_qp_alpha(
            left_dofs, J_elbow_l, J_hand_l,
            np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64),
            x_elbow_l, x_hand_l, Wc, q_ref_arm=q_ref_l, w_ref_arm=w_ref)

        # g_tracking = g_full - g_const  (the part that scales with α)
        g_r_track = g_r_full - g_r_const
        g_l_track = g_l_full - g_l_const

        # Build per-α g vectors: g(α) = α * g_track + g_const
        g_r_list = [alpha * g_r_track + g_r_const for alpha in alphas]
        g_l_list = [alpha * g_l_track + g_l_const for alpha in alphas]

        # =====================================================================
        # Solve ALL α-QPs in a SINGLE GPU call
        # =====================================================================
        fused_solver = self._gpu_qp_solver_right

        all_solutions = BatchedGPUQPSolver.solve_pair_multi_g_all(
            fused_solver,
            H_r, H_l,
            g_r_list, g_l_list,
            C_r, C_l,
            l_r, u_r, l_l, u_l)

        # =====================================================================
        # Select max-feasible-α with progress guarantee
        # =====================================================================
        best_k = -1  # Will pick largest feasible k

        for k in range(n_alpha - 1, -1, -1):  # Iterate from largest α to smallest
            dq_r_k, dq_l_k = all_solutions[k]

            # Feasibility check 1: step limit
            dq_r_norm = np.linalg.norm(dq_r_k)
            dq_l_norm = np.linalg.norm(dq_l_k)
            if dq_r_norm > dq_max or dq_l_norm > dq_max:
                continue

            # Feasibility check 2: joint limits (already enforced by QP, but check)
            q_new_r = q_fb[right_dofs] + dq_r_k
            q_new_l = q_fb[left_dofs] + dq_l_k
            if np.any(q_new_r < qj_min[right_dofs] - 1e-4) or np.any(q_new_r > qj_max[right_dofs] + 1e-4):
                continue
            if np.any(q_new_l < qj_min[left_dofs] - 1e-4) or np.any(q_new_l > qj_max[left_dofs] + 1e-4):
                continue

            # Feasibility check 3: CBF (slack below threshold)
            # Check C @ dq >= l (lower bounds include CBF)
            Cx_r = C_r @ dq_r_k
            Cx_l = C_l @ dq_l_k
            cbf_viol_r = np.maximum(l_r - Cx_r, 0).sum()
            cbf_viol_l = np.maximum(l_l - Cx_l, 0).sum()
            if cbf_viol_r > 1e-3 or cbf_viol_l > 1e-3:
                continue

            # Progress check: predicted final error using linear model
            # ê_final^{(k)} = e_final - J * Δq^{(k)}
            e_pred_elbow_r = e_elbow_r_final[3:] - J_elbow_r_sub[3:, :] @ dq_r_k
            e_pred_hand_r = e_hand_r_final[3:] - J_hand_r_sub[3:, :] @ dq_r_k
            e_pred_elbow_l = e_elbow_l_final[3:] - J_elbow_l_sub[3:, :] @ dq_l_k
            e_pred_hand_l = e_hand_l_final[3:] - J_hand_l_sub[3:, :] @ dq_l_k

            e_pred_norm = np.sqrt(
                np.sum(e_pred_elbow_r**2) + np.sum(e_pred_hand_r**2) +
                np.sum(e_pred_elbow_l**2) + np.sum(e_pred_hand_l**2))

            # Require: ||ê_final^{(k)}|| ≤ ||e0|| - δ
            if e_pred_norm <= e0_norm - delta_progress:
                best_k = k
                break  # Found largest feasible α

        # Fallback: if no α satisfies progress, use smallest α (most conservative)
        if best_k < 0:
            best_k = 0

        dq_right, dq_left = all_solutions[best_k]

        # =====================================================================
        # Combine
        # =====================================================================
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
            print("WARNING: Final q_des/dq_des contains NaN/Inf! Returning current state.")
            return self.q, np.zeros(DOF, dtype=np.float32)

        self._ik_ratchet_q_des = q_des
        self._ik_ratchet_dq_des = dq_des

        return q_des, dq_des
