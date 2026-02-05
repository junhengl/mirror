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
                       "Ry", "Rx", "Ry", "Rx", "Ry", "Rx", "Ry",
                       "Ry", "Rx", "Ry", "Rx", "Ry", "Rx", "Ry",
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
        j = body_index +1 
        
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
        J = self.compute_body_jacobian(end_effector_index, Xend)
        
        return rpypos, J
    
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
        We = 1*np.diag([0.0, 0.0, 0.0, 150.0, 150.0, 150.0]).astype(np.float64)  # position only
        
        # Hand weight matrix (6x6: orientation + position)
        Wh = 1*np.diag([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]).astype(np.float64)  # position only
        
        # COM weight matrix (6x6)
        Wc = np.eye(6, dtype=np.float64) * 50000
        
        # Joint regularization weight (for ||dq||^2_Wq)
        # Must be much smaller than task weights for good tracking
        Wq = np.eye(DOF, dtype=np.float64) * 100
        Wq[:6, :6] = 0  # no regularization on floating base (it's fixed anyway)
        
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
        #
        # P = sum(J^T W J) + Wq
        # c = sum(-J^T W e)  (the -2 factor is absorbed into 0.5 in QP standard form)
        # =====================================================================
        
        # P matrix (Hessian)
        P = np.zeros((DOF, DOF), dtype=np.float64)
        P += J_elbow_l.T @ We @ J_elbow_l
        P += J_elbow_r.T @ We @ J_elbow_r
        P += J_hand_l.T @ Wh @ J_hand_l
        P += J_hand_r.T @ Wh @ J_hand_r
        P += J_com.T @ Wc @ J_com
        P += Wq  # regularization
        
        # Make P symmetric and positive definite
        P = 0.5 * (P + P.T)
        P += np.eye(DOF) * 1e-6  # numerical stability
        
        # c vector (linear term): c = -J^T W e
        c = np.zeros(DOF, dtype=np.float64)
        c -= J_elbow_l.T @ We @ e_elbow_l
        c -= J_elbow_r.T @ We @ e_elbow_r
        c -= J_hand_l.T @ Wh @ e_hand_l
        c -= J_hand_r.T @ Wh @ e_hand_r
        c -= J_com.T @ Wc @ e_com
        # Note: Wq term contributes 0 to c since we minimize ||dq||^2 (reference is 0)
        
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
