"""
Hardware Pipeline Visualizer — zero-overhead MuJoCo viewer.

Runs a passive MuJoCo viewer on a **separate thread** that:
  • Loads the robot MJCF (visual only — no physics stepping)
  • Reads q_des from SharedState, converts KinDynLib→MuJoCo convention,
    and sets qpos so the mesh matches the commanded pose
  • Computes actual hand / elbow positions via KinDynLib FK (same FK the
    retargeting node uses) — no mj_forward needed
  • Renders desired / actual hand & elbow markers (same style as sim)
  • Refreshes at ~30 Hz — never touches the 1 kHz command loop

Usage (from integrated_hw_wbc.py):
    viz = HardwareVisualizer(shared_state, config)
    viz.start()          # non-blocking
    ...
    viz.stop()           # on shutdown
"""

import os
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── KinDynLib imports (same path setup as retargeting_node.py) ────────
_PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KINDYN_DIR = os.path.join(_PROJ_DIR, "KinDynLib")
if _KINDYN_DIR not in sys.path:
    sys.path.insert(0, _KINDYN_DIR)

import robot_const as themis_const           # noqa: E402
sys.modules['robot_const'] = themis_const
from robot_dynamics import Robot             # noqa: E402
from dynamics_lib import Xtrans             # noqa: E402

from real_time_sim.shared_state import SharedState, RetargetingOutput
from real_time_sim.config import PipelineConfig
from real_time_sim.joint_mapping import JointMapping


class HardwareVisualizer:
    """Non-blocking MuJoCo viewer for the hardware pipeline."""

    RENDER_HZ = 30.0  # target refresh rate

    # Link indices & offsets — must match retargeting_node.py
    HAND_R_LINK  = 23
    HAND_L_LINK  = 30
    ELBOW_R_LINK = 21
    ELBOW_L_LINK = 28

    HAND_R_OFFSET  = np.array([0.08, 0.0, 0.0], dtype=np.float64)
    HAND_L_OFFSET  = np.array([0.08, 0.0, 0.0], dtype=np.float64)
    ELBOW_R_OFFSET = np.array([0.0, 0.0, 0.0],  dtype=np.float64)
    ELBOW_L_OFFSET = np.array([0.0, 0.0, 0.0],  dtype=np.float64)

    def __init__(self, shared: SharedState, config: PipelineConfig,
                 model_path: str | None = None):
        self.shared = shared
        self.config = config
        self.joint_mapping = JointMapping(config.joint_mapping)
        self.base_height = config.sim.base_height

        # KinDynLib robot model (its own instance — thread-safe)
        self.robot = Robot()
        self.q_kin = np.zeros(themis_const.DOF, dtype=np.float64)
        self.dq_kin = np.zeros(themis_const.DOF, dtype=np.float64)

        # Resolve MJCF path
        if model_path is None:
            model_path = os.path.join(_PROJ_DIR, config.sim.model_path)
        self._model_path = model_path

        # Will be populated on start()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer: mujoco.viewer.Handle | None = None

        # Thread plumbing
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Marker positions (updated each frame)
        self._markers: dict[str, np.ndarray] = {}

    # ─── public API ──────────────────────────────────────────────────
    def start(self):
        """Load model, launch viewer, and start the render thread."""
        print(f"[Viz] Loading model from {self._model_path}")
        self.model = mujoco.MjModel.from_xml_path(self._model_path)
        self.data = mujoco.MjData(self.model)

        # Set robot base height (welded base — body_pos)
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
        if base_id >= 0:
            self.model.body_pos[base_id] = [0.0, 0.0, self.base_height]

        # Initial FK so mesh is valid before viewer opens
        mujoco.mj_forward(self.model, self.data)

        # Launch passive (non-blocking) viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False, show_right_ui=False,
        )
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [0.0, 0.0, self.base_height]
            self.viewer.cam.distance = 2.5
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 135

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="hw-viz")
        self._thread.start()
        print("[Viz] Viewer started (30 Hz)")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        print("[Viz] Stopped")

    def is_running(self) -> bool:
        if self.viewer is None:
            return False
        return self.viewer.is_running()

    # ─── render loop (background thread) ─────────────────────────────
    def _loop(self):
        dt = 1.0 / self.RENDER_HZ
        while not self._stop.is_set():
            if self.viewer is not None and not self.viewer.is_running():
                break  # user closed the window

            t0 = time.perf_counter()
            self._update()
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _update(self):
        """Read SharedState, compute FK via KinDynLib, update MuJoCo qpos, sync viewer."""
        retarget: RetargetingOutput = self.shared.get_retarget_output()

        base_offset = np.array([0.0, 0.0, self.base_height], dtype=np.float64)

        if retarget.valid:
            # ── 1. Compute actual end-effector positions via KinDynLib FK ──
            # q_des is 28 joints in KinDynLib convention.
            # Build full 34-DOF state vector: [base_6dof, joints_28]
            self.q_kin[6:6+28] = retarget.q_des
            self.q_kin[2] = self.base_height  # base Z in KinDynLib frame
            self.dq_kin[:] = 0.0
            self.robot.update(self.q_kin, self.dq_kin)

            x_hand_r  = self.robot.compute_forward_kinematics(self.HAND_R_LINK,  self.HAND_R_OFFSET)
            x_hand_l  = self.robot.compute_forward_kinematics(self.HAND_L_LINK,  self.HAND_L_OFFSET)
            x_elbow_r = self.robot.compute_forward_kinematics(self.ELBOW_R_LINK, self.ELBOW_R_OFFSET)
            x_elbow_l = self.robot.compute_forward_kinematics(self.ELBOW_L_LINK, self.ELBOW_L_OFFSET)

            # ── 2. Convert q_des from KinDynLib → MuJoCo/XML convention for mesh ──
            q_mujoco = self.joint_mapping.reverse_q(retarget.q_des)
            self.data.qpos[0:28] = q_mujoco
            mujoco.mj_forward(self.model, self.data)   # position mesh only

            # ── 3. Build markers ──
            self._markers = {
                # desired (from ZED tracking targets, already in world frame)
                "hand_l_des":  retarget.hand_l_des.copy(),
                "hand_r_des":  retarget.hand_r_des.copy(),
                "elbow_l_des": retarget.elbow_l_des.copy(),
                "elbow_r_des": retarget.elbow_r_des.copy(),
                # actual (from our own KinDynLib FK, positions are indices [3:6])
                "hand_l_act":  x_hand_l[3:6].astype(np.float64),
                "hand_r_act":  x_hand_r[3:6].astype(np.float64),
                "elbow_l_act": x_elbow_l[3:6].astype(np.float64),
                "elbow_r_act": x_elbow_r[3:6].astype(np.float64),
                # orientation
                "hand_l_orient_mat": retarget.hand_l_orient_mat.copy(),
                "hand_r_orient_mat": retarget.hand_r_orient_mat.copy(),
            }
        else:
            self._markers = {}

        self._render_markers()

        if self.viewer is not None:
            self.viewer.sync()

    # ─── marker rendering (same visual style as mujoco_sim.py) ───────
    def _render_markers(self):
        if self.viewer is None:
            return

        m = self._markers
        with self.viewer.lock():
            ngeom = 0

            # ── Desired hand markers as orientation arrows ───────────
            arrow_length = 0.15
            arrow_radius = 0.012
            cone_radius = 0.03
            cone_length = 0.05
            shaft_length = arrow_length - cone_length

            hand_arrow_defs = {
                "hand_l_des": ("hand_l_orient_mat", [1.00, 0.08, 0.18, 1.0]),
                "hand_r_des": ("hand_r_orient_mat", [0.08, 0.55, 1.00, 1.0]),
            }
            for key, (mat_key, color) in hand_arrow_defs.items():
                pos = m.get(key, np.zeros(3))
                orient = m.get(mat_key, np.eye(3))
                if np.any(pos != 0) and ngeom + 1 < self.viewer.user_scn.maxgeom:
                    arm_dir = orient[:, 2]
                    # shaft
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    g.size[:] = [arrow_radius, shaft_length * 0.5, 0]
                    g.pos[:] = pos + arm_dir * (shaft_length * 0.5)
                    g.mat[:] = orient
                    g.rgba[:] = color
                    ngeom += 1
                    # arrowhead
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    g.size[:] = [cone_radius, cone_length * 0.5, 0]
                    g.pos[:] = pos + arm_dir * shaft_length + arm_dir * (cone_length * 0.5)
                    g.mat[:] = orient
                    g.rgba[:] = color
                    ngeom += 1

            # ── Desired elbow markers (spheres) ──────────────────────
            for key, color in [
                ("elbow_l_des", [1.00, 0.18, 0.28, 1.0]),
                ("elbow_r_des", [0.18, 0.72, 1.00, 1.0]),
            ]:
                pos = m.get(key, np.zeros(3))
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.05, 0.05, 0.05]
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = color
                    ngeom += 1

            # ── Actual hand / elbow markers (slightly smaller) ───────
            for key, color in [
                ("hand_l_act",  [0.85, 0.06, 0.12, 0.90]),
                ("hand_r_act",  [0.06, 0.40, 0.72, 0.90]),
                ("elbow_l_act", [0.90, 0.12, 0.18, 0.90]),
                ("elbow_r_act", [0.14, 0.58, 0.82, 0.90]),
            ]:
                pos = m.get(key, np.zeros(3))
                if np.any(pos != 0) and ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[ngeom]
                    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    g.size[:] = [0.045, 0.045, 0.045]
                    g.pos[:] = pos
                    g.mat[:] = np.eye(3)
                    g.rgba[:] = color
                    ngeom += 1

            self.viewer.user_scn.ngeom = ngeom
