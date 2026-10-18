import os

import mujoco
import numpy as np

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "scene.xml"
)

EE_BODY = "eof"
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# Reachable workspace: a sphere of 370 mm radius centered on the robot base.
BASE_ORIGIN = np.array([0.0, 0.0, 0.0])
REACH_RADIUS = 0.370

POS_TOL = 1e-4
MAX_ITERS = 200
DAMPING = 0.05
RESTARTS = 5

WRIST_ITERS = 150
WRIST_GAIN = 0.3
WRIST_DAMPING_SQ = 0.02


class IKSolver:
    def __init__(self, scene_path=SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.nv = self.model.nv
        self.jnt_lo = self.model.jnt_range[:, 0].copy()
        self.jnt_hi = self.model.jnt_range[:, 1].copy()
        self.home_qpos = np.zeros(self.nv)
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    def in_workspace(self, target_pos):
        return np.linalg.norm(np.asarray(target_pos) - BASE_ORIGIN) <= REACH_RADIUS

    def forward(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.ee_id].copy(), self.data.xquat[self.ee_id].copy()

    def solve(self, target_pos, point_dir=None, q_init=None):
        """Solve for target_pos. If point_dir is given, also swing the wrist's
        tool axis toward that world-frame direction using the arm's spare
        (redundant) degrees of freedom, without disturbing target_pos."""
        target_pos = np.asarray(target_pos, dtype=float)

        if not self.in_workspace(target_pos):
            return None, False

        if q_init is not None:
            seeds = [np.array(q_init, dtype=float)]
        else:
            seeds = [self.home_qpos.copy()]
            rng = np.random.default_rng(0)
            for _ in range(RESTARTS - 1):
                seeds.append(rng.uniform(self.jnt_lo, self.jnt_hi))

        for seed in seeds:
            q, ok = self._solve_position(target_pos, seed)
            if ok:
                if point_dir is not None:
                    q = self._align_wrist(q, target_pos, np.asarray(point_dir, dtype=float))
                return q, True

        return None, False

    def _solve_position(self, target_pos, q):
        for _ in range(MAX_ITERS):
            ee_pos, _ = self.forward(q)
            err = target_pos - ee_pos

            if np.linalg.norm(err) < POS_TOL:
                if np.all(q >= self.jnt_lo) and np.all(q <= self.jnt_hi):
                    return q, True
                return None, False

            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, ee_pos, self.ee_id)
            JJt = self._jacp @ self._jacp.T
            lam_sq = DAMPING * DAMPING * np.eye(3)
            dq = self._jacp.T @ np.linalg.solve(JJt + lam_sq, err)

            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)

        return None, False

    def _align_wrist(self, q, target_pos, point_dir):
        point_dir = point_dir / np.linalg.norm(point_dir)
        best_q, best_ang = q.copy(), np.pi

        for _ in range(WRIST_ITERS):
            ee_pos, ee_quat = self.forward(q)
            tool_axis = np.zeros(3)
            mujoco.mju_rotVecQuat(tool_axis, TOOL_AXIS_LOCAL, ee_quat)
            omega_des = np.cross(tool_axis, point_dir)
            ang = np.arccos(np.clip(np.dot(tool_axis, point_dir), -1.0, 1.0))

            if np.linalg.norm(target_pos - ee_pos) < 5e-4 and ang < best_ang:
                best_q, best_ang = q.copy(), ang

            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, ee_pos, self.ee_id)

            jp_pinv = self._jacp.T @ np.linalg.inv(self._jacp @ self._jacp.T + 1e-6 * np.eye(3))
            nullspace = np.eye(self.nv) - jp_pinv @ self._jacp

            jr_pinv = self._jacr.T @ np.linalg.inv(
                self._jacr @ self._jacr.T + WRIST_DAMPING_SQ * np.eye(3)
            )
            dq_orient = nullspace @ (jr_pinv @ omega_des)
            norm = np.linalg.norm(dq_orient)
            if norm > 0.1:
                dq_orient = dq_orient / norm * 0.1
            dq_pos = jp_pinv @ (target_pos - ee_pos)

            q = np.clip(q + WRIST_GAIN * dq_orient + dq_pos, self.jnt_lo, self.jnt_hi)

        return best_q
