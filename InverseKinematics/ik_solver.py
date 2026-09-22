import os

import mujoco
import numpy as np

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Kikobot-mujoco", "scene.xml"
)

EE_BODY = "eof"

# Reachable workspace: a sphere of 370 mm diameter centered on the robot base.
BASE_ORIGIN = np.array([0.0, 0.0, 0.0])
REACH_RADIUS = 0.185

POS_TOL = 1e-4
ROT_TOL = 1e-3
MAX_ITERS = 200
DAMPING = 0.05
RESTARTS = 5


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

    def solve(self, target_pos, target_quat=None, q_init=None):
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
            q, ok = self._solve_from(target_pos, target_quat, seed)
            if ok:
                return q, True

        return None, False

    def _solve_from(self, target_pos, target_quat, q):
        for _ in range(MAX_ITERS):
            ee_pos, ee_quat = self.forward(q)
            pos_err = target_pos - ee_pos

            if target_quat is not None:
                rot_err = np.zeros(3)
                mujoco.mju_subQuat(rot_err, target_quat, ee_quat)
                err = np.concatenate([pos_err, rot_err])
            else:
                err = pos_err

            if np.linalg.norm(pos_err) < POS_TOL and (
                target_quat is None or np.linalg.norm(err[3:]) < ROT_TOL
            ):
                if np.all(q >= self.jnt_lo) and np.all(q <= self.jnt_hi):
                    return q, True
                return None, False

            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, ee_pos, self.ee_id)
            J = self._jacp if target_quat is None else np.vstack([self._jacp, self._jacr])

            JJt = J @ J.T
            lam_sq = DAMPING * DAMPING * np.eye(JJt.shape[0])
            dq = J.T @ np.linalg.solve(JJt + lam_sq, err)

            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)

        return None, False
