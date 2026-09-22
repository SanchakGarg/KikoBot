import time

import mujoco
import mujoco.viewer
import numpy as np

from ik_solver import IKSolver

IDLE_QPOS = np.zeros(6)
POINT_A = (-0.15, 0.08, 0.06)
POINT_B = (0.15, -0.08, 0.06)

MOVE_DURATION = 1.5
HOLD_DURATION = 0.5


def move_to(model, data, viewer, q_start, q_target, duration):
    steps = max(1, int(duration / model.opt.timestep))
    for i in range(1, steps + 1):
        alpha = i / steps
        data.ctrl[:] = q_start + alpha * (q_target - q_start)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def hold(model, data, viewer, duration):
    steps = max(1, int(duration / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def main():
    ik = IKSolver()
    model, data = ik.model, ik.data

    targets = {}
    for name, pos in (("A", POINT_A), ("B", POINT_B)):
        q, ok = ik.solve(pos)
        if not ok:
            print(f"point {name} {pos} is not reachable, skipping")
            continue
        targets[name] = q

    if not targets:
        print("no reachable points given, nothing to do")
        return

    order = [name for name in ("A", "B") if name in targets]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = IDLE_QPOS
    data.ctrl[:] = IDLE_QPOS
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for name in order:
                move_to(model, data, viewer, data.qpos.copy(), IDLE_QPOS, MOVE_DURATION)
                hold(model, data, viewer, HOLD_DURATION)
                move_to(model, data, viewer, data.qpos.copy(), targets[name], MOVE_DURATION)
                hold(model, data, viewer, HOLD_DURATION)


if __name__ == "__main__":
    main()
