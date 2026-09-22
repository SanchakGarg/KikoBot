import time

import mujoco
import mujoco.viewer
import numpy as np

from ik_solver import IKSolver, BASE_ORIGIN

POINT_A = (-0.15, 0.0, 0.05)
POINT_B = (0.15, 0.0, 0.05)
LIFT = 0.03

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


def solve_waypoint(ik, pos, label):
    point_dir = np.asarray(pos) - BASE_ORIGIN
    q, ok = ik.solve(pos, point_dir=point_dir)
    if not ok:
        print(f"{label} {pos} is not reachable")
    return q


def main():
    ik = IKSolver()
    model, data = ik.model, ik.data

    waypoints = []
    for name, pos in (("A", POINT_A), ("B", POINT_B)):
        down = pos
        up = (pos[0], pos[1], pos[2] + LIFT)
        q_down = solve_waypoint(ik, down, f"{name} down")
        q_up = solve_waypoint(ik, up, f"{name} up")
        if q_down is None or q_up is None:
            continue
        waypoints.append((f"{name} up", q_up))
        waypoints.append((f"{name} down", q_down))
        waypoints.append((f"{name} up", q_up))

    if len(waypoints) < 4:
        print("not enough reachable points, nothing to do")
        return

    q_start = waypoints[0][1]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _, q_target in waypoints:
                move_to(model, data, viewer, data.qpos.copy(), q_target, MOVE_DURATION)
                hold(model, data, viewer, HOLD_DURATION)


if __name__ == "__main__":
    main()
