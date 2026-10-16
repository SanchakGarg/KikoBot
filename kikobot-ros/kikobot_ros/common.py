JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
# Ready pose used by the manual pick-and-place controller and simulation startup.
READY_POSE_RAD = [1.5707963268, -0.7853981634, 1.8325957146, 1.0471975512, -1.5882496193, 0.0]
# Ranges copied from Kikobot-mujoco/robot.xml (radians).
JOINT_LIMITS = [
    (-2.8797932658, 2.8797932658),
    (-2.1816615650, 2.1816615650),
    (-2.4434609528, 2.4434609528),
    (-2.4434609528, 2.4434609528),
    (-2.4434609528, 2.4434609528),
    (-3.0543261910, 3.0543261910),
]


def clip_targets(values):
    return [max(lo, min(hi, float(value))) for value, (lo, hi) in zip(values, JOINT_LIMITS)]
