"""ROS 2 Cartesian pick-and-place controller using the KikoBot MuJoCo model."""

import math
import os
import select
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float64MultiArray

from .common import JOINT_LIMITS, JOINT_NAMES, READY_POSE_RAD, clip_targets


DEADZONE = 0.12
DAMPING = 0.035
MAX_IK_STEP_RAD = math.radians(2.0)
IK_PERIOD = 0.02
PLANE_TOLERANCE = math.sin(math.radians(1.0))
IK_POSITION_DEADBAND_M = 0.0005
IK_PLANE_DEADBAND = math.sin(math.radians(0.2))
WORLD_Z = np.array([0.0, 0.0, 1.0])
READY_POSE = np.asarray(READY_POSE_RAD)


class PickPlaceController(Node):
    def __init__(self, mode):
        super().__init__("pick_place_controller")
        self.declare_parameter("input_mode", mode)
        self.declare_parameter("joy_left_x_axis", 0)
        self.declare_parameter("joy_left_y_axis", 1)
        self.declare_parameter("joy_left_trigger_axis", 2)
        self.declare_parameter("joy_right_x_axis", 3)
        self.declare_parameter("joy_right_y_axis", 4)
        self.declare_parameter("joy_right_trigger_axis", 5)
        self.declare_parameter("joy_a_button", 0)
        self.declare_parameter("deadzone", DEADZONE)
        self.declare_parameter("start_in_ready_pose", False)
        self.mode = str(self.get_parameter("input_mode").value)
        self.model_path = os.path.join(
            get_package_share_directory("kikobot_ros"), "model", "scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.ik_data = mujoco.MjData(self.model)
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ], dtype=int)
        if np.any(self.joint_ids < 0):
            raise RuntimeError("MuJoCo model must contain J1 through J6")
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids[:4]]
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.ik_qpos_ids = self.model.jnt_qposadr[self.joint_ids[:4]]
        self.eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "j5")
        if self.eef_id < 0:
            raise RuntimeError("MuJoCo model must contain the j5 wrist-pivot body")
        self.low = self.model.jnt_range[self.joint_ids, 0].copy()
        self.high = self.model.jnt_range[self.joint_ids, 1].copy()
        self.targets = np.zeros(6)
        self.cartesian_target = np.zeros(3)
        self.initialized = False
        self.latest_joy = None
        self.last_joy_time = 0.0
        self.last_tick = time.monotonic()
        self.manual_wrist_moved = False
        self.cartesian_moved = False
        self.plane_blocked = False
        self.last_center_button = False
        self.key_lock = threading.Lock()
        self.keys_down = {}
        self.key_events = []
        self.publisher = self.create_publisher(Float64MultiArray, f"/input/{self.mode}", 10)
        self.state_sub = self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.joy_sub = None
        if self.mode == "joystick":
            self.joy_sub = self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.timer = self.create_timer(IK_PERIOD, self.update)
        if self.mode == "keyboard":
            threading.Thread(target=self.read_keyboard, daemon=True).start()
            self.get_logger().info(
                "Cartesian keys: arrows XY, Q/E Z, R/F J5, Z/X J6, C centre wrist, H ready pose"
            )
        else:
            self.get_logger().info("Xbox-style /joy input: left stick XY, triggers Z, right stick wrist, A centre")

    def on_state(self, message):
        if self.initialized:
            return
        by_name = dict(zip(message.name, message.position))
        if not all(name in by_name for name in JOINT_NAMES):
            return
        self.targets = clip_targets([by_name[name] for name in JOINT_NAMES])
        if bool(self.get_parameter("start_in_ready_pose").value):
            self.targets = np.clip(READY_POSE, self.low, self.high)
        self.ik_data.qpos[:] = self.model.qpos0
        self.ik_data.qpos[self.qpos_ids] = self.targets
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)
        self.cartesian_target = self.ik_data.xpos[self.eef_id].copy()
        self.initialized = True
        self.get_logger().info("Initialized Cartesian target from measured joint state")
        self.publish_targets()

    def on_joy(self, message):
        self.latest_joy = message
        self.last_joy_time = time.monotonic()

    def read_keyboard(self):
        if not sys.stdin.isatty():
            self.get_logger().error("Keyboard control needs a TTY; run the launch command in an interactive terminal")
            return
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                key = sys.stdin.read(1)
                if key == "\x1b":
                    # Windows Terminal/WSL may send either CSI (ESC [ A) or
                    # application cursor mode (ESC O A). Allow the sequence
                    # enough time to arrive across the WSL terminal bridge.
                    sequence = ""
                    for _ in range(2):
                        more, _, _ = select.select([sys.stdin], [], [], 0.03)
                        if more:
                            sequence += sys.stdin.read(1)
                    key = {
                        "[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT",
                        "OA": "UP", "OB": "DOWN", "OC": "RIGHT", "OD": "LEFT",
                    }.get(sequence, "")
                else:
                    key = key.lower()
                with self.key_lock:
                    if key in ("c", "h"):
                        self.key_events.append(key)
                    else:
                        self.keys_down[key] = time.monotonic()
        except Exception as exc:
            self.get_logger().error(f"Keyboard reader stopped: {exc}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def handle_key(self, key):
        if not self.initialized:
            return
        translation = 0.005
        wrist_step = math.radians(3.0)
        moves = {
            "RIGHT": np.array([translation, 0.0, 0.0]),
            "LEFT": np.array([-translation, 0.0, 0.0]),
            "UP": np.array([0.0, translation, 0.0]),
            "DOWN": np.array([0.0, -translation, 0.0]),
        }
        if key in moves:
            self.translate(moves[key])
        elif key == "q":
            self.translate(np.array([0.0, 0.0, translation]))
        elif key == "e":
            self.translate(np.array([0.0, 0.0, -translation]))
        elif key in ("r", "f", "z", "x"):
            joint = 4 if key in ("r", "f") else 5
            sign = 1.0 if key in ("r", "z") else -1.0
            self.move_wrist(joint, sign * wrist_step)
        elif key == "c":
            self.targets[4:] = 0.0
            self.manual_wrist_moved = True
        elif key == "h":
            self.reset_ready()

    def update_keyboard(self, keys, events, dt):
        if not self.initialized:
            return
        dx = ("RIGHT" in keys) - ("LEFT" in keys)
        dy = ("UP" in keys) - ("DOWN" in keys)
        dz = ("q" in keys) - ("e" in keys)
        direction = np.array([dx, dy, dz], dtype=float)
        magnitude = np.linalg.norm(direction)
        if magnitude > 1.0:
            direction /= magnitude
        if magnitude:
            self.translate(direction * (0.10 * dt))
        self.move_wrist(4, (("r" in keys) - ("f" in keys)) * math.radians(60.0) * dt)
        self.move_wrist(5, (("z" in keys) - ("x" in keys)) * math.radians(60.0) * dt)
        for key in events:
            if key == "c":
                self.targets[4:] = 0.0
                self.manual_wrist_moved = True
            elif key == "h":
                self.reset_ready()

    def translate(self, delta):
        candidate = self.cartesian_target + delta
        candidate[2] = np.clip(candidate[2], 0.06, 0.55)
        xy_norm = np.linalg.norm(candidate[:2])
        if xy_norm > 0.42:
            candidate[:2] *= 0.42 / xy_norm
        self.cartesian_target = candidate
        self.cartesian_moved = True

    def update_joystick(self, dt):
        joy = self.latest_joy
        if joy is None or time.monotonic() - self.last_joy_time > 0.5:
            return
        def axis(name):
            index = int(self.get_parameter(name).value)
            return float(joy.axes[index]) if index < len(joy.axes) else 0.0
        def deadzone(value):
            return value if abs(value) >= float(self.get_parameter("deadzone").value) else 0.0
        lx = deadzone(axis("joy_left_x_axis"))
        ly = deadzone(axis("joy_left_y_axis"))
        rx = deadzone(axis("joy_right_x_axis"))
        ry = deadzone(axis("joy_right_y_axis"))
        # sensor_msgs/Joy trigger axes are commonly 0 at rest and +1 pressed.
        lt = max(0.0, min(1.0, axis("joy_left_trigger_axis")))
        rt = max(0.0, min(1.0, axis("joy_right_trigger_axis")))
        delta = np.array([lx, -ly, rt - lt]) * 0.12 * dt
        if np.any(delta):
            self.translate(delta)
        self.move_wrist(4, rx * math.radians(75.0) * dt)
        self.move_wrist(5, -ry * math.radians(75.0) * dt)
        button = int(self.get_parameter("joy_a_button").value)
        pressed = button < len(joy.buttons) and joy.buttons[button] != 0
        if pressed and not self.last_center_button:
            self.targets[4:] = 0.0
            self.manual_wrist_moved = True
        self.last_center_button = pressed

    def move_wrist(self, index, delta):
        if abs(delta) < 1e-12:
            return
        self.sync_ik_state()
        if index == 4 and abs(self.ik_data.xaxis[self.joint_ids[4], 2]) > PLANE_TOLERANCE:
            if not self.plane_blocked:
                self.get_logger().warning("J5 locked: its rotation plane is not perpendicular to world XY")
            self.plane_blocked = True
            return
        previous = self.targets[index]
        self.targets[index] = np.clip(previous + delta, self.low[index], self.high[index])
        if not np.isclose(previous, self.targets[index]):
            self.manual_wrist_moved = True

    def sync_ik_state(self):
        self.ik_data.qpos[:] = self.model.qpos0
        self.ik_data.qpos[self.qpos_ids] = self.targets
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)

    def candidate_preserves_plane(self, candidate):
        self.sync_ik_state()
        self.ik_data.qpos[self.ik_qpos_ids] = candidate
        mujoco.mj_forward(self.model, self.ik_data)
        return abs(self.ik_data.xaxis[self.joint_ids[4], 2]) <= PLANE_TOLERANCE

    def solve_ik(self):
        self.sync_ik_state()
        position_error = self.cartesian_target - self.ik_data.xpos[self.eef_id]
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.ik_data, jac_position, jac_rotation, self.eef_id)
        j5_axis = self.ik_data.xaxis[self.joint_ids[4]].copy()
        # Do not chase solver noise once the requested pose is reached.
        if (np.linalg.norm(position_error) <= IK_POSITION_DEADBAND_M
                and abs(j5_axis[2]) <= IK_PLANE_DEADBAND):
            return
        plane_jac = np.cross(j5_axis, WORLD_Z) @ jac_rotation
        task_jac = np.vstack((jac_position[:, self.dof_ids], plane_jac[self.dof_ids]))
        task_error = np.concatenate((position_error, [-j5_axis[2]]))
        if np.linalg.cond(task_jac) < 1e7:
            delta_q = np.linalg.solve(task_jac, task_error)
        else:
            regularized = task_jac @ task_jac.T + DAMPING**2 * np.eye(4)
            delta_q = task_jac.T @ np.linalg.solve(regularized, task_error)
        maximum = np.max(np.abs(delta_q))
        if maximum > MAX_IK_STEP_RAD:
            delta_q *= MAX_IK_STEP_RAD / maximum
        scale = 1.0
        for _ in range(14):
            candidate = np.clip(self.targets[:4] + scale * delta_q, self.low[:4], self.high[:4])
            if self.candidate_preserves_plane(candidate):
                self.targets[:4] = candidate
                self.plane_blocked = scale < 0.999
                return
            scale *= 0.5
        if not self.plane_blocked:
            self.get_logger().warning("Arm held at J5-plane boundary; reverse Cartesian motion to retreat")
        self.plane_blocked = True

    def reset_ready(self):
        self.targets = np.clip(READY_POSE, self.low, self.high)
        self.sync_ik_state()
        self.cartesian_target = self.ik_data.xpos[self.eef_id].copy()
        self.cartesian_moved = True

    def publish_targets(self):
        message = Float64MultiArray()
        message.data = clip_targets(self.targets)
        self.publisher.publish(message)

    def update(self):
        if not self.initialized:
            return
        now = time.monotonic()
        dt = min(0.05, max(0.0, now - self.last_tick))
        self.last_tick = now
        self.manual_wrist_moved = False
        self.cartesian_moved = False
        if self.mode == "joystick":
            self.update_joystick(dt)
        else:
            with self.key_lock:
                now = time.monotonic()
                # Terminal auto-repeat is the only held-key signal available. Expire
                # it quickly after key-up so the arm never continues on stale input.
                self.keys_down = {key: stamp for key, stamp in self.keys_down.items() if now - stamp < 0.16}
                keys = set(self.keys_down)
                events = list(self.key_events)
                self.key_events.clear()
            self.update_keyboard(keys, events, dt)
        wrist_only = self.manual_wrist_moved and not self.cartesian_moved
        if wrist_only:
            self.sync_ik_state()
            self.cartesian_target = self.ik_data.xpos[self.eef_id].copy()
        elif self.cartesian_moved:
            # A stationary input means hold the last joint target exactly.
            # Re-solving IK while idle can turn tiny numerical/model errors into
            # unwanted joint motion (most visible as J5 drift).
            self.solve_ik()
        self.publish_targets()


def main_for_mode(mode, args=None):
    rclpy.init(args=args)
    node = PickPlaceController(mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    main_for_mode("keyboard", args)
