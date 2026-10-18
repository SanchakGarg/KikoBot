"""Interactive Cartesian pick-and-place controller for the Kiko-Bot MuJoCo model.

J1--J4 are solved with differential inverse kinematics.  J5 and J6 are
deliberately excluded from the IK solve: they are manual wrist axes and start
at their centred (0 rad) positions.
"""

from __future__ import annotations

import os
import time
import ctypes
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

try:
    import glfw  # Installed with MuJoCo on the standard desktop installation.
except ImportError:  # Keyboard control remains available without gamepad support.
    glfw = None


from kikobot import PROJECT_ROOT, SCENE_PATH

# The viewer callback reports presses only; Windows polls held state instead.
KEYBOARD_TRANSLATION_STEP_M = 0.005
KEYBOARD_WRIST_STEP_RAD = np.deg2rad(3.0)
KEYBOARD_TRANSLATION_SPEED_MPS = 0.10
KEYBOARD_WRIST_SPEED_RADPS = np.deg2rad(60.0)

# Xbox controls are velocity commands, multiplied by elapsed real time.
GAMEPAD_TRANSLATION_SPEED_MPS = 0.12
GAMEPAD_WRIST_SPEED_RADPS = np.deg2rad(75.0)
DEADZONE = 0.12

# Jaw travel per second while held. Full stroke ~27 mm, so roughly a second
# to close, and the jaws can be stopped part-way onto an object.
GRIPPER_SPEED_MPS = 0.03
GRIPPER_STEP_M = 0.0015

# Differential IK tuning.
DAMPING = 0.035
MAX_IK_STEP_RAD = np.deg2rad(2.0)
IK_UPDATE_PERIOD_S = 0.02
# Run on the kinematic copy during a reset: the CAD pose puts the gripper
# below the table, so the arm is placed at the lifted pose, not driven there.
IK_WARM_START_ITERATIONS = 300
PLANE_AXIS_Z_TOLERANCE = np.sin(np.deg2rad(1.0))

WORLD_Z = np.array([0.0, 0.0, 1.0])
# The all-zero keyframe is near-singular for Cartesian motion. This bent pose
# moves well locally and starts J5's hinge axis horizontal.
READY_POSE_RAD = np.deg2rad([90.0, -45.0, 105.0, 60.0, -91.0, 0.0])

# Keeps targets inside the physical 540 mm envelope and above the base/floor.
WORKSPACE_XY_RADIUS_M = 0.42
WORKSPACE_MIN_Z_M = 0.06
WORKSPACE_MAX_Z_M = 0.55

# The gripper hangs ~0.10 m below the J5 task point, so start the target
# high enough that the tip clears the table.
READY_POSE_START_Z_M = 0.20

# Windows virtual-key codes, so polling does not depend on GLFW internals.
KEY_UP = 0x26
KEY_DOWN = 0x28
KEY_LEFT = 0x25
KEY_RIGHT = 0x27


@dataclass
class Controls:
    """Mutable commanded pose and Cartesian target."""

    target_position: np.ndarray
    joint_targets: np.ndarray
    gripper_ctrl: float = 0.0


class PickPlaceController:
    """Maps input devices to a Cartesian target and solves the arm IK."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        # Independent kinematic state: the arm sags slightly under gravity
        # between updates, and planning from that made the plane gate misfire.
        self.ik_data = mujoco.MjData(model)

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("J1", "J2", "J3", "J4", "J5", "J6")
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("The model must contain joints J1 through J6.")

        self.ik_joint_ids = self.joint_ids[:4]
        self.ik_dof_ids = model.jnt_dofadr[self.ik_joint_ids]
        self.ik_qpos_ids = model.jnt_qposadr[self.ik_joint_ids]
        self.wrist_qpos_ids = model.jnt_qposadr[self.joint_ids[4:]]
        # Task point is the J5 wrist pivot, not `eof`: J1--J4 then only
        # position the arm, leaving J5/J6 manual.
        self.eef_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "j5"
        )
        if self.eef_body_id < 0:
            raise RuntimeError("The model must contain the J5 body.")

        # The finger midpoint is where an object actually ends up. Each mesh
        # appears as both a visual and a collision geom; either will do.
        finger_geoms = {}
        for geom_id in range(model.ngeom):
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name in ("gripper_l", "gripper_r"):
                finger_geoms.setdefault(mesh_name, geom_id)
        self.finger_geom_ids = [
            finger_geoms[name] for name in ("gripper_l", "gripper_r")
            if name in finger_geoms
        ]

        self.joint_low = model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = model.jnt_range[self.joint_ids, 1].copy()
        # One servo on finger_l; finger_r mirrors it via the model equality.
        # Limits come from the actuator's ctrlrange so a re-export stays right.
        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_l"
        )
        if self.gripper_actuator_id >= 0:
            closed, opened = model.actuator_ctrlrange[self.gripper_actuator_id]
            self.gripper_closed_ctrl = float(closed)
            self.gripper_open_ctrl = float(opened)
        else:
            self.gripper_closed_ctrl = self.gripper_open_ctrl = 0.0

        self.last_gamepad_present = False
        self.last_a_pressed = False
        self.next_ik_time = 0.0
        self.last_key_down = {"c": False, "h": False}
        self.manual_wrist_moved = False
        self.cartesian_target_moved = False
        self.plane_constraint_blocked = False
        try:
            self.get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        except AttributeError:
            self.get_async_key_state = None

        # The model's reset pose is the CAD lower stop; start bent instead.
        self.reset_to_ready_pose()

    def reset_to_ready_pose(self) -> None:
        """Reset to a bent Cartesian-control pose and centre both wrist axes."""
        home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home_key < 0:
            raise RuntimeError("The robot model does not define the 'home' keyframe.")

        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[self.model.jnt_qposadr[self.joint_ids]] = READY_POSE_RAD
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        targets = READY_POSE_RAD.copy()  # J5 and J6 are centred by default.
        start_position = self.data.xpos[self.eef_body_id].copy()
        start_position[2] = np.clip(
            READY_POSE_START_Z_M, WORKSPACE_MIN_Z_M, WORKSPACE_MAX_Z_M
        )
        self.controls = Controls(
            target_position=start_position,
            joint_targets=targets,
            gripper_ctrl=self.gripper_open_ctrl,  # Start with the jaws open.
        )
        # Converge on the kinematic copy, then place the arm there. Letting
        # physics drive it up instead buries the gripper in the table.
        for _ in range(IK_WARM_START_ITERATIONS):
            self.solve_ik_step()
        self.data.qpos[self.model.jnt_qposadr[self.joint_ids]] = (
            self.controls.joint_targets
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.plane_constraint_blocked = False
        self.next_ik_time = self.data.time
        self.apply_controls()

    def apply_controls(self) -> None:
        """Send the position-servo targets to the arm and gripper actuators."""
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            if name in {"J1", "J2", "J3", "J4", "J5", "J6"}:
                joint_index = int(name[1]) - 1
                self.data.ctrl[actuator_id] = self.controls.joint_targets[joint_index]
        if self.gripper_actuator_id >= 0:
            self.data.ctrl[self.gripper_actuator_id] = self.controls.gripper_ctrl

    def grasp_centre(self, data: mujoco.MjData | None = None) -> np.ndarray:
        """Midpoint between the fingers: where a grasped object ends up."""
        source = self.data if data is None else data
        left, right = self.finger_geom_ids
        return 0.5 * (source.geom_xpos[left] + source.geom_xpos[right])

    def gripper_is_open(self) -> bool:
        midpoint = 0.5 * (self.gripper_open_ctrl + self.gripper_closed_ctrl)
        return self.controls.gripper_ctrl > midpoint

    def move_gripper(self, amount: float) -> None:
        """Open (positive) or close (negative) the jaws by `amount` metres.

        Gradual rather than a toggle, so the jaws can be stopped as soon as
        they meet the object instead of always driving to a hard stop.
        """
        if self.gripper_actuator_id < 0 or abs(amount) < 1e-12:
            return
        span = self.gripper_open_ctrl - self.gripper_closed_ctrl
        low = min(self.gripper_open_ctrl, self.gripper_closed_ctrl)
        high = max(self.gripper_open_ctrl, self.gripper_closed_ctrl)
        self.controls.gripper_ctrl = float(
            np.clip(self.controls.gripper_ctrl + amount * np.sign(span), low, high)
        )

    def gripper_opening_m(self) -> float:
        """Commanded jaw travel from fully closed, in metres."""
        return abs(self.controls.gripper_ctrl - self.gripper_closed_ctrl)

    def toggle_gripper(self) -> None:
        """Flip the jaws between fully open and fully closed."""
        if self.gripper_actuator_id < 0:
            return
        self.controls.gripper_ctrl = (
            self.gripper_closed_ctrl
            if self.gripper_is_open()
            else self.gripper_open_ctrl
        )
        print("Gripper " + ("open." if self.gripper_is_open() else "closed."))

    def translate_target(self, delta: np.ndarray) -> None:
        """Move the Cartesian target, keeping it in a conservative work volume."""
        if not np.any(delta):
            return
        candidate = self.controls.target_position + delta
        candidate[2] = np.clip(candidate[2], WORKSPACE_MIN_Z_M, WORKSPACE_MAX_Z_M)
        xy_norm = np.linalg.norm(candidate[:2])
        if xy_norm > WORKSPACE_XY_RADIUS_M:
            candidate[:2] *= WORKSPACE_XY_RADIUS_M / xy_norm
        self.controls.target_position = candidate
        self.cartesian_target_moved = True

    def j5_rotation_plane_is_vertical(self) -> bool:
        """A horizontal J5 axis means the corresponding rotation plane is vertical."""
        self._sync_ik_state()
        return abs(self.ik_data.xaxis[self.joint_ids[4], 2]) <= PLANE_AXIS_Z_TOLERANCE

    def _sync_ik_state(self) -> None:
        """Forward-kinematics state from the current commanded joint angles."""
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[self.model.jnt_qposadr[self.joint_ids]] = (
            self.controls.joint_targets
        )
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)

    def move_wrist(self, joint_index: int, delta: float) -> bool:
        """Manually change J5/J6; reject J5 if its plane is not vertical."""
        if joint_index not in (4, 5):
            raise ValueError("Only J5 and J6 are manual wrist joints.")
        if abs(delta) < 1e-12:
            return False
        if joint_index == 4 and not self.j5_rotation_plane_is_vertical():
            if not self.plane_constraint_blocked:
                print("J5 locked: its rotation plane is not perpendicular to XY.")
            self.plane_constraint_blocked = True
            return False

        previous = self.controls.joint_targets[joint_index]
        self.controls.joint_targets[joint_index] = np.clip(
            self.controls.joint_targets[joint_index] + delta,
            self.joint_low[joint_index],
            self.joint_high[joint_index],
        )
        changed = not np.isclose(previous, self.controls.joint_targets[joint_index])
        self.manual_wrist_moved |= changed
        return changed

    def begin_input_frame(self) -> None:
        """Clear per-frame input flags before polling keyboard and controller."""
        self.manual_wrist_moved = False
        self.cartesian_target_moved = False

    def hold_cartesian_target_at_current_pose(self) -> None:
        """Prevent a manual wrist motion from causing automatic arm compensation."""
        self.controls.target_position = self.data.xpos[self.eef_body_id].copy()

    def _candidate_preserves_j5_plane(self, candidate_q: np.ndarray) -> bool:
        """Test a proposed J1--J4 target without permanently changing state."""
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[self.model.jnt_qposadr[self.joint_ids]] = (
            self.controls.joint_targets
        )
        self.ik_data.qpos[self.ik_qpos_ids] = candidate_q
        mujoco.mj_forward(self.model, self.ik_data)
        return (
            abs(self.ik_data.xaxis[self.joint_ids[4], 2])
            <= PLANE_AXIS_Z_TOLERANCE
        )

    def solve_ik_step(self) -> None:
        """Perform one damped-least-squares IK update for J1--J4.

        Besides XYZ position, the fourth task keeps J5's world-Z axis component
        at zero. Therefore J5's rotation plane remains perpendicular to XY.
        J5/J6 remain manual; wrist movement never causes arm compensation.
        """
        self._sync_ik_state()
        current_position = self.ik_data.xpos[self.eef_body_id].copy()
        position_error = self.controls.target_position - current_position

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model,
            self.ik_data,
            jacobian_position,
            jacobian_rotation,
            self.eef_body_id,
        )

        # d(axis_z)/dq = (axis x world_z)^T * angular_jacobian.
        j5_axis_world = self.ik_data.xaxis[self.joint_ids[4]].copy()
        plane_constraint_jacobian = np.cross(j5_axis_world, WORLD_Z) @ jacobian_rotation
        task_jacobian = np.vstack(
            (
                jacobian_position[:, self.ik_dof_ids],
                plane_constraint_jacobian[self.ik_dof_ids],
            )
        )
        task_error = np.concatenate((position_error, [-j5_axis_world[2]]))

        # Four joints, four task equations: solve exactly when well-conditioned
        # so damping cannot soften the J5-plane condition, else fall back to DLS.
        if np.linalg.cond(task_jacobian) < 1e7:
            delta_q = np.linalg.solve(task_jacobian, task_error)
        else:
            regularized_task_space = (
                task_jacobian @ task_jacobian.T
                + (DAMPING**2) * np.eye(task_jacobian.shape[0])
            )
            delta_q = task_jacobian.T @ np.linalg.solve(
                regularized_task_space, task_error
            )
        max_change = np.max(np.abs(delta_q))
        if max_change > MAX_IK_STEP_RAD:
            delta_q *= MAX_IK_STEP_RAD / max_change

        # A differential step leaves a residual, so backtrack until the
        # candidate satisfies the plane constraint. If none does, keep the
        # requested target so a reverse command can retreat from the boundary.
        accepted_targets = None
        accepted_scale = 0.0
        scale = 1.0
        for _ in range(14):
            candidate_targets = np.clip(
                self.controls.joint_targets[:4] + scale * delta_q,
                self.joint_low[:4],
                self.joint_high[:4],
            )
            if self._candidate_preserves_j5_plane(candidate_targets):
                accepted_targets = candidate_targets
                accepted_scale = scale
                break
            scale *= 0.5

        if accepted_targets is None:
            if not self.plane_constraint_blocked:
                print("Arm held at plane boundary; reverse the Cartesian command to retreat.")
            self.plane_constraint_blocked = True
            return

        self.controls.joint_targets[:4] = accepted_targets
        self.plane_constraint_blocked = accepted_scale < 0.999

    def solve_ik_if_due(self) -> None:
        """Run IK at 50 Hz, giving the physical position servos time to track."""
        if self.data.time + 1e-9 >= self.next_ik_time:
            self.solve_ik_step()
            self.next_ik_time = self.data.time + IK_UPDATE_PERIOD_S

    def _key_is_down(self, virtual_key: int) -> bool:
        """Return the physical key state on Windows, or False when unavailable."""
        return bool(
            self.get_async_key_state is not None
            and (self.get_async_key_state(virtual_key) & 0x8000)
        )

    def update_keyboard(self, dt: float) -> None:
        """Apply continuous Windows keyboard motion while the keys are held."""
        if self.get_async_key_state is None:
            return

        # Arrows move across the table, Q/E for height. WASD is avoided
        # because the MuJoCo viewer binds those itself.
        motion = np.array(
            [
                float(self._key_is_down(KEY_UP)) - float(self._key_is_down(KEY_DOWN)),
                float(self._key_is_down(KEY_LEFT))
                - float(self._key_is_down(KEY_RIGHT)),
                float(self._key_is_down(ord("Q"))) - float(self._key_is_down(ord("E"))),
            ]
        )
        # Diagonals must not outpace the axes.
        motion_norm = np.linalg.norm(motion)
        if motion_norm > 1.0:
            motion /= motion_norm
        self.translate_target(motion * KEYBOARD_TRANSLATION_SPEED_MPS * dt)

        j5_direction = float(self._key_is_down(ord("R"))) - float(
            self._key_is_down(ord("F"))
        )
        j6_direction = float(self._key_is_down(ord("Z"))) - float(
            self._key_is_down(ord("X"))
        )
        self.move_wrist(4, j5_direction * KEYBOARD_WRIST_SPEED_RADPS * dt)
        self.move_wrist(5, j6_direction * KEYBOARD_WRIST_SPEED_RADPS * dt)

        # Held keys, so the jaws can be stopped part-way onto an object.
        jaw_direction = float(self._key_is_down(ord("O"))) - float(
            self._key_is_down(ord("P"))
        )
        self.move_gripper(jaw_direction * GRIPPER_SPEED_MPS * dt)

        # C and H are one-shot actions rather than repeated actions.
        for key, virtual_key in (("c", ord("C")), ("h", ord("H"))):
            down = self._key_is_down(virtual_key)
            if down and not self.last_key_down[key]:
                if key == "c":
                    self.controls.joint_targets[4:] = 0.0
                else:
                    self.reset_to_ready_pose()
            self.last_key_down[key] = down

    def on_key(self, keycode: int) -> None:
        """Fallback for non-Windows systems, which use key-repeat events."""
        step = KEYBOARD_TRANSLATION_STEP_M
        wrist_step = KEYBOARD_WRIST_STEP_RAD
        arrow_map = {
            265: np.array([step, 0.0, 0.0]),   # GLFW_KEY_UP
            264: np.array([-step, 0.0, 0.0]),  # GLFW_KEY_DOWN
            263: np.array([0.0, step, 0.0]),   # GLFW_KEY_LEFT
            262: np.array([0.0, -step, 0.0]),  # GLFW_KEY_RIGHT
        }
        if keycode in arrow_map:
            self.translate_target(arrow_map[keycode])
            return

        key = chr(keycode).lower() if 32 <= keycode <= 126 else ""
        if key == "q":
            self.translate_target(np.array([0.0, 0.0, step]))
        elif key == "e":
            self.translate_target(np.array([0.0, 0.0, -step]))
        elif key == "r":
            self.move_wrist(4, wrist_step)   # Manual J5 positive rotation.
        elif key == "f":
            self.move_wrist(4, -wrist_step)  # Manual J5 negative rotation.
        elif key == "z":
            self.move_wrist(5, wrist_step)   # Manual J6 positive rotation.
        elif key == "x":
            self.move_wrist(5, -wrist_step)  # Manual J6 negative rotation.
        elif key == "o":
            self.move_gripper(GRIPPER_STEP_M)
        elif key == "p":
            self.move_gripper(-GRIPPER_STEP_M)
        elif key == "c":
            self.controls.joint_targets[4:] = 0.0
        elif key == "h":
            self.reset_to_ready_pose()

    @staticmethod
    def _deadzone(value: float) -> float:
        return value if abs(value) >= DEADZONE else 0.0

    def update_gamepad(self, dt: float) -> None:
        """Apply an attached Xbox-compatible GLFW gamepad, if one is present."""
        if glfw is None:
            return
        try:
            present = glfw.joystick_is_gamepad(glfw.JOYSTICK_1)
            state = glfw.get_gamepad_state(glfw.JOYSTICK_1) if present else None
        except Exception:  # No initialized GLFW context or no readable joystick.
            present, state = False, None

        if present != self.last_gamepad_present:
            print("Xbox-compatible controller " + ("connected." if present else "disconnected."))
            self.last_gamepad_present = present
        if state is None:
            return

        axes = state.axes
        left_x = self._deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_X])
        left_y = self._deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        right_x = self._deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        right_y = self._deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_Y])
        # GLFW maps Xbox trigger rest to -1 and full press to +1.
        left_trigger = (axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) * 0.5
        right_trigger = (axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) * 0.5

        self.translate_target(
            np.array(
                [
                    left_x * GAMEPAD_TRANSLATION_SPEED_MPS * dt,
                    -left_y * GAMEPAD_TRANSLATION_SPEED_MPS * dt,
                    (right_trigger - left_trigger) * GAMEPAD_TRANSLATION_SPEED_MPS * dt,
                ]
            )
        )
        self.move_wrist(4, right_x * GAMEPAD_WRIST_SPEED_RADPS * dt)
        self.move_wrist(5, -right_y * GAMEPAD_WRIST_SPEED_RADPS * dt)

        a_pressed = state.buttons[glfw.GAMEPAD_BUTTON_A] == glfw.PRESS
        if a_pressed and not self.last_a_pressed:
            self.controls.joint_targets[4:] = 0.0
        self.last_a_pressed = a_pressed

        # Bumpers work the jaws gradually, matching the O/P keys.
        jaw_direction = float(
            state.buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] == glfw.PRESS
        ) - float(state.buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] == glfw.PRESS)
        self.move_gripper(jaw_direction * GRIPPER_SPEED_MPS * dt)

    def status(self) -> str:
        """Short state shown periodically in the terminal."""
        position_mm = 1000.0 * self.data.xpos[self.eef_body_id]
        wrist_deg = np.rad2deg(self.controls.joint_targets[4:])
        plane_state = "vertical" if self.j5_rotation_plane_is_vertical() else "LOCKED"
        return (
            f"target: {1000 * self.controls.target_position[0]:7.1f} "
            f"{1000 * self.controls.target_position[1]:7.1f} "
            f"{1000 * self.controls.target_position[2]:7.1f} mm | "
            f"eef: {position_mm[0]:7.1f} {position_mm[1]:7.1f} {position_mm[2]:7.1f} mm | "
            f"J5/J6: {wrist_deg[0]:6.1f}/{wrist_deg[1]:6.1f} deg | "
            f"J5 plane: {plane_state}"
        )


def print_controls() -> None:
    print("""
Kiko-Bot manual pick-and-place controller

  Move the gripper
    Arrows   : forward / back / left / right
    Q / E    : up / down        (+Z / -Z)

  Wrist
    R / F    : rotate J5 + / -
    Z / X    : rotate J6 + / -
    C        : centre J5 and J6

  Jaws (hold; stop as soon as they grip)
    O        : open
    P        : close

  H          : return the robot to the ready pose

Xbox controller (when connected):
  Left stick : XY target movement
  LT / RT    : -Z / +Z target movement
  Right stick: J5 / J6 manual rotation
  LB / RB    : open / close the jaws
  A          : centre J5 and J6
""".strip())


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    controller = PickPlaceController(model, data)
    print_controls()

    last_time = time.perf_counter()
    next_status_time = last_time
    with mujoco.viewer.launch_passive(
        model,
        data,
        # Windows polls the held state directly; other OSes retain key-repeat.
        key_callback=None if controller.get_async_key_state else controller.on_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        while viewer.is_running():
            now = time.perf_counter()
            dt = min(now - last_time, 0.05)
            last_time = now

            controller.begin_input_frame()
            controller.update_keyboard(dt)
            controller.update_gamepad(dt)
            wrist_only_motion = (
                controller.manual_wrist_moved
                and not controller.cartesian_target_moved
            )
            if not wrist_only_motion:
                controller.solve_ik_if_due()
            controller.apply_controls()
            mujoco.mj_step(model, data)
            if wrist_only_motion:
                # Capture the post-step pose so the next frame has no
                # Cartesian error for J1--J4 to compensate.
                controller.hold_cartesian_target_at_current_pose()
            viewer.sync()

            if now >= next_status_time:
                print(controller.status())
                next_status_time = now + 1.0
            time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
