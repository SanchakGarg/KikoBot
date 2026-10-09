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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_PATH = os.path.join(PROJECT_ROOT, "Kikobot-mujoco", "scene.xml")

# The Windows key-state API provides true press-and-hold control even though
# MuJoCo's passive-viewer callback reports presses only.
KEYBOARD_TRANSLATION_STEP_M = 0.005
KEYBOARD_WRIST_STEP_RAD = np.deg2rad(3.0)
KEYBOARD_TRANSLATION_SPEED_MPS = 0.10
KEYBOARD_WRIST_SPEED_RADPS = np.deg2rad(60.0)

# Xbox controls are velocity commands, multiplied by elapsed real time.
GAMEPAD_TRANSLATION_SPEED_MPS = 0.12
GAMEPAD_WRIST_SPEED_RADPS = np.deg2rad(75.0)
DEADZONE = 0.12

# Differential IK tuning.
DAMPING = 0.035
MAX_IK_STEP_RAD = np.deg2rad(2.0)
IK_UPDATE_PERIOD_S = 0.02
PLANE_AXIS_Z_TOLERANCE = np.sin(np.deg2rad(1.0))

WORLD_Z = np.array([0.0, 0.0, 1.0])
# The all-zero model keyframe is nearly straight-up and singular for Cartesian
# motion. This bent pose has useful local motion and a horizontal J5 hinge
# axis, so the J5 rotation plane starts perpendicular to world XY.
READY_POSE_RAD = np.deg2rad([0.0, -45.0, 105.0, 60.0, 0.0, 0.0])

# Windows virtual-key codes.  The dictionary also makes the polling code
# independent of the GLFW window internals used by MuJoCo's passive viewer.
KEY_UP = 0x26
KEY_DOWN = 0x28
KEY_LEFT = 0x25
KEY_RIGHT = 0x27


@dataclass
class Controls:
    """Mutable commanded pose and Cartesian target."""

    target_position: np.ndarray
    joint_targets: np.ndarray


class PickPlaceController:
    """Maps input devices to a Cartesian target and solves the arm IK."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        # Keep an independent kinematic state for IK. The simulated arm can
        # sag slightly under gravity between actuator updates; planning from
        # that transient pose caused the plane gate to report false limits.
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
        # The Cartesian task point is the J5 frame, not the final `eof` mesh.
        # J5 is the wrist pivot: treating it as the end effector makes J1--J4
        # responsible only for positioning the arm, while J5/J6 stay manual.
        self.eef_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "j5"
        )
        if self.eef_body_id < 0:
            raise RuntimeError("The model must contain the J5 body.")

        self.joint_low = model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = model.jnt_range[self.joint_ids, 1].copy()
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

        # The supplied model's normal reset pose is the CAD lower-stop pose.
        # Start in a deliberately bent, nonsingular ready pose instead.
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
        self.controls = Controls(
            target_position=self.data.xpos[self.eef_body_id].copy(),
            joint_targets=targets,
        )
        self.next_ik_time = self.data.time
        self.apply_controls()

    def apply_controls(self) -> None:
        """Send the position-servo targets to actuators named J1 through J6."""
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            if name in {"J1", "J2", "J3", "J4", "J5", "J6"}:
                joint_index = int(name[1]) - 1
                self.data.ctrl[actuator_id] = self.controls.joint_targets[joint_index]

    def translate_target(self, delta: np.ndarray) -> None:
        """Move the Cartesian target, keeping it in a conservative work volume."""
        if not np.any(delta):
            return
        candidate = self.controls.target_position + delta
        # A 0.42 m XY radius and 0.06--0.55 m height keep targets close to the
        # physical 540 mm home-pose envelope and above the fixed base/floor.
        candidate[2] = np.clip(candidate[2], 0.06, 0.55)
        xy_norm = np.linalg.norm(candidate[:2])
        if xy_norm > 0.42:
            candidate[:2] *= 0.42 / xy_norm
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

        # Four controlled joints serve four task equations. Solve the equality
        # system directly when it is well-conditioned so damping does not
        # soften the J5-plane condition. Near singularities use DLS instead.
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

        # Integrate from the same commanded pose used by the kinematic copy.
        # Nonlinear kinematics leave a small residual after a differential
        # update. Backtrack until the candidate satisfies the plane constraint.
        # Keep the requested Cartesian target if no step is feasible so a
        # reverse command can retreat from the boundary.
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

        xy = np.array(
            [
                float(self._key_is_down(KEY_RIGHT)) - float(self._key_is_down(KEY_LEFT)),
                float(self._key_is_down(KEY_UP)) - float(self._key_is_down(KEY_DOWN)),
                0.0,
            ]
        )
        z = float(self._key_is_down(ord("Q"))) - float(self._key_is_down(ord("E")))
        motion = np.array([xy[0], xy[1], z])
        # Diagonal movement has the same maximum speed as axial movement.
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
        key_map = {
            265: np.array([0.0, step, 0.0]),   # GLFW_KEY_UP
            264: np.array([0.0, -step, 0.0]),  # GLFW_KEY_DOWN
            263: np.array([-step, 0.0, 0.0]),  # GLFW_KEY_LEFT
            262: np.array([step, 0.0, 0.0]),   # GLFW_KEY_RIGHT
        }
        if keycode in key_map:
            self.translate_target(key_map[keycode])
            return

        key = chr(keycode).lower() if 32 <= keycode <= 126 else ""
        if key == "q":
            self.translate_target(np.array([0.0, 0.0, step]))
        elif key == "e":
            self.translate_target(np.array([0.0, 0.0, -step]))
        elif key == "r":
            self.move_wrist(4, wrist_step)  # Manual J5 positive rotation.
        elif key == "f":
            self.move_wrist(4, -wrist_step)  # Manual J5 negative rotation.
        elif key == "z":
            self.move_wrist(5, wrist_step)   # Manual J6 positive rotation.
        elif key == "x":
            self.move_wrist(5, -wrist_step)  # Manual J6 negative rotation.
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
  Arrow keys : move Cartesian target in world XY
  Q / E      : move target +Z / -Z
  R / F      : manually rotate J5 + / -
  Z / X      : manually rotate J6 + / -
  C          : centre J5 and J6
  H          : return the robot to the ready pose

Xbox controller (when connected):
  Left stick : XY target movement
  LT / RT    : -Z / +Z target movement
  Right stick: J5 / J6 manual rotation
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
                # Capture the post-step pose, after the manual wrist has moved.
                # The next frame will therefore have no Cartesian error to
                # compensate with J1--J4.
                controller.hold_cartesian_target_at_current_pose()
            viewer.sync()

            if now >= next_status_time:
                print(controller.status())
                next_status_time = now + 1.0
            time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
