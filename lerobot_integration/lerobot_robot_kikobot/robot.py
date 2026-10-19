"""The Kiko-Bot MuJoCo scene behind LeRobot's `Robot` interface.

The sim runs in this process, so `send_action` steps physics directly. Actions
are end-effector deltas, handed to the existing `PickPlaceController`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

import kikobot
from kikobot.sim.controller import (
    WORKSPACE_MAX_Z_M,
    WORKSPACE_MIN_Z_M,
    WORKSPACE_XY_RADIUS_M,
)
from kikobot.sim.training_env import TrainingController
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_not_connected

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")

# Physics steps per control tick. 1/30 s is not a whole number of the model's
# default 0.002 s steps, so `connect` retimes it to 1/(fps*SUBSTEPS).
SUBSTEPS = 17


@RobotConfig.register_subclass("kikobot_sim")
@dataclass
class KikobotConfig(RobotConfig):
    scene: str = kikobot.PICK_PLACE_SCENE_PATH
    fps: int = 30
    # The scene's offscreen framebuffer caps out at 1280x720
    # (models/pick_place_scene.xml's <visual><global offwidth/offheight>);
    # raising these further needs that cap raised too.
    width: int = 640
    height: int = 480
    # Not `cameras`: RobotConfig reserves that name for CameraConfig objects.
    camera_names: tuple[str, ...] = ("record",)
    seed: int | None = None
    # A native passive viewer window, decoupled from the recorded camera and
    # Rerun's slower image-logging path. Windows: plain `python`. macOS needs
    # `mjpython` -- `launch_passive` must own the real main thread there.
    viewer: bool = False


class Kikobot(Robot):
    config_class = KikobotConfig
    name = "kikobot_sim"

    def __init__(self, config: KikobotConfig):
        super().__init__(config)
        self.config = config
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.controller: TrainingController | None = None
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

    # ---------------------------------------------------------------- features

    @property
    def _joints_ft(self) -> dict[str, type]:
        # EE pose is in the state because the actions are deltas.
        return {
            **{f"{j}.pos": float for j in ARM_JOINTS},
            "gripper.pos": float,
            "ee_x.pos": float,
            "ee_y.pos": float,
            "ee_z.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.height, self.config.width, 3)
            for cam in self.config.camera_names
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joints_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Absolute commanded targets. The `.pos` suffix is required:
        # lerobot-rollout drops features without a `.pos`/`.vel` suffix, which
        # would empty the action space.
        return {
            "ee_x.pos": float,
            "ee_y.pos": float,
            "ee_z.pos": float,
            "j5.pos": float,
            "j6.pos": float,
            "gripper.pos": float,
        }

    # ------------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self.controller is not None

    def connect(self, calibrate: bool = True) -> None:
        self.model = mujoco.MjModel.from_xml_path(self.config.scene)
        self.model.opt.timestep = 1.0 / (self.config.fps * SUBSTEPS)
        self.data = mujoco.MjData(self.model)
        self.controller = TrainingController(
            self.model, self.data, np.random.default_rng(self.config.seed)
        )
        self._renderer = mujoco.Renderer(
            self.model, height=self.config.height, width=self.config.width
        )
        self._qpos_adr = {
            j: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in (*ARM_JOINTS, "finger_l")
        }
        if self.config.viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    @property
    def is_calibrated(self) -> bool:
        return True  # Nothing to calibrate in sim.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        self.controller = None
        self.model = self.data = None

    # ------------------------------------------------------------------- i/o

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        c, d = self.controller, self.data
        obs: dict[str, Any] = {
            f"{j}.pos": float(d.qpos[self._qpos_adr[j]]) for j in ARM_JOINTS
        }
        obs["gripper.pos"] = float(d.qpos[self._qpos_adr["finger_l"]])
        ee = d.xpos[c.eef_body_id]
        obs["ee_x.pos"], obs["ee_y.pos"], obs["ee_z.pos"] = (float(v) for v in ee)
        for cam in self.config.camera_names:
            self._renderer.update_scene(d, camera=cam)
            obs[cam] = self._renderer.render()
        return obs

    def commanded_action(self) -> RobotAction:
        """The pose the controller is currently commanding, in action form."""
        c = self.controller
        x, y, z = c.controls.target_position
        return {
            "ee_x.pos": float(x),
            "ee_y.pos": float(y),
            "ee_z.pos": float(z),
            "j5.pos": float(c.controls.joint_targets[4]),
            "j6.pos": float(c.controls.joint_targets[5]),
            "gripper.pos": float(c.controls.gripper_ctrl),
        }

    def clamp_action(self, action: RobotAction) -> RobotAction:
        """Limit an absolute target to the work volume, wrist range and jaw range."""
        c = self.controller
        xyz = np.array([action["ee_x.pos"], action["ee_y.pos"], action["ee_z.pos"]], dtype=float)
        xyz[2] = np.clip(xyz[2], WORKSPACE_MIN_Z_M, WORKSPACE_MAX_Z_M)
        norm = np.linalg.norm(xyz[:2])
        if norm > WORKSPACE_XY_RADIUS_M:
            xyz[:2] *= WORKSPACE_XY_RADIUS_M / norm
        jaw_low = min(c.gripper_open_ctrl, c.gripper_closed_ctrl)
        jaw_high = max(c.gripper_open_ctrl, c.gripper_closed_ctrl)
        return {
            "ee_x.pos": float(xyz[0]),
            "ee_y.pos": float(xyz[1]),
            "ee_z.pos": float(xyz[2]),
            "j5.pos": float(np.clip(action["j5.pos"], c.joint_low[4], c.joint_high[4])),
            "j6.pos": float(np.clip(action["j6.pos"], c.joint_low[5], c.joint_high[5])),
            "gripper.pos": float(np.clip(action["gripper.pos"], jaw_low, jaw_high)),
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        c = self.controller
        target = self.clamp_action(action)
        current = self.commanded_action()
        c.begin_input_frame()
        c.translate_target(
            np.array([target[k] - current[k] for k in ("ee_x.pos", "ee_y.pos", "ee_z.pos")])
        )
        c.move_wrist(4, target["j5.pos"] - current["j5.pos"])
        c.move_wrist(5, target["j6.pos"] - current["j6.pos"])
        c.controls.gripper_ctrl = target["gripper.pos"]

        # A pure wrist move must not make J1--J4 chase the Cartesian target.
        wrist_only = c.manual_wrist_moved and not c.cartesian_target_moved
        if not wrist_only:
            c.solve_ik_if_due()
        c.apply_controls()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        if wrist_only:
            c.hold_cartesian_target_at_current_pose()
        if self._viewer is not None:
            self._viewer.sync()
        return action

    # ---------------------------------------------------------------- episode

    def reset_scene(self) -> None:
        """Re-centre the arm and respawn cube and pad.

        LeRobot has no episode-boundary hook; the record script calls this.
        """
        self.controller.reset_to_ready_pose()

    def task_succeeded(self) -> bool:
        """Cube resting on the place pad."""
        cube = self.controller.objects.cube_position()
        pad = self.controller.objects.pad_position()
        from kikobot.sim.training_env import (
            CUBE_HALF_EXTENT_M,
            PAD_HALF_HEIGHT_M,
            PAD_RADIUS_M,
        )

        in_xy = np.linalg.norm(cube[:2] - pad[:2]) <= PAD_RADIUS_M
        resting_z = 2 * PAD_HALF_HEIGHT_M + CUBE_HALF_EXTENT_M
        on_top = abs(cube[2] - resting_z) <= 0.01
        return bool(in_xy and on_top)
