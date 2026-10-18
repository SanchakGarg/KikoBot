"""The Kiko-Bot MuJoCo scene behind LeRobot's `Robot` interface.

The sim runs in this process, so `send_action` steps physics directly. Actions
are end-effector deltas, handed to the existing `PickPlaceController`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import mujoco
import numpy as np

import kikobot
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
    width: int = 320
    height: int = 240
    # Not `cameras`: RobotConfig reserves that name for CameraConfig objects.
    camera_names: tuple[str, ...] = ("record",)
    seed: int | None = None


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
        # `.vel` is required: lerobot-rollout drops features without a
        # `.pos`/`.vel` suffix, which would empty the action space.
        return {
            "ee_x.vel": float,
            "ee_y.vel": float,
            "ee_z.vel": float,
            "j5.vel": float,
            "j6.vel": float,
            "gripper.vel": float,
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

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        c = self.controller
        c.begin_input_frame()
        c.translate_target(
            np.array(
                [
                    float(action.get("ee_x.vel", 0.0)),
                    float(action.get("ee_y.vel", 0.0)),
                    float(action.get("ee_z.vel", 0.0)),
                ]
            )
        )
        c.move_wrist(4, float(action.get("j5.vel", 0.0)))
        c.move_wrist(5, float(action.get("j6.vel", 0.0)))
        c.move_gripper(float(action.get("gripper.vel", 0.0)))

        # A pure wrist move must not make J1--J4 chase the Cartesian target.
        wrist_only = c.manual_wrist_moved and not c.cartesian_target_moved
        if not wrist_only:
            c.solve_ik_if_due()
        c.apply_controls()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        if wrist_only:
            c.hold_cartesian_target_at_current_pose()
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
