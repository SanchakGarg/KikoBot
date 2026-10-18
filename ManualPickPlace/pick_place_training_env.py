"""Pick-and-place training scene driven by the existing Cartesian IK.

Loads `Kikobot-mujoco/pick_place_scene.xml` (arm on a table, plus a recording
camera), spawns the red cube and the green place pad at random reachable
positions, and hands control to the same teleoperation used by
`pick_place_controller.py`.

Spawn positions are restricted to a cone around the direction the arm is
currently pointing. That is not cosmetic: the Cartesian solver is a local
differential IK with a J5-plane constraint, so it cannot swing the base
across a large heading change. Measured against this model, targets converge
exactly within roughly +/-45 deg of the current heading and at a table-level
object radius of 0.23--0.37 m; outside that band the solve stalls at the
plane boundary. Every spawn in that band has been checked to be both
reachable and graspable.
"""

from __future__ import annotations

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

import pick_place_controller as ppc

try:
    import imageio.v2 as imageio  # Optional: only needed to record video.
except ImportError:
    imageio = None


SCENE_PATH = os.path.join(
    ppc.PROJECT_ROOT, "Kikobot-mujoco", "pick_place_scene.xml"
)

# Reachable band for the OBJECT (not the IK target) at table height, measured
# by driving this model's own IK onto objects across a radius and heading
# sweep. The grasp centre sits outside and below the J5 frame the solver
# aims, so the object band is shifted outwards from the J5 band.
SPAWN_RADIUS_MIN_M = 0.23
SPAWN_RADIUS_MAX_M = 0.37
SPAWN_CONE_HALF_ANGLE_DEG = 45.0

# Starting guess for how far the grasp centre sits below the J5 IK target.
# Only a seed: ik_target_for() refines it against the real finger positions.
GRASP_HEIGHT_OFFSET_M = 0.1163

# The fingertips reach ~48 mm below the grasp centre that ik_target_for()
# aims. A top-down grasp therefore puts the grasp centre ABOVE the object;
# aiming at the object itself drives the fingers through the tabletop and
# stalls the shoulder against its torque limit.
FINGER_REACH_BELOW_GRASP_CENTRE_M = 0.048
TABLE_CLEARANCE_M = 0.008

# Must match the geometry declared in pick_place_scene.xml.
CUBE_HALF_EXTENT_M = 0.016
PAD_HALF_HEIGHT_M = 0.008
PAD_RADIUS_M = 0.055
# Keep the pad clear of the cube so a grasp is never fouled by the pad rim.
MIN_CUBE_PAD_SEPARATION_M = PAD_RADIUS_M + CUBE_HALF_EXTENT_M + 0.05

RECORDING_FPS = 30
RECORDING_SIZE = (1280, 720)
RECORDING_PATH = os.path.join(ppc.PROJECT_ROOT, "pick_place_recording.mp4")


class SceneObjects:
    """Teleports the cube and the place pad without disturbing the arm."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        cube_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free"
        )
        if cube_joint < 0:
            raise RuntimeError("The scene must contain the 'target_cube_free' joint.")
        self.cube_qpos_adr = int(model.jnt_qposadr[cube_joint])
        self.cube_qvel_adr = int(model.jnt_dofadr[cube_joint])

        pad_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_pad")
        self.pad_mocap_id = int(model.body_mocapid[pad_body]) if pad_body >= 0 else -1
        if self.pad_mocap_id < 0:
            raise RuntimeError("The scene must contain the 'place_pad' mocap body.")

    def place_cube(self, xy: np.ndarray, yaw: float) -> None:
        """Drop the cube flat on the table at `xy`, rotated about world Z."""
        adr = self.cube_qpos_adr
        self.data.qpos[adr : adr + 3] = (xy[0], xy[1], CUBE_HALF_EXTENT_M)
        self.data.qpos[adr + 3 : adr + 7] = (
            np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
        )
        self.data.qvel[self.cube_qvel_adr : self.cube_qvel_adr + 6] = 0.0

    def place_pad(self, xy: np.ndarray) -> None:
        self.data.mocap_pos[self.pad_mocap_id] = (xy[0], xy[1], PAD_HALF_HEIGHT_M)

    def cube_position(self) -> np.ndarray:
        adr = self.cube_qpos_adr
        return self.data.qpos[adr : adr + 3].copy()

    def pad_position(self) -> np.ndarray:
        return self.data.mocap_pos[self.pad_mocap_id].copy()


class TrainingController(ppc.PickPlaceController):
    """The teleoperation controller plus randomised object placement."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
    ) -> None:
        # Both are needed by reset_to_ready_pose, which the base __init__ calls.
        self.objects = SceneObjects(model, data)
        self.rng = rng
        self.last_respawn_key_down = False
        super().__init__(model, data)

    def reset_to_ready_pose(self) -> None:
        """Reset the arm, then re-place the objects.

        The model's `home` keyframe is shorter than the scene's qpos, so
        MuJoCo pads it with each body's declared pose. Resetting to it
        therefore teleports the cube back to its XML position, and the
        episode's spawn has to be re-applied afterwards.
        """
        super().reset_to_ready_pose()
        self.respawn_objects()

    def current_heading_rad(self) -> float:
        """Horizontal direction the arm is currently reaching in."""
        x, y = self.controls.target_position[:2]
        return float(np.arctan2(y, x))

    def ik_target_for(
        self, object_position: np.ndarray, iterations: int = 6
    ) -> np.ndarray:
        """The J5 target that puts the grasp centre on `object_position`.

        The gripper hangs off the J5 frame by an amount that changes with the
        arm's configuration, so there is no fixed offset to apply. Instead,
        solve on the kinematic copy, measure where the fingers actually land,
        and shift the target by that error until it settles.

        Returns a target for `controls.target_position`; the live commanded
        pose is left untouched.
        """
        object_position = np.asarray(object_position, dtype=float)
        saved_target = self.controls.target_position.copy()
        saved_joints = self.controls.joint_targets.copy()
        target = object_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET_M])
        try:
            for _ in range(iterations):
                self.controls.target_position = target.copy()
                for _ in range(ppc.IK_WARM_START_ITERATIONS):
                    self.solve_ik_step()
                self._sync_ik_state()
                error = object_position - self.grasp_centre(self.ik_data)
                target = target + error
                if np.linalg.norm(error) < 1e-4:
                    break
        finally:
            self.controls.target_position = saved_target
            self.controls.joint_targets = saved_joints
        return target

    def grasp_target_for(self, object_position: np.ndarray) -> np.ndarray:
        """IK target for a top-down grasp of an object resting on the table.

        Height is set by how far the fingers hang below the grasp centre, not
        by the object, so the fingertips straddle it and clear the table.
        """
        aim = np.asarray(object_position, dtype=float).copy()
        aim[2] = FINGER_REACH_BELOW_GRASP_CENTRE_M + TABLE_CLEARANCE_M
        return self.ik_target_for(aim)

    def sample_reachable_point(self, heading_rad: float) -> np.ndarray:
        cone = np.deg2rad(SPAWN_CONE_HALF_ANGLE_DEG)
        radius = self.rng.uniform(SPAWN_RADIUS_MIN_M, SPAWN_RADIUS_MAX_M)
        angle = heading_rad + self.rng.uniform(-cone, cone)
        return np.array([radius * np.cos(angle), radius * np.sin(angle)])

    def respawn_objects(self) -> None:
        """Randomise cube and pad within reach of the current heading."""
        heading = self.current_heading_rad()
        cube_xy = self.sample_reachable_point(heading)

        pad_xy = self.sample_reachable_point(heading)
        for _ in range(100):
            if np.linalg.norm(pad_xy - cube_xy) >= MIN_CUBE_PAD_SEPARATION_M:
                break
            pad_xy = self.sample_reachable_point(heading)

        self.objects.place_cube(cube_xy, yaw=self.rng.uniform(-np.pi, np.pi))
        self.objects.place_pad(pad_xy)
        mujoco.mj_forward(self.model, self.data)
        print(
            f"Spawned cube at {1000 * cube_xy[0]:6.1f} {1000 * cube_xy[1]:6.1f} mm, "
            f"pad at {1000 * pad_xy[0]:6.1f} {1000 * pad_xy[1]:6.1f} mm "
            f"(heading {np.rad2deg(heading):5.1f} deg)"
        )

    def update_keyboard(self, dt: float) -> None:
        super().update_keyboard(dt)
        if self.get_async_key_state is None:
            return
        down = self._key_is_down(ord("N"))
        if down and not self.last_respawn_key_down:
            self.respawn_objects()
        self.last_respawn_key_down = down

    def on_key(self, keycode: int) -> None:
        if 32 <= keycode <= 126 and chr(keycode).lower() == "n":
            self.respawn_objects()
            return
        super().on_key(keycode)


class SceneRecorder:
    """Writes frames from the scene's `record` camera to an MP4."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.renderer: mujoco.Renderer | None = None
        self.writer = None
        self.next_frame_time = 0.0

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def toggle(self, data: mujoco.MjData) -> None:
        if self.is_recording:
            self.stop()
            return
        if imageio is None:
            print("Recording needs imageio: pip install 'imageio[ffmpeg]'")
            return
        width, height = RECORDING_SIZE
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.writer = imageio.get_writer(RECORDING_PATH, fps=RECORDING_FPS)
        self.next_frame_time = data.time
        print(f"Recording to {RECORDING_PATH}")

    def capture_if_due(self, data: mujoco.MjData) -> None:
        if not self.is_recording or data.time < self.next_frame_time:
            return
        self.renderer.update_scene(data, camera="record")
        self.writer.append_data(self.renderer.render())
        self.next_frame_time = data.time + 1.0 / RECORDING_FPS

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"Recording saved to {RECORDING_PATH}")


def print_controls() -> None:
    ppc.print_controls()
    print("""
Training scene:
  N          : respawn the cube and pad in the arm's current direction
  V          : start / stop recording the 'record' camera to MP4
""".rstrip())


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    controller = TrainingController(model, data, np.random.default_rng())
    recorder = SceneRecorder(model)
    print_controls()

    last_time = time.perf_counter()
    next_status_time = last_time
    last_record_key_down = False

    with mujoco.viewer.launch_passive(
        model,
        data,
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
                controller.manual_wrist_moved and not controller.cartesian_target_moved
            )
            if not wrist_only_motion:
                controller.solve_ik_if_due()
            controller.apply_controls()
            mujoco.mj_step(model, data)
            if wrist_only_motion:
                controller.hold_cartesian_target_at_current_pose()

            if controller.get_async_key_state is not None:
                down = controller._key_is_down(ord("V"))
                if down and not last_record_key_down:
                    recorder.toggle(data)
                last_record_key_down = down

            recorder.capture_if_due(data)
            viewer.sync()

            if now >= next_status_time:
                jaws = "open" if controller.gripper_is_open() else "closed"
                print(f"{controller.status()} | jaws: {jaws}")
                next_status_time = now + 1.0

    recorder.stop()


if __name__ == "__main__":
    main()
