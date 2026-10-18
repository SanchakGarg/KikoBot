"""Slider viewer: loads the arm and hands it to MuJoCo's own control panel."""

import mujoco
import mujoco.viewer

from kikobot import SCENE_PATH

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# qpos0 is each joint's hard stop, so reset to the "home" keyframe.
HOME_KEY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)

print("=" * 60)
print("Kiko-Bot MuJoCo simulation")
print(f"Scene : {SCENE_PATH}")
print("=" * 60)
for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    low, high = model.actuator_ctrlrange[actuator_id]
    print(f"{name} | {low:+7.3f} .. {high:+7.3f} rad | home {data.ctrl[actuator_id]:+.3f} rad")
print("=" * 60)
print("Open the Control panel in the viewer to drive the joints.")
print("Sliders are in RADIANS.")
print("=" * 60)

mujoco.viewer.launch(model, data)
