import os

import mujoco
import mujoco.viewer


# ============================================================
# CONFIG
# ============================================================

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Kikobot-mujoco",
    "scene.xml"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)


# ============================================================
# HOME POSE
# ============================================================

mujoco.mj_resetData(model, data)

data.ctrl[:] = 0.0


# ============================================================
# INFO
# ============================================================

print("=" * 60)
print("Kiko-Bot MuJoCo simulation")
print(f"Scene : {SCENE_PATH}")
print("=" * 60)

for actuator_id in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id
    )

    low, high = model.actuator_ctrlrange[actuator_id]

    print(
        f"{name} | "
        f"{low:+7.1f}° .. {high:+7.1f}° | "
        f"home {data.ctrl[actuator_id]:+.1f}°"
    )

print("=" * 60)
print("Open the Control panel in the viewer to drive the joints.")
print("Sliders are in DEGREES.")
print("=" * 60)


# ============================================================
# VIEWER
# ============================================================

mujoco.viewer.launch(model, data)
