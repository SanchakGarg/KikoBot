"""Kiko-Bot: a 6-DOF arm, its MuJoCo model, and the controllers that drive it.

Modules import `MODELS_DIR` rather than walking up from `__file__`, so they
run from any working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# str, not Path: mujoco.MjModel.from_xml_path wants a string.
SCENE_PATH = str(MODELS_DIR / "scene.xml")
PICK_PLACE_SCENE_PATH = str(MODELS_DIR / "pick_place_scene.xml")

__all__ = [
    "PROJECT_ROOT",
    "MODELS_DIR",
    "SCENE_PATH",
    "PICK_PLACE_SCENE_PATH",
]
