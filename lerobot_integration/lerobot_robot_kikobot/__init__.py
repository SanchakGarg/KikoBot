"""Kiko-Bot MuJoCo sim as a LeRobot device pair.

Importing this package registers both devices, so one distribution covers the
robot and the teleoperator.
"""

from .robot import Kikobot, KikobotConfig
from .teleop import KikobotTeleop, KikobotTeleopConfig

__all__ = ["Kikobot", "KikobotConfig", "KikobotTeleop", "KikobotTeleopConfig"]
