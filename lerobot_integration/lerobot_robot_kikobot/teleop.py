"""Keyboard and gamepad teleoperation for the Kiko-Bot sim.

LeRobot's own `keyboard_ee`/`gamepad` teleops have no wrist axes, and the cube
spawns at a random yaw, so J6 must be steerable. Key map: `docs/controls.md`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from kikobot.sim.controller import (
    DEADZONE,
    GRIPPER_SPEED_MPS,
    KEYBOARD_TRANSLATION_SPEED_MPS,
    KEYBOARD_WRIST_SPEED_RADPS,
)
from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.decorators import check_if_not_connected

logger = logging.getLogger(__name__)

AXES = ("ee_x.vel", "ee_y.vel", "ee_z.vel", "j5.vel", "j6.vel", "gripper.vel")
ZERO: RobotAction = dict.fromkeys(AXES, 0.0)


@TeleoperatorConfig.register_subclass("kikobot_teleop")
@dataclass
class KikobotTeleopConfig(TeleoperatorConfig):
    backend: str = "keyboard"  # keyboard | gamepad | scripted
    fps: int = 30


class KikobotTeleop(Teleoperator):
    config_class = KikobotTeleopConfig
    name = "kikobot_teleop"

    def __init__(self, config: KikobotTeleopConfig):
        super().__init__(config)
        self.config = config
        self._backend: _Backend | None = None
        self._last_t: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(AXES, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._backend is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.config.backend == "keyboard":
            self._backend = _KeyboardBackend()
        elif self.config.backend == "gamepad":
            self._backend = _GamepadBackend()
        elif self.config.backend == "scripted":
            self._backend = _ScriptedBackend()
        else:
            raise ValueError(
                f"Unknown backend {self.config.backend!r}; "
                "expected 'keyboard', 'gamepad' or 'scripted'."
            )
        self._backend.start()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Real elapsed time keeps held-key speed in m/s; clamped against stalls.
        now = time.perf_counter()
        dt = 1.0 / self.config.fps if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now
        return self._backend.poll(dt)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._backend is not None:
            self._backend.stop()
        self._backend = None


class _Backend:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def poll(self, dt: float) -> RobotAction: ...


class _KeyboardBackend(_Backend):
    """Held-key polling via pynput, replacing the Windows-only GetAsyncKeyState path."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._listener = None

    def start(self) -> None:
        from lerobot.utils.keyboard_input import pynput_can_capture

        if not pynput_can_capture():
            raise RuntimeError(
                "pynput cannot capture keys here. Install it (pip install pynput). "
                "Linux needs a real X11 session; Windows and macOS need no display server."
            )
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=lambda k: self._pressed.add(self._name(k)),
            on_release=lambda k: self._pressed.discard(self._name(k)),
        )
        self._listener.start()
        self._listener.wait()

        # On macOS the Accessibility grant is only observable after start.
        from lerobot.utils.keyboard_input import pynput_listener_is_trusted

        if not pynput_listener_is_trusted(self._listener):
            self.stop()
            raise RuntimeError(
                "pynput started but is not trusted: grant this terminal Accessibility "
                "permission in System Settings > Privacy & Security > Accessibility. "
                "Without it the arm never moves."
            )
        logger.info("Keyboard teleop ready: arrows=XY  Q/E=Z  R/F=J5  Z/X=J6  O/P=jaws")

    @staticmethod
    def _name(key) -> str:
        return getattr(key, "name", None) or getattr(key, "char", None) or str(key)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        self._listener = None

    def _down(self, *names: str) -> float:
        return 1.0 if any(n in self._pressed for n in names) else 0.0

    def poll(self, dt: float) -> RobotAction:
        motion = np.array(
            [
                self._down("up") - self._down("down"),
                self._down("left") - self._down("right"),
                self._down("q") - self._down("e"),
            ]
        )
        # Diagonals must not outpace the axes.
        norm = np.linalg.norm(motion)
        if norm > 1.0:
            motion /= norm
        motion *= KEYBOARD_TRANSLATION_SPEED_MPS * dt

        return {
            "ee_x.vel": float(motion[0]),
            "ee_y.vel": float(motion[1]),
            "ee_z.vel": float(motion[2]),
            "j5.vel": float((self._down("r") - self._down("f")) * KEYBOARD_WRIST_SPEED_RADPS * dt),
            "j6.vel": float((self._down("z") - self._down("x")) * KEYBOARD_WRIST_SPEED_RADPS * dt),
            "gripper.vel": float((self._down("o") - self._down("p")) * GRIPPER_SPEED_MPS * dt),
        }


class _GamepadBackend(_Backend):
    """GLFW gamepad; works with no window once `glfw.init()` has run."""

    def __init__(self) -> None:
        self._glfw = None

    def start(self) -> None:
        import glfw

        if not glfw.init():
            raise RuntimeError("glfw.init() failed; cannot read the gamepad.")
        self._glfw = glfw
        if not glfw.joystick_is_gamepad(glfw.JOYSTICK_1):
            logger.warning("No gamepad detected on JOYSTICK_1; actions will be zero until one is plugged in.")

    def stop(self) -> None:
        if self._glfw is not None:
            self._glfw.terminate()
        self._glfw = None

    @staticmethod
    def _dead(v: float) -> float:
        return v if abs(v) >= DEADZONE else 0.0

    def poll(self, dt: float) -> RobotAction:
        glfw = self._glfw
        if not glfw.joystick_is_gamepad(glfw.JOYSTICK_1):
            return dict(ZERO)
        s = glfw.get_gamepad_state(glfw.JOYSTICK_1)
        if s is None:
            return dict(ZERO)

        lx = self._dead(s.axes[glfw.GAMEPAD_AXIS_LEFT_X])
        ly = self._dead(s.axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        ry = self._dead(s.axes[glfw.GAMEPAD_AXIS_RIGHT_Y])
        # GLFW triggers rest at -1, full press +1.
        lt = (s.axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) * 0.5
        rt = (s.axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) * 0.5
        wrist = float(s.buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] == glfw.PRESS) - float(
            s.buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] == glfw.PRESS
        )

        return {
            "ee_x.vel": float(ly * KEYBOARD_TRANSLATION_SPEED_MPS * dt),
            "ee_y.vel": float(lx * KEYBOARD_TRANSLATION_SPEED_MPS * dt),
            "ee_z.vel": float(-ry * KEYBOARD_TRANSLATION_SPEED_MPS * dt),
            "j5.vel": 0.0,  # Right stick left/right is unmapped; no J5 control.
            "j6.vel": float(wrist * KEYBOARD_WRIST_SPEED_RADPS * dt),
            "gripper.vel": float((lt - rt) * GRIPPER_SPEED_MPS * dt),
        }


class _ScriptedBackend(_Backend):
    """A canned sweep, so the pipeline can be tested with nobody at the keyboard.

    Not a demonstrator: it does not reliably complete the task.
    """

    def __init__(self) -> None:
        self._t = 0.0

    def poll(self, dt: float) -> RobotAction:
        self._t += dt
        phase = self._t % 8.0
        reach = phase < 3.0          # out and down toward the table
        close = 3.0 <= phase < 4.0   # shut the jaws
        lift = 4.0 <= phase < 6.0    # pick it up
        return {
            "ee_x.vel": 0.04 * dt if reach else (-0.03 * dt if lift else 0.0),
            "ee_y.vel": 0.0,
            "ee_z.vel": -0.05 * dt if reach else (0.06 * dt if lift else 0.0),
            "j5.vel": 0.0,
            "j6.vel": 0.3 * dt if reach else 0.0,
            "gripper.vel": -GRIPPER_SPEED_MPS * dt if close else 0.0,
        }
