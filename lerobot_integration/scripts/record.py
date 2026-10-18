#!/usr/bin/env python
"""Record Kiko-Bot pick-and-place demonstrations into a LeRobotDataset.

`lerobot-record` has no episode-boundary hook, so this wraps LeRobot's own
`record_loop` and adds the `reset_scene()` call between episodes.

Keys: right/n save, left/r re-record, esc/q stop.
"""

from __future__ import annotations

import argparse
import logging

from lerobot.datasets import LeRobotDataset
from lerobot.processor.factory import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_kikobot import Kikobot, KikobotConfig, KikobotTeleop, KikobotTeleopConfig

DEFAULT_TASK = "Pick up the red cube and place it on the green pad"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episode-time", type=float, default=40.0, help="seconds per episode")
    p.add_argument("--reset-time", type=float, default=5.0, help="unrecorded seconds between episodes")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--backend", choices=("keyboard", "gamepad", "scripted"), default="keyboard")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--repo-id", default="local/kikobot_pickplace")
    p.add_argument("--root", default="datasets/kikobot_pickplace")
    p.add_argument("--no-display", action="store_true", help="skip the rerun viewer")
    p.add_argument(
        "--viewer",
        action="store_true",
        help="open a native MuJoCo window for teleop preview (smoother than rerun's camera feed)",
    )
    p.add_argument("--no-sound", action="store_true", help="skip spoken prompts")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=640, help="record camera width (max 1280)")
    p.add_argument("--height", type=int, default=480, help="record camera height (max 720)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot = Kikobot(
        KikobotConfig(
            id="kikobot_sim",
            fps=args.fps,
            seed=args.seed,
            viewer=args.viewer,
            width=args.width,
            height=args.height,
        )
    )
    teleop = KikobotTeleop(KikobotTeleopConfig(id="kikobot_teleop", backend=args.backend, fps=args.fps))

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=combine_feature_dicts(
            hw_to_dataset_features(robot.action_features, ACTION, True),
            hw_to_dataset_features(robot.observation_features, OBS_STR, True),
        ),
        root=args.root,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    teleop.connect()  # Before the robot, as lerobot-record does.
    robot.connect()
    listener, events = init_keyboard_listener()
    if not args.no_display:
        init_rerun(session_name="kikobot_record")

    loop_kwargs = dict(
        robot=robot,
        events=events,
        fps=args.fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        single_task=args.task,
        display_data=not args.no_display,
    )

    try:
        recorded = 0
        while recorded < args.episodes and not events["stop_recording"]:
            robot.reset_scene()
            log_say(f"Recording episode {recorded + 1} of {args.episodes}", not args.no_sound)
            record_loop(dataset=dataset, control_time_s=args.episode_time, **loop_kwargs)

            if events["rerecord_episode"]:
                log_say("Re-recording episode", not args.no_sound)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded += 1
            print(f"  saved episode {recorded}  (task {'SUCCEEDED' if robot.task_succeeded() else 'not completed'})")

            # Unrecorded settle time, so the next episode starts from rest.
            if recorded < args.episodes and not events["stop_recording"]:
                record_loop(control_time_s=args.reset_time, **loop_kwargs)
    finally:
        log_say("Stop recording", not args.no_sound)
        robot.disconnect()
        teleop.disconnect()
        if listener is not None:
            listener.stop()
        dataset.finalize()
        print(f"\nDataset written to {args.root} ({dataset.num_episodes} episodes)")


if __name__ == "__main__":
    main()
