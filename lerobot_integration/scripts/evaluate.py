#!/usr/bin/env python
"""Run a trained policy on the Kiko-Bot sim and score it.

`lerobot-rollout` drives the same robot but cannot report whether the cube
reached the pad. Same inference path, plus a success count.
"""

from __future__ import annotations

import argparse
import logging

import torch
from lerobot.common.control_utils import predict_action
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features

from lerobot_robot_kikobot import Kikobot, KikobotConfig

DEFAULT_TASK = "Pick up the red cube and place it on the green pad"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="path to a pretrained_model directory")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-time", type=float, default=40.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--device", default=None, help="defaults to the policy's own device")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot = Kikobot(KikobotConfig(id="kikobot_eval", fps=args.fps, seed=args.seed))
    policy = ACTPolicy.from_pretrained(args.policy)
    # policy.config.device is a string; predict_action wants a torch.device.
    device = torch.device(args.device or str(policy.config.device))

    features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, True),
        hw_to_dataset_features(robot.observation_features, OBS_STR, True),
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    robot.connect()
    successes = 0
    try:
        for ep in range(args.episodes):
            robot.reset_scene()
            policy.reset()
            for _ in range(int(args.episode_time * args.fps)):
                obs = robot.get_observation()
                frame = build_dataset_frame(features, obs, prefix=OBS_STR)
                action = predict_action(
                    observation=frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=device.type == "cuda",
                    task=args.task,
                    robot_type=robot.name,
                )
                robot.send_action(make_robot_action(action, features))
                if robot.task_succeeded():
                    break
            ok = robot.task_succeeded()
            successes += ok
            print(f"episode {ep + 1:3d}/{args.episodes}: {'SUCCESS' if ok else 'failed '}")
    finally:
        robot.disconnect()

    rate = successes / args.episodes if args.episodes else 0.0
    print(f"\nsuccess rate: {successes}/{args.episodes} = {rate:.1%}")


if __name__ == "__main__":
    main()
