"""Publish exactly one selected hand-eye transform."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _publisher(context):
    mode = LaunchConfiguration("mode").perform(context)
    config_path = LaunchConfiguration("config_file").perform(context)
    allow_unverified = LaunchConfiguration("allow_unverified").perform(context).lower()
    with open(config_path, encoding="utf-8") as stream:
        modes = yaml.safe_load(stream)["hand_eye"]
    if mode not in modes:
        raise ValueError(f"unknown hand-eye mode: {mode}")
    config = modes[mode]
    if not config.get("verified", False) and allow_unverified not in {"1", "true", "yes"}:
        raise ValueError(
            f"hand-eye mode {mode} is not verified; set allow_unverified:=true only for calibration"
        )
    translation = [str(value) for value in config["translation"]]
    quaternion = [str(value) for value in config["quaternion"]]
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="hand_eye_tf",
            arguments=[
                "--x", translation[0], "--y", translation[1], "--z", translation[2],
                "--qx", quaternion[0], "--qy", quaternion[1],
                "--qz", quaternion[2], "--qw", quaternion[3],
                "--frame-id", config["parent_frame"],
                "--child-frame-id", config["child_frame"],
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("robot_description"), "config", "hand_eye.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="eye_in_hand"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("allow_unverified", default_value="false"),
            OpaqueFunction(function=_publisher),
        ]
    )
