"""Launch the complete system; arm connection is disabled by default."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("cow_bringup")
    vision = os.path.join(share, "launch", "vision.launch.py")
    robot = os.path.join(share, "launch", "robot.launch.py")
    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_ip", default_value="192.168.1.30"),
            DeclareLaunchArgument("debug", default_value="true"),
            DeclareLaunchArgument("hand_eye_mode", default_value="eye_in_hand"),
            DeclareLaunchArgument("start_arm", default_value="false"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(vision),
                launch_arguments={
                    "camera_ip": LaunchConfiguration("camera_ip"),
                    "debug": LaunchConfiguration("debug"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot),
                launch_arguments={
                    "hand_eye_mode": LaunchConfiguration("hand_eye_mode"),
                    "start_arm": LaunchConfiguration("start_arm"),
                    "dry_run": LaunchConfiguration("dry_run"),
                }.items(),
            ),
        ]
    )
