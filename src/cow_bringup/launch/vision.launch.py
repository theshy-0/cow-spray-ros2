"""Launch camera and the complete teat perception chain."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera_launch = os.path.join(
        get_package_share_directory("camera"), "launch", "camera.launch.py"
    )
    detection_launch = os.path.join(
        get_package_share_directory("teat_detection"),
        "launch",
        "detection.launch.py",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_ip", default_value="192.168.1.30"),
            DeclareLaunchArgument("debug", default_value="true"),
            DeclareLaunchArgument("start_leg_entry", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(camera_launch),
                launch_arguments={"ip": LaunchConfiguration("camera_ip")}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(detection_launch),
                launch_arguments={
                    "debug": LaunchConfiguration("debug"),
                    "start_leg_entry": LaunchConfiguration("start_leg_entry"),
                }.items(),
            ),
        ]
    )
