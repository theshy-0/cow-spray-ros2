"""Launch estun_driver (5-point sequence + 250Hz PBVS + CRI)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("arm"),
        "config",
        "estun_driver.yaml",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("dry_run", default_value="false"),
            Node(
                package="arm",
                executable="estun_driver",
                name="estun_driver",
                parameters=[
                    config,
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("dry_run"), value_type=bool
                        ),
                    },
                ],
                output="screen",
            )
        ]
    )
