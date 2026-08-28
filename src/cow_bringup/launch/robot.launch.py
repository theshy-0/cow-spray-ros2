"""Launch hand-eye TF and optionally connect the physical ESTUN arm."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    tf_launch = os.path.join(
        get_package_share_directory("robot_description"), "launch", "tf.launch.py"
    )
    arm_launch = os.path.join(
        get_package_share_directory("arm"), "launch", "estun_driver.launch.py"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("hand_eye_mode", default_value="eye_in_hand"),
            DeclareLaunchArgument("start_arm", default_value="false"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tf_launch),
                launch_arguments={"mode": LaunchConfiguration("hand_eye_mode")}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(arm_launch),
                launch_arguments={"dry_run": LaunchConfiguration("dry_run")}.items(),
                condition=IfCondition(LaunchConfiguration("start_arm")),
            ),
        ]
    )
