"""Launch read-only robot feedback for hand-eye calibration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("arm"),
        "config",
        "handeye_feedback.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="arm",
                executable="estun_feedback_node",
                name="estun_feedback_node",
                parameters=[config],
                output="screen",
            )
        ]
    )
