"""Launch YOLO/depth detection, semantic teat IDs, and optional debug image."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("teat_detection")
    config = os.path.join(share, "config", "detection.yaml")
    model = os.path.join(share, "models", "best.pt")
    return LaunchDescription(
        [
            DeclareLaunchArgument("debug", default_value="true"),
            Node(
                package="teat_detection",
                executable="detector_node",
                name="detector_node",
                parameters=[config, {"model_path": model}],
                output="screen",
            ),
            Node(
                package="teat_detection",
                executable="teat_id_node",
                name="teat_id_node",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="teat_detection",
                executable="debug_node",
                name="sick_yolo_debug",
                parameters=[config],
                output="screen",
                condition=IfCondition(LaunchConfiguration("debug")),
            ),
        ]
    )
