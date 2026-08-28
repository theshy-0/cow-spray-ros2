"""SICK ToF 相机驱动启动文件（仅相机节点）。

完整视觉系统和手眼标定入口位于 ``sick_bringup`` 包。

启动：
  ros2 launch camera camera.launch.py
  ros2 launch camera camera.launch.py ip:=192.168.1.30
  ros2 launch camera camera.launch.py publish_pointcloud:=true   # RViz 调试用
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('camera')
    camera_yaml = os.path.join(pkg_share, 'config', 'camera.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('ip', default_value='192.168.1.30'),
        DeclareLaunchArgument('publish_pointcloud', default_value='false'),
        Node(
            package='camera', executable='camera_node',
            name='camera_node', output='screen',
            parameters=[camera_yaml, {'ip': LaunchConfiguration('ip'),
                                    'publish_pointcloud': ParameterValue(
                                        LaunchConfiguration('publish_pointcloud'), value_type=bool)}],
            respawn=True, respawn_delay=3.0,
        ),
    ])
