#!/usr/bin/env python3
"""连续图像采集节点，用于 YOLO 标注训练数据采集。

订阅相机图像话题，按固定间隔保存为图片文件，自动接续编号。

用法：
  ros2 run camera image_collect_node [--ros-args -p <param>:=<value> ...]

参数：
  image_topic   订阅的图像话题（默认 /camera_node/intensity）
  save_dir      保存目录（默认 dataset/images，相对当前目录）
  save_interval 保存间隔（秒），0 = 每帧都保存
  prefix        文件名前缀（默认 teat）
  image_format  图片格式 png / jpg（默认 png）
  convert_16bit_to_8bit  是否把 16 位图转成 8 位（默认 true，YOLO 用）
  intensity_vmax   0 = 自动按该帧最值归一化；>0 = 固定归一化上限（默认 0）
"""

import os
import re
import time

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image

from ..camera_node import _SENSOR_QOS


class ImageCollectNode(Node):
    def __init__(self) -> None:
        super().__init__('image_collect_node')

        self.declare_parameter('image_topic', '/camera_node/intensity')
        self.declare_parameter('save_dir', 'dataset/images')
        self.declare_parameter('save_interval', 0.2)
        self.declare_parameter('prefix', 'teat')
        self.declare_parameter('image_format', 'png')
        self.declare_parameter('convert_16bit_to_8bit', True)
        self.declare_parameter('intensity_vmax', 0)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.save_dir = str(self.get_parameter('save_dir').value)
        self.save_interval = float(self.get_parameter('save_interval').value)
        self.prefix = str(self.get_parameter('prefix').value)
        self.image_format = str(self.get_parameter('image_format').value)
        self.convert_16bit_to_8bit = bool(self.get_parameter('convert_16bit_to_8bit').value)
        self.intensity_vmax = float(self.get_parameter('intensity_vmax').value)

        self.bridge = CvBridge()
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_index = self.find_next_index()
        self.last_save_time = 0.0
        self.received_frames = 0
        self.saved_frames = 0
        self._last_shapes = {}

        self.subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, _SENSOR_QOS
        )

        self.get_logger().info(
            '\n'
            '========================================\n'
            '       Image Collector Started\n'
            '========================================\n'
            f'Image topic    : {self.image_topic}\n'
            f'Save directory : {os.path.abspath(self.save_dir)}\n'
            f'Save interval  : {self.save_interval:.3f} s\n'
            f'Filename       : {self.prefix}_XXXXXX.{self.image_format}\n'
            f'Start index    : {self.image_index}\n'
            '----------------------------------------\n'
            'Press Ctrl+C to stop.\n'
            '========================================'
        )

    def find_next_index(self) -> int:
        max_index = 0
        pattern = re.compile(
            rf'^{re.escape(self.prefix)}_(\d+)\.'
            rf'{re.escape(self.image_format)}$'
        )
        try:
            files = os.listdir(self.save_dir)
        except OSError:
            return 1
        for filename in files:
            match = pattern.match(filename)
            if match:
                try:
                    max_index = max(max_index, int(match.group(1)))
                except ValueError:
                    pass
        return max_index + 1

    def convert_image(self, msg: Image) -> np.ndarray | None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return None

        if cv_image.dtype == np.uint16 and self.convert_16bit_to_8bit:
            if self.intensity_vmax > 0:
                max_value = self.intensity_vmax
                min_value = 0
            else:
                min_value = float(np.min(cv_image))
                max_value = float(np.max(cv_image))
            if max_value > min_value:
                normalized = (cv_image.astype(np.float32) - min_value) * 255.0 / (max_value - min_value)
                cv_image = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                cv_image = np.zeros(cv_image.shape, dtype=np.uint8)

        if msg.encoding == 'rgb8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        return cv_image

    def image_callback(self, msg: Image) -> None:
        self.received_frames += 1

        current_time = time.monotonic()
        if current_time - self.last_save_time < self.save_interval:
            return

        cv_image = self.convert_image(msg)
        if cv_image is None:
            return

        filename = f'{self.prefix}_{self.image_index:06d}.{self.image_format}'
        save_path = os.path.join(self.save_dir, filename)

        success = cv2.imwrite(save_path, cv_image)
        if not success:
            self.get_logger().error(f'Failed to save: {save_path}')
            return

        self.last_save_time = current_time
        self.saved_frames += 1

        self._last_shapes = {
            'shape': cv_image.shape,
            'dtype': str(cv_image.dtype),
        }

        self.get_logger().info(
            f'[{self.saved_frames:05d}] Saved: {filename} '
            f'| shape={cv_image.shape} dtype={cv_image.dtype}'
        )

        self.image_index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            '\n'
            '========================================\n'
            '        Image Collector Stopped\n'
            '========================================\n'
            f'Received frames : {node.received_frames}\n'
            f'Saved images    : {node.saved_frames}\n'
            f'Save directory  : {os.path.abspath(node.save_dir)}\n'
            '========================================'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()