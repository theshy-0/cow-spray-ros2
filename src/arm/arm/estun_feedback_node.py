"""Read-only Estun TCP feedback for hand-eye calibration."""

from __future__ import annotations

import time

import rclpy
from codroid import CodroidClient
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from .tool import convert_tcp_pose


class EstunFeedbackNode(Node):
    """Publish ``base_link -> tool0`` without enabling or commanding motion."""

    def __init__(self) -> None:
        super().__init__("estun_feedback_node")
        self.declare_parameter("robot_ip", "192.168.1.136")
        self.declare_parameter("local_ip", "192.168.1.10")
        self.declare_parameter("udp_port", 10086)
        self.declare_parameter("feedback_period_ms", 10)
        self.declare_parameter("feedback_timeout", 0.30)
        self.declare_parameter("connection_timeout", 5.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")

        value = lambda name: self.get_parameter(name).value
        self.robot_ip = str(value("robot_ip"))
        self.local_ip = str(value("local_ip"))
        self.udp_port = int(value("udp_port"))
        self.feedback_period_ms = int(value("feedback_period_ms"))
        self.feedback_timeout = float(value("feedback_timeout"))
        self.connection_timeout = float(value("connection_timeout"))
        self.base_frame = str(value("base_frame"))
        self.tool_frame = str(value("tool_frame"))
        if self.feedback_period_ms <= 0:
            raise ValueError("feedback_period_ms must be positive")
        if self.feedback_timeout <= 0.0 or self.connection_timeout <= 0.0:
            raise ValueError("timeouts must be positive")

        self.robot = None
        self.last_feedback_timestamp = None
        self.last_feedback_change = None
        try:
            self.robot = CodroidClient(
                host=self.robot_ip,
                local_ip=self.local_ip,
                udp_port=self.udp_port,
            )
            self.robot.__enter__()
            self.robot.StartListenUdp()
            self.robot.StartCriDataPush(
                ip=self.local_ip,
                port=self.udp_port,
                duration=self.feedback_period_ms,
            )
            data = self.robot.WaitForCriData(timeout=self.connection_timeout)
            self._record_feedback(data)
        except Exception:
            self._cleanup()
            raise

        self.broadcaster = TransformBroadcaster(self)
        self.create_timer(self.feedback_period_ms / 1000.0, self._publish_feedback)
        self.get_logger().info(
            "READ-ONLY hand-eye feedback active: "
            f"{self.base_frame} -> {self.tool_frame}"
        )

    def _record_feedback(self, data) -> None:
        timestamp = getattr(data, "timestamp", None)
        if timestamp is None:
            raise RuntimeError("CRI feedback has no timestamp")
        if timestamp != self.last_feedback_timestamp:
            self.last_feedback_timestamp = timestamp
            self.last_feedback_change = time.monotonic()

    def _publish_feedback(self) -> None:
        data = self.robot.CriData if self.robot is not None else None
        if data is None:
            return
        try:
            self._record_feedback(data)
            if time.monotonic() - self.last_feedback_change > self.feedback_timeout:
                self.get_logger().warning(
                    "CRI feedback stale; TF publication paused",
                    throttle_duration_sec=2.0,
                )
                return
            position, quaternion, _ = convert_tcp_pose(data.tcp_pose)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = self.tool_frame
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.broadcaster.sendTransform(transform)

    def _cleanup(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.StopCriDataPush(ip=self.local_ip, port=self.udp_port)
        except Exception as exc:
            self.get_logger().warning(f"Failed to stop CRI feedback: {exc}")
        try:
            self.robot.__exit__(None, None, None)
        except Exception as exc:
            self.get_logger().warning(f"Failed to disconnect robot feedback: {exc}")
        self.robot = None

    def destroy_node(self):
        self._cleanup()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = EstunFeedbackNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
