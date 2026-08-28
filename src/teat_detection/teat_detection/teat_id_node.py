"""Read the unchanged camera detector output and publish four teat task frames."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .transforms import matrix_to_quaternion
from .teat_id import TEAT_NAMES, TeatDetection, TeatTracker


class TeatIdNode(Node):
    def __init__(self) -> None:
        super().__init__("teat_id_node")
        self.declare_parameter("detections_topic", "/detector_node/detections")
        self.declare_parameter("tracked_detections_topic", "/udder/tracked_detections")
        self.declare_parameter("target_class_id", "0")
        self.declare_parameter("camera_frame_override", "")
        self.declare_parameter("udder_frame", "udder_frame")
        self.declare_parameter("publish_legacy_teat_tf", True)
        self.declare_parameter("expected_z_camera", [0.0, 0.0, 1.0])
        self.declare_parameter("front_is_smaller_v", True)
        self.declare_parameter("left_is_smaller_u", True)
        self.declare_parameter("min_detection_score", 0.35)
        self.declare_parameter("max_match_distance", 0.08)
        self.declare_parameter("max_fit_residual", 0.02)
        self.declare_parameter("two_point_timeout", 0.30)
        self.declare_parameter("max_prediction_translation", 0.05)
        self.declare_parameter("max_prediction_rotation_deg", 15.0)
        self.declare_parameter("reacquire_after_failures", 5)
        self.declare_parameter("reacquire_stable_frames", 5)
        self.tracker = TeatTracker(
            expected_z=self.get_parameter("expected_z_camera").value,
            front_is_smaller_v=bool(self.get_parameter("front_is_smaller_v").value),
            left_is_smaller_u=bool(self.get_parameter("left_is_smaller_u").value),
            min_score=float(self.get_parameter("min_detection_score").value),
            max_match_distance=float(self.get_parameter("max_match_distance").value),
            max_fit_residual=float(self.get_parameter("max_fit_residual").value),
            two_point_timeout=float(self.get_parameter("two_point_timeout").value),
            max_prediction_translation=float(
                self.get_parameter("max_prediction_translation").value
            ),
            max_prediction_rotation_deg=float(
                self.get_parameter("max_prediction_rotation_deg").value
            ),
            reacquire_after_failures=int(
                self.get_parameter("reacquire_after_failures").value
            ),
            reacquire_stable_frames=int(
                self.get_parameter("reacquire_stable_frames").value
            ),
        )
        self.udder_frame = str(self.get_parameter("udder_frame").value)
        self._tf_broadcaster = None
        self.status_pub = self.create_publisher(String, "/udder/status", 1)
        self.tracked_detections_pub = self.create_publisher(
            Detection2DArray,
            str(self.get_parameter("tracked_detections_topic").value),
            10,
        )
        self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detections_topic").value),
            self._on_detections,
            10,
        )

    @staticmethod
    def _stamp_seconds(msg: Detection2DArray) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _on_detections(self, msg: Detection2DArray) -> None:
        detections: list[TeatDetection] = []
        target_class = str(self.get_parameter("target_class_id").value)
        for source_index, detection in enumerate(msg.detections):
            for result in detection.results:
                if target_class and str(result.hypothesis.class_id) != target_class:
                    continue
                position = result.pose.pose.position
                if position.z <= 0.0:
                    continue
                detections.append(
                    TeatDetection(
                        (detection.bbox.center.position.x, detection.bbox.center.position.y),
                        (position.x, position.y, position.z),
                        float(result.hypothesis.score),
                        source_index,
                    )
                )
                break
        stamp = self._stamp_seconds(msg)
        try:
            observation = self.tracker.update_partial(detections, stamp)
        except ValueError as exc:
            # 极少（stamp 不递增等）；此时保留空输出
            status = {
                "valid": False,
                "initialized": self.tracker.initialized,
                "detections": len(detections),
                "reacquiring": self.tracker.reacquiring,
                "reason": str(exc),
                "lost": list(TEAT_NAMES),
            }
            tracked = self._build_empty_tracked(msg)
        else:
            if observation is None:
                # 模板尚未初始化成功：无依据输出四点
                status = {
                    "valid": False,
                    "initialized": False,
                    "detections": len(detections),
                    "reacquiring": self.tracker.reacquiring,
                    "stable_frames": self.tracker.stable_frames,
                    "consecutive_failures": self.tracker.consecutive_failures,
                    "reason": "awaiting template initialization",
                    "lost": list(TEAT_NAMES),
                }
                tracked = self._build_empty_tracked(msg)
            else:
                if bool(self.get_parameter("publish_legacy_teat_tf").value):
                    self._publish_frames(msg, observation)
                # 初始化后：无论本帧观测几个，固定输出四点（失败时是历史/几何传播）
                tracked = self._build_tracked_msg(msg, observation)
                status = {
                    "valid": not observation.stale,
                    "initialized": True,
                    "detections": len(detections),
                    "observed": list(observation.observed_names),
                    "predicted": list(observation.predicted_names),
                    "lost": [
                        name
                        for name in TEAT_NAMES
                        if name not in observation.observed_names
                        and name not in observation.predicted_names
                    ],
                    "fit_residual_m": float(observation.residual),
                    "assignments": {
                        f"teat_{name}": int(index)
                        for name, index in observation.source_indices.items()
                    },
                    "predicted_points": {
                        f"teat_{name}": [float(value) for value in observation.teats[name]]
                        for name in observation.predicted_names
                        if name in observation.teats
                    },
                    "stale": observation.stale,
                    "stamp": stamp,
                    "camera_frame": str(
                        self.get_parameter("camera_frame_override").value
                    ).strip()
                    or msg.header.frame_id,
                    "publish_legacy_teat_tf": bool(
                        self.get_parameter("publish_legacy_teat_tf").value
                    ),
                }
        self.tracked_detections_pub.publish(tracked)
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))

    def _build_tracked_msg(
        self, msg: Detection2DArray, observation
    ) -> Detection2DArray:
        """构造固定四乳头的 tracked 消息（不修改原始 detector msg）。"""
        tracked = Detection2DArray()
        tracked.header = msg.header
        by_name: dict[str, Detection2D] = {}
        # 实测点：deepcopy 原始 detection，改 id，保留 bbox/score/pose
        for name, source_index in observation.source_indices.items():
            if 0 <= source_index < len(msg.detections):
                det = deepcopy(msg.detections[source_index])
                det.id = f"teat_{name}"
                by_name[f"teat_{name}"] = det
        # 补点：新建 Detection2D（bbox=0, score=0, pose=observation.teats[name]）
        for name in observation.predicted_names:
            if name not in observation.teats:
                continue
            pos = observation.teats[name]
            det = Detection2D()
            det.header = msg.header
            det.id = f"teat_{name}"
            det.bbox.center.position.x = 0.0
            det.bbox.center.position.y = 0.0
            det.bbox.size_x = 0.0
            det.bbox.size_y = 0.0
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = f"teat_{name}"
            hypothesis.hypothesis.score = 0.0
            hypothesis.pose.pose.position.x = float(pos[0])
            hypothesis.pose.pose.position.y = float(pos[1])
            hypothesis.pose.pose.position.z = float(pos[2])
            det.results.append(hypothesis)
            by_name[f"teat_{name}"] = det
        # 固定按 TEAT_NAMES 顺序输出
        for name in TEAT_NAMES:
            key = f"teat_{name}"
            if key in by_name:
                tracked.detections.append(by_name[key])
        return tracked

    def _build_empty_tracked(self, msg: Detection2DArray) -> Detection2DArray:
        tracked = Detection2DArray()
        tracked.header = msg.header
        return tracked

    def _publish_frames(self, msg, observation) -> None:
        """发布 udder_frame + 四个乳头 TF（供 tf_legacy 控制模式使用）。

        相机是 eye-in-hand：这里发布 camera->udder_frame 和 udder_frame->teat_*，
        控制器通过 base->camera->udder->teat 的 TF 链取目标（TF 有缓冲+插值，运动更平滑）。
        """
        if self._tf_broadcaster is None:
            from tf2_ros import TransformBroadcaster
            self._tf_broadcaster = TransformBroadcaster(self)
        parent_override = str(self.get_parameter("camera_frame_override").value).strip()
        camera_frame = parent_override or msg.header.frame_id
        if not camera_frame:
            return
        frames = []
        # camera -> udder_frame
        udder = TransformStamped()
        udder.header.stamp = msg.header.stamp
        udder.header.frame_id = camera_frame
        udder.child_frame_id = self.udder_frame
        udder.transform.translation.x = float(observation.origin[0])
        udder.transform.translation.y = float(observation.origin[1])
        udder.transform.translation.z = float(observation.origin[2])
        qx, qy, qz, qw = matrix_to_quaternion(observation.rotation)
        udder.transform.rotation.x = qx
        udder.transform.rotation.y = qy
        udder.transform.rotation.z = qz
        udder.transform.rotation.w = qw
        frames.append(udder)
        # udder_frame -> teat_xxx（用 observation.teats，含几何补点）
        for name, pos in observation.teats.items():
            local = observation.rotation.T @ (pos - observation.origin)
            teat = TransformStamped()
            teat.header.stamp = msg.header.stamp
            teat.header.frame_id = self.udder_frame
            teat.child_frame_id = f"teat_{name}"
            teat.transform.translation.x = float(local[0])
            teat.transform.translation.y = float(local[1])
            teat.transform.translation.z = float(local[2])
            teat.transform.rotation.w = 1.0
            frames.append(teat)
        self._tf_broadcaster.sendTransform(frames)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeatIdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
