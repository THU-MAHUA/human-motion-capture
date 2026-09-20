"""Optional ROS 2 publishers; imports happen only when ROS is requested."""

from __future__ import annotations

import numpy as np


class ROSPublisher:
    def __init__(self, frame_id: str = "realsense_color_optical"):
        try:
            import rclpy
            from sensor_msgs.msg import PointCloud2, PointField
            from std_msgs.msg import Header
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:
            raise RuntimeError("ROS 2 Humble Python messages are required for --ros") from exc
        self.rclpy, self.PointCloud2, self.PointField = rclpy, PointCloud2, PointField
        self.Header, self.Marker, self.MarkerArray = Header, Marker, MarkerArray
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("realsense_motion_capture")
        self.points_pub = self.node.create_publisher(PointCloud2, "~/joints", 10)
        self.markers_pub = self.node.create_publisher(MarkerArray, "~/skeleton", 10)
        self.frame_id = frame_id

    def publish(self, frame: dict, edges: tuple[tuple[int, int], ...]) -> None:
        msg = self.PointCloud2()
        msg.header = self.Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        xyz = np.asarray(frame["joints_xyz"], dtype=np.float32)
        confidence = np.asarray(frame["confidence"], dtype=np.float32)
        valid = np.asarray(frame["valid_mask"], dtype=bool)
        joint_count = len(xyz)
        msg.height, msg.width, msg.is_bigendian, msg.is_dense = 1, joint_count, False, False
        msg.fields = [self.PointField(name=name, offset=offset, datatype=self.PointField.FLOAT32, count=1)
                      for name, offset in (("x", 0), ("y", 4), ("z", 8), ("confidence", 12))]
        msg.point_step, msg.row_step = 16, joint_count * 16
        payload = np.zeros((joint_count, 4), dtype=np.float32)
        payload[:, :3], payload[:, 3] = xyz, confidence
        payload[~valid, :3] = np.nan
        msg.data = payload.tobytes()
        self.points_pub.publish(msg)

        markers = self.MarkerArray()
        for marker_id, (a, b) in enumerate(edges):
            marker = self.Marker()
            marker.header = msg.header
            marker.ns, marker.id, marker.type, marker.action = "skeleton", marker_id, self.Marker.LINE_LIST, self.Marker.ADD
            marker.scale.x, marker.color.r, marker.color.g, marker.color.a = 0.02, 0.1, 0.9, 0.9
            for index in (a, b):
                point = type(marker.pose.position)()
                point.x, point.y, point.z = (float(v) for v in xyz[index])
                marker.points.append(point)
            markers.markers.append(marker)
        self.markers_pub.publish(markers)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()
