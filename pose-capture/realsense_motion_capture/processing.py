"""Hardware-independent 3D skeleton processing utilities.

The functions in this module deliberately use NumPy only.  They define the
data contract shared by the live capture process, offline replay, and human
motion dataset loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

# COCO-WholeBody order: body (17), feet (6), face (68), hands (42).
WHOLEBODY_NAMES = COCO17_NAMES + (
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel",
) + tuple(f"face_{index:02d}" for index in range(68)) + tuple(
    f"left_hand_{index:02d}" for index in range(21)) + tuple(
    f"right_hand_{index:02d}" for index in range(21))

COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
)

WHOLEBODY_EDGES = COCO17_EDGES + (
    (15, 17), (17, 18), (18, 19), (16, 20), (20, 21), (21, 22),
) + tuple((i, i + 1) for i in range(23, 90) if (i - 23) % 4 != 3) + tuple(
    (i, i + 1) for i in range(91, 132) if (i - 91) % 4 != 3)


def _as_float_array(value: np.ndarray, shape: tuple[int, ...], name: str):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def depth_to_xyz(
    keypoints_xy: np.ndarray,
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_scale: float = 0.001,
    window: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deproject 2D keypoints using a depth image aligned to color.

    Args:
        keypoints_xy: ``(N, 2)`` pixel coordinates in the color image.
        depth: ``(H, W)`` uint16 or float depth image.
        intrinsics: ``(fx, fy, cx, cy)`` for the image containing ``depth``.
        depth_scale: multiplier converting depth units to metres.
        window: radius of the median sampling window around each keypoint.

    Returns:
        ``(xyz, valid)`` where xyz is ``(N, 3)`` in camera optical coordinates
        and valid marks joints with a usable depth sample.
    """
    points = np.asarray(keypoints_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"keypoints_xy must have shape (N, 2), got {points.shape}")
    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_array.shape}")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    fx, fy, cx, cy = (float(x) for x in intrinsics)
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")

    height, width = depth_array.shape
    xyz = np.full((len(points), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(points), dtype=bool)
    radius = max(0, int(window))
    for index, (u, v) in enumerate(points):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        px, py = int(round(float(u))), int(round(float(v)))
        if px < 0 or py < 0 or px >= width or py >= height:
            continue
        x0, x1 = max(0, px - radius), min(width, px + radius + 1)
        y0, y1 = max(0, py - radius), min(height, py + radius + 1)
        values = np.asarray(depth_array[y0:y1, x0:x1], dtype=np.float32)
        values = values[np.isfinite(values) & (values > 0)] * float(depth_scale)
        values = values[(values >= min_depth_m) & (values <= max_depth_m)]
        if values.size == 0:
            continue
        z = float(np.median(values))
        xyz[index] = ((float(u) - cx) * z / fx,
                      (float(v) - cy) * z / fy,
                      z)
        valid[index] = True
    return xyz, valid


def _safe_fill(points: np.ndarray, valid: np.ndarray, previous: Optional[np.ndarray]):
    result = np.asarray(points, dtype=np.float32).copy()
    if previous is not None:
        missing = ~valid
        result[missing] = previous[missing]
    return result


def _root_quaternion(joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Estimate a pelvis orientation and return an ``xyzw`` quaternion."""
    if not (valid[5] and valid[6] and valid[11] and valid[12]):
        return np.array([0., 0., 0., 1.], dtype=np.float32)
    right = joints[12] - joints[11]
    up = ((joints[5] + joints[6]) * 0.5
          - (joints[11] + joints[12]) * 0.5)
    right_norm = np.linalg.norm(right)
    up_norm = np.linalg.norm(up)
    if right_norm < 1e-5 or up_norm < 1e-5:
        return np.array([0., 0., 0., 1.], dtype=np.float32)
    right /= right_norm
    up /= up_norm
    forward = np.cross(right, up)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-5:
        return np.array([0., 0., 0., 1.], dtype=np.float32)
    forward /= forward_norm
    up = np.cross(forward, right)
    rotation = np.column_stack((right, up, forward))
    trace = float(np.trace(rotation))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw, qx = 0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s
        qy, qz = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            qw, qx = (rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s
            qy, qz = (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            qw, qx = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s
            qy, qz = 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            qw, qx = (rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s
            qy, qz = (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    return quaternion / norm if norm > 1e-5 else np.array([0., 0., 0., 1.], dtype=np.float32)


@dataclass
class SkeletonFrame:
    timestamp: float
    joints_xyz: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    tracking_state: int


class SkeletonProcessor:
    """Add root-relative coordinates and finite-difference derivatives."""

    def __init__(self, joint_count: int = 17, smoothing: float = 0.35,
                 root_index: tuple[int, int] = (11, 12)):
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = float(smoothing)
        if joint_count < 17:
            raise ValueError("joint_count must include the COCO body joints")
        self.joint_count = int(joint_count)
        self.root_index = tuple(root_index)
        self._previous_xyz: Optional[np.ndarray] = None
        self._previous_filtered: Optional[np.ndarray] = None
        self._previous_velocity: Optional[np.ndarray] = None
        self._previous_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._previous_xyz = None
        self._previous_filtered = None
        self._previous_velocity = None
        self._previous_timestamp = None

    def update(self, frame: SkeletonFrame) -> dict[str, np.ndarray | float | int]:
        xyz = _as_float_array(frame.joints_xyz, (self.joint_count, 3), "joints_xyz")
        confidence = _as_float_array(frame.confidence, (self.joint_count,), "confidence")
        valid = np.asarray(frame.valid_mask, dtype=bool)
        if valid.shape != (self.joint_count,):
            raise ValueError(f"valid_mask must have shape ({self.joint_count},), got {valid.shape}")
        filtered = _safe_fill(xyz, valid, self._previous_filtered)
        if self._previous_filtered is not None:
            present = valid[:, None] & np.isfinite(xyz)
            filtered[present] = (
                self.smoothing * xyz[present]
                + (1.0 - self.smoothing) * self._previous_filtered[present])
        filtered[~np.isfinite(filtered)] = 0.0

        dt = 0.0 if self._previous_timestamp is None else max(
            1e-4, float(frame.timestamp) - self._previous_timestamp)
        velocity = np.zeros_like(filtered)
        if self._previous_filtered is not None and dt > 0:
            velocity = (filtered - self._previous_filtered) / dt
        acceleration = np.zeros_like(filtered)
        if self._previous_velocity is not None and dt > 0:
            acceleration = (velocity - self._previous_velocity) / dt

        root_valid = bool(valid[list(self.root_index)].any())
        root = np.mean(filtered[list(self.root_index)], axis=0)
        if not root_valid:
            root[:] = 0.0
        relative = filtered - root

        self._previous_xyz = xyz.copy()
        self._previous_filtered = filtered.copy()
        self._previous_velocity = velocity.copy()
        self._previous_timestamp = float(frame.timestamp)
        return {
            "timestamp": float(frame.timestamp),
            "joints_xyz": filtered,
            "root_xyz": root.astype(np.float32),
            "root_rotation": _root_quaternion(filtered, valid),
            "joints_root_relative": relative.astype(np.float32),
            "joint_velocity": velocity.astype(np.float32),
            "joint_acceleration": acceleration.astype(np.float32),
            "confidence": np.clip(confidence, 0., 1.).astype(np.float32),
            "valid_mask": valid,
            "tracking_state": int(frame.tracking_state),
        }
