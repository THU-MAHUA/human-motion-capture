"""Intel RealSense D4xx RGB-D capture adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray
    depth: np.ndarray
    timestamp_us: int
    color_timestamp_us: int
    depth_timestamp_us: int
    intrinsics: tuple[float, float, float, float]
    depth_scale: float
    timestamp_delta_ms: float = 0.0


class RealSenseD4xxCapture:
    """Capture aligned color and depth frames from one RealSense D4xx."""

    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        bag_path: str | None = None,
        sync_tolerance_ms: float = 5.0,
    ):
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.bag_path = bag_path
        self.sync_tolerance_ms = float(sync_tolerance_ms)
        self._rs: Any = None
        self._pipeline: Any = None
        self._profile: Any = None
        self._align: Any = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self.depth_scale = 0.001
        self.device_name = ""
        self.device_serial = ""
        self.firmware_version = ""
        self.usb_type = ""
        self.sync_drops = 0
        self.timestamp_drops = 0
        self._last_timestamp_us: int | None = None

    @staticmethod
    def _device_info(device: Any, rs: Any, field: Any) -> str:
        try:
            return device.get_info(field)
        except RuntimeError:
            return ""

    @classmethod
    def _select_device(
        cls,
        devices: list[Any],
        rs: Any,
        serial: str | None,
    ) -> Any:
        if not devices:
            raise RuntimeError("no Intel RealSense device found")
        available = [
            cls._device_info(device, rs, rs.camera_info.serial_number)
            for device in devices
        ]
        if serial is None:
            if len(devices) > 1:
                raise RuntimeError(
                    "multiple RealSense devices are connected; pass --serial "
                    f"with one of {available}"
                )
            return devices[0]
        for device, device_serial in zip(devices, available):
            if device_serial == serial:
                return device
        raise RuntimeError(
            f"RealSense serial {serial!r} was not found; available: {available}"
        )

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required; install the Intel RealSense Python SDK"
            ) from exc

        self._rs = rs
        devices = list(rs.context().query_devices())
        selected = self._select_device(devices, rs, self.serial)

        self.device_name = self._device_info(selected, rs, rs.camera_info.name)
        self.device_serial = self._device_info(
            selected, rs, rs.camera_info.serial_number
        )
        self.firmware_version = self._device_info(
            selected, rs, rs.camera_info.firmware_version
        )
        self.usb_type = self._device_info(
            selected, rs, rs.camera_info.usb_type_descriptor
        )

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.device_serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        if self.bag_path:
            bag = Path(self.bag_path).expanduser().resolve()
            bag.parent.mkdir(parents=True, exist_ok=True)
            config.enable_record_to_file(str(bag))

        try:
            profile = pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(
                f"could not start {self.device_name} ({self.device_serial}): {exc}"
            ) from exc

        self._pipeline = pipeline
        self._profile = profile
        self._align = rs.align(rs.stream.color)
        self._last_timestamp_us = None
        self.depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )

    def read(self, timeout_ms: int = 1000) -> RGBDFrame | None:
        if self._pipeline is None:
            raise RuntimeError("capture is not open")

        for _ in range(5):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms)
            except RuntimeError:
                return None
            aligned = self._align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            color_ts_ms = float(color.get_timestamp())
            depth_ts_ms = float(depth.get_timestamp())
            delta_ms = abs(color_ts_ms - depth_ts_ms)
            if delta_ms > self.sync_tolerance_ms:
                self.sync_drops += 1
                continue

            color_array = np.asanyarray(color.get_data()).copy()
            depth_array = np.asanyarray(depth.get_data()).copy()
            intrinsic = color.profile.as_video_stream_profile().intrinsics
            self._intrinsics = (
                float(intrinsic.fx),
                float(intrinsic.fy),
                float(intrinsic.ppx),
                float(intrinsic.ppy),
            )
            color_ts_us = int(round(color_ts_ms * 1000.0))
            depth_ts_us = int(round(depth_ts_ms * 1000.0))
            timestamp_us = max(color_ts_us, depth_ts_us)
            if (
                self._last_timestamp_us is not None
                and timestamp_us <= self._last_timestamp_us
            ):
                self.timestamp_drops += 1
                continue
            self._last_timestamp_us = timestamp_us
            return RGBDFrame(
                color_bgr=color_array,
                depth=depth_array,
                timestamp_us=timestamp_us,
                color_timestamp_us=color_ts_us,
                depth_timestamp_us=depth_ts_us,
                intrinsics=self._intrinsics,
                depth_scale=self.depth_scale,
                timestamp_delta_ms=delta_ms,
            )
        return None

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.stop()
        self._profile = None
        self._align = None
        self._last_timestamp_us = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_args):
        self.close()
