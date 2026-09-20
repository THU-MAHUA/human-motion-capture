"""Re-run pose extraction on an Intel RealSense ``.bag`` recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .dataset import EpisodeWriter, HumanEpisodeWriter
    from .hmr2_adapter import HMR2OnePerson, HMR2Process
    from .pose import OnePersonPose
    from .processing import (
        COCO17_NAMES,
        WHOLEBODY_NAMES,
        SkeletonFrame,
        SkeletonProcessor,
        depth_to_xyz,
    )
    from .realsense import RGBDFrame
except ImportError:
    from dataset import EpisodeWriter, HumanEpisodeWriter
    from hmr2_adapter import HMR2OnePerson, HMR2Process
    from pose import OnePersonPose
    from processing import (
        COCO17_NAMES,
        WHOLEBODY_NAMES,
        SkeletonFrame,
        SkeletonProcessor,
        depth_to_xyz,
    )
    from realsense import RGBDFrame


def _device_info(device, rs, field):
    try:
        return device.get_info(field)
    except RuntimeError:
        return ""


def replay_frames(
    bag_path: str,
    metadata: dict | None = None,
    sync_tolerance_ms: float = 5.0,
):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for bag replay") from exc

    bag = str(Path(bag_path).expanduser().resolve())
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag, repeat_playback=False)
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    depth_scale = float(device.first_depth_sensor().get_depth_scale())

    if metadata is not None:
        metadata.update({
            "camera": _device_info(device, rs, rs.camera_info.name),
            "serial": _device_info(device, rs, rs.camera_info.serial_number),
            "firmware_version": _device_info(
                device, rs, rs.camera_info.firmware_version
            ),
        })

    try:
        last_timestamp_us = None
        while True:
            try:
                frames = pipeline.wait_for_frames(1000)
            except RuntimeError:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                continue
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            color_ts_ms = float(color.get_timestamp())
            depth_ts_ms = float(depth.get_timestamp())
            delta_ms = abs(color_ts_ms - depth_ts_ms)
            if delta_ms > sync_tolerance_ms:
                continue

            intrinsic = color.profile.as_video_stream_profile().intrinsics
            intrinsics = (
                float(intrinsic.fx),
                float(intrinsic.fy),
                float(intrinsic.ppx),
                float(intrinsic.ppy),
            )
            color_ts_us = int(round(color_ts_ms * 1000.0))
            depth_ts_us = int(round(depth_ts_ms * 1000.0))
            timestamp_us = max(color_ts_us, depth_ts_us)
            if last_timestamp_us is not None and timestamp_us <= last_timestamp_us:
                continue
            last_timestamp_us = timestamp_us
            yield RGBDFrame(
                color_bgr=np.asanyarray(color.get_data()).copy(),
                depth=np.asanyarray(depth.get_data()).copy(),
                timestamp_us=timestamp_us,
                color_timestamp_us=color_ts_us,
                depth_timestamp_us=depth_ts_us,
                intrinsics=intrinsics,
                depth_scale=depth_scale,
                timestamp_delta_ms=delta_ms,
            )
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pose2d",
        default="rtmw-x_8xb320-270e_cocktail14-384x288",
    )
    parser.add_argument("--sync-tolerance-ms", type=float, default=5.0)
    parser.add_argument(
        "--hmr2",
        action="store_true",
        help="enable optional HMR2 SMPL inference",
    )
    parser.add_argument(
        "--hmr2-root",
        default=None,
        help="optional legacy 4D-Humans checkout override",
    )
    parser.add_argument("--hmr2-checkpoint", default=None)
    parser.add_argument("--hmr2-device", default="cuda:0")
    parser.add_argument(
        "--hmr2-python",
        default=None,
        help="Python executable for a separate 4D-Humans environment",
    )
    args = parser.parse_args()

    pose = OnePersonPose(args.pose2d, args.device)
    hmr2 = None
    hmr2_enabled = bool(
        args.hmr2
        or args.hmr2_root
        or args.hmr2_checkpoint
        or args.hmr2_python
    )
    if hmr2_enabled:
        hmr2 = (
            HMR2Process(
                args.hmr2_python,
                args.hmr2_root,
                args.hmr2_checkpoint,
                args.hmr2_device,
            )
            if args.hmr2_python
            else HMR2OnePerson(
                args.hmr2_root,
                args.hmr2_checkpoint,
                args.hmr2_device,
            )
        )
    pose.open()
    if hmr2 is not None:
        hmr2.open()

    processor = SkeletonProcessor(joint_count=pose.joint_count)
    metadata = {
        "format": (
            "realsense_human_motion_hmr2_replay"
            if hmr2
            else "realsense_motion_capture_v1_replay"
        ),
        "joint_names": list(
            WHOLEBODY_NAMES if pose.joint_count == 133 else COCO17_NAMES
        ),
        "source_bag": str(Path(args.bag).resolve()),
        "pose2d": args.pose2d,
        "device": args.device,
        "coordinate_frame": "realsense_color_optical",
        "coordinate_convention": "x right, y down, z forward; metres",
        "sync_tolerance_ms": args.sync_tolerance_ms,
    }
    if hmr2 is not None:
        metadata.update({
            "smpl_coordinate_frame": "hmr2_camera",
            "hmr2_source": (
                str(hmr2.hmr2_root)
                if hmr2.hmr2_root is not None
                else "installed hmr2 package"
            ),
            "hmr2_detector": "regnety",
            "hmr2_checkpoint": hmr2.checkpoint_path,
            "smpl_model": "SMPL neutral",
        })
    writer = (
        HumanEpisodeWriter(args.output_dir, metadata)
        if hmr2
        else EpisodeWriter(args.output_dir, metadata)
    )

    try:
        for frame in replay_frames(
            args.bag,
            writer.metadata,
            args.sync_tolerance_ms,
        ):
            if "intrinsics" not in writer.metadata:
                writer.metadata.update({
                    "intrinsics": list(frame.intrinsics),
                    "depth_scale": float(frame.depth_scale),
                    "depth_scale_measured": True,
                })
            writer.metadata["last_timestamp_delta_ms"] = frame.timestamp_delta_ms
            writer.metadata["max_timestamp_delta_ms"] = max(
                frame.timestamp_delta_ms,
                writer.metadata.get("max_timestamp_delta_ms", 0.0),
            )
            keypoints, scores, tracking_state = pose.predict(frame.color_bgr)
            xyz, depth_valid = depth_to_xyz(
                keypoints,
                frame.depth,
                frame.intrinsics,
                frame.depth_scale,
            )
            valid = (
                np.isfinite(keypoints).all(axis=1)
                & (scores >= 0.25)
                & depth_valid
            )
            processed = processor.update(
                SkeletonFrame(
                    frame.timestamp_us / 1e6,
                    xyz,
                    scores,
                    valid,
                    tracking_state,
                )
            )
            if hmr2 is not None:
                processed.update(hmr2.predict(frame.color_bgr).as_dict())
            writer.append(processed)
        if len(writer):
            print(f"Saved {len(writer)} frames to {writer.save()}")
    finally:
        if hmr2 is not None:
            hmr2.close()


if __name__ == "__main__":
    main()
