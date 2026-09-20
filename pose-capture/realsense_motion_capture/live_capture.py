"""Live RealSense D4xx -> MMPose/HMR2 -> human-motion episode capture.

Example::

    PYTHONPATH=. python projects/realsense_motion_capture/live_capture.py \
        --output-dir recordings/wave --serial 313522072015 --device cuda:0

Controls: ``r`` toggles recording of processed frames, ``q`` quits. The raw
RealSense bag starts when the process starts unless ``--no-bag`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .runtime_env import prepare_opencv_import, require_environment_package
except ImportError:  # direct ``python live_capture.py`` execution
    from runtime_env import prepare_opencv_import, require_environment_package

prepare_opencv_import()

import cv2
import numpy as np

require_environment_package(cv2, "OpenCV")

try:
    from .dataset import EpisodeWriter, HumanEpisodeWriter
    from .hmr2_adapter import HMR2OnePerson, HMR2Process
    from .realsense import RealSenseD4xxCapture
    from .pose import OnePersonPose
    from .processing import (COCO17_EDGES, COCO17_NAMES, WHOLEBODY_EDGES,
                             WHOLEBODY_NAMES, SkeletonFrame, SkeletonProcessor,
                             depth_to_xyz)
except ImportError:  # direct ``python live_capture.py`` execution
    from dataset import EpisodeWriter, HumanEpisodeWriter
    from hmr2_adapter import HMR2OnePerson, HMR2Process
    from realsense import RealSenseD4xxCapture
    from pose import OnePersonPose
    from processing import (COCO17_EDGES, COCO17_NAMES, WHOLEBODY_EDGES,
                            WHOLEBODY_NAMES, SkeletonFrame, SkeletonProcessor,
                            depth_to_xyz)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="recordings/session")
    parser.add_argument("--serial", default=None, help="RealSense camera serial number")
    parser.add_argument("--device", default="cuda:0", help="MMPose device, e.g. cuda:0 or cpu")
    parser.add_argument("--pose2d", default="rtmw-x_8xb320-270e_cocktail14-384x288", help="MMPose 2D model alias/config")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sync-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--depth-window", type=int, default=2)
    parser.add_argument("--no-bag", action="store_true", help="do not save raw RealSense .bag data")
    parser.add_argument("--ros", action="store_true", help="publish joints and markers on ROS 2 topics")
    parser.add_argument("--record-immediately", action="store_true", help="save processed frames without pressing r")
    parser.add_argument("--headless", action="store_true", help="capture without opening an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop cleanly after this many camera frames")
    parser.add_argument("--hmr2-root", default=None,
                        help="mesh-capture/4D-Humans checkout; enables SMPL human output")
    parser.add_argument("--hmr2-checkpoint", default=None)
    parser.add_argument("--hmr2-device", default="cuda:0")
    parser.add_argument("--hmr2-python", default=None,
                        help="Python executable for a separate 4D-Humans environment")
    parser.add_argument(
        "--mesh-preview",
        action="store_true",
        help="show a split skeleton and live white HMR2 mesh preview",
    )
    parser.add_argument(
        "--preview-panel-width",
        type=int,
        default=800,
        help="width of each panel in split mesh preview mode",
    )
    return parser.parse_args()


def draw_skeleton(image, points, valid, confidence, edges):
    output = image.copy()
    for first, second in edges:
        if valid[first] and valid[second]:
            cv2.line(output, tuple(np.int32(points[first])), tuple(np.int32(points[second])), (0, 220, 80), 2)
    for index, (point, is_valid) in enumerate(zip(points, valid)):
        if is_valid:
            cv2.circle(output, tuple(np.int32(point)), 4, (40, 80, 255), -1)
            cv2.putText(output, str(index), tuple(np.int32(point + (5, -5))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return output


def make_split_preview(
    skeleton_bgr: np.ndarray,
    mesh_bgr: np.ndarray | None,
    panel_width: int,
) -> np.ndarray:
    if panel_width < 320:
        raise ValueError("preview panel width must be at least 320 pixels")
    height, width = skeleton_bgr.shape[:2]
    panel_height = max(1, int(round(height * panel_width / width)))
    size = (panel_width, panel_height)
    left = cv2.resize(skeleton_bgr, size, interpolation=cv2.INTER_AREA)
    right_source = mesh_bgr if mesh_bgr is not None else np.zeros_like(skeleton_bgr)
    right = cv2.resize(right_source, size, interpolation=cv2.INTER_AREA)

    if mesh_bgr is None:
        cv2.putText(
            right,
            "HMR2 MESH UNAVAILABLE",
            (20, panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    for panel, label in ((left, "RTMO-L SKELETON"), (right, "HMR2 WHITE MESH")):
        cv2.rectangle(panel, (0, 0), (panel_width, 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return np.hstack((left, right))


def main():
    args = parse_args()
    if args.mesh_preview and not args.hmr2_root:
        raise SystemExit("--mesh-preview requires --hmr2-root")
    if args.mesh_preview and args.headless:
        raise SystemExit("--mesh-preview cannot be combined with --headless")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bag_path = None if args.no_bag else str(output_dir / "rgbd.bag")
    capture = RealSenseD4xxCapture(
        args.serial,
        args.width,
        args.height,
        args.fps,
        bag_path=bag_path,
        sync_tolerance_ms=args.sync_tolerance_ms,
    )
    pose = OnePersonPose(args.pose2d, args.device)
    hmr2 = None
    if args.hmr2_root:
        hmr2 = (HMR2Process(
                    args.hmr2_python,
                    args.hmr2_root,
                    args.hmr2_checkpoint,
                    args.hmr2_device,
                    mesh_preview=args.mesh_preview)
                if args.hmr2_python else
                HMR2OnePerson(
                    args.hmr2_root,
                    args.hmr2_checkpoint,
                    args.hmr2_device,
                    mesh_preview=args.mesh_preview))
    ros = None
    writer = None
    joint_names = WHOLEBODY_NAMES if pose.joint_count == 133 else COCO17_NAMES
    joint_edges = WHOLEBODY_EDGES if pose.joint_count == 133 else COCO17_EDGES
    processor = SkeletonProcessor(joint_count=pose.joint_count)
    recording = bool(args.record_immediately)
    frame_count = 0
    dropped = 0
    started = time.monotonic()
    try:
        capture.open()
        pose.open()
        if hmr2 is not None:
            hmr2.open()
        metadata = {
            "format": "realsense_human_motion_hmr2" if hmr2 else "realsense_motion_capture_v1",
            "joint_names": list(joint_names),
            "joint_edges": [list(edge) for edge in joint_edges],
            "coordinate_frame": "realsense_color_optical",
            "coordinate_convention": "x right, y down, z forward; metres",
            "fps": args.fps,
            "camera": capture.device_name,
            "camera_model": capture.device_name,
            "serial": capture.device_serial,
            "firmware_version": capture.firmware_version,
            "usb_type": capture.usb_type,
            "color_profile": [args.width, args.height, "bgr8", args.fps],
            "depth_profile": [args.width, args.height, "z16", args.fps],
            "sync_tolerance_ms": args.sync_tolerance_ms,
            "pose2d": args.pose2d,
            "device": args.device,
            "intrinsics": list(capture._intrinsics or ()),
            "depth_scale": capture.depth_scale,
        }
        if hmr2 is not None:
            metadata.update({
                "smpl_coordinate_frame": "hmr2_camera",
                "hmr2_root": str(hmr2.hmr2_root),
                "hmr2_detector": hmr2.detector_name,
                "hmr2_checkpoint": hmr2.checkpoint_path,
                "smpl_model": "SMPL neutral",
                "mesh_preview": args.mesh_preview,
            })
            writer = HumanEpisodeWriter(output_dir, metadata)
        else:
            writer = EpisodeWriter(output_dir, metadata)
        if args.ros:
            try:
                from .ros_publish import ROSPublisher
            except ImportError:
                from ros_publish import ROSPublisher
            ros = ROSPublisher()
        print(f"Connected to {capture.device_name} ({capture.device_serial})")
        print("Press r to start/stop processed recording; q or ESC to quit.")
        while True:
            frame = capture.read()
            if frame is None:
                dropped += 1
                continue
            keypoints, scores, tracking_state = pose.predict(frame.color_bgr)
            joints_xyz, depth_valid = depth_to_xyz(
                keypoints, frame.depth, frame.intrinsics, frame.depth_scale, args.depth_window)
            keypoint_valid = np.isfinite(keypoints).all(axis=1) & (scores >= 0.25)
            valid = keypoint_valid & depth_valid
            processed = processor.update(SkeletonFrame(
                frame.timestamp_us / 1e6, joints_xyz, scores, valid, tracking_state))
            smpl_valid = None
            mesh_preview_bgr = None
            if hmr2 is not None:
                smpl = hmr2.predict(frame.color_bgr)
                smpl_valid = smpl.valid
                mesh_preview_bgr = smpl.mesh_preview_bgr
                processed.update(smpl.as_dict())
            if recording:
                writer.append(processed)
            preview = draw_skeleton(frame.color_bgr, keypoints, keypoint_valid, scores, joint_edges)
            status = f"FPS {frame_count / max(1e-6, time.monotonic()-started):.1f} | {'REC' if recording else 'READY'} | joints {valid.sum()}/{pose.joint_count}"
            if smpl_valid is not None:
                status += f" | HMR2 {'OK' if smpl_valid else 'MISS'}"
            cv2.putText(preview, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if args.mesh_preview:
                preview = make_split_preview(
                    preview,
                    mesh_preview_bgr,
                    args.preview_panel_width,
                )
            if writer is not None:
                if not writer.metadata.get("depth_scale_measured"):
                    writer.metadata.update({
                        "intrinsics": list(frame.intrinsics),
                        "depth_scale": frame.depth_scale,
                        "depth_scale_measured": True,
                    })
                writer.metadata["last_timestamp_delta_ms"] = frame.timestamp_delta_ms
                writer.metadata["max_timestamp_delta_ms"] = max(
                    frame.timestamp_delta_ms,
                    writer.metadata.get("max_timestamp_delta_ms", 0.0),
                )
            if not args.headless:
                window_title = (
                    "RealSense D435 | RTMO-L skeleton + HMR2 mesh"
                    if args.mesh_preview
                    else "RealSense D435 motion capture"
                )
                cv2.imshow(window_title, preview)
            if ros is not None:
                ros.publish(processed, joint_edges)
            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
            key = cv2.waitKey(1) & 0xFF if not args.headless else -1
            if key == ord("r"):
                recording = not recording
                if recording:
                    processor.reset()
                    print("Recording started")
                else:
                    print("Recording paused")
            elif key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        if writer is not None and len(writer):
            episode_path = writer.save()
            print(f"Saved {len(writer)} frames to {episode_path}")
            (output_dir / "runtime.json").write_text(
                json.dumps({
                    "dropped_frames": dropped,
                    "sync_drops": capture.sync_drops,
                    "timestamp_drops": capture.timestamp_drops,
                }, indent=2),
                encoding="utf-8",
            )
        if ros is not None:
            ros.close()
        capture.close()
        if hmr2 is not None:
            hmr2.close()


if __name__ == "__main__":
    main()
