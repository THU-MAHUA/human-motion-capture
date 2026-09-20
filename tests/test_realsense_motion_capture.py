import numpy as np
import pytest

from realsense_motion_capture.dataset import (
    EpisodeWriter, HumanEpisodeWriter, load_episode, load_human_episode)
from realsense_motion_capture.hmr2_adapter import HMR2OnePerson, SMPLFrame
from realsense_motion_capture.live_capture import make_split_preview
from realsense_motion_capture.processing import (
    SkeletonFrame, SkeletonProcessor, depth_to_xyz)
from realsense_motion_capture.realsense import RealSenseD4xxCapture


class _CameraInfo:
    serial_number = "serial_number"


class _RS:
    camera_info = _CameraInfo()


class _Device:
    def __init__(self, serial):
        self.serial = serial

    def get_info(self, field):
        assert field == _CameraInfo.serial_number
        return self.serial


def test_realsense_selection_requires_serial_for_multiple_devices():
    devices = [_Device("d435"), _Device("d435i")]
    with pytest.raises(RuntimeError, match="multiple RealSense"):
        RealSenseD4xxCapture._select_device(devices, _RS, None)
    assert (
        RealSenseD4xxCapture._select_device(devices, _RS, "d435")
        is devices[0]
    )
    with pytest.raises(RuntimeError, match="missing"):
        RealSenseD4xxCapture._select_device(devices, _RS, "missing")


def test_realsense_read_aligns_and_discards_unsynchronized_frames():
    class Intrinsics:
        fx, fy, ppx, ppy = 600.0, 601.0, 640.0, 360.0

    class Profile:
        def as_video_stream_profile(self):
            return self

        intrinsics = Intrinsics()

    class Frame:
        profile = Profile()

        def __init__(self, timestamp, data):
            self.timestamp = timestamp
            self.data = data

        def __bool__(self):
            return True

        def get_timestamp(self):
            return self.timestamp

        def get_data(self):
            return self.data

    class FrameSet:
        def __init__(self, color_timestamp, depth_timestamp):
            self.color = Frame(
                color_timestamp, np.zeros((2, 3, 3), dtype=np.uint8))
            self.depth = Frame(
                depth_timestamp, np.full((2, 3), 1234, dtype=np.uint16))

        def get_color_frame(self):
            return self.color

        def get_depth_frame(self):
            return self.depth

    class Pipeline:
        def __init__(self):
            self.frames = iter([
                FrameSet(100.0, 110.0),
                FrameSet(133.025, 133.0),
            ])

        def wait_for_frames(self, _timeout):
            return next(self.frames)

    class Align:
        calls = 0

        def process(self, frames):
            self.calls += 1
            return frames

    capture = RealSenseD4xxCapture(sync_tolerance_ms=5.0)
    capture._pipeline = Pipeline()
    capture._align = Align()
    capture.depth_scale = 0.001
    frame = capture.read()
    assert capture.sync_drops == 1
    assert capture._align.calls == 2
    assert frame.timestamp_us == 133025
    assert frame.timestamp_delta_ms == pytest.approx(0.025)
    assert frame.intrinsics == (600.0, 601.0, 640.0, 360.0)
    assert frame.depth_scale == pytest.approx(0.001)
    assert frame.color_bgr.shape == (2, 3, 3)
    assert frame.depth.shape == (2, 3)


def test_depth_to_xyz_uses_median_and_intrinsics():
    depth = np.full((8, 8), 2000, dtype=np.uint16)
    depth[3:6, 3:6] = np.array([[2000, 2100, 2000], [2000, 2200, 2000], [2000, 2000, 2000]], dtype=np.uint16)
    xyz, valid = depth_to_xyz(np.array([[4., 4.], [-1., 2.]], dtype=np.float32), depth, (100., 100., 4., 4.))
    assert valid.tolist() == [True, False]
    assert np.allclose(xyz[0], [0., 0., 2.], atol=0.001)


def test_processor_derivatives_and_quaternion():
    processor = SkeletonProcessor(smoothing=1.0)
    xyz = np.zeros((17, 3), dtype=np.float32)
    xyz[5], xyz[6], xyz[11], xyz[12] = [-1, 0, 2], [1, 0, 2], [-1, 0, 0], [1, 0, 0]
    first = processor.update(SkeletonFrame(1.0, xyz, np.ones(17), np.ones(17, bool), 1))
    xyz[0, 0] = 1.
    second = processor.update(SkeletonFrame(2.0, xyz, np.ones(17), np.ones(17, bool), 1))
    assert second["joint_velocity"][0, 0] == pytest.approx(1.)
    assert second["joint_acceleration"][0, 0] == pytest.approx(1.)
    assert np.isclose(np.linalg.norm(first["root_rotation"]), 1.)


def test_episode_round_trip(tmp_path):
    processor = SkeletonProcessor()
    writer = EpisodeWriter(tmp_path, {"fps": 30})
    for index in range(3):
        frame = processor.update(SkeletonFrame(float(index + 1), np.zeros((17, 3), np.float32),
                                               np.ones(17), np.ones(17, bool), 1))
        writer.append(frame)
    path = writer.save()
    arrays = load_episode(path)
    assert arrays["joints_xyz"].shape == (3, 17, 3)
    assert arrays["valid_mask"].dtype == bool


def test_episode_rejects_non_monotonic_timestamps(tmp_path):
    writer = EpisodeWriter(tmp_path)
    base = {"timestamp": 2.0}
    with pytest.raises(ValueError):
        writer.append(base)
        writer.append({"timestamp": 1.0})


def test_human_episode_round_trip(tmp_path):
    processor = SkeletonProcessor()
    writer = HumanEpisodeWriter(tmp_path, {"format": "human_test"})
    for index in range(2):
        frame = processor.update(SkeletonFrame(
            float(index + 1), np.zeros((17, 3), np.float32),
            np.ones(17), np.ones(17, bool), 1))
        frame.update(SMPLFrame.empty(44).as_dict())
        writer.append(frame)
    arrays = load_human_episode(writer.save())
    assert arrays["smpl_global_orient"].shape == (2, 3, 3)
    assert arrays["smpl_body_pose"].shape == (2, 23, 3, 3)
    assert arrays["smpl_joints_root_relative"].shape == (2, 44, 3)
    assert arrays["smpl_valid"].dtype == bool


def test_smpl_preview_is_not_added_to_episode_schema():
    frame = SMPLFrame.empty(44)
    frame.mesh_preview_bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    assert "mesh_preview_bgr" not in frame.as_dict()


def test_split_preview_has_two_stable_panels():
    skeleton = np.zeros((720, 1280, 3), dtype=np.uint8)
    mesh = np.full_like(skeleton, 255)
    preview = make_split_preview(skeleton, mesh, panel_width=800)
    assert preview.shape == (450, 1600, 3)
    assert preview.dtype == np.uint8


def test_hmr2_bbox_tracking_prefers_temporal_continuity():
    estimator = HMR2OnePerson(".", bbox_smoothing=1.0)
    estimator._previous_bbox = np.array([10, 10, 30, 50], dtype=np.float32)

    class Boxes:
        tensor = np.array([[11, 10, 31, 50], [80, 10, 120, 90]], dtype=np.float32)

    class Instances:
        pred_classes = np.array([0, 0])
        scores = np.array([0.7, 0.95], dtype=np.float32)
        pred_boxes = Boxes()

        def to(self, _device):
            return self

    estimator._detector = lambda _image: {"instances": Instances()}
    bbox, score = estimator._select_bbox(np.zeros((100, 140, 3), dtype=np.uint8))
    assert np.allclose(bbox, [11, 10, 31, 50])
    assert score == pytest.approx(0.7)


def test_hmr2_bbox_survives_short_detector_loss():
    estimator = HMR2OnePerson(".", max_missed_detections=2)
    estimator._previous_bbox = np.array([10, 10, 30, 50], dtype=np.float32)

    class Boxes:
        tensor = np.empty((0, 4), dtype=np.float32)

    class Instances:
        pred_classes = np.empty(0, dtype=np.int64)
        scores = np.empty(0, dtype=np.float32)
        pred_boxes = Boxes()

        def to(self, _device):
            return self

    estimator._detector = lambda _image: {"instances": Instances()}
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    assert estimator._select_bbox(image)[0] is not None
    assert estimator._select_bbox(image)[0] is not None
    assert estimator._select_bbox(image)[0] is None
