"""Optional 4D-Humans HMR2 inference adapter.

The adapter deliberately imports 4D-Humans and Detectron2 only from ``open``.
This keeps RealSense capture, replay, and the existing MMPose tools usable in
their dedicated environment.
"""

from __future__ import annotations

import sys
import json
import os
import struct
import subprocess
import tarfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SMPLFrame:
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    cam_t: np.ndarray
    joints_root_relative: np.ndarray
    valid: bool
    detection_score: float = 0.0
    mesh_preview_bgr: np.ndarray | None = None

    @classmethod
    def empty(cls, joint_count: int, body_joint_count: int = 23) -> "SMPLFrame":
        return cls(
            global_orient=np.eye(3, dtype=np.float32),
            body_pose=np.tile(np.eye(3, dtype=np.float32), (body_joint_count, 1, 1)),
            betas=np.zeros(10, dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
            joints_root_relative=np.zeros((joint_count, 3), dtype=np.float32),
            valid=False,
        )

    def as_dict(self) -> dict[str, np.ndarray | bool]:
        return {
            "smpl_global_orient": self.global_orient,
            "smpl_body_pose": self.body_pose,
            "smpl_betas": self.betas,
            "smpl_cam_t": self.cam_t,
            "smpl_joints_root_relative": self.joints_root_relative,
            "smpl_valid": self.valid,
        }


class HMR2OnePerson:
    """Run HMR2.0 on exactly one detected person in a BGR image."""

    def __init__(self, hmr2_root: str | Path | None = None,
                 checkpoint: str | None = None,
                 device: str = "cuda:0",
                 detector: str = "regnety",
                 bbox_smoothing: float = 0.5,
                 min_detection_score: float = 0.5,
                 max_missed_detections: int = 5,
                 mesh_preview: bool = False):
        self.hmr2_root = (
            Path(hmr2_root).expanduser().resolve()
            if hmr2_root is not None
            else None
        )
        self.checkpoint = checkpoint
        self.checkpoint_path = ""
        self.device_name = device
        self.detector_name = detector
        self.bbox_smoothing = float(bbox_smoothing)
        self.min_detection_score = float(min_detection_score)
        self.max_missed_detections = max(0, int(max_missed_detections))
        self.mesh_preview = bool(mesh_preview)
        if not 0.0 < self.bbox_smoothing <= 1.0:
            raise ValueError("bbox_smoothing must be in (0, 1]")
        self._model: Any = None
        self._cfg: Any = None
        self._detector: Any = None
        self._torch: Any = None
        self._recursive_to: Any = None
        self._cam_crop_to_full: Any = None
        self._renderer: Any = None
        self._previous_bbox: np.ndarray | None = None
        self._missed_detections = 0
        self.smpl_joint_count = 0
        self.body_joint_count = 23

    def open(self) -> None:
        if self.hmr2_root is not None:
            if not self.hmr2_root.is_dir():
                raise FileNotFoundError(
                    f"4D-Humans repository not found: {self.hmr2_root}"
                )
            root = str(self.hmr2_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            import hmr2
            import torch
            from hmr2.models import download_models, load_hmr2
            from hmr2.utils import recursive_to
            from hmr2.utils.renderer import Renderer, cam_crop_to_full
            from hmr2.configs import CACHE_DIR_4DHUMANS
        except ImportError as exc:
            raise RuntimeError(
                "4D-Humans dependencies are missing; use its Python 3.10 "
                f"environment (original error: {exc})") from exc

        try:
            if self.hmr2_root is None:
                self.hmr2_root = Path(hmr2.__file__).resolve().parent.parent
            cache_smpl = (
                Path(CACHE_DIR_4DHUMANS)
                / "data"
                / "smpl"
                / "SMPL_NEUTRAL.pkl"
            )
            checkout_smpl = (
                self.hmr2_root
                / "data"
                / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
            )
            if not cache_smpl.is_file() and not checkout_smpl.is_file():
                raise FileNotFoundError(
                    "The licensed neutral SMPL model is missing. Run "
                    "`motion-capture-download-models --install-smpl PATH` "
                    f"to install it at {cache_smpl}."
                )
            if self.checkpoint is None:
                download_models(CACHE_DIR_4DHUMANS)
                from hmr2.models import DEFAULT_CHECKPOINT
                checkpoint = DEFAULT_CHECKPOINT
                if not Path(checkpoint).is_file():
                    archive = Path(CACHE_DIR_4DHUMANS) / "hmr2_data.tar.gz"
                    archive_note = ""
                    if archive.exists():
                        try:
                            with tarfile.open(archive, "r:*") as tar:
                                tar.next()
                        except (tarfile.TarError, EOFError, OSError) as exc:
                            archive_note = (
                                f" Cached archive {archive} is incomplete ({exc})."
                            )
                    raise RuntimeError(
                        "HMR2 model data is not fully installed; expected "
                        f"checkpoint/config at {checkpoint}." + archive_note +
                        " Remove or rename the incomplete archive and resume "
                        "the 2.5 GB download before starting capture.")
            else:
                checkpoint = str(Path(self.checkpoint).expanduser().resolve())
            self.checkpoint_path = str(Path(checkpoint).expanduser().resolve())
            # HMR2 checkpoints contain an OmegaConf object in addition to the
            # tensor state dict. PyTorch 2.6 changed torch.load's default to
            # weights_only=True, which rejects that trusted local checkpoint.
            # Keep the compatibility override local to model initialization;
            # the checkpoint is downloaded by 4D-Humans and explicitly chosen
            # by the user.
            load_checkpoint = torch.load
            def load_legacy_checkpoint(*args, **kwargs):
                if kwargs.get("weights_only") is None:
                    kwargs["weights_only"] = False
                return load_checkpoint(*args, **kwargs)
            torch.load = load_legacy_checkpoint
            previous_cwd = Path.cwd()
            try:
                # 4D-Humans checks the licensed neutral SMPL file using a
                # repository-relative path.
                os.chdir(self.hmr2_root)
                model, cfg = load_hmr2(checkpoint)
            finally:
                os.chdir(previous_cwd)
                torch.load = load_checkpoint
            device = torch.device(self.device_name)
            model = model.to(device).eval()
            self._model, self._cfg, self._torch = model, cfg, torch
            self._recursive_to, self._cam_crop_to_full = recursive_to, cam_crop_to_full
            if self.mesh_preview:
                self._renderer = Renderer(cfg, faces=model.smpl.faces)
            self._detector = self._build_detector()
            self.body_joint_count = int(getattr(cfg.SMPL, "NUM_BODY_JOINTS", 23))
            self.smpl_joint_count = int(model.smpl.joint_map.numel())
            if hasattr(model.smpl, "joint_regressor_extra"):
                self.smpl_joint_count += int(model.smpl.joint_regressor_extra.shape[0])
        except Exception as exc:
            raise RuntimeError(f"could not initialize HMR2: {exc}") from exc

    def _build_detector(self):
        if self.detector_name != "regnety":
            raise ValueError("only the faster RegNetY detector is supported")
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
            cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
                trained=True)
            cfg.model.roi_heads.box_predictor.test_score_thresh = self.min_detection_score
            cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            return DefaultPredictor_Lazy(cfg)
        except ImportError as exc:
            raise RuntimeError(
                "Detectron2 is required by the HMR2 person detector") from exc

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float:
        x0, y0 = np.maximum(first[:2], second[:2])
        x1, y1 = np.minimum(first[2:], second[2:])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = area_first + area_second - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _select_bbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        output = self._detector(image_bgr)
        instances = output["instances"].to("cpu")
        classes = np.asarray(instances.pred_classes)
        scores = np.asarray(instances.scores, dtype=np.float32)
        boxes = np.asarray(instances.pred_boxes.tensor, dtype=np.float32)
        candidates = np.flatnonzero((classes == 0) & (scores >= self.min_detection_score))
        if candidates.size == 0:
            self._missed_detections += 1
            if self._previous_bbox is not None and self._missed_detections <= self.max_missed_detections:
                return self._previous_bbox.copy(), 0.0
            self._previous_bbox = None
            return None, 0.0
        if self._previous_bbox is None:
            index = int(candidates[np.argmax(scores[candidates])])
        else:
            index = int(max(candidates, key=lambda i: float(scores[i]) + self._iou(boxes[i], self._previous_bbox)))
        bbox = boxes[index].copy()
        if self._previous_bbox is not None:
            bbox = self.bbox_smoothing * bbox + (1.0 - self.bbox_smoothing) * self._previous_bbox
        height, width = image_bgr.shape[:2]
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        self._previous_bbox = bbox
        self._missed_detections = 0
        return bbox, float(scores[index])

    def _render_mesh_preview(
        self,
        image_bgr: np.ndarray,
        vertices: np.ndarray,
        camera_translation: np.ndarray,
        focal_length: float,
    ) -> np.ndarray | None:
        if self._renderer is None:
            return None
        height, width = image_bgr.shape[:2]
        # The upstream renderer prints mesh dimensions for each frame. Keep the
        # binary worker's diagnostics quiet while retaining its tested renderer.
        with redirect_stdout(StringIO()):
            rgba = self._renderer.render_rgba_multiple(
                [vertices],
                cam_t=[camera_translation],
                render_res=[width, height],
                focal_length=focal_length,
                mesh_base_color=(1.0, 1.0, 1.0),
                scene_bg_color=(0.0, 0.0, 0.0),
            )
        rgba = np.asarray(rgba, dtype=np.float32)
        alpha = np.clip(rgba[:, :, 3:4], 0.0, 1.0)
        camera_rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        overlay_rgb = rgba[:, :, :3] * alpha + camera_rgb * (1.0 - alpha)
        return np.clip(overlay_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)

    def predict(self, image_bgr: np.ndarray) -> SMPLFrame:
        if self._model is None:
            raise RuntimeError("HMR2 estimator is not open")
        bbox, detection_score = self._select_bbox(image_bgr)
        if bbox is None:
            return SMPLFrame.empty(self.smpl_joint_count, self.body_joint_count)
        try:
            from hmr2.datasets.vitdet_dataset import ViTDetDataset
            dataset = ViTDetDataset(self._cfg, image_bgr, bbox[None])
            loader = self._torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            batch = self._recursive_to(next(iter(loader)), self._torch.device(self.device_name))
            with self._torch.no_grad():
                output = self._model(batch)
            params = output["pred_smpl_params"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            image_size = batch["img_size"].float()
            focal = self._cfg.EXTRA.FOCAL_LENGTH / self._cfg.MODEL.IMAGE_SIZE * image_size.max()
            cam_t = self._cam_crop_to_full(
                output["pred_cam"], box_center, box_size, image_size, focal)
            camera_translation = cam_t[0].detach().cpu().numpy().astype(np.float32)
            mesh_preview_bgr = self._render_mesh_preview(
                image_bgr,
                output["pred_vertices"][0].detach().cpu().numpy().astype(np.float32),
                camera_translation,
                float(focal.detach().cpu()),
            )
            joints = output["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
            joints -= joints[[8]] if joints.shape[0] > 8 else joints.mean(axis=0, keepdims=True)
            return SMPLFrame(
                global_orient=params["global_orient"][0, 0].detach().cpu().numpy().astype(np.float32),
                body_pose=params["body_pose"][0].detach().cpu().numpy().astype(np.float32),
                betas=params["betas"][0].detach().cpu().numpy().astype(np.float32),
                cam_t=camera_translation,
                joints_root_relative=joints,
                valid=True,
                detection_score=detection_score,
                mesh_preview_bgr=mesh_preview_bgr,
            )
        except Exception:
            # Preserve a fixed-shape, explicitly invalid record on transient
            # detector/model failures; the RGB-D episode remains replayable.
            return SMPLFrame.empty(self.smpl_joint_count, self.body_joint_count)

    def close(self) -> None:
        self._model = self._detector = self._renderer = None
        self._previous_bbox = None
        self._missed_detections = 0


class HMR2Process:
    """HMR2 client backed by a dedicated Python 3.10 worker process."""

    def __init__(self, python: str | Path, hmr2_root: str | Path | None = None,
                 checkpoint: str | None = None, device: str = "cuda:0",
                 mesh_preview: bool = False):
        self.python = str(Path(python).expanduser().resolve())
        self.hmr2_root = (
            Path(hmr2_root).expanduser().resolve()
            if hmr2_root is not None
            else None
        )
        self.checkpoint = checkpoint
        self.checkpoint_path = ""
        self.device_name = device
        self.mesh_preview = bool(mesh_preview)
        self.detector_name = "regnety"
        self.smpl_joint_count = 0
        self.body_joint_count = 23
        self._process: subprocess.Popen | None = None

    @staticmethod
    def _read_exact(stream, length: int) -> bytes:
        result = bytearray()
        while len(result) < length:
            chunk = stream.read(length - len(result))
            if not chunk:
                raise EOFError("HMR2 worker closed its output stream")
            result.extend(chunk)
        return bytes(result)

    def _exchange(self, payload: bytes) -> bytes:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("HMR2 worker is not open")
        self._process.stdin.write(struct.pack("!Q", len(payload)))
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        size = struct.unpack("!Q", self._read_exact(self._process.stdout, 8))[0]
        return self._read_exact(self._process.stdout, size)

    def open(self) -> None:
        command = [
            self.python,
            "-m",
            "realsense_motion_capture.hmr2_worker",
            "--device",
            self.device_name,
        ]
        if self.hmr2_root is not None:
            command += ["--hmr2-root", str(self.hmr2_root)]
        if self.checkpoint:
            command += ["--checkpoint", self.checkpoint]
        if self.mesh_preview:
            command.append("--mesh-preview")
        # Keep the worker isolated from the ROS/user Python installation while
        # making the CUDA libraries installed in the HMR2 environment visible
        # to Detectron2's compiled extension.
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        python_root = Path(self.python).resolve().parent.parent
        cuda_paths = [
            python_root / "lib",
            python_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_runtime" / "lib",
            python_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cublas" / "lib",
            python_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusparse" / "lib",
            python_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusolver" / "lib",
        ]
        existing = environment.get("LD_LIBRARY_PATH", "")
        library_paths = [str(path) for path in cuda_paths if path.is_dir()]
        if existing:
            library_paths.append(existing)
        if library_paths:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, bufsize=0, env=environment,
            cwd=str(self.hmr2_root) if self.hmr2_root is not None else None)
        try:
            hello = json.loads(self._exchange(b"HELLO").decode("utf-8"))
            if "error" in hello:
                raise RuntimeError(hello["error"])
            self.smpl_joint_count = int(hello["smpl_joint_count"])
            self.body_joint_count = int(hello["body_joint_count"])
            self.checkpoint_path = str(hello["checkpoint"])
        except Exception:
            self.close()
            raise

    def predict(self, image_bgr: np.ndarray) -> SMPLFrame:
        try:
            import cv2
            ok, encoded = cv2.imencode(".png", image_bgr,
                                       [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise RuntimeError("could not encode HMR2 input frame")
            response = self._exchange(encoded.tobytes())
            from io import BytesIO
            with np.load(BytesIO(response), allow_pickle=False) as data:
                mesh_preview_bgr = None
                if "mesh_preview_jpeg" in data and data["mesh_preview_jpeg"].size:
                    mesh_preview_bgr = cv2.imdecode(
                        data["mesh_preview_jpeg"], cv2.IMREAD_COLOR)
                return SMPLFrame(
                    global_orient=data["global_orient"].copy(),
                    body_pose=data["body_pose"].copy(),
                    betas=data["betas"].copy(),
                    cam_t=data["cam_t"].copy(),
                    joints_root_relative=data["joints_root_relative"].copy(),
                    valid=bool(data["valid"]),
                    detection_score=float(data["detection_score"]),
                    mesh_preview_bgr=mesh_preview_bgr,
                )
        except Exception:
            return SMPLFrame.empty(self.smpl_joint_count, self.body_joint_count)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
