"""Episode persistence for RealSense human-motion datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ARRAYS = (
    "timestamps", "joints_xyz", "root_xyz", "root_rotation",
    "joints_root_relative", "joint_velocity", "joint_acceleration",
    "confidence", "valid_mask", "tracking_state",
)

SMPL_ARRAYS = (
    "smpl_global_orient", "smpl_body_pose", "smpl_betas", "smpl_cam_t",
    "smpl_joints_root_relative", "smpl_valid",
)


class EpisodeWriter:
    """Collect processed frames and write one compressed, self-describing episode."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self.frames: list[dict[str, Any]] = []

    def append(self, frame: dict[str, Any]) -> None:
        if self.frames and float(frame["timestamp"]) <= float(self.frames[-1]["timestamp"]):
            raise ValueError("episode timestamps must be strictly increasing")
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, filename: str = "episode.npz") -> Path:
        if not self.frames:
            raise ValueError("cannot save an empty episode")
        arrays = {name: np.stack([np.asarray(frame[name]) for frame in self.frames])
                  for name in REQUIRED_ARRAYS if name != "timestamps"}
        arrays["timestamps"] = np.asarray([frame["timestamp"] for frame in self.frames], dtype=np.float64)
        arrays["tracking_state"] = arrays["tracking_state"].astype(np.int8)
        arrays["valid_mask"] = arrays["valid_mask"].astype(bool)
        path = self.output_dir / filename
        np.savez_compressed(path, **arrays)
        metadata = dict(self.metadata)
        metadata.update({"num_frames": len(self.frames),
                         "joint_count": int(arrays["joints_xyz"].shape[1]),
                         "array_shapes": {key: list(value.shape) for key, value in arrays.items()}})
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return path


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    """Load an episode and validate its required shapes."""
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_ARRAYS if name not in data]
        if missing:
            raise ValueError(f"episode is missing arrays: {', '.join(missing)}")
        arrays = {name: data[name].copy() for name in REQUIRED_ARRAYS}
    length = len(arrays["timestamps"])
    if arrays["joints_xyz"].ndim != 3 or arrays["joints_xyz"].shape[0] != length or arrays["joints_xyz"].shape[2] != 3:
        raise ValueError("joints_xyz must have shape (T, K, 3)")
    joint_count = arrays["joints_xyz"].shape[1]
    if arrays["confidence"].shape != (length, joint_count):
        raise ValueError("confidence must have shape (T, K)")
    if arrays["valid_mask"].shape != (length, joint_count):
        raise ValueError("valid_mask must have shape (T, K)")
    if length > 1 and np.any(np.diff(arrays["timestamps"]) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return arrays


class HumanEpisodeWriter(EpisodeWriter):
    """Write the base RGB-D skeleton plus one HMR2 SMPL result per frame."""

    def save(self, filename: str = "episode.npz") -> Path:
        if not self.frames:
            raise ValueError("cannot save an empty episode")
        missing = [name for name in (*REQUIRED_ARRAYS, *SMPL_ARRAYS)
                   if name != "timestamps" and name not in self.frames[0]]
        if missing:
            raise ValueError(f"human episode is missing arrays: {', '.join(missing)}")
        arrays = {name: np.stack([np.asarray(frame[name]) for frame in self.frames])
                  for name in (*REQUIRED_ARRAYS, *SMPL_ARRAYS) if name != "timestamps"}
        arrays["timestamps"] = np.asarray(
            [frame["timestamp"] for frame in self.frames], dtype=np.float64)
        arrays["tracking_state"] = arrays["tracking_state"].astype(np.int8)
        arrays["valid_mask"] = arrays["valid_mask"].astype(bool)
        arrays["smpl_valid"] = arrays["smpl_valid"].astype(bool)
        path = self.output_dir / filename
        np.savez_compressed(path, **arrays)
        metadata = dict(self.metadata)
        metadata.update({
            "num_frames": len(self.frames),
            "joint_count": int(arrays["joints_xyz"].shape[1]),
            "smpl_joint_count": int(arrays["smpl_joints_root_relative"].shape[1]),
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        })
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8")
        return path


def load_human_episode(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a RealSense RGB-D + HMR2 human episode."""
    arrays = load_episode(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in SMPL_ARRAYS if name not in data]
        if missing:
            raise ValueError(f"human episode is missing arrays: {', '.join(missing)}")
        arrays.update({name: data[name].copy() for name in SMPL_ARRAYS})
    length = len(arrays["timestamps"])
    if arrays["smpl_global_orient"].shape != (length, 3, 3):
        raise ValueError("smpl_global_orient must have shape (T, 3, 3)")
    if arrays["smpl_body_pose"].ndim != 4 or arrays["smpl_body_pose"].shape[:1] != (length,) \
            or arrays["smpl_body_pose"].shape[-2:] != (3, 3):
        raise ValueError("smpl_body_pose must have shape (T, J, 3, 3)")
    if arrays["smpl_betas"].shape != (length, 10):
        raise ValueError("smpl_betas must have shape (T, 10)")
    if arrays["smpl_cam_t"].shape != (length, 3):
        raise ValueError("smpl_cam_t must have shape (T, 3)")
    if arrays["smpl_joints_root_relative"].ndim != 3 \
            or arrays["smpl_joints_root_relative"].shape[0] != length \
            or arrays["smpl_joints_root_relative"].shape[2] != 3:
        raise ValueError("smpl_joints_root_relative must have shape (T, J, 3)")
    if arrays["smpl_valid"].shape != (length,):
        raise ValueError("smpl_valid must have shape (T,)")
    return arrays
