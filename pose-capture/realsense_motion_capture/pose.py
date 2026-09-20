"""MMPose inference adapter for one-person live capture."""

from __future__ import annotations

from typing import Any

import numpy as np


class OnePersonPose:
    def __init__(self, pose2d: str = "rtmw-x_8xb320-270e_cocktail14-384x288", device: str = "cuda:0",
                 keypoint_threshold: float = 0.25):
        self.pose2d, self.device = pose2d, device
        self.keypoint_threshold = float(keypoint_threshold)
        self.joint_count = 133 if any(token in pose2d.lower() for token in ("rtmw", "wholebody", "133")) else 17
        self._inferencer: Any = None

    def open(self) -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "MMPose dependencies are missing; install the project runtime "
                f"dependencies (original error: {exc})") from exc
        # There is exactly one performer in this application.  Whole-image
        # mode avoids a second detector process and keeps the live path light.
        self._inferencer = MMPoseInferencer(
            pose2d=self.pose2d, device=self.device, det_model="whole_image")

    def predict(self, color_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        if self._inferencer is None:
            raise RuntimeError("pose estimator is not open")
        result = next(self._inferencer(color_bgr, return_vis=False, show=False))
        predictions = result.get("predictions", [])
        while predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        if not predictions:
            return np.full((self.joint_count, 2), np.nan, np.float32), np.zeros(self.joint_count, np.float32), 0
        # One-person policy: select the highest-confidence detected person.
        def person_score(item):
            score = item.get("bbox_score", item.get("bbox_scores", 0.0))
            score = np.asarray(score).reshape(-1)
            return float(score[0]) if score.size else 0.0

        person = max(predictions, key=person_score)
        keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
        scores = np.asarray(person.get("keypoint_scores", person.get("keypoint_scores", [])), dtype=np.float32)
        if keypoints.shape[0] < self.joint_count:
            return np.full((self.joint_count, 2), np.nan, np.float32), np.zeros(self.joint_count, np.float32), 0
        keypoints, scores = keypoints[:self.joint_count, :2], scores[:self.joint_count]
        valid = np.isfinite(keypoints).all(axis=1) & (scores >= self.keypoint_threshold)
        return keypoints, scores, int(valid.sum() >= 5)
