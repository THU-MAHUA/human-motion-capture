"""Binary stdio worker for running HMR2 in a dedicated environment."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import struct
import sys

import cv2
import numpy as np

PROTOCOL_OUTPUT = sys.stdout.buffer

try:
    from .hmr2_adapter import HMR2OnePerson
except ImportError:
    from hmr2_adapter import HMR2OnePerson


def read_exact(length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        chunk = sys.stdin.buffer.read(length - len(result))
        if not chunk:
            raise EOFError
        result.extend(chunk)
    return bytes(result)


def send(payload: bytes) -> None:
    PROTOCOL_OUTPUT.write(struct.pack("!Q", len(payload)))
    PROTOCOL_OUTPUT.write(payload)
    PROTOCOL_OUTPUT.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmr2-root",
        default=None,
        help="optional legacy 4D-Humans checkout override",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mesh-preview", action="store_true")
    args = parser.parse_args()
    # HMR2's ViTDetDataset prints crop diagnostics. Keep stdout exclusively
    # for the binary protocol and direct normal logging to stderr.
    sys.stdout = sys.stderr
    estimator = HMR2OnePerson(
        args.hmr2_root,
        args.checkpoint,
        args.device,
        mesh_preview=args.mesh_preview,
    )
    try:
        estimator.open()
        while True:
            try:
                size = struct.unpack("!Q", read_exact(8))[0]
                payload = read_exact(size)
            except EOFError:
                break
            if payload == b"HELLO":
                send(json.dumps({
                    "smpl_joint_count": estimator.smpl_joint_count,
                    "body_joint_count": estimator.body_joint_count,
                    "checkpoint": estimator.checkpoint_path,
                }).encode("utf-8"))
                continue
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            result = estimator.predict(image)
            mesh_preview_jpeg = np.empty(0, dtype=np.uint8)
            if result.mesh_preview_bgr is not None:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    result.mesh_preview_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                if ok:
                    mesh_preview_jpeg = encoded
            output = BytesIO()
            np.savez(output,
                     global_orient=result.global_orient,
                     body_pose=result.body_pose,
                     betas=result.betas,
                     cam_t=result.cam_t,
                     joints_root_relative=result.joints_root_relative,
                     valid=np.asarray(result.valid),
                     detection_score=np.asarray(result.detection_score),
                     mesh_preview_jpeg=mesh_preview_jpeg)
            send(output.getvalue())
    except Exception as exc:
        try:
            send(json.dumps({"error": str(exc)}).encode("utf-8"))
        except Exception:
            pass
        raise
    finally:
        estimator.close()


if __name__ == "__main__":
    main()
