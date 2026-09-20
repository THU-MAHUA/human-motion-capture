"""Download supported MMPose/HMR2 assets and install a licensed SMPL model."""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = (
    ("rtmo-l_16xb16-600e_coco-640x640", 17),
    ("rtmpose-s_8xb256-420e_coco-256x192", 17),
    ("rtmw-x_8xb320-270e_cocktail14-384x288", 133),
)


def download_pose_models(device: str) -> None:
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise RuntimeError(
            "MMPose is not installed. Run `pip install -e '.[pose]'` in the "
            "pose environment first."
        ) from exc
    for alias, joint_count in MODELS:
        print(f"Loading {alias} ({joint_count} joints)...", flush=True)
        MMPoseInferencer(
            pose2d=alias,
            device=device,
            det_model="whole_image",
        )
        print(f"Ready: {alias}", flush=True)


def download_hmr2_models() -> None:
    try:
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import DEFAULT_CHECKPOINT, download_models
    except ImportError as exc:
        raise RuntimeError(
            "HMR2 is not installed. Run `pip install -e '.[mesh]'` in the "
            "mesh environment first."
        ) from exc
    download_models(CACHE_DIR_4DHUMANS)
    checkpoint = Path(DEFAULT_CHECKPOINT).expanduser()
    if not checkpoint.is_file():
        raise RuntimeError(
            f"HMR2 download completed without the expected checkpoint: {checkpoint}"
        )
    print(f"Ready: HMR2 checkpoint at {checkpoint}", flush=True)


def install_smpl_model(source: str) -> None:
    try:
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import convert_pkl
    except ImportError as exc:
        raise RuntimeError(
            "HMR2 is not installed. Install the mesh extra before installing "
            "the licensed SMPL model."
        ) from exc
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"SMPL model not found: {source_path}")
    destination = (
        Path(CACHE_DIR_4DHUMANS)
        / "data"
        / "smpl"
        / "SMPL_NEUTRAL.pkl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    convert_pkl(str(source_path), str(destination))
    print(f"Installed licensed SMPL model at {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", action="store_true",
                        help="download all supported MMPose checkpoints")
    parser.add_argument("--hmr2", action="store_true",
                        help="download the HMR2 checkpoint and support data")
    parser.add_argument("--all", action="store_true",
                        help="download both pose and HMR2 model assets")
    parser.add_argument("--device", default="cuda:0",
                        help="device used to validate MMPose models")
    parser.add_argument(
        "--install-smpl",
        metavar="PATH",
        help="convert and install a separately licensed neutral SMPL model",
    )
    args = parser.parse_args()

    pose_requested = args.pose or args.all
    hmr2_requested = args.hmr2 or args.all
    if not pose_requested and not hmr2_requested and not args.install_smpl:
        pose_requested = True
    if args.install_smpl:
        install_smpl_model(args.install_smpl)
    if pose_requested:
        download_pose_models(args.device)
    if hmr2_requested:
        download_hmr2_models()


if __name__ == "__main__":
    main()
