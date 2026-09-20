"""Download and load the supported MMPose checkpoints."""

from __future__ import annotations

import argparse

from mmpose.apis import MMPoseInferencer

MODELS = (
    ("rtmo-l_16xb16-600e_coco-640x640", 17),
    ("rtmpose-s_8xb256-420e_coco-256x192", 17),
    ("rtmw-x_8xb320-270e_cocktail14-384x288", 133),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    for alias, joint_count in MODELS:
        print(f"Loading {alias} ({joint_count} joints)...", flush=True)
        MMPoseInferencer(
            pose2d=alias,
            device=args.device,
            det_model="whole_image",
        )
        print(f"Ready: {alias}", flush=True)


if __name__ == "__main__":
    main()
