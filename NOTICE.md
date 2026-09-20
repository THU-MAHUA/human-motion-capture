# Third-Party Notices

This repository contains original integration code and installs third-party
packages at setup time. It does not vendor the upstream source trees or model
weights.

## MMPose

- Project: OpenMMLab MMPose
- Repository: https://github.com/open-mmlab/mmpose
- Installed version: 1.3.2
- License: Apache License 2.0
- Copyright: OpenMMLab and MMPose contributors

MMPose provides the RTMO, RTMPose, and RTMW pose models and unified inference
API used by this project. Individual algorithms, model checkpoints, and
training datasets may have additional terms. Users are responsible for
reviewing the model documentation for their intended use.

## 4D-Humans

- Project: 4D-Humans / HMR2
- Repository: https://github.com/shubham-goel/4D-Humans
- Pinned revision: efe18deff163b29dff87ddbd575fa29b716a356c
- License: MIT License
- Copyright: 2023 UC Regents, Shubham Goel

4D-Humans provides the HMR2 model and renderer used for optional human mesh
reconstruction.

## Detectron2

- Project: Detectron2
- Repository: https://github.com/facebookresearch/detectron2
- Pinned revision: a2f4a8771ab77e8411c26b27f24f9489a28a2453
- License: Apache License 2.0
- Copyright: Facebook, Inc. and its affiliates

Detectron2 provides the person detector used by the HMR2 worker.

## Chumpy

- Project: Chumpy
- Repository: https://github.com/mattloper/chumpy
- Pinned revision: 580566eafc9ac68b2614b64d6f7aaa84eebb70da
- License: MIT License

Chumpy is an HMR2 compatibility dependency. The mesh installer pins it
explicitly because the upstream HMR2 package declares an unpinned Git URL.

## Intel RealSense SDK

- Project: Intel RealSense SDK 2.0 / librealsense
- Repository: https://github.com/IntelRealSense/librealsense
- Python package: pyrealsense2 2.57.7.10387
- License: Apache License 2.0

The SDK provides camera access, color/depth alignment, bag recording, and bag
playback.

## SMPL

SMPL is separately licensed and is not distributed by this repository. Users
must register with the SMPL provider, accept its terms, and install their own
neutral SMPL model. The model must not be committed to this repository.

## Model Weights And Data

MMPose and HMR2 checkpoints are downloaded on the user's machine and are not
part of this repository. Checkpoint and dataset terms may differ from the
source-code licenses listed above.
