# RealSense Human Motion Capture

Live Intel RealSense D435/D435i human motion capture with:

- MMPose RTMO-L, RTMPose-S, or RTMW-X skeleton inference
- aligned metric depth and 3D joint reconstruction
- optional 4D-Humans HMR2 white SMPL mesh inference
- split live skeleton and mesh preview
- RealSense bag recording and replay

This repository does not require separate MMPose or 4D-Humans source
checkouts. They are installed as pinned Python dependencies. Model checkpoints,
camera recordings, and the separately licensed neutral SMPL model are not
stored in Git.

## Live Demo

The split preview shows the RTMO-L skeleton on the left and the live HMR2
white mesh on the right.

https://github.com/user-attachments/assets/d9fd9525-1ac0-404e-a862-185a4528a498

## Requirements

- Ubuntu or another compatible x86-64 Linux distribution
- Python 3.10
- NVIDIA driver compatible with the selected PyTorch CUDA wheels
- Intel RealSense D435 or D435i
- Git and a C/C++ compiler for the optional Detectron2 mesh environment

The pose and mesh processes use separate environments because their validated
dependency stacks use different CUDA wheels and NumPy versions.

## Pose Environment

Clone the project and create the reproducible pose environment:

```bash
git clone https://github.com/THU-MAHUA/human-motion-capture.git
cd human-motion-capture

conda env create -f environments/pose-capture.yml
conda activate pose-capture

python -m pip install -e ".[pose,test]" \
  --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

The environment file installs PyTorch 2.1 with CUDA 12.1. To use another CUDA
wheel, install the matching PyTorch and MMCV builds before installing this
project.

Check the installation and run the focused tests:

```bash
./scripts/check_environment.sh
python -m pytest -q
```

Download and validate the supported MMPose checkpoints:

```bash
motion-capture-download-models --pose --device cuda:0
```

The supported aliases are:

```text
rtmo-l_16xb16-600e_coco-640x640
rtmpose-s_8xb256-420e_coco-256x192
rtmw-x_8xb320-270e_cocktail14-384x288
```

## Live Skeleton

The default launcher uses D435 serial `313522072015`, RTMO-L, CUDA device 0,
and immediate recording:

```bash
conda activate pose-capture
cd /path/to/human-motion-capture

./scripts/vision_motion_capture \
  --output-dir "recordings/rtmo-d435-$(date +%Y%m%d-%H%M%S)"
```

Select another camera or pose model with normal command-line overrides:

```bash
./scripts/vision_motion_capture \
  --serial 239122072192 \
  --pose2d rtmpose-s_8xb256-420e_coco-256x192 \
  --output-dir "recordings/rtmpose-s-$(date +%Y%m%d-%H%M%S)"
```

Press `r` to pause or resume processed recording. Press `q` or `Esc` to exit.
Use `--no-bag` to disable raw RGB-D recording and `--headless` to disable the
OpenCV preview.

## Mesh Environment

Create the separate HMR2 environment:

```bash
conda env create -f environments/mesh-capture.yml
conda activate mesh-capture

./mesh-capture/install_hmr2_detectron2.sh
```

The installer builds the pinned Detectron2 revision for the environment and
installs the pinned 4D-Humans package. No external source checkout is needed.

Download the HMR2 checkpoint and support data:

```bash
motion-capture-download-models --hmr2
```

HMR2 also requires the neutral SMPL model. Register with the SMPL provider,
download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, and install your
licensed copy into the standard 4D-Humans cache:

```bash
motion-capture-download-models \
  --install-smpl /path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

The resulting local file is:

```text
~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
```

It remains outside this repository and must not be committed.

## Live Skeleton And White Mesh

Activate the pose environment and point the launcher to the mesh environment's
Python executable:

```bash
conda activate pose-capture
cd /path/to/human-motion-capture

export MESH_CAPTURE_PYTHON="$HOME/anaconda3/envs/mesh-capture/bin/python"

./scripts/live_mesh_motion_capture \
  --output-dir "recordings/live-mesh-$(date +%Y%m%d-%H%M%S)" \
  --preview-panel-width 640
```

The split window shows the live RTMO-L skeleton on the left and the live white
HMR2 mesh on the right. The processed session saves `episode.npz`,
`metadata.json`, and `runtime.json`; raw RGB-D is saved as `rgbd.bag` unless
`--no-bag` is supplied.

Advanced users may still pass `--hmr2-root` to test a modified local
4D-Humans checkout, but it is not required for normal installation.

## Replay

Reprocess a RealSense bag:

```bash
conda activate pose-capture

motion-capture-replay recordings/session/rgbd.bag \
  --output-dir "recordings/replay-$(date +%Y%m%d-%H%M%S)" \
  --pose2d rtmo-l_16xb16-600e_coco-640x640 \
  --device cuda:0
```

Enable HMR2 replay with:

```bash
motion-capture-replay recordings/session/rgbd.bag \
  --output-dir "recordings/replay-mesh-$(date +%Y%m%d-%H%M%S)" \
  --pose2d rtmpose-s_8xb256-420e_coco-256x192 \
  --hmr2 \
  --hmr2-python "$HOME/anaconda3/envs/mesh-capture/bin/python"
```

Only RealSense bags are supported. Existing Orbbec recordings are not
migrated.

## Acknowledgments

This project builds on the following open-source research projects:

- [MMPose](https://github.com/open-mmlab/mmpose), developed by the MMPose
  Contributors and released under the Apache License 2.0, provides RTMO,
  RTMPose, RTMW, and the unified inference interface.
- [4D-Humans](https://github.com/shubham-goel/4D-Humans), developed by
  Shubham Goel and collaborators and released under the MIT License, provides
  the HMR2 human mesh reconstruction model.
- [Detectron2](https://github.com/facebookresearch/detectron2) provides the
  HMR2 person detector.
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
  provides D4xx camera capture, alignment, recording, and playback.

See [NOTICE.md](NOTICE.md) for dependency revisions and license notes.

If this project is used in research, please cite the upstream projects:

```bibtex
@misc{mmpose2020,
  title={OpenMMLab Pose Estimation Toolbox and Benchmark},
  author={MMPose Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmpose}},
  year={2020}
}

@inproceedings{goel2023humans,
  title={Humans in 4D: Reconstructing and Tracking Humans with Transformers},
  author={Goel, Shubham and Pavlakos, Georgios and Rajasegaran, Jathushan
          and Kanazawa, Angjoo and Malik, Jitendra},
  booktitle={ICCV},
  year={2023}
}
```

## License

The original integration code in this repository is released under the MIT
License. Third-party packages, checkpoints, datasets, and SMPL assets retain
their own licenses and terms.
