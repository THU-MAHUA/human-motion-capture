# RealSense Human Motion Capture

Standalone Intel RealSense D435 motion capture using MMPose and optional
4D-Humans HMR2 mesh inference.

The repository uses two deliberately separate environments:

- `pose-capture`: RealSense, MMPose, and the main capture process
- `mesh-capture`: HMR2, SMPL rendering, and the optional worker process

Model weights are downloaded during setup and are not stored in Git.
The licensed neutral SMPL model must also be downloaded separately.

The published default targets D435 serial `313522072015`. On this machine,
the currently connected camera is a D435i with serial `239122072192`; use the
`--serial 239122072192` override shown below only when testing that camera.

## Local Test Setup

The existing working environments can be cloned with the new names:

```bash
conda create --name pose-capture --clone mmpose-realsense
conda create --name mesh-capture --clone 4D-humans
```

The commands below use the new environment names and this repository folder.
The old environments are intentionally left available until testing is
complete.

Set the HMR2 checkout for mesh capture:

```bash
export MESH_CAPTURE_ROOT=/path/to/4D-Humans
```

Check the pose environment:

```bash
./scripts/check_environment.sh
```

Run the focused tests:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_PLUGIN_PATH \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD/pose-capture" \
  "$HOME/anaconda3/envs/pose-capture/bin/python" \
  -m pytest -q tests/test_realsense_motion_capture.py
```

## Live Skeleton

```bash
./scripts/vision_motion_capture \
  --output-dir "recordings/rtmo-d435-$(date +%Y%m%d-%H%M%S)"
```

Use another supported MMPose model:

```bash
./scripts/vision_motion_capture \
  --output-dir "recordings/rtmpose-s-d435-$(date +%Y%m%d-%H%M%S)" \
  --pose2d rtmpose-s_8xb256-420e_coco-256x192
```

Run a short headless smoke test without writing a bag:

```bash
./scripts/vision_motion_capture \
  --output-dir "recordings/headless-test-$(date +%Y%m%d-%H%M%S)" \
  --headless --no-bag --max-frames 90
```

Test the currently connected D435i explicitly:

```bash
./scripts/vision_motion_capture \
  --serial 239122072192 \
  --output-dir "recordings/d435i-test-$(date +%Y%m%d-%H%M%S)" \
  --headless --no-bag --max-frames 90
```

## Live Skeleton And White Mesh

The split window shows the RTMO-L skeleton on the left and the live white HMR2
mesh on the right:

```bash
MESH_CAPTURE_ROOT="${MESH_CAPTURE_ROOT}" \
./scripts/live_mesh_motion_capture \
  --output-dir "recordings/live-mesh-$(date +%Y%m%d-%H%M%S)"
```

For a smaller window:

```bash
./scripts/live_mesh_motion_capture \
  --output-dir "recordings/live-mesh-$(date +%Y%m%d-%H%M%S)" \
  --preview-panel-width 640
```

The session saves `rgbd.bag`, `episode.npz`, `metadata.json`, and
`runtime.json`. The split preview itself is not saved as a video.

## Model Prefetch

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_PLUGIN_PATH \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD/pose-capture" \
  "$HOME/anaconda3/envs/pose-capture/bin/python" \
  pose-capture/realsense_motion_capture/prefetch_models.py \
  --device cuda:0
```

This downloads and validates RTMO-L, RTMPose-S, and RTMW-X. HMR2 downloads
its checkpoint into its normal cache when the mesh worker is opened.

## HMR2 Installation

The optional mesh environment can be recreated from:

```bash
conda env create -f environments/mesh-capture.yml
conda activate mesh-capture
./mesh-capture/install_hmr2_detectron2.sh
```

The neutral SMPL file is licensed and must be placed in the HMR2 checkout at:

```text
<MESH_CAPTURE_ROOT>/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

For a clean installation instead of cloning the already-tested local
environments:

```bash
conda env create -f environments/pose-capture.yml
conda env create -f environments/mesh-capture.yml
conda activate pose-capture
python -m pip install -e .
```

## RealSense And Replay

```bash
realsense-viewer
rs-enumerate-devices
```

Replay a RealSense bag:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_PLUGIN_PATH \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD/pose-capture" \
  "$HOME/anaconda3/envs/pose-capture/bin/python" \
  pose-capture/realsense_motion_capture/replay.py \
  recordings/rtmo-d435/rgbd.bag \
  --output-dir "recordings/replay-$(date +%Y%m%d-%H%M%S)" \
  --pose2d rtmo-l_16xb16-600e_coco-640x640 \
  --device cuda:0
```

Only RealSense bags are supported. Existing Orbbec recordings are not migrated.
