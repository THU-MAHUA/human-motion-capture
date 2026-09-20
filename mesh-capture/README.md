# Mesh Capture Environment

The optional mesh worker installs 4D-Humans/HMR2 and a CUDA-enabled
Detectron2 build directly from the revisions pinned in `pyproject.toml`.
A separate 4D-Humans checkout is not required.

```bash
conda env create -f environments/mesh-capture.yml
conda activate mesh-capture
./mesh-capture/install_hmr2_detectron2.sh
```

Download the HMR2 checkpoint:

```bash
motion-capture-download-models --hmr2
```

Install a separately downloaded, licensed neutral SMPL model:

```bash
motion-capture-download-models \
  --install-smpl /path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

The worker reads HMR2 assets from `~/.cache/4DHumans`. From the pose
environment, set `MESH_CAPTURE_PYTHON` to this environment's Python executable
before starting the split live preview.
