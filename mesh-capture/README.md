# mesh-capture

This directory contains the Detectron2 build helper for the optional
4D-Humans HMR2 worker.

The HMR2 source checkout and licensed SMPL model are external assets. Set
`MESH_CAPTURE_ROOT` to the 4D-Humans checkout before using the live mesh
launcher:

```bash
export MESH_CAPTURE_ROOT=/path/to/4D-Humans
./mesh-capture/install_hmr2_detectron2.sh
./scripts/live_mesh_motion_capture
```
