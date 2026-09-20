#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${POSE_CAPTURE_PYTHON:-}"

if [[ -z "${python}" && -n "${CONDA_PREFIX:-}" ]]; then
  python="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${python}" ]]; then
  python="$(command -v python3 || true)"
fi
if [[ -z "${python}" || ! -x "${python}" ]]; then
  echo "Activate pose-capture or set POSE_CAPTURE_PYTHON." >&2
  exit 1
fi

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_PLUGIN_PATH \
  PYTHONNOUSERSITE=1 \
  "${python}" - <<'PY'
import cv2
import pyrealsense2 as rs
import torch

print("OpenCV:", cv2.__file__)
print("RealSense:", rs.__file__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
