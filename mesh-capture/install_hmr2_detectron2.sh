#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the mesh-capture environment before running this script." >&2
  exit 1
fi

site_packages="${CONDA_PREFIX}/lib/python3.10/site-packages"
cuda_runtime="${site_packages}/nvidia/cuda_runtime"
cuda_paths=(
  "${CONDA_PREFIX}/targets/x86_64-linux/include"
  "${cuda_runtime}/include"
  "${site_packages}/nvidia/cublas/include"
  "${site_packages}/nvidia/cusparse/include"
  "${site_packages}/nvidia/cusolver/include"
)
library_paths=(
  "${CONDA_PREFIX}/lib"
  "${cuda_runtime}/lib"
  "${site_packages}/nvidia/cublas/lib"
  "${site_packages}/nvidia/cusparse/lib"
  "${site_packages}/nvidia/cusolver/lib"
)

link_dir="${TMPDIR:-/tmp}/4d-humans-cuda-link"
mkdir -p "${link_dir}"
ln -sf "${cuda_runtime}/lib/libcudart.so.11.0" "${link_dir}/libcudart.so"

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CPATH="$(IFS=:; echo "${cuda_paths[*]}")"
export LIBRARY_PATH="${link_dir}:$(IFS=:; echo "${library_paths[*]}")"
export LD_LIBRARY_PATH="${link_dir}:$(IFS=:; echo "${library_paths[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python -m pip install --no-build-isolation --no-cache-dir \
  'detectron2 @ https://codeload.github.com/facebookresearch/detectron2/tar.gz/refs/heads/main'
