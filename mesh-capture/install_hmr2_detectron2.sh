#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the mesh-capture environment before running this script." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
  -e "${repo_root}[mesh]"

installed_revision() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 python - "$1" <<'PY'
import json
import sys
from importlib.metadata import distribution
from pathlib import Path

try:
    metadata = distribution(sys.argv[1])
    direct_url = Path(metadata._path) / "direct_url.json"
    payload = json.loads(direct_url.read_text())
    print(payload.get("vcs_info", {}).get("commit_id", ""))
except Exception:
    print("")
PY
}

ensure_revision() {
  local package="$1"
  local revision="$2"
  local requirement="$3"

  if [[ "$(installed_revision "${package}")" == "${revision}" ]]; then
    return
  fi

  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    python -m pip install --force-reinstall --no-deps \
    --no-build-isolation --no-cache-dir "${requirement}"
}

ensure_revision \
  chumpy \
  580566eafc9ac68b2614b64d6f7aaa84eebb70da \
  "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
ensure_revision \
  hmr2 \
  efe18deff163b29dff87ddbd575fa29b716a356c \
  "hmr2 @ git+https://github.com/shubham-goel/4D-Humans.git@efe18deff163b29dff87ddbd575fa29b716a356c"
ensure_revision \
  detectron2 \
  a2f4a8771ab77e8411c26b27f24f9489a28a2453 \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
