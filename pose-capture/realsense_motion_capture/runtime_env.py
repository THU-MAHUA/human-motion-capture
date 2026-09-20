"""Runtime isolation helpers for the motion-capture entry points."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _resolved(path: str) -> Path | None:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return None


def prepare_opencv_import() -> None:
    """Prevent user-installed OpenCV/Qt files from shadowing the conda env."""
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    blocked = {_resolved(path) for path in user_sites}
    blocked.discard(None)

    sys.path[:] = [
        entry
        for entry in sys.path
        if _resolved(entry) not in blocked
    ]

    # OpenCV sets its own environment-local Qt plugin path during import.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)


def require_environment_package(module, package_name: str) -> None:
    """Raise a useful error if a package still came from outside sys.prefix."""
    module_path = _resolved(getattr(module, "__file__", ""))
    environment = _resolved(sys.prefix)
    if module_path is None or environment is None:
        return
    try:
        module_path.relative_to(environment)
    except ValueError as exc:
        raise RuntimeError(
            f"{package_name} loaded from {module_path}, outside the active "
            f"environment {environment}. Run with PYTHONNOUSERSITE=1 or use "
            "scripts/vision_motion_capture from the repository root."
        ) from exc
