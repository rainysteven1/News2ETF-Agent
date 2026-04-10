"""Runtime shim for ``src.utils``."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_REPO_UTILS = Path(__file__).resolve().parents[4] / "src" / "utils"
if _REPO_UTILS.exists():
    repo_utils_str = str(_REPO_UTILS)
    if repo_utils_str not in __path__:
        __path__.append(repo_utils_str)
