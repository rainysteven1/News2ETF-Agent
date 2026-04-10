"""Runtime shim package for the standalone agent app.

This package keeps ``runtime/agent`` importable as a self-contained app while
still delegating unresolved submodules to the repository-level ``src`` package
until the full migration is complete.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
repo_root_str = str(_REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

if _REPO_SRC.exists():
    repo_src_str = str(_REPO_SRC)
    if repo_src_str not in __path__:
        __path__.append(repo_src_str)
