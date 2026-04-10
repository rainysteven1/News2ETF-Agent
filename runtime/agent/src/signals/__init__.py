"""Runtime shim for ``src.signals``."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_REPO_SIGNALS = Path(__file__).resolve().parents[4] / "src" / "signals"
if _REPO_SIGNALS.exists():
    repo_signals_str = str(_REPO_SIGNALS)
    if repo_signals_str not in __path__:
        __path__.append(repo_signals_str)
