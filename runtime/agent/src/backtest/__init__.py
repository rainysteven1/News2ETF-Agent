"""Runtime shim for ``src.backtest``."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_REPO_BACKTEST = Path(__file__).resolve().parents[4] / "src" / "backtest"
if _REPO_BACKTEST.exists():
    repo_backtest_str = str(_REPO_BACKTEST)
    if repo_backtest_str not in __path__:
        __path__.append(repo_backtest_str)
