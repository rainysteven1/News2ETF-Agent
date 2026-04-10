"""Runtime shim for ``src.agent``.

Local modules placed under ``runtime/agent/src/agent`` override the repository
version first; unresolved submodules still fall back to ``<repo>/src/agent``.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_REPO_AGENT = Path(__file__).resolve().parents[4] / "src" / "agent"
if _REPO_AGENT.exists():
    repo_agent_str = str(_REPO_AGENT)
    if repo_agent_str not in __path__:
        __path__.append(repo_agent_str)
