"""Runtime context for the active CLI invocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuntimeContext:
    run_id: str | None = None
    checkpoint_dir: Path | None = None


_runtime: RuntimeContext | None = None


def init_runtime(*, run_id: str | None = None, checkpoint_dir: Path | None = None) -> RuntimeContext:
    global _runtime
    _runtime = RuntimeContext(run_id=run_id, checkpoint_dir=checkpoint_dir)
    return _runtime


def get_runtime() -> RuntimeContext:
    global _runtime
    if _runtime is None:
        _runtime = RuntimeContext()
    return _runtime
