"""Minimal .env loader without external dependency."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import runtime_root


def load_project_env(start: Path | None = None) -> None:
    base = (start or runtime_root()).resolve()
    env_path: Path | None = None
    for candidate_base in (base, *base.parents):
        candidate = candidate_base / ".env"
        if candidate.exists():
            env_path = candidate
            break
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)
