"""Standalone W&B handler — runs alongside loguru, not instead of it.

Loguru handles console output (logger.info/success).
WandbHandler pushes metrics to wandb dashboard.
Both run simultaneously and independently.

Registry-based multi-instance pattern:

  # FinBERT — single instance
  WandbRegistry.init("finbert")
  wb = WandbRegistry.get("finbert")

  # SetFit — one per major, registered upfront
  WandbRegistry.init("setfit_major1", tags=["setfit", "major1"])
  WandbRegistry.init("setfit_major2", tags=["setfit", "major2"])
  wb1 = WandbRegistry.get("setfit_major1")
  wb2 = WandbRegistry.get("setfit_major2")

  # No-args convenience: uses key="default"
  WandbRegistry.init()
  wb = WandbRegistry.get()
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path
from typing import Any

from loguru import logger

import wandb
from trainer.config import get_config


def _generate_wandb_run_name(prefix: str) -> str:
    """Generate a unique W&B run name with a 4-char random suffix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}_{suffix}"


# ── Registry ──────────────────────────────────────────────────────────────────


class WandbRegistry:
    """Registry for multiple WandbHandler instances."""

    _handlers: dict[str, WandbHandler] = {}

    @classmethod
    def init(cls, key: str = "default", run_name: str | None = None, tags: list[str] | None = None) -> None:
        """Register (or replace) a named WandbHandler in the registry.

        Args:
            key: Unique identifier for this handler (e.g. "finbert", "setfit_technology").
                 Subsequent calls with the same key replace the existing handler.
            tags: Override config.toml tags for this handler only.
        """
        handler = WandbHandler(tags=tags)
        cls._handlers[key] = handler

    @classmethod
    def get(cls, key: str = "default") -> WandbHandler:
        """Return a named handler from the registry."""
        assert key in cls._handlers, f"WandbHandler '{key}' not found. Call WandbRegistry.init('{key}', ...) first."
        return cls._handlers[key]

    @classmethod
    def finish_all(cls) -> None:
        """Finish all registered handlers."""
        for handler in cls._handlers.values():
            handler.finish()


# ── Handler ───────────────────────────────────────────────────────────────────


class WandbHandler:
    """W&B metrics handler. Settings come from config.toml, optionally overridden per-instance."""

    def __init__(
        self,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._cfg = get_config().wandb
        self._run = None
        self._run_id: str | None = None

        # Override config with explicitly passed values
        self._tags = tags if tags is not None else self._cfg.tags

        self._login()
        self._init_run(run_name or _generate_wandb_run_name("trainer"))

    @property
    def id(self) -> str | None:
        """W&B run ID."""
        return self._run_id

    def _login(self):
        """Login to W&B using API key from environment variable."""
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key and self._cfg.mode == "online":
            raise ValueError("W&B API key not found in environment variable 'WANDB_API_KEY'")
        wandb.login(key=api_key)

    def _init_run(self, run_name: str):
        self._run = wandb.init(
            project=self._cfg.project,
            entity=self._cfg.entity,
            name=run_name,
            tags=self._tags,
            mode=self._cfg.mode,
        )
        self._run_id = self._run.id if self._run is not None else None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb dashboard."""
        wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics to W&B run summary."""
        if self._run is not None:
            for key, value in metrics.items():
                self._run.summary[key] = value

    def log_artifact(
        self,
        artifact_path: str | Path,
        name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ):
        """Upload a file as a W&B artifact."""
        if not self._run:
            logger.info(f"[Wandb] Artifact upload skipped (disabled): {name}")
            return

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"[Wandb] Artifact path does not exist: {artifact_path}")
            return

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path), name=artifact_path.name)

        self._run.log_artifact(artifact, aliases=aliases or [])
        logger.info(f"[Wandb] Artifact uploaded: {name} ({artifact_type})")

    def finish(self) -> None:
        """Finish the wandb run."""
        if self._run is not None:
            self._run.finish()
            logger.info("[Wandb] Run finished.")
