"""Standalone W&B handler — runs alongside loguru, not instead of it."""

from __future__ import annotations

import os
import random
import string
from pathlib import Path
from typing import Any

from loguru import logger

import wandb
from trainer.src.config import get_config


def _generate_wandb_run_name(prefix: str) -> str:
    """Generate a unique W&B run name with a 4-char random suffix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}_{suffix}"


class WandbRegistry:
    """Registry for multiple WandbHandler instances."""

    _handlers: dict[str, WandbHandler] = {}
    _logged_in: bool = False

    @classmethod
    def _login(cls) -> None:
        """Login to W&B using API key from environment variable. Called once globally."""
        if cls._logged_in:
            return
        cfg = get_config().wandb
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key and cfg.mode == "online":
            raise ValueError("W&B API key not found in environment variable 'WANDB_API_KEY'")

        wandb.login(key=api_key)
        cls._logged_in = True

    @classmethod
    def init(
        cls,
        key: str = "default",
        run_name: str | None = None,
        cfg_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register (or replace) a named WandbHandler in the registry."""
        cls._login()
        handler = WandbHandler()
        handler.init_run(run_name or _generate_wandb_run_name(key), cfg_dict, tags)
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


class WandbHandler:
    """W&B metrics handler."""

    def __init__(self) -> None:
        self._cfg = get_config().wandb
        self._run = None
        self._run_id: str | None = None

    @property
    def id(self) -> str | None:
        return self._run_id

    def init_run(
        self,
        run_name: str,
        cfg_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._run = wandb.init(
            project=self._cfg.project,
            entity=self._cfg.entity,
            name=run_name,
            config=cfg_dict,
            tags=tags,
            mode=self._cfg.mode,
        )
        self._run_id = self._run.id if self._run is not None else None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del step
        wandb.log(metrics)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            for key, value in metrics.items():
                self._run.summary[key] = value

    def log_confusion_matrix(
        self,
        cm: list[list[float]],
        labels: list[str],
        title: str = "Confusion Matrix",
    ) -> None:
        if self._run is None:
            return
        import numpy as np
        import plotly.express as px
        import plotly.figure_factory as ff

        cm_arr = np.array(cm) * 100
        fig = ff.create_annotated_heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            annotation_text=[[f"{v:.1f}" for v in row] for row in cm_arr],
            colorscale="Blues",
            showscale=True,
            zmin=0,
            zmax=100,
        )
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="True",
            font=dict(size=11),
        )
        fig.update_xaxes(side="bottom", tickangle=45)
        self._run.log({f"eval/{title}": fig})

    def log_artifact(
        self,
        artifact_path: str | Path,
        name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ):
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
        if self._run is not None:
            self._run.finish()
            logger.info("[Wandb] Run finished.")
