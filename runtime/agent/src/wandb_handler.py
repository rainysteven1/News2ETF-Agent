"""Lightweight W&B handler for src runtime metrics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import get_config
from src.logger import logger


class WandbHandler:
    """W&B metrics/artifact handler."""

    def __init__(self) -> None:
        self._cfg = get_config().wandb
        self._run = None
        self._run_id: str | None = None
        self._tables: dict[str, Any] = {}

    @property
    def id(self) -> str | None:
        return self._run_id

    def init_run(
        self,
        run_name: str,
        cfg_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        try:
            import wandb
        except ModuleNotFoundError:
            logger.warning("[Wandb] wandb package not installed, run logging disabled")
            return

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
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def log_table_row(self, name: str, row: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            import wandb
        except ModuleNotFoundError:
            return

        table = self._tables.get(name)
        columns = list(row.keys())
        if table is None:
            table = wandb.Table(columns=columns)
            self._tables[name] = table
        elif list(table.columns) != columns:
            logger.warning("[Wandb] Table '{}' column mismatch, row skipped", name)
            return

        table.add_data(*[row.get(col) for col in columns])
        self._run.log({name: table}, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in metrics.items():
            self._run.summary[key] = value

    def log_artifact(
        self,
        artifact_path: str | Path,
        *,
        name: str,
        artifact_type: str = "dataset",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        if self._run is None:
            return

        import wandb

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning("[Wandb] Artifact path does not exist: {}", artifact_path)
            return

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path), name=artifact_path.name)
        self._run.log_artifact(artifact, aliases=aliases or [])

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            logger.info("[Wandb] Run finished.")


class WandbRegistry:
    """Global W&B handler registry."""

    _handlers: dict[str, WandbHandler] = {}
    _logged_in: bool = False

    @classmethod
    def _login(cls) -> None:
        if cls._logged_in:
            return
        cfg = get_config().wandb
        if cfg.mode == "disabled":
            cls._logged_in = True
            return

        try:
            import wandb
        except ModuleNotFoundError:
            logger.warning("[Wandb] wandb package not installed, registry disabled")
            cls._logged_in = True
            return

        api_key = os.getenv("WANDB_API_KEY")
        if cfg.mode == "online" and not api_key:
            raise ValueError("WANDB_API_KEY is required when wandb.mode=online")
        wandb.login(key=api_key)
        cls._logged_in = True

    @classmethod
    def init(
        cls,
        key: str = "default",
        *,
        run_name: str,
        cfg_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        cls._login()
        handler = WandbHandler()
        handler.init_run(run_name=run_name, cfg_dict=cfg_dict, tags=tags)
        cls._handlers[key] = handler

    @classmethod
    def get(cls, key: str = "default") -> WandbHandler | None:
        return cls._handlers.get(key)

    @classmethod
    def finish_all(cls) -> None:
        for handler in cls._handlers.values():
            handler.finish()
        cls._handlers.clear()
