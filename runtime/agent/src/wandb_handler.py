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
        self._run_name: str | None = None
        self._tags: list[str] = []

    @property
    def id(self) -> str | None:
        return self._run_id

    def metadata(self) -> dict[str, Any]:
        run_name = self._run.name if self._run is not None else self._run_name
        return {
            "wandb_run_id": self._run_id,
            "wandb_run_name": run_name,
            "project": self._cfg.project,
            "entity": self._cfg.entity,
            "mode": self._cfg.mode,
            "tags": list(self._tags),
        }

    def init_run(
        self,
        run_name: str,
        cfg_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        existing_run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        try:
            import wandb
        except ModuleNotFoundError:
            logger.warning("[Wandb] wandb package not installed, run logging disabled")
            return

        self._run_name = run_name
        self._tags = list(tags or [])

        init_kwargs: dict[str, Any] = {
            "project": self._cfg.project,
            "entity": self._cfg.entity,
            "name": run_name,
            "config": cfg_dict,
            "tags": self._tags,
            "mode": self._cfg.mode,
        }
        if existing_run_id:
            init_kwargs["id"] = existing_run_id
            init_kwargs["resume"] = resume or "must"

        self._run = wandb.init(**init_kwargs)
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
        del step

    def log_table_rows(self, name: str, rows: list[dict[str, Any]], step: int | None = None) -> None:
        if self._run is None or not rows:
            return

        self._tables.pop(name, None)
        for row in rows:
            self.log_table_row(name, row)

        table = self._tables.get(name)
        if table is not None:
            self._run.log({name: table}, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in metrics.items():
            self._run.summary[key] = value

    def add_tags(self, tags: list[str]) -> None:
        run_tags = list(getattr(self._run, "tags", []) or []) if self._run is not None else []
        merged = list(dict.fromkeys([*self._tags, *run_tags, *[str(tag) for tag in tags if tag]]))
        self._tags = merged
        if self._run is None:
            return
        try:
            self._run.tags = tuple(merged)
        except Exception as exc:
            logger.warning("[Wandb] Failed to update run tags: {}", exc)

    def log_images(
        self,
        images: dict[str, str | Path],
        *,
        captions: dict[str, str] | None = None,
        gallery_key: str | None = None,
        step: int | None = None,
    ) -> None:
        if self._run is None or not images:
            return

        import wandb

        payload: dict[str, Any] = {}
        gallery: list[Any] = []
        for key, image_path in images.items():
            path = Path(image_path)
            if not path.exists():
                logger.warning("[Wandb] Image path does not exist: {}", path)
                continue
            image = wandb.Image(str(path), caption=(captions or {}).get(key, path.stem))
            payload[key] = image
            gallery.append(image)

        if gallery_key and gallery:
            payload[gallery_key] = gallery

        if payload:
            self._run.log(payload, step=step)

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
        existing_run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        cls._login()
        handler = WandbHandler()
        handler.init_run(
            run_name=run_name,
            cfg_dict=cfg_dict,
            tags=tags,
            existing_run_id=existing_run_id,
            resume=resume,
        )
        cls._handlers[key] = handler

    @classmethod
    def get(cls, key: str = "default") -> WandbHandler | None:
        return cls._handlers.get(key)

    @classmethod
    def finish_all(cls) -> None:
        for handler in cls._handlers.values():
            handler.finish()
        cls._handlers.clear()
