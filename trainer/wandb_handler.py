"""Standalone W&B handler — runs alongside loguru, not instead of it.

Loguru handles console output (logger.info/success).
WandbHandler pushes metrics to wandb dashboard.
Both run simultaneously and independently.

Usage:
    wb = WandbHandler(project="news2etf", name="run-001", tags=["signals"])
    wb.log({"loss": 0.5}, step=1)
    wb.log_epoch("pretrain", epoch=1, loss=0.3, extras={"reg_loss": 0.1})
    wb.finish()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

import wandb


def _build_lstm_config_dict(cfg: Any) -> dict[str, Any]:
    """Build wandb config dict from TrainerConfig (LSTM pipeline)."""
    return {
        "lstm_hidden_size": cfg.signals.hidden_size,
        "lstm_num_layers": cfg.signals.num_layers,
        "lstm_dropout": cfg.signals.dropout,
        "seq_len": cfg.signals.sequence_length,
        "epochs_pretrain": cfg.training.epochs_pretrain,
        "epochs_finetune": cfg.training.epochs_finetune,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "num_heads": cfg.training.num_heads,
        "anomaly_threshold": cfg.training.anomaly_threshold,
        "lgbm_num_leaves": cfg.lightgbm.num_leaves,
        "lgbm_lr": cfg.lightgbm.learning_rate,
        "lgbm_n_estimators": cfg.lightgbm.n_estimators,
    }


class WandbHandler:
    """Handles wandb metrics logging. Works alongside loguru (console output is separate)."""

    def __init__(
        self,
        project: str = "news2etf",
        name: str | None = None,
        config: Any = None,
        config_dict: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        entity: str | None = None,
    ):
        self.enabled = mode != "disabled" and (mode != "online" or bool(os.environ.get("WANDB_API_KEY")))
        self._run = None
        self._run_id: str | None = None
        self._tags = tags or []

        if config is not None:
            cfg_dict = _build_lstm_config_dict(config)
        else:
            cfg_dict = config_dict or {}

        if self.enabled:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=cfg_dict,
                tags=self._tags,
                mode=mode,  # type: ignore
            )
            self._run_id = self._run.id
            logger.info(f"[Wandb] Started run: {self._run.url} (tags={self._tags}, mode={mode})")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb dashboard."""
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_epoch(
        self,
        stage: str,
        epoch: int,
        loss: float,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Log per-epoch metrics with stage and epoch context.

        Logs to wandb with step=epoch so each epoch is a separate data point.
        """
        if not self.enabled:
            return
        d: dict[str, Any] = {
            "stage": stage,
            "epoch": epoch,
            "loss": loss,
        }
        if extras:
            d.update(extras)
        wandb.log(d, step=epoch)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Write final scalars to the run summary."""
        if not self.enabled or not self._run:
            return
        for key, value in metrics.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self._run is not None:
            self._run.finish()
            logger.info("[Wandb] Run finished.")

    @property
    def run_id(self) -> str | None:
        """W&B run ID."""
        return self._run_id

    def upload_artifact(
        self,
        artifact_path: str | Path,
        name: str,
        artifact_type: str = "model",
        aliases: list[str] | None = None,
    ) -> None:
        """Upload a local file or directory as a W&B artifact."""
        if not self.enabled:
            logger.info(f"[Wandb] Artifact upload skipped (disabled): {name}")
            return

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"[Wandb] Artifact path does not exist: {artifact_path}")
            return
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path), name=artifact_path.name)

        if self._run is not None:
            self._run.log_artifact(artifact, aliases=aliases or [])
            logger.info(f"[Wandb] Artifact uploaded: {name} ({artifact_type})")
