"""Pipelines package — expose entry points without eager optional imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "train_major",
    "train_signals",
    "setfit_trainer",
    "predict_run",
]


def __getattr__(name: str) -> Any:
    if name == "predict_run":
        from trainer.src.pipelines.predict import run

        return run
    if name == "train_major":
        from trainer.src.pipelines.train_major import train_major

        return train_major
    if name == "train_signals":
        from trainer.src.pipelines.train_signals import run_training

        return run_training
    if name == "setfit_trainer":
        from trainer.src.pipelines.train_sub_setfit import SetFitMultiMajorTrainer

        return SetFitMultiMajorTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
