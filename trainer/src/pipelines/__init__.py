"""Pipelines package — training and inference pipelines."""
from trainer.src.pipelines.predict import run as predict_run
from trainer.src.pipelines.train_major import train_major
from trainer.src.pipelines.train_sub_setfit import SetFitMultiMajorTrainer as setfit_trainer

__all__ = [
    "train_major",
    "setfit_trainer",
    "predict_run",
]
