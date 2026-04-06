"""Root trainer configuration — aggregates all model configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from trainer.src.config.major import MajorConfig
from trainer.src.config.sub import SubConfig


class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "news2etf"
    entity: str | None = None
    tags: list[str] = []


class PredictionConfig(BaseModel):
    finbert_onnx_dir: Path | None = None
    finbert_output_path: Path | None = None
    finbert_workers: int = 8
    setfit_base_dir: Path | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    batch_size: int = 64
    finbert_max_length: int = 128
    setfit_max_length: int = 256


class RootConfig(BaseModel):
    app: Literal["major", "sub", "signals"] = "major"
    seed: int = 42
    wandb: WandbConfig = WandbConfig()
    major: MajorConfig = MajorConfig()
    sub: SubConfig = SubConfig()
    prediction: PredictionConfig = PredictionConfig()

    def to_wandb(self):
        config: dict[str, Any] = {"seed": self.seed}
        if self.app == "major":
            config["major"] = self.major.to_wandb()
        return config


_config_instance: RootConfig | None = None


def init_config(app: Literal["major", "sub", "signals"], config_path: str | None = None) -> None:
    """Initialize the singleton config from a TOML config file."""
    global _config_instance
    if config_path is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        cfg_path = root_dir / "config.toml"
    else:
        cfg_path = Path(config_path)

    import tomllib

    with open(cfg_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    _config_instance = RootConfig.model_validate(raw)
    _config_instance.app = app

    if app == "major":
        _config_instance.wandb.tags = ["major"]
    elif app == "signals":
        _config_instance.wandb.tags = ["signals"]
    elif app == "sub":
        _config_instance.wandb.tags = ["sub"]


def get_config() -> RootConfig:
    """Return the singleton config. Must call init_config() first."""
    assert _config_instance is not None, "Config not initialized. Call init_config() first."
    return _config_instance


def load_config(path: str | Path | None = None) -> RootConfig:
    """Load trainer/config.toml and resolve relative paths."""
    import tomllib

    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config.toml"

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    filtered: dict[str, Any] = {}

    if "wandb" in raw:
        filtered["wandb"] = raw["wandb"]

    if "predict" in raw:
        _ROOT = Path(__file__).resolve().parent.parent.parent
        predict_section = raw["predict"]
        resolved: dict[str, Any] = {}
        for key, val in predict_section.items():
            if key.endswith("_path") or key.endswith("_dir") or key.endswith("_file"):
                resolved[key] = _ROOT / val
            else:
                resolved[key] = val
        filtered["prediction"] = resolved

    return RootConfig.model_validate(filtered)
