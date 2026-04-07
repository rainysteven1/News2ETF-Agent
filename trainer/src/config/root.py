"""Root trainer configuration — aggregates all model configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from trainer.src.config.major import MajorConfig
from trainer.src.config.signals import SignalsConfig
from trainer.src.config.sub import SubConfig


class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "news2etf"
    entity: str | None = None
    tags: list[str] = []


class PredictionConfig(BaseModel):
    major_onnx_dir: Path | None = None
    major_output_path: Path | None = None
    major_output_dir: Path | None = None
    major_workers: int = 8
    major_shard_workers: int = 4
    major_input_dir: Path | None = None
    major_input_glob: str = "*.parquet"
    major_input_path: Path | None = None
    major_input_paths: list[Path] | None = None
    sub_onnx_dir: Path | None = None
    sub_shard_workers: int = 4
    sub_major_workers: int = 8
    sub_input_dir: Path | None = None
    sub_input_glob: str = "*.parquet"
    sub_input_path: Path | None = None
    sub_input_paths: list[Path] | None = None
    input_dir: Path | None = None
    input_glob: str = "*.parquet"
    input_path: Path | None = None
    input_paths: list[Path] | None = None
    output_path: Path | None = None
    output_dir: Path | None = None
    major_batch_size: int = 64
    sub_batch_size: int = 256
    major_max_length: int = 128
    sub_max_length: int = 256


class RootConfig(BaseModel):
    app: Literal["major", "sub", "signals", "predict"] = "major"
    seed: int = 42
    wandb: WandbConfig = WandbConfig()
    major: MajorConfig = MajorConfig()
    sub: SubConfig = SubConfig()
    signals: SignalsConfig = SignalsConfig()
    prediction: PredictionConfig = PredictionConfig()

    def to_wandb(self):
        config: dict[str, Any] = {"seed": self.seed}
        if self.app == "major":
            config["major"] = self.major.to_wandb()
        elif self.app == "sub":
            config["sub"] = self.sub.to_wandb()
        elif self.app == "signals":
            config["signals"] = {
                "tcn": self.signals.tcn.model_dump(),
                "training": self.signals.training.model_dump(),
                "isolation_forest": self.signals.isolation_forest.model_dump(),
                "lightgbm": self.signals.lightgbm.model_dump(),
                "dataset": self.signals.dataset.model_dump(),
                "ohlcv": self.signals.ohlcv.model_dump(),
            }
        return config


_config_instance: RootConfig | None = None


def init_config(app: Literal["major", "sub", "signals", "predict"], config_path: str | None = None) -> None:
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

    _config_instance = _build_root_config(raw, cfg_path.parent)
    _config_instance.app = app

    if app == "major":
        _config_instance.wandb.tags = ["major"]
    elif app == "signals":
        _config_instance.wandb.tags = ["signals"]
    elif app == "sub":
        _config_instance.wandb.tags = ["sub"]
    elif app == "predict":
        _config_instance.wandb.tags = ["predict"]


def get_config() -> RootConfig:
    """Return the singleton config. Must call init_config() first."""
    assert _config_instance is not None, "Config not initialized. Call init_config() first."
    return _config_instance


def load_config(path: str | Path | None = None) -> RootConfig:
    """Load trainer/config.toml and resolve relative paths."""
    import tomllib

    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config.toml"
    path = Path(path)

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    return _build_root_config(raw, path.parent)


def _resolve_path_fields(section: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, val in section.items():
        if isinstance(val, list) and key.endswith("_paths"):
            resolved[key] = [root / item if isinstance(item, str) else item for item in val]
        elif isinstance(val, str) and (
            key.endswith("_path") or key.endswith("_dir") or key.endswith("_file") or key.endswith("_checkpoint")
        ):
            resolved[key] = root / val
        else:
            resolved[key] = val
    return resolved


def _build_root_config(raw: dict[str, Any], root: Path) -> RootConfig:
    filtered: dict[str, Any] = {}

    if "wandb" in raw:
        filtered["wandb"] = raw["wandb"]
    if "major" in raw:
        filtered["major"] = raw["major"]
    if "sub" in raw:
        filtered["sub"] = raw["sub"]
    if "signals" in raw:
        signals_section = dict(raw["signals"])
        if "dataset" in signals_section:
            signals_section["dataset"] = _resolve_path_fields(signals_section["dataset"], root.parent)
        if "training" in signals_section:
            signals_section["training"] = _resolve_path_fields(signals_section["training"], root.parent)
        if "ohlcv" in signals_section:
            signals_section["ohlcv"] = _resolve_path_fields(signals_section["ohlcv"], root.parent)
        filtered["signals"] = signals_section
    if "predict" in raw:
        filtered["prediction"] = _resolve_path_fields(raw["predict"], root.parent)

    return RootConfig.model_validate(filtered)
