# Trainer configuration — shared configs (Wandb, Predict).
# Model-specific configs live in their own modules:
#   trainer.signals.config   -> SignalsConfig (TCN, LightGBM, IsolationForest, Dataset, OHLCV)
#   trainer.finbert.config   -> FinBERTConfig (model + training)
#   trainer.setfit_module.config -> SetFitConfig (model + training)

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from trainer.finbert.config import FinBERTConfig
from trainer.setfit_module.config import SetFitConfig
from trainer.signals.config import SignalsConfig

# ─── WandB ──────────────────────────────────────────────────────────────────────


class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "news2etf"
    entity: str | None = None
    tags: list[str] = []


# ─── Predict ───────────────────────────────────────────────────────────────────


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


# ─── Trainer Root Config ───────────────────────────────────────────────────────


class TrainerConfig(BaseModel):
    app: Literal["finbert", "setfit", "signals"] = "finbert"
    seed: int = 42
    wandb: WandbConfig = WandbConfig()
    finbert: FinBERTConfig = FinBERTConfig()
    setfit: SetFitConfig = SetFitConfig()
    signals: SignalsConfig = SignalsConfig()
    prediction: PredictionConfig = PredictionConfig()

    def to_wandb(self):
        config: dict[str, Any] = {
            "seed": self.seed,
        }

        if self.app == "finbert":
            config["finbert"] = self.finbert.to_wandb()

        return config


def load_config(path: str | Path | None = None) -> TrainerConfig:
    """Load trainer/config.toml and resolve relative paths.

    Only loads shared configs (wandb, predict). Model configs are loaded
    via their own load_*_config() functions from sub-modules.
    """
    import tomllib

    if path is None:
        path = Path(__file__).resolve().parent / "config.toml"

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    filtered: dict[str, Any] = {}

    if "wandb" in raw:
        filtered["wandb"] = raw["wandb"]

    if "predict" in raw:
        _ROOT = Path(__file__).resolve().parent
        predict_section = raw["predict"]
        resolved: dict[str, Any] = {}
        for key, val in predict_section.items():
            if key.endswith("_path") or key.endswith("_dir") or key.endswith("_file"):
                resolved[key] = _ROOT / val
            else:
                resolved[key] = val
        filtered["predict"] = resolved

    return TrainerConfig.model_validate(filtered)


# ── Module-level singleton ─────────────────────────────────────────────────────


_config_instance: TrainerConfig | None = None


def init_config(app: Literal["finbert", "setfit", "signals"], config_path: str | None = None) -> None:
    """Initialize the singleton config from a TOML or JSON config file."""
    global _config_instance
    if config_path is None:
        root_dir = Path(__file__).parent
        cfg_path = root_dir / "config.toml"
    else:
        cfg_path = Path(config_path)

    raw: dict[str, Any] | None = None
    if cfg_path.suffix == ".toml":
        import tomllib

        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)

    assert raw is not None, f"Unsupported config file format: {cfg_path.suffix}"
    _config_instance = TrainerConfig.model_validate(raw)

    _config_instance.app = app

    if app == "finbert":
        _config_instance.wandb.tags = ["finbert"]
    elif app == "signals":
        _config_instance.wandb.tags = ["signals"]


def get_config() -> TrainerConfig:
    """Return the singleton config. Must call init_config() first."""
    assert _config_instance is not None, "Config not initialized. Call init_config() first."
    return _config_instance
