"""Explicit signals inference pipeline using deployed ONNX bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from loguru import logger

_RUNTIME_ROOT = Path(__file__).resolve().parents[3] / "runtime" / "agent"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from src.signals.signals_inference import SignalsONNXInferencePipeline
from trainer.src.config import get_config, load_config
from trainer.src.datasets.signals import WeeklySignalDataset


def run_inference(
    *,
    bundle_dir: Path | None = None,
    output_path: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    force_dataset: bool = False,
) -> Path:
    try:
        cfg = get_config().signals
    except AssertionError:
        cfg = load_config().signals

    dataset = WeeklySignalDataset(
        cfg.dataset,
        force=force_dataset,
        ohlcv_cfg=cfg.ohlcv,
    )
    assert dataset.sentiment_df is not None, "Signals sentiment dataset is unavailable"
    sentiment_df = dataset.sentiment_df

    mapping_path = Path("data/meta_sector_mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError("Missing data/meta_sector_mapping.json for signals inference.")
    with open(mapping_path, encoding="utf-8") as f:
        meta_map = json.load(f)
    del meta_map

    resolved_bundle = bundle_dir or cfg.training.deploy_onnx_dir
    if resolved_bundle is None:
        raise ValueError("signals.training.deploy_onnx_dir is not configured and --bundle-dir was not provided.")
    resolved_output = output_path or Path("data/agent_features.oof.parquet")

    logger.info(f"[Signals/Infer] bundle={resolved_bundle}")
    logger.info(f"[Signals/Infer] output={resolved_output}")
    if start_date or end_date:
        logger.info(f"[Signals/Infer] date filter | start={start_date or '-'} | end={end_date or '-'}")

    pipeline = SignalsONNXInferencePipeline(
        bundle_dir=resolved_bundle,
        meta_sector_mapping_path=mapping_path,
    )
    feature_df = pipeline.infer_feature_frame(
        sentiment_df,
        start_date=start_date,
        end_date=end_date,
        output_path=resolved_output,
    )
    logger.info(f"[Signals/Infer] wrote {len(feature_df)} rows -> {resolved_output}")
    return resolved_output
