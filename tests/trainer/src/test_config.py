from pathlib import Path

from trainer.src.config.root import load_config


def test_load_config_resolves_predict_paths():
    cfg = load_config()

    assert isinstance(cfg.prediction.major_onnx_dir, Path)
    assert isinstance(cfg.prediction.major_output_dir, Path)
    assert isinstance(cfg.prediction.sub_onnx_dir, Path)
    assert isinstance(cfg.prediction.major_input_dir, Path)
    assert isinstance(cfg.prediction.sub_input_dir, Path)
    assert isinstance(cfg.prediction.output_dir, Path)
    assert cfg.prediction.major_shard_workers > 0
    assert cfg.prediction.sub_shard_workers > 0
    assert cfg.prediction.sub_major_workers > 0
    assert cfg.prediction.sub_backend in {"setfit", "supervised"}
    assert cfg.prediction.major_input_dir.is_absolute()
    assert cfg.prediction.sub_input_dir.is_absolute()
    assert cfg.prediction.output_dir.is_absolute()


def test_load_config_includes_signals_section_with_resolved_paths():
    cfg = load_config()

    assert cfg.signals.dataset.raw_data_path is not None
    assert cfg.signals.dataset.output_sentiment is not None
    assert cfg.signals.training.output_checkpoint is not None
    assert cfg.signals.ohlcv.ohlcv_path is not None
    assert cfg.signals.dataset.raw_data_path.is_absolute()
    assert cfg.signals.training.output_checkpoint.is_absolute()
