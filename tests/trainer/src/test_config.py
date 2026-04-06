from pathlib import Path

from trainer.src.config.root import load_config


def test_load_config_resolves_predict_paths():
    cfg = load_config()

    assert isinstance(cfg.prediction.finbert_onnx_dir, Path)
    assert isinstance(cfg.prediction.finbert_output_path, Path)
    assert isinstance(cfg.prediction.input_path, Path)
    assert isinstance(cfg.prediction.output_path, Path)
    assert cfg.prediction.input_path.is_absolute()
    assert cfg.prediction.output_path.is_absolute()


def test_load_config_includes_signals_section_with_resolved_paths():
    cfg = load_config()

    assert cfg.signals.dataset.raw_data_path is not None
    assert cfg.signals.dataset.output_sentiment is not None
    assert cfg.signals.training.output_checkpoint is not None
    assert cfg.signals.ohlcv.ohlcv_path is not None
    assert cfg.signals.dataset.raw_data_path.is_absolute()
    assert cfg.signals.training.output_checkpoint.is_absolute()
