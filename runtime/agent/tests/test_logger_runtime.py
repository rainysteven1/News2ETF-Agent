from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from src.logger import get_logger, init_logger, logger


def _load_runtime_main():
    entry_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("runtime_agent_main_test", entry_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logger_is_singleton_and_file_sink_excludes_debug_messages(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"

    first = init_logger(log_path)
    second = init_logger(log_path)

    assert first is second
    assert get_logger() is first

    logger.debug("debug-only")
    logger.info("info-visible")

    content = log_path.read_text(encoding="utf-8")
    assert "info-visible" in content
    assert "debug-only" not in content


def test_backtest_defaults_log_path_to_runtime_checkpoint_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    cfg = SimpleNamespace()
    main = _load_runtime_main()

    def fake_init_src(config_path, log_path, **kwargs):
        captured["config_path"] = config_path
        captured["log_path"] = Path(log_path)
        captured["run_id"] = kwargs.get("run_id")
        captured["checkpoint_dir"] = kwargs.get("checkpoint_dir")
        return cfg

    class DummyEngine:
        def __init__(self, cfg, checkpoint_dir):
            captured["engine_checkpoint_dir"] = checkpoint_dir

        def run(self, *args, **kwargs):
            captured["engine_run_id"] = kwargs["run_id"]

    monkeypatch.setattr(main, "_ROOT", tmp_path)
    monkeypatch.setattr(main, "_init_src", fake_init_src)
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "build_workflow", lambda cfg: object())
    monkeypatch.setattr(main, "WalkForwardEngine", DummyEngine)

    main.backtest(
        start_date="2024-01-01",
        end_date="2024-01-31",
        config=None,
        log_file=None,
        run_id="bt_runtime_logger",
        resume_from_week=None,
        resume_to_week=None,
        resume_latest=False,
    )

    assert captured["run_id"] == "bt_runtime_logger"
    assert captured["checkpoint_dir"] == tmp_path / "checkpoints"
    assert captured["engine_run_id"] == "bt_runtime_logger"
    assert captured["engine_checkpoint_dir"] == tmp_path / "checkpoints"
    assert captured["log_path"] == tmp_path / "checkpoints" / "bt_runtime_logger" / "backtest.log"
