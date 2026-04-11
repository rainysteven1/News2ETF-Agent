from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import main
import pytest
from src.logger import get_logger, init_logger, logger


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


def test_backtest_defaults_log_path_to_checkpoint_run_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    cfg = SimpleNamespace()

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
        run_id="bt_test_logger",
        resume_from_week=None,
        resume_to_week=None,
        resume_latest=False,
    )

    assert captured["run_id"] == "bt_test_logger"
    assert captured["checkpoint_dir"] == tmp_path / "checkpoints"
    assert captured["engine_run_id"] == "bt_test_logger"
    assert captured["engine_checkpoint_dir"] == tmp_path / "checkpoints"
    assert captured["log_path"] == tmp_path / "checkpoints" / "bt_test_logger" / "backtest.log"


def test_backtest_preflight_reports_missing_sentiment_dataset(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            output_agent_features_oof=tmp_path / "missing.oof.parquet",
            output_agent_features=tmp_path / "missing.parquet",
            output_sentiment=tmp_path / "missing_sentiment.parquet",
        ),
        predict=SimpleNamespace(signals_onnx_dir=tmp_path / "bundle"),
    )

    with pytest.raises(FileNotFoundError, match="signals sentiment dataset"):
        main._validate_backtest_inputs(cfg)


def test_backtest_preflight_accepts_existing_sentiment_and_bundle(tmp_path: Path) -> None:
    sentiment_path = tmp_path / "sentiment.parquet"
    bundle_dir = tmp_path / "bundle"
    sentiment_path.write_text("ok", encoding="utf-8")
    bundle_dir.mkdir()

    cfg = SimpleNamespace(
        data=SimpleNamespace(
            output_agent_features_oof=tmp_path / "missing.oof.parquet",
            output_agent_features=tmp_path / "missing.parquet",
            output_sentiment=sentiment_path,
        ),
        predict=SimpleNamespace(signals_onnx_dir=bundle_dir),
    )

    main._validate_backtest_inputs(cfg)


def test_init_src_resumes_wandb_from_run_meta(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    logs: list[tuple[object, ...]] = []

    class DummyCfg:
        seed = 42

        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            return {"seed": 42}

    class DummyHandler:
        def metadata(self) -> dict[str, object]:
            return {
                "wandb_run_id": "wandb-123",
                "wandb_run_name": "bt_test",
                "project": "demo",
                "entity": "demo-team",
                "mode": "online",
                "tags": ["backtest"],
            }

    run_dir = tmp_path / "checkpoints" / "bt_test"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        '{"wandb_run_id":"wandb-123","wandb_run_name":"bt_test","project":"demo","entity":"demo-team","mode":"online","created_at":"2026-04-10T00:00:00+00:00"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_ROOT", tmp_path)
    monkeypatch.setattr(main, "init_config", lambda path: None)
    monkeypatch.setattr(main, "init_logger", lambda path: None)
    monkeypatch.setattr(main, "get_config", lambda: DummyCfg())
    monkeypatch.setattr(main, "_init_seed", lambda seed: None)
    monkeypatch.setattr(main, "init_runtime", lambda **kwargs: None)
    monkeypatch.setattr(main.logger, "info", lambda *args: logs.append(args))
    monkeypatch.setattr(
        main.WandbRegistry,
        "init",
        lambda key, **kwargs: captured.update({"key": key, **kwargs}),
    )
    monkeypatch.setattr(main.WandbRegistry, "get", lambda key: DummyHandler())

    main._init_src(
        None,
        None,
        run_id="bt_test",
        checkpoint_dir=tmp_path / "checkpoints",
        init_wandb=True,
        wandb_tags=["backtest"],
    )

    assert captured["key"] == "backtest"
    assert captured["existing_run_id"] == "wandb-123"
    assert captured["resume"] == "must"

    payload = main._load_run_meta(tmp_path / "checkpoints", "bt_test")
    assert payload["wandb_run_id"] == "wandb-123"
    assert payload["run_id"] == "bt_test"
    assert payload["created_at"] == "2026-04-10T00:00:00+00:00"
    assert isinstance(payload["updated_at"], str)
    assert any("Resuming W&B run" in str(item[0]) for item in logs)


def test_visualize_backtest_upload_resumes_wandb_from_run_meta(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    image_calls: list[tuple[dict[str, Path], dict[str, str] | None]] = []

    class DummyCfg:
        data = SimpleNamespace(
            output_backtest=tmp_path / "unused_results.parquet",
            output_backtest_metrics=tmp_path / "unused_metrics.parquet",
        )

        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            return {"seed": 42}

    class DummyResult:
        run_id = "bt_test"
        output_dir = tmp_path / "checkpoints" / "bt_test" / "visualizations"
        report_path = output_dir / "report.html"
        image_paths = [output_dir / "equity_curve.png"]

    class DummyHandler:
        tags = ["backtest"]

        def add_tags(self, tags: list[str]) -> None:
            self.tags = list(dict.fromkeys([*self.tags, *tags]))

        def log_images(
            self,
            images: dict[str, Path],
            *,
            captions: dict[str, str] | None = None,
            gallery_key: str | None = None,
            step=None,
        ) -> None:
            del gallery_key, step
            image_calls.append((images, captions))

        def metadata(self) -> dict[str, object]:
            return {
                "wandb_run_id": "wandb-123",
                "wandb_run_name": "bt_test_existing",
                "project": "demo",
                "entity": "demo-team",
                "mode": "online",
                "tags": list(self.tags),
            }

    run_dir = tmp_path / "checkpoints" / "bt_test"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        '{"run_id":"bt_test","wandb_run_id":"wandb-123","wandb_run_name":"bt_test_existing","tags":["backtest"],"latest_total_value":1012345.0,"created_at":"2026-04-10T00:00:00+00:00"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_ROOT", tmp_path)
    monkeypatch.setattr(main, "_init_src", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "get_config", lambda: DummyCfg())
    monkeypatch.setattr(main, "visualize_backtest", lambda **kwargs: DummyResult())
    monkeypatch.setattr(
        main.WandbRegistry,
        "init",
        lambda key, **kwargs: captured.update({"key": key, **kwargs}),
    )
    monkeypatch.setattr(main.WandbRegistry, "get", lambda key: DummyHandler())
    monkeypatch.setattr(main.WandbRegistry, "finish_all", lambda: None)

    main.visualize_backtest_cmd(
        config=None,
        run_id="bt_test",
        results_path=None,
        metrics_path=None,
        output_dir=None,
        upload_wandb=True,
    )

    assert captured["key"] == "visualize-backtest"
    assert captured["existing_run_id"] == "wandb-123"
    assert captured["resume"] == "must"
    assert captured["run_name"] == "bt_test_existing"
    assert captured["tags"] == ["backtest", "visualized"]
    assert image_calls[0][0] == {"backtest/visualizations/equity_curve": DummyResult.image_paths[0]}

    payload = main._load_run_meta(tmp_path / "checkpoints", "bt_test")
    assert payload["wandb_run_id"] == "wandb-123"
    assert payload["tags"] == ["backtest", "visualized"]
    assert payload["latest_total_value"] == 1012345.0
    assert payload["created_at"] == "2026-04-10T00:00:00+00:00"
    assert isinstance(payload["updated_at"], str)
