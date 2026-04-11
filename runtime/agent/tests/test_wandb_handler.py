from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src import wandb_handler as wandb_handler_module


def test_log_images_uses_wandb_image(monkeypatch, tmp_path: Path) -> None:
    captured_images: list[tuple[str, str]] = []

    def fake_image(path: str, caption: str):
        captured_images.append((path, caption))
        return {"path": path, "caption": caption}

    class DummyRun:
        def __init__(self) -> None:
            self.log_calls: list[tuple[dict[str, object], int | None]] = []

        def log(self, payload: dict[str, object], step: int | None = None) -> None:
            self.log_calls.append((payload, step))

    monkeypatch.setattr(
        wandb_handler_module,
        "get_config",
        lambda: SimpleNamespace(wandb=SimpleNamespace(project="demo", entity=None, mode="disabled")),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=fake_image))

    image_path = tmp_path / "equity_curve.png"
    image_path.write_bytes(b"png")
    handler = wandb_handler_module.WandbHandler()
    dummy_run = DummyRun()
    handler._run = dummy_run

    handler.log_images(
        {"backtest/visualizations/equity_curve": image_path},
        captions={"backtest/visualizations/equity_curve": "bt_demo equity curve"},
        gallery_key="backtest_visualizations",
        step=3,
    )

    assert captured_images == [(str(image_path), "bt_demo equity curve")]
    assert dummy_run.log_calls == [
        (
            {
                "backtest/visualizations/equity_curve": {"path": str(image_path), "caption": "bt_demo equity curve"},
                "backtest_visualizations": [{"path": str(image_path), "caption": "bt_demo equity curve"}],
            },
            3,
        )
    ]


def test_add_tags_updates_run_tags(monkeypatch) -> None:
    class DummyRun:
        name = "bt_demo"
        tags = ("backtest",)

    monkeypatch.setattr(
        wandb_handler_module,
        "get_config",
        lambda: SimpleNamespace(wandb=SimpleNamespace(project="demo", entity=None, mode="disabled")),
    )

    handler = wandb_handler_module.WandbHandler()
    dummy_run = DummyRun()
    handler._run = dummy_run
    handler._tags = ["backtest"]

    handler.add_tags(["visualized", "backtest"])

    assert dummy_run.tags == ("backtest", "visualized")
    assert handler.metadata()["tags"] == ["backtest", "visualized"]
