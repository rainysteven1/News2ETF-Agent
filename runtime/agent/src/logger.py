"""Singleton loguru logger for src modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class SrcLogger:
    """Thin wrapper around loguru.logger with console DEBUG and file INFO sinks."""

    def __init__(self, log_path: Path | str | None = None):
        from loguru import logger as loguru_logger

        self.logger = loguru_logger
        self.log_path: Path | None = None
        self.configure(log_path)

    def configure(self, log_path: Path | str | None = None) -> None:
        self.logger.remove()
        self.log_path = Path(log_path) if log_path is not None else None

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.add(
                self.log_path,
                rotation="10 MB",
                retention="7 days",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            )

        self.logger.add(
            sys.stderr,
            level="DEBUG",
            format="<cyan>{time:HH:mm:ss}</cyan> | <level>{level}</level> | {message}",
            colorize=True,
        )


class _LoggerProxy:
    """Lazy proxy so modules can import `logger` before init_logger() runs."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_logger().logger, name)


_instance: SrcLogger | None = None
logger = _LoggerProxy()


def init_logger(log_path: Path | str | None = None) -> SrcLogger:
    """Initialize or reconfigure the singleton logger."""
    global _instance
    if _instance is None:
        _instance = SrcLogger(log_path)
    else:
        _instance.configure(log_path)
    return _instance


def get_logger() -> SrcLogger:
    """Return the singleton logger instance."""
    global _instance
    if _instance is None:
        _instance = SrcLogger()
    return _instance
