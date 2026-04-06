"""Logger management using loguru."""

import sys
from pathlib import Path


class TrainerLogger:
    """Thin wrapper around loguru.logger with file + console output."""

    def __init__(self, log_path: Path | None = None):
        from loguru import logger

        self.logger = logger
        self.logger.remove()

        if log_path is not None:
            self.logger.add(
                log_path,
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

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


_instance: TrainerLogger | None = None


def init_logger(log_path: Path | None = None) -> TrainerLogger:
    """Initialize (or return existing) the singleton logger instance."""
    global _instance
    if _instance is None:
        _instance = TrainerLogger(log_path)
    return _instance


def get_logger() -> TrainerLogger:
    """Return the singleton logger instance. Must call init_logger(...) first."""
    assert _instance is not None, "Logger not initialized. Call init_logger(...) first."
    return _instance
