"""
Logging utilities.

Provides:
- Console logging
- File logging
- Rotating log files
- Configurable log levels
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.paths import LOGS_DIR


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_FILE = LOGS_DIR / "application.log"


def get_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or return a configured logger.

    Parameters
    ----------
    name
        Logger name.

    level
        Logging level.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # ----------------------------
    # Console
    # ----------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter,
    )

    logger.addHandler(
        console_handler,
    )

    # ----------------------------
    # Rotating File
    # ----------------------------

    file_handler = RotatingFileHandler(
        DEFAULT_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter,
    )

    logger.addHandler(
        file_handler,
    )

    logger.propagate = False

    return logger


def set_log_level(
    logger: logging.Logger,
    level: int,
) -> None:
    """
    Update logger level.
    """

    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)