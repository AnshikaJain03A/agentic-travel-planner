"""
Application-wide logging configuration.

Call :func:`configure_logging` once at process start-up (the API and the
Streamlit app both do this). Everywhere else simply use::

    from core.logging_config import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from logging import Logger

from core.config import settings

_CONFIGURED = False

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or settings.log_level).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(log_level)
    # Avoid duplicate handlers when reloaded (e.g. uvicorn --reload, Streamlit).
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "chromadb", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)
