"""
Thin SQLite access layer.

Provides connection management and schema initialisation. Higher-level
persistence logic (preferences, history) lives in :mod:`memory.memory_manager`,
which builds on the helpers here.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.config import settings
from core.logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA_FILE = Path(__file__).with_name("schema.sql")
_init_lock = threading.Lock()
_initialised = False


def _connect() -> sqlite3.Connection:
    """Create a new SQLite connection with sensible defaults."""
    conn = sqlite3.connect(
        settings.sqlite_db_path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Create tables from ``schema.sql`` if they do not already exist."""
    global _initialised
    with _init_lock:
        if _initialised:
            return
        Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        ddl = _SCHEMA_FILE.read_text(encoding="utf-8")
        with _connect() as conn:
            conn.executescript(ddl)
            conn.commit()
        logger.info("SQLite initialised at %s", settings.sqlite_db_path)
        _initialised = True


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection, committing on success.

    Ensures the schema exists before handing out a connection.
    """
    if not _initialised:
        init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:  # noqa: BLE001 - re-raise after rollback
        conn.rollback()
        logger.exception("Database transaction failed; rolled back")
        raise
    finally:
        conn.close()
