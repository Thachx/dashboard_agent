"""Read-only DuckDB access helpers shared by the agent and pipelines.

The hourly ingestion transaction takes DuckDB's exclusive file lock for its
whole duration. Any concurrent reader process would normally fail to connect
with a conflicting-lock error; these helpers retry briefly so short overlaps
degrade into slightly slower answers instead of wrong ones.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_ATTEMPTS = 8
DEFAULT_BACKOFF_SECONDS = 1.0


def _is_transient_lock_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "lock" in message
        or "conflict" in message
        or "being used by another" in message
        or "could not set lock" in message
    )


def connect_read_only(
    database_path: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> Any:
    """Open a DuckDB database read-only, retrying while an ingest holds the lock."""

    import duckdb

    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return duckdb.connect(str(database_path), read_only=True)
        except Exception as exc:
            if not _is_transient_lock_error(exc):
                raise
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Database stayed locked after {attempts} attempts: {last_error}")
