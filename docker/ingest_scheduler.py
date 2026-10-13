#!/usr/bin/env python3
"""In-container S3 -> DuckDB ingestion scheduler.

Replaces the host systemd timer (dashboard-agent-ingest.timer, hourly at
:25) for the Dockerized deployment. Runs the same pipeline the host service
runs, on the same schedule:

    OnCalendar=*-*-* *:25:00  RandomizedDelaySec=2min  Persistent=true

Behavior:
  - Fires at minute 25 of every hour (plus up to 2 minutes of jitter).
  - After startup, if the last successful run was more than 75 minutes ago
    (i.e. the container (re)started across a missed window), runs once to
    catch up — the Docker equivalent of Persistent=true.
  - Each run is the pipeline with --lock-timeout-seconds 60, matching the
    host oneshot service. A failed run is logged and retried next hour;
    it never crashes the container.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # /app
PIPELINE = ROOT / "run_s3_duckdb_pipeline.py"
LAST_RUN_FILE = Path(os.environ.get("INGEST_STATE_FILE", "/app/data/ingest_last_run.txt"))
FIRE_MINUTE = int(os.environ.get("INGEST_FIRE_MINUTE", "25"))
MAX_JITTER_SECONDS = int(os.environ.get("INGEST_MAX_JITTER_SECONDS", "120"))
CATCHUP_MAX_AGE_MINUTES = int(os.environ.get("INGEST_CATCHUP_MAX_AGE_MINUTES", "75"))


def log(message: str) -> None:
    print(f"[ingest-scheduler {datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def last_success_age_minutes() -> float | None:
    try:
        stamp = float(LAST_RUN_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return (time.time() - stamp) / 60.0


def record_success() -> None:
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(f"{time.time():.3f}\n")


def run_pipeline() -> bool:
    env = dict(os.environ)
    # The pipeline loads backend/.env itself when present; keep container env authoritative.
    command = [sys.executable, str(PIPELINE), "--lock-timeout-seconds", "60"]
    log("starting: " + " ".join(command))
    try:
        result = subprocess.run(command, cwd=str(ROOT), env=env, timeout=45 * 60)
    except subprocess.TimeoutExpired:
        log("pipeline timed out after 45 minutes")
        return False
    if result.returncode == 0:
        record_success()
        log("pipeline finished OK")
        return True
    log(f"pipeline failed with exit code {result.returncode}")
    return False


def seconds_until_next_fire(now: datetime) -> float:
    """Seconds until the next hour's fire minute (+ jitter), like the timer."""
    target = now.replace(minute=FIRE_MINUTE, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(hours=1)
    return (target - now).total_seconds() + random.uniform(0, MAX_JITTER_SECONDS)


def main() -> None:
    log(f"scheduler starting (fire minute={FIRE_MINUTE}, jitter<={MAX_JITTER_SECONDS}s)")
    # Persistent=true equivalent: catch up if we missed the window.
    age = last_success_age_minutes()
    if age is None or age > CATCHUP_MAX_AGE_MINUTES:
        log("missed-window catch-up run (Persistent=true equivalent)")
        run_pipeline()
    while True:
        wait = seconds_until_next_fire(datetime.now(timezone.utc))
        log(f"next run in {wait / 60:.1f} minutes")
        time.sleep(max(wait, 30))
        run_pipeline()


if __name__ == "__main__":
    main()
