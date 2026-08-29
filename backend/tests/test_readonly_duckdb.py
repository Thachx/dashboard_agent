from __future__ import annotations

import subprocess
import sys

import pytest

from dashboard_agent.readonly_duckdb import connect_read_only


def test_connect_read_only_retries_until_writer_releases(tmp_path):
    path = tmp_path / "locked.duckdb"
    setup = duckdb.connect(str(path))
    setup.execute("CREATE TABLE t (a INTEGER)")
    setup.execute("INSERT INTO t VALUES (1)")
    setup.close()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import duckdb, time\n"
                f"con = duckdb.connect({str(path)!r})\n"
                "con.execute('CREATE TABLE IF NOT EXISTS w AS SELECT 42 AS v')\n"
                "time.sleep(0.8)\n"
                "con.close()\n"
            ),
        ]
    )
    try:
        con = connect_read_only(str(path), attempts=12, backoff_seconds=0.25)
        try:
            assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 1
        finally:
            con.close()
    finally:
        holder.wait(timeout=10)


def test_connect_read_only_fails_fast_on_missing_file(tmp_path):
    with pytest.raises(Exception):
        connect_read_only(str(tmp_path / "missing.duckdb"), attempts=2, backoff_seconds=0.05)


import duckdb  # noqa: E402
