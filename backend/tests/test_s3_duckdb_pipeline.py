from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import dashboard_agent.s3_duckdb_pipeline as pipeline_module

from dashboard_agent.s3_duckdb_pipeline import (
    dataset_from_key,
    discover_latest_snapshots,
    initialize_legacy_base,
    run_history_backfill,
    run_pipeline,
)


class FakePaginator:
    def __init__(self, objects: list[dict]):
        self.objects = objects

    def paginate(self, **_kwargs):
        yield {"Contents": list(self.objects)}


class FakeS3Client:
    def __init__(self, objects: list[dict], files: dict[str, Path]):
        self.objects = objects
        self.files = files

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)

    def download_file(self, bucket: str, key: str, filename: str):
        assert bucket == "lead-etl"
        Path(filename).write_bytes(self.files[key].read_bytes())


def _parquet(path: Path, rows: list[tuple[int, str]]) -> Path:
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE source(id INTEGER, label VARCHAR)")
        con.executemany("INSERT INTO source VALUES (?, ?)", rows)
        con.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        con.close()
    return path


def _enrollment_parquet(path: Path) -> Path:
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TABLE source (
                enrollment_id BIGINT,
                user_id BIGINT,
                course_id VARCHAR,
                username VARCHAR,
                email VARCHAR,
                course_type VARCHAR,
                enrolled_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO source VALUES
                (101, 1, 'course-v1:legacy', 'updated-user', 'updated@example.com', 'self', '2026-08-14'),
                (102, 2, 'course-v1:new', 'new-user', 'new@example.com', 'instructor', '2026-08-14')
            """
        )
        con.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        con.close()
    return path


def _legacy_database(path: Path) -> Path:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE unified_records (
                source_path VARCHAR NOT NULL,
                source_format VARCHAR NOT NULL,
                source_table VARCHAR NOT NULL,
                record_index BIGINT NOT NULL,
                payload_json JSON,
                ingested_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO unified_records VALUES "
            "('legacy/source', 'json', 'legacy', 0, '{\"id\":1}', now()), "
            "('legacy/source', 'json', 'legacy', 1, '{\"id\":2}', now())"
        )
        con.execute(
            "CREATE TABLE source_summary AS SELECT source_path, source_format, source_table, "
            "count(*) AS records FROM unified_records GROUP BY 1, 2, 3"
        )
        con.execute("CREATE TABLE legacy_sentinel(value VARCHAR)")
        con.execute("INSERT INTO legacy_sentinel VALUES ('preserve-me')")
        con.execute(
            "CREATE TABLE dashboard_agent_user_dim "
            "(user_id BIGINT, username VARCHAR, email VARCHAR, full_name VARCHAR)"
        )
        con.execute(
            "INSERT INTO dashboard_agent_user_dim VALUES "
            "(1, 'old-user', 'old@example.com', 'Legacy Name')"
        )
        con.execute(
            "CREATE TABLE dashboard_agent_course_dim "
            "(course_id VARCHAR, course_type VARCHAR, subject_name VARCHAR)"
        )
        con.execute(
            "INSERT INTO dashboard_agent_course_dim VALUES "
            "('course-v1:legacy', 'old-type', 'Legacy Subject')"
        )
        con.execute(
            "CREATE TABLE dashboard_agent_user_course_fact "
            "(user_id BIGINT, course_id VARCHAR, username VARCHAR, email VARCHAR, "
            "course_type VARCHAR, full_name VARCHAR)"
        )
        con.execute(
            "INSERT INTO dashboard_agent_user_course_fact VALUES "
            "(1, 'course-v1:legacy', 'old-user', 'old@example.com', 'old-type', 'Legacy Name')"
        )
    finally:
        con.close()
    return path


def _item(key: str, hour: int, path: Path) -> dict:
    return {
        "Key": key,
        "ETag": f'"etag-{hour}-{path.stat().st_size}"',
        "Size": path.stat().st_size,
        "LastModified": datetime(2026, 8, 14, hour, tzinfo=timezone.utc),
    }


def test_dataset_from_partitioned_key_is_stable():
    key = "openedx/raw/mysql/fct_grade_detail/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    assert dataset_from_key(key) == "openedx/raw/mysql/fct_grade_detail"


def test_discovery_uses_newest_snapshot_per_dataset(tmp_path):
    older = _parquet(tmp_path / "older.parquet", [(1, "old")])
    newer = _parquet(tmp_path / "newer.parquet", [(2, "new")])
    other = _parquet(tmp_path / "other.parquet", [(3, "other")])
    old_key = "openedx/raw/mysql/course/ingest_date=2026-08-14/ingest_hour=07/old.parquet"
    new_key = "openedx/raw/mysql/course/ingest_date=2026-08-14/ingest_hour=08/new.parquet"
    other_key = "bookroll/raw/mysql/notes/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    objects = [_item(old_key, 7, older), _item(new_key, 8, newer), _item(other_key, 8, other)]
    client = FakeS3Client(objects, {old_key: older, new_key: newer, other_key: other})

    snapshots = discover_latest_snapshots("s3://lead-etl", client=client)

    assert [(item.dataset, item.key) for item in snapshots] == [
        ("bookroll/raw/mysql/notes", other_key),
        ("openedx/raw/mysql/course", new_key),
    ]


def test_pipeline_replaces_changed_snapshot_without_duplicate_rows(tmp_path):
    first = _parquet(tmp_path / "first.parquet", [(1, "old")])
    second = _parquet(tmp_path / "second.parquet", [(2, "new"), (3, "newer")])
    other = _parquet(tmp_path / "other.parquet", [(9, "stable")])
    first_key = "openedx/raw/mysql/course/ingest_date=2026-08-14/ingest_hour=07/data.parquet"
    second_key = "openedx/raw/mysql/course/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    other_key = "bookroll/raw/mysql/notes/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    objects = [_item(first_key, 7, first), _item(other_key, 8, other)]
    files = {first_key: first, second_key: second, other_key: other}
    client = FakeS3Client(objects, files)
    database = tmp_path / "warehouse.duckdb"

    initial = run_pipeline("s3://lead-etl", database, client=client)
    unchanged = run_pipeline("s3://lead-etl", database, client=client)
    client.objects[:] = [_item(second_key, 8, second), _item(other_key, 8, other)]
    updated = run_pipeline("s3://lead-etl", database, client=client)

    assert initial.status == "updated"
    assert unchanged.status == "unchanged"
    assert updated.changed_datasets == 1
    assert updated.total_records == 3
    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM unified_records").fetchone()[0] == 3
        assert con.execute(
            "SELECT json_extract_string(payload_json, '$.label') FROM unified_records "
            "WHERE source_table = 'openedx/raw/mysql/course' ORDER BY record_index"
        ).fetchall() == [("new",), ("newer",)]
        assert con.execute("SELECT count(*) FROM lead_etl_object_manifest").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM lead_etl_pipeline_runs").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM lead_etl_openedx_raw_mysql_course").fetchone()[0] == 2
    finally:
        con.close()


def test_history_backfill_reconstructs_incremental_and_snapshot_datasets(tmp_path):
    incremental_first = _parquet(tmp_path / "incremental-first.parquet", [(1, "one"), (2, "old")])
    incremental_latest = _parquet(tmp_path / "incremental-latest.parquet", [(2, "new"), (3, "three")])
    snapshot_first = _parquet(tmp_path / "snapshot-first.parquet", [(10, "stable")])
    snapshot_latest = _parquet(tmp_path / "snapshot-latest.parquet", [(10, "stable"), (11, "added")])
    incremental_first_key = "openedx/raw/mysql/fct_events/ingest_date=2026-08-14/ingest_hour=07/data.parquet"
    incremental_latest_key = "openedx/raw/mysql/fct_events/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    snapshot_first_key = "bookroll/raw/mysql/fct_notes/ingest_date=2026-08-14/ingest_hour=07/data.parquet"
    snapshot_latest_key = "bookroll/raw/mysql/fct_notes/ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    files = {
        incremental_first_key: incremental_first,
        incremental_latest_key: incremental_latest,
        snapshot_first_key: snapshot_first,
        snapshot_latest_key: snapshot_latest,
    }
    objects = [
        _item(incremental_first_key, 7, incremental_first),
        _item(incremental_latest_key, 8, incremental_latest),
        _item(snapshot_first_key, 7, snapshot_first),
        _item(snapshot_latest_key, 8, snapshot_latest),
    ]
    client = FakeS3Client(objects, files)
    database = tmp_path / "warehouse.duckdb"
    run_pipeline("s3://lead-etl", database, client=client)

    result = run_history_backfill("s3://lead-etl", database, client=client)

    assert result.datasets == 2
    assert result.objects == 4
    assert result.current_records == 5
    assert result.history_records == 6
    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute(
            "SELECT id, label FROM lead_etl_openedx_raw_mysql_fct_events ORDER BY id"
        ).fetchall() == [(1, "one"), (2, "new"), (3, "three")]
        assert con.execute(
            "SELECT id, label FROM lead_etl_bookroll_raw_mysql_fct_notes ORDER BY id"
        ).fetchall() == [(10, "stable"), (11, "added")]
        assert con.execute(
            "SELECT dataset, ingestion_mode, key_columns_json FROM lead_etl_object_manifest ORDER BY dataset"
        ).fetchall() == [
            ("bookroll/raw/mysql/fct_notes", "snapshot", "[]"),
            ("openedx/raw/mysql/fct_events", "incremental", '["id"]'),
        ]
    finally:
        con.close()


def test_hourly_run_merges_future_incremental_delta_after_backfill(tmp_path):
    first = _parquet(tmp_path / "first.parquet", [(1, "one"), (2, "old")])
    second = _parquet(tmp_path / "second.parquet", [(2, "new"), (3, "three")])
    third = _parquet(tmp_path / "third.parquet", [(3, "updated"), (4, "four")])
    keys = [
        "openedx/raw/mysql/fct_events/ingest_date=2026-08-14/ingest_hour=07/data.parquet",
        "openedx/raw/mysql/fct_events/ingest_date=2026-08-14/ingest_hour=08/data.parquet",
        "openedx/raw/mysql/fct_events/ingest_date=2026-08-14/ingest_hour=09/data.parquet",
    ]
    files = dict(zip(keys, (first, second, third)))
    objects = [_item(keys[0], 7, first), _item(keys[1], 8, second)]
    client = FakeS3Client(objects, files)
    database = tmp_path / "warehouse.duckdb"
    run_pipeline("s3://lead-etl", database, client=client)
    run_history_backfill("s3://lead-etl", database, client=client)
    client.objects.append(_item(keys[2], 9, third))

    result = run_pipeline("s3://lead-etl", database, client=client)

    assert result.changed_datasets == 1
    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute(
            "SELECT id, label FROM lead_etl_openedx_raw_mysql_fct_events ORDER BY id"
        ).fetchall() == [(1, "one"), (2, "new"), (3, "updated"), (4, "four")]
        assert con.execute("SELECT count(*) FROM lead_etl_history_openedx_raw_mysql_fct_events").fetchone()[0] == 6
        assert con.execute("SELECT count(*) FROM lead_etl_history_objects").fetchone()[0] == 3
    finally:
        con.close()


def test_legacy_base_preserves_old_tables_and_merges_compatible_columns(tmp_path):
    legacy = _legacy_database(tmp_path / "legacy.duckdb")
    enrollment = _enrollment_parquet(tmp_path / "enrollment.parquet")
    key = (
        "openedx/raw/mysql/fct_course_enrollment/"
        "ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    )
    client = FakeS3Client([_item(key, 8, enrollment)], {key: enrollment})
    database = tmp_path / "warehouse.duckdb"

    result = initialize_legacy_base(
        legacy,
        database,
        "s3://lead-etl",
        client=client,
    )

    assert result.legacy_records == 2
    assert result.new_current_records == 2
    assert result.combined_records == 4
    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute("SELECT * FROM legacy_sentinel").fetchall() == [("preserve-me",)]
        assert con.execute(
            "SELECT user_id, username, email, full_name FROM dashboard_agent_user_dim ORDER BY user_id"
        ).fetchall() == [
            (1, "updated-user", "updated@example.com", "Legacy Name"),
            (2, "new-user", "new@example.com", None),
        ]
        assert con.execute(
            "SELECT course_id, course_type, subject_name FROM dashboard_agent_course_dim ORDER BY course_id"
        ).fetchall() == [
            ("course-v1:legacy", "self", "Legacy Subject"),
            ("course-v1:new", "instructor", None),
        ]
        assert con.execute(
            "SELECT user_id, course_id, username, email, course_type, full_name "
            "FROM dashboard_agent_user_course_fact ORDER BY user_id"
        ).fetchall() == [
            (1, "course-v1:legacy", "updated-user", "updated@example.com", "self", "Legacy Name"),
            (2, "course-v1:new", "new-user", "new@example.com", "instructor", None),
        ]
        assert con.execute("SELECT count(*) FROM lead_etl_legacy_merge_audit").fetchone()[0] == 3
    finally:
        con.close()


def test_hourly_in_place_update_does_not_copy_legacy_warehouse(tmp_path, monkeypatch):
    legacy = _legacy_database(tmp_path / "legacy.duckdb")
    enrollment = _enrollment_parquet(tmp_path / "enrollment.parquet")
    key_08 = (
        "openedx/raw/mysql/fct_course_enrollment/"
        "ingest_date=2026-08-14/ingest_hour=08/data.parquet"
    )
    key_09 = (
        "openedx/raw/mysql/fct_course_enrollment/"
        "ingest_date=2026-08-14/ingest_hour=09/data.parquet"
    )
    client = FakeS3Client([_item(key_08, 8, enrollment)], {key_08: enrollment, key_09: enrollment})
    database = tmp_path / "warehouse.duckdb"
    initialize_legacy_base(legacy, database, "s3://lead-etl", client=client)
    client.objects[:] = [_item(key_09, 9, enrollment)]

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("in-place hourly updates must not copy the warehouse")

    monkeypatch.setattr(pipeline_module.shutil, "copy2", fail_copy)
    result = run_pipeline("s3://lead-etl", database, client=client, in_place=True)

    assert result.status == "updated"
    assert result.changed_datasets == 1
    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM unified_records").fetchone()[0] == 4
        assert con.execute("SELECT count(*) FROM legacy_sentinel").fetchone()[0] == 1
    finally:
        con.close()
