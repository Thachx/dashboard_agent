from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dashboard_agent.s3_source import parse_s3_uri


PARTITION_RE = re.compile(r"/(?:ingest_date|ingest_hour)=[^/]+")
IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")

# These are the legacy analytical grains that can safely accept exact-name,
# type-compatible columns from the new Open edX facts.  Renamed or merely
# similar fields are deliberately not guessed.
LEGACY_COMPATIBLE_TARGETS: dict[str, tuple[str, ...]] = {
    "dashboard_agent_user_dim": ("user_id",),
    "dashboard_agent_course_dim": ("course_id",),
    "dashboard_agent_user_course_fact": ("user_id", "course_id"),
}


@dataclass(frozen=True)
class S3Snapshot:
    dataset: str
    bucket: str
    key: str
    etag: str
    size: int
    last_modified: datetime

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def table_name(self) -> str:
        return physical_table_name(self.dataset)


@dataclass(frozen=True)
class PipelineResult:
    status: str
    discovered_datasets: int
    changed_datasets: int
    total_records: int
    database_path: Path
    backup_path: Path | None = None


@dataclass(frozen=True)
class HistoryBackfillResult:
    datasets: int
    objects: int
    current_records: int
    history_records: int
    database_path: Path


@dataclass(frozen=True)
class LegacyBaseResult:
    datasets: int
    objects: int
    legacy_records: int
    new_current_records: int
    combined_records: int
    history_records: int
    database_path: Path
    replaced_database_backup: Path | None = None


def dataset_from_key(key: str) -> str:
    normalized = key.replace("\\", "/").strip("/")
    partition = PARTITION_RE.search(f"/{normalized}")
    if partition:
        normalized = normalized[: partition.start() - 1]
    else:
        normalized = normalized.rsplit("/", 1)[0] if "/" in normalized else normalized
    return normalized.strip("/")


def physical_table_name(dataset: str) -> str:
    suffix = IDENTIFIER_RE.sub("_", dataset.lower()).strip("_")
    if not suffix:
        raise ValueError(f"Cannot derive a table name from dataset {dataset!r}")
    return f"lead_etl_{suffix}"


def discover_latest_snapshots(
    s3_uri: str,
    *,
    region: str | None = None,
    client: Any | None = None,
) -> list[S3Snapshot]:
    grouped = discover_snapshots(s3_uri, region=region, client=client)
    return [items[-1] for _dataset, items in sorted(grouped.items())]


def discover_snapshots(
    s3_uri: str,
    *,
    region: str | None = None,
    client: Any | None = None,
) -> dict[str, list[S3Snapshot]]:
    bucket, prefix = parse_s3_uri(s3_uri)
    if client is None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("boto3 is required for S3 ingestion") from exc
        client = boto3.client("s3", region_name=region)

    grouped: dict[str, list[S3Snapshot]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.lower().endswith(".parquet"):
                continue
            last_modified = item.get("LastModified")
            if not isinstance(last_modified, datetime):
                continue
            dataset = dataset_from_key(key)
            snapshot = S3Snapshot(
                dataset=dataset,
                bucket=bucket,
                key=key,
                etag=str(item.get("ETag") or "").strip('"'),
                size=int(item.get("Size") or 0),
                last_modified=last_modified,
            )
            grouped.setdefault(dataset, []).append(snapshot)
    for snapshots in grouped.values():
        snapshots.sort(key=lambda item: (item.last_modified, item.key))
    return grouped


def run_history_backfill(
    s3_uri: str,
    database_path: str | Path,
    *,
    region: str | None = None,
    client: Any | None = None,
    lock_timeout_seconds: float = 0,
) -> HistoryBackfillResult:
    database_path = Path(database_path).resolve()
    if not _is_managed_database(database_path):
        raise RuntimeError(f"History backfill requires a managed lead-etl database: {database_path}")
    lock_path = database_path.with_suffix(database_path.suffix + ".ingest.lock")
    with pipeline_lock(lock_path, timeout_seconds=lock_timeout_seconds):
        grouped = discover_snapshots(s3_uri, region=region, client=client)
        if not grouped:
            raise RuntimeError(f"No Parquet snapshots found under {s3_uri}")
        if client is None:
            import boto3

            client = boto3.client("s3", region_name=region)

        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("history-%Y%m%dT%H%M%S.%fZ")
        work_path = database_path.with_suffix(database_path.suffix + f".{run_id}.tmp")
        if work_path.exists():
            work_path.unlink()
        shutil.copy2(database_path, work_path)
        try:
            with tempfile.TemporaryDirectory(prefix="lead-etl-history-") as temp_dir_name:
                _apply_history_backfill(
                    work_path,
                    grouped=grouped,
                    temp_dir=Path(temp_dir_name),
                    client=client,
                    s3_uri=s3_uri,
                    run_id=run_id,
                    started_at=started_at,
                )
            os.replace(work_path, database_path)
        finally:
            if work_path.exists():
                work_path.unlink()

        import duckdb

        con = duckdb.connect(str(database_path), read_only=True)
        try:
            current_records = int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
            history_records = int(con.execute("SELECT count(*) FROM lead_etl_history_records").fetchone()[0])
        finally:
            con.close()
        return HistoryBackfillResult(
            datasets=len(grouped),
            objects=sum(len(items) for items in grouped.values()),
            current_records=current_records,
            history_records=history_records,
            database_path=database_path,
        )


def initialize_legacy_base(
    legacy_database_path: str | Path,
    database_path: str | Path,
    s3_uri: str,
    *,
    region: str | None = None,
    client: Any | None = None,
    lock_timeout_seconds: float = 0,
) -> LegacyBaseResult:
    legacy_database_path = Path(legacy_database_path).resolve()
    database_path = Path(database_path).resolve()
    legacy_records = _validate_legacy_base(legacy_database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = database_path.with_suffix(database_path.suffix + ".ingest.lock")
    with pipeline_lock(lock_path, timeout_seconds=lock_timeout_seconds):
        grouped = discover_snapshots(s3_uri, region=region, client=client)
        if not grouped:
            raise RuntimeError(f"No Parquet snapshots found under {s3_uri}")
        if client is None:
            import boto3

            client = boto3.client("s3", region_name=region)

        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("legacy-base-%Y%m%dT%H%M%S.%fZ")
        work_path = database_path.with_suffix(database_path.suffix + f".{run_id}.tmp")
        if work_path.exists():
            work_path.unlink()
        shutil.copy2(legacy_database_path, work_path)
        try:
            with tempfile.TemporaryDirectory(prefix="lead-etl-legacy-base-") as temp_dir_name:
                _apply_history_backfill(
                    work_path,
                    grouped=grouped,
                    temp_dir=Path(temp_dir_name),
                    client=client,
                    s3_uri=s3_uri,
                    run_id=run_id,
                    started_at=started_at,
                )
            work_combined, work_new, _work_history = _lead_etl_counts(work_path)
            if work_combined != legacy_records + work_new:
                raise RuntimeError(
                    "Combined warehouse row-count reconciliation failed before publish: "
                    f"legacy={legacy_records}, new={work_new}, combined={work_combined}"
                )
            replaced_backup = _publish_replacement_with_label(
                work_path,
                database_path,
                label="lead-etl-only",
            )
        finally:
            if work_path.exists():
                work_path.unlink()

        combined_records, new_current_records, history_records = _lead_etl_counts(database_path)
        return LegacyBaseResult(
            datasets=len(grouped),
            objects=sum(len(items) for items in grouped.values()),
            legacy_records=legacy_records,
            new_current_records=new_current_records,
            combined_records=combined_records,
            history_records=history_records,
            database_path=database_path,
            replaced_database_backup=replaced_backup,
        )


def run_pipeline(
    s3_uri: str,
    database_path: str | Path,
    *,
    region: str | None = None,
    client: Any | None = None,
    lock_timeout_seconds: float = 0,
    preserve_unmanaged_database: bool = True,
    in_place: bool = False,
) -> PipelineResult:
    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = database_path.with_suffix(database_path.suffix + ".ingest.lock")
    with pipeline_lock(lock_path, timeout_seconds=lock_timeout_seconds):
        snapshots = discover_latest_snapshots(s3_uri, region=region, client=client)
        if not snapshots:
            raise RuntimeError(f"No Parquet snapshots found under {s3_uri}")
        _validate_physical_table_names({snapshot.dataset: [snapshot] for snapshot in snapshots})

        managed_database = _is_managed_database(database_path)
        current = _current_snapshots(database_path) if managed_database else {}
        changed = [
            snapshot
            for snapshot in snapshots
            if current.get(snapshot.dataset) != (snapshot.key, snapshot.etag, snapshot.size)
        ]
        removed = sorted(set(current) - {snapshot.dataset for snapshot in snapshots})
        if managed_database and not changed and not removed:
            return PipelineResult(
                status="unchanged",
                discovered_datasets=len(snapshots),
                changed_datasets=0,
                total_records=_total_records(database_path),
                database_path=database_path,
            )

        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        with tempfile.TemporaryDirectory(prefix="lead-etl-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if in_place and managed_database:
                _apply_snapshots(
                    database_path,
                    snapshots=snapshots,
                    changed=changed,
                    removed=removed,
                    temp_dir=temp_dir,
                    client=client,
                    region=region,
                    s3_uri=s3_uri,
                    run_id=run_id,
                    started_at=started_at,
                )
                return PipelineResult(
                    status="updated",
                    discovered_datasets=len(snapshots),
                    changed_datasets=len(changed) + len(removed),
                    total_records=_total_records(database_path),
                    database_path=database_path,
                )
            work_path = database_path.with_suffix(database_path.suffix + f".{run_id}.tmp")
            if work_path.exists():
                work_path.unlink()
            if managed_database:
                shutil.copy2(database_path, work_path)
            try:
                _apply_snapshots(
                    work_path,
                    snapshots=snapshots,
                    changed=changed,
                    removed=removed,
                    temp_dir=temp_dir,
                    client=client,
                    region=region,
                    s3_uri=s3_uri,
                    run_id=run_id,
                    started_at=started_at,
                )
                backup_path = _publish_database(
                    work_path,
                    database_path,
                    preserve_existing=preserve_unmanaged_database and database_path.exists() and not managed_database,
                )
            finally:
                if work_path.exists():
                    work_path.unlink()

        return PipelineResult(
            status="updated",
            discovered_datasets=len(snapshots),
            changed_datasets=len(changed) + len(removed),
            total_records=_total_records(database_path),
            database_path=database_path,
            backup_path=backup_path,
        )


def _apply_snapshots(
    work_path: Path,
    *,
    snapshots: list[S3Snapshot],
    changed: list[S3Snapshot],
    removed: list[str],
    temp_dir: Path,
    client: Any | None,
    region: str | None,
    s3_uri: str,
    run_id: str,
    started_at: datetime,
) -> None:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("duckdb is required for the analytical warehouse") from exc
    if client is None:
        import boto3

        client = boto3.client("s3", region_name=region)

    con = duckdb.connect(str(work_path))
    try:
        _ensure_metadata_tables(con)
        con.execute("BEGIN TRANSACTION")
        for dataset in removed:
            _remove_dataset(con, dataset)
        for index, snapshot in enumerate(changed):
            local_path = temp_dir / f"{index:04d}.parquet"
            client.download_file(snapshot.bucket, snapshot.key, str(local_path))
            _replace_dataset(con, snapshot, local_path, run_id=run_id, ingested_at=started_at)
        _refresh_legacy_compatible_tables(con, run_id=run_id, refreshed_at=started_at)
        _refresh_source_summary(
            con,
            datasets=[snapshot.dataset for snapshot in changed] + removed,
        )
        finished_at = datetime.now(timezone.utc)
        total_records = int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
        con.execute(
            """
            INSERT INTO lead_etl_pipeline_runs
                (run_id, source_uri, started_at, finished_at, status, discovered_datasets, changed_datasets, total_records)
            VALUES (?, ?, ?, ?, 'ok', ?, ?, ?)
            """,
            [run_id, s3_uri, started_at, finished_at, len(snapshots), len(changed) + len(removed), total_records],
        )
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception:
        with contextlib.suppress(Exception):
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _ensure_metadata_tables(con: Any) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS unified_records (
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
        """
        CREATE TABLE IF NOT EXISTS lead_etl_legacy_merge_audit (
            run_id VARCHAR NOT NULL,
            target_table VARCHAR NOT NULL,
            source_tables_json VARCHAR NOT NULL,
            matched_columns_json VARCHAR NOT NULL,
            inserted_rows BIGINT NOT NULL,
            updated_rows BIGINT NOT NULL,
            refreshed_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (run_id, target_table)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lead_etl_object_manifest (
            dataset VARCHAR PRIMARY KEY,
            physical_table VARCHAR NOT NULL,
            s3_uri VARCHAR NOT NULL,
            s3_key VARCHAR NOT NULL,
            etag VARCHAR NOT NULL,
            size_bytes BIGINT NOT NULL,
            source_last_modified TIMESTAMPTZ NOT NULL,
            row_count BIGINT NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            run_id VARCHAR NOT NULL
        )
        """
    )
    manifest_columns = {
        str(row[1]) for row in con.execute("PRAGMA table_info('lead_etl_object_manifest')").fetchall()
    }
    if "ingestion_mode" not in manifest_columns:
        con.execute("ALTER TABLE lead_etl_object_manifest ADD COLUMN ingestion_mode VARCHAR DEFAULT 'snapshot'")
    if "key_columns_json" not in manifest_columns:
        con.execute("ALTER TABLE lead_etl_object_manifest ADD COLUMN key_columns_json VARCHAR DEFAULT '[]'")
    if "order_columns_json" not in manifest_columns:
        con.execute("ALTER TABLE lead_etl_object_manifest ADD COLUMN order_columns_json VARCHAR DEFAULT '[]'")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lead_etl_history_objects (
            dataset VARCHAR NOT NULL,
            s3_uri VARCHAR PRIMARY KEY,
            s3_key VARCHAR NOT NULL,
            etag VARCHAR NOT NULL,
            size_bytes BIGINT NOT NULL,
            source_last_modified TIMESTAMPTZ NOT NULL,
            processed_at TIMESTAMPTZ NOT NULL,
            run_id VARCHAR NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lead_etl_history_records (
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
        """
        CREATE TABLE IF NOT EXISTS lead_etl_pipeline_runs (
            run_id VARCHAR PRIMARY KEY,
            source_uri VARCHAR NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ NOT NULL,
            status VARCHAR NOT NULL,
            discovered_datasets INTEGER NOT NULL,
            changed_datasets INTEGER NOT NULL,
            total_records BIGINT NOT NULL
        )
        """
    )


def _replace_dataset(con: Any, snapshot: S3Snapshot, local_path: Path, *, run_id: str, ingested_at: datetime) -> None:
    manifest = con.execute(
        "SELECT ingestion_mode, key_columns_json, order_columns_json "
        "FROM lead_etl_object_manifest WHERE dataset = ?",
        [snapshot.dataset],
    ).fetchone()
    mode = str(manifest[0]) if manifest and manifest[0] else "snapshot"
    key_columns = _decode_key_columns(manifest[1] if manifest else None)
    order_columns = _decode_key_columns(manifest[2] if manifest else None)
    incoming_name = "__lead_etl_incoming"
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(incoming_name)}")
    _create_parquet_table(con, incoming_name, local_path, snapshot)

    if mode == "incremental" and key_columns and _table_exists(con, snapshot.table_name):
        _replace_from_union(
            con,
            snapshot.table_name,
            incoming_name,
            key_columns=key_columns,
            order_columns=order_columns,
        )
    else:
        con.execute(f"DROP TABLE IF EXISTS {quote_identifier(snapshot.table_name)}")
        con.execute(
            f"ALTER TABLE {quote_identifier(incoming_name)} RENAME TO {quote_identifier(snapshot.table_name)}"
        )

    history_table = history_table_name(snapshot.dataset)
    if _table_exists(con, history_table):
        _append_exact_history(con, history_table, snapshot.table_name if mode == "snapshot" else incoming_name)
        _refresh_history_dataset(con, snapshot.dataset, history_table, ingested_at)
        con.execute("DELETE FROM lead_etl_history_objects WHERE s3_uri = ?", [snapshot.uri])
        con.execute(
            """
            INSERT INTO lead_etl_history_objects
                (dataset, s3_uri, s3_key, etag, size_bytes, source_last_modified, processed_at, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.dataset,
                snapshot.uri,
                snapshot.key,
                snapshot.etag,
                snapshot.size,
                snapshot.last_modified,
                ingested_at,
                run_id,
            ],
        )

    _refresh_unified_dataset(con, snapshot.dataset, snapshot.table_name, ingested_at)
    row_count = int(
        con.execute(f"SELECT count(*) FROM {quote_identifier(snapshot.table_name)}").fetchone()[0]
    )
    _write_manifest(
        con,
        snapshot,
        row_count=row_count,
        ingested_at=ingested_at,
        run_id=run_id,
        mode=mode,
        key_columns=key_columns,
        order_columns=order_columns,
    )
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(incoming_name)}")


def _refresh_legacy_compatible_tables(con: Any, *, run_id: str, refreshed_at: datetime) -> None:
    """Upsert exact-name compatible lead columns into legacy dimensional tables.

    The legacy table remains authoritative: unmatched columns and every existing
    row are preserved.  A lead value is accepted only when it can be losslessly
    cast to the legacy column type; otherwise the transaction fails.
    """
    manifest_rows = con.execute(
        "SELECT physical_table FROM lead_etl_object_manifest ORDER BY physical_table"
    ).fetchall()
    source_tables = [str(row[0]) for row in manifest_rows if _table_exists(con, str(row[0]))]
    if not source_tables:
        return

    for target_table, key_columns in LEGACY_COMPATIBLE_TARGETS.items():
        if not _table_exists(con, target_table):
            continue
        target_types = _table_column_types(con, target_table)
        if not all(column in target_types for column in key_columns):
            continue

        source_specs: list[tuple[str, dict[str, str]]] = []
        matched_columns: set[str] = set()
        for source_table in source_tables:
            source_types = _table_column_types(con, source_table)
            if not all(column in source_types for column in key_columns):
                continue
            shared = set(target_types) & set(source_types)
            shared.difference_update({"_source_path", "_source_last_modified"})
            if not (shared - set(key_columns)):
                continue
            source_specs.append((source_table, source_types))
            matched_columns.update(shared)
        if not source_specs:
            continue

        ordered_columns = [
            column for column in target_types if column in matched_columns or column in key_columns
        ]
        value_columns = [column for column in ordered_columns if column not in key_columns]
        for source_table, source_types in source_specs:
            for column in ordered_columns:
                if column not in source_types:
                    continue
                failed = int(
                    con.execute(
                        f"SELECT count(*) FROM {quote_identifier(source_table)} "
                        f"WHERE {quote_identifier(column)} IS NOT NULL "
                        f"AND try_cast({quote_identifier(column)} AS {target_types[column]}) IS NULL"
                    ).fetchone()[0]
                )
                if failed:
                    raise RuntimeError(
                        f"Cannot safely merge {source_table}.{column} into "
                        f"{target_table}.{column}: {failed} value(s) do not cast to {target_types[column]}"
                    )

        union_parts: list[str] = []
        for source_table, source_types in source_specs:
            projections = []
            for column in ordered_columns:
                if column in source_types:
                    projections.append(
                        f"try_cast({quote_identifier(column)} AS {target_types[column]}) "
                        f"AS {quote_identifier(column)}"
                    )
                else:
                    projections.append(
                        f"NULL::{target_types[column]} AS {quote_identifier(column)}"
                    )
            projections.extend(
                [
                    "_source_last_modified AS __lead_modified",
                    "_source_path AS __lead_path",
                ]
            )
            union_parts.append(
                f"SELECT {', '.join(projections)} FROM {quote_identifier(source_table)}"
            )

        incoming_table = f"__lead_etl_legacy_{target_table}"
        con.execute(f"DROP TABLE IF EXISTS {quote_identifier(incoming_table)}")
        group_keys = ", ".join(quote_identifier(column) for column in key_columns)
        aggregates = [
            f"first({quote_identifier(column)} ORDER BY __lead_modified DESC, __lead_path DESC) "
            f"FILTER (WHERE {quote_identifier(column)} IS NOT NULL) AS {quote_identifier(column)}"
            for column in value_columns
        ]
        non_null_keys = " AND ".join(
            f"{quote_identifier(column)} IS NOT NULL" for column in key_columns
        )
        con.execute(
            f"CREATE TEMP TABLE {quote_identifier(incoming_table)} AS "
            f"SELECT {group_keys}{', ' if aggregates else ''}{', '.join(aggregates)} "
            f"FROM ({' UNION ALL BY NAME '.join(union_parts)}) AS source "
            f"WHERE {non_null_keys} GROUP BY {group_keys}"
        )

        join_condition = " AND ".join(
            f"target.{quote_identifier(column)} = incoming.{quote_identifier(column)}"
            for column in key_columns
        )
        existing_rows = int(
            con.execute(
                f"SELECT count(*) FROM {quote_identifier(incoming_table)} AS incoming "
                f"WHERE EXISTS (SELECT 1 FROM {quote_identifier(target_table)} AS target "
                f"WHERE {join_condition})"
            ).fetchone()[0]
        )
        if value_columns:
            assignments = ", ".join(
                f"{quote_identifier(column)} = coalesce(incoming.{quote_identifier(column)}, "
                f"target.{quote_identifier(column)})"
                for column in value_columns
            )
            con.execute(
                f"UPDATE {quote_identifier(target_table)} AS target SET {assignments} "
                f"FROM {quote_identifier(incoming_table)} AS incoming WHERE {join_condition}"
            )

        insert_columns = list(key_columns) + value_columns
        insert_list = ", ".join(quote_identifier(column) for column in insert_columns)
        select_list = ", ".join(f"incoming.{quote_identifier(column)}" for column in insert_columns)
        con.execute(
            f"INSERT INTO {quote_identifier(target_table)} ({insert_list}) "
            f"SELECT {select_list} FROM {quote_identifier(incoming_table)} AS incoming "
            f"WHERE NOT EXISTS (SELECT 1 FROM {quote_identifier(target_table)} AS target "
            f"WHERE {join_condition})"
        )
        incoming_rows = int(
            con.execute(f"SELECT count(*) FROM {quote_identifier(incoming_table)}").fetchone()[0]
        )
        con.execute(
            """
            INSERT INTO lead_etl_legacy_merge_audit
                (run_id, target_table, source_tables_json, matched_columns_json,
                 inserted_rows, updated_rows, refreshed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                target_table,
                json.dumps([table for table, _types in source_specs]),
                json.dumps(ordered_columns),
                incoming_rows - existing_rows,
                existing_rows,
                refreshed_at,
            ],
        )
        con.execute(f"DROP TABLE IF EXISTS {quote_identifier(incoming_table)}")


def _table_column_types(con: Any, table_name: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2])
        for row in con.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    }


def _create_parquet_table(con: Any, table_name: str, local_path: Path, snapshot: S3Snapshot) -> None:
    con.execute(
        f"""
        CREATE TABLE {quote_identifier(table_name)} AS
        SELECT source.*, ?::VARCHAR AS _source_path, ?::TIMESTAMPTZ AS _source_last_modified
        FROM read_parquet(?) AS source
        """,
        [snapshot.uri, snapshot.last_modified, str(local_path)],
    )


def _replace_from_union(
    con: Any,
    target: str,
    incoming: str,
    *,
    key_columns: list[str],
    order_columns: list[str],
) -> None:
    merged = "__lead_etl_merged"
    keys = ", ".join(quote_identifier(column) for column in key_columns)
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(merged)}")
    order_by = _dedupe_order_expression(order_columns)
    con.execute(
        f"""
        CREATE TABLE {quote_identifier(merged)} AS
        SELECT * EXCLUDE (_lead_etl_rank)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY {keys}
                ORDER BY _source_last_modified DESC, {order_by}, _source_path DESC
            ) AS _lead_etl_rank
            FROM (
                SELECT * FROM {quote_identifier(target)}
                UNION ALL BY NAME
                SELECT * FROM {quote_identifier(incoming)}
            )
        )
        WHERE _lead_etl_rank = 1
        """
    )
    con.execute(f"DROP TABLE {quote_identifier(target)}")
    con.execute(f"ALTER TABLE {quote_identifier(merged)} RENAME TO {quote_identifier(target)}")


def _append_exact_history(con: Any, history_table: str, incoming_table: str) -> None:
    merged = "__lead_etl_history_merged"
    business_columns = _business_columns(con, history_table)
    if not business_columns:
        return
    hash_expression = _row_hash_expression(business_columns)
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(merged)}")
    con.execute(
        f"""
        CREATE TABLE {quote_identifier(merged)} AS
        SELECT * EXCLUDE (_lead_etl_rank)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY {hash_expression}
                ORDER BY _source_last_modified DESC, _source_path DESC
            ) AS _lead_etl_rank
            FROM (
                SELECT * FROM {quote_identifier(history_table)}
                UNION ALL BY NAME
                SELECT * FROM {quote_identifier(incoming_table)}
            )
        )
        WHERE _lead_etl_rank = 1
        """
    )
    con.execute(f"DROP TABLE {quote_identifier(history_table)}")
    con.execute(f"ALTER TABLE {quote_identifier(merged)} RENAME TO {quote_identifier(history_table)}")


def _refresh_unified_dataset(con: Any, dataset: str, table_name: str, ingested_at: datetime) -> None:
    con.execute("DELETE FROM unified_records WHERE source_table = ?", [dataset])
    con.execute(
        f"""
        INSERT INTO unified_records
        SELECT
            regexp_replace(source._source_path, '/ingest_date=.*$', ''),
            'parquet',
            ?,
            row_number() OVER () - 1,
            to_json(source),
            ?
        FROM {quote_identifier(table_name)} AS source
        """,
        [dataset, ingested_at],
    )


def _refresh_history_dataset(con: Any, dataset: str, table_name: str, ingested_at: datetime) -> None:
    history_dataset = f"history/{dataset}"
    con.execute("DELETE FROM lead_etl_history_records WHERE source_table = ?", [history_dataset])
    con.execute(
        f"""
        INSERT INTO lead_etl_history_records
        SELECT
            source._source_path,
            'parquet',
            ?,
            row_number() OVER () - 1,
            to_json(source),
            ?
        FROM {quote_identifier(table_name)} AS source
        """,
        [history_dataset, ingested_at],
    )


def _write_manifest(
    con: Any,
    snapshot: S3Snapshot,
    *,
    row_count: int,
    ingested_at: datetime,
    run_id: str,
    mode: str,
    key_columns: list[str],
    order_columns: list[str],
) -> None:
    con.execute("DELETE FROM lead_etl_object_manifest WHERE dataset = ?", [snapshot.dataset])
    con.execute(
        """
        INSERT INTO lead_etl_object_manifest
            (dataset, physical_table, s3_uri, s3_key, etag, size_bytes, source_last_modified,
             row_count, ingested_at, run_id, ingestion_mode, key_columns_json, order_columns_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            snapshot.dataset,
            snapshot.table_name,
            snapshot.uri,
            snapshot.key,
            snapshot.etag,
            snapshot.size,
            snapshot.last_modified,
            row_count,
            ingested_at,
            run_id,
            mode,
            json.dumps(key_columns),
            json.dumps(order_columns),
        ],
    )


def _apply_history_backfill(
    work_path: Path,
    *,
    grouped: dict[str, list[S3Snapshot]],
    temp_dir: Path,
    client: Any,
    s3_uri: str,
    run_id: str,
    started_at: datetime,
) -> None:
    import duckdb

    _validate_physical_table_names(grouped)
    con = duckdb.connect(str(work_path))
    try:
        _ensure_metadata_tables(con)
        con.execute("BEGIN TRANSACTION")
        for index, (dataset, snapshots) in enumerate(sorted(grouped.items())):
            dataset_dir = temp_dir / f"{index:04d}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            local_files: list[tuple[Path, S3Snapshot]] = []
            for object_index, snapshot in enumerate(snapshots):
                local_path = dataset_dir / f"{object_index:06d}.parquet"
                client.download_file(snapshot.bucket, snapshot.key, str(local_path))
                local_files.append((local_path, snapshot))
            _backfill_dataset(con, dataset, snapshots, local_files, run_id=run_id, ingested_at=started_at)

        _refresh_legacy_compatible_tables(con, run_id=run_id, refreshed_at=started_at)
        _refresh_source_summary(con)
        con.execute(
            """
            CREATE OR REPLACE TABLE lead_etl_history_source_summary AS
            SELECT source_path, source_format, source_table, count(*) AS records
            FROM lead_etl_history_records
            GROUP BY 1, 2, 3
            ORDER BY records DESC, source_path
            """
        )
        finished_at = datetime.now(timezone.utc)
        total_records = int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
        con.execute(
            """
            INSERT INTO lead_etl_pipeline_runs
                (run_id, source_uri, started_at, finished_at, status, discovered_datasets, changed_datasets, total_records)
            VALUES (?, ?, ?, ?, 'history_backfill', ?, ?, ?)
            """,
            [run_id, s3_uri, started_at, finished_at, len(grouped), len(grouped), total_records],
        )
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception:
        with contextlib.suppress(Exception):
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _backfill_dataset(
    con: Any,
    dataset: str,
    snapshots: list[S3Snapshot],
    local_files: list[tuple[Path, S3Snapshot]],
    *,
    run_id: str,
    ingested_at: datetime,
) -> None:
    raw_table = "__lead_etl_history_raw"
    object_table = "__lead_etl_history_files"
    for table in (raw_table, object_table):
        con.execute(f"DROP TABLE IF EXISTS {quote_identifier(table)}")
    con.execute(
        f"CREATE TEMP TABLE {quote_identifier(object_table)} "
        "(filename VARCHAR, source_path VARCHAR, source_last_modified TIMESTAMPTZ)"
    )
    file_names = [str(path.resolve()) for path, _snapshot in local_files]
    con.executemany(
        f"INSERT INTO {quote_identifier(object_table)} VALUES (?, ?, ?)",
        [
            (str(path.resolve()), snapshot.uri, snapshot.last_modified)
            for path, snapshot in local_files
        ],
    )
    con.execute(
        f"""
        CREATE TEMP TABLE {quote_identifier(raw_table)} AS
        SELECT parquet.* EXCLUDE (filename), files.source_path AS _source_path,
               files.source_last_modified AS _source_last_modified
        FROM read_parquet(?, union_by_name=true, filename=true) AS parquet
        JOIN {quote_identifier(object_table)} AS files USING (filename)
        """,
        [file_names],
    )
    business_columns = _business_columns(con, raw_table)
    mode = _classify_ingestion_mode(con, raw_table, snapshots, business_columns)
    order_columns = _infer_order_columns(business_columns) if mode == "incremental" else []
    key_columns = (
        _infer_key_columns(
            con,
            raw_table,
            dataset,
            snapshots,
            business_columns,
            order_columns=order_columns,
        )
        if mode == "incremental"
        else []
    )
    if mode == "incremental" and not key_columns:
        raise RuntimeError(f"Could not infer a stable business key for incremental dataset {dataset}")

    history_table = history_table_name(dataset)
    hash_expression = _row_hash_expression(business_columns)
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(history_table)}")
    con.execute(
        f"""
        CREATE TABLE {quote_identifier(history_table)} AS
        SELECT * EXCLUDE (_lead_etl_rank)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY {hash_expression}
                ORDER BY _source_last_modified DESC, _source_path DESC
            ) AS _lead_etl_rank
            FROM {quote_identifier(raw_table)}
        )
        WHERE _lead_etl_rank = 1
        """
    )

    current_table = physical_table_name(dataset)
    con.execute(f"DROP TABLE IF EXISTS {quote_identifier(current_table)}")
    if mode == "snapshot":
        con.execute(
            f"""
            CREATE TABLE {quote_identifier(current_table)} AS
            SELECT * FROM {quote_identifier(raw_table)} WHERE _source_path = ?
            """,
            [snapshots[-1].uri],
        )
    else:
        keys = ", ".join(quote_identifier(column) for column in key_columns)
        order_by = _dedupe_order_expression(order_columns)
        con.execute(
            f"""
            CREATE TABLE {quote_identifier(current_table)} AS
            SELECT * EXCLUDE (_lead_etl_rank)
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY {keys}
                    ORDER BY _source_last_modified DESC, {order_by}, _source_path DESC
                ) AS _lead_etl_rank
                FROM {quote_identifier(raw_table)}
            )
            WHERE _lead_etl_rank = 1
            """
        )

    _refresh_unified_dataset(con, dataset, current_table, ingested_at)
    _refresh_history_dataset(con, dataset, history_table, ingested_at)
    row_count = int(con.execute(f"SELECT count(*) FROM {quote_identifier(current_table)}").fetchone()[0])
    _write_manifest(
        con,
        snapshots[-1],
        row_count=row_count,
        ingested_at=ingested_at,
        run_id=run_id,
        mode=mode,
        key_columns=key_columns,
        order_columns=order_columns,
    )
    con.execute("DELETE FROM lead_etl_history_objects WHERE dataset = ?", [dataset])
    con.executemany(
        """
        INSERT INTO lead_etl_history_objects
            (dataset, s3_uri, s3_key, etag, size_bytes, source_last_modified, processed_at, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dataset,
                snapshot.uri,
                snapshot.key,
                snapshot.etag,
                snapshot.size,
                snapshot.last_modified,
                ingested_at,
                run_id,
            )
            for snapshot in snapshots
        ],
    )


def _classify_ingestion_mode(
    con: Any,
    raw_table: str,
    snapshots: list[S3Snapshot],
    business_columns: list[str],
) -> str:
    if len(snapshots) < 2:
        return "snapshot"
    previous, latest = snapshots[-2], snapshots[-1]
    select_columns = ", ".join(quote_identifier(column) for column in business_columns)
    previous_rows = int(
        con.execute(
            f"SELECT count(*) FROM {quote_identifier(raw_table)} WHERE _source_path = ?", [previous.uri]
        ).fetchone()[0]
    )
    latest_rows = int(
        con.execute(
            f"SELECT count(*) FROM {quote_identifier(raw_table)} WHERE _source_path = ?", [latest.uri]
        ).fetchone()[0]
    )
    overlap = int(
        con.execute(
            f"""
            SELECT count(*) FROM (
                SELECT {select_columns} FROM {quote_identifier(raw_table)} WHERE _source_path = ?
                INTERSECT
                SELECT {select_columns} FROM {quote_identifier(raw_table)} WHERE _source_path = ?
            )
            """,
            [previous.uri, latest.uri],
        ).fetchone()[0]
    )
    overlap_ratio = overlap / max(min(previous_rows, latest_rows), 1)
    return "snapshot" if overlap_ratio >= 0.8 else "incremental"


def _infer_key_columns(
    con: Any,
    raw_table: str,
    dataset: str,
    snapshots: list[S3Snapshot],
    business_columns: list[str],
    *,
    order_columns: list[str],
) -> list[str]:
    id_columns = [
        column for column in business_columns if column.lower() == "id" or column.lower().endswith("_id")
    ]
    if not id_columns:
        return []
    base = dataset.rsplit("/", 1)[-1].lower()
    if base.startswith("fct_"):
        base = base[4:]
    preferred = ["id", f"{base}_id"]
    largest = max(
        snapshots,
        key=lambda snapshot: int(
            con.execute(
                f"SELECT count(*) FROM {quote_identifier(raw_table)} WHERE _source_path = ?", [snapshot.uri]
            ).fetchone()[0]
        ),
    )
    for candidate in preferred:
        actual = next((column for column in id_columns if column.lower() == candidate), None)
        if actual and _keys_identify_unique_rows(
            con, raw_table, largest.uri, [actual], business_columns
        ):
            return [actual]
    complete_ids = [
        column
        for column in id_columns
        if _column_is_complete(con, raw_table, largest.uri, column)
    ]
    if complete_ids and _keys_identify_unique_rows(
        con, raw_table, largest.uri, complete_ids, business_columns
    ):
        return complete_ids
    if complete_ids and order_columns and _key_coverage(
        con, raw_table, largest.uri, complete_ids, business_columns
    ) >= 0.99:
        return complete_ids
    return []


def _infer_order_columns(business_columns: list[str]) -> list[str]:
    preferred_tokens = (
        "updated_at",
        "last_updated",
        "last_update",
        "modified_at",
        "date_complete",
        "completed_at",
    )
    lowered = {column.lower(): column for column in business_columns}
    return [lowered[token] for token in preferred_tokens if token in lowered]


def _key_coverage(
    con: Any,
    table: str,
    source_path: str,
    key_columns: list[str],
    business_columns: list[str],
) -> float:
    keys = ", ".join(quote_identifier(column) for column in key_columns)
    row_hash = _row_hash_expression(business_columns)
    distinct_keys, distinct_rows = con.execute(
        f"""
        SELECT count(DISTINCT ({keys})), count(DISTINCT {row_hash})
        FROM {quote_identifier(table)} WHERE _source_path = ?
        """,
        [source_path],
    ).fetchone()
    return int(distinct_keys) / max(int(distinct_rows), 1)


def _column_is_complete(con: Any, table: str, source_path: str, column: str) -> bool:
    row = con.execute(
        f"""
        SELECT count(*), count({quote_identifier(column)})
        FROM {quote_identifier(table)} WHERE _source_path = ?
        """,
        [source_path],
    ).fetchone()
    return int(row[0]) > 0 and int(row[0]) == int(row[1])


def _keys_identify_unique_rows(
    con: Any,
    table: str,
    source_path: str,
    key_columns: list[str],
    business_columns: list[str],
) -> bool:
    keys = ", ".join(quote_identifier(column) for column in key_columns)
    row_hash = _row_hash_expression(business_columns)
    row = con.execute(
        f"""
        SELECT count(DISTINCT ({keys})) AS distinct_keys,
               count(DISTINCT {row_hash}) AS distinct_rows
        FROM {quote_identifier(table)} WHERE _source_path = ?
        """,
        [source_path],
    ).fetchone()
    return int(row[0]) > 0 and int(row[0]) == int(row[1])


def _business_columns(con: Any, table_name: str) -> list[str]:
    return [
        str(row[1])
        for row in con.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
        if str(row[1]) not in {"_source_path", "_source_last_modified"}
    ]


def _row_hash_expression(columns: list[str]) -> str:
    if not columns:
        raise ValueError("Cannot hash a row without business columns")
    return "hash(" + ", ".join(quote_identifier(column) for column in columns) + ")"


def _table_exists(con: Any, table_name: str) -> bool:
    return bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",
            [table_name],
        ).fetchone()[0]
    )


def history_table_name(dataset: str) -> str:
    current = physical_table_name(dataset)
    return current.replace("lead_etl_", "lead_etl_history_", 1)


def _decode_key_columns(value: Any) -> list[str]:
    try:
        decoded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _dedupe_order_expression(columns: list[str]) -> str:
    if not columns:
        return "NULL DESC"
    return ", ".join(f"{quote_identifier(column)} DESC NULLS LAST" for column in columns)


def _validate_physical_table_names(grouped: dict[str, list[S3Snapshot]]) -> None:
    by_table: dict[str, str] = {}
    for dataset in grouped:
        table = physical_table_name(dataset)
        previous = by_table.get(table)
        if previous and previous != dataset:
            raise RuntimeError(f"Dataset table-name collision: {previous!r} and {dataset!r} -> {table!r}")
        by_table[table] = dataset


def _remove_dataset(con: Any, dataset: str) -> None:
    row = con.execute(
        "SELECT physical_table FROM lead_etl_object_manifest WHERE dataset = ?", [dataset]
    ).fetchone()
    if row:
        con.execute(f"DROP TABLE IF EXISTS {quote_identifier(str(row[0]))}")
    con.execute("DELETE FROM unified_records WHERE source_table = ?", [dataset])
    con.execute("DELETE FROM lead_etl_object_manifest WHERE dataset = ?", [dataset])


def _refresh_source_summary(con: Any, *, datasets: list[str] | None = None) -> None:
    if datasets and _table_exists(con, "source_summary"):
        unique_datasets = sorted(set(datasets))
        placeholders = ", ".join("?" for _dataset in unique_datasets)
        con.execute(
            f"DELETE FROM source_summary WHERE source_table IN ({placeholders})",
            unique_datasets,
        )
        con.execute(
            f"""
            INSERT INTO source_summary
            SELECT source_path, source_format, source_table, count(*) AS records
            FROM unified_records
            WHERE source_table IN ({placeholders})
            GROUP BY 1, 2, 3
            """,
            unique_datasets,
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TABLE source_summary AS
            SELECT source_path, source_format, source_table, count(*) AS records
            FROM unified_records
            GROUP BY 1, 2, 3
            ORDER BY records DESC, source_path
            """
        )
    if _table_exists(con, "lead_etl_history_records"):
        con.execute(
            """
            CREATE OR REPLACE TABLE lead_etl_history_source_summary AS
            SELECT source_path, source_format, source_table, count(*) AS records
            FROM lead_etl_history_records
            GROUP BY 1, 2, 3
            ORDER BY records DESC, source_path
            """
        )


def _is_managed_database(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import duckdb

        con = duckdb.connect(str(path), read_only=True)
        try:
            return bool(
                con.execute(
                    """
                    SELECT count(*)
                    FROM information_schema.tables
                    WHERE table_schema = 'main' AND table_name = 'lead_etl_object_manifest'
                    """
                ).fetchone()[0]
            )
        finally:
            con.close()
    except Exception:
        return False


def _current_snapshots(path: Path) -> dict[str, tuple[str, str, int]]:
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = con.execute("SELECT dataset, s3_key, etag, size_bytes FROM lead_etl_object_manifest").fetchall()
        return {str(dataset): (str(key), str(etag), int(size)) for dataset, key, etag, size in rows}
    finally:
        con.close()


def _total_records(path: Path) -> int:
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        return int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
    finally:
        con.close()


def _publish_database(work_path: Path, database_path: Path, *, preserve_existing: bool) -> Path | None:
    backup_path: Path | None = None
    if preserve_existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = database_path.with_suffix(database_path.suffix + f".legacy-{stamp}.bak")
        os.replace(database_path, backup_path)
    try:
        os.replace(work_path, database_path)
    except Exception:
        if backup_path and backup_path.exists() and not database_path.exists():
            os.replace(backup_path, database_path)
        raise
    return backup_path


def _publish_replacement_with_label(work_path: Path, database_path: Path, *, label: str) -> Path | None:
    backup_path: Path | None = None
    if database_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = database_path.with_suffix(database_path.suffix + f".{label}-{stamp}.bak")
        os.replace(database_path, backup_path)
    try:
        os.replace(work_path, database_path)
    except Exception:
        if backup_path and backup_path.exists() and not database_path.exists():
            os.replace(backup_path, database_path)
        raise
    return backup_path


def _validate_legacy_base(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy DuckDB database not found: {path}")
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
        required_tables = {"unified_records", "source_summary"}
        missing = sorted(required_tables - tables)
        if missing:
            raise RuntimeError(f"Legacy database is missing required tables: {', '.join(missing)}")
        if "lead_etl_object_manifest" in tables:
            raise RuntimeError(f"Legacy base already contains lead-etl metadata: {path}")
        columns = {
            str(row[1]) for row in con.execute("PRAGMA table_info('unified_records')").fetchall()
        }
        required_columns = {
            "source_path",
            "source_format",
            "source_table",
            "record_index",
            "payload_json",
            "ingested_at",
        }
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise RuntimeError(
                "Legacy unified_records is missing required columns: " + ", ".join(missing_columns)
            )
        return int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
    finally:
        con.close()


def _lead_etl_counts(path: Path) -> tuple[int, int, int]:
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        combined = int(con.execute("SELECT count(*) FROM unified_records").fetchone()[0])
        current = int(
            con.execute("SELECT coalesce(sum(row_count), 0) FROM lead_etl_object_manifest").fetchone()[0]
        )
        history = int(con.execute("SELECT count(*) FROM lead_etl_history_records").fetchone()[0])
        return combined, current, history
    finally:
        con.close()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@contextlib.contextmanager
def pipeline_lock(path: Path, *, timeout_seconds: float = 0) -> Iterator[None]:
    deadline = time.monotonic() + max(timeout_seconds, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        while True:
            try:
                _lock_file(handle)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Another ingestion run holds {path}")
                time.sleep(min(1.0, max(deadline - time.monotonic(), 0.05)))
        yield
    finally:
        with contextlib.suppress(OSError):
            _unlock_file(handle)
        handle.close()


def _lock_file(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - deployment is Linux
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - deployment is Linux
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
