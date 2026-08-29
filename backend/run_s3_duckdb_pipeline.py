from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_agent.duckdb_ingest import ingest_duckdb_database
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_duckdb_pipeline import initialize_legacy_base, run_history_backfill, run_pipeline


DEFAULT_DATABASE = ROOT.parent / "data" / "_warehouse" / "dashboard_agent.duckdb"
DEFAULT_GRAPH = ROOT / "data" / "s3-json-graph.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the newest Parquet snapshot per S3 dataset into DuckDB.")
    parser.add_argument("--s3-uri", default=os.getenv("S3_DATA_URI", "s3://lead-etl"))
    parser.add_argument("--database", type=Path, default=Path(os.getenv("DUCKDB_PATH", str(DEFAULT_DATABASE))))
    parser.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    parser.add_argument("--lock-timeout-seconds", type=float, default=0)
    parser.add_argument("--no-backup", action="store_true", help="Do not preserve an unmanaged database on first migration.")
    parser.add_argument("--no-graph-refresh", action="store_true")
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Atomically rebuild the graph from the current DuckDB warehouse without ingesting S3.",
    )
    parser.add_argument(
        "--legacy-base",
        type=Path,
        help="Copy this legacy warehouse once, then merge all S3 history into it before publishing.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Apply an hourly update transaction directly to a managed warehouse (avoids copying a large database).",
    )
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Ingest every historical S3 Parquet object, deduplicate row versions, and rebuild complete current tables.",
    )
    parser.add_argument("--graph-path", type=Path, default=Path(os.getenv("GRAPH_PATH", str(DEFAULT_GRAPH))))
    parser.add_argument("--graph-sample-records", type=int, default=3)
    return parser.parse_args()


def _prune_stale_work_files(database_path: Path, *, max_age_seconds: float = 86400) -> list[str]:
    """Remove leftover *.tmp work copies older than a day; returns removed paths."""

    import time as _time

    removed: list[str] = []
    cutoff = _time.time() - max_age_seconds
    pattern = database_path.with_suffix(database_path.suffix + ".*.tmp")
    for candidate in database_path.parent.glob(pattern.name):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed.append(candidate.name)
        except OSError:
            continue
    return removed



def _prune_database_backups(database_path: Path, *, keep: int = 3, max_age_days: float = 14) -> list[str]:
    """Delete warehouse *.bak copies that are both stale and outside the newest keep set."""

    import time as _time

    cutoff = _time.time() - max_age_days * 86_400
    backups = []
    for candidate in database_path.parent.glob(f"{database_path.name}*.bak*"):
        try:
            if candidate.is_file():
                backups.append((candidate.stat().st_mtime, candidate))
        except OSError:
            continue
    backups.sort(key=lambda item: (-item[0], str(item[1])))
    removed = []
    for index, (mtime, candidate) in enumerate(backups):
        if index < keep:
            continue
        if mtime <= cutoff:
            try:
                candidate.unlink()
                removed.append(candidate.name)
            except OSError:
                continue
    return removed


def _warehouse_backup_inventory(database_path: Path) -> list[dict]:
    inventory = []
    for candidate in sorted(database_path.parent.glob(f"{database_path.name}*.bak")):
        try:
            stat = candidate.stat()
            inventory.append({"file": candidate.name, "size_bytes": stat.st_size})
        except OSError:
            continue
    return inventory


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()
    if args.graph_only:
        graph_status = _refresh_graph(args.graph_path, args.database, args.graph_sample_records)
        print(
            json.dumps(
                {
                    "status": "graph_refreshed",
                    "database": str(args.database),
                    "graph": graph_status,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.legacy_base:
        if not args.backfill_history:
            raise SystemExit("--legacy-base requires --backfill-history")
        result = initialize_legacy_base(
            args.legacy_base,
            args.database,
            args.s3_uri,
            region=args.region,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
        graph_status = None
        if not args.no_graph_refresh:
            graph_status = _refresh_graph(args.graph_path, result.database_path, args.graph_sample_records)
        print(
            json.dumps(
                {
                    "status": "legacy_base_initialized",
                    "source": args.s3_uri,
                    "database": str(result.database_path),
                    "datasets": result.datasets,
                    "objects": result.objects,
                    "legacyRecords": result.legacy_records,
                    "newCurrentRecords": result.new_current_records,
                    "combinedRecords": result.combined_records,
                    "historyRecords": result.history_records,
                    "replacedDatabaseBackup": (
                        str(result.replaced_database_backup) if result.replaced_database_backup else None
                    ),
                    "graph": graph_status,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.backfill_history:
        result = run_history_backfill(
            args.s3_uri,
            args.database,
            region=args.region,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
        graph_status = None
        if not args.no_graph_refresh:
            graph_status = _refresh_graph(args.graph_path, result.database_path, args.graph_sample_records)
        print(
            json.dumps(
                {
                    "status": "history_backfilled",
                    "source": args.s3_uri,
                    "database": str(result.database_path),
                    "datasets": result.datasets,
                    "objects": result.objects,
                    "currentRecords": result.current_records,
                    "historyRecords": result.history_records,
                    "graph": graph_status,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    result = run_pipeline(
        args.s3_uri,
        args.database,
        region=args.region,
        lock_timeout_seconds=args.lock_timeout_seconds,
        preserve_unmanaged_database=not args.no_backup,
        in_place=args.in_place,
    )
    pruned = _prune_stale_work_files(args.database)
    pruned_backups = _prune_database_backups(args.database)
    graph_status = None
    if not args.no_graph_refresh and result.status == "updated":
        graph_status = _refresh_graph(args.graph_path, result.database_path, args.graph_sample_records)
    print(
        json.dumps(
            {
                "status": result.status,
                "source": args.s3_uri,
                "database": str(args.database),
                "backup": str(result.backup_path) if result.backup_path else None,
                "discoveredDatasets": result.discovered_datasets,
                "changedDatasets": result.changed_datasets,
                "totalRecords": result.total_records,
                "prunedWorkFiles": pruned,
                "prunedBackups": pruned_backups,
                "warehouseBackups": _warehouse_backup_inventory(args.database),
                "graph": graph_status,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _refresh_graph(graph_path: Path, database_path: Path, sample_records: int) -> dict:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = graph_path.with_suffix(graph_path.suffix + ".tmp")
    temporary_index = temporary_path.with_suffix(temporary_path.suffix + ".search.json")
    for path in (temporary_path, temporary_index):
        if path.exists():
            path.unlink()
    store = JsonGraphStore(temporary_path, load_existing=False)
    try:
        status = ingest_duckdb_database(
            store,
            database_path,
            table="unified_records",
            sample_records_per_source=sample_records,
        )
        os.replace(temporary_path, graph_path)
        final_index = graph_path.with_suffix(graph_path.suffix + ".search.json")
        if temporary_index.exists():
            os.replace(temporary_index, final_index)
        return status
    finally:
        for path in (temporary_path, temporary_index):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
