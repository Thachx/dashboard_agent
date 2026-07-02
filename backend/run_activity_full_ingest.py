"""Run a full S3 JSON-lines aggregate ingest into the local graph store.

By default this scans every JSON object under S3_DATA_URI / --s3-uri. Existing
successful full-scan aggregates are reused when the S3 object's ETag and size
still match the graph, so repeated runs only stream new or changed files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.duckdb_ingest import ingest_duckdb_database
from dashboard_agent.graphify_ingest import ingest_s3_with_graphify
from dashboard_agent.s3_source import S3JsonSource


DEFAULT_ENV_PATHS = (ROOT / ".env", ROOT.parent / "agent-chat-ui" / ".env")
DEFAULT_GRAPH_PATH = ROOT / "data" / "s3-json-graph.json"
DEFAULT_GRAPHIFY_OUTPUT = ROOT / "data" / "graphify-out"
DEFAULT_AGGREGATE_CACHE = ROOT / "data" / "full-scan-aggregates.json"
DEFAULT_DUCKDB_PATH = ROOT.parent / "data" / "_warehouse" / "dashboard_agent.duckdb"
DEFAULT_S3_URI = "s3://edx-nectec-demo"
AGGREGATE_FIELDS = (
    "full_scan_status",
    "full_scan_error",
    "full_record_count",
    "full_counts_json",
    "full_time_buckets_json",
    "full_distinct_time_buckets_json",
    "full_user_time_buckets_json",
)
AGGREGATE_STATE_FIELDS = (
    "full_scan_offset",
    "full_parser",
    "_full_counts_state_json",
    "_full_time_buckets_state_json",
    "_full_distinct_time_buckets_state_json",
    "_full_user_time_buckets_state_json",
)
REUSABLE_FULL_SCAN_STATUSES = {"ok", "skipped"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-scan all S3 JSON/JSON-lines objects into the dashboard graph. "
            "Already scanned unchanged objects are reused from the graph."
        )
    )
    parser.add_argument(
        "--source",
        choices=("s3", "duckdb"),
        default=os.getenv("FULL_INGEST_SOURCE", "s3"),
        help="Ingest source backend: s3 or duckdb.",
    )
    parser.add_argument("--s3-uri", help="S3 URI to scan. Defaults to S3_DATA_URI/.env or s3://edx-nectec-demo.")
    parser.add_argument(
        "--activity-key",
        default="",
        help=(
            "Optional substring filter for JSON object keys. Empty means scan all JSON objects. "
            "Example: ae-activity-data-stream.json"
        ),
    )
    parser.add_argument("--graph-path", default=str(DEFAULT_GRAPH_PATH), help="Output graph JSON path.")
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB_PATH), help="DuckDB database path when --source duckdb.")
    parser.add_argument("--duckdb-table", default="unified_records", help="DuckDB table to ingest when --source duckdb.")
    parser.add_argument("--duckdb-sample-records", type=int, default=3, help="Sample records per DuckDB source file/table.")
    parser.add_argument("--duckdb-max-sources", type=int, default=None, help="Optional limit on DuckDB sources for testing.")
    parser.add_argument("--graphify-output", default=str(DEFAULT_GRAPHIFY_OUTPUT), help="Graphify output directory.")
    parser.add_argument(
        "--aggregate-cache",
        default=str(DEFAULT_AGGREGATE_CACHE),
        help="Cache file for completed full-scan aggregates, used to skip unchanged files on later runs.",
    )
    parser.add_argument("--extensions", default=".json,.parquet", help="Comma-separated S3 extensions to index.")
    parser.add_argument("--sample-limit", type=int, default=200, help="Maximum preview records kept per JSON object.")
    parser.add_argument("--sample-bytes", type=int, default=262_144, help="Bytes per JSON preview range.")
    parser.add_argument("--sample-ranges", type=int, default=8, help="Preview ranges across each JSON object.")
    parser.add_argument("--progress-seconds", type=float, default=10.0, help="Seconds between full-scan progress lines.")
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore existing full-scan aggregates and stream every matching object again.",
    )
    return parser.parse_args()


def main() -> int:
    for env_path in DEFAULT_ENV_PATHS:
        load_env_file(env_path)

    args = parse_args()
    graph_path = Path(args.graph_path)
    if args.source == "duckdb":
        started = time.monotonic()
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        store = JsonGraphStore(graph_path, load_existing=False)
        status = ingest_duckdb_database(
            store,
            Path(args.duckdb_path),
            table=args.duckdb_table,
            sample_records_per_source=args.duckdb_sample_records,
            max_sources=args.duckdb_max_sources,
        )
        elapsed = time.monotonic() - started
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        print(f"duckdb ingest finished in {elapsed:,.1f}s", flush=True)
        return 0

    s3_uri = args.s3_uri or os.getenv("S3_DATA_URI") or DEFAULT_S3_URI
    aggregate_cache = Path(args.aggregate_cache)
    key_filter = args.activity_key.strip()

    source = S3JsonSource(
        s3_uri,
        region=os.getenv("AWS_REGION") or None,
        include_extensions=_split_extensions(args.extensions),
    )

    existing_aggregates: dict[str, dict[str, Any]] = {}
    incremental_seeds: dict[str, dict[str, Any]] = {}
    if not args.force_rescan:
        print(
            f"planning full scan for {s3_uri}: checking reusable aggregate cache...",
            flush=True,
        )
        existing_aggregates, incremental_seeds = _existing_aggregate_plan(
            graph_path,
            aggregate_cache,
            source,
            key_filter,
        )

    filter_label = key_filter or "<all json objects>"
    if existing_aggregates:
        print(
            f"full scan starting for {s3_uri} filter={filter_label}; "
            f"will reuse {len(existing_aggregates):,} unchanged aggregate(s), "
            f"incrementally update {len(incremental_seeds):,}",
            flush=True,
        )
    elif incremental_seeds:
        print(
            f"full scan starting for {s3_uri} filter={filter_label}; "
            f"will incrementally update {len(incremental_seeds):,} aggregate(s)",
            flush=True,
        )
    else:
        print(f"full scan starting for {s3_uri} filter={filter_label}; no reusable aggregates found", flush=True)

    started = time.monotonic()
    objects = source.load_objects(
        metadata_only=True,
        sample_json_bytes=args.sample_bytes,
        sample_record_limit=args.sample_limit,
        sample_json_ranges=args.sample_ranges,
        sample_key_contains=key_filter or None,
        aggregate_json_lines=True,
        aggregate_key_contains=key_filter or None,
        aggregate_progress_seconds=args.progress_seconds,
        aggregate_existing_values=existing_aggregates,
        aggregate_seed_values=incremental_seeds,
    )
    objects = _with_progress(objects, aggregate_cache=aggregate_cache)
    source.load = lambda: objects  # type: ignore[method-assign]

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    store = JsonGraphStore(graph_path, load_existing=False)
    status = ingest_s3_with_graphify(
        source,
        store,
        graphify_output_dir=Path(args.graphify_output),
        graphify_bin=os.getenv("GRAPHIFY_BIN", "graphify"),
        graphify_enabled=True,
    )
    status["reused_aggregates"] = len(existing_aggregates)
    status["incremental_aggregates"] = len(incremental_seeds)

    elapsed = time.monotonic() - started
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    print(f"full scan finished in {elapsed:,.1f}s", flush=True)
    return 0


def _with_progress(objects: Iterable[Any], *, aggregate_cache: Path) -> Iterable[Any]:
    for count, item in enumerate(objects, start=1):
        value = getattr(item, "value", {}) or {}
        key = getattr(item, "key", "") or value.get("key") or value.get("source_key") or "<unknown>"
        if value.get("aggregate_reused"):
            print(f"reused existing full scan for {key}", flush=True)
        elif value.get("full_scan_status"):
            records = value.get("full_record_count", 0)
            if value.get("full_scan_status") == "skipped":
                print(f"skipped full scan for {key}: {value.get('full_scan_error')}", flush=True)
            else:
                print(f"ingested full scan for {key}: {records:,} records", flush=True)
            if value.get("full_scan_status") in REUSABLE_FULL_SCAN_STATUSES:
                _save_completed_aggregate(aggregate_cache, key, value)
        elif count == 1 or count % 100 == 0:
            print(f"metadata objects processed: {count:,}", flush=True)
        yield item


def _split_extensions(raw: str) -> set[str]:
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return {item if item.startswith(".") else f".{item}" for item in values}


def _existing_aggregate_plan(
    graph_path: Path,
    aggregate_cache: Path,
    source: S3JsonSource,
    key_filter: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    states = _existing_aggregate_states(graph_path, key_filter)
    states.update(_cached_aggregate_states(aggregate_cache, key_filter))
    if not states:
        return {}, {}

    print(f"validating {len(states):,} cached aggregate object state(s) against S3...", flush=True)
    started = time.monotonic()
    last_progress = started
    unchanged: dict[str, dict[str, Any]] = {}
    incremental: dict[str, dict[str, Any]] = {}
    for index, (key, state) in enumerate(states.items(), start=1):
        current_state = _current_s3_object_state(source, key)
        if current_state is None:
            continue
        if _is_current(state, current_state):
            unchanged[key] = {field: state[field] for field in _reusable_aggregate_fields(state)}
        elif _can_incremental_update(state, current_state):
            incremental[key] = dict(state)
        now = time.monotonic()
        if now - last_progress >= 5 or index == len(states):
            print(
                "aggregate cache validation progress "
                f"{index:,}/{len(states):,}; reuse {len(unchanged):,}, incremental {len(incremental):,}",
                flush=True,
            )
            last_progress = now

    return unchanged, incremental


def _cached_aggregate_states(cache_path: Path, key_filter: str) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    states: dict[str, dict[str, Any]] = {}
    for key, state in payload.items():
        if not isinstance(key, str) or not isinstance(state, dict):
            continue
        if key_filter and key_filter not in key:
            continue
        if state.get("full_scan_status") not in REUSABLE_FULL_SCAN_STATUSES:
            continue
        if not state.get("etag") or state.get("size_bytes") is None:
            continue
        states[key] = dict(state)
    return states


def _reusable_aggregate_fields(state: dict[str, Any]) -> list[str]:
    fields = list(AGGREGATE_FIELDS)
    fields.extend(AGGREGATE_STATE_FIELDS)
    return [field for field in fields if field in state]


def _has_required_aggregate_fields(state: dict[str, Any]) -> bool:
    return (
        "full_counts_json" in state
        and "full_time_buckets_json" in state
        and (
            "full_distinct_time_buckets_json" in state
            or "full_user_time_buckets_json" in state
        )
    )


def _can_incremental_update(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_size = int(existing.get("full_scan_offset") or existing.get("size_bytes") or 0)
    current_size = int(current.get("size_bytes") or 0)
    if previous_size <= 0 or current_size <= previous_size:
        return False
    if existing.get("full_scan_status") != "ok":
        return False
    if not existing.get("full_record_count"):
        return False
    return bool(
        existing.get("_full_counts_state_json")
        and existing.get("_full_time_buckets_state_json")
        and (
            existing.get("_full_distinct_time_buckets_state_json")
            or existing.get("_full_user_time_buckets_state_json")
        )
    )


def _save_completed_aggregate(cache_path: Path, key: str, value: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    payload[key] = {
        "key": key,
        "etag": value.get("etag"),
        "size_bytes": value.get("size_bytes") or value.get("size"),
        **{field: value.get(field) for field in AGGREGATE_FIELDS if field in value},
    }
    for field in AGGREGATE_STATE_FIELDS:
        if field in value:
            payload[key][field] = value.get(field)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)


def _existing_aggregate_states(graph_path: Path, key_filter: str) -> dict[str, dict[str, Any]]:
    if not graph_path.exists():
        return {}

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return {}

    states: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        key = _node_key(node)
        if not key:
            continue
        if key_filter and key_filter not in key:
            continue

        state = states.setdefault(key, {"key": key})
        for source_name, dest_name in (
            ("etag", "etag"),
            ("size", "size_bytes"),
            ("size_bytes", "size_bytes"),
            ("last_modified", "last_modified"),
            ("s3_uri", "s3_uri"),
        ):
            if source_name in node and node[source_name] not in (None, ""):
                state[dest_name] = node[source_name]

        label = str(node.get("label") or node.get("name") or node.get("field") or "")
        value = node.get("value")
        if label in ("etag", "size_bytes") and value not in (None, ""):
            state[label] = value
        if label in AGGREGATE_FIELDS + AGGREGATE_STATE_FIELDS and value not in (None, ""):
            state[label] = value
        if node.get("field") in AGGREGATE_FIELDS + AGGREGATE_STATE_FIELDS and value not in (None, ""):
            state[str(node["field"])] = value

    return {
        key: state
        for key, state in states.items()
        if state.get("full_scan_status") in REUSABLE_FULL_SCAN_STATUSES
        and state.get("etag")
        and state.get("size_bytes") is not None
        and state.get("full_record_count") is not None
    }


def _node_key(node: dict[str, Any]) -> str | None:
    for field in ("key", "source_key", "object_key"):
        value = node.get(field)
        if isinstance(value, str) and value:
            return value

    s3_uri = node.get("s3_uri") or node.get("uri")
    if isinstance(s3_uri, str) and s3_uri.startswith("s3://"):
        return s3_uri.split("/", 3)[3] if s3_uri.count("/") >= 3 else ""

    node_id = str(node.get("id") or "")
    if node_id.startswith("object::"):
        return node_id[len("object::") :]
    if "::$." in node_id:
        return node_id.split("::$.", 1)[0]
    return None


def _current_s3_object_state(source: S3JsonSource, key: str) -> dict[str, Any] | None:
    try:
        import boto3

        client = boto3.client("s3", region_name=source.region)
        response = client.head_object(Bucket=source.bucket, Key=key)
    except Exception as exc:  # pragma: no cover - depends on live S3/IAM state
        print(f"could not validate existing aggregate for s3://{source.bucket}/{key}: {exc}", file=sys.stderr)
        return None

    return {
        "etag": str(response.get("ETag", "")).strip('"'),
        "size_bytes": int(response.get("ContentLength") or 0),
    }


def _is_current(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    return (
        _has_required_aggregate_fields(existing)
        and str(existing.get("etag") or "").strip('"') == str(current.get("etag") or "").strip('"')
        and int(existing.get("size_bytes") or 0) == int(current.get("size_bytes") or 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
