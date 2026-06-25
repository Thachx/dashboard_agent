from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_agent.config import Settings, _split_extensions
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.graphify_ingest import ingest_s3_with_graphify
from dashboard_agent.s3_source import JsonObject, S3JsonSource, parse_s3_uri


DEFAULT_ACTIVITY_KEY = "ae-activity-data-stream.json"


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
    parser = argparse.ArgumentParser(
        description=(
            "Import the S3 bucket graph and fully aggregate the large activity JSON-lines file. "
            "This streams the matched file from S3 and stores aggregate counts in the graph; "
            "it does not download the raw 11GB file to disk."
        )
    )
    parser.add_argument("--s3-uri", default="", help="S3 URI to ingest. Defaults to S3_DATA_URI/.env.")
    parser.add_argument("--activity-key", default=DEFAULT_ACTIVITY_KEY, help="Substring of the activity object key.")
    parser.add_argument("--extensions", default=".json,.parquet", help="Comma-separated S3 extensions to index.")
    parser.add_argument("--graph-path", default="", help="Output path for the app graph JSON.")
    parser.add_argument("--graphify-output-dir", default="", help="Graphify output directory, retained for status only.")
    parser.add_argument("--sample-json-bytes", type=int, default=262_144, help="Bytes per preview range.")
    parser.add_argument("--sample-json-ranges", type=int, default=8, help="Preview ranges across the activity file.")
    parser.add_argument("--sample-record-limit", type=int, default=200, help="Preview records kept for the table.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N S3 objects.")
    parser.add_argument(
        "--aggregate-progress-seconds",
        type=float,
        default=10.0,
        help="Print full-scan byte/record progress every N seconds.",
    )
    parser.add_argument("--search", default="eventCategory", help="Optional search query after ingest.")
    parser.add_argument("--limit", type=int, default=8, help="Search result limit.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the activity aggregate already exists.")
    return parser.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()
    settings = Settings.from_env()

    s3_uri = args.s3_uri or settings.s3_data_uri
    graph_path = Path(args.graph_path) if args.graph_path else settings.graph_path
    graphify_output_dir = (
        Path(args.graphify_output_dir) if args.graphify_output_dir else settings.graphify_output_dir
    )
    extensions = _split_extensions(args.extensions)

    existing = _existing_activity_state(graph_path, args.activity_key)
    if existing and not args.force:
        current = _current_s3_object_state(s3_uri, str(existing.get("key") or ""), settings.aws_region)
        if _is_current(existing, current):
            print(
                json.dumps(
                    {
                        "status": "skipped-existing",
                        "reason": "Activity aggregate already exists and S3 object is unchanged.",
                        "graph_path": str(graph_path),
                        "activity_key": existing.get("key"),
                        "etag": existing.get("etag"),
                        "size_bytes": existing.get("size_bytes"),
                        "full_record_count": existing.get("full_record_count"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

    source = S3JsonSource(
        s3_uri,
        region=settings.aws_region,
        max_object_bytes=settings.graph_max_object_bytes,
        include_extensions=extensions,
    )
    source.load = lambda: _with_progress(  # type: ignore[method-assign]
        source.load_objects(
            metadata_only=True,
            sample_json_bytes=args.sample_json_bytes,
            sample_record_limit=args.sample_record_limit,
            sample_json_ranges=args.sample_json_ranges,
            sample_key_contains=args.activity_key,
            aggregate_json_lines=True,
            aggregate_key_contains=args.activity_key,
            aggregate_progress_seconds=args.aggregate_progress_seconds,
        ),
        every=args.progress_every,
    )

    store = JsonGraphStore(graph_path, load_existing=False)
    status = ingest_s3_with_graphify(
        source,
        store,
        graphify_output_dir=graphify_output_dir,
        graphify_bin=settings.graphify_bin,
        graphify_enabled=False,
    )

    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    if args.search:
        print(json.dumps(store.search(args.search, limit=args.limit), ensure_ascii=False, indent=2, default=str))
    return 0


def _with_progress(objects: Iterable[JsonObject], *, every: int) -> Iterable[JsonObject]:
    for count, item in enumerate(objects, start=1):
        if every > 0 and count % every == 0:
            print(f"loaded {count} S3 objects...", file=sys.stderr, flush=True)
        if DEFAULT_ACTIVITY_KEY in item.key:
            print(f"streamed full aggregate for {item.key}", file=sys.stderr, flush=True)
        yield item


def _existing_activity_state(graph_path: Path, activity_key: str) -> dict[str, object] | None:
    if not graph_path.exists():
        return None
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        return None

    state: dict[str, object] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        path = str(node.get("path") or "")
        source = str(node.get("source") or path)
        if activity_key not in source and activity_key not in path:
            continue
        label = str(node.get("label") or "")
        if label in {
            "key",
            "s3_uri",
            "etag",
            "size_bytes",
            "full_scan_status",
            "full_record_count",
        }:
            state[label] = node.get("value")
        if node.get("type") == "dataset" or str(node.get("id") or "").startswith("object::"):
            state.setdefault("key", node.get("path") or node.get("label"))
            state.setdefault("etag", node.get("etag"))
            state.setdefault("size_bytes", node.get("size"))

    if state.get("full_scan_status") == "ok" and state.get("key"):
        return state
    return None


def _current_s3_object_state(s3_uri: str, key: str, region: str | None) -> dict[str, object] | None:
    if not key:
        return None
    try:
        import boto3
    except ImportError:
        return None
    bucket, prefix = parse_s3_uri(s3_uri)
    object_key = key
    if prefix and not object_key.startswith(prefix):
        object_key = f"{prefix.rstrip('/')}/{object_key.lstrip('/')}"
    try:
        response = boto3.client("s3", region_name=region).head_object(Bucket=bucket, Key=object_key)
    except Exception:
        return None
    return {
        "key": object_key,
        "etag": str(response.get("ETag", "")).strip('"'),
        "size_bytes": int(response.get("ContentLength", 0)),
    }


def _is_current(existing: dict[str, object], current: dict[str, object] | None) -> bool:
    if current is None:
        return False
    return (
        str(existing.get("etag") or "") == str(current.get("etag") or "")
        and int(existing.get("size_bytes") or 0) == int(current.get("size_bytes") or 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
