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

from dashboard_agent.config import Settings, _split_extensions
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.graphify_ingest import ingest_s3_with_graphify
from dashboard_agent.s3_source import S3JsonSource


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
    parser = argparse.ArgumentParser(description="Temporary S3-to-graph ingest runner.")
    parser.add_argument("--s3-uri", help="S3 URI to ingest, e.g. s3://edx-nectec-demo/parquet/fact_student_course")
    parser.add_argument("--extensions", help="Comma-separated extensions, e.g. .json,.parquet")
    parser.add_argument("--graph-path", help="Output path for the app graph JSON")
    parser.add_argument("--graphify-output-dir", help="Output directory for Graphify graph.json")
    parser.add_argument("--graphify-bin", help="Graphify executable name or path")
    parser.add_argument("--no-graphify", action="store_true", help="Skip Graphify export")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Import S3 object metadata without downloading object bodies.",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N objects")
    parser.add_argument(
        "--sample-json-bytes",
        type=int,
        default=0,
        help="Read this many leading bytes from JSON objects and store parsed sample records.",
    )
    parser.add_argument("--sample-record-limit", type=int, default=20, help="Maximum JSON sample records per object")
    parser.add_argument("--sample-json-ranges", type=int, default=1, help="Number of S3 byte ranges to sample per JSON object")
    parser.add_argument(
        "--sample-key-contains",
        default="",
        help="Only attach JSON content samples to keys containing this text.",
    )
    parser.add_argument(
        "--aggregate-json-lines",
        action="store_true",
        help="Stream matching JSON-lines objects and store full aggregate counts.",
    )
    parser.add_argument(
        "--aggregate-key-contains",
        default="",
        help="Only full-scan keys containing this text.",
    )
    parser.add_argument(
        "--aggregate-progress-seconds",
        type=float,
        default=10.0,
        help="Print full-scan byte/record progress every N seconds.",
    )
    parser.add_argument("--search", default="", help="Optional graph search query after ingest")
    parser.add_argument("--limit", type=int, default=5, help="Search result limit")
    return parser.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()
    settings = Settings.from_env()

    s3_uri = args.s3_uri or settings.s3_data_uri
    extensions = _split_extensions(args.extensions) if args.extensions else settings.s3_include_extensions
    graph_path = Path(args.graph_path) if args.graph_path else settings.graph_path
    graphify_output_dir = (
        Path(args.graphify_output_dir) if args.graphify_output_dir else settings.graphify_output_dir
    )
    graphify_bin = args.graphify_bin or settings.graphify_bin
    graphify_enabled = settings.graphify_enabled and not args.no_graphify

    source = S3JsonSource(
        s3_uri,
        region=settings.aws_region,
        max_object_bytes=settings.graph_max_object_bytes,
        include_extensions=extensions,
    )
    if args.metadata_only:
        source.load = lambda: _with_progress(  # type: ignore[method-assign]
            source.load_objects(
                metadata_only=True,
                sample_json_bytes=args.sample_json_bytes,
                sample_record_limit=args.sample_record_limit,
                sample_json_ranges=args.sample_json_ranges,
                sample_key_contains=args.sample_key_contains,
                aggregate_json_lines=args.aggregate_json_lines,
                aggregate_key_contains=args.aggregate_key_contains,
                aggregate_progress_seconds=args.aggregate_progress_seconds,
            ),
            every=args.progress_every,
        )
    store = JsonGraphStore(graph_path, load_existing=False)
    status = ingest_s3_with_graphify(
        source,
        store,
        graphify_output_dir=graphify_output_dir,
        graphify_bin=graphify_bin,
        graphify_enabled=graphify_enabled,
    )

    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    if args.search:
        print(json.dumps(store.search(args.search, limit=args.limit), ensure_ascii=False, indent=2, default=str))
    return 0


def _with_progress(objects, *, every: int):
    for count, item in enumerate(objects, start=1):
        if every > 0 and count % every == 0:
            print(f"loaded {count} S3 objects...", file=sys.stderr, flush=True)
        yield item


if __name__ == "__main__":
    raise SystemExit(main())
