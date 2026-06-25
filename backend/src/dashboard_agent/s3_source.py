from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class JsonObject:
    key: str
    etag: str
    value: Any
    object_type: str = "json"
    size: int = 0


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class S3JsonSource:
    def __init__(
        self,
        uri: str,
        *,
        region: str | None = None,
        max_object_bytes: int = 10 * 1024 * 1024,
        include_extensions: tuple[str, ...] = (".json", ".parquet"),
    ):
        self.uri = uri
        self.bucket, self.prefix = parse_s3_uri(uri)
        self.region = region
        self.max_object_bytes = max_object_bytes
        self.include_extensions = tuple(extension.lower() for extension in include_extensions)

    def load(self) -> Iterable[JsonObject]:
        yield from self.load_objects(metadata_only=False)

    def load_objects(
        self,
        *,
        metadata_only: bool = False,
        sample_json_bytes: int = 0,
        sample_record_limit: int = 20,
        sample_json_ranges: int = 1,
        sample_key_contains: str = "",
        aggregate_json_lines: bool = False,
        aggregate_key_contains: str = "",
        aggregate_progress_seconds: float = 10.0,
    ) -> Iterable[JsonObject]:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 ingestion; install backend requirements") from exc

        client = boto3.client("s3", region_name=self.region)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                size = int(item.get("Size", 0))
                extension = _object_extension(key)
                if extension not in self.include_extensions:
                    continue
                etag = str(item.get("ETag", "")).strip('"')
                if metadata_only:
                    value = _metadata_value(self.bucket, key, extension, size, etag, item.get("LastModified"))
                    should_sample = not sample_key_contains or sample_key_contains in key
                    if extension == ".json" and sample_json_bytes > 0 and size > 0 and should_sample:
                        value.update(
                            _sample_json_content(
                                client,
                                self.bucket,
                                key,
                                sample_bytes=min(sample_json_bytes, size),
                                record_limit=sample_record_limit,
                                object_size=size,
                                sample_ranges=sample_json_ranges,
                            )
                        )
                    should_aggregate = aggregate_json_lines and (
                        not aggregate_key_contains or aggregate_key_contains in key
                    )
                    if extension == ".json" and should_aggregate:
                        value.update(
                            _aggregate_json_lines_content(
                                client,
                                self.bucket,
                                key,
                                object_size=size,
                                progress_seconds=aggregate_progress_seconds,
                            )
                        )
                    yield JsonObject(
                        key=key,
                        etag=etag,
                        value=value,
                        object_type=extension.lstrip("."),
                        size=size,
                    )
                    continue

                if extension == ".json":
                    if size > self.max_object_bytes:
                        value = _metadata_value(self.bucket, key, extension, size, etag, item.get("LastModified"))
                        value.update({"metadata_reason": "oversized-json"})
                        if sample_json_bytes > 0:
                            value.update(
                                _sample_json_content(
                                    client,
                                    self.bucket,
                                    key,
                                    sample_bytes=min(sample_json_bytes, size),
                                    record_limit=sample_record_limit,
                                    object_size=size,
                                    sample_ranges=sample_json_ranges,
                                )
                            )
                        yield JsonObject(
                            key=key,
                            etag=etag,
                            value=value,
                            object_type="json",
                            size=size,
                        )
                        continue
                    response = client.get_object(Bucket=self.bucket, Key=key)
                    try:
                        value = json.loads(response["Body"].read())
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"S3 object s3://{self.bucket}/{key} is not valid JSON") from exc
                    yield JsonObject(key=key, etag=etag, value=value, object_type="json", size=size)
                    continue

                yield JsonObject(
                    key=key,
                    etag=etag,
                    value=_metadata_value(self.bucket, key, extension, size, etag, item.get("LastModified")),
                    object_type=extension.lstrip("."),
                    size=size,
                )


def _object_extension(key: str) -> str:
    dot_index = key.rfind(".")
    return key[dot_index:].lower() if dot_index >= 0 else ""


def _metadata_value(bucket: str, key: str, extension: str, size: int, etag: str, last_modified: Any) -> dict[str, Any]:
    return {
        "s3_uri": f"s3://{bucket}/{key}",
        "bucket": bucket,
        "key": key,
        "object_type": extension.lstrip("."),
        "size_bytes": size,
        "etag": etag,
        "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else last_modified,
    }


def _sample_json_content(
    client: Any,
    bucket: str,
    key: str,
    *,
    sample_bytes: int,
    record_limit: int,
    object_size: int,
    sample_ranges: int = 1,
) -> dict[str, Any]:
    chunks: list[bytes] = []
    records: list[Any] = []
    starts = _range_starts(object_size, sample_bytes, sample_ranges)
    per_range_limit = max(1, (record_limit + max(len(starts), 1) - 1) // max(len(starts), 1))
    for start in starts:
        end = min(start + sample_bytes - 1, object_size - 1)
        response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
        body = response["Body"].read()
        chunks.append(body)
        text = body.decode("utf-8", errors="ignore")
        records.extend(_sample_json_records(text, record_limit=per_range_limit))
    records = records[:record_limit]
    fields = sorted({field for record in records if isinstance(record, dict) for field in record})
    preview = "\n".join(chunk.decode("utf-8", errors="ignore")[:1200] for chunk in chunks)
    return {
        "content_sample_type": "json-lines" if records else "text",
        "content_sample_bytes": sum(len(chunk) for chunk in chunks),
        "content_sample_ranges": len(chunks),
        "sample_record_count": len(records),
        "sample_fields": fields,
        "sample_records": records,
        "content_preview": preview[:4000],
    }


def _range_starts(object_size: int, sample_bytes: int, sample_ranges: int) -> list[int]:
    if object_size <= 0 or sample_bytes <= 0:
        return []
    ranges = max(1, sample_ranges)
    if ranges == 1 or object_size <= sample_bytes:
        return [0]
    max_start = max(object_size - sample_bytes, 0)
    return sorted({round(max_start * index / (ranges - 1)) for index in range(ranges)})


def _sample_json_records(text: str, *, record_limit: int) -> list[Any]:
    records: list[Any] = []
    if record_limit <= 0:
        return records
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in {"[", "]"}:
            continue
        if line.startswith("["):
            line = line[1:].strip()
        if line.endswith("]"):
            line = line[:-1].strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(records) >= record_limit:
            break
    return records


def _aggregate_json_lines_content(
    client: Any,
    bucket: str,
    key: str,
    *,
    object_size: int,
    progress_seconds: float,
) -> dict[str, Any]:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    record_count = 0
    bytes_read = 0
    started_at = time.monotonic()
    last_report_at = started_at
    counts: dict[str, dict[str, int]] = {
        "event": {},
        "eventCategory": {},
        "courseID": {},
        "appID": {},
        "userID": {},
    }
    time_buckets: dict[str, int] = {}

    print(f"full scan started for s3://{bucket}/{key} ({object_size:,} bytes)", file=sys.stderr, flush=True)
    for raw_line in body.iter_lines(chunk_size=1024 * 1024):
        if not raw_line:
            continue
        bytes_read += len(raw_line) + 1
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        record_count += 1
        for field in counts:
            value = record.get(field)
            if value is None or value == "":
                continue
            key_value = str(value)
            counts[field][key_value] = counts[field].get(key_value, 0) + 1
        timestamp = str(record.get("@timestamp") or record.get("timestamp") or "")
        if len(timestamp) >= 13:
            bucket_label = timestamp[:13]
            time_buckets[bucket_label] = time_buckets.get(bucket_label, 0) + 1
        now = time.monotonic()
        if progress_seconds > 0 and now - last_report_at >= progress_seconds:
            _print_aggregate_progress(
                key=key,
                bytes_read=bytes_read,
                object_size=object_size,
                record_count=record_count,
                elapsed=now - started_at,
            )
            last_report_at = now

    _print_aggregate_progress(
        key=key,
        bytes_read=bytes_read,
        object_size=object_size,
        record_count=record_count,
        elapsed=max(time.monotonic() - started_at, 0.001),
        done=True,
    )

    top_counts = {
        field: _top_count_rows(field_counts, limit=50)
        for field, field_counts in counts.items()
    }


def _print_aggregate_progress(
    *,
    key: str,
    bytes_read: int,
    object_size: int,
    record_count: int,
    elapsed: float,
    done: bool = False,
) -> None:
    percent = (bytes_read / object_size * 100) if object_size else 0
    mb_read = bytes_read / 1024 / 1024
    mb_total = object_size / 1024 / 1024 if object_size else 0
    mbps = mb_read / elapsed if elapsed > 0 else 0
    prefix = "full scan complete" if done else "full scan progress"
    print(
        f"{prefix} {key}: {percent:5.1f}% "
        f"({mb_read:,.1f}/{mb_total:,.1f} MB), "
        f"{record_count:,} records, {mbps:,.1f} MB/s",
        file=sys.stderr,
        flush=True,
    )
    return {
        "full_scan_status": "ok",
        "full_record_count": record_count,
        "full_counts_json": json.dumps(top_counts, ensure_ascii=False),
        "full_time_buckets_json": json.dumps(_top_time_rows(time_buckets, limit=200), ensure_ascii=False),
    }


def _top_count_rows(counts: dict[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "value": value}
        for label, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _top_time_rows(counts: dict[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "value": counts[label]}
        for label in sorted(counts)[:limit]
    ]
