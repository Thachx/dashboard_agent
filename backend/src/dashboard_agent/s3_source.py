from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


DEFAULT_AGGREGATE_MAX_FIELDS = 80
DEFAULT_AGGREGATE_MAX_VALUES_PER_FIELD = 1000
DEFAULT_AGGREGATE_MAX_DISTINCT_FIELDS = 40
DEFAULT_AGGREGATE_MAX_DISTINCT_VALUES_PER_BUCKET = 10000
DEFAULT_AGGREGATE_MAX_SCALAR_LENGTH = 200


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
        aggregate_existing_values: dict[str, dict[str, Any]] | None = None,
        aggregate_seed_values: dict[str, dict[str, Any]] | None = None,
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
                        existing_value = (aggregate_existing_values or {}).get(key)
                        if existing_value:
                            value.update(existing_value)
                            value["aggregate_reused"] = True
                        else:
                            seed_value = (aggregate_seed_values or {}).get(key)
                            value.update(
                                _aggregate_json_lines_content(
                                    client,
                                    self.bucket,
                                    key,
                                    object_size=size,
                                    progress_seconds=aggregate_progress_seconds,
                                    seed_value=seed_value,
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
    seed_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _aggregate_json_lines_content_resumable(
        client,
        bucket,
        key,
        object_size=object_size,
        progress_seconds=progress_seconds,
        seed_value=seed_value,
    )

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
    return {
        "full_scan_status": "ok",
        "full_record_count": record_count,
        "full_counts_json": json.dumps(top_counts, ensure_ascii=False),
        "full_time_buckets_json": json.dumps(_top_time_rows(time_buckets, limit=200), ensure_ascii=False),
    }


def _aggregate_json_lines_content_resumable(
    client: Any,
    bucket: str,
    key: str,
    *,
    object_size: int,
    progress_seconds: float,
    seed_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    head = _safe_head_object(client, bucket, key)
    etag = str(head.get("ETag") or "").strip('"')
    size = int(head.get("ContentLength") or object_size or 0)
    checkpoint_path = _aggregate_checkpoint_path(bucket, key, etag, size)
    checkpoint = _load_aggregate_checkpoint(checkpoint_path, bucket, key, etag, size)
    if int(checkpoint.get("next_start") or 0) <= 0 and seed_value:
        checkpoint = _checkpoint_from_seed(seed_value, bucket=bucket, key=key, etag=etag, size=size) or checkpoint

    counts = checkpoint["counts"]
    time_buckets = checkpoint["time_buckets"]
    distinct_time_buckets = checkpoint.get("distinct_time_buckets", {})
    record_count = int(checkpoint["record_count"])
    next_start = int(checkpoint["next_start"])
    tail = bytes.fromhex(str(checkpoint.get("tail_hex") or ""))
    started_at = time.monotonic()
    last_progress_at = started_at
    range_bytes = int(os.getenv("S3_FULL_SCAN_RANGE_BYTES", str(64 * 1024 * 1024)))
    range_bytes = max(1024 * 1024, range_bytes)
    tail_limit = int(os.getenv("S3_FULL_SCAN_MAX_TAIL_BYTES", str(8 * 1024 * 1024)))
    tail_limit = max(1024 * 1024, tail_limit)

    if next_start > 0:
        print(
            f"full scan resume/update s3://{bucket}/{key}: "
            f"{next_start / 1024 / 1024:,.1f}/{size / 1024 / 1024:,.1f} MB already checkpointed, "
            f"{record_count:,} records",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"full scan started for s3://{bucket}/{key} ({size:,} bytes)", file=sys.stderr, flush=True)

    while next_start < size:
        range_end = min(size - 1, next_start + range_bytes - 1)
        chunk = _read_s3_range(client, bucket, key, next_start, range_end)
        combined = tail + chunk
        lines = combined.split(b"\n")
        if range_end + 1 < size:
            tail = lines.pop()
        else:
            tail = b""
        if len(tail) > tail_limit:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return _aggregate_json_object_stream_content_resumable(
                client,
                bucket,
                key,
                object_size=size,
                progress_seconds=progress_seconds,
                seed_value=seed_value,
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue

            record_count += _update_aggregate_from_record(record, counts, time_buckets, distinct_time_buckets)

        next_start = range_end + 1
        _save_aggregate_checkpoint(
            checkpoint_path,
            bucket=bucket,
            key=key,
            etag=etag,
            size=size,
            next_start=next_start,
            tail=tail,
            record_count=record_count,
            counts=counts,
            time_buckets=time_buckets,
            distinct_time_buckets=distinct_time_buckets,
        )

        now = time.monotonic()
        if progress_seconds <= 0 or now - last_progress_at >= progress_seconds or next_start >= size:
            _print_aggregate_progress(
                key=key,
                bytes_read=next_start,
                object_size=size,
                record_count=record_count,
                elapsed=now - started_at,
                done=next_start >= size,
            )
            last_progress_at = now

    if tail.strip():
        try:
            record = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            record = None
        if isinstance(record, dict):
            record_count += _update_aggregate_from_record(record, counts, time_buckets, distinct_time_buckets)

    top_counts = {
        field: _top_count_rows(field_counts, limit=200)
        for field, field_counts in counts.items()
    }
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return {
        "full_scan_status": "ok",
        "full_record_count": record_count,
        "full_scan_offset": size,
        "full_counts_json": json.dumps(top_counts, ensure_ascii=False),
        "full_time_buckets_json": json.dumps(_top_time_rows(time_buckets, limit=200), ensure_ascii=False),
        "full_distinct_time_buckets_json": json.dumps(
            _top_distinct_time_rows_by_field(distinct_time_buckets, limit=200),
            ensure_ascii=False,
        ),
        "full_user_time_buckets_json": json.dumps(
            _top_legacy_user_time_rows(distinct_time_buckets, limit=200),
            ensure_ascii=False,
        ),
        "_full_counts_state_json": json.dumps(counts, ensure_ascii=False),
        "_full_time_buckets_state_json": json.dumps(time_buckets, ensure_ascii=False),
        "_full_distinct_time_buckets_state_json": json.dumps(distinct_time_buckets, ensure_ascii=False),
    }


def _aggregate_json_object_stream_content_resumable(
    client: Any,
    bucket: str,
    key: str,
    *,
    object_size: int,
    progress_seconds: float,
    seed_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    head = _safe_head_object(client, bucket, key)
    etag = str(head.get("ETag") or "").strip('"')
    size = int(head.get("ContentLength") or object_size or 0)
    checkpoint_path = _aggregate_checkpoint_path(bucket, f"{key}#json-object-stream", etag, size)
    checkpoint = _load_aggregate_checkpoint(checkpoint_path, bucket, f"{key}#json-object-stream", etag, size)
    if int(checkpoint.get("next_start") or 0) <= 0 and seed_value:
        checkpoint = _checkpoint_from_seed(seed_value, bucket=bucket, key=key, etag=etag, size=size) or checkpoint

    counts = checkpoint["counts"]
    time_buckets = checkpoint["time_buckets"]
    record_count = int(checkpoint["record_count"])
    next_start = int(checkpoint["next_start"])
    tail = bytes.fromhex(str(checkpoint.get("tail_hex") or ""))
    started_at = time.monotonic()
    last_progress_at = started_at
    range_bytes = int(os.getenv("S3_FULL_SCAN_RANGE_BYTES", str(64 * 1024 * 1024)))
    range_bytes = max(1024 * 1024, range_bytes)
    tail_limit = int(os.getenv("S3_FULL_SCAN_MAX_OBJECT_TAIL_BYTES", str(128 * 1024 * 1024)))
    tail_limit = max(8 * 1024 * 1024, tail_limit)

    print(
        f"full scan object-stream parser for s3://{bucket}/{key} ({size:,} bytes)",
        file=sys.stderr,
        flush=True,
    )

    while next_start < size:
        range_end = min(size - 1, next_start + range_bytes - 1)
        chunk = _read_s3_range(client, bucket, key, next_start, range_end)
        objects, tail = _extract_complete_json_objects(tail + chunk)
        if len(tail) > tail_limit:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return _skipped_full_scan_result(
                size=size,
                reason=f"JSON object larger than {tail_limit:,} bytes; cannot safely stream full content",
            )

        for raw_object in objects:
            try:
                record = json.loads(raw_object.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        record_count += _update_aggregate_from_record(record, counts, time_buckets, distinct_time_buckets)

        next_start = range_end + 1
        _save_aggregate_checkpoint(
            checkpoint_path,
            bucket=bucket,
            key=f"{key}#json-object-stream",
            etag=etag,
            size=size,
            next_start=next_start,
            tail=tail,
            record_count=record_count,
            counts=counts,
            time_buckets=time_buckets,
        )

        now = time.monotonic()
        if progress_seconds <= 0 or now - last_progress_at >= progress_seconds or next_start >= size:
            _print_aggregate_progress(
                key=key,
                bytes_read=next_start,
                object_size=size,
                record_count=record_count,
                elapsed=now - started_at,
                done=next_start >= size,
            )
            last_progress_at = now

    remaining = tail.strip().strip(b",").strip()
    if remaining and remaining not in (b"]", b"}"):
        objects, tail = _extract_complete_json_objects(remaining)
        for raw_object in objects:
            try:
                record = json.loads(raw_object.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            record_count += _update_aggregate_from_record(record, counts, time_buckets)

    top_counts = {
        field: _top_count_rows(field_counts, limit=200)
        for field, field_counts in counts.items()
    }
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return {
        "full_scan_status": "ok",
        "full_record_count": record_count,
        "full_scan_offset": size,
        "full_parser": "json_object_stream",
        "full_counts_json": json.dumps(top_counts, ensure_ascii=False),
        "full_time_buckets_json": json.dumps(_top_time_rows(time_buckets, limit=200), ensure_ascii=False),
        "full_distinct_time_buckets_json": json.dumps(
            _top_distinct_time_rows_by_field(distinct_time_buckets, limit=200),
            ensure_ascii=False,
        ),
        "_full_counts_state_json": json.dumps(counts, ensure_ascii=False),
        "_full_time_buckets_state_json": json.dumps(time_buckets, ensure_ascii=False),
        "_full_distinct_time_buckets_state_json": json.dumps(distinct_time_buckets, ensure_ascii=False),
    }


def _extract_complete_json_objects(buffer: bytes) -> tuple[list[bytes], bytes]:
    objects: list[bytes] = []
    index = 0
    length = len(buffer)
    while index < length:
        while index < length and buffer[index] in b" \r\n\t[,]":
            index += 1
        if index >= length:
            return objects, b""
        if buffer[index] != ord("{"):
            index += 1
            continue

        start = index
        depth = 0
        in_string = False
        escaped = False
        while index < length:
            byte = buffer[index]
            if in_string:
                if escaped:
                    escaped = False
                elif byte == ord("\\"):
                    escaped = True
                elif byte == ord('"'):
                    in_string = False
            else:
                if byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        objects.append(buffer[start : index + 1])
                        index += 1
                        break
            index += 1
        else:
            return objects, buffer[start:]
    return objects, b""


def _update_aggregate_from_record(
    record: Any,
    counts: dict[str, dict[str, int]],
    time_buckets: dict[str, int],
    distinct_time_buckets: dict[str, dict[str, dict[str, int]]] | None = None,
) -> int:
    if not isinstance(record, dict):
        return 0
    scalar_fields = list(_iter_aggregate_scalar_fields(record))
    for field, value in scalar_fields:
        _increment_aggregate_count(counts, field, value)

    bucket_key = _record_time_bucket(record)
    if bucket_key:
        time_buckets[bucket_key] = int(time_buckets.get(bucket_key, 0)) + 1
        if distinct_time_buckets is not None:
            for field, value in scalar_fields:
                if _is_time_like_field(field):
                    continue
                _add_distinct_time_value(distinct_time_buckets, field, bucket_key, value)
    return 1


def _aggregate_limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _iter_aggregate_scalar_fields(record: dict[str, Any]) -> Iterable[tuple[str, str]]:
    max_scalar_length = _aggregate_limit("S3_AGGREGATE_MAX_SCALAR_LENGTH", DEFAULT_AGGREGATE_MAX_SCALAR_LENGTH)
    for field, raw_value in record.items():
        if not isinstance(field, str) or not field:
            continue
        if isinstance(raw_value, bool):
            value = str(raw_value).lower()
        elif isinstance(raw_value, (int, float)):
            value = str(raw_value)
        elif isinstance(raw_value, str):
            value = raw_value.strip()
        else:
            continue
        if not value or len(value) > max_scalar_length:
            continue
        yield field, value


def _increment_aggregate_count(counts: dict[str, dict[str, int]], field: str, value: str) -> None:
    max_fields = _aggregate_limit("S3_AGGREGATE_MAX_FIELDS", DEFAULT_AGGREGATE_MAX_FIELDS)
    max_values = _aggregate_limit("S3_AGGREGATE_MAX_VALUES_PER_FIELD", DEFAULT_AGGREGATE_MAX_VALUES_PER_FIELD)
    if field not in counts and len(counts) >= max_fields:
        return
    field_counts = counts.setdefault(field, {})
    if value not in field_counts and len(field_counts) >= max_values:
        return
    field_counts[value] = int(field_counts.get(value, 0)) + 1


def _record_time_bucket(record: dict[str, Any]) -> str:
    candidates: list[tuple[int, str]] = []
    for field, value in _iter_aggregate_scalar_fields(record):
        bucket = _time_bucket_from_value(value)
        if not bucket:
            continue
        priority = 0 if _is_time_like_field(field) else 1
        candidates.append((priority, bucket))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][1]


def _is_time_like_field(field: str) -> bool:
    normalized = field.lower().replace("_", "").replace("-", "")
    return "timestamp" in normalized or normalized in {"time", "date", "datetime", "createdat", "updatedat"}


def _time_bucket_from_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 13 and re_match_iso_timestamp(text):
        return text[:13]
    return ""


def re_match_iso_timestamp(value: str) -> bool:
    return len(value) >= 10 and value[4:5] == "-" and value[7:8] == "-"


def _add_distinct_time_value(
    distinct_time_buckets: dict[str, dict[str, dict[str, int]]],
    field: str,
    bucket: str,
    value: str,
) -> None:
    max_fields = _aggregate_limit("S3_AGGREGATE_MAX_DISTINCT_FIELDS", DEFAULT_AGGREGATE_MAX_DISTINCT_FIELDS)
    max_values = _aggregate_limit(
        "S3_AGGREGATE_MAX_DISTINCT_VALUES_PER_BUCKET",
        DEFAULT_AGGREGATE_MAX_DISTINCT_VALUES_PER_BUCKET,
    )
    if field not in distinct_time_buckets and len(distinct_time_buckets) >= max_fields:
        return
    field_buckets = distinct_time_buckets.setdefault(field, {})
    bucket_values = field_buckets.setdefault(bucket, {})
    if value not in bucket_values and len(bucket_values) >= max_values:
        return
    bucket_values[value] = 1


def _safe_head_object(client: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        return dict(client.head_object(Bucket=bucket, Key=key))
    except Exception:
        return {}


def _skipped_full_scan_result(*, size: int, reason: str) -> dict[str, Any]:
    return {
        "full_scan_status": "skipped",
        "full_scan_error": reason,
        "full_record_count": 0,
        "full_scan_offset": size,
        "full_counts_json": "{}",
        "full_time_buckets_json": "[]",
        "full_distinct_time_buckets_json": "{}",
        "full_user_time_buckets_json": "[]",
        "_full_counts_state_json": "{}",
        "_full_time_buckets_state_json": "{}",
        "_full_distinct_time_buckets_state_json": "{}",
        "_full_user_time_buckets_state_json": "{}",
    }


def _checkpoint_from_seed(
    seed_value: dict[str, Any],
    *,
    bucket: str,
    key: str,
    etag: str,
    size: int,
) -> dict[str, Any] | None:
    previous_size = int(seed_value.get("full_scan_offset") or seed_value.get("size_bytes") or 0)
    previous_records = int(seed_value.get("full_record_count") or 0)
    if previous_size <= 0 or previous_size > size:
        return None

    counts = _json_string_mapping(seed_value.get("_full_counts_state_json"))
    time_buckets = _json_flat_mapping(seed_value.get("_full_time_buckets_state_json"))
    distinct_time_buckets = _json_distinct_time_mapping(seed_value.get("_full_distinct_time_buckets_state_json"))
    if not distinct_time_buckets:
        distinct_time_buckets = _legacy_user_time_to_distinct_mapping(
            _json_string_mapping(seed_value.get("_full_user_time_buckets_state_json"))
        )
    if not counts:
        counts = _rows_json_to_counts(seed_value.get("full_counts_json"))
    if not time_buckets:
        time_buckets = _rows_json_to_time_buckets(seed_value.get("full_time_buckets_json"))
    if not distinct_time_buckets and previous_records > 0:
        return None
    if not counts and previous_records > 0:
        return None

    return {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "size": size,
        "next_start": previous_size,
        "tail_hex": "",
        "record_count": previous_records,
        "counts": counts,
        "time_buckets": time_buckets,
        "distinct_time_buckets": distinct_time_buckets,
    }


def _read_s3_range(client: Any, bucket: str, key: str, start: int, end: int) -> bytes:
    attempts = int(os.getenv("S3_FULL_SCAN_RANGE_RETRIES", "5"))
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(min(30.0, 2.0 * attempt))
    raise RuntimeError("unreachable S3 range retry state")


def _aggregate_checkpoint_path(bucket: str, key: str, etag: str, size: int) -> Path:
    digest = hashlib.sha256(f"{bucket}\n{key}\n{etag}\n{size}".encode("utf-8")).hexdigest()
    return Path(__file__).resolve().parents[2] / "data" / "full-scan-checkpoints" / f"{digest}.json"


def _load_aggregate_checkpoint(
    checkpoint_path: Path,
    bucket: str,
    key: str,
    etag: str,
    size: int,
) -> dict[str, Any]:
    empty = {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "size": size,
        "next_start": 0,
        "tail_hex": "",
        "record_count": 0,
        "counts": {},
        "time_buckets": {},
        "distinct_time_buckets": {},
    }
    if not checkpoint_path.exists():
        return empty
    max_checkpoint_bytes = int(os.getenv("S3_FULL_SCAN_MAX_CHECKPOINT_BYTES", str(64 * 1024 * 1024)))
    if checkpoint_path.stat().st_size > max_checkpoint_bytes:
        checkpoint_path.unlink()
        return empty
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    if payload.get("bucket") != bucket or payload.get("key") != key:
        return empty
    if str(payload.get("etag") or "") != etag or int(payload.get("size") or 0) != size:
        return empty
    return {
        **empty,
        "next_start": int(payload.get("next_start") or 0),
        "tail_hex": str(payload.get("tail_hex") or ""),
        "record_count": int(payload.get("record_count") or 0),
        "counts": _string_int_mapping(payload.get("counts")),
        "time_buckets": _flat_int_mapping(payload.get("time_buckets")),
        "distinct_time_buckets": _nested_string_int_mapping(payload.get("distinct_time_buckets"))
        or _legacy_user_time_to_distinct_mapping(_string_int_mapping(payload.get("user_time_buckets"))),
    }


def _save_aggregate_checkpoint(
    checkpoint_path: Path,
    *,
    bucket: str,
    key: str,
    etag: str,
    size: int,
    next_start: int,
    tail: bytes,
    record_count: int,
    counts: dict[str, dict[str, int]],
    time_buckets: dict[str, int],
    distinct_time_buckets: dict[str, dict[str, dict[str, int]]] | None = None,
) -> None:
    tail_limit = int(os.getenv("S3_FULL_SCAN_MAX_TAIL_BYTES", str(8 * 1024 * 1024)))
    if len(tail) > tail_limit:
        raise ValueError(f"checkpoint tail is too large to serialize: {len(tail):,} bytes")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "size": size,
        "next_start": next_start,
        "tail_hex": tail.hex(),
        "record_count": record_count,
        "counts": counts,
        "time_buckets": time_buckets,
        "distinct_time_buckets": distinct_time_buckets or {},
        "updated_at": time.time(),
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def _string_int_mapping(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, int]] = {}
    for outer_key, inner in value.items():
        if not isinstance(inner, dict):
            continue
        result[str(outer_key)] = _flat_int_mapping(inner)
    return result


def _nested_string_int_mapping(value: Any) -> dict[str, dict[str, dict[str, int]]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, dict[str, int]]] = {}
    for field, buckets in value.items():
        if not isinstance(buckets, dict):
            continue
        parsed_buckets = _string_int_mapping(buckets)
        if parsed_buckets:
            result[str(field)] = parsed_buckets
    return result


def _flat_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        try:
            result[str(key)] = int(count)
        except (TypeError, ValueError):
            continue
    return result


def _json_string_mapping(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return _string_int_mapping(parsed)


def _json_distinct_time_mapping(value: Any) -> dict[str, dict[str, dict[str, int]]]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return _nested_string_int_mapping(parsed)


def _legacy_user_time_to_distinct_mapping(value: dict[str, dict[str, int]]) -> dict[str, dict[str, dict[str, int]]]:
    return {"user": value} if value else {}


def _json_flat_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return _flat_int_mapping(parsed)


def _rows_json_to_counts(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    result: dict[str, dict[str, int]] = {}
    for field, rows in parsed.items():
        if not isinstance(rows, list):
            continue
        field_counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = row.get("label")
            count = row.get("value")
            if label is None:
                continue
            try:
                field_counts[str(label)] = int(count)
            except (TypeError, ValueError):
                continue
        if field_counts:
            result[str(field)] = field_counts
    return result


def _top_distinct_time_rows_by_field(
    counts: dict[str, dict[str, dict[str, int]]],
    *,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        field: [
            {"label": label, "value": len(field_counts[label])}
            for label in sorted(field_counts)[:limit]
        ]
        for field, field_counts in sorted(counts.items())
    }


def _top_legacy_user_time_rows(
    counts: dict[str, dict[str, dict[str, int]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    user_fields = [
        field
        for field in counts
        if "user" in field.lower() or "learner" in field.lower() or "student" in field.lower()
    ]
    if not user_fields:
        return []
    field_counts = counts[sorted(user_fields, key=len)[0]]
    return [
        {"label": label, "value": len(field_counts[label])}
        for label in sorted(field_counts)[:limit]
    ]


def _rows_json_to_time_buckets(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}

    result: dict[str, int] = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        count = row.get("value")
        if label is None:
            continue
        try:
            result[str(label)] = int(count)
        except (TypeError, ValueError):
            continue
    return result


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
