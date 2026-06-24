from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class JsonObject:
    key: str
    etag: str
    value: Any


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class S3JsonSource:
    def __init__(self, uri: str, *, region: str | None = None, max_object_bytes: int = 10 * 1024 * 1024):
        self.bucket, self.prefix = parse_s3_uri(uri)
        self.region = region
        self.max_object_bytes = max_object_bytes

    def load(self) -> Iterable[JsonObject]:
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
                if not key.lower().endswith(".json") or size > self.max_object_bytes:
                    continue
                response = client.get_object(Bucket=self.bucket, Key=key)
                try:
                    value = json.loads(response["Body"].read())
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"S3 object s3://{self.bucket}/{key} is not valid JSON") from exc
                yield JsonObject(key=key, etag=str(item.get("ETag", "")).strip('"'), value=value)
