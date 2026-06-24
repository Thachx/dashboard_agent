from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    s3_data_uri: str = "s3://edx-nectec-demo/data"
    aws_region: str | None = None
    graph_path: Path = Path("data/s3-json-graph.json")
    graph_refresh_seconds: int = 300
    graph_max_object_bytes: int = 10 * 1024 * 1024
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    power_bi_embed_url: str | None = None
    power_bi_report_id: str | None = None
    power_bi_access_token: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            s3_data_uri=os.getenv("S3_DATA_URI", cls.s3_data_uri),
            aws_region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            graph_path=Path(os.getenv("GRAPH_PATH", str(cls.graph_path))),
            graph_refresh_seconds=int(os.getenv("GRAPH_REFRESH_SECONDS", str(cls.graph_refresh_seconds))),
            graph_max_object_bytes=int(os.getenv("GRAPH_MAX_OBJECT_BYTES", str(cls.graph_max_object_bytes))),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            power_bi_embed_url=os.getenv("POWER_BI_EMBED_URL"),
            power_bi_report_id=os.getenv("POWER_BI_REPORT_ID"),
            power_bi_access_token=os.getenv("POWER_BI_ACCESS_TOKEN"),
        )
