from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    s3_data_uri: str = "s3://edx-nectec-demo"
    aws_region: str | None = None
    graph_path: Path = Path("data/s3-json-graph.json")
    graphify_output_dir: Path = Path("data/graphify-out")
    graphify_bin: str = "graphify"
    graphify_enabled: bool = True
    graph_refresh_seconds: int = 300
    graph_max_object_bytes: int = 10 * 1024 * 1024
    s3_include_extensions: tuple[str, ...] = (".json", ".parquet")
    llm_mode: str = "auto"
    llm_provider: str = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_main_model: str = "openai/gpt-4.1-mini"
    openrouter_reserve_model_1: str | None = None
    openrouter_reserve_model_2: str | None = None
    openrouter_http_referer: str | None = None
    openrouter_title: str = "Dashboard Graph Agent"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            s3_data_uri=os.getenv("S3_DATA_URI", cls.s3_data_uri),
            aws_region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            graph_path=Path(os.getenv("GRAPH_PATH", str(cls.graph_path))),
            graphify_output_dir=Path(os.getenv("GRAPHIFY_OUTPUT_DIR", str(cls.graphify_output_dir))),
            graphify_bin=os.getenv("GRAPHIFY_BIN", cls.graphify_bin),
            graphify_enabled=os.getenv("GRAPHIFY_ENABLED", str(cls.graphify_enabled)).strip().lower()
            not in {"0", "false", "no", "off"},
            graph_refresh_seconds=int(os.getenv("GRAPH_REFRESH_SECONDS", str(cls.graph_refresh_seconds))),
            graph_max_object_bytes=int(os.getenv("GRAPH_MAX_OBJECT_BYTES", str(cls.graph_max_object_bytes))),
            s3_include_extensions=_split_extensions(
                os.getenv("S3_INCLUDE_EXTENSIONS", ",".join(cls.s3_include_extensions))
            ),
            llm_mode=os.getenv("LLM_MODE", cls.llm_mode).strip().lower(),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider).strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", cls.openrouter_base_url),
            openrouter_main_model=os.getenv("OPENROUTER_MAIN_MODEL", cls.openrouter_main_model),
            openrouter_reserve_model_1=os.getenv("OPENROUTER_RESERVE_MODEL_1"),
            openrouter_reserve_model_2=os.getenv("OPENROUTER_RESERVE_MODEL_2"),
            openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            openrouter_title=os.getenv("OPENROUTER_TITLE", cls.openrouter_title),
        )


def _split_extensions(value: str) -> tuple[str, ...]:
    extensions: list[str] = []
    for item in value.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(extensions)
