from __future__ import annotations

import json
import re
from typing import Any


START = "<<<GRAPH_DASHBOARD_WIDGET>>>"
END = "<<<END_GRAPH_DASHBOARD_WIDGET>>>"
HUMAN_LABEL_RE = re.compile(r"([a-z0-9])([A-Z])")
HUMAN_KEEP_ALL_CAPS = {"API", "CSV", "DB", "ETAG", "ID", "JSON", "LLM", "SQL", "S3", "UI", "URL", "UTC"}


def graph_dashboard_marker(
    *,
    status: dict[str, Any],
    results: list[dict[str, Any]],
    activity: dict[str, Any] | None = None,
    title: str = "Dashboard graph",
) -> str:
    datasets = dataset_summaries(results)
    result_chart = [
        {
            "label": human_label(item.get("label") or item.get("path") or item.get("source") or "result")[:48],
            "score": float(item.get("score") or 0),
        }
        for item in results[:8]
    ]
    payload = {
        "version": 1,
        "kind": "graph-dashboard-widget",
        "title": title,
        "status": {
            "objects": status.get("objects", 0),
            "nodes": status.get("nodes", 0),
            "edges": status.get("edges", 0),
            "updatedAt": status.get("updated_at"),
        },
        "results": results[:8],
        "datasets": datasets,
        "activity": activity or {},
        "charts": {
            "status": [
                {"label": "Objects", "value": status.get("objects", 0)},
                {"label": "Nodes", "value": status.get("nodes", 0)},
                {"label": "Edges", "value": status.get("edges", 0)},
            ],
            "results": result_chart,
        },
    }
    return f"{START}\n{json.dumps(payload, ensure_ascii=False, default=str)}\n{END}"


def dataset_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for item in results[:16]:
        source = str(item.get("source") or item.get("path") or item.get("id") or "")
        dataset_key = _dataset_key(source)
        if not dataset_key:
            continue
        dataset = datasets.setdefault(
            dataset_key,
            {
                "key": dataset_key,
                "label": human_label(dataset_key),
                "score": 0,
                "matchedFields": [],
            },
        )
        dataset["score"] = max(float(dataset.get("score") or 0), float(item.get("score") or 0))
        label = str(item.get("label") or item.get("path") or "").removeprefix("$.")
        value = item.get("value")
        if label in {"s3_uri", "bucket", "key", "object_type", "size_bytes", "ETAG", "last_modified", "source_paths"}:
            dataset[_camel(label)] = value
        if label and label not in dataset["matchedFields"]:
            dataset["matchedFields"].append(human_label(label))

    return sorted(datasets.values(), key=lambda item: (-float(item.get("score") or 0), str(item["key"])))[:6]


def _dataset_key(source: str) -> str:
    if source.startswith("object::"):
        return source.removeprefix("object::")
    if "::$" in source:
        return source.split("::$", 1)[0]
    return source


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def human_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = re.sub(r"^\$\.?", "", text)
    text = re.sub(r"\.(json|csv|tsv|parquet|ndjson|jsonl|sql|db|duckdb|txt)$", "", text, flags=re.IGNORECASE)
    text = HUMAN_LABEL_RE.sub(r"\1 \2", text)
    text = re.sub(r"[_\-.]+", " ", text)
    words: list[str] = []
    for raw_word in text.split():
        word = raw_word.strip()
        if not word:
            continue
        if word.upper() in HUMAN_KEEP_ALL_CAPS:
            words.append(word.upper() if len(word) <= 4 else word.title())
            continue
        if word.isupper() and len(word) <= 4:
            words.append(word)
            continue
        if word.isdigit():
            words.append(word)
            continue
        words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words) if words else "Unknown"
