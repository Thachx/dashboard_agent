from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


START = "<<<GRAPH_DASHBOARD_WIDGET>>>"
END = "<<<END_GRAPH_DASHBOARD_WIDGET>>>"
HUMAN_LABEL_RE = re.compile(r"([a-z0-9])([A-Z])")
HUMAN_KEEP_ALL_CAPS = {"API", "CSV", "DB", "ETAG", "ID", "JSON", "LLM", "SQL", "S3", "UI", "URL", "UTC"}


def graph_dashboard_payload(
    *,
    status: dict[str, Any],
    results: list[dict[str, Any]],
    activity: dict[str, Any] | None = None,
    title: str = "Dashboard graph",
) -> dict[str, Any]:
    """Build the graph-dashboard widget payload from status/results/activity.

    Single source of the payload: both the text marker (web chat widget) and
    the structured DashboardDocument (MCP Apps) derive from this, so the two
    render paths can never diverge.
    """
    datasets = dataset_summaries(results)
    result_chart = [
        {
            "label": human_label(item.get("label") or item.get("path") or item.get("source") or "result")[:48],
            "score": float(item.get("score") or 0),
        }
        for item in results[:8]
    ]
    return {
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


def render_dashboard_marker(payload: dict[str, Any]) -> str:
    return f"{START}\n{json.dumps(payload, ensure_ascii=False, default=str)}\n{END}"


def graph_dashboard_marker(
    *,
    status: dict[str, Any],
    results: list[dict[str, Any]],
    activity: dict[str, Any] | None = None,
    title: str = "Dashboard graph",
) -> str:
    payload = graph_dashboard_payload(status=status, results=results, activity=activity, title=title)
    return render_dashboard_marker(payload)


def dataset_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for item in results[:16]:
        source = str(item.get("source") or item.get("path") or item.get("id") or "")
        source_label = str(item.get("sourceLabel") or item.get("fileLabel") or human_label(source))
        dataset_key = _dataset_key(source)
        if not dataset_key:
            continue
        dataset = datasets.setdefault(
            dataset_key,
            {
                "key": dataset_key,
                "label": source_label,
                "sourceLabel": source_label,
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

# ---------------------------------------------------------------------------
# DashboardDocument builder
#
# The backend is the single source of truth for how an `activity` context
# (chartSlots, layoutSpec, summary, datasets, decisionTrace, ...) becomes the
# renderable DashboardDocument. The MCP Apps server forwards this document
# verbatim instead of re-implementing the conversion; the same conversion
# used to live in mcp/src/shared/dashboard-contract.ts and has been ported
# here so both render paths (the web chat widget marker and the MCP iframe)
# are produced by exactly this code.
#
# The output shape is the `DashboardDocumentSchema` in mcp/src/shared/
# dashboard-contract.ts (schemaVersion "1.0"); keep the two in sync.
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    "totalRecords",
    "sampleRecords",
    "distinctEvents",
    "distinctCourses",
    "distinctUsers",
    "distinctDimensionValues",
    "totalDistinctMeasure",
    "topDimensionValue",
]
DEFAULT_METRIC_LABELS = {
    "totalRecords": "Activity rows",
    "sampleRecords": "Sample rows",
    "distinctEvents": "Event types",
    "distinctCourses": "Courses",
    "distinctUsers": "Users",
    "distinctDimensionValues": "Groups",
    "totalDistinctMeasure": "Matched values",
    "topDimensionValue": "Leading value",
}
def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
def _text(value: Any, fallback: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback
def _numeric(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None
def _normalize_chart_data(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    data: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        source = _record(item)
        label = _text(source.get("label", source.get("name", source.get("key"))), f"Item {index + 1}")
        datum: dict[str, Any] = {
            "label": label,
            "value": _numeric(source.get("value")),
            "score": _numeric(source.get("score")),
            "series": _text(source.get("series")) or None,
            "rawLabel": _text(source.get("rawLabel")) or None,
        }
        if datum["value"] is None and datum["score"] is None:
            continue
        data.append(datum)
    return data
def _normalize_sources(payload: dict[str, Any], activity: dict[str, Any]) -> list[dict[str, Any]]:
    source_values: list[Any] = []
    if isinstance(payload.get("datasets"), list):
        source_values.extend(payload["datasets"])
    activity_datasets = activity.get("datasets")
    if isinstance(activity_datasets, list):
        source_values.extend(activity_datasets)
    elif isinstance(activity_datasets, dict):
        source_values.extend(activity_datasets.values())
    unique: dict[str, dict[str, Any]] = {}
    for value in source_values:
        source = _record(value)
        path = _text(source.get("path", source.get("source", source.get("key", source.get("s3Uri")))))
        source_id = _text(source.get("id", source.get("key", path)))
        if not source_id:
            continue
        matched = source.get("matchedFields")
        matched_fields = (
            [field for field in matched if isinstance(field, str)] if isinstance(matched, list) else None
        )
        unique[source_id] = {
            "id": source_id,
            "label": _text(source.get("label", source.get("sourceLabel", source.get("fileLabel"))), source_id),
            "path": path or None,
            "objectType": _text(source.get("objectType", source.get("object_type"))) or None,
            "matchedFields": matched_fields,
        }
    return list(unique.values())
def _infer_execution(activity: dict[str, Any]) -> dict[str, Any]:
    summary = _record(activity.get("summary"))
    source = " ".join(
        [
            _text(summary.get("executionSource")),
            _text(summary.get("source")),
            _text(activity.get("executionSource")),
        ]
    ).lower()
    graph_used = "graph" in source or bool(activity.get("graphContext"))
    duckdb_used = "duckdb" in source or "dashboard_agent_" in source
    if graph_used and duckdb_used:
        return {"source": "hybrid", "graphUsed": graph_used, "duckdbUsed": duckdb_used}
    if graph_used:
        return {"source": "graph", "graphUsed": graph_used, "duckdbUsed": duckdb_used}
    if duckdb_used:
        return {"source": "duckdb", "graphUsed": graph_used, "duckdbUsed": duckdb_used}
    return {"source": "unknown", "graphUsed": False, "duckdbUsed": False}
def build_dashboard_document(
    *,
    payload: dict[str, Any],
    query: str,
    locale: str = "en",
    timezone: str = "UTC",
    dashboard_id: str | None = None,
) -> dict[str, Any]:
    """Build a DashboardDocument from the graph-dashboard marker payload.

    This is the single conversion from the backend `activity` context to the
    renderable document. It was ported 1:1 from the MCP server's
    dashboardFromGraphPayload; the MCP now forwards the result instead of
    re-deriving it.
    """
    activity = _record(payload.get("activity"))
    summary = _record(activity.get("summary"))
    layout_spec = _record(activity.get("layoutSpec"))
    metric_labels = _record(summary.get("metricLabels"))
    metrics: list[dict[str, Any]] = []
    for metric_id in METRIC_KEYS:
        value = summary.get(metric_id)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        metrics.append(
            {
                "id": metric_id,
                "label": _text(metric_labels.get(metric_id), DEFAULT_METRIC_LABELS[metric_id]),
                "value": value,
                "hint": (
                    "Distinct values included in the aggregate"
                    if metric_id.startswith("distinct") or metric_id.startswith("totalDistinct")
                    else None
                ),
            }
        )
    charts: list[dict[str, Any]] = []
    slots = activity.get("chartSlots")
    for index, slot_value in enumerate(slots if isinstance(slots, list) else []):
        slot = _record(slot_value)
        data = _normalize_chart_data(slot.get("data"))
        if not data:
            continue
        charts.append(
            {
                "id": _text(slot.get("id"), f"chart-{index + 1}"),
                "title": _text(slot.get("title"), f"Chart {index + 1}"),
                "chartType": _text(slot.get("chartType"), "bar"),
                "field": _text(slot.get("field"), "value"),
                "reason": _text(slot.get("reason")) or None,
                "data": data,
            }
        )
    chart_ids = {chart["id"] for chart in charts}
    metric_ids = {metric["id"] for metric in metrics}
    layout: list[dict[str, Any]] = []
    blocks = layout_spec.get("blocks")
    for block_value in blocks if isinstance(blocks, list) else []:
        block = _record(block_value)
        block_type = _text(block.get("type"))
        if block_type == "metric" and _text(block.get("id")) in metric_ids:
            layout.append({"type": "metric", "id": _text(block.get("id")), "span": _numeric(block.get("span"))})
        elif block_type == "chart" and _text(block.get("slotId")) in chart_ids:
            layout.append({"type": "chart", "slotId": _text(block.get("slotId")), "span": _numeric(block.get("span"))})
        elif block_type == "records":
            layout.append({"type": "records", "span": _numeric(block.get("span"))})
    if not layout:
        layout = [
            *[{"type": "metric", "id": metric["id"]} for metric in metrics],
            *[{"type": "chart", "slotId": chart["id"]} for chart in charts],
        ]
    reasoning_summary: list[dict[str, Any]] = []
    trace = activity.get("decisionTrace")
    for value in trace if isinstance(trace, list) else []:
        item = _record(value)
        detail = _text(item.get("detail", item.get("summary", item.get("message"))))
        if not detail:
            continue
        reasoning_summary.append({"stage": _text(item.get("stage", item.get("name")), "Analysis"), "detail": detail})
    title = _text(layout_spec.get("title", payload.get("title")), "Dashboard")
    subtitle = _text(layout_spec.get("subtitle")) or None
    top_chart = charts[0] if charts else None
    summary_text = subtitle or (
        f"{top_chart['title']} with {len(top_chart['data'])} plotted values."
        if top_chart
        else f"{len(metrics)} dashboard metrics."
    )
    return _strip_none({
        "schemaVersion": "1.0",
        "dashboardId": dashboard_id or _new_dashboard_id(),
        "query": query,
        "title": title,
        "subtitle": subtitle,
        "locale": locale,
        "timezone": timezone,
        "generatedAt": _now_iso(),
        "summary": summary_text,
        "metrics": metrics,
        "charts": charts,
        "layout": layout,
        "sources": _normalize_sources(payload, activity),
        "reasoningSummary": reasoning_summary,
        "execution": _infer_execution(activity),
    })
def _strip_none(value: Any) -> Any:
    """Drop null values: the document contract treats absent keys as optional, while JSON null is rejected by the schema."""
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def _new_dashboard_id() -> str:
    return uuid4().hex
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
