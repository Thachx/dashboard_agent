from __future__ import annotations

import json
import re
import threading
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from dashboard_agent.config import Settings
from dashboard_agent.dashboard_widget import dataset_summaries, graph_dashboard_marker


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]


refresh_lock = threading.Lock()
settings: Settings | None = None
_settings: Settings | None = None
_store: Any | None = None
SAMPLE_FIELD_RE = re.compile(r"^\$\.sample_records\[(\d+)\]\.(.+)$")
DASHBOARD_BUILDING_INSTRUCTIONS = (
    "Build dashboards for the user's data domain, not for graph internals. "
    "Use this workflow: 1) decide which chart types fit the retrieved fields and grain, "
    "2) create empty chart widget slots with titles and intended encodings, "
    "3) query/bind the retrieved graph data into those chart slots. "
    "Prefer reader-usable sections in this order: KPI cards for the matched dataset/content, "
    "distribution charts over meaningful fields, sample records or rows with real values, "
    "source/dataset coverage and caveats, then graph node/edge/object counts only as technical metadata. "
    "If the source is represented by sampled content, label values as sampled and do not claim full-file aggregates."
)


def get_settings() -> Settings:
    global _settings
    if settings is not None:
        return settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def get_store() -> Any:
    global _store
    settings = get_settings()
    if _store is None or _store.path != settings.graph_path:
        from dashboard_agent.graph_store import JsonGraphStore

        _store = JsonGraphStore(settings.graph_path)
    return _store


def refresh_graph(force: bool = False) -> dict[str, Any]:
    global _store
    settings = get_settings()
    if force:
        from dashboard_agent.graph_store import JsonGraphStore

        _store = JsonGraphStore(settings.graph_path, load_existing=False)
        store = _store
    else:
        store = get_store()
    age = time.time() - store.updated_at if store.updated_at else float("inf")
    if not force and store.graph.number_of_nodes():
        return store.status()
    with refresh_lock:
        settings = get_settings()
        if force:
            from dashboard_agent.graph_store import JsonGraphStore

            _store = JsonGraphStore(settings.graph_path, load_existing=False)
            store = _store
        else:
            store = get_store()
        age = time.time() - store.updated_at if store.updated_at else float("inf")
        if not force and store.graph.number_of_nodes():
            return store.status()
        from dashboard_agent.graphify_ingest import ingest_s3_with_graphify
        from dashboard_agent.s3_source import S3JsonSource

        source = S3JsonSource(
            settings.s3_data_uri,
            region=settings.aws_region,
            max_object_bytes=settings.graph_max_object_bytes,
            include_extensions=settings.s3_include_extensions,
        )
        return ingest_s3_with_graphify(
            source,
            store,
            graphify_output_dir=settings.graphify_output_dir,
            graphify_bin=settings.graphify_bin,
            graphify_enabled=settings.graphify_enabled,
        )


def _last_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _wants_dashboard(question: str) -> bool:
    lowered = question.lower()
    return "dashboard" in lowered or "graph" in lowered or "widget" in lowered


def _wants_refresh(question: str) -> bool:
    lowered = question.lower()
    return "refresh" in lowered or "reload" in lowered or "rebuild" in lowered


def _fallback_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return "I do not have matching S3 JSON graph context yet."
    datasets = dataset_summaries(results)
    lines = ["I found this graph context:"]
    if datasets:
        lines.append("")
        lines.append("Relevant datasets:")
        for dataset in datasets[:4]:
            details = []
            if dataset.get("objectType"):
                details.append(str(dataset["objectType"]))
            if dataset.get("sizeBytes") is not None:
                details.append(_format_bytes(dataset["sizeBytes"]))
            if dataset.get("lastModified"):
                details.append(f"modified {dataset['lastModified']}")
            suffix = f" ({', '.join(details)})" if details else ""
            s3_uri = dataset.get("s3Uri") or dataset.get("key")
            lines.append(f"- `{dataset.get('key')}`{suffix}")
            if s3_uri:
                lines.append(f"  S3: `{s3_uri}`")
            matched_fields = dataset.get("matchedFields") or []
            if matched_fields:
                lines.append(f"  Evidence fields: `{', '.join(matched_fields[:8])}`")
    lines.append("")
    lines.append("Retrieved graph evidence:")
    for item in results[:10]:
        value = item.get("value")
        detail = str(value) if value is not None else str(item.get("text", ""))
        lines.append(f"- `{item.get('source')}` / `{item.get('path')}`: {detail[:240]}")
    return "\n".join(lines)


def _rag_context(question: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question": question,
        "datasets": dataset_summaries(results),
        "evidence_nodes": results[:16],
        "instruction": (
            "Use datasets as the primary retrieved context. Use evidence_nodes as citations. "
            "Do not invent row-level facts when an object is represented by metadata only. "
            + DASHBOARD_BUILDING_INSTRUCTIONS
        ),
    }


def _activity_dashboard_context(store: Any, results: list[dict[str, Any]]) -> dict[str, Any]:
    sources = []
    for item in results:
        source = item.get("source")
        if source and source not in sources:
            sources.append(source)
    if not sources:
        return {}

    records_by_source: dict[str, dict[int, dict[str, Any]]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}
    for attrs in _iter_graph_node_attrs(store):
        source = attrs.get("source") or attrs.get("path")
        if source not in sources:
            continue
        path = str(attrs.get("path") or "")
        label = str(attrs.get("label") or "")
        value = attrs.get("value")
        if path.startswith("$.") and "[" not in path and label in {
            "s3_uri",
            "bucket",
            "key",
            "object_type",
            "size_bytes",
            "last_modified",
            "sample_record_count",
            "sample_fields",
            "content_sample_bytes",
            "content_sample_ranges",
            "full_scan_status",
            "full_record_count",
            "full_counts_json",
            "full_time_buckets_json",
        }:
            dataset_meta.setdefault(str(source), {})[label] = value
        match = SAMPLE_FIELD_RE.match(path)
        if not match:
            continue
        index = int(match.group(1))
        field = match.group(2)
        if "." in field:
            continue
        records_by_source.setdefault(str(source), {}).setdefault(index, {})[field] = value

    records: list[dict[str, Any]] = []
    for source in sources:
        for index, record in sorted(records_by_source.get(str(source), {}).items()):
            if record:
                records.append({"source": source, "index": index, **record})
            if len(records) >= 200:
                break
        if len(records) >= 200:
            break
    aggregate_meta = next((meta for meta in dataset_meta.values() if meta.get("full_record_count")), {})
    full_counts = _json_rows_by_field(aggregate_meta.get("full_counts_json"))
    full_time_buckets = _json_rows(aggregate_meta.get("full_time_buckets_json"))
    if not records and not full_counts:
        return {}

    chart_plan = _design_activity_chart_plan(records, full_counts=full_counts, full_time_buckets=full_time_buckets)
    chart_slots = _bind_activity_chart_slots(
        chart_plan,
        records,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
    )
    return {
        "datasets": dataset_meta,
        "records": records,
        "chartPlan": chart_plan,
        "chartSlots": chart_slots,
        "charts": {slot["id"]: slot["data"] for slot in chart_slots},
        "summary": {
            "sampleRecords": len(records),
            "totalRecords": aggregate_meta.get("full_record_count"),
            "isFullAggregate": bool(aggregate_meta.get("full_record_count")),
            "distinctEvents": len(full_counts.get("event", []))
            or len({record.get("event") for record in records if record.get("event")}),
            "distinctCourses": len(full_counts.get("courseID", []))
            or len({record.get("courseID") for record in records if record.get("courseID")}),
            "distinctUsers": len(full_counts.get("userID", []))
            or len({record.get("userID") for record in records if record.get("userID")}),
        },
    }


def _design_activity_chart_plan(
    records: list[dict[str, Any]],
    *,
    full_counts: dict[str, list[dict[str, Any]]] | None = None,
    full_time_buckets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidate_fields = full_counts or {
        "event": _top_counts(records, "event"),
        "eventCategory": _top_counts(records, "eventCategory"),
        "courseID": _top_counts(records, "courseID"),
        "appID": _top_counts(records, "appID"),
        "userID": _top_counts(records, "userID"),
    }
    plan: list[dict[str, Any]] = []
    if full_time_buckets or _time_buckets(records):
        plan.append(
            {
                "id": "activityTimeline",
                "title": "Activity over sampled time",
                "chartType": "line",
                "field": "@timestamp",
                "reason": "A line chart fits timestamped records and shows activity order over the sample window.",
            }
        )
    if candidate_fields["event"]:
        plan.append(
            {
                "id": "events",
                "title": "Event mix",
                "chartType": "donut",
                "field": "event",
                "reason": "A donut chart fits a small categorical split of activity event types.",
            }
        )
    if candidate_fields["courseID"]:
        plan.append(
            {
                "id": "courses",
                "title": "Courses in the sample",
                "chartType": "horizontal_bar",
                "field": "courseID",
                "reason": "Course IDs are long labels, so horizontal bars are easiest to read.",
            }
        )
    if candidate_fields["eventCategory"]:
        chart_type = "stat" if len(candidate_fields["eventCategory"]) == 1 else "donut"
        plan.append(
            {
                "id": "categories",
                "title": "Event categories",
                "chartType": chart_type,
                "field": "eventCategory",
                "reason": (
                    "A single category is better as an insight card than a chart."
                    if chart_type == "stat"
                    else "A donut chart fits a small categorical split."
                ),
            }
        )
    if candidate_fields["appID"]:
        chart_type = "stat" if len(candidate_fields["appID"]) == 1 else "column"
        plan.append(
            {
                "id": "apps",
                "title": "Applications",
                "chartType": chart_type,
                "field": "appID",
                "reason": (
                    "A single application is better as an insight card than a chart."
                    if chart_type == "stat"
                    else "A column chart compares sampled application IDs."
                ),
            }
        )
    return plan[:5]


def _bind_activity_chart_slots(
    chart_plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    full_counts: dict[str, list[dict[str, Any]]] | None = None,
    full_time_buckets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    slots = []
    for item in chart_plan:
        field = str(item["field"])
        if item["chartType"] == "line":
            data = full_time_buckets or _time_buckets(records)
        else:
            data = (full_counts or {}).get(field) or _top_counts(records, field)
        slots.append({**item, "data": data})
    return slots


def _json_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    return parsed if isinstance(parsed, list) else []


def _json_rows_by_field(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    else:
        parsed = value
    if not isinstance(parsed, dict):
        return {}
    return {str(key): rows for key, rows in parsed.items() if isinstance(rows, list)}


def _time_buckets(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get("@timestamp") or record.get("timestamp")
        if not value:
            continue
        timestamp = str(value)
        label = timestamp[11:16] if len(timestamp) >= 16 else timestamp
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "value": counts[label]} for label in sorted(counts)[:limit]]


def _iter_graph_node_attrs(store: Any) -> list[dict[str, Any]]:
    if store.graph.number_of_nodes():
        return [dict(attrs, id=node_id) for node_id, attrs in store.graph.nodes(data=True)]
    try:
        payload = json.loads(store.path.read_text(encoding="utf-8"))
    except Exception:
        return []
    nodes = payload.get("nodes") if isinstance(payload, dict) else []
    return [node for node in nodes if isinstance(node, dict)]


def _top_counts(records: list[dict[str, Any]], field: str, limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"label": label, "value": value}
        for label, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return str(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def _openrouter_headers() -> dict[str, str]:
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_title:
        headers["X-OpenRouter-Title"] = settings.openrouter_title
    return headers


def _openrouter_models() -> list[str]:
    settings = get_settings()
    models = [
        settings.openrouter_main_model,
        settings.openrouter_reserve_model_1,
        settings.openrouter_reserve_model_2,
    ]
    ordered: list[str] = []
    for model in models:
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _llm_candidates() -> list[dict[str, Any]]:
    settings = get_settings()
    candidates: list[dict[str, Any]] = []
    provider = settings.llm_provider
    use_openrouter = provider in {"auto", "openrouter"}
    use_openai = provider in {"auto", "openai"}
    if use_openrouter and settings.openrouter_api_key:
        headers = _openrouter_headers()
        for model in _openrouter_models():
            candidates.append(
                {
                    "provider": "openrouter",
                    "api_key": settings.openrouter_api_key,
                    "model": model,
                    "base_url": settings.openrouter_base_url,
                    "headers": headers,
                }
            )
    if use_openai and settings.openai_api_key:
        candidates.append(
            {
                "provider": "openai",
                "api_key": settings.openai_api_key,
                "model": settings.openai_model,
                "base_url": settings.openai_base_url,
                "headers": {},
            }
        )
    return candidates


def _should_use_llm(results: list[dict[str, Any]]) -> bool:
    settings = get_settings()
    mode = settings.llm_mode
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode == "auto":
        return bool(results)
    return bool(results)


def _llm_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not _should_use_llm(results):
        return _fallback_answer(question, results)
    candidates = _llm_candidates()
    if not candidates:
        return _fallback_answer(question, results)

    from langchain_openai import ChatOpenAI

    context = json.dumps(_rag_context(question, results), ensure_ascii=False, default=str)
    messages = [
        (
            "system",
            (
                "Answer using only supplied graph RAG context. Cite S3 keys or JSON paths in backticks. "
                "Summarize retrieved datasets before raw nodes. State when evidence is metadata-only or insufficient. "
                + DASHBOARD_BUILDING_INSTRUCTIONS
            ),
        ),
        ("human", f"Question: {question}\n\nGraph context:\n{context}"),
    ]
    failures: list[str] = []
    for candidate in candidates:
        try:
            model = ChatOpenAI(
                api_key=candidate["api_key"],
                model=candidate["model"],
                base_url=candidate.get("base_url"),
                default_headers=candidate.get("headers") or None,
                temperature=0,
            )
            response = model.invoke(messages)
            return str(response.content)
        except Exception as exc:
            failures.append(f"{candidate['provider']}:{candidate['model']} failed: {exc}")
    fallback = _fallback_answer(question, results)
    return f"{fallback}\n\nLLM synthesis unavailable after trying configured models: {'; '.join(failures)}"


def run_agent(state: AgentState) -> dict[str, list[AIMessage]]:
    question = _last_question(state)
    store = get_store()
    notices: list[str] = []
    wants_refresh = _wants_refresh(question)
    try:
        if wants_refresh:
            status = refresh_graph(force=True)
            graphify_status = status.get("graphify_status")
            graphify_detail = ""
            if graphify_status == "ok":
                graphify_detail = f" Graphify graph written to {status.get('graphify_path')}."
            elif graphify_status:
                graphify_detail = f" Graphify export status: {graphify_status}."
            notices.append(
                f"S3 graph refreshed: {status['objects']} objects, {status['nodes']} nodes, {status['edges']} edges."
                f"{graphify_detail}"
            )
        else:
            status = store.status()
    except Exception as exc:
        if not store.graph.number_of_nodes():
            notices.append(f"S3 ingestion unavailable: {exc}")
        else:
            notices.append(f"Using last saved graph because S3 refresh failed: {exc}")

    results = store.search(question, limit=24)
    if _wants_dashboard(question):
        activity = _activity_dashboard_context(store, results)
        title = "Activity dashboard" if activity else "Dashboard graph"
        answer = graph_dashboard_marker(status=store.status(), results=results, activity=activity, title=title)
    else:
        answer = _llm_answer(question, results)
    if notices:
        answer = "\n\n".join(notices + [answer])
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(AgentState)
builder.add_node("dashboard_agent", run_agent)
builder.add_edge(START, "dashboard_agent")
builder.add_edge("dashboard_agent", END)

graph = builder.compile()
