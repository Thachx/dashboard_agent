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
_duckdb_field_cache: dict[str, dict[str, str]] = {}
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
    if not force and _has_loaded_graph_context(store):
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
        if not force and _has_loaded_graph_context(store):
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


def _has_loaded_graph_context(store: Any) -> bool:
    return bool(
        store.graph.number_of_nodes()
        or getattr(store, "search_rows", None)
        or getattr(store, "index_status", None)
    )


def _last_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _wants_dashboard(question: str) -> bool:
    return bool(question.strip()) and not _wants_refresh(question)


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


def _aggregate_cache_results(store: Any, question: str) -> list[dict[str, Any]]:
    requested = _requested_json_names(question)
    cache_path = getattr(store, "path", None)
    if cache_path is None:
        return []
    cache_path = cache_path.parent / "full-scan-aggregates.json"
    if not cache_path.exists():
        return []

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    requested_keys = set(requested)
    if not requested_keys:
        requested_keys = _ranked_aggregate_cache_keys(payload, question)
    if not requested_keys:
        return []

    rows: list[dict[str, Any]] = []
    for key, meta in payload.items():
        if not isinstance(key, str) or not isinstance(meta, dict):
            continue
        key_name = key.rsplit("/", 1)[-1].lower()
        if not any(name == key_name or name in key.lower() or name == key for name in requested_keys):
            continue
        text_parts = [
            key,
            f"records {meta.get('full_record_count', 0)}",
            str(meta.get("full_counts_json", ""))[:4000],
            str(meta.get("full_time_buckets_json", ""))[:2000],
            str(meta.get("full_distinct_time_buckets_json", ""))[:3000],
            str(meta.get("full_user_time_buckets_json", ""))[:2000],
        ]
        public_meta = {
            field: value
            for field, value in meta.items()
            if not field.startswith("_full_")
        }
        rows.append(
            {
                "id": f"aggregate::{key}",
                "label": key.rsplit("/", 1)[-1],
                "path": key,
                "source": key,
                "value": {
                    "key": key,
                    "s3_uri": f"s3://edx-nectec-demo/{key}",
                    **public_meta,
                },
                "text": " ".join(text_parts),
                "score": 100.0,
            }
        )
    return rows


def _ranked_aggregate_cache_keys(payload: dict[str, Any], question: str) -> set[str]:
    lowered = question.lower()
    wants_user = "user" in lowered or "learner" in lowered
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    wants_event = "event" in lowered or "activity" in lowered
    wants_course = "course" in lowered

    scored: list[tuple[float, str]] = []
    for key, meta in payload.items():
        if not isinstance(key, str) or not isinstance(meta, dict):
            continue
        full_counts = _json_rows_by_field(meta.get("full_counts_json"))
        time_rows = _json_rows(meta.get("full_time_buckets_json"))
        distinct_time_rows = _json_rows_by_field(meta.get("full_distinct_time_buckets_json"))
        user_time_rows = _json_rows(meta.get("full_user_time_buckets_json"))
        score = 0.0
        if wants_user and _field_matches_question("user learner student", question, full_counts):
            score += 5.0
        if wants_time and (time_rows or distinct_time_rows or user_time_rows):
            score += 5.0
        if wants_user and wants_time and (user_time_rows or _field_matches_question("user learner student", question, distinct_time_rows)):
            score += 6.0
        if wants_event and (full_counts.get("event") or full_counts.get("eventCategory")):
            score += 3.0
        if wants_course and full_counts.get("courseID"):
            score += 3.0
        for token in re.findall(r"[a-z0-9_-]{3,}", lowered):
            if token in key.lower():
                score += 1.0
        try:
            score += min(float(meta.get("full_record_count") or 0) / 1_000_000, 3.0)
        except (TypeError, ValueError):
            pass
        if score > 0:
            scored.append((score, key))

    scored.sort(reverse=True)
    return {key for _, key in scored[:3]}


def _requested_json_names(question: str) -> list[str]:
    names = re.findall(r"[\w./-]+\.json", question, flags=re.IGNORECASE)
    result: list[str] = []
    for name in names:
        normalized = name.strip("`'\".,;:()[]{}").replace("\\", "/").rsplit("/", 1)[-1].lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _field_matches_question(field_hint: str, question: str, fields: dict[str, Any]) -> bool:
    terms = set(_intent_terms(f"{field_hint} {question}"))
    return any(_field_score(field, terms) > 0 for field in fields)


def _intent_terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 2 and term not in {"over", "time", "trend", "number", "count", "total", "show", "have"}
    ]


def _field_terms(field: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", field)
    return set(re.findall(r"[a-z0-9]+", normalized.lower()))


def _field_score(field: str, intent_terms: set[str]) -> float:
    terms = _field_terms(field)
    if not terms:
        return 0.0
    score = float(len(terms & intent_terms) * 4)
    field_lower = field.lower()
    for term in intent_terms:
        if term in field_lower:
            score += 1.0
    return score


def _humanize_field(field: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", field).replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() if word.islower() else word for word in spaced.split())


def _slot_safe_field(field: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", field).strip("-")
    return value[:48] or "field"


def _duckdb_quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _aggregate_cache_activity(store: Any, question: str) -> dict[str, Any]:
    rows = _aggregate_cache_results(store, question)
    if not rows:
        return {}

    row = rows[0]
    value = row.get("value") if isinstance(row.get("value"), dict) else {}
    full_counts = _json_rows_by_field(value.get("full_counts_json"))
    full_time_buckets = _json_rows(value.get("full_time_buckets_json"))
    full_distinct_time_buckets = _json_rows_by_field(value.get("full_distinct_time_buckets_json"))
    full_user_time_buckets = _json_rows(value.get("full_user_time_buckets_json"))
    if full_user_time_buckets and not full_distinct_time_buckets:
        full_distinct_time_buckets = {"user": full_user_time_buckets}
    records: list[dict[str, Any]] = []
    chart_plan = _design_activity_chart_plan(
        records,
        question=question,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
        full_distinct_time_buckets=full_distinct_time_buckets,
        full_user_time_buckets=full_user_time_buckets,
    )
    chart_slots = _bind_activity_chart_slots(
        chart_plan,
        records,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
        full_distinct_time_buckets=full_distinct_time_buckets,
        full_user_time_buckets=full_user_time_buckets,
    )
    chart_slots = _retitle_slots_for_prompt(question, chart_slots)
    chart_plan = [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots]
    distinct_events = len(full_counts.get("event", []))
    distinct_courses = len(full_counts.get("courseID", []))
    distinct_users = len(full_counts.get("userID", []))
    total_records = value.get("full_record_count", 0)
    source = row.get("path") or row.get("source") or row.get("label")
    activity = {
        "datasets": [
            {
                "source": source,
                "sampleRecords": 0,
                "totalRecords": total_records,
                "isFullAggregate": True,
                "distinctEvents": distinct_events,
                "distinctCourses": distinct_courses,
                "distinctUsers": distinct_users,
            }
        ],
        "records": records,
        "chartPlan": chart_plan,
        "chartSlots": chart_slots,
        "charts": chart_slots,
        "summary": {
            "source": source,
            "sampleRecords": 0,
            "totalRecords": total_records,
            "isFullAggregate": True,
            "distinctEvents": distinct_events,
            "distinctCourses": distinct_courses,
            "distinctUsers": distinct_users,
        },
    }
    activity["layoutSpec"] = _activity_layout_spec(question, activity)
    return activity


def _duckdb_activity_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    if not any(term in lowered for term in ("activity", "event", "user", "learner", "student", "course", "time", "trend")):
        return {}
    db_path = getattr(store, "path", None)
    if db_path is None:
        return {}
    db_path = db_path.parent / "dashboard_agent.duckdb"
    if not db_path.exists():
        return {}
    try:
        import duckdb
    except ImportError:
        return {}

    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return {}
    try:
        source = _select_duckdb_source(con, question)
        if not source:
            return {}
        source_path = str(source.get("source_path") or "")
        fields = _infer_duckdb_json_fields(con, source_path)
        timestamp_field = fields.get("timestamp")
        if not timestamp_field:
            return {}
        user_field = fields.get("user")
        event_field = fields.get("event")
        category_field = fields.get("category")
        course_field = fields.get("course")
        app_field = fields.get("app")
        wants_user_time = (
            bool(user_field)
            and ("user" in lowered or "learner" in lowered or "student" in lowered)
            and ("time" in lowered or "trend" in lowered or "over time" in lowered)
        )

        timeline = _duckdb_time_buckets(con, source_path, timestamp_field, user_field)
        counts: dict[str, list[dict[str, Any]]] = {}
        count_fields: list[str] = []
        if not wants_user_time:
            count_fields = [field for field in (event_field, category_field, course_field, app_field, user_field) if field]
        else:
            if "event" in lowered:
                count_fields.extend(field for field in (event_field, category_field) if field)
            if "course" in lowered and course_field:
                count_fields.append(course_field)
        for field in count_fields:
            if field:
                counts[field] = _duckdb_top_counts(con, source_path, field)

        records = _duckdb_sample_records(con, source_path, limit=12)
        full_time_buckets = [{"label": row["label"], "value": row["value"]} for row in timeline]
        full_user_time_buckets = [
            {"label": row["label"], "value": row["users"]}
            for row in timeline
            if row.get("users") is not None
        ]
        distinct_time_buckets = {}
        if wants_user_time and full_user_time_buckets:
            chart_plan = [
                {
                    "id": "userTimeline",
                    "title": "Users over time",
                    "chartType": "line",
                    "field": f"{user_field}@timestamp",
                    "reason": "A line chart shows distinct users by detected time bucket.",
                },
                {
                    "id": "activityTimeline",
                    "title": "Activity over time",
                    "chartType": "line",
                    "field": "@timestamp",
                    "reason": "A line chart shows activity volume by detected time bucket.",
                },
            ]
        else:
            distinct_time_buckets = {user_field: full_user_time_buckets} if user_field and full_user_time_buckets else {}
            chart_plan = _design_activity_chart_plan(
                records,
                question=question,
                full_counts=counts,
                full_time_buckets=full_time_buckets,
                full_distinct_time_buckets=distinct_time_buckets,
                full_user_time_buckets=full_user_time_buckets,
            )
        chart_slots = _bind_activity_chart_slots(
            chart_plan,
            records,
            full_counts=counts,
            full_time_buckets=full_time_buckets,
            full_distinct_time_buckets=distinct_time_buckets,
            full_user_time_buckets=full_user_time_buckets,
        )
        chart_slots = _retitle_slots_for_prompt(question, chart_slots)
        chart_plan = [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots]
        total_records = source.get("records")
        distinct_users = _duckdb_distinct_activity_users(con) if wants_user_time else (
            _duckdb_distinct_count(con, source_path, user_field) if user_field else 0
        )
        activity = {
            "datasets": {
                source_path: {
                    "source": source_path,
                    "key": source_path,
                    "object_type": source.get("source_format"),
                    "source_table": source.get("source_table"),
                    "sampleRecords": len(records),
                    "totalRecords": total_records,
                    "isFullAggregate": True,
                }
            },
            "records": records,
            "chartPlan": chart_plan,
            "chartSlots": chart_slots,
            "charts": {slot["id"]: slot["data"] for slot in chart_slots},
            "summary": {
                "source": source_path,
                "sampleRecords": len(records),
                "totalRecords": total_records,
                "isFullAggregate": True,
                "distinctEvents": len(counts.get(event_field or "", [])),
                "distinctCourses": len(counts.get(course_field or "", [])),
                "distinctUsers": distinct_users,
            },
        }
        activity["layoutSpec"] = _activity_layout_spec(question, activity)
        return activity
    except Exception:
        return {}
    finally:
        con.close()


def _graph_ranked_dimension_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    if not any(term in lowered for term in ("most", "top", "highest", "largest", "max", "rank")):
        return {}
    aggregates = _graph_aggregate_values(store, "ranked_dimension")
    if not aggregates:
        return {}
    query_terms = set(_intent_terms(_expand_ranked_query_terms(question)))
    measure_terms = _measure_terms(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for aggregate in aggregates:
        semantic_terms = set(str(term).lower() for term in aggregate.get("semantic_terms") or [])
        dimension = str(aggregate.get("dimension_field") or "")
        measure = str(aggregate.get("measure_field") or "")
        score = float(len(query_terms & semantic_terms) * 2)
        score += _ranked_dimension_score(dimension, query_terms)
        score += _ranked_measure_score(measure, measure_terms)
        if score > 0:
            scored.append((score, aggregate))
    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("duckdb_table") or ""),
            str(item[1].get("dimension_field") or ""),
            str(item[1].get("measure_field") or ""),
        )
    )
    if not scored:
        return {}
    return _ranked_dimension_activity_from_payload(scored[0][1], question, source_kind="graph")


def _graph_time_series_activity_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "over time" in lowered or "overtime" in lowered
    if not wants_time:
        return {}
    aggregates = _graph_aggregate_values(store, "time_series")
    if not aggregates:
        return {}
    aggregate = aggregates[0]
    wants_user = "user" in lowered or "learner" in lowered or "student" in lowered
    users_series = aggregate.get("users_series") if isinstance(aggregate.get("users_series"), list) else []
    records_series = aggregate.get("records_series") if isinstance(aggregate.get("records_series"), list) else []
    chart_slots = []
    if wants_user and users_series:
        chart_slots.append(
            {
                "id": "userTimeline",
                "title": "Users over time",
                "chartType": "line",
                "field": "users@time",
                "reason": "Graph aggregate of cumulative distinct users by time bucket.",
                "data": users_series,
            }
        )
    if records_series:
        chart_slots.append(
            {
                "id": "activityTimeline",
                "title": "Activity over time",
                "chartType": "line",
                "field": "records@time",
                "reason": "Graph aggregate of cumulative activity records by time bucket.",
                "data": records_series,
            }
        )
    if not chart_slots:
        return {}
    summary = {
        "source": aggregate.get("duckdb_table") or "graph aggregate",
        "sampleRecords": len(chart_slots[0].get("data") or []),
        "totalRecords": aggregate.get("total_records"),
        "isFullAggregate": True,
        "distinctUsers": aggregate.get("total_users"),
    }
    activity = {
        "datasets": {
            "graph/time_series": {
                "source": "graph/time_series",
                "key": "graph/time_series",
                "object_type": "graph_aggregate",
                "sampleRecords": summary["sampleRecords"],
                "totalRecords": summary["totalRecords"],
                "isFullAggregate": True,
            }
        },
        "records": [],
        "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
        "chartSlots": chart_slots,
        "charts": {slot["id"]: slot["data"] for slot in chart_slots},
        "summary": summary,
    }
    activity["layoutSpec"] = _activity_layout_spec(question, activity)
    return activity


def _graph_aggregate_values(store: Any, aggregate_kind: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in getattr(store, "search_rows", None) or []:
        value = row.get("value")
        if isinstance(value, dict) and value.get("aggregate_kind") == aggregate_kind:
            values.append(value)
    if values:
        return values
    for attrs in _iter_graph_node_attrs(store):
        value = attrs.get("value")
        if isinstance(value, dict) and value.get("aggregate_kind") == aggregate_kind:
            values.append(value)
    return values


def _ranked_dimension_activity_from_payload(
    aggregate: dict[str, Any],
    question: str,
    *,
    source_kind: str,
) -> dict[str, Any]:
    chart_data = aggregate.get("chart_data") if isinstance(aggregate.get("chart_data"), list) else []
    if not chart_data:
        return {}
    table = str(aggregate.get("duckdb_table") or "graph aggregate")
    dimension = str(aggregate.get("dimension_field") or "dimension")
    measure = str(aggregate.get("measure_field") or "measure")
    dimension_label = _dimension_display_name(dimension, question)
    measure_label = _measure_display_name(measure)
    top_label = str(aggregate.get("top_label") or chart_data[0].get("label") or "")
    top_value = int(aggregate.get("top_value") or chart_data[0].get("value") or 0)
    total_measure = int(aggregate.get("total_measure_values") or 0)
    total_dimensions = int(aggregate.get("distinct_dimension_values") or 0)
    slot_id = f"top-{_slot_safe_field(dimension)}-by-{_slot_safe_field(measure)}"
    chart_slots = [
        {
            "id": slot_id,
            "title": f"{dimension_label} by {measure_label}",
            "chartType": "horizontal_bar",
            "field": dimension,
            "reason": f"Ranked aggregate read from {source_kind} graph context.",
            "data": chart_data,
        }
    ]
    return {
        "datasets": {
            table: {
                "source": table,
                "key": table,
                "object_type": f"{source_kind}_aggregate",
                "sampleRecords": len(chart_data),
                "totalRecords": total_measure,
                "isFullAggregate": True,
            }
        },
        "records": [
            {"source": table, "index": index, dimension: item.get("label"), measure: item.get("value")}
            for index, item in enumerate(chart_data)
        ],
        "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
        "chartSlots": chart_slots,
        "charts": {slot_id: chart_data},
        "summary": {
            "source": table,
            "sampleRecords": len(chart_data),
            "totalRecords": total_measure,
            "isFullAggregate": True,
            "topDimensionLabel": top_label,
            "topDimensionValue": top_value,
            "topDimensionField": dimension,
            "topDimensionName": dimension_label,
            "measureField": measure,
            "measureName": measure_label,
            "totalDistinctMeasure": total_measure,
            "distinctDimensionValues": total_dimensions,
        },
        "layoutSpec": {
            "title": f"Top {dimension_label} by {measure_label}",
            "subtitle": f"{top_label} has the highest distinct {measure_label}.",
            "blocks": [
                {"type": "metric", "id": "topDimensionValue", "span": 1},
                {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
                {"type": "chart", "slotId": slot_id, "span": 2},
            ],
        },
    }


def _duckdb_ranked_dimension_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    wants_rank = any(term in lowered for term in ("most", "top", "highest", "largest", "max", "rank"))
    if not wants_rank:
        return {}
    db_path = getattr(store, "path", None)
    if db_path is None:
        return {}
    db_path = db_path.parent / "dashboard_agent.duckdb"
    if not db_path.exists():
        return {}
    try:
        import duckdb
    except ImportError:
        return {}
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return {}
    try:
        plan = _ranked_dimension_plan(con, question)
        if not plan:
            return {}
        table = str(plan["table"])
        dimension = str(plan["dimension"])
        measure = str(plan["measure"])
        dimension_expr = _duckdb_quote_identifier(dimension)
        measure_expr = _duckdb_quote_identifier(measure)
        rows = con.execute(
            f"""
            select
                {dimension_expr} as label,
                count(distinct {measure_expr}) as value
            from {_duckdb_quote_identifier(table)}
            where {dimension_expr} is not null
              and cast({dimension_expr} as varchar) <> ''
              and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
              and {measure_expr} is not null
            group by 1
            order by value desc, label
            limit 12
            """
        ).fetchall()
        if not rows:
            return {}
        total_measure = int(
            con.execute(
                f"""
                select count(distinct {measure_expr})
                from {_duckdb_quote_identifier(table)}
                where {measure_expr} is not null
                  and {dimension_expr} is not null
                  and cast({dimension_expr} as varchar) <> ''
                  and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
                """
            ).fetchone()[0]
            or 0
        )
        total_dimensions = int(
            con.execute(
                f"""
                select count(distinct {dimension_expr})
                from {_duckdb_quote_identifier(table)}
                where {dimension_expr} is not null
                  and cast({dimension_expr} as varchar) <> ''
                  and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
                """
            ).fetchone()[0]
            or 0
        )
        chart_data = [{"label": str(label), "value": int(value)} for label, value in rows]
        top_label = chart_data[0]["label"]
        top_value = chart_data[0]["value"]
        dimension_label = _dimension_display_name(dimension, question)
        measure_label = _measure_display_name(measure)
        slot_id = f"top-{_slot_safe_field(dimension)}-by-{_slot_safe_field(measure)}"
        chart_slots = [
            {
                "id": slot_id,
                "title": f"{dimension_label} by {measure_label}",
                "chartType": "horizontal_bar",
                "field": dimension,
                "reason": "A ranked bar chart answers the requested top dimension by distinct measure.",
                "data": chart_data,
            }
        ]
        activity = {
            "datasets": {
                table: {
                    "source": table,
                    "key": table,
                    "object_type": "duckdb",
                    "sampleRecords": len(chart_data),
                    "totalRecords": total_measure,
                    "isFullAggregate": True,
                }
            },
            "records": [
                {"source": table, "index": index, dimension: label, measure: value}
                for index, (label, value) in enumerate(rows)
            ],
            "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
            "chartSlots": chart_slots,
            "charts": {slot_id: chart_data},
            "summary": {
                "source": table,
                "sampleRecords": len(chart_data),
                "totalRecords": total_measure,
                "isFullAggregate": True,
                "topDimensionLabel": top_label,
                "topDimensionValue": top_value,
                "topDimensionField": dimension,
                "topDimensionName": dimension_label,
                "measureField": measure,
                "measureName": measure_label,
                "totalDistinctMeasure": total_measure,
                "distinctDimensionValues": total_dimensions,
            },
            "layoutSpec": {
                "title": f"Top {dimension_label} by {measure_label}",
                "subtitle": f"{top_label} has the highest distinct {measure_label}.",
                "blocks": [
                    {"type": "metric", "id": "topDimensionValue", "span": 1},
                    {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
                    {"type": "chart", "slotId": slot_id, "span": 2},
                ],
            },
        }
        return activity
    except Exception:
        return {}
    finally:
        con.close()


def _ranked_dimension_plan(con: Any, question: str) -> dict[str, str]:
    tables = [
        str(row[0])
        for row in con.execute(
            """
            select table_name
            from information_schema.tables
            where table_schema = 'main'
              and table_name like 'dashboard_agent_%'
            order by table_name
            """
        ).fetchall()
    ]
    intent_terms = set(_intent_terms(_expand_ranked_query_terms(question)))
    measure_terms = _measure_terms(question)
    candidates: list[tuple[float, str, str, str]] = []
    for table in tables:
        if any(skip in table for skip in ("cache", "map")):
            continue
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
        measure_columns = [column for column in columns if _is_measure_column(column, measure_terms)]
        dimension_columns = [column for column in columns if _is_dimension_column(column)]
        for dimension in dimension_columns:
            dimension_score = _ranked_dimension_score(dimension, intent_terms)
            if dimension_score <= 0:
                continue
            for measure in measure_columns:
                measure_score = _ranked_measure_score(measure, measure_terms)
                if measure_score <= 0:
                    continue
                table_score = 1.0 if "fact" in table or "joined" in table else 0.0
                if "name" in intent_terms and "name" in _field_terms(dimension):
                    dimension_score += 2.0
                candidates.append((dimension_score + measure_score + table_score, table, dimension, measure))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    for _, table, dimension, measure in candidates:
        if _ranked_dimension_has_values(con, table, dimension, measure):
            return {"table": table, "dimension": dimension, "measure": measure}
    return {}


def _expand_ranked_query_terms(question: str) -> str:
    lowered = question.lower()
    extra: list[str] = []
    if "institute" in lowered or "institution" in lowered:
        extra.extend(["school", "school_name", "institute_id", "organization", "name"])
    if "user" in lowered or "learner" in lowered or "student" in lowered:
        extra.extend(["user_id", "activity_user_id", "student_id", "learner_id"])
    if "course" in lowered:
        extra.extend(["course_id", "subject_name", "course"])
    return f"{question} {' '.join(extra)}"


def _dimension_display_name(field: str, question: str) -> str:
    lowered = question.lower()
    if ("institute" in lowered or "institution" in lowered) and field in {"school_name", "institute_id"}:
        return "Institute Name" if field == "school_name" else "Institute"
    return _humanize_field(field)


def _measure_display_name(field: str) -> str:
    lowered = field.lower()
    if lowered in {"user_id", "activity_user_id", "student_id", "learner_id"} or lowered.endswith("_user_id"):
        return "Users"
    if lowered == "course_id":
        return "Courses"
    return _humanize_field(field)


def _measure_terms(question: str) -> set[str]:
    lowered = question.lower()
    if "user" in lowered or "learner" in lowered or "student" in lowered:
        return {"user", "users", "user_id", "activity_user_id", "student", "student_id", "learner"}
    if "course" in lowered:
        return {"course", "course_id"}
    if "event" in lowered or "activity" in lowered or "row" in lowered:
        return {"activity", "event", "record", "row"}
    return set(_intent_terms(question))


def _is_measure_column(column: str, measure_terms: set[str]) -> bool:
    lowered = column.lower()
    if lowered.endswith("_payload_json") or "name" in lowered or lowered in {"email", "full_name"}:
        return False
    if {"user", "users", "student", "learner"} & measure_terms:
        return lowered in {"user_id", "activity_user_id", "student_id", "learner_id"} or lowered.endswith("_user_id")
    terms = _field_terms(column)
    if not terms:
        return False
    if measure_terms & terms:
        return True
    return any(term in lowered for term in measure_terms)


def _is_dimension_column(column: str) -> bool:
    lowered = column.lower()
    if lowered.endswith("_payload_json") or lowered in {"activity_payload_json", "user_payload_json"}:
        return False
    if lowered in {"user_id", "activity_user_id", "record_index", "activity_record_index"}:
        return False
    return any(
        token in lowered
        for token in ("name", "school", "institute", "department", "province", "course", "category", "type", "status")
    )


def _ranked_dimension_score(column: str, intent_terms: set[str]) -> float:
    score = _field_score(column, intent_terms)
    field_terms = _field_terms(column)
    if "name" in field_terms:
        score += 1.5
    if {"school", "institute"} & intent_terms and {"school", "institute"} & field_terms:
        score += 4.0
    lowered = column.lower()
    if "course" in intent_terms:
        if lowered in {"course_id", "subject_name"}:
            score += 5.0
        if any(term in lowered for term in ("teacher", "faculty", "org")):
            score -= 3.0
    return score


def _ranked_measure_score(column: str, measure_terms: set[str]) -> float:
    score = _field_score(column, measure_terms)
    lowered = column.lower()
    if lowered.endswith("_id"):
        score += 1.0
    return score


def _ranked_dimension_has_values(con: Any, table: str, dimension: str, measure: str) -> bool:
    dimension_expr = _duckdb_quote_identifier(dimension)
    measure_expr = _duckdb_quote_identifier(measure)
    try:
        value = con.execute(
            f"""
            select count(*)
            from (
                select {dimension_expr}, count(distinct {measure_expr}) as value
                from {_duckdb_quote_identifier(table)}
                where {dimension_expr} is not null
                  and cast({dimension_expr} as varchar) <> ''
                  and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
                  and {measure_expr} is not null
                group by 1
                limit 1
            )
            """
        ).fetchone()[0]
    except Exception:
        return False
    return bool(value)


def _select_duckdb_source(con: Any, question: str) -> dict[str, Any]:
    try:
        rows = con.execute(
            """
            select source_path, source_format, source_table, records
            from source_summary
            order by records desc
            limit 80
            """
        ).fetchall()
    except Exception:
        return {}
    intent_terms = set(_intent_terms(question))
    lowered = question.lower()
    wants_activity = "activity" in lowered or "event" in lowered
    wants_user = "user" in lowered or "learner" in lowered or "student" in lowered
    wants_time = "time" in lowered or "trend" in lowered or "over time" in lowered
    scored: list[tuple[float, dict[str, Any]]] = []
    for source_path, source_format, source_table, records in rows:
        source = {
            "source_path": source_path,
            "source_format": source_format,
            "source_table": source_table,
            "records": records,
        }
        haystack = f"{source_path} {source_table}".lower()
        score = 0.0
        for term in intent_terms:
            if term in haystack:
                score += 2.0
        if wants_activity and ("activity" in haystack or "event" in haystack or "log" in haystack):
            score += 8.0
        if wants_user and ("user" in haystack or "student" in haystack or "learner" in haystack):
            score += 2.0
        if wants_time:
            score += 1.0
        try:
            score += min(float(records or 0) / 1_000_000, 3.0)
        except (TypeError, ValueError):
            pass
        fields = _infer_duckdb_json_fields(con, str(source_path))
        if fields.get("timestamp"):
            score += 6.0 if wants_time else 2.0
        if fields.get("user"):
            score += 4.0 if wants_user else 1.0
        if fields.get("event"):
            score += 4.0 if wants_activity else 1.0
        if fields.get("course") and "course" in lowered:
            score += 4.0
        if score > 0:
            scored.append((score, source))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("source_path") or "")))
    return scored[0][1] if scored else {}


def _infer_duckdb_json_fields(con: Any, source_path: str) -> dict[str, str]:
    cached = _duckdb_field_cache.get(source_path)
    if cached is not None:
        return cached
    try:
        rows = con.execute(
            """
            select payload_json
            from unified_records
            where source_path = ?
              and payload_json is not null
            limit 60
            """,
            [source_path],
        ).fetchall()
    except Exception:
        return {}
    fields: set[str] = set()
    for (payload_json,) in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            fields.update(str(key) for key in payload)
    inferred = {
        "timestamp": _choose_field(fields, ("@timestamp", "timestamp", "created_at", "created", "updated_at", "date", "time")),
        "user": _choose_field(fields, ("userID", "user_id", "userid", "username", "user", "actorID", "student_id", "learner_id")),
        "event": _choose_field(fields, ("event", "event_type", "action", "name", "verb")),
        "category": _choose_field(fields, ("eventCategory", "event_category", "category", "type")),
        "course": _choose_field(fields, ("courseID", "course_id", "course_key", "course")),
        "app": _choose_field(fields, ("appID", "app_id", "application", "app")),
    }
    _duckdb_field_cache[source_path] = inferred
    return inferred


def _choose_field(fields: set[str], preferred: tuple[str, ...]) -> str:
    lower_to_field = {field.lower(): field for field in fields}
    for candidate in preferred:
        if candidate.lower() in lower_to_field:
            return lower_to_field[candidate.lower()]
    preferred_terms = set(_intent_terms(" ".join(preferred)))
    ranked = sorted(fields, key=lambda field: (-_field_score(field, preferred_terms), field.lower()))
    if ranked and _field_score(ranked[0], preferred_terms) > 0:
        return ranked[0]
    return ""


def _json_extract_expr(field: str) -> str:
    escaped = field.replace("\\", "\\\\").replace('"', '\\"')
    return f"json_extract_string(payload_json, '$.\"{escaped}\"')"


def _timestamp_expr(field: str) -> str:
    value = _json_extract_expr(field)
    return f"try_cast(replace(substr({value}, 1, 19), 'T', ' ') as timestamp)"


def _duckdb_time_buckets(con: Any, source_path: str, timestamp_field: str, user_field: str | None) -> list[dict[str, Any]]:
    cached_rows = _duckdb_cached_time_buckets(con, source_path, timestamp_field, user_field)
    if cached_rows:
        return cached_rows
    timestamp_sql = _timestamp_expr(timestamp_field)
    user_sql = _json_extract_expr(user_field) if user_field else "NULL"
    rows = con.execute(
        f"""
        with extracted as (
            select
                date_trunc('hour', {timestamp_sql}) as bucket,
                nullif({user_sql}, '') as user_value
            from unified_records
            where source_path = ?
              and payload_json is not null
        ),
        buckets as (
            select
                bucket,
                count(*) as records,
                count(distinct user_value) as users
            from extracted
            where bucket is not null
            group by bucket
            order by bucket desc
            limit 24
        )
        select strftime(bucket, '%Y-%m-%d %H:%M') as label, records, users
        from buckets
        order by label
        """,
        [source_path],
    ).fetchall()
    return [{"label": label, "value": int(records), "users": int(users)} for label, records, users in rows]


def _duckdb_cached_time_buckets(
    con: Any,
    source_path: str,
    timestamp_field: str,
    user_field: str | None,
) -> list[dict[str, Any]]:
    try:
        exists = con.execute(
            """
            select count(*)
            from information_schema.tables
            where table_schema = 'main'
              and table_name = 'dashboard_agent_activity_hourly_cache'
            """
        ).fetchone()[0]
        if not exists:
            return []
        rows = con.execute(
            """
            with ordered as (
                select
                    label,
                    coalesce(cumulative_records, records) as records,
                    coalesce(cumulative_users, users) as users,
                    row_number() over (order by label) as rn,
                    count(*) over () as total_rows
                from dashboard_agent_activity_hourly_cache
                where source_path = ?
                  and timestamp_field = ?
                  and coalesce(user_field, '') = ?
            ),
            sampled as (
                select
                    *,
                    case
                        when total_rows <= 24 then rn
                        when rn = 1 then 1
                        when rn = total_rows then 24
                        else 2 + cast(floor(((rn - 2) * 22.0) / greatest(total_rows - 2, 1)) as integer)
                    end as sample_bucket
                from ordered
            ),
            bucketed as (
                select
                    *,
                    row_number() over (
                        partition by sample_bucket
                        order by
                            case when sample_bucket = 1 then rn end asc,
                            case when sample_bucket = 24 then rn end desc,
                            rn asc
                    ) as bucket_rank
                from sampled
            )
            select label, records, users
            from bucketed
            where bucket_rank = 1
            order by label
            """,
            [source_path, timestamp_field, user_field or ""],
        ).fetchall()
    except Exception:
        return []
    return [{"label": label, "value": int(records), "users": int(users)} for label, records, users in rows]


def _duckdb_table_exists(con: Any, table_name: str) -> bool:
    try:
        return bool(
            con.execute(
                """
                select count(*)
                from information_schema.tables
                where table_schema = 'main'
                  and table_name = ?
                """,
                [table_name],
            ).fetchone()[0]
        )
    except Exception:
        return False


def _duckdb_top_counts(con: Any, source_path: str, field: str, limit: int = 8) -> list[dict[str, Any]]:
    value_sql = _json_extract_expr(field)
    rows = con.execute(
        f"""
        select nullif({value_sql}, '') as label, count(*) as value
        from unified_records
        where source_path = ?
          and payload_json is not null
        group by label
        having label is not null
        order by value desc, label
        limit {int(limit)}
        """,
        [source_path],
    ).fetchall()
    return [{"label": str(label), "value": int(value)} for label, value in rows]


def _duckdb_distinct_count(con: Any, source_path: str, field: str) -> int:
    value_sql = _json_extract_expr(field)
    try:
        value = con.execute(
            f"""
            select count(distinct nullif({value_sql}, ''))
            from unified_records
            where source_path = ?
              and payload_json is not null
            """,
            [source_path],
        ).fetchone()[0]
    except Exception:
        return 0
    return int(value or 0)


def _duckdb_distinct_activity_users(con: Any) -> int:
    if _duckdb_table_exists(con, "dashboard_agent_activity_joined"):
        try:
            value = con.execute(
                """
                select count(distinct activity_user_id)
                from dashboard_agent_activity_joined
                where activity_user_id is not null
                  and activity_user_id <> ''
                """
            ).fetchone()[0]
            return int(value or 0)
        except Exception:
            pass
    if _duckdb_table_exists(con, "dashboard_agent_activity_hourly_cache"):
        try:
            value = con.execute("select max(users) from dashboard_agent_activity_hourly_cache").fetchone()[0]
            return int(value or 0)
        except Exception:
            pass
    return 0


def _duckdb_sample_records(con: Any, source_path: str, limit: int = 12) -> list[dict[str, Any]]:
    if (
        source_path == "edx-elastic/ae-activity-data-stream.json"
        and _duckdb_table_exists(con, "dashboard_agent_activity_joined")
    ):
        rows = con.execute(
            """
            select
                activity_record_index,
                event_at,
                activity_user_id,
                user_id,
                username,
                email,
                full_name,
                school_name,
                school_province,
                course_id,
                subject_name,
                department_name,
                course_type,
                enrolled_users,
                user_course_activity_count,
                learning_status,
                app_id,
                event_category,
                event_name,
                session_id
            from dashboard_agent_activity_joined
            order by activity_record_index
            limit ?
            """,
            [int(limit)],
        ).fetchall()
        columns = [
            "activity_record_index",
            "event_at",
            "activity_user_id",
            "user_id",
            "username",
            "email",
            "full_name",
            "school_name",
            "school_province",
            "course_id",
            "subject_name",
            "department_name",
            "course_type",
            "enrolled_users",
            "user_course_activity_count",
            "learning_status",
            "app_id",
            "event_category",
            "event_name",
            "session_id",
        ]
        return [
            {
                "source": "dashboard_agent_activity_joined",
                "index": index,
                **{key: value for key, value in zip(columns, row)},
            }
            for index, row in enumerate(rows)
        ]
    rows = con.execute(
        """
        select payload_json
        from unified_records
        where source_path = ?
          and payload_json is not null
        limit ?
        """,
        [source_path, int(limit)],
    ).fetchall()
    records: list[dict[str, Any]] = []
    for index, (payload_json,) in enumerate(rows):
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append({"source": source_path, "index": index, **payload})
    return records


def _activity_dashboard_context(store: Any, results: list[dict[str, Any]], question: str = "") -> dict[str, Any]:
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
                "full_distinct_time_buckets_json",
                "full_user_time_buckets_json",
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
    full_distinct_time_buckets = _json_rows_by_field(aggregate_meta.get("full_distinct_time_buckets_json"))
    full_user_time_buckets = _json_rows(aggregate_meta.get("full_user_time_buckets_json"))
    if full_user_time_buckets and not full_distinct_time_buckets:
        full_distinct_time_buckets = {"user": full_user_time_buckets}
    if not records and not full_counts:
        return {}

    chart_plan = _design_activity_chart_plan(
        records,
        question=question,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
        full_distinct_time_buckets=full_distinct_time_buckets,
        full_user_time_buckets=full_user_time_buckets,
    )
    chart_slots = _bind_activity_chart_slots(
        chart_plan,
        records,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
        full_distinct_time_buckets=full_distinct_time_buckets,
        full_user_time_buckets=full_user_time_buckets,
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


def _design_generic_chart_plan(
    records: list[dict[str, Any]],
    *,
    question: str = "",
    full_counts: dict[str, list[dict[str, Any]]] | None = None,
    full_time_buckets: list[dict[str, Any]] | None = None,
    full_distinct_time_buckets: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    full_counts = full_counts or {}
    full_distinct_time_buckets = full_distinct_time_buckets or {}
    if not full_counts and not full_time_buckets and not full_distinct_time_buckets:
        return []

    intent_terms = set(_intent_terms(question))
    plan: list[dict[str, Any]] = []

    distinct_fields = _rank_fields(full_distinct_time_buckets, intent_terms)
    for field in distinct_fields[:3]:
        field_label = _humanize_field(field)
        plan.append(
            {
                "id": f"distinctTime-{_slot_safe_field(field)}",
                "title": f"{field_label} over time",
                "chartType": "line",
                "field": f"{field}@time",
                "sourceField": field,
                "aggregate": "distinct_time",
                "reason": "A line chart shows distinct values for this matched field by time bucket.",
            }
        )

    if full_time_buckets:
        plan.append(
            {
                "id": "activityTimeline",
                "title": "Activity over time",
                "chartType": "line",
                "field": "@time",
                "reason": "A line chart shows record volume by detected time bucket.",
            }
        )

    for field in _rank_fields(full_counts, intent_terms)[:6]:
        rows = full_counts.get(field) or []
        if not rows:
            continue
        field_label = _humanize_field(field)
        chart_type = "stat" if len(rows) == 1 else "horizontal_bar"
        plan.append(
            {
                "id": f"fieldCounts-{_slot_safe_field(field)}",
                "title": f"{field_label} distribution",
                "chartType": chart_type,
                "field": field,
                "sourceField": field,
                "aggregate": "count",
                "reason": "A distribution chart compares the most frequent values for this matched field.",
            }
        )

    return plan[:6]


def _rank_fields(rows_by_field: dict[str, Any], intent_terms: set[str]) -> list[str]:
    def sort_key(field: str) -> tuple[float, int, str]:
        rows = rows_by_field.get(field)
        row_count = len(rows) if isinstance(rows, list) else 0
        return (-_field_score(field, intent_terms), -row_count, field.lower())

    return sorted((field for field in rows_by_field if isinstance(field, str)), key=sort_key)


def _design_activity_chart_plan(
    records: list[dict[str, Any]],
    *,
    question: str = "",
    full_counts: dict[str, list[dict[str, Any]]] | None = None,
    full_time_buckets: list[dict[str, Any]] | None = None,
    full_distinct_time_buckets: dict[str, list[dict[str, Any]]] | None = None,
    full_user_time_buckets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    generic_plan = _design_generic_chart_plan(
        records,
        question=question,
        full_counts=full_counts,
        full_time_buckets=full_time_buckets,
        full_distinct_time_buckets=full_distinct_time_buckets,
    )
    if generic_plan:
        return generic_plan

    candidate_fields = {
        "event": _top_counts(records, "event"),
        "eventCategory": _top_counts(records, "eventCategory"),
        "courseID": _top_counts(records, "courseID"),
        "appID": _top_counts(records, "appID"),
        "userID": _top_counts(records, "userID"),
    }
    if full_counts:
        candidate_fields.update(full_counts)
    plan: list[dict[str, Any]] = []
    if full_user_time_buckets:
        plan.append(
            {
                "id": "userTimeline",
                "title": "Users over time",
                "chartType": "line",
                "field": "userID@timestamp",
                "reason": "A line chart fits distinct user counts by timestamp bucket.",
            }
        )
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
    if candidate_fields.get("userID"):
        plan.append(
            {
                "id": "users",
                "title": "Top users by activity",
                "chartType": "horizontal_bar",
                "field": "userID",
                "reason": "User identifiers are long labels, so horizontal bars are easiest to scan.",
            }
        )
    return plan[:6]


def _bind_activity_chart_slots(
    chart_plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    full_counts: dict[str, list[dict[str, Any]]] | None = None,
    full_time_buckets: list[dict[str, Any]] | None = None,
    full_distinct_time_buckets: dict[str, list[dict[str, Any]]] | None = None,
    full_user_time_buckets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    slots = []
    for item in chart_plan:
        field = str(item["field"])
        if item.get("aggregate") == "distinct_time":
            data = (full_distinct_time_buckets or {}).get(str(item.get("sourceField") or field), [])
        elif item["id"] == "userTimeline":
            data = full_user_time_buckets or []
        elif item["chartType"] == "line":
            data = full_time_buckets or _time_buckets(records)
        else:
            data = (full_counts or {}).get(field) or _top_counts(records, field)
        slots.append({**item, "data": data})
    return slots


def _retitle_slots_for_prompt(question: str, chart_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered = question.lower()
    wants_user = "user" in lowered or "learner" in lowered or "student" in lowered
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    wants_course = "course" in lowered
    wants_event = "event" in lowered or "activity" in lowered or "click" in lowered
    retitled: list[dict[str, Any]] = []
    for slot in chart_slots:
        updated = dict(slot)
        slot_id = str(updated.get("id", ""))
        if slot_id == "userTimeline" and wants_user and wants_time:
            updated["title"] = "Users over time"
        elif slot_id == "activityTimeline":
            if wants_time:
                updated["title"] = "Activity trend over time"
        elif slot_id == "users" and wants_user:
            updated["title"] = "Most active users"
        elif slot_id == "courses" and wants_course:
            updated["title"] = "Course activity"
        elif slot_id == "events" and wants_event:
            updated["title"] = "Activity event mix"
        retitled.append(updated)
    return retitled


def _activity_layout_spec(question: str, activity: dict[str, Any]) -> dict[str, Any]:
    chart_slots = activity.get("chartSlots") if isinstance(activity.get("chartSlots"), list) else []
    available_slots = {
        str(slot.get("id"))
        for slot in chart_slots
        if isinstance(slot, dict) and slot.get("id") and slot.get("data")
    }
    lowered = question.lower()
    if (
        ("user" in lowered or "learner" in lowered or "student" in lowered)
        and ("time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered)
        and "userTimeline" not in available_slots
    ):
        available_slots.discard("activityTimeline")
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    metrics = {
        "totalRecords": summary.get("totalRecords") or summary.get("sampleRecords"),
        "sampleRecords": summary.get("sampleRecords"),
        "distinctEvents": summary.get("distinctEvents"),
        "distinctCourses": summary.get("distinctCourses"),
        "distinctUsers": summary.get("distinctUsers"),
    }
    available_metrics = {key for key, value in metrics.items() if value not in (None, "", 0)}
    fallback = _fallback_activity_layout_spec(question, available_slots, available_metrics)
    if summary.get("isFullAggregate") and available_slots:
        return fallback
    llm_spec = _llm_activity_layout_spec(question, chart_slots, summary, available_metrics)
    return _sanitize_activity_layout_spec(llm_spec, available_slots, available_metrics, fallback)


def _fallback_activity_layout_spec(
    question: str,
    available_slots: set[str],
    available_metrics: set[str],
) -> dict[str, Any]:
    lowered = question.lower()
    wants_user = "user" in lowered or "learner" in lowered or "student" in lowered
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    wants_course = "course" in lowered
    wants_event = "event" in lowered or "activity" in lowered or "click" in lowered

    def metric(metric_id: str, span: int = 1) -> dict[str, Any]:
        return {"type": "metric", "id": metric_id, "span": span}

    def chart(slot_id: str, span: int = 1) -> dict[str, Any]:
        return {"type": "chart", "slotId": slot_id, "span": span}

    blocks: list[dict[str, Any]] = []
    title = "Activity dashboard"
    subtitle = "Dashboard selected from the fields and aggregates matched to the prompt."
    if wants_user and wants_time:
        title = "Users over time"
        subtitle = (
            "Shows distinct users by time bucket from the full aggregate."
            if "userTimeline" in available_slots
            else "User time buckets are not in the current aggregate cache; refresh ingest to build this trend."
        )
        for metric_id in ("distinctUsers", "totalRecords"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        if "userTimeline" in available_slots:
            blocks.append(chart("userTimeline", 2))
        if "users" in available_slots:
            blocks.append(chart("users", 2 if "userTimeline" not in available_slots else 1))
    elif wants_user:
        title = "User activity"
        for metric_id in ("distinctUsers", "totalRecords"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        if "users" in available_slots:
            blocks.append(chart("users", 2))
        if "activityTimeline" in available_slots:
            blocks.append(chart("activityTimeline", 2))
    elif wants_course:
        title = "Course activity"
        for metric_id in ("distinctCourses", "totalRecords"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        if "courses" in available_slots:
            blocks.append(chart("courses", 2))
        if "activityTimeline" in available_slots:
            blocks.append(chart("activityTimeline", 2))
    elif wants_event:
        title = "Activity events"
        for metric_id in ("totalRecords", "distinctEvents"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in ("activityTimeline", "events", "categories"):
            if slot_id in available_slots:
                blocks.append(chart(slot_id, 2 if slot_id == "activityTimeline" else 1))
    else:
        for metric_id in ("totalRecords", "distinctEvents", "distinctCourses", "distinctUsers"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in ("userTimeline", "activityTimeline", "events", "courses", "categories", "users", "apps"):
            if slot_id in available_slots:
                blocks.append(chart(slot_id, 2 if slot_id in {"userTimeline", "activityTimeline"} else 1))

    if not blocks:
        blocks = [chart(slot_id, 2 if slot_id in {"userTimeline", "activityTimeline"} else 1) for slot_id in sorted(available_slots)]
    blocks.append({"type": "source", "span": 2})
    return {"title": title, "subtitle": subtitle, "blocks": blocks[:8]}


def _llm_activity_layout_spec(
    question: str,
    chart_slots: list[dict[str, Any]],
    summary: dict[str, Any],
    available_metrics: set[str],
) -> dict[str, Any] | None:
    if get_settings().llm_mode == "never":
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    from langchain_openai import ChatOpenAI

    slot_context = [
        {
            "id": slot.get("id"),
            "title": slot.get("title"),
            "chartType": slot.get("chartType"),
            "field": slot.get("field"),
            "rows": len(slot.get("data") or []),
        }
        for slot in chart_slots
        if isinstance(slot, dict)
    ]
    messages = [
        (
            "system",
            (
                "You are choosing a dashboard layout from existing chart slots. "
                "Return only JSON with title, subtitle, and blocks. "
                "Blocks must be one of: {\"type\":\"metric\",\"id\":metricId,\"span\":1}, "
                "{\"type\":\"chart\",\"slotId\":slotId,\"span\":1 or 2}, {\"type\":\"source\",\"span\":1 or 2}. "
                "Use only supplied metric ids and chart slot ids. Prefer the user's requested metric over a fixed overview."
            ),
        ),
        (
            "human",
            json.dumps(
                {
                    "question": question,
                    "availableMetricIds": sorted(available_metrics),
                    "chartSlots": slot_context,
                    "summary": summary,
                },
                ensure_ascii=False,
                default=str,
            ),
        ),
    ]
    for candidate in candidates[:1]:
        try:
            model = ChatOpenAI(
                api_key=candidate["api_key"],
                model=candidate["model"],
                base_url=candidate.get("base_url"),
                default_headers=candidate.get("headers") or None,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            response = model.invoke(messages)
            content = str(response.content)
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _sanitize_activity_layout_spec(
    spec: dict[str, Any] | None,
    available_slots: set[str],
    available_metrics: set[str],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    source = spec if isinstance(spec, dict) else fallback
    blocks: list[dict[str, Any]] = []
    for block in source.get("blocks", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        span = 2 if block.get("span") == 2 else 1
        if block_type == "metric" and block.get("id") in available_metrics:
            blocks.append({"type": "metric", "id": block["id"], "span": span})
        elif block_type == "chart" and block.get("slotId") in available_slots:
            blocks.append({"type": "chart", "slotId": block["slotId"], "span": span})
        elif block_type == "source":
            blocks.append({"type": "source", "span": span})
        elif block_type == "records":
            blocks.append({"type": "records", "span": 2})
    if not blocks:
        blocks = fallback.get("blocks", [])
    title = str(source.get("title") or fallback.get("title") or "Activity dashboard").strip()
    subtitle = str(source.get("subtitle") or fallback.get("subtitle") or "").strip()
    return {"title": title[:80], "subtitle": subtitle[:180], "blocks": blocks[:10]}


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

    cache_results = _aggregate_cache_results(store, question)
    results = cache_results + store.search(question, limit=24)
    if _wants_dashboard(question):
        activity = (
            _aggregate_cache_activity(store, question)
            or _graph_ranked_dimension_context(store, question)
            or _graph_time_series_activity_context(store, question)
            or _duckdb_ranked_dimension_context(store, question)
            or _duckdb_activity_context(store, question)
            or _activity_dashboard_context(store, results, question)
        )
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
