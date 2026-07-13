from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from dashboard_agent.config import Settings
from dashboard_agent.dashboard_planner import build_complex_dashboard
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
HUMAN_LABEL_RE = re.compile(r"([a-z0-9])([A-Z])")
HUMAN_KEEP_ALL_CAPS = {"API", "CSV", "DB", "ETAG", "ID", "JSON", "LLM", "SQL", "S3", "UI", "URL", "UTC"}
INTENT_SYNONYMS = {
    "finish": {"pass", "passed", "complete", "completed", "completion", "status", "result"},
    "finished": {"pass", "passed", "complete", "completed", "completion", "status", "result"},
    "complete": {"pass", "passed", "finished", "completion", "status", "result"},
    "completed": {"pass", "passed", "finished", "complete", "completion", "status", "result"},
    "pass": {"passed", "complete", "completed", "finish", "finished", "status"},
    "passed": {"pass", "complete", "completed", "finish", "finished", "status"},
    "read": {"reading", "learn", "learning", "content", "course"},
    "reading": {"read", "learn", "learning", "content", "course"},
    "learn": {"learning", "read", "reading", "course"},
    "learning": {"learn", "read", "reading", "course", "status"},
    "education": {"level", "grade"},
    "level": {"education", "grade"},
    "teacher": {"instructor", "faculty"},
    "instructor": {"teacher", "faculty"},
    "faculty": {"teacher", "instructor"},
    "institute": {"institution", "school", "organization", "name"},
    "institution": {"institute", "school", "organization", "name"},
    "school": {"institute", "institution", "organization", "name"},
    "id": {"ids"},
    "ids": {"id"},
}
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


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return ""


def _last_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return _message_text(content)
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
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 2
        and term
        not in {
            "a",
            "an",
            "and",
            "are",
            "can",
            "for",
            "have",
            "of",
            "over",
            "please",
            "show",
            "that",
            "the",
            "time",
            "to",
            "total",
            "trend",
            "with",
            "number",
            "count",
        }
    ]
    expanded = list(terms)
    seen = set(expanded)
    for term in terms:
        for synonym in INTENT_SYNONYMS.get(term, set()):
            if synonym not in seen:
                seen.add(synonym)
                expanded.append(synonym)
    return expanded


QUERY_DIMENSION_STOP_TERMS = {
    "activity",
    "all",
    "ask",
    "breakdown",
    "chart",
    "compare",
    "complete",
    "completed",
    "count",
    "dashboard",
    "distribution",
    "finish",
    "finished",
    "graph",
    "highest",
    "largest",
    "learn",
    "many",
    "max",
    "most",
    "number",
    "pass",
    "passed",
    "rank",
    "reading",
    "result",
    "show",
    "total",
    "user",
    "users",
    "learner",
    "learners",
    "student",
    "students",
    "value",
    "values",
}


def _query_dimension_phrases(question: str) -> list[str]:
    lowered = f" {question.lower()} "
    phrase_patterns = [
        r"\b(?:grouped by|group by|split by|breakdown by|compare by|by|per)\s+([a-z0-9 _-]+)",
        r"\bdistribution of\s+([a-z0-9 _-]+)",
        r"\bshow\s+([a-z0-9 _-]+?)\s+distribution\b",
        r"\b([a-z0-9 _-]+?)\s+distribution\b",
        r"\b(?:which|what)\s+([a-z0-9 _-]+?)\s+(?:has|have|had|contains?|includes?)\b",
        r"\bcompare\s+([a-z0-9 _-]+)",
        r"\bshow\s+([a-z0-9 _-]+?)\s+by\b",
        r"\b(?:top|most|highest|largest)\s+([a-z0-9 _-]+?)\s+by\b",
    ]
    phrases: list[str] = []
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, lowered):
            phrase = re.split(
                r"\b(?:after|and|for|from|having|that|to|where|when|which|who|with)\b",
                match.group(1),
                maxsplit=1,
            )[0]
            phrase = re.sub(r"\b(?:user|users|student|students|learner|learners|number|count|total)\b", " ", phrase)
            phrase = re.sub(r"[^a-z0-9 _-]+", " ", phrase)
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _query_dimension_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for phrase in _query_dimension_phrases(question):
        terms.update(_intent_terms(phrase))
    if terms:
        return terms - QUERY_DIMENSION_STOP_TERMS
    fallback = set(_intent_terms(question)) - QUERY_DIMENSION_STOP_TERMS
    return fallback


def _primary_group_dimension_terms(question: str) -> set[str]:
    lowered = f" {question.lower()} "
    phrases: list[str] = []
    for pattern in (
        r"\b(?:grouped by|group by|split by|breakdown by|by|per)\s+([a-z0-9 _-]+)",
    ):
        for match in re.finditer(pattern, lowered):
            phrase = re.split(
                r"\b(?:after|and|compare|for|from|having|that|to|where|when|which|who|with)\b",
                match.group(1),
                maxsplit=1,
            )[0]
            phrase = re.sub(r"\b(?:user|users|student|students|learner|learners|number|count|total)\b", " ", phrase)
            phrase = re.sub(r"[^a-z0-9 _-]+", " ", phrase)
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase:
                phrases.append(phrase)
    terms: set[str] = set()
    for phrase in phrases:
        terms.update(_intent_terms(phrase))
    return terms - QUERY_DIMENSION_STOP_TERMS


def _primary_ranked_dimension_terms(question: str) -> set[str]:
    lowered = f" {question.lower()} "
    patterns = [
        r"\b(?:which|what)\s+([a-z0-9 _-]+?)\s+(?:has|have|had|contains?|includes?)\b",
        r"\b(?:top|most|highest|largest)\s+([a-z0-9 _-]+?)\s+by\b",
        r"\bshow\s+([a-z0-9 _-]+?)\s+by\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        phrase = re.split(
            r"\b(?:after|and|compare|for|from|having|that|to|where|when|which|who|with)\b",
            match.group(1),
            maxsplit=1,
        )[0]
        phrase = re.sub(r"\b(?:user|users|student|students|learner|learners|number|count|total)\b", " ", phrase)
        phrase = re.sub(r"[^a-z0-9 _-]+", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        terms = set(_intent_terms(phrase)) - QUERY_DIMENSION_STOP_TERMS
        if terms:
            return terms
    return set()


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


def _dimension_field_score(field: str, question: str, intent_terms: set[str] | None = None) -> float:
    intent_terms = intent_terms or set(_intent_terms(question))
    score = _field_score(field, intent_terms)
    focus_terms = _query_dimension_terms(question)
    if focus_terms:
        field_terms = _field_terms(field)
        focus_overlap = field_terms & focus_terms
        score += float(len(focus_overlap) * 10)
        field_text = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", field).lower()))
        for phrase in _query_dimension_phrases(question):
            phrase_terms = set(_intent_terms(phrase))
            if phrase_terms and phrase_terms <= (field_terms | focus_terms) and phrase_terms & field_terms:
                score += float(len(phrase_terms & field_terms) * 6)
            phrase_text = " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))
            if phrase_text and phrase_text in field_text:
                score += 20.0
    return score


def _query_filter_intent_terms(question: str) -> set[str]:
    lowered = question.lower()
    raw_terms = set(re.findall(r"[a-z0-9]+", lowered))
    terms = set(_intent_terms(question))
    filter_terms = {
        "active",
        "certificate",
        "certified",
        "complete",
        "completed",
        "completion",
        "finish",
        "finished",
        "inactive",
        "pass",
        "passed",
        "result",
    }
    if "distribution" in lowered and "pass" in raw_terms and not (raw_terms & {"passed", "finish", "finished", "complete", "completed"}):
        terms -= {"pass", "passed", "complete", "completed", "finish", "finished"}
    return terms & filter_terms


def _needs_duckdb_filter(question: str) -> bool:
    return bool(_query_filter_intent_terms(question))


def _should_show_unfiltered_distribution(question: str, field: str) -> bool:
    field_terms = _field_terms(field)
    filter_terms = _query_filter_intent_terms(question)
    if filter_terms and field_terms & {"status", "result", "pass", "passed", "complete", "completed", "certificate", "learning"}:
        return True
    lowered = question.lower()
    wants_distribution = any(term in lowered for term in ("distribution", "breakdown", "split", "compare", "group"))
    return wants_distribution and _field_score(field, set(_intent_terms(question))) > 0


def _chart_type_for_count_field(field: str, rows: list[dict[str, Any]], question: str) -> str:
    if len(rows) <= 1:
        return "stat"
    lowered = question.lower()
    field_terms = _field_terms(field)
    wants_composition = any(term in lowered for term in ("composition", "mix", "percent", "percentage", "proportion", "share"))
    wants_distribution = any(term in lowered for term in ("distribution", "breakdown", "split"))
    label_lengths = [len(str(row.get("label") or "")) for row in rows[:8]]
    max_label_length = max(label_lengths or [0])
    row_count = len(rows)
    if row_count <= 6 and (wants_composition or wants_distribution or field_terms & {"status", "category", "type"}):
        return "donut"
    if row_count >= 10 and (wants_composition or "top" in lowered or "most" in lowered):
        return "treemap"
    if max_label_length <= 18 and row_count <= 8:
        return "column"
    return "horizontal_bar"


def _ranked_chart_type_for_dimension(dimension: str, rows: list[dict[str, Any]], question: str) -> str:
    chart_type = _chart_type_for_count_field(dimension, rows, question)
    if chart_type == "donut" and not any(term in question.lower() for term in ("composition", "mix", "percent", "percentage", "proportion", "share")):
        return "column"
    return chart_type if chart_type in {"column", "treemap", "horizontal_bar"} else "horizontal_bar"


def _ranked_filter_sql(con: Any, table: str, question: str) -> tuple[str, list[Any], list[str]]:
    filter_terms = _query_filter_intent_terms(question)
    if not filter_terms:
        return "", [], []
    try:
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
    except Exception:
        return "", [], []
    clauses: list[str] = []
    params: list[Any] = []
    labels: list[str] = []
    completion_terms = filter_terms & {"complete", "completed", "completion", "finish", "finished", "pass", "passed", "status", "result"}
    active_terms = filter_terms & {"active", "inactive"}
    for column in columns:
        column_terms = _field_terms(column)
        column_expr = _duckdb_quote_identifier(column)
        lowered = column.lower()
        if completion_terms and column_terms & {"status", "result", "learning"}:
            try:
                values = con.execute(
                    f"""
                    select distinct lower(cast({column_expr} as varchar)) as value
                    from {_duckdb_quote_identifier(table)}
                    where {column_expr} is not null
                      and cast({column_expr} as varchar) <> ''
                    limit 100
                    """
                ).fetchall()
            except Exception:
                values = []
            matched_values = [
                str(value)
                for (value,) in values
                if value is not None and (_field_terms(str(value)) & completion_terms)
            ]
            if matched_values:
                placeholders = ", ".join("?" for _ in matched_values)
                clauses.append(f"lower(cast({column_expr} as varchar)) in ({placeholders})")
                params.extend(matched_values)
                labels.append(f"{_humanize_field(column)} in {', '.join(matched_values[:4])}")
        if active_terms and lowered in {"is_active", "active", "enroll_active"}:
            expected = 0 if "inactive" in active_terms else 1
            clauses.append(f"try_cast({column_expr} as integer) = ?")
            params.append(expected)
            labels.append(f"{_humanize_field(column)} = {expected}")
    if not clauses:
        return "", [], []
    return " and (" + " or ".join(f"({clause})" for clause in clauses) + ")", params, labels


def _json_source_filter_sql(con: Any, source_path: str, question: str) -> tuple[str, list[Any], list[str]]:
    filter_terms = _query_filter_intent_terms(question)
    if not filter_terms:
        return "", [], []
    fields = _duckdb_json_field_names(con, source_path)
    if not fields:
        return "", [], []
    clauses: list[str] = []
    params: list[Any] = []
    labels: list[str] = []
    completion_terms = filter_terms & {"complete", "completed", "completion", "finish", "finished", "pass", "passed", "status", "result"}
    active_terms = filter_terms & {"active", "inactive"}
    for field in fields:
        field_terms = _field_terms(field)
        value_expr = _json_extract_expr(field)
        lowered = field.lower()
        if completion_terms and field_terms & {"status", "result", "learning"}:
            try:
                values = con.execute(
                    f"""
                    select distinct lower(cast({value_expr} as varchar)) as value
                    from unified_records
                    where source_path = ?
                      and payload_json is not null
                      and {value_expr} is not null
                      and cast({value_expr} as varchar) <> ''
                    limit 100
                    """,
                    [source_path],
                ).fetchall()
            except Exception:
                values = []
            matched_values = [
                str(value)
                for (value,) in values
                if value is not None and (_field_terms(str(value)) & completion_terms)
            ]
            if matched_values:
                placeholders = ", ".join("?" for _ in matched_values)
                clauses.append(f"lower(cast({value_expr} as varchar)) in ({placeholders})")
                params.extend(matched_values)
                labels.append(f"{_humanize_field(field)} in {', '.join(matched_values[:4])}")
        if active_terms and lowered in {"is_active", "active", "enroll_active"}:
            expected = 0 if "inactive" in active_terms else 1
            clauses.append(f"try_cast({value_expr} as integer) = ?")
            params.append(expected)
            labels.append(f"{_humanize_field(field)} = {expected}")
    if not clauses:
        return "", [], []
    return " and (" + " or ".join(f"({clause})" for clause in clauses) + ")", params, labels


def _humanize_field(field: str) -> str:
    text = str(field or "").strip()
    if not text:
        return "Unknown"
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = re.sub(r"^\$\.?", "", text)
    text = re.sub(r"\.(json|csv|tsv|parquet|ndjson|jsonl|sql|db|duckdb|txt)$", "", text, flags=re.IGNORECASE)
    text = HUMAN_LABEL_RE.sub(r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
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


def _humanize_source_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown Source"
    text = re.sub(r"^s3://[^/]+/", "", text.replace("\\", "/"))
    text = re.sub(r"^duckdb/[^/]+/", "", text)
    parts = [part for part in text.split("/") if part]
    if not parts:
        return _humanize_field(text)
    basename = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""
    label = _humanize_field(basename)
    parent_label = _humanize_field(parent) if parent and parent.lower() not in {"data", "json", "parquet", "duckdb"} else ""
    return f"{parent_label} - {label}" if parent_label and parent_label not in label else label


def _slot_safe_field(field: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", field).strip("-")
    return value[:48] or "field"


def _duckdb_quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _duckdb_database_path(store: Any) -> Path | None:
    candidates: list[Path] = []
    store_path = getattr(store, "path", None)
    for raw_path in (store_path, get_settings().graph_path):
        if raw_path is None:
            continue
        path = Path(raw_path)
        candidates.append(path.parent / "dashboard_agent.duckdb")
        candidates.append(path.parent / "_warehouse" / "dashboard_agent.duckdb")
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "data" / "_warehouse" / "dashboard_agent.duckdb",
            cwd.parent / "data" / "_warehouse" / "dashboard_agent.duckdb",
            cwd / "data" / "dashboard_agent.duckdb",
            cwd.parent / "data" / "dashboard_agent.duckdb",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


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
    activity["decisionTrace"] = _activity_decision_trace(
        question=question,
        source_kind="aggregate cache",
        source=str(source or ""),
        source_paths=[str(source)] if source else [],
        chart_slots=chart_slots,
        layout_spec=activity["layoutSpec"],
        summary=activity["summary"],
    )
    return activity


def _duckdb_activity_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    if not any(
        term in lowered
        for term in (
            "activity",
            "breakdown",
            "compare",
            "distribution",
            "event",
            "group",
            "learner",
            "status",
            "student",
            "user",
            "course",
            "time",
            "trend",
        )
    ):
        return {}
    db_path = _duckdb_database_path(store)
    if db_path is None or not db_path.exists():
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

        extra_where, filter_params, filter_labels = _json_source_filter_sql(con, source_path, question)
        timeline = _duckdb_time_buckets(con, source_path, timestamp_field, user_field)
        counts: dict[str, list[dict[str, Any]]] = {}
        count_fields: list[str] = []
        intent_terms = set(_intent_terms(question))
        source_fields = _duckdb_json_field_names(con, source_path)
        matched_count_fields = [
            field
            for field in sorted(source_fields, key=lambda field: (-_dimension_field_score(field, question, intent_terms), field.lower()))
            if _dimension_field_score(field, question, intent_terms) > 0
        ]
        if not wants_user_time:
            count_fields = [
                field
                for field in (
                    *matched_count_fields[:6],
                    event_field,
                    category_field,
                    course_field,
                    app_field,
                    user_field,
                )
                if field
            ]
        else:
            if "event" in lowered:
                count_fields.extend(field for field in (event_field, category_field) if field)
            if "course" in lowered and course_field:
                count_fields.append(course_field)
        count_fields = list(dict.fromkeys(count_fields))[:8]
        for field in count_fields:
            if field:
                if _should_show_unfiltered_distribution(question, field):
                    counts[field] = _duckdb_top_counts(con, source_path, field)
                else:
                    counts[field] = _duckdb_top_counts(con, source_path, field, extra_where=extra_where, params=filter_params)

        records = _duckdb_sample_records(con, source_path, limit=12, extra_where=extra_where, params=filter_params)
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
        total_records = _duckdb_source_record_count(con, source_path, extra_where=extra_where, params=filter_params)
        source_paths = [source_path] if source_path else []
        source_samples = _source_sample_records(con, source_paths, limit_per_source=50)
        distinct_users = _duckdb_distinct_activity_users(con) if wants_user_time else (
            _duckdb_distinct_count(con, source_path, user_field, extra_where=extra_where, params=filter_params) if user_field else 0
        )
        activity = {
            "datasets": {
                source_path: {
                    "source": source_path,
                    "key": source_path,
                    "source_paths": source_paths,
                    "source_samples": source_samples,
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
                "sourcePaths": source_paths,
                "sourceSamples": source_samples,
                "sampleRecords": len(records),
                "totalRecords": total_records,
                "isFullAggregate": True,
                "distinctEvents": len(counts.get(event_field or "", [])),
                "distinctCourses": len(counts.get(course_field or "", [])),
                "distinctUsers": distinct_users,
                "filters": filter_labels,
            },
        }
        activity["layoutSpec"] = _activity_layout_spec(question, activity)
        activity["decisionTrace"] = _activity_decision_trace(
            question=question,
            source_kind="duckdb",
            source=source_path,
            source_paths=source_paths,
            chart_slots=chart_slots,
            layout_spec=activity["layoutSpec"],
            summary=activity["summary"],
            fields=[field for field in (timestamp_field, user_field, event_field, category_field, course_field, app_field) if field],
        )
        return activity
    except Exception:
        return {}
    finally:
        con.close()


def _graph_ranked_dimension_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "over time" in lowered or "overtime" in lowered
    if wants_time:
        return {}
    if not any(term in lowered for term in ("most", "top", "highest", "largest", "max", "rank")):
        return {}
    if _needs_duckdb_filter(question):
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
        score += _ranked_dimension_score(dimension, query_terms, question)
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
                "chartType": "area",
                "field": "records@time",
                "reason": "Graph aggregate of cumulative activity record volume by time bucket.",
                "data": records_series,
            }
        )
    if not chart_slots:
        return {}
    source_paths = _aggregate_source_paths(aggregate, question, ["time", "records", "users"])
    source_samples = _source_sample_records_from_duckdb(aggregate, source_paths)
    summary = {
        "source": aggregate.get("duckdb_table") or "graph aggregate",
        "sourcePaths": source_paths,
        "sourceSamples": source_samples,
        "sampleRecords": len(chart_slots[0].get("data") or []),
        "totalRecords": aggregate.get("total_records"),
        "isFullAggregate": True,
        "distinctUsers": aggregate.get("total_users"),
    }
    decision_trace = _time_series_decision_trace(
        question=question,
        source_kind="graph aggregate",
        table=str(summary["source"]),
        source_paths=source_paths,
        chart_slots=chart_slots,
        total_records=summary["totalRecords"],
        total_users=summary["distinctUsers"],
    )
    activity = {
        "datasets": {
            "graph/time_series": {
                "source": "graph/time_series",
                "key": "graph/time_series",
                "source_paths": source_paths,
                "source_samples": source_samples,
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
        "decisionTrace": decision_trace,
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


def _aggregate_source_paths(
    aggregate: dict[str, Any],
    question: str = "",
    fields: list[str] | None = None,
) -> list[str]:
    explicit = aggregate.get("source_paths")
    if isinstance(explicit, list):
        paths = _dedupe_source_paths([str(path) for path in explicit if path])
        if paths:
            return paths
    table = str(aggregate.get("duckdb_table") or "")
    db_paths = [aggregate.get("duckdb_database"), _duckdb_database_path(get_store())]
    for db_path in db_paths:
        if not db_path:
            continue
        paths = _source_paths_from_duckdb_database(db_path, table, question=question, fields=fields or [])
        if paths:
            return paths
    return []


def _dedupe_source_paths(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = path.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _source_paths_from_duckdb_database(
    db_path: Any,
    table: str,
    *,
    question: str = "",
    fields: list[str] | None = None,
) -> list[str]:
    if not db_path or not table:
        return []
    try:
        import duckdb
    except ImportError:
        return []
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return []
    try:
        return _source_paths_for_table(con, table, question=question, fields=fields or [])
    except Exception:
        return []
    finally:
        con.close()


def _source_paths_for_table(
    con: Any,
    table: str,
    *,
    question: str = "",
    fields: list[str] | None = None,
) -> list[str]:
    if not table:
        return []
    if _duckdb_table_exists(con, "dashboard_agent_table_lineage"):
        rows = con.execute(
            """
            select distinct source_path
            from dashboard_agent_table_lineage
            where table_name = ?
              and source_path is not null
              and source_path <> ''
            order by source_path
            """,
            [table],
        ).fetchall()
        paths = _dedupe_source_paths([str(row[0]) for row in rows])
        if paths:
            return paths

    columns = _duckdb_table_columns(con, table)
    source_columns = [column for column in columns if column == "source_path" or column.endswith("_source_path")]
    if source_columns:
        selects = [
            f"select distinct {_duckdb_quote_identifier(column)} as source_path from {_duckdb_quote_identifier(table)}"
            for column in source_columns
        ]
        rows = con.execute(
            " union ".join(selects) + " order by source_path limit 50"
        ).fetchall()
        paths = _dedupe_source_paths([str(row[0]) for row in rows if row[0]])
        if paths:
            return paths

    return _infer_source_paths_for_table(con, table, columns, question=question, fields=fields or [])


def _duckdb_table_columns(con: Any, table: str) -> list[str]:
    try:
        return [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
    except Exception:
        return []


def _infer_source_paths_for_table(
    con: Any,
    table: str,
    columns: list[str],
    *,
    question: str = "",
    fields: list[str] | None = None,
) -> list[str]:
    if not _duckdb_table_exists(con, "source_summary"):
        return []
    target_text = " ".join([question, " ".join(fields or [])]).strip()
    if not target_text:
        target_text = " ".join([table, " ".join(columns)])
    target_terms = _lineage_terms(target_text)
    dimension_text_parts = (fields or [])[:1]
    lowered_question = question.lower()
    if "institute" in lowered_question or "institution" in lowered_question:
        dimension_text_parts += ["institute institution school name"]
    dimension_terms = _lineage_terms(" ".join(dimension_text_parts))
    table_terms = _lineage_terms(table)
    if not target_terms:
        return []
    rows = con.execute(
        """
        select source_path, source_table, records
        from source_summary
        where records > 0
        order by records desc, source_path
        """
    ).fetchall()
    scored_by_path: dict[str, tuple[int, int, str]] = {}
    for source_path, source_table, records in rows:
        source_path = str(source_path)
        lineage_path = _lineage_source_path(source_path)
        source_label = _humanize_source_path(source_path)
        source_terms = _lineage_terms(f"{source_path} {source_label} {source_table or ''}")
        score = len(target_terms & source_terms)
        score += 8 * len(dimension_terms & source_terms)
        if "fact" in table_terms and "fact" in source_terms:
            score += 6
        if "dim" in source_terms:
            score += 4
        if score > 0:
            previous = scored_by_path.get(lineage_path)
            next_value = (score, int(records or 0), lineage_path)
            if previous is None:
                scored_by_path[lineage_path] = next_value
            else:
                scored_by_path[lineage_path] = (max(previous[0], score), previous[1] + int(records or 0), lineage_path)
    scored = list(scored_by_path.values())
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return _dedupe_source_paths([source_path for _score, _records, source_path in scored[:8]])


def _lineage_source_path(source_path: str) -> str:
    normalized = str(source_path or "").replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3 and parts[0] == "parquet":
        return f"{parts[0]}/{parts[1]}.parquet"
    if len(parts) == 3 and parts[2] == f"{parts[1]}.json":
        return f"{parts[0]}/{parts[2]}"
    return normalized


def _lineage_terms(text: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    terms: set[str] = set()
    for term in re.findall(r"[a-z0-9]+", normalized.lower()):
        if len(term) < 2 or term in {"over", "time", "trend", "number", "count", "total", "show", "have"}:
            continue
        terms.add(term)
        if term.endswith("s") and len(term) > 3:
            terms.add(term[:-1])
    return terms


def _ranked_sample_records_from_duckdb(
    aggregate: dict[str, Any],
    table: str,
    dimension: str,
    top_label: str,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    db_path = aggregate.get("duckdb_database")
    if not db_path:
        return []
    try:
        import duckdb
    except ImportError:
        return []
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return []
    try:
        return _ranked_sample_records(con, table, dimension, top_label, limit=limit)
    except Exception:
        return []
    finally:
        con.close()


def _source_sample_records_from_duckdb(
    aggregate: dict[str, Any],
    source_paths: list[str],
    *,
    limit_per_source: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    db_path = aggregate.get("duckdb_database")
    if not db_path or not source_paths:
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
        return _source_sample_records(con, source_paths, limit_per_source=limit_per_source)
    except Exception:
        return {}
    finally:
        con.close()


def _source_sample_records(
    con: Any,
    source_paths: list[str],
    *,
    limit_per_source: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    if not source_paths:
        return {}
    if not _duckdb_table_exists(con, "unified_records"):
        return {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for source_path in source_paths:
        rows = con.execute(
            """
            select record_index, payload_json
            from unified_records
            where source_path = ?
              and payload_json is not null
            order by record_index
            limit ?
            """,
            [source_path, int(limit_per_source)],
        ).fetchall()
        source_samples: list[dict[str, Any]] = []
        for record_index, payload_json in rows:
            try:
                payload = json.loads(str(payload_json))
            except json.JSONDecodeError:
                payload = {"payload_json": str(payload_json)}
            if isinstance(payload, dict):
                source_samples.append({"source": source_path, "record_index": int(record_index), **payload})
        if source_samples:
            samples[source_path] = source_samples
    return samples


def _ranked_sample_records(
    con: Any,
    table: str,
    dimension: str,
    top_label: str,
    *,
    limit: int = 12,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
    except Exception:
        return []
    if not columns:
        return []
    selected_columns = _sample_record_columns(columns)
    if not selected_columns:
        selected_columns = columns[:16]
    select_sql = ", ".join(_duckdb_quote_identifier(column) for column in selected_columns)
    rows = con.execute(
        f"""
        select {select_sql}
        from {_duckdb_quote_identifier(table)}
        where {_duckdb_quote_identifier(dimension)} is not null
          and cast({_duckdb_quote_identifier(dimension)} as varchar) = ?
          {extra_where}
        limit ?
        """,
        [top_label, *(params or []), int(limit)],
    ).fetchall()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = {"source": table, "index": index}
        record.update({column: _json_safe_value(value) for column, value in zip(selected_columns, row)})
        records.append(record)
    return records


def _sample_record_columns(columns: list[str]) -> list[str]:
    preferred_tokens = (
        "school",
        "institute",
        "user",
        "course",
        "subject",
        "event",
        "category",
        "province",
        "activity",
        "grade",
        "status",
        "date",
        "time",
        "name",
        "id",
    )
    scored: list[tuple[int, int, str]] = []
    for index, column in enumerate(columns):
        lowered = column.lower()
        if lowered.endswith("_payload_json"):
            continue
        score = sum(1 for token in preferred_tokens if token in lowered)
        scored.append((-score, index, column))
    return [column for _score, _index, column in sorted(scored)[:18]]


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _decision_trace_item(step: str, detail: str, evidence: list[str] | None = None) -> dict[str, Any]:
    return {
        "step": step,
        "detail": detail,
        "evidence": [str(item) for item in (evidence or []) if item is not None and str(item) != ""],
    }


def _result_type_label(
    dashboard_type: str,
    *,
    dimension_label: str | None = None,
    measure_label: str | None = None,
    chart_titles: list[str] | None = None,
) -> str:
    if dashboard_type == "ranked":
        return f"ranked comparison: {dimension_label or 'dimension'} by {measure_label or 'measure'}"
    if dashboard_type == "time_series":
        charts = ", ".join(chart_titles or [])
        return f"time-series trend{f': {charts}' if charts else ''}"
    if dashboard_type == "distribution":
        return f"distribution: {dimension_label or 'category'} by {measure_label or 'count'}"
    if dashboard_type == "activity":
        charts = ", ".join(chart_titles or [])
        return f"field-matched activity dashboard{f': {charts}' if charts else ''}"
    return dashboard_type.replace("_", " ")


def _ranked_decision_trace(
    *,
    question: str,
    source_kind: str,
    table: str,
    dimension: str,
    measure: str,
    dimension_label: str,
    measure_label: str,
    source_paths: list[str],
    chart_data: list[dict[str, Any]],
    total_measure: int,
    total_dimensions: int,
    chart_type: str,
) -> list[dict[str, Any]]:
    return [
        _decision_trace_item(
            "Interpret request",
            "Agent decided this should be a ranked aggregate because the request asks for the highest values.",
            [
                f"User request: {question}",
                f"Agent decided: {_result_type_label('ranked', dimension_label=dimension_label, measure_label=measure_label)}",
            ],
        ),
        _decision_trace_item(
            "Select data",
            f"Selected {table} from {source_kind} context because it contains the ranked dimension and measure.",
            [f"Dimension field: {dimension}", f"Measure field: {measure}", f"Rows in chart: {len(chart_data)}"],
        ),
        _decision_trace_item(
            "Collect lineage",
            "Resolved dashboard sources from DuckDB lineage metadata, source path columns, or source_summary fallback.",
            source_paths[:8],
        ),
        _decision_trace_item(
            "Choose layout",
            f"Used KPI cards plus a {chart_type.replace('_', ' ')} chart because it fits the selected ranked field shape.",
            [
                f"Top metric: {chart_data[0].get('label') if chart_data else 'not available'}",
                f"Total distinct {measure_label}: {total_measure}",
                f"Distinct {dimension_label}: {total_dimensions}",
            ],
        ),
        _decision_trace_item(
            "Bind chart",
            f"Bound {dimension_label} to chart labels and distinct {measure_label} to chart values.",
            [f"Chart type: {chart_type}", f"Values shown: {len(chart_data)}"],
        ),
    ]


def _time_series_decision_trace(
    *,
    question: str,
    source_kind: str,
    table: str,
    source_paths: list[str],
    chart_slots: list[dict[str, Any]],
    total_records: Any,
    total_users: Any,
) -> list[dict[str, Any]]:
    return [
        _decision_trace_item(
            "Interpret request",
            "Agent decided this should be a time-series dashboard because the request asks for change over time.",
            [
                f"User request: {question}",
                f"Agent decided: {_result_type_label('time_series', chart_titles=[str(slot.get('title', slot.get('id', 'chart'))) for slot in chart_slots])}",
            ],
        ),
        _decision_trace_item(
            "Select data",
            f"Selected {table} from {source_kind} context because it has precomputed time buckets.",
            [f"Activity rows: {total_records}", f"Distinct users: {total_users}"],
        ),
        _decision_trace_item(
            "Collect lineage",
            "Resolved original sources from DuckDB lineage metadata, source path columns, or source_summary fallback.",
            source_paths[:8],
        ),
        _decision_trace_item(
            "Choose layout",
            "Used KPI cards plus line chart blocks because ordered time buckets are best read as trends.",
            [f"Charts selected: {', '.join(slot.get('title', slot.get('id', 'chart')) for slot in chart_slots)}"],
        ),
        _decision_trace_item(
            "Bind chart",
            "Bound time bucket labels to the x-axis and aggregate counts to the y-axis.",
            [f"Chart count: {len(chart_slots)}"],
        ),
    ]


def _ranked_layout_spec(
    *,
    question: str,
    dimension_label: str,
    measure_label: str,
    slot_id: str,
    chart_data: list[dict[str, Any]],
    filter_labels: list[str] | None = None,
) -> dict[str, Any]:
    filter_labels = filter_labels or []
    fallback = _fallback_ranked_layout_spec(
        question=question,
        dimension_label=dimension_label,
        measure_label=measure_label,
        slot_id=slot_id,
        chart_data=chart_data,
        filter_labels=filter_labels,
    )
    llm_spec = _llm_ranked_layout_spec(
        question=question,
        dimension_label=dimension_label,
        measure_label=measure_label,
        slot_id=slot_id,
        chart_data=chart_data,
        filter_labels=filter_labels,
    )
    return _sanitize_ranked_layout_spec(llm_spec, slot_id, fallback)


def _fallback_ranked_layout_spec(
    *,
    question: str,
    dimension_label: str,
    measure_label: str,
    slot_id: str,
    chart_data: list[dict[str, Any]],
    filter_labels: list[str],
) -> dict[str, Any]:
    lowered = question.lower()
    if {"finish", "finished", "complete", "completed", "pass", "passed"} & set(_intent_terms(question)):
        focus = "Completed"
    elif filter_labels:
        focus = "Filtered"
    else:
        focus = ""
    measure_text = measure_label.lower() if focus else measure_label
    dimension_text = dimension_label.lower() if focus else dimension_label
    title = f"{focus} {measure_text} by {dimension_text}".strip()
    subtitle_parts = []
    if chart_data:
        subtitle_parts.append(f"{chart_data[0].get('label')} leads this comparison")
    if filter_labels:
        subtitle_parts.append(f"filters: {', '.join(filter_labels[:3])}")
    subtitle = ". ".join(subtitle_parts) + ("." if subtitle_parts else "")
    return {
        "title": title,
        "subtitle": subtitle,
        "chartTitle": f"{measure_label} by {dimension_label}",
        "metricLabels": {
            "topDimensionValue": f"Leading {measure_label}",
            "totalDistinctMeasure": f"Matched {measure_label}",
            "distinctDimensionValues": f"{dimension_label} groups",
        },
        "blocks": [
            {"type": "metric", "id": "topDimensionValue", "span": 1},
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            {"type": "chart", "slotId": slot_id, "span": 2},
        ],
    }


def _llm_ranked_layout_spec(
    *,
    question: str,
    dimension_label: str,
    measure_label: str,
    slot_id: str,
    chart_data: list[dict[str, Any]],
    filter_labels: list[str],
) -> dict[str, Any] | None:
    if get_settings().llm_mode == "never":
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    from langchain_openai import ChatOpenAI

    messages = [
        (
            "system",
            (
                "You are planning a dashboard for the user's actual analytical request. "
                "Return only JSON with title, subtitle, chartTitle, metricLabels, and blocks. "
                "Do not use a fixed template like 'Top X by Y' unless the user's wording requires exactly that. "
                "Use concise human labels derived from the request, filters, dimension, and measure. "
                "Allowed metric ids: topDimensionValue, totalDistinctMeasure, distinctDimensionValues. "
                f"Allowed chart slot id: {slot_id}. "
                "Blocks must use only these ids and this chart slot."
            ),
        ),
        (
            "human",
            json.dumps(
                {
                    "question": question,
                    "dimension": dimension_label,
                    "measure": measure_label,
                    "filters": filter_labels,
                    "chartSlotId": slot_id,
                    "topRows": chart_data[:5],
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
            parsed = json.loads(str(response.content))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _sanitize_ranked_layout_spec(
    spec: dict[str, Any] | None,
    slot_id: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return fallback
    title = str(spec.get("title") or fallback["title"]).strip() or fallback["title"]
    subtitle = str(spec.get("subtitle") or fallback.get("subtitle") or "").strip()
    chart_title = str(spec.get("chartTitle") or fallback.get("chartTitle") or "").strip()
    metric_labels = spec.get("metricLabels") if isinstance(spec.get("metricLabels"), dict) else {}
    allowed_metrics = {"topDimensionValue", "totalDistinctMeasure", "distinctDimensionValues"}
    blocks: list[dict[str, Any]] = []
    raw_blocks = spec.get("blocks") if isinstance(spec.get("blocks"), list) else []
    for block in raw_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        span = 2 if block.get("span") == 2 else 1
        if block_type == "metric" and block.get("id") in allowed_metrics:
            blocks.append({"type": "metric", "id": block["id"], "span": span})
        elif block_type == "chart" and block.get("slotId") == slot_id:
            blocks.append({"type": "chart", "slotId": slot_id, "span": 2})
    if not any(block.get("type") == "chart" for block in blocks):
        blocks.append({"type": "chart", "slotId": slot_id, "span": 2})
    if not blocks:
        blocks = fallback["blocks"]
    return {
        "title": title,
        "subtitle": subtitle,
        "chartTitle": chart_title or fallback.get("chartTitle"),
        "metricLabels": {
            **fallback.get("metricLabels", {}),
            **{str(key): str(value) for key, value in metric_labels.items() if key in allowed_metrics and value},
        },
        "blocks": blocks[:8],
    }


def _activity_decision_trace(
    *,
    question: str,
    source_kind: str,
    source: str,
    source_paths: list[str] | None,
    chart_slots: list[dict[str, Any]],
    layout_spec: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    summary = summary or {}
    layout_spec = layout_spec or {}
    field_names = [str(field) for field in (fields or []) if field]
    if not field_names:
        field_names = [
            str(slot.get("field"))
            for slot in chart_slots
            if isinstance(slot, dict) and slot.get("field")
        ]
    chart_titles = [
        str(slot.get("title") or slot.get("id") or "chart")
        for slot in chart_slots
        if isinstance(slot, dict)
    ]
    layout_blocks = layout_spec.get("blocks") if isinstance(layout_spec.get("blocks"), list) else []
    chart_types = [
        str(slot.get("chartType"))
        for slot in chart_slots
        if isinstance(slot, dict) and slot.get("chartType")
    ]
    lineage = _dedupe_source_paths([str(path) for path in (source_paths or []) if path])
    return [
        _decision_trace_item(
            "Interpret request",
            "Agent decided this should be a field-matched activity dashboard because no stronger ranked or time-series graph aggregate matched the request.",
            [
                f"User request: {question}",
                f"Agent decided: {_result_type_label('activity', chart_titles=chart_titles)}",
            ],
        ),
        _decision_trace_item(
            "Select data",
            f"Selected {source or 'matched source'} from {source_kind} context using fields and aggregates matched to the prompt.",
            [
                f"Rows available: {summary.get('totalRecords') or summary.get('sampleRecords') or 'not provided'}",
                f"Fields matched: {', '.join(field_names[:10]) if field_names else 'not provided'}",
            ],
        ),
        _decision_trace_item(
            "Collect lineage",
            "Resolved original dashboard sources from graph or DuckDB lineage metadata.",
            lineage[:8],
        ),
        _decision_trace_item(
            "Choose layout",
            f"Used the {layout_spec.get('title') or 'activity'} layout because its available metrics and chart slots matched the prompt.",
            [
                f"Layout blocks: {len(layout_blocks)}",
                f"Charts selected: {', '.join(chart_titles) if chart_titles else 'none'}",
            ],
        ),
        _decision_trace_item(
            "Bind chart",
            "Bound selected fields to the available chart types and used KPI cards for aggregate counts.",
            [
                f"Chart types: {', '.join(chart_types) if chart_types else 'none'}",
                f"Chart count: {len(chart_slots)}",
            ],
        ),
    ]


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
    source_paths = _aggregate_source_paths(aggregate, question, [dimension, measure])
    dimension_label = _dimension_display_name(dimension, question)
    measure_label = _measure_display_name(measure)
    top_label = str(aggregate.get("top_label") or chart_data[0].get("label") or "")
    top_value = int(aggregate.get("top_value") or chart_data[0].get("value") or 0)
    total_measure = int(aggregate.get("total_measure_values") or 0)
    total_dimensions = int(aggregate.get("distinct_dimension_values") or 0)
    records = _ranked_sample_records_from_duckdb(aggregate, table, dimension, top_label) or [
        {"source": table, "index": index, dimension: item.get("label"), measure: item.get("value")}
        for index, item in enumerate(chart_data)
    ]
    source_samples = _source_sample_records_from_duckdb(aggregate, source_paths)
    slot_id = f"top-{_slot_safe_field(dimension)}-by-{_slot_safe_field(measure)}"
    layout_spec = _ranked_layout_spec(
        question=question,
        dimension_label=dimension_label,
        measure_label=measure_label,
        slot_id=slot_id,
        chart_data=chart_data,
    )
    chart_type = _ranked_chart_type_for_dimension(dimension, chart_data, question)
    chart_slots = [
        {
            "id": slot_id,
            "title": layout_spec.get("chartTitle") or f"{measure_label} by {dimension_label}",
            "chartType": chart_type,
            "field": dimension,
            "reason": f"Ranked aggregate read from {source_kind} graph context.",
            "data": chart_data,
        }
    ]
    decision_trace = _ranked_decision_trace(
        question=question,
        source_kind=source_kind,
        table=table,
        dimension=dimension,
        measure=measure,
        dimension_label=dimension_label,
        measure_label=measure_label,
        source_paths=source_paths,
        chart_data=chart_data,
        total_measure=total_measure,
        total_dimensions=total_dimensions,
        chart_type=chart_type,
    )
    return {
        "datasets": {
            table: {
                "source": table,
                "key": table,
                "source_paths": source_paths,
                "source_samples": source_samples,
                "object_type": f"{source_kind}_aggregate",
                "sampleRecords": len(chart_data),
                "totalRecords": total_measure,
                "isFullAggregate": True,
            }
        },
        "records": records,
        "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
        "chartSlots": chart_slots,
        "charts": {slot_id: chart_data},
        "decisionTrace": decision_trace,
        "summary": {
            "source": table,
            "sourcePaths": source_paths,
            "sourceSamples": source_samples,
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
            "metricLabels": layout_spec.get("metricLabels") or {},
        },
        "layoutSpec": {key: value for key, value in layout_spec.items() if key != "metricLabels"},
    }


def _duckdb_ranked_dimension_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "over time" in lowered or "overtime" in lowered
    if wants_time:
        return {}
    wants_grouped = " by " in lowered and any(term in lowered for term in ("user", "users", "student", "learner", "course"))
    wants_rank = any(term in lowered for term in ("most", "top", "highest", "largest", "max", "rank")) or wants_grouped
    if not wants_rank:
        return {}
    db_path = _duckdb_database_path(store)
    if db_path is None or not db_path.exists():
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
        extra_where, filter_params, filter_labels = _ranked_filter_sql(con, table, question)
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
              {extra_where}
            group by 1
            order by value desc, label
            limit 12
            """
            ,
            filter_params,
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
                  {extra_where}
                """
                ,
                filter_params,
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
                  {extra_where}
                """
                ,
                filter_params,
            ).fetchone()[0]
            or 0
        )
        chart_data = [{"label": str(label), "value": int(value)} for label, value in rows]
        top_label = chart_data[0]["label"]
        top_value = chart_data[0]["value"]
        dimension_label = _dimension_display_name(dimension, question)
        measure_label = _measure_display_name(measure)
        source_paths = _source_paths_for_table(con, table, question=question, fields=[dimension, measure])
        source_samples = _source_sample_records(con, source_paths, limit_per_source=50)
        sample_records = _ranked_sample_records(
            con,
            table,
            dimension,
            top_label,
            limit=12,
            extra_where=extra_where,
            params=filter_params,
        )
        slot_id = f"top-{_slot_safe_field(dimension)}-by-{_slot_safe_field(measure)}"
        layout_spec = _ranked_layout_spec(
            question=question,
            dimension_label=dimension_label,
            measure_label=measure_label,
            slot_id=slot_id,
            chart_data=chart_data,
            filter_labels=filter_labels,
        )
        chart_type = _ranked_chart_type_for_dimension(dimension, chart_data, question)
        chart_slots = [
            {
                "id": slot_id,
                "title": layout_spec.get("chartTitle") or f"{measure_label} by {dimension_label}",
                "chartType": chart_type,
                "field": dimension,
                "reason": "A ranked chart answers the requested top dimension by distinct measure after applying prompt filters.",
                "data": chart_data,
            }
        ]
        split_slot = _duckdb_split_dimension_slot(
            con,
            table,
            question,
            dimension=dimension,
            measure=measure,
            extra_where=extra_where,
            params=filter_params,
        )
        if split_slot:
            chart_slots.insert(0, split_slot)
            layout_spec["chartTitle"] = split_slot.get("title") or layout_spec.get("chartTitle")
            layout_spec["blocks"] = _replace_primary_chart_block(
                layout_spec.get("blocks"),
                old_slot_id=slot_id,
                new_slot_id=str(split_slot["id"]),
            )
        if not split_slot:
            chart_slots.extend(
                _duckdb_companion_dimension_slots(
                    con,
                    table,
                    question,
                    measure=measure,
                    excluded_dimensions={dimension},
                    extra_where=extra_where,
                    params=filter_params,
                    primary_dimension=dimension if _should_scope_companion_to_primary(question) else None,
                    primary_value=top_label if _should_scope_companion_to_primary(question) else None,
                    max_slots=3,
                )
            )
        if len(chart_slots) > 1 and not split_slot:
            layout_spec["blocks"] = _extend_layout_blocks_with_slots(layout_spec.get("blocks"), chart_slots[1:], max_blocks=8)
        decision_trace = _ranked_decision_trace(
            question=question,
            source_kind="duckdb",
            table=table,
            dimension=dimension,
            measure=measure,
            dimension_label=dimension_label,
            measure_label=measure_label,
            source_paths=source_paths,
            chart_data=chart_data,
            total_measure=total_measure,
            total_dimensions=total_dimensions,
            chart_type=chart_type,
        )
        activity = {
            "datasets": {
                table: {
                    "source": table,
                    "key": table,
                    "source_paths": source_paths,
                    "source_samples": source_samples,
                    "object_type": "duckdb",
                    "sampleRecords": len(chart_data),
                    "totalRecords": total_measure,
                    "isFullAggregate": True,
                }
            },
            "records": sample_records
            or [
                {"source": table, "index": index, dimension: label, measure: value}
                for index, (label, value) in enumerate(rows)
            ],
            "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
            "chartSlots": chart_slots,
            "charts": {slot_id: chart_data},
            "decisionTrace": decision_trace,
            "summary": {
                "source": table,
                "sourcePaths": source_paths,
                "sourceSamples": source_samples,
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
                "filters": filter_labels,
                "metricLabels": layout_spec.get("metricLabels") or {},
            },
            "layoutSpec": {key: value for key, value in layout_spec.items() if key != "metricLabels"},
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
    primary_dimension_terms = _primary_ranked_dimension_terms(question)
    measure_terms = _measure_terms(question)
    requires_filter = _needs_duckdb_filter(question)
    candidates: list[tuple[float, str, str, str]] = []
    for table in tables:
        if any(skip in table for skip in ("cache", "map")):
            continue
        if requires_filter and not _ranked_filter_sql(con, table, question)[0]:
            continue
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
        measure_columns = [column for column in columns if _is_measure_column(column, measure_terms)]
        dimension_columns = [column for column in columns if _is_dimension_column(column)]
        for dimension in dimension_columns:
            dimension_score = _ranked_dimension_score(dimension, intent_terms, question)
            if primary_dimension_terms:
                primary_overlap = _field_terms(dimension) & primary_dimension_terms
                if not primary_overlap:
                    continue
                dimension_score += float(len(primary_overlap) * 30)
            if dimension_score <= 0:
                continue
            for measure in measure_columns:
                measure_score = _ranked_measure_score(measure, measure_terms)
                if measure_score <= 0:
                    continue
                table_score = 1.0 if "fact" in table or "joined" in table else 0.0
                table_score += _secondary_dimension_table_score(
                    con,
                    table,
                    columns,
                    question,
                    measure=measure,
                    excluded={dimension, measure},
                )
                if "name" in intent_terms and "name" in _field_terms(dimension):
                    dimension_score += 2.0
                candidates.append((dimension_score + measure_score + table_score, table, dimension, measure))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    for _, table, dimension, measure in candidates:
        if _ranked_dimension_has_values(con, table, dimension, measure):
            return {"table": table, "dimension": dimension, "measure": measure}
    return {}


def _secondary_dimension_table_score(
    con: Any,
    table: str,
    columns: list[str],
    question: str,
    *,
    measure: str,
    excluded: set[str],
) -> float:
    lowered = question.lower()
    if not any(term in lowered for term in (" and ", "also", "compare", "composition", "distribution", "breakdown", "split", "inside", "within")):
        return 0.0
    focus_terms = _query_dimension_terms(question)
    if not focus_terms:
        return 0.0
    score = 0.0
    for column in columns:
        if column in excluded or not _is_dimension_column(column):
            continue
        field_terms = _field_terms(column)
        overlap = field_terms & focus_terms
        if overlap:
            if _ranked_dimension_has_values(con, table, column, measure):
                score += min(float(len(overlap) * 3), 9.0)
            else:
                score -= 3.0
    return min(score, 12.0)


def _extend_layout_blocks_with_slots(
    blocks: Any,
    slots: list[dict[str, Any]],
    *,
    max_blocks: int = 8,
) -> list[dict[str, Any]]:
    result = [dict(block) for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []
    existing = {str(block.get("slotId")) for block in result if block.get("type") == "chart" and block.get("slotId")}
    for slot in slots:
        slot_id = str(slot.get("id") or "")
        if not slot_id or slot_id in existing or not slot.get("data"):
            continue
        result.append({"type": "chart", "slotId": slot_id, "span": 2})
        existing.add(slot_id)
        if len(result) >= max_blocks:
            break
    return result[:max_blocks]


def _replace_primary_chart_block(blocks: Any, *, old_slot_id: str, new_slot_id: str) -> list[dict[str, Any]]:
    result = [dict(block) for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []
    replaced = False
    for block in result:
        if block.get("type") == "chart" and block.get("slotId") == old_slot_id:
            block["slotId"] = new_slot_id
            block["span"] = 2
            replaced = True
            break
    if not replaced:
        result.append({"type": "chart", "slotId": new_slot_id, "span": 2})
    return result[:8]


def _split_dimension_terms(question: str) -> set[str]:
    lowered = f" {question.lower()} "
    for pattern in (
        r"\b(?:split by|breakdown by|grouped by|group by)\s+([a-z0-9 _-]+)",
        r"\b(?:split|breakdown|group)\s+.+?\s+by\s+([a-z0-9 _-]+)",
    ):
        match = re.search(pattern, lowered)
        if not match:
            continue
        phrase = re.split(
            r"\b(?:after|and|compare|for|from|having|that|to|where|when|which|who|with)\b",
            match.group(1),
            maxsplit=1,
        )[0]
        phrase = re.sub(r"\b(?:user|users|student|students|learner|learners|number|count|total)\b", " ", phrase)
        phrase = re.sub(r"[^a-z0-9 _-]+", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        terms = set(_intent_terms(phrase)) - QUERY_DIMENSION_STOP_TERMS
        if terms:
            return terms
    return set()


def _split_dimension_for_prompt(columns: list[str], question: str, *, excluded: set[str]) -> str:
    split_terms = _split_dimension_terms(question)
    if not split_terms:
        return ""
    candidates: list[tuple[float, str]] = []
    for column in columns:
        if column in excluded or not _is_dimension_column(column):
            continue
        field_terms = _field_terms(column)
        overlap = field_terms & split_terms
        if not overlap:
            continue
        candidates.append((float(len(overlap) * 10) + _ranked_dimension_score(column, split_terms, question), column))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1] if candidates else ""


def _duckdb_split_dimension_slot(
    con: Any,
    table: str,
    question: str,
    *,
    dimension: str,
    measure: str,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> dict[str, Any]:
    try:
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
    except Exception:
        return {}
    split_dimension = _split_dimension_for_prompt(columns, question, excluded={dimension, measure})
    if not split_dimension:
        return {}
    dimension_expr = _duckdb_quote_identifier(dimension)
    split_expr = _duckdb_quote_identifier(split_dimension)
    measure_expr = _duckdb_quote_identifier(measure)
    query_params = list(params or [])
    try:
        rows = con.execute(
            f"""
            with top_dimensions as (
                select cast({dimension_expr} as varchar) as label, count(distinct {measure_expr}) as total_value
                from {_duckdb_quote_identifier(table)}
                where {dimension_expr} is not null
                  and cast({dimension_expr} as varchar) <> ''
                  and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
                  and {split_expr} is not null
                  and cast({split_expr} as varchar) <> ''
                  and {measure_expr} is not null
                  {extra_where}
                group by 1
                order by total_value desc, label
                limit 8
            )
            select
                cast(t.{dimension_expr} as varchar) as label,
                cast(t.{split_expr} as varchar) as series,
                count(distinct t.{measure_expr}) as value
            from {_duckdb_quote_identifier(table)} t
            join top_dimensions d on cast(t.{dimension_expr} as varchar) = d.label
            where t.{dimension_expr} is not null
              and cast(t.{dimension_expr} as varchar) <> ''
              and t.{split_expr} is not null
              and cast(t.{split_expr} as varchar) <> ''
              and t.{measure_expr} is not null
              {extra_where}
            group by 1, 2
            order by max(d.total_value) desc, label, value desc, series
            """,
            [*query_params, *query_params],
        ).fetchall()
    except Exception:
        return {}
    data = [
        {"label": str(label), "series": str(series), "value": int(value or 0)}
        for label, series, value in rows
        if label not in (None, "") and series not in (None, "")
    ]
    if not data:
        return {}
    dimension_label = _dimension_display_name(dimension, question)
    split_label = _dimension_display_name(split_dimension, question)
    measure_label = _measure_display_name(measure)
    return {
        "id": f"split-{_slot_safe_field(dimension)}-by-{_slot_safe_field(split_dimension)}",
        "title": f"{measure_label} by {dimension_label} split by {split_label}",
        "chartType": "stacked_bar",
        "field": dimension,
        "splitField": split_dimension,
        "reason": f"Split chart keeps {dimension_label} as the main dimension and separates values by {split_label}.",
        "data": data,
    }


def _duckdb_dimension_rows(
    con: Any,
    table: str,
    dimension: str,
    measure: str,
    *,
    extra_where: str = "",
    params: list[Any] | None = None,
    primary_dimension: str | None = None,
    primary_value: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    dimension_expr = _duckdb_quote_identifier(dimension)
    measure_expr = _duckdb_quote_identifier(measure)
    where_sql = f"""
        where {dimension_expr} is not null
          and cast({dimension_expr} as varchar) <> ''
          and lower(cast({dimension_expr} as varchar)) not in ('unknown', 'none', 'null')
          and {measure_expr} is not null
          {extra_where}
    """
    query_params = list(params or [])
    if primary_dimension and primary_value not in (None, ""):
        where_sql += f"\n          and cast({_duckdb_quote_identifier(primary_dimension)} as varchar) = ?"
        query_params.append(str(primary_value))
    rows = con.execute(
        f"""
        select cast({dimension_expr} as varchar) as label, count(distinct {measure_expr}) as value
        from {_duckdb_quote_identifier(table)}
        {where_sql}
        group by 1
        order by value desc, label
        limit ?
        """,
        [*query_params, int(limit)],
    ).fetchall()
    return [{"label": str(label), "value": int(value or 0)} for label, value in rows]


def _should_scope_companion_to_primary(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("inside", "within", "in that", "for that", "of that", "there"))


def _duckdb_companion_dimension_slots(
    con: Any,
    table: str,
    question: str,
    *,
    measure: str,
    excluded_dimensions: set[str],
    extra_where: str = "",
    params: list[Any] | None = None,
    primary_dimension: str | None = None,
    primary_value: str | None = None,
    max_slots: int = 3,
) -> list[dict[str, Any]]:
    lowered = question.lower()
    wants_companion = any(
        term in lowered
        for term in (
            " and ",
            " also ",
            "compare",
            "composition",
            "distribution",
            "breakdown",
            "split",
            "inside",
            "within",
            "by ",
        )
    )
    if not wants_companion:
        return []
    try:
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
    except Exception:
        return []
    intent_terms = set(_intent_terms(question))
    raw_candidates: list[tuple[float, bool, str]] = []
    for column in columns:
        if column in excluded_dimensions or column == measure or not _is_dimension_column(column):
            continue
        score = _ranked_dimension_score(column, intent_terms, question)
        exact_match = _dimension_matches_requested_phrase(column, question)
        if exact_match:
            score += 100.0
        if _field_terms(column) & {"status", "type", "category", "province", "school", "institute", "course", "level"}:
            score += 1.0
        if score <= 0:
            continue
        raw_candidates.append((score, exact_match, column))
    exact_candidates = [item for item in raw_candidates if item[1]]
    candidates = exact_candidates or raw_candidates
    candidates.sort(key=lambda item: (-item[0], item[2]))
    slots: list[dict[str, Any]] = []
    used: set[str] = set()
    measure_label = _measure_display_name(measure)
    for _score, exact_match, dimension in candidates:
        if dimension in used:
            continue
        try:
            rows = _duckdb_dimension_rows(
                con,
                table,
                dimension,
                measure,
                extra_where=extra_where,
                params=params,
                primary_dimension=primary_dimension,
                primary_value=primary_value,
            )
        except Exception:
            continue
        if len(rows) < 2 and not exact_match:
            continue
        dimension_label = _dimension_display_name(dimension, question)
        slot_id = f"companion-{_slot_safe_field(dimension)}-by-{_slot_safe_field(measure)}"
        if primary_dimension and primary_value not in (None, ""):
            primary_label = _dimension_display_name(primary_dimension, question)
            title = f"{dimension_label} within top {primary_label}"
            reason = f"Companion chart answers the secondary dimension requested in the prompt within the leading {primary_label}."
        else:
            title = f"{measure_label} by {dimension_label}"
            reason = "Companion chart answers an additional dimension requested in the same prompt."
        slots.append(
            {
                "id": slot_id,
                "title": title,
                "chartType": _chart_type_for_count_field(dimension, rows, question) if len(rows) > 1 else "stat",
                "field": dimension,
                "reason": reason,
                "data": rows,
            }
        )
        used.add(dimension)
        if len(slots) >= max_slots:
            break
    return slots


def _dimension_matches_requested_phrase(column: str, question: str) -> bool:
    field_terms = _field_terms(column)
    if not field_terms:
        return False
    primary_terms = _primary_ranked_dimension_terms(question) | _primary_group_dimension_terms(question)
    for phrase in _query_dimension_phrases(question):
        phrase_terms = set(_intent_terms(phrase)) - QUERY_DIMENSION_STOP_TERMS
        if primary_terms and phrase_terms and phrase_terms <= primary_terms:
            continue
        if phrase_terms and phrase_terms <= field_terms:
            return True
    return False


def _time_column_score(column: str, question: str) -> float:
    lowered = column.lower()
    terms = _field_terms(column)
    score = 0.0
    if "date" in terms or "time" in terms or "timestamp" in terms:
        score += 6.0
    if "activity" in lowered or "last" in terms:
        score += 3.0
    if "enroll" in lowered:
        score += 2.0
    if "cert" in lowered or "certificate" in lowered:
        score += 2.0
    query = question.lower()
    if any(term in query for term in ("finish", "finished", "complete", "completed", "pass", "passed", "certificate")):
        if "cert" in lowered or "complete" in lowered or "pass" in lowered:
            score += 5.0
        if "last" in terms or "activity" in lowered:
            score += 2.0
    if any(term in query for term in ("activity", "active", "read", "reading", "learn", "learning")):
        if "activity" in lowered or "last" in terms:
            score += 5.0
    return score


def _is_time_column(column: str, column_type: str, question: str) -> bool:
    lowered_type = column_type.lower()
    if "date" in lowered_type or "time" in lowered_type:
        return True
    return _time_column_score(column, question) > 0


def _grouped_time_series_plan(con: Any, question: str) -> dict[str, str]:
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
    primary_dimension_terms = _primary_group_dimension_terms(question)
    dimension_terms = primary_dimension_terms or _query_dimension_terms(question)
    measure_terms = _measure_terms(question)
    candidates: list[tuple[float, str, str, str, str]] = []
    for table in tables:
        if any(skip in table for skip in ("cache", "map")):
            continue
        pragma_rows = con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()
        columns = [(str(row[1]), str(row[2])) for row in pragma_rows]
        time_columns = [column for column, column_type in columns if _is_time_column(column, column_type, question)]
        dimension_columns = [
            column
            for column, _column_type in columns
            if _is_dimension_column(column) or _dimension_field_score(column, question, set(_intent_terms(question))) > 0
        ]
        measure_columns = [column for column, _column_type in columns if _is_measure_column(column, measure_terms)]
        for dimension in dimension_columns:
            dimension_score = _dimension_field_score(dimension, question, set(_intent_terms(question)))
            if primary_dimension_terms:
                primary_overlap = _field_terms(dimension) & primary_dimension_terms
                if not primary_overlap:
                    continue
                dimension_score += float(len(primary_overlap) * 20)
            if dimension_terms and not (_field_terms(dimension) & dimension_terms) and dimension_score <= 0:
                continue
            for time_column in time_columns:
                time_score = _time_column_score(time_column, question)
                if time_score <= 0:
                    continue
                for measure in measure_columns:
                    measure_score = _ranked_measure_score(measure, measure_terms)
                    if measure_score <= 0:
                        continue
                    if not _grouped_time_series_has_values(con, table, time_column, dimension, measure):
                        continue
                    table_score = 4.0 if "fact" in table or "joined" in table else 0.0
                    candidates.append((dimension_score + time_score + measure_score + table_score, table, time_column, dimension, measure))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    return {"table": candidates[0][1], "time": candidates[0][2], "dimension": candidates[0][3], "measure": candidates[0][4]} if candidates else {}


def _grouped_time_series_has_values(con: Any, table: str, time_column: str, dimension: str, measure: str) -> bool:
    try:
        value = con.execute(
            f"""
            select count(*)
            from {_duckdb_quote_identifier(table)}
            where {_duckdb_quote_identifier(time_column)} is not null
              and {_duckdb_quote_identifier(dimension)} is not null
              and cast({_duckdb_quote_identifier(dimension)} as varchar) <> ''
              and {_duckdb_quote_identifier(measure)} is not null
            limit 1
            """
        ).fetchone()[0]
    except Exception:
        return False
    return int(value or 0) > 0


def _requested_dimension_values(question: str) -> set[str]:
    lowered = question.lower()
    values: set[str] = set()
    known_values = {
        "passed": ("passed", "pass", "finished", "complete", "completed"),
        "in_progress": ("in progress", "in-progress", "in_progress"),
        "inactive": ("inactive",),
        "active": ("active",),
    }
    for canonical, terms in known_values.items():
        if any(term in lowered for term in terms):
            values.add(canonical)
    return values


def _dimension_value_matches_request(value: Any, requested_values: set[str]) -> bool:
    if not requested_values:
        return True
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in requested_values:
        return True
    if normalized == "passed" and {"passed"} & requested_values:
        return True
    return False


def _duckdb_grouped_time_series_context(store: Any, question: str) -> dict[str, Any]:
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "over time" in lowered or "overtime" in lowered
    wants_group = any(term in lowered for term in (" by ", "group", "split", "breakdown", "compare"))
    if not (wants_time and wants_group):
        return {}
    db_path = _duckdb_database_path(store)
    if db_path is None or not db_path.exists():
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
        plan = _grouped_time_series_plan(con, question)
        if not plan:
            return {}
        table = plan["table"]
        time_column = plan["time"]
        dimension = plan["dimension"]
        measure = plan["measure"]
        time_expr = _duckdb_quote_identifier(time_column)
        dimension_expr = _duckdb_quote_identifier(dimension)
        measure_expr = _duckdb_quote_identifier(measure)
        requested_group_values = _requested_dimension_values(question)
        if requested_group_values and (_field_terms(dimension) & {"status", "result", "pass", "passed", "complete", "completed", "learning"}):
            extra_where, filter_params, filter_labels = "", [], []
        else:
            extra_where, filter_params, filter_labels = _ranked_filter_sql(con, table, question)
        group_limit = 50 if requested_group_values else 4
        top_groups = con.execute(
            f"""
            select cast({dimension_expr} as varchar) as label, count(distinct {measure_expr}) as value
            from {_duckdb_quote_identifier(table)}
            where {dimension_expr} is not null
              and cast({dimension_expr} as varchar) <> ''
              and {time_expr} is not null
              {extra_where}
            group by 1
            order by value desc, label
            limit {int(group_limit)}
            """,
            filter_params,
        ).fetchall()
        if requested_group_values:
            top_groups = [
                (label, value)
                for label, value in top_groups
                if _dimension_value_matches_request(label, requested_group_values)
            ]
        if not top_groups:
            return {}
        dimension_label = _dimension_display_name(dimension, question)
        measure_label = _measure_display_name(measure)
        chart_slots: list[dict[str, Any]] = []
        charts: dict[str, list[dict[str, Any]]] = {}
        for index, (group_label, _group_value) in enumerate(top_groups):
            rows = con.execute(
                f"""
                with grouped as (
                    select
                        date_trunc('month', {time_expr}) as bucket,
                        count(distinct {measure_expr}) as value
                    from {_duckdb_quote_identifier(table)}
                    where {dimension_expr} is not null
                      and cast({dimension_expr} as varchar) = ?
                      and {time_expr} is not null
                      {extra_where}
                    group by 1
                    order by bucket
                ),
                numbered as (
                    select
                        bucket,
                        value,
                        row_number() over (order by bucket) as rn,
                        count(*) over () as total_rows
                    from grouped
                ),
                sampled as (
                    select
                        *,
                        case
                            when total_rows <= 18 then rn
                            when rn = 1 then 1
                            when rn = total_rows then 18
                            else 2 + cast(floor(((rn - 2) * 16.0) / greatest(total_rows - 2, 1)) as integer)
                        end as sample_bucket
                    from numbered
                ),
                bucketed as (
                    select *, row_number() over (partition by sample_bucket order by rn) as bucket_rank
                    from sampled
                )
                select strftime(bucket, '%Y-%m') as label, value
                from bucketed
                where bucket_rank = 1
                order by label
                """,
                [str(group_label), *filter_params],
            ).fetchall()
            data = [{"label": str(label), "value": int(value or 0)} for label, value in rows]
            if not data:
                continue
            slot_id = f"groupedTime-{_slot_safe_field(dimension)}-{index}"
            title = f"{measure_label} over time - {group_label}"
            slot = {
                "id": slot_id,
                "title": title,
                "chartType": "line",
                "field": f"{measure}@{time_column}|{dimension}",
                "reason": f"A grouped time-series line chart tracks distinct {_humanize_field(measure)} over {_humanize_field(time_column)} by {_humanize_field(dimension)}.",
                "data": data,
            }
            chart_slots.append(slot)
            charts[slot_id] = data
        if not chart_slots:
            return {}
        companion_slots: list[dict[str, Any]] = []
        chart_slots.extend(companion_slots)
        for slot in companion_slots:
            charts[str(slot["id"])] = slot.get("data") or []
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_duckdb_quote_identifier(table)})").fetchall()]
        source_paths = _infer_source_paths_for_table(con, table, columns, question=question, fields=[dimension, measure, time_column])
        source_samples = _source_sample_records(con, source_paths, limit_per_source=50)
        total_measure = int(
            con.execute(
                f"""
                select count(distinct {measure_expr})
                from {_duckdb_quote_identifier(table)}
                where {time_expr} is not null
                  {extra_where}
                """,
                filter_params,
            ).fetchone()[0]
            or 0
        )
        total_records = int(
            con.execute(
                f"""
                select count(*)
                from {_duckdb_quote_identifier(table)}
                where {time_expr} is not null
                  {extra_where}
                """,
                filter_params,
            ).fetchone()[0]
            or 0
        )
        total_dimensions = int(
            con.execute(
                f"""
                select count(distinct {dimension_expr})
                from {_duckdb_quote_identifier(table)}
                where {dimension_expr} is not null
                  and {time_expr} is not null
                  {extra_where}
                """,
                filter_params,
            ).fetchone()[0]
            or 0
        )
        decision_trace = _time_series_decision_trace(
            question=question,
            source_kind="duckdb",
            table=table,
            source_paths=source_paths,
            chart_slots=chart_slots,
            total_records=total_records,
            total_users=total_measure,
        )
        activity = {
            "datasets": {
                table: {
                    "source": table,
                    "key": table,
                    "source_paths": source_paths,
                    "source_samples": source_samples,
                    "object_type": "duckdb_grouped_time_series",
                    "sampleRecords": sum(len(slot.get("data") or []) for slot in chart_slots),
                    "totalRecords": total_records,
                    "isFullAggregate": True,
                }
            },
            "records": [
                {"source": table, "series": str(label), dimension: str(label), measure: int(value or 0)}
                for label, value in top_groups
            ],
            "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots],
            "chartSlots": chart_slots,
            "charts": charts,
            "decisionTrace": decision_trace,
            "summary": {
                "source": table,
                "sourcePaths": source_paths,
                "sourceSamples": source_samples,
                "sampleRecords": sum(len(slot.get("data") or []) for slot in chart_slots),
                "totalRecords": total_records,
                "isFullAggregate": True,
                "distinctUsers": total_measure,
                "topDimensionField": dimension,
                "topDimensionName": dimension_label,
                "measureField": measure,
                "measureName": measure_label,
                "totalDistinctMeasure": total_measure,
                "distinctDimensionValues": total_dimensions,
                "filters": filter_labels,
                "metricLabels": {
                    "totalDistinctMeasure": f"Total {measure_label}",
                    "distinctDimensionValues": f"{dimension_label} groups",
                },
            },
            "layoutSpec": {
                "title": f"{measure_label} over time by {dimension_label}",
                "subtitle": f"Shows distinct {measure_label} over time grouped by {dimension_label}.",
                "blocks": [
                    {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
                    {"type": "metric", "id": "distinctDimensionValues", "span": 1},
                    *[
                        {"type": "chart", "slotId": str(slot["id"]), "span": 2}
                        for slot in chart_slots[:4]
                    ],
                ],
            },
        }
        return activity
    except Exception:
        return {}
    finally:
        con.close()


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
        for token in ("name", "school", "institute", "department", "province", "course", "category", "type", "status", "education", "level")
    )


def _ranked_dimension_score(column: str, intent_terms: set[str], question: str = "") -> float:
    score = _dimension_field_score(column, question, intent_terms) if question else _field_score(column, intent_terms)
    field_terms = _field_terms(column)
    if "name" in field_terms:
        score += 1.5
    if {"school", "institute"} & intent_terms and {"school", "institute"} & field_terms:
        score += 4.0
    if {"school", "institute"} & intent_terms:
        if "name" in field_terms or "school" in field_terms:
            score += 4.0
        if "id" in field_terms and "id" not in _query_dimension_terms(question):
            score -= 4.0
    lowered = column.lower()
    if "course" in intent_terms:
        if lowered in {"course_id", "subject_name"} and not (_query_dimension_terms(question) - {"course"}):
            score += 12.0
        course_focus_terms = _query_dimension_terms(question)
        course_metadata_terms = {"teacher", "faculty", "org", "organization", "type", "category"}
        if any(term in lowered for term in course_metadata_terms) and not (course_focus_terms & course_metadata_terms):
            score -= 12.0
    metric_like_terms = {"activity", "avg", "average", "count", "date", "grade", "max", "min", "rate", "score", "total"}
    focus_terms = _query_dimension_terms(question)
    if field_terms & metric_like_terms and not (focus_terms & metric_like_terms):
        score -= 20.0
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
    completion_terms = intent_terms & {"complete", "completed", "completion", "finish", "finished", "result", "status"}
    learning_terms = intent_terms & {"content", "course", "learn", "learning", "read", "reading"}
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
        source_label = _humanize_source_path(source_path)
        haystack = f"{source_path} {source_label} {source_table}".lower()
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
        source_fields = _duckdb_json_field_names(con, str(source_path))
        field_match_score = sum(_dimension_field_score(field, question, intent_terms) for field in source_fields)
        if field_match_score:
            score += min(field_match_score, 10.0)
        focus_terms = _query_dimension_terms(question)
        if focus_terms:
            best_focus_score = max((_dimension_field_score(field, question, intent_terms) for field in source_fields), default=0.0)
            if best_focus_score:
                score += min(best_focus_score, 30.0)
        if completion_terms:
            has_completion_field = any(_field_score(field, completion_terms) > 0 for field in source_fields)
            score += 15.0 if has_completion_field else -8.0
        if learning_terms:
            has_learning_field = any(_field_score(field, learning_terms) > 0 for field in source_fields)
            if has_learning_field:
                score += 8.0
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
        "category": _choose_field(fields, ("eventCategory", "event_category", "category", "type", "status", "state", "result")),
        "course": _choose_field(fields, ("courseID", "course_id", "course_key", "course")),
        "app": _choose_field(fields, ("appID", "app_id", "application", "app")),
    }
    _duckdb_field_cache[source_path] = inferred
    return inferred


def _duckdb_json_field_names(con: Any, source_path: str) -> set[str]:
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
        return set()
    fields: set[str] = set()
    for (payload_json,) in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            fields.update(str(key) for key in payload)
    return fields


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


def _duckdb_source_record_count(
    con: Any,
    source_path: str,
    *,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> int:
    try:
        value = con.execute(
            f"""
            select count(*)
            from unified_records
            where source_path = ?
              and payload_json is not null
              {extra_where}
            """,
            [source_path, *(params or [])],
        ).fetchone()[0]
    except Exception:
        return 0
    return int(value or 0)


def _duckdb_top_counts(
    con: Any,
    source_path: str,
    field: str,
    limit: int = 8,
    *,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    value_sql = _json_extract_expr(field)
    rows = con.execute(
        f"""
        select nullif({value_sql}, '') as label, count(*) as value
        from unified_records
        where source_path = ?
          and payload_json is not null
          {extra_where}
        group by label
        having label is not null
        order by value desc, label
        limit {int(limit)}
        """,
        [source_path, *(params or [])],
    ).fetchall()
    return [{"label": str(label), "value": int(value)} for label, value in rows]


def _duckdb_distinct_count(
    con: Any,
    source_path: str,
    field: str,
    *,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> int:
    value_sql = _json_extract_expr(field)
    try:
        value = con.execute(
            f"""
            select count(distinct nullif({value_sql}, ''))
            from unified_records
            where source_path = ?
              and payload_json is not null
              {extra_where}
            """,
            [source_path, *(params or [])],
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


def _duckdb_sample_records(
    con: Any,
    source_path: str,
    limit: int = 12,
    *,
    extra_where: str = "",
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
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
        f"""
        select payload_json
        from unified_records
        where source_path = ?
          and payload_json is not null
          {extra_where}
        limit ?
        """,
        [source_path, *(params or []), int(limit)],
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
    chart_slots = _retitle_slots_for_prompt(question, chart_slots)
    chart_plan = [{key: value for key, value in slot.items() if key != "data"} for slot in chart_slots]
    source_paths = _dedupe_source_paths(
        [
            str(meta.get("s3_uri") or meta.get("key") or source)
            for source, meta in dataset_meta.items()
            if source or meta.get("s3_uri") or meta.get("key")
        ]
    )
    summary = {
        "sampleRecords": len(records),
        "totalRecords": aggregate_meta.get("full_record_count"),
        "isFullAggregate": bool(aggregate_meta.get("full_record_count")),
        "distinctEvents": len(full_counts.get("event", []))
        or len({record.get("event") for record in records if record.get("event")}),
        "distinctCourses": len(full_counts.get("courseID", []))
        or len({record.get("courseID") for record in records if record.get("courseID")}),
        "distinctUsers": len(full_counts.get("userID", []))
        or len({record.get("userID") for record in records if record.get("userID")}),
    }
    activity = {
        "datasets": dataset_meta,
        "records": records,
        "chartPlan": chart_plan,
        "chartSlots": chart_slots,
        "charts": {slot["id"]: slot["data"] for slot in chart_slots},
        "summary": summary,
    }
    activity["layoutSpec"] = _activity_layout_spec(question, activity)
    activity["decisionTrace"] = _activity_decision_trace(
        question=question,
        source_kind="graph",
        source=", ".join(sources[:3]),
        source_paths=source_paths,
        chart_slots=chart_slots,
        layout_spec=activity["layoutSpec"],
        summary=summary,
    )
    return activity


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
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    plan: list[dict[str, Any]] = []

    if wants_time:
        distinct_fields = _rank_fields(full_distinct_time_buckets, intent_terms, question)
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

    if wants_time and full_time_buckets:
        plan.append(
            {
                "id": "activityTimeline",
                "title": "Activity over time",
                "chartType": "area",
                "field": "@time",
                "reason": "An area chart emphasizes activity volume by detected time bucket.",
            }
        )

    for field in _rank_fields(full_counts, intent_terms, question)[:6]:
        rows = full_counts.get(field) or []
        if not rows:
            continue
        field_label = _humanize_field(field)
        chart_type = _chart_type_for_count_field(field, rows, question)
        plan.append(
            {
                "id": f"fieldCounts-{_slot_safe_field(field)}",
                "title": f"{field_label} distribution",
                "chartType": chart_type,
                "field": field,
                "sourceField": field,
                "aggregate": "count",
                "reason": f"A {chart_type.replace('_', ' ')} chart fits this field's cardinality, label length, and requested comparison.",
            }
        )

    return plan[:6]


def _rank_fields(rows_by_field: dict[str, Any], intent_terms: set[str], question: str = "") -> list[str]:
    def sort_key(field: str) -> tuple[float, int, str]:
        rows = rows_by_field.get(field)
        row_count = len(rows) if isinstance(rows, list) else 0
        score = _dimension_field_score(field, question, intent_terms) if question else _field_score(field, intent_terms)
        return (-score, -row_count, field.lower())

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
    lowered = question.lower()
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    if wants_time and full_user_time_buckets:
        plan.append(
            {
                "id": "userTimeline",
                "title": "Users over time",
                "chartType": "line",
                "field": "userID@timestamp",
                "reason": "A line chart fits distinct user counts by timestamp bucket.",
            }
        )
    if wants_time and (full_time_buckets or _time_buckets(records)):
        plan.append(
            {
                "id": "activityTimeline",
                "title": "Activity over sampled time",
                "chartType": "area",
                "field": "@timestamp",
                "reason": "An area chart fits timestamped record volume over the sample window.",
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
    fallback = _fallback_activity_layout_spec(question, available_slots, available_metrics, summary)
    if summary.get("isFullAggregate") and available_slots:
        return fallback
    llm_spec = _llm_activity_layout_spec(question, chart_slots, summary, available_metrics)
    return _sanitize_activity_layout_spec(llm_spec, available_slots, available_metrics, fallback)


def _fallback_activity_layout_spec(
    question: str,
    available_slots: set[str],
    available_metrics: set[str],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    lowered = question.lower()
    wants_user = "user" in lowered or "learner" in lowered or "student" in lowered
    wants_time = "time" in lowered or "trend" in lowered or "overtime" in lowered or "over time" in lowered
    wants_count = "number" in lowered or "count" in lowered or "how many" in lowered or "total" in lowered
    wants_rank = any(term in lowered for term in ("most", "top", "highest", "largest", "max", "rank"))
    wants_breakdown = any(term in lowered for term in ("by", "breakdown", "distribution", "compare", "split", "group"))
    wants_course = "course" in lowered
    wants_event = "event" in lowered or "activity" in lowered or "click" in lowered

    def metric(metric_id: str, span: int = 1) -> dict[str, Any]:
        return {"type": "metric", "id": metric_id, "span": span}

    def chart(slot_id: str, span: int = 1) -> dict[str, Any]:
        return {"type": "chart", "slotId": slot_id, "span": span}

    def dynamic_slots(*excluded: str) -> list[str]:
        excluded_ids = {"userTimeline", "activityTimeline", *excluded}
        intent_terms = set(_intent_terms(question))
        return sorted(
            (slot_id for slot_id in available_slots if slot_id not in excluded_ids),
            key=lambda slot_id: (-_dimension_field_score(slot_id, question, intent_terms), slot_id),
        )

    blocks: list[dict[str, Any]] = []
    title = "Activity dashboard"
    subtitle = "Dashboard selected from the fields and aggregates matched to the prompt."
    filters = [str(item) for item in summary.get("filters") or [] if item]
    count_only = wants_count and not wants_time and not wants_rank and not wants_breakdown
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
    elif wants_breakdown:
        focused_slots = dynamic_slots("events", "courses", "categories", "users", "apps")
        primary_slot = focused_slots[0] if focused_slots else ""
        primary_label = _humanize_field(primary_slot.replace("fieldCounts-", "").replace("-", "_")) if primary_slot else "Distribution"
        title = primary_label
        subtitle = "Distribution matched to the requested field."
        for metric_id in ("totalRecords", "distinctUsers"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in focused_slots[:4]:
            blocks.append(chart(slot_id, 2 if slot_id == primary_slot else 1))
        for slot_id in ("events", "courses", "categories", "users", "apps"):
            if slot_id in available_slots and slot_id not in focused_slots:
                blocks.append(chart(slot_id, 1))
    elif wants_user:
        if count_only:
            if _query_filter_intent_terms(question):
                title = "Completed users" if {"finish", "finished", "complete", "completed", "pass", "passed"} & set(_intent_terms(question)) else "Filtered users"
                subtitle = f"Distinct users after applying: {', '.join(filters[:3])}." if filters else "Distinct users after applying the matched filter."
            else:
                title = "User count"
                subtitle = "Distinct users matched to the request."
            for metric_id in ("distinctUsers", "totalRecords"):
                if metric_id in available_metrics:
                    blocks.append(metric(metric_id))
            focused_slots = dynamic_slots("users")
            if focused_slots:
                blocks.append(chart(focused_slots[0], 2))
            blocks.append({"type": "source", "span": 2})
            return {"title": title, "subtitle": subtitle, "blocks": blocks[:8]}
        title = "User count" if wants_count else "User activity"
        for metric_id in ("distinctUsers", "totalRecords"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in dynamic_slots("users"):
            blocks.append(chart(slot_id, 2))
        if "users" in available_slots:
            blocks.append(chart("users", 2))
        if wants_time and "activityTimeline" in available_slots:
            blocks.append(chart("activityTimeline", 2))
    elif wants_course:
        title = "Course activity"
        for metric_id in ("distinctCourses", "totalRecords"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in dynamic_slots("courses"):
            blocks.append(chart(slot_id, 2))
        if "courses" in available_slots:
            blocks.append(chart("courses", 2))
        if wants_time and "activityTimeline" in available_slots:
            blocks.append(chart("activityTimeline", 2))
    elif wants_event:
        title = "Activity events"
        for metric_id in ("totalRecords", "distinctEvents"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        for slot_id in (*dynamic_slots("events", "categories"), "events", "categories"):
            if slot_id in available_slots:
                blocks.append(chart(slot_id, 2 if slot_id.startswith("fieldCounts-") else 1))
        if wants_time and "activityTimeline" in available_slots:
            blocks.append(chart("activityTimeline", 2))
    else:
        for metric_id in ("totalRecords", "distinctEvents", "distinctCourses", "distinctUsers"):
            if metric_id in available_metrics:
                blocks.append(metric(metric_id))
        general_slots = [
            *dynamic_slots("events", "courses", "categories", "users", "apps"),
            "events",
            "courses",
            "categories",
            "users",
            "apps",
        ]
        if wants_time:
            general_slots = ["userTimeline", "activityTimeline", *general_slots]
        for slot_id in general_slots:
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


def _supported_chart_types(slot: dict[str, Any]) -> set[str]:
    data = slot.get("data") if isinstance(slot.get("data"), list) else []
    has_series = any(isinstance(item, dict) and item.get("series") for item in data)
    current = str(slot.get("chartType") or "")
    if has_series:
        if current in {"line", "area", "multi_line"} or "time" in str(slot.get("field") or "").lower() or "date" in str(slot.get("field") or "").lower():
            return {"multi_line", "stacked_column", "stacked_bar"}
        return {"stacked_bar", "stacked_column", "multi_line"}
    count = len(data)
    if count <= 1:
        return {"stat", "horizontal_bar"}
    if current in {"line", "area"}:
        return {"line", "area", "column"}
    supported = {"horizontal_bar", "column"}
    if 2 <= count <= 8:
        supported.update({"donut", "pie", "radial_bar"})
    if 3 <= count <= 8:
        supported.add("radar")
    if 2 <= count <= 10:
        supported.add("funnel")
    if 3 <= count <= 20:
        supported.add("treemap")
    return supported


def _apply_chart_type_choices(activity: dict[str, Any], choices: Any) -> dict[str, Any]:
    if not isinstance(choices, dict):
        return activity
    slots = activity.get("chartSlots") if isinstance(activity.get("chartSlots"), list) else []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        requested = str(choices.get(str(slot.get("id"))) or "")
        if requested not in _supported_chart_types(slot):
            continue
        slot["chartType"] = requested
        slot["reason"] = f"{slot.get('reason') or ''} LLM selected {requested} from the compatible chart types.".strip()
    activity["chartPlan"] = [
        {key: value for key, value in slot.items() if key != "data"}
        for slot in slots
        if isinstance(slot, dict)
    ]
    return activity


def _prompt_chart_type_choice(question: str) -> str:
    lowered = question.lower()
    chart_phrases = {
        "multi_line": ("multi line", "multiple lines"),
        "stacked_column": ("stacked column", "stacked vertical"),
        "stacked_bar": ("stacked bar", "split bar"),
        "horizontal_bar": ("horizontal bar",),
        "radial_bar": ("radial bar", "circular bar"),
        "donut": ("donut", "ring chart"),
        "pie": ("pie chart", " as pie", " pie "),
        "treemap": ("treemap", "tree map"),
        "radar": ("radar", "spider chart"),
        "funnel": ("funnel",),
        "area": ("area chart",),
        "line": ("line chart",),
        "column": ("column chart", "vertical bar"),
    }
    padded = f" {lowered} "
    for chart_type, phrases in chart_phrases.items():
        if any(phrase in padded for phrase in phrases):
            return chart_type
    return ""


def _llm_design_complex_dashboard(question: str, activity: dict[str, Any]) -> dict[str, Any]:
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    slots = activity.get("chartSlots") if isinstance(activity.get("chartSlots"), list) else []
    if not slots or get_settings().llm_mode == "never":
        return activity
    explicit_chart_type = _prompt_chart_type_choice(question)
    if explicit_chart_type:
        _apply_chart_type_choices(
            activity,
            {
                str(slot.get("id")): explicit_chart_type
                for slot in slots
                if isinstance(slot, dict) and explicit_chart_type in _supported_chart_types(slot)
            },
        )
        return activity
    candidates = _llm_candidates()
    if not candidates:
        return activity
    from langchain_openai import ChatOpenAI

    slot_context = [
        {
            "id": slot.get("id"),
            "title": slot.get("title"),
            "field": slot.get("field"),
            "splitField": slot.get("splitField"),
            "rows": len(slot.get("data") or []),
            "hasSeries": any(isinstance(item, dict) and item.get("series") for item in slot.get("data") or []),
            "currentChartType": slot.get("chartType"),
            "compatibleChartTypes": sorted(_supported_chart_types(slot)),
        }
        for slot in slots
        if isinstance(slot, dict)
    ]
    messages = [
        (
            "system",
            (
                "You are the dashboard visualization planner. Return only JSON with chartTypes, title, and subtitle. "
                "chartTypes must map each supplied slot id to one compatibleChartTypes value. "
                "Choose by analytical job: line or area for change over time; multi_line for comparing time series; "
                "horizontal_bar for ranking and long labels; column for compact comparison; donut or pie for part-to-whole with few categories; "
                "treemap for hierarchical composition; radar only for comparing a small common profile; radial_bar for a compact circular comparison; "
                "funnel only for ordered stages; stacked_bar or stacked_column for composition split by a second dimension. "
                "Prefer the simplest truthful chart and do not use every available type merely for variety."
            ),
        ),
        (
            "human",
            json.dumps(
                {
                    "question": question,
                    "analyticalPlan": summary.get("analyticalPlan"),
                    "chartSlots": slot_context,
                },
                ensure_ascii=False,
                default=str,
            ),
        ),
    ]
    candidate = candidates[0]
    try:
        model = ChatOpenAI(
            api_key=candidate["api_key"],
            model=candidate["model"],
            base_url=candidate.get("base_url"),
            default_headers=candidate.get("headers") or None,
            temperature=0,
            timeout=10,
            max_retries=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        parsed = json.loads(str(model.invoke(messages).content))
        if not isinstance(parsed, dict):
            return activity
        _apply_chart_type_choices(activity, parsed.get("chartTypes"))
        layout = activity.get("layoutSpec") if isinstance(activity.get("layoutSpec"), dict) else {}
        if isinstance(parsed.get("title"), str) and parsed["title"].strip():
            layout["title"] = parsed["title"].strip()[:100]
        if isinstance(parsed.get("subtitle"), str) and parsed["subtitle"].strip():
            layout["subtitle"] = parsed["subtitle"].strip()[:180]
        activity["layoutSpec"] = layout
    except Exception:
        return activity
    return activity


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
        database_path = _duckdb_database_path(store)
        activity = (
            (build_complex_dashboard(database_path, question) if database_path else {})
            or _duckdb_grouped_time_series_context(store, question)
            or _graph_time_series_activity_context(store, question)
            or _duckdb_ranked_dimension_context(store, question)
            or _graph_ranked_dimension_context(store, question)
            or _aggregate_cache_activity(store, question)
            or _duckdb_activity_context(store, question)
            or _activity_dashboard_context(store, results, question)
        )
        if activity:
            activity = _llm_design_complex_dashboard(question, activity)
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
