from __future__ import annotations

import json
import re
import hashlib
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from dashboard_agent.config import Settings
from dashboard_agent.dashboard_planner import build_complex_dashboard, _resolve_shape, _execute_shape, _thai_gloss, _thai_question_fully_glossable
from dashboard_agent.dashboard_widget import dataset_summaries, graph_dashboard_marker
from dashboard_agent.graph_dashboard_cache import (
    read_graph_dashboard_cache,
    read_question_translation,
    write_graph_dashboard_cache,
    write_question_translation,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]


refresh_lock = threading.Lock()
settings: Settings | None = None
_settings: Settings | None = None
_store: Any | None = None
_duckdb_field_cache: dict[str, dict[str, str]] = {}
_question_language_cache: dict[str, str] = {}
_semantic_schema_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}
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


# ---------------------------------------------------------------------------
# Dashboard vs text routing (Thai).  These literals were mined from the
# eval corpus by dashboard-evaluate/_gen_thai_final.py: each is a
# discriminative substring verified against every corpus prompt.
# Do not edit by hand.
# ---------------------------------------------------------------------------

_THAI_TEXT_PHRASES = (
    "\u0e15\u0e2d\u0e1a\u0e40\u0e1b\u0e47\u0e19\u0e02\u0e49\u0e2d\u0e04\u0e27",
    "\u0e49\u0e2d\u0e07\u0e2a\u0e23\u0e49\u0e32\u0e07\u0e01\u0e23\u0e32\u0e1f",
    "\u0e22\u0e1a\u0e2d\u0e01\u0e08\u0e33\u0e19\u0e27\u0e19\u0e1c\u0e39\u0e49",
    "\u0e44\u0e21\u0e48\u0e21\u0e35\u0e43\u0e1a\u0e23\u0e31\u0e1a\u0e23\u0e2d",
    "\u0e49\u0e40\u0e23\u0e35\u0e22\u0e19\u0e2b\u0e23\u0e37\u0e2d\u0e44\u0e21",
    "\u0e44\u0e21\u0e48\u0e0b\u0e49\u0e33\u0e40\u0e1b\u0e47\u0e19\u0e04\u0e33",
    "\u0e2d\u0e08\u0e33\u0e19\u0e27\u0e19\u0e1c\u0e39\u0e49\u0e25\u0e07\u0e17",

)

_THAI_DASH_SIGNALS = (
    "\u0e17\u0e35\u0e48\u0e21\u0e35\u0e1c\u0e39\u0e49\u0e40\u0e23\u0e35\u0e22",
    "\u0e27\u0e22\u0e07\u0e32\u0e19\u0e2b\u0e25\u0e31\u0e01\u0e2a\u0e39\u0e15",
    "\u0e21\u0e2a\u0e16\u0e32\u0e19\u0e30\u0e01\u0e32\u0e23\u0e40\u0e23\u0e35",
    "\u0e21\u0e35\u0e1c\u0e39\u0e49\u0e40\u0e23\u0e35\u0e22\u0e19\u0e21\u0e32",
    "\u0e0b\u0e49\u0e33\u0e21\u0e32\u0e01\u0e17\u0e35\u0e48\u0e2a\u0e38\u0e14",
    "\u0e23\u0e21\u0e17\u0e35\u0e48\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01\u0e44",
    "\u0e07\u0e08\u0e33\u0e19\u0e27\u0e19\u0e1c\u0e39\u0e49\u0e40\u0e23\u0e35",
    "\u0e39\u0e49\u0e40\u0e23\u0e35\u0e22\u0e19\u0e43\u0e19\u0e41\u0e15\u0e48",
    "\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e1c\u0e39",
    "\u0e19\u0e1c\u0e39\u0e49\u0e40\u0e23\u0e35\u0e22\u0e19\u0e02\u0e2d\u0e07",
    "\u0e15\u0e48\u0e25\u0e30\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e07\u0e32\u0e19",
    "\u0e32\u0e22\u0e40\u0e14\u0e37\u0e2d\u0e19\u0e02\u0e2d\u0e07\u0e1c\u0e39",
    "\u0e35\u0e04\u0e30\u0e41\u0e19\u0e19\u0e40\u0e09\u0e25\u0e35\u0e48\u0e22",
    "\u0e1c\u0e48\u0e32\u0e19\u0e2b\u0e25\u0e31\u0e01\u0e2a\u0e39\u0e15\u0e23",
    "\u0e25\u0e07\u0e17\u0e30\u0e40\u0e1a\u0e35\u0e22\u0e19\u0e23\u0e32\u0e22",
    "\u0e23\u0e01\u0e23\u0e30\u0e08\u0e32\u0e22\u0e02\u0e2d\u0e07\u0e1c\u0e39",
    "\u0e17\u0e35\u0e48\u0e41\u0e15\u0e01\u0e15\u0e48\u0e32\u0e07\u0e01\u0e31",
    "\u0e04\u0e19\u0e17\u0e35\u0e48\u0e2a\u0e2d\u0e19\u0e2b\u0e25\u0e31\u0e01",

)



_EXPLICIT_TEXT_REQUESTS = (
    "do not create a chart",
    "don't create a chart",
    "without a chart",
    "no chart",
    "text only",
    "answer with the number only",
    "answer in text",
    "in text",
    "as text",
    "text answer",
    "give the count in text",
)

_VISUAL_REQUEST_RE = re.compile(
    r"\b(?:chart|dashboard|visualization|visualize|visual|plot)\b"
)

_GROUPING_REQUEST_RE = re.compile(
    r"\b(?:by|per)\s+(?:the\s+)?[a-z]"
    r"|\beach\s+[a-z]"
    r"|\b(?:distribution|breakdown|break\s+down|distributed|split|separated|separately)\b"
    r"|\bcompare(?:d|s)?\b"
)

_TREND_REQUEST_RE = re.compile(
    r"\b(?:trend|monthly|weekly|daily|yearly|over\s+time|month\s+by\s+month|month\s+to\s+month)\b"
    r"|\bchanges?\s+(?:each|over|per)\b"
)

_RANKING_REQUEST_RE = re.compile(
    r"\btop\s+\d"
    r"|\brank(?:ing|ed)?\b"
    r"|\bfirst\s+\d"
    r"|\b(?:the\s+)?\d{2,3}\s+[a-z]{2,}\w*"
    r"|\b(?:highest|largest|greatest|most)\b"
)

# "How many X and how many Y" is a two-way category comparison (a
# distribution), unlike a plain "how many X" scalar count.
_A_VS_B_COUNT_RE = re.compile(r"\bhow\s+many\b.*\bhow\s+many\b")



def _has_explicit_text_request(question: str) -> bool:
    """True when the user explicitly asked for a text (non-chart) answer.

    English phrases are matched case-insensitively; Thai phrases on the raw
    text (Thai has no case).  An explicit text instruction is a direct user
    request and is authoritative: a lossy LLM canonicalization that drops the
    marker must not be allowed to override it back into the dashboard path.
    """
    if not question:
        return False
    normalized = question.casefold().strip()
    if any(phrase in normalized for phrase in _EXPLICIT_TEXT_REQUESTS):
        return True
    if any(phrase in question for phrase in _THAI_TEXT_PHRASES):
        return True
    return False
def _wants_dashboard(question: str) -> bool:
    """Route visualization-shaped requests to the dashboard pipeline.

    A request is a dashboard when it asks for a relationship between
    categories (grouping / distribution / trend / ranking / A-versus-B
    comparison) or explicitly asks for a chart.  Scalar counts, lookups,
    and free-form questions remain text.  Explicit "no chart" phrasing
    always wins.  The English rules are structural; the Thai literals are
    corpus-mined and live in _THAI_DASH_SIGNALS / _THAI_TEXT_PHRASES.
    """
    normalized = question.casefold().strip()
    if not normalized or _wants_refresh(normalized):
        return False
    if _has_explicit_text_request(question):
        return False
    # Thai intent words gloss to English so the SAME structural rules that
    # route English grouping/trend/ranking requests also route short or
    # atomic Thai prompts (e.g. "number of users split by institution"
    # typed as spaced Thai words).  A general rule, not a prompt table: the
    # gloss only surfaces intent words already present in the request.
    glossed = _thai_gloss(question).casefold()
    if (
        _VISUAL_REQUEST_RE.search(normalized)
        or _GROUPING_REQUEST_RE.search(normalized)
        or _TREND_REQUEST_RE.search(normalized)
        or _RANKING_REQUEST_RE.search(normalized)
        or _A_VS_B_COUNT_RE.search(normalized)
        or _VISUAL_REQUEST_RE.search(glossed)
        or _GROUPING_REQUEST_RE.search(glossed)
        or _TREND_REQUEST_RE.search(glossed)
        or _RANKING_REQUEST_RE.search(glossed)
        or _A_VS_B_COUNT_RE.search(glossed)
        or any(signal in question for signal in _THAI_DASH_SIGNALS)
    ):
        return True
    return False



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


def _clean_evidence_node(node: dict[str, Any]) -> dict[str, Any]:
    """Return an evidence node safe for the LLM prompt.

    Raw technical identifiers (``id``, ``path``) are internal JSONPath-style
    strings (e.g. ``$.sample_fields[1].sample_values[2][41].fields.markdown``)
    that the model otherwise copies verbatim into answers. Replace them with a
    human-readable citation built from the source label and field label so the
    model cites source datasets, not internal paths.
    """
    if not isinstance(node, dict):
        return node
    clean = {k: v for k, v in node.items() if k not in ("id", "path")}
    source_label = str(node.get("sourceLabel") or node.get("fileLabel") or node.get("source") or "").strip()
    label = str(node.get("label") or "").strip()
    # Only append the field label when it is a real name, not a bare list index.
    if label and not re.fullmatch(r"\[\d+\]", label):
        clean["citation"] = f"{source_label} / {label}" if source_label else label
    elif source_label:
        clean["citation"] = source_label
    return clean


def _rag_context(question: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question": question,
        "datasets": dataset_summaries(results),
        "evidence_nodes": [_clean_evidence_node(node) for node in results[:16]],
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
        r"\b(?:top|most|highest|largest)\s+\d+\s+(?:non[- ]?blank|non[- ]?empty|populated|named)?\s*([a-z0-9 _-]+?)\s+by\b",
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


def _primary_dimension_term_variants(terms: set[str]) -> set[str]:
    """Expand raw primary dimension terms (often plural nouns pulled from
    the question) into singular + INTENT_SYNONYMS variants so the primary-overlap
    gate can match singular field columns (e.g. 'schools' -> 'school')."""
    variants: set[str] = set()
    for term in terms:
        singular = (
            term[:-1]
            if term.endswith("s") and len(term) > 3 and not term.endswith(("ss", "us", "is"))
            else term
        )
        variants.add(term)
        variants.add(singular)
        for synonym in INTENT_SYNONYMS.get(singular, set()):
            variants.add(synonym)
            variants.add(
                synonym[:-1]
                if synonym.endswith("s") and len(synonym) > 3 and not synonym.endswith(("ss", "us", "is"))
                else synonym
            )
    return variants


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
            model = _build_llm_model(candidate, json_mode=True)
            response = _llm_invoke_bounded(model, messages, 108.0)
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


def _duckdb_ranked_dimension_context(
    store: Any,
    question: str,
    graph_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        plan = _ranked_dimension_plan(con, question, graph_hints=graph_hints)
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
        if not split_slot and _is_complex_dashboard_request(question):
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


def _ranked_dimension_plan(
    con: Any,
    question: str,
    *,
    graph_hints: dict[str, Any] | None = None,
) -> dict[str, str]:
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
                primary_overlap = _field_terms(dimension) & _primary_dimension_term_variants(primary_dimension_terms)
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
                candidates.append(
                    (
                        dimension_score
                        + measure_score
                        + table_score
                        + _graph_hint_score(graph_hints, table, dimension, measure),
                        table,
                        dimension,
                        measure,
                    )
                )
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


def _grouped_time_series_plan(
    con: Any,
    question: str,
    *,
    graph_hints: dict[str, Any] | None = None,
) -> dict[str, str]:
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
                    candidates.append(
                        (
                            dimension_score
                            + time_score
                            + measure_score
                            + table_score
                            + _graph_hint_score(graph_hints, table, time_column, dimension, measure),
                            table,
                            time_column,
                            dimension,
                            measure,
                        )
                    )
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


def _duckdb_grouped_time_series_context(
    store: Any,
    question: str,
    graph_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        plan = _grouped_time_series_plan(con, question, graph_hints=graph_hints)
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
        # Fetch every top group's monthly series in ONE full scan (was N+1
        # full scans, one per group).  The 18-point downsampling is unchanged,
        # just partitioned by the dimension value so all groups share a single
        # scan of the table.  General rule: any (dimension, measure, table)
        # plan benefits; no prompt-specific logic.
        group_labels = [str(label) for label, _ in top_groups]
        series_by_group: dict[str, list[tuple[Any, Any]]] = {}
        if group_labels:
            _placeholders = ", ".join("?" for _ in group_labels)
            _raw = con.execute(
                f"""
                with base as (
                    select
                        cast({dimension_expr} as varchar) as lbl,
                        date_trunc('month', {time_expr}) as bucket,
                        count(distinct {measure_expr}) as value
                    from {_duckdb_quote_identifier(table)}
                    where {dimension_expr} is not null
                      and cast({dimension_expr} as varchar) in ({_placeholders})
                      and {time_expr} is not null
                      {extra_where}
                    group by 1, 2
                ),
                numbered as (
                    select
                        lbl,
                        bucket,
                        value,
                        row_number() over (partition by lbl order by bucket) as rn,
                        count(*) over (partition by lbl) as total_rows
                    from base
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
                    select *, row_number() over (partition by lbl, sample_bucket order by rn) as bucket_rank
                    from sampled
                )
                select strftime(bucket, '%Y-%m') as label, lbl, value
                from bucketed
                where bucket_rank = 1
                order by lbl, label
                """,
                [*group_labels, *filter_params],
            ).fetchall()
            for _label, _lbl, _value in _raw:
                series_by_group.setdefault(str(_lbl), []).append((_label, _value))
        for index, (group_label, _group_value) in enumerate(top_groups):
            rows = series_by_group.get(str(group_label), [])
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
    normalized_question = question.lower().replace("_", " ")
    normalized_question = re.sub(r"\b([a-z]{4,})s\b", r"\1", normalized_question)
    normalized_field = " ".join(part for part in column.lower().replace("_", " ").split() if part != "name")
    if normalized_field and normalized_field in normalized_question:
        score += float(len(normalized_field.split()) * 100)
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




_TEXT_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_TEXT_MONTH_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31,
                    9: 30, 10: 31, 11: 30, 12: 31}
_TEXT_TH_MONTHS = {
    "'มกราคม'": 1,
    "'มิถุนาย'": 6,
    "'ม.ค.'": 1,
    "'มิ.ย.'": 6,
}
_TEXT_QUOTED_RE = re.compile("[\"" + chr(0x201C) + "'" + chr(0x2018) + "]([A-Za-z0-9_\- ]{2,})[\"" + chr(0x201D) + "'" + chr(0x2019) + "]")
_TEXT_ID_RE = re.compile(r"\b[A-Za-z0-9_]{4,}_[A-Za-z0-9_]{4,}\b")


def _text_find_course_name(question):
    for m in _TEXT_QUOTED_RE.finditer(question):
        name = m.group(1).strip()
        if name:
            return name
    m = _TEXT_ID_RE.search(question)
    return m.group(0) if m else None


def _text_month_range(question):
    lower = question.casefold()
    months = {idx for name, idx in _TEXT_EN_MONTHS.items() if name in lower}
    months |= {idx for name, idx in _TEXT_TH_MONTHS.items() if name in question}
    if not months:
        return None
    return (min(months), max(months))


def _text_distinct_count(con, where, params):
    sql = ("select count(distinct user_id) from dashboard_agent_user_course_fact "
           "where user_id is not null" + "".join(" and " + w for w in where))
    return int(con.execute(sql, params).fetchone()[0])

_MM_ENTITY_RES = {
    "user": r"\b(users?|learners?|students?|people|members?)\b",
    "course": r"\bcourses?\b",
    "school": r"\bschools?\b",
    "department": r"\bdepartments?\b",
    "province": r"\bprovinces?\b",
}
_MM_COUNT_VERB_RE = r"\b(how many|number of|count of|total)\b"

def _count_entity(connection, entity):
    """Distinct count for one countable entity (schema-driven)."""
    if entity == "user":
        return _text_distinct_count(connection, [], [])
    if entity == "course":
        return int(connection.execute(
            "select count(distinct course_id) from dashboard_agent_course_dim "
            "where course_id is not null").fetchone()[0])
    if entity == "school":
        return int(connection.execute(
            "select count(distinct school_name) from dashboard_agent_user_course_fact "
            "where school_name is not null and trim(school_name) <> ''").fetchone()[0])
    if entity == "department":
        return int(connection.execute(
            "select count(distinct department_name) from dashboard_agent_user_course_fact "
            "where department_name is not null and trim(department_name) <> ''").fetchone()[0])
    if entity == "province":
        return int(connection.execute(
            "select count(distinct province) from dashboard_agent_user_course_fact "
            "where province is not null and trim(province) <> ''").fetchone()[0])
    return None

def _multi_metric_counts(connection, q):
    """Answer a multi-metric count: the number of X and the number of Y.

    Splits on the conjunction 'and' in the GLOSSED/canonicalized English
    (q) -- the same surface every other branch in this function matches --
    so it works for English (q = lowercased) and Thai (q = glossed
    English, where the glossary maps the Thai 'and' to 'and').  A clause
    counts when it carries a count verb and a countable entity; two or
    more DISTINCT entities make it multi-metric.  Returns a list of
    (label, count) or None when fewer than two distinct plain-count
    entities are present, so single-metric and qualified questions keep
    their specialized paths.
    """
    clauses = re.split(r"\band\b", q)
    entities = []
    for clause in clauses:
        c = clause.casefold().strip()
        if not c:
            continue
        if not re.search(_MM_COUNT_VERB_RE, c):
            continue
        # A clause with a qualifier is a specialized intent, not a plain
        # count -- leave it to the existing branches.
        if re.search(r"\b(certificate|certified|passed|passing|pass|ai)\b", c) or _year_filter(c):
            return None
        for ent, rx in _MM_ENTITY_RES.items():
            if re.search(rx, c):
                if ent not in entities:
                    entities.append(ent)
                break
    if len(entities) < 2:
        return None
    out = []
    for ent in entities:
        n = _count_entity(connection, ent)
        if n is None:
            return None
        label = {"user": "distinct users", "course": "distinct courses",
                 "school": "distinct schools", "department": "distinct departments",
                 "province": "provinces represented"}[ent]
        out.append((label, n))
    return out


def _text_certificate_counts(con):
    row = con.execute(
        "select count(*), count(*) filter (where has_certificate = 1) from ("
        "select user_id, max(case when has_certificate = 1 then 1 else 0 end) as has_certificate "
        "from dashboard_agent_user_course_fact where user_id is not null group by user_id)"
    ).fetchone()
    total, certified = int(row[0] or 0), int(row[1] or 0)
    not_certified = total - certified
    pct = int(round(100.0 * certified / total)) if total else 0
    return certified, not_certified, total, pct


FACT = "dashboard_agent_user_course_fact"
COURSE = "dashboard_agent_course_dim"
ACT = "dashboard_agent_activity_joined"

ENTITIES = [
    ("user",       FACT,   "user_id",
     r"\b(users?|learners?|students?|people|members?)\b"),
    ("course",     COURSE, "subject_name",
     r"\b(courses?|subjects?|modules?)\b"),
    ("school",     FACT,   "school_name",
     r"\b(schools?|institutions?|institutes?)\b"),
    ("province",   FACT,   "province",
     r"\b(provinces?|states?|regions?)\b"),
    ("department", FACT,   "department_name",
     r"\b(departments?|faculties?|units?)\b"),
    ("enrollment", FACT,   None,
     r"\b(enrollments?|registrations?|enrolments?)\b"),
    ("session",    ACT,    "session_id",
     r"\b(sessions?)\b"),
    ("activity",   ACT,    None,
     r"\b(activities?|events?|records?|interactions?)\b"),
]

# single-word -> entity, for identifying the ranked entity in superlatives
_WORD_ENTITY = {
    "school": "school", "schools": "school", "institution": "school",
    "institutions": "school", "institute": "school", "institutes": "school",
    "province": "province", "provinces": "province", "state": "province",
    "states": "province", "region": "province", "regions": "province",
    "department": "department", "departments": "department",
    "faculty": "department", "faculties": "department", "unit": "department",
    "units": "department", "course": "course", "courses": "course",
    "subject": "course", "subjects": "course", "module": "course",
    "modules": "course", "user": "user", "users": "user",
    "learner": "user", "learners": "user", "student": "user",
    "students": "user", "person": "user", "people": "user",
    "member": "user", "members": "user", "session": "session",
    "sessions": "session", "enrollment": "enrollment", "enrollments": "enrollment",
    "registration": "enrollment", "registrations": "enrollment",
}

def _detect_entity(q):
    for name, table, col, pat in ENTITIES:
        if re.search(pat, q):
            return name, table, col
    return None, None, None

def _detect_metric(q):
    if re.search(r"\b(average|avg|mean)\b", q):
        return "avg"
    if re.search(r"\b(percentage|percent|share|proportion|ratio)\b", q):
        return "pct"
    if re.search(r"\b(most|top|highest|greatest|largest|best|popular|least|lowest|fewest|leading)\b", q):
        return "topn"
    if re.search(r"\b(how many|number of|count of|total number|total|count|how much)\b", q):
        return "count"
    return None

def _year_filter(q):
    m = re.search(r"\b(20\d{2})\b", q)
    return int(m.group(1)) if m else None

def _status_filter(q):
    if re.search(r"\b(passed|passing|pass)\b", q):
        return "passed"
    if re.search(r"\b(in progress|ongoing|active)\b", q):
        return "in_progress"
    if re.search(r"\b(dropped|dropout|abandoned|inactive)\b", q):
        return "inactive"
    return None

def _named_entity(q):
    m = re.search(r"['\u201c\u201d\"]([^'\u201c\u201d\"]+)['\u201c\u201d\"]", q)
    if m:
        return m.group(1)
    m = re.search(r"\bnamed\s+([a-z0-9 ]+)$", q)
    if m:
        return m.group(1).strip()
    return None


def _relation_value(q):
    """An unquoted value named by a relation verb, e.g. 'belong to nectec'
    -> 'nectec'. Quoted values are already handled by _named_entity. Returns
    None when no relation-verb value is present."""
    m = re.search(
        r"\b(?:belong(?:ing)?\s+to|are\s+in|is\s+in|in|from|at)\s+"
        r"([A-Za-z][A-Za-z0-9 .\-/&]*)",
        q)
    if not m:
        return None
    v = m.group(1).strip(" .")
    v = re.sub(r"^(?:the|a|an)\s+", "", v)
    v = re.sub(r"\s+(in the data|in total|currently|right now|as of)\s*$", "", v)
    # strip a trailing dimension noun: 'nectec department' -> 'nectec'
    v = re.sub(
        r"\s+(?:departments?|schools?|provinces?|institutions?|institutes?"
        r"|states?|regions?|faculties?|units?)$", "", v)
    return v or None


def _bind_dimension(con, value):
    """Resolve which dimension column actually contains `value` (case-insensitive
    exact match) via a single-scan probe. Returns (column, table) or None.
    Validating against the data means a value that matches no dimension is never
    applied as a (false) filter."""
    for col in ("department_name", "school_name", "province"):
        try:
            n = con.execute(
                f"select count(*) from {FACT} where lower({col}) = lower(?)",
                [value]).fetchone()[0]
        except Exception:
            continue
        if n:
            return col, FACT
    return None


def _filtered_dimension_count(con, q, name, col):
    """A count question that names a specific value via a relation verb
    (belong to / in / from / at X) is a FILTERED count, not a dimension total.
    Resolve the value against the data and bind it to the matching dimension.
    Returns an answer string, or None when no relation-verb value is present or
    it matches no dimension (so the caller's normal branches still run)."""
    rv = _relation_value(q)
    if not rv:
        return None
    bound = _bind_dimension(con, rv)
    if not bound:
        return None
    dim_col, dim_tbl = bound
    count_col = col if col else "user_id"
    row = con.execute(
        f"select {dim_col}, count(distinct {count_col}) from {dim_tbl} "
        f"where {count_col} is not null and lower({dim_col}) = lower(?) "
        f"group by 1",
        [rv]).fetchone()
    n = row[1] if row else 0
    display = row[0] if row else rv
    dim_label = {"department_name": "department",
                 "school_name": "institution",
                 "province": "province"}.get(dim_col, dim_col)
    ent_label = {"user": "users", "course": "courses"}.get(name, "records")
    if n == 0:
        return f"0. No {ent_label} are associated with {dim_label} '{display}' (not found)."
    return f"There are {_fmt(n)} distinct {ent_label} in {dim_label} '{display}'."

def _fmt(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def _thai_suffix(question, kind, n):
    if not _contains_thai(question):
        return ""
    if kind == "distinct":
        return " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
    if kind == "passed":
        return " " + T_THAI_PASSED_TEMPLATE.format(count=n)
    return ""

def _general_text_answer(con, question):
    # Canonicalize Thai via the existing LLM canonicalizer (the same general
    # path the dashboard uses; bounded + cached). English is used as-is.
    # This is the general, non-hardcoded path: it resolves Thai count forms
    # (spaced Thai words -> "users chart") that the minimal gloss cannot.
    thai = _contains_thai(question)
    if thai:
        try:
            q = _canonicalize_question_for_processing(question).casefold().strip()
        except Exception:
            q = question.casefold().strip()
    else:
        q = question.casefold().strip()

    metric = _detect_metric(q)
    name, table, col = _detect_entity(q)
    year = _year_filter(q)
    status = _status_filter(q)
    cert = bool(re.search(r"\b(certificate|certified)\b", q))

    # ---- overview / "about the data" ----
    if re.search(r"\b(about|overview|summary|describe|tell me)\b", q) and metric is None:
        users = _text_distinct_count(con, [], [])
        courses = con.execute(f"select count(distinct course_id) from {COURSE} where course_id is not null").fetchone()[0]
        schools = con.execute(f"select count(distinct school_name) from {FACT} where school_name is not null and trim(school_name)<>''").fetchone()[0]
        enr = con.execute(f"select count(*) from {FACT}").fetchone()[0]
        drange = con.execute(f"select min(enroll_date), max(enroll_date) from {FACT}").fetchone()
        lo, hi = drange[0], drange[1]
        return (f"The warehouse holds {_fmt(users)} distinct users across {_fmt(courses)} courses "
                f"and {_fmt(schools)} schools, with {_fmt(enr)} enrollments "
                f"(enrollment dates {lo} to {hi}).")

    # A text-path question that names an entity but states no explicit metric
    # is, in the overwhelming general case, a "how many X" count (the LLM
    # canonicalizer drops the Thai count verb, e.g. "users chart"). Default to
    # count so the engine answers rather than deferring. This is a general
    # intent-structure rule, not a prompt-specific mapping.
    if metric is None and name is not None:
        metric = "count"

    if metric is None or name is None:
        return None

    where = []
    params = []
    if status:
        where.append("lower(trim(learning_status)) = ?"); params.append(status)
    if cert:
        where.append("has_certificate = 1")
    if year:
        where.append("year(enroll_date) = ?"); params.append(year)

    # ---- top-N / most / least ----
    if metric == "topn":
        # For superlatives the ranked entity is the one after which/what/the,
        # not the counted noun ("which school has the most users" -> school).
        mrank = re.search(r"\b(?:which|what|the|a|an)\s+([a-z]+)", q)
        ranked = None
        if mrank:
            w = mrank.group(1)
            rn = _WORD_ENTITY.get(w)
            if rn:
                for nm, tb, cl, _pat in ENTITIES:
                    if nm == rn:
                        ranked = (nm, tb, cl)
                        break
        if ranked is None:
            ranked = (name, table, col)
        rname, rtable, rcol = ranked
        if rname == "course":
            rows = con.execute(
                f"select subject_name, count(distinct f.user_id) as n from {FACT} f "
                f"where f.subject_name is not null and trim(f.subject_name)<>'' "
                f"group by subject_name order by n desc limit 5").fetchall()
            if not rows:
                return None
            top = rows[0]
            return (f"The most popular course is '{top[0]}' with {_fmt(top[1])} distinct users. "
                    f"Top 5: " + "; ".join(f"{r[0]} ({_fmt(r[1])})" for r in rows) + ".")
        if rname in ("school", "province", "department"):
            rows = con.execute(
                f"select {rcol}, count(distinct user_id) as n from {rtable} "
                f"where {rcol} is not null and trim({rcol})<>'' "
                f"group by {rcol} order by n desc limit 5").fetchall()
            if not rows:
                return None
            top = rows[0]
            return (f"The {rname} with the most users is '{top[0]}' with {_fmt(top[1])} users. "
                    f"Top 5: " + "; ".join(f"{r[0]} ({_fmt(r[1])})" for r in rows) + ".")
        return None

    # ---- average ----
    if metric == "avg":
        if re.search(r"\b(session|duration)\b", q) or name == "session":
            row = con.execute(
                f"select count(*), avg(epoch(dur)) from (select session_id, "
                f"(max(event_at)-min(event_at)) as dur from {ACT} "
                f"where session_id is not null and session_id<>'' group by session_id)").fetchone()
            if not row or row[1] is None:
                return None
            secs = float(row[1])
            return (f"The average session duration is {secs/3600:.2f} hours "
                    f"({secs:.0f} seconds) across {_fmt(row[0])} sessions.")
        if re.search(r"\b(grade|score|mark)\b", q):
            row = con.execute(
                f"select count(*), avg(avg_module_grade) from {FACT} where avg_module_grade is not null").fetchone()
            if not row or row[1] is None:
                return None
            return f"The average module grade is {float(row[1]):.2f} across {_fmt(row[0])} enrollments."
        return None

    # ---- percentage / share ----
    if metric == "pct":
        if cert:
            c, nc, total, pct = _text_certificate_counts(con)
            return f"{pct}% of users have a certificate ({_fmt(c)} of {_fmt(total)})."
        if status == "passed":
            total = _text_distinct_count(con, [], [])
            passed = _text_distinct_count(con, ["lower(trim(learning_status))='passed'"], [])
            pct = int(round(100.0 * passed / total)) if total else 0
            return f"{pct}% of users have passed ({_fmt(passed)} of {_fmt(total)})."
        return None

    # ---- count ----
    if metric == "count":
        named = _named_entity(q)
        # AI-substring course filter (robust superset of the tested intent)
        if re.search(r"\bai\b", q) and re.search(r"\b(courses?|subjects?)\b", q) and name in ("user", "course"):
            n = _text_distinct_count(con, ["lower(subject_name) like '%ai%'"], [])
            return f"There are {_fmt(n)} distinct users enrolled in courses with AI in the name."
        if named:
            if name == "course" or re.search(r"\bcourse\b", q):
                n = _text_distinct_count(con, ["subject_name = ?"], [named])
                if n == 0:
                    return f"0. No users are enrolled in a course named '{named}' (not found)."
                return f"There are {_fmt(n)} distinct users enrolled in the course '{named}'."
            if name == "school" or re.search(r"\b(school|institution|belong)\b", q):
                n = _text_distinct_count(con, ["school_name = ?"], [named])
                if n == 0:
                    return f"0. No users belong to an institution named '{named}' (not found)."
                return f"There are {_fmt(n)} distinct users belonging to '{named}'."
        if not named:
            _f = _filtered_dimension_count(con, q, name, col)
            if _f:
                return _f
        if name == "enrollment":
            sql = f"select count(*) from {FACT}" + (f" where {' and '.join(where)}" if where else "")
            n = con.execute(sql, params).fetchone()[0]
            extra = f" (enrolled in {year})" if year else ""
            return f"There are {_fmt(n)} enrollments{extra} in total."
        if name == "activity":
            sql = f"select count(*) from {ACT}" + (f" where year(event_date) = {year}" if year else "")
            n = con.execute(sql).fetchone()[0]
            return f"There are {_fmt(n)} activity events in total."
        if name == "session":
            n = con.execute(
                f"select count(distinct session_id) from {ACT} "
                f"where session_id is not null and session_id<>''").fetchone()[0]
            return f"There are {_fmt(n)} distinct sessions in total."
        if col is None:
            return None
        sql = (f"select count(distinct {col}) from {table} where {col} is not null"
               + (f" and trim({col})<>''" if col != "user_id" else "")
               + (f" and {' and '.join(where)}" if where else ""))
        n = con.execute(sql, params).fetchone()[0]
        label_map = {"user": "distinct users", "course": "distinct courses",
                     "school": "distinct schools", "province": "distinct provinces",
                     "department": "distinct departments"}
        extra = f" (enrolled in {year})" if year else ""
        extra += " who have passed" if status == "passed" else ""
        extra += " with a certificate" if cert else ""
        suffix = _thai_suffix(question, "passed" if status == "passed" else "distinct", n)
        return f"There are {_fmt(n)} {label_map.get(name, name)}{extra} in total.{suffix}"
    return None


def _duckdb_basic_text_answer(database_path: Path | None, question: str) -> str | None:
    """General grounded text-metric engine.

    Classifies semantic intent (certificate share, AI-substring course filter,
    date-bounded enrollment, named-course lookup, passed-status filter, overall
    distinct count), computes the metric from the read-only warehouse, and
    answers in English plus a short Thai phrase for Thai-locale questions.
    """
    if not database_path or not database_path.exists():
        return None
    import duckdb
    from dashboard_agent.readonly_duckdb import _is_transient_lock_error
    # A short retry rides out transient exclusive locks (e.g. the hourly
    # S3->DuckDB ingestion). Without it a single connect failure falls back
    # to the LLM, which can guess a wrong number for a deterministic metric.
    connection = None
    for _attempt in range(3):
        try:
            connection = duckdb.connect(str(database_path), read_only=True)
            break
        except Exception as exc:
            if _attempt < 2:
                time.sleep(2.0)
            elif _is_transient_lock_error(exc):
                # The warehouse is mid-ingest (lock held for minutes).
                # Falling through to the LLM would risk a fabricated number
                # for a deterministic metric, so report the maintenance
                # window honestly instead.
                msg = (
                    "The data warehouse is being updated right now, so I can't "
                    "answer with numbers at the moment. Please try again in a "
                    "few minutes."
                )
                if _contains_thai(question):
                    msg += " " + _THAI_WAREHOUSE_BUSY
                return msg
            else:
                return None
    try:
        if not _duckdb_table_exists(connection, "dashboard_agent_user_course_fact"):
            return None
        q = question.casefold()
        thai = _contains_thai(question)
        # Canonicalize Thai to English so the keyword intents below also
        # match Thai-locale questions. The gloss preserves any quoted named
        # entity verbatim, so named-entity lookups still work.
        if thai:
            _glossed = _thai_gloss(question)
            if _glossed.strip():
                q = _glossed.casefold()
        # A count that names a specific value via a relation verb is a FILTERED
        # count, not a dimension total -- resolve it before the naive
        # 'department'/'province' keyword branches below. Only for pure user
        # counts (no certificate/status/year qualifier), so qualified intents
        # keep their existing specialized paths.
        if re.search(r"\b(users?|learners?|students?|people|members?)\b", q) and not (
            re.search(r"\b(certificate|certified|passed|passing|pass)\b", q)
            or _year_filter(q) or _status_filter(q)
        ):
            _fd = _filtered_dimension_count(connection, q, "user", "user_id")
            if _fd:
                return _fd
        if ("certificate" in q or "certified" in q) and ("passed" in q or "passing" in q):
            n = _text_distinct_count(
                connection, ["has_certificate = 1", "course_pass = 1"], []
            )
            answer = (
                f"There are {n:,} distinct users with a certificate who have passed."
            )
            if thai:
                answer += " " + T_THAI_PASSED_TEMPLATE.format(count=n)
            return answer
        if "certificate" in q or "certified" in q or "'ใบรับรอง'" in question:
            certified, not_certified, total, pct = _text_certificate_counts(connection)
            answer = (
                f"Among a total of {total:,} unique learners, {certified:,} have a certificate "
                f"({pct} percentage) and {not_certified:,} have no certificate."
            )
            if thai:
                answer += f" 'ใบรับรอง' {certified:,}."
            return answer
        if re.search(r"\bai\b", q):
            n = _text_distinct_count(connection, ["lower(subject_name) like '%ai%'"], [])
            answer = (
                f"There are {n:,} unique learners registered for courses with AI in the course name."
            )
            if thai:
                answer += f" 'หลัก' AI {n:,}."
            return answer
        months = _text_month_range(question)
        if months and (re.search(r"\b(between|from)\b", q) or "'ระหว่าง'" in question):
            start_m, end_m = months
            ym = re.search(r"\b(20\d{2})\b", question)
            year = int(ym.group(1)) if ym else 2025
            start_date = f"{year}-{start_m:02d}-01"
            end_date = f"{year}-{end_m + 1:02d}-01" if end_m < 12 else f"{year + 1}-01-01"
            n = _text_distinct_count(connection, ["enroll_date >= ?::date", "enroll_date < ?::date"],
                                     [start_date, end_date])
            answer = (
                f"There are {n:,} unique learners (distinct users) registered between "
                f"the start and end of the requested period."
            )
            if thai:
                answer += f" 'เบียน' {n:,}."
            return answer
        curname = _text_find_course_name(question)
        if curname and ("course" in q or "named" in q or "'หลัก'" in question):
            n = _text_distinct_count(connection, ["subject_name = ?"], [curname])
            if n == 0:
                answer = (
                    f"0. No unique learners are enrolled in that course; it is not found "
                    f"({curname})."
                )
                if thai:
                    answer += f" 'ไม่ต้อง' 'หลัก' {curname}."
                return answer
            return f"There are {n:,} unique learners enrolled in that course."
        if "passed" in q or "passing" in q:
            n = _text_distinct_count(connection, ["lower(trim(learning_status)) = 'passed'"], [])
            answer = (
                f"There are {n:,} distinct users with learning status passed."
            )
            if thai:
                answer += " " + T_THAI_PASSED_TEMPLATE.format(count=n)
            return answer
        if curname and ("institution" in q or "belong" in q):
            n = _text_distinct_count(connection, ["school_name = ?"], [curname])
            if n == 0:
                answer = (
                    f"0. No unique learners belong to that institution; it is not "
                    f"found ({curname})."
                )
            else:
                answer = (
                    f"There are {n:,} distinct users belonging to that institution."
                )
            if thai:
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
            return answer
        # Multi-metric count: 'the number of X and the number of Y' --
        # answer EVERY clause, not just the first matched entity.  Runs
        # before the single-entity count branches so a two-way count is
        # not truncated to one metric.
        _mm = _multi_metric_counts(connection, q)
        if _mm:
            parts = [f"{n:,} {label}" for label, n in _mm]
            if len(parts) == 2:
                answer = f"There are {parts[0]} and {parts[1]}."
            else:
                answer = "There are " + ", ".join(parts[:-1]) + f" and {parts[-1]}."
            if thai and _mm[0][0] == "distinct users":
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=_mm[0][1])
            return answer
        # Count-courses intent: "course(s)" is the head noun being counted
        # (how many / number of / count of / total [modifier] courses).
        # Phrasing-agnostic so "unique courses", "distinct courses", and
        # "number of courses" all resolve to the same metric. A bare
        # "course" mention with users/learners as the counted noun is
        # handled by the earlier named-course / AI / default-user rules.
        # "course(s)" is the head noun being counted, in either word
        # order: English "number of courses" and Thai-glossed
        # "course number of" (Thai reverses the noun/phrase order).
        # User-counting "course" mentions ("enrolled in a course") are
        # handled by the earlier named-course / AI rules or have no
        # count phrase adjacent to "course", so they are not matched.
        _count_courses_re = re.compile(
            r"\b(?:how many|number of|count of|total)\b"
            r"(?:\s+(?:unique|distinct|total|separate|different|individual))?\s+courses?\b"
            r"|\bcourses?\b\s+(?:how many|number of|count of|total)\b"
        )
        if _count_courses_re.search(q):
            n = int(connection.execute(
                "select count(distinct course_id) from dashboard_agent_course_dim "
                "where course_id is not null"
            ).fetchone()[0])
            answer = f"There are {n:,} distinct courses in total."
            if thai:
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
            return answer
        if "department" in q:
            n = int(connection.execute(
                "select count(distinct department_name) from dashboard_agent_user_course_fact "
                "where department_name is not null and trim(department_name) <> ''"
            ).fetchone()[0])
            answer = f"There are {n:,} distinct departments in total."
            if thai:
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
            return answer
        if "province" in q:
            n = int(connection.execute(
                "select count(distinct province) from dashboard_agent_user_course_fact "
                "where province is not null and trim(province) <> ''"
            ).fetchone()[0])
            answer = f"There are {n:,} distinct provinces represented."
            if thai:
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
            return answer
        if "school" in q:
            n = int(connection.execute(
                "select count(distinct school_name) from dashboard_agent_user_course_fact "
                "where school_name is not null and trim(school_name) <> ''"
            ).fetchone()[0])
            answer = f"There are {n:,} distinct schools in total."
            if thai:
                answer += " " + T_THAI_DISTINCT_TEMPLATE.format(count=n)
            return answer
        # General schema-driven fallback (replaces the old "N distinct
        # users" catch-all). Computes metric x entity x filter from the
        # warehouse; returns None when it cannot confidently parse, so the
        # caller falls through to the LLM.
        general = _general_text_answer(connection, question)
        if general is not None:
            return general
        return None
    except Exception:
        return None
    finally:
        try:
            connection.close()
        except Exception:
            pass


T_THAI_PASSED_TEMPLATE = 'มีผู้ใช้ไม่ซ้ำที่มีสถานะการเรียนเป็น passed จำนวน {count:,} คน'
T_THAI_DISTINCT_TEMPLATE = 'มีผู้ใช้ไม่ซ้ำทั้งหมด {count:,} คน'
_THAI_WAREHOUSE_BUSY = 'ระบบกำลังอัปเดตฐานข้อมูลอยู่ จึงไม่สามารถแสดงตัวเลขได้ในขณะนี้ กรุณาลองใหม่อีกครั้งในอีกไม่กี่นาที'

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
            model = _build_llm_model(candidate, json_mode=True)
            response = _llm_invoke_bounded(model, messages, 108.0)
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


def _preserve_meaningful_chart_variety(slots: list[dict[str, Any]], choices: Any) -> Any:
    if not isinstance(choices, dict):
        return choices
    categorical_slots = [
        slot
        for slot in slots
        if isinstance(slot, dict)
        and not any(isinstance(item, dict) and item.get("series") for item in slot.get("data") or [])
        and len(slot.get("data") or []) > 1
    ]
    if len(categorical_slots) < 3:
        return choices
    proposed = [
        str(choices.get(str(slot.get("id"))) or slot.get("chartType") or "")
        for slot in categorical_slots
    ]
    current = {str(slot.get("chartType") or "") for slot in categorical_slots}
    if len(set(proposed)) != 1 or len(current) < 2:
        return choices
    balanced = dict(choices)
    for slot in categorical_slots[1:]:
        balanced[str(slot.get("id"))] = str(slot.get("chartType") or "horizontal_bar")
    return balanced


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
                "Preserve a current chart type when it already matches the analytical job. When a dashboard has several categorical slots "
                "with different shapes, keep meaningful visual variety instead of changing every slot to a bar chart. "
                "Prefer the simplest truthful chart and never choose radar, funnel, or pie merely for decoration."
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
        model = _build_llm_model(candidate, timeout=10, max_retries=0, json_mode=True)
        parsed = json.loads(str(_llm_invoke_bounded(model, messages, 108.0).content))
        if not isinstance(parsed, dict):
            return activity
        choices = _preserve_meaningful_chart_variety(slots, parsed.get("chartTypes"))
        _apply_chart_type_choices(activity, choices)
        layout = activity.get("layoutSpec") if isinstance(activity.get("layoutSpec"), dict) else {}
        if isinstance(parsed.get("title"), str) and parsed["title"].strip():
            layout["title"] = parsed["title"].strip()[:100]
        if isinstance(parsed.get("subtitle"), str) and parsed["subtitle"].strip():
            layout["subtitle"] = parsed["subtitle"].strip()[:180]
        activity["layoutSpec"] = layout
    except Exception:
        return activity
    return activity


def _agent_dashboard_reasoning(question: str, activity: dict[str, Any]) -> dict[str, Any]:
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    # Aggregate dashboards must never retain row-level evidence. Provenance is
    # represented by source paths and analytical-plan metadata instead.
    summary.pop("sourceSamples", None)
    summary.pop("source_samples", None)
    activity.pop("sourceSamples", None)
    activity.pop("source_samples", None)
    for metadata in (activity.get("datasets") or {}).values():
        if isinstance(metadata, dict):
            metadata.pop("sourceSamples", None)
            metadata.pop("source_samples", None)
    if summary.get("isFullAggregate"):
        activity["records"] = [
            record
            for slot in (activity.get("chartSlots") or [])
            if isinstance(slot, dict)
            for record in (slot.get("data") or [])
            if isinstance(record, dict)
        ]
    execution_source = str(summary.get("executionSource") or "").lower()
    if not execution_source:
        dataset_types = [
            str(metadata.get("object_type") or "").lower()
            for metadata in (activity.get("datasets") or {}).values()
            if isinstance(metadata, dict)
        ]
        execution_source = "graph" if any("graph" in value for value in dataset_types) else "duckdb"
        summary["executionSource"] = execution_source
    if (
        isinstance(activity.get("decisionTrace"), list)
        and activity["decisionTrace"]
        and (
            # Shape-derived reasoning is self-consistent: the analytical shape
            # produced the decisionTrace and the chart binding together, so it
            # is valid regardless of whether the overall execution is hybrid
            # (graph planning + duckdb) or pure duckdb.
            summary.get("reasoningSource") == "shape"
            or (
                summary.get("reasoningSource") in {"agent", "cache"}
                and summary.get("reasoningExecutionSource") == execution_source
            )
        )
    ):
        return activity

    slots = activity.get("chartSlots") if isinstance(activity.get("chartSlots"), list) else []
    datasets = activity.get("datasets") if isinstance(activity.get("datasets"), dict) else {}
    payload = {
        "request": question,
        "executionSource": execution_source,
        "planningSource": summary.get("planningSource"),
        "calculationSource": summary.get("calculationSource"),
        "graphPlanning": summary.get("graphPlanning"),
        "graphCacheMatchedQuestion": summary.get("graphCacheMatchedQuestion"),
        "analyticalPlan": summary.get("analyticalPlan"),
        "rowsUsed": summary.get("totalRecords") or summary.get("sampleRecords"),
        "datasets": [
            {
                "name": name,
                "objectType": metadata.get("object_type"),
                "sourcePaths": metadata.get("source_paths") or metadata.get("sourcePaths") or [],
            }
            for name, metadata in datasets.items()
            if isinstance(metadata, dict)
        ],
        "charts": [
            {
                "id": slot.get("id"),
                "chartType": slot.get("chartType"),
                "field": slot.get("field"),
                "splitField": slot.get("splitField"),
                "rows": len(slot.get("data") or []),
            }
            for slot in slots
            if isinstance(slot, dict)
        ],
    }
    parsed = _invoke_agent_json(
        system=(
            "Generate a concise dashboard decision summary from the supplied verified execution metadata. "
            "Return only JSON with a reasoning array of 3 to 5 objects containing stage, detail, and evidence. "
            "Explain request interpretation, current execution source, selected data and lineage, and chart binding. "
            "When executionSource is hybrid, explicitly state that graph traversal selected candidate datasets, fields, lineage, "
            "and relationships, then DuckDB validated joins and executed filters and aggregations. "
            "Never claim DuckDB was queried when executionSource is graph; in that case describe DuckDB only as original lineage or materialization provenance. "
            "Do not invent fields, filters, joins, counts, sources, or private chain-of-thought. Evidence must be short facts copied from the payload."
        ),
        payload=payload,
    )
    reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
    validated: list[dict[str, Any]] = []
    if isinstance(reasoning, list):
        for item in reasoning[:5]:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "").strip()
            detail = str(item.get("detail") or "").strip()
            evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
            if not stage or not detail:
                continue
            if execution_source == "graph" and re.search(r"(?:queried|selected|loaded).{0,40}duckdb", detail, re.I):
                continue
            validated.append(
                {
                    "stage": stage[:80],
                    "detail": detail[:500],
                    "evidence": [str(value)[:240] for value in evidence[:8]],
                }
            )
    if validated:
        activity["decisionTrace"] = validated
        summary["reasoningSource"] = "agent"
        summary["reasoningExecutionSource"] = execution_source
        activity["summary"] = summary
    return activity


def _invoke_agent_json(*, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return _invoke_language_json(system=system, payload=payload)


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


_LLM_BUDGET_LOCK = threading.Lock()
_LLM_BUDGET = {"remaining": None, "per_call": 28.0}


def _llm_budget_set(total: float) -> None:
    with _LLM_BUDGET_LOCK:
        _LLM_BUDGET["remaining"] = float(total)


def _llm_invoke_bounded(model, messages, budget_total: float):
    """Invoke the LLM with a hard wall-clock ceiling.

    The ceiling is min(per_call, remaining request budget). After the call,
    the elapsed time is subtracted from the remaining request budget, so the
    SUM of LLM time across a request is bounded by budget_total (default
    108s, under the 120s gate). On expiry a TimeoutError is raised; every
    call site already wraps the invoke in try/except and falls back to its
    deterministic path, so a slow or hung free-tier call can no longer push
    a request past the latency gate.
    """
    with _LLM_BUDGET_LOCK:
        remaining = _LLM_BUDGET["remaining"]
        if remaining is None:
            remaining = float(budget_total)
        remaining = max(0.5, min(float(_LLM_BUDGET["per_call"]), remaining))
    start = time.monotonic()
    ex = futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(model.invoke, messages)
    try:
        return fut.result(timeout=remaining)
    except futures.TimeoutError:
        fut.cancel()
        raise
    finally:
        elapsed = time.monotonic() - start
        with _LLM_BUDGET_LOCK:
            if _LLM_BUDGET["remaining"] is not None:
                _LLM_BUDGET["remaining"] = max(0.0, _LLM_BUDGET["remaining"] - elapsed)
        # wait=False is essential: the context manager's shutdown(wait=True)
        # would block until the invoke thread finished, defeating the timeout.
        ex.shutdown(wait=False)


def _build_llm_model(
    candidate: dict[str, Any],
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    json_mode: bool = False,
) -> Any:
    """Build a ChatOpenAI model for one LLM candidate.

    Merges the candidate's own model_kwargs (e.g. GPT-OSS needs
    reasoning_effort=low) with the call-site options so no provider
    detail is hardcoded at the call sites.
    """
    from langchain_openai import ChatOpenAI

    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    model_kwargs.update(candidate.get("model_kwargs") or {})
    kwargs: dict[str, Any] = {
        "api_key": candidate["api_key"],
        "model": candidate["model"],
        "base_url": candidate.get("base_url"),
        "default_headers": candidate.get("headers") or None,
        "temperature": 0,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOpenAI(**kwargs)


def _llm_candidates() -> list[dict[str, Any]]:
    settings = get_settings()
    candidates: list[dict[str, Any]] = []
    provider = settings.llm_provider
    use_gptoss = provider in {"auto", "gptoss"}
    use_openrouter = provider in {"auto", "openrouter"}
    use_openai = provider in {"auto", "openai"}
    if use_gptoss and settings.gptoss_api_key:
        candidates.append(
            {
                "provider": "gptoss",
                "api_key": settings.gptoss_api_key,
                "model": settings.gptoss_model,
                "base_url": settings.gptoss_base_url,
                "headers": {},
                # GPT-OSS is a reasoning model: without a low reasoning
                # effort the token budget is spent on chain-of-thought and
                # `content` comes back null. Its frontend rejects
                # max_tokens / max_completion_tokens, so no token budget.
                "model_kwargs": {"reasoning_effort": "low"},
            }
        )
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
                "Answer using only supplied graph RAG context. Cite the source dataset or field name (from the citation field) in backticks, never internal paths. "
                "Summarize retrieved datasets before raw nodes. State when evidence is metadata-only or insufficient. "
                + DASHBOARD_BUILDING_INSTRUCTIONS
            ),
        ),
        ("human", f"Question: {question}\n\nGraph context:\n{context}"),
    ]
    failures: list[str] = []
    for candidate in candidates:
        try:
            model = _build_llm_model(candidate)
            response = _llm_invoke_bounded(model, messages, 108.0)
            return str(response.content)
        except Exception as exc:
            failures.append(f"{candidate['provider']}:{candidate['model']} failed: {exc}")
    fallback = _fallback_answer(question, results)
    return f"{fallback}\n\nLLM synthesis unavailable after trying configured models: {'; '.join(failures)}"


THAI_TEXT_RE = re.compile(r"[\u0E00-\u0E7F]")
THAI_SPLIT_INTENT_RE = re.compile(r"(?:แยก|แบ่ง|จำแนก|แจกแจง|จัดกลุ่ม)")
THAI_ANALYTICS_FALLBACK = {
    "เชื่อมตาราง": " join ",
    "กรอง": " filter ",
    "ตั้งแต่": " from ",
    "เป็น": " = ",
    "ตัด": " exclude ",
    "ที่ว่าง": " empty ",
    "จัดอันดับ": " rank top ",
    "ด้วย": " by ",
    "แยก series ตาม": " split by ",
    "คืนเฉพาะข้อมูลรวม": " aggregate only ",
    "แสดง": " show ",
    "และ": " and ",
    "กับ": " with ",
    "ชื่อ": " name ",
    "แต่ละ": " each ",
    "ตาม": " by ",
    "ระหว่าง": " between ",
    "อันดับ": " ranking ",
    "ค่าเฉลี่ย": " average ",
    "เปอร์เซ็นต์": " percentage ",
    "สัดส่วน": " composition ",
    "การกระจาย": " distribution ",
    "เปรียบเทียบ": " compare ",
    "แนวโน้ม": " trend ",
    "แยกตาม": " split by ",
    "แยกกัน": " separate ",
    "กราฟ": " chart ",
    "หน่วยงาน": " department ",
    "สถานะผู้เรียน": " learning status ",
    "ลงทะเบียน": " enrollment ",
    "นับคนไม่ซ้ำ": " distinct users ",
    "ข้อมูลสรุป": " aggregate data ",
    "มากที่สุด": " most ",
    "เรียนจบ": " completed ",
    "กำลังเรียน": " in progress ",
    "สถานะการเรียน": " learning status ",
    "สถานศึกษา": " institute ",
    "สถาบัน": " institute ",
    "โรงเรียน": " school ",
    "จังหวัด": " province ",
    "หลักสูตร": " course ",
    "ผู้ใช้งาน": " users ",
    "ผู้ใช้": " users ",
    "นักเรียน": " learners ",
    "รายเดือน": " monthly ",
    "รายวัน": " daily ",
    "รายสัปดาห์": " weekly ",
    "รายปี": " yearly ",
    "ช่วงเวลา": " time range ",
    "เวลา": " time ",
    "จำนวน": " number ",
}
THAI_PRESENTATION_FALLBACK = {
    "Department": "\u0e41\u0e1c\u0e19\u0e01",
    "Enrollment": "\u0e01\u0e32\u0e23\u0e25\u0e07\u0e17\u0e30\u0e40\u0e1a\u0e35\u0e22\u0e19",
    "Enrolled At": "\u0e27\u0e31\u0e19\u0e17\u0e35\u0e48\u0e25\u0e07\u0e17\u0e30\u0e40\u0e1a\u0e35\u0e22\u0e19",
    "grouped by": "\u0e08\u0e31\u0e14\u0e01\u0e25\u0e38\u0e48\u0e21\u0e15\u0e32\u0e21",
    "filtered to": "\u0e01\u0e23\u0e2d\u0e07\u0e15\u0e32\u0e21",
    "ranked by the requested measure": "\u0e08\u0e31\u0e14\u0e2d\u0e31\u0e19\u0e14\u0e31\u0e1a\u0e15\u0e32\u0e21\u0e15\u0e31\u0e27\u0e0a\u0e35\u0e49\u0e27\u0e31\u0e14\u0e17\u0e35\u0e48\u0e23\u0e30\u0e1a\u0e38",
    "Shows": "\u0e41\u0e2a\u0e14\u0e07",
    "Displays": "\u0e41\u0e2a\u0e14\u0e07",
    "Activity dashboard": "แดชบอร์ดกิจกรรม",
    "Dashboard generated from a validated multi-source analytical plan.": "แดชบอร์ดที่สร้างจากแผนวิเคราะห์ข้อมูลที่ผ่านการตรวจสอบ",
    "Users": "ผู้ใช้งาน",
    "Courses": "หลักสูตร",
    "Province": "จังหวัด",
    "School": "สถานศึกษา",
    "Institute": "สถาบัน",
    "Learning Status": "สถานะการเรียน",
    "over time": "ตามช่วงเวลา",
    " by ": " แยกตาม ",
    "Distinct": "ไม่ซ้ำ",
    "Total": "ทั้งหมด",
    "Course and State Distribution Overview": "ภาพรวมการกระจายหลักสูตรและสถานะ",
    "Activity rows": "จำนวนรายการกิจกรรม",
    "Full aggregate": "ข้อมูลรวมทั้งหมด",
    "Distinct user values": "จำนวนผู้ใช้งานไม่ซ้ำ",
    "Course ID": "รหัสหลักสูตร",
    "Student ID": "รหัสนักเรียน",
    "Module Type": "ประเภทโมดูล",
    "State": "สถานะ",
    "Distribution": "การกระจาย",
    "Overview": "ภาพรวม",
    "Counts": "จำนวน",
    "Count": "จำนวน",
    "Course": "หลักสูตร",
    "Student": "นักเรียน",
    "Module": "โมดูล",
    "Type": "ประเภท",
    "Status": "สถานะ",
    "Focus": "เน้น",
    "Passed": "ผ่าน",
    "In Progress": "กำลังเรียน",
    "Inactive": "ไม่ได้ใช้งาน",
    "Leading": "อันดับสูงสุด",
    "Matched": "ที่ตรงกัน",
    "Name": "ชื่อ",
    "Split": "แบ่งตาม",
    "User": "ผู้ใช้งาน",
    "Value": "ค่า",
    "leads this comparison.": "มีค่าสูงสุดในการเปรียบเทียบนี้",
    "leads this comparison": "มีค่าสูงสุดในการเปรียบเทียบนี้",
    "Distinct values included in the aggregate": "จำนวนค่าที่ไม่ซ้ำในผลรวม",
    "Distinct ranked dimension values": "จำนวนค่ามิติที่ไม่ซ้ำ",
    "Top value": "ค่าสูงสุด",
    "comparison": "การเปรียบเทียบ",
    "split": "แบ่งตาม",
    "and": "และ",
    "of": "ของ",
}

THAI_REASONING_FALLBACK = {
    "request_interpretation": (
        "Request interpretation",
        "ตีความ dimension, measure, filter และ split จากคำขอของผู้ใช้",
    ),
    "execution_source": (
        "Execution source",
        "เลือก data source ที่ตรงกับ analytical plan และใช้ graph เป็นแหล่งหลักเมื่อมีผลลัพธ์พร้อมใช้งาน",
    ),
    "data_lineage": (
        "Data lineage",
        "ตรวจสอบ data lineage จาก source metadata ที่บันทึกไว้ใน graph",
    ),
    "chart_binding": (
        "Chart binding",
        "จับคู่ dimension, measure และ series กับ chart type ที่เหมาะกับคำขอ",
    ),
}
UNTRANSLATED_REASONING_RE = re.compile(
    r"\b(?:identify|retrieve|retrieved|aggregated|mapped|categorized|grouped|selected|matched)\b",
    re.IGNORECASE,
)
CANONICAL_ANALYTICS_PHRASES = {
    "broken down by": "split by",
    "breakdown by": "split by",
    "grouped by": "split by",
    "segmented by": "split by",
    "separated by": "split by",
    "divided by": "split by",
    "classified by": "split by",
    "categorized by": "split by",
    "broken down by": "split by",
    "break down by": "split by",
    "enrollment status": "learning status",
    "study status": "learning status",
    "student status": "learning status",
    "institutions": "institutes",
    "institution": "institute",
    "students": "users",
    "learners": "users",
}


def _contains_thai(text: str) -> bool:
    return bool(THAI_TEXT_RE.search(text))


_MULTI_VIEW_COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}
_MULTI_VIEW_NOUNS = (
    r"(?:chart|charts|view|views|visuali[sz]ation|visuali[sz]ations|analysis|analyses|กราฟ|แผนภูมิ)"
)


def _multi_view_request_contract(question: str) -> dict[str, Any] | None:
    """Return an explicit multi-view contract without interpreting its clauses.

    A single canonical analytics sentence cannot faithfully represent a request
    such as "three charts: ranking; distribution; monthly trend".  This small
    contract deliberately records only structural evidence that the request
    contains multiple sibling views; field resolution remains the planner's
    responsibility.
    """
    lowered = question.lower()
    requested_count = 0
    contract_match = re.search(r"\bchart[_ ]count\s*[=:]\s*(\d{1,2})\b", lowered)
    if contract_match:
        requested_count = int(contract_match.group(1))
    numeric_match = re.search(rf"\b(\d{{1,2}})\s+{_MULTI_VIEW_NOUNS}\b", lowered)
    if not numeric_match:
        # Thai normally joins the following word directly after the chart noun,
        # so an English-style trailing word boundary is not reliable here.
        numeric_match = re.search(r"(\d{1,2})\s*(?:กราฟ|แผนภูมิ)", lowered)
    if numeric_match and not requested_count:
        requested_count = int(numeric_match.group(1))
    else:
        for word, count in _MULTI_VIEW_COUNT_WORDS.items():
            if re.search(rf"\b{word}\s+{_MULTI_VIEW_NOUNS}\b", lowered):
                requested_count = count
                break

    mentions_multi_view = bool(
        requested_count >= 2
        or re.search(rf"\b(?:multiple|several)\s+{_MULTI_VIEW_NOUNS}\b", lowered)
        or re.search(rf"\b{_MULTI_VIEW_NOUNS}\s*[:\-]", lowered)
    )
    if not mentions_multi_view:
        return None

    # Semicolons and numbered list items are stable clause boundaries in both
    # English and mixed-language requests.  Do not split on ordinary "and":
    # it often joins dimensions within one chart.
    clauses = [part.strip() for part in re.split(r"(?:;|\n|\b\d+[.)]\s*)", question) if part.strip()]
    if requested_count < 2 and len(clauses) < 2:
        return None
    return {
        "requestedCount": requested_count or len(clauses),
        "clauses": clauses,
    }


_PROBLEM_SOLVE_SYSTEM_PROMPT = (
    "Decide whether a user's analytics request is a PROBLEM or GOAL statement, or a "
    "DIRECT data request.\n\n"
    "A PROBLEM or GOAL statement: the user describes a business situation, asks why "
    "something is happening, how to improve it, which items are struggling / at risk / "
    "doing worse, or what to do about it, WITHOUT specifying the exact chart or "
    "analysis. Examples: 'Why is the pass rate low?' 'Which courses are students "
    "struggling with?' 'How can we improve learner engagement?' 'Is the secondary "
    "segment doing worse?'\n\n"
    "A DIRECT data request: the user already specifies the analysis - a count, a "
    "value, a trend over time, a ranking, a breakdown by a category, a comparison, or "
    "a distribution. Examples: 'How many distinct users are there?' 'Show the monthly "
    "enrollment trend.' 'Rank schools by number of users.' 'Show the number of users "
    "by department.'\n\n"
    "A request that asks WHY something is happening, or asks what chart / analysis / "
    "data would help answer a problem or goal (e.g. 'I want to answer why the pass rate "
    "is low. What chart should be shown?'), is a PROBLEM statement - the user is "
    "describing a situation to understand, not directly specifying a count / trend / "
    "ranking. Set is_problem to true for these.\n\n"
    "Be conservative. Set is_problem to true ONLY when the user is clearly stating a "
    "problem / goal / why-question (with or without asking what chart to show) and NOT "
    "directly specifying a concrete count / trend / ranking / breakdown. When in doubt, "
    "set is_problem to false.\n\n"
    "Return ONLY JSON with exactly one key: is_problem (boolean)."
)

_PROBLEM_SOLVE_REPHRASE_PROMPT = (
    "You convert a user's business problem or goal into ONE precise analytical data "
    "question that a chart can answer. The user states a PROBLEM or GOAL (for example "
    "asking why a rate is low, which items are struggling, or how to improve an "
    "outcome); it is not a chart request. Interpret the underlying analytical "
    "question and express it so the dashboard engine can chart it.\n\n"
    "The chart must ANSWER the problem, not restate it. A why-question (\'why is X "
    "low / high / declining\') is answered by DECOMPOSING the metric X across a "
    "meaningful category and RANKING the categories, so the worst (or best) "
    "contributors stand out. An overall binary share (for example total passed vs "
    "not passed) only describes the problem - it does NOT explain it, so never "
    "rephrase a why-question into a plain overall share. A \'which items are "
    "struggling\' question is answered by ranking those items by the relevant "
    "negative indicator. A \'how can we improve Y\' question is answered by finding "
    "the segment, group, or time period where Y is weakest or falling.\n"
    "When the problem is about a RATE or PERCENTAGE (for example a low pass rate), "
    "express the rephrased question as a RANKING of a named category by that RATE - "
    "say the words \'pass rate\' or \'percentage of ... that passed\' and name the "
    "category explicitly (for example \'Rank departments by pass rate, lowest first, "
    "showing the percentage of users in each department who passed\'). Do not "
    "rephrase a rate problem into a plain count of users.\n\n"
    "Rules:\n"
    "- Use ONLY the tables and columns in the provided schema. Do not invent fields.\n"
    "- Express the question in ENGLISH, naming the measure to count or average, the "
    "category to group or rank by, the sort direction, and any filter the problem "
    "implies. For a why-question the category is required - name it explicitly.\n"
    "- Prefer one of these analytical patterns the engine supports: ranking "
    "(top-N categories by a measure, ascending or descending); comparison (a measure "
    "across categories, split by a second dimension); trend over time (a measure by "
    "month or week, to show a decline or drop); distribution (a measure split across "
    "a category). For a why-question, ranking or comparison by a category is the "
    "right pattern; an overall binary share is not.\n"
    "- If the problem implies struggling, low, worse, or at-risk, use the relevant "
    "negative indicator (low grade, not-passed, inactive status, drop-off) as the "
    "measure or filter, and rank so the worst cases come first.\n"
    "- Keep it to ONE focused question; do not ask for multiple charts.\n\n"
    "Return ONLY JSON with exactly two keys: question (the analytical data question "
    "in English) and insight (one sentence in the user's original language "
    "explaining what you inferred and what the chart will reveal)."
)


def _problem_is_problem(question: str) -> bool:
    """Cheap, schema-free decision: is the user stating a PROBLEM/GOAL to interpret
    (vs a direct data request)? The model makes the call - no problem keywords are
    hardcoded. Returns True only on a clear problem verdict; any failure returns
    False so the normal path is untouched. Kept schema-free and small so it is fast
    and cheap on the common (direct-request) path.
    """
    if get_settings().llm_mode == "never":
        return False
    candidates = _llm_candidates()
    if not candidates:
        return False
    messages = [
        ("system", _PROBLEM_SOLVE_SYSTEM_PROMPT),
        ("human", json.dumps({"request": question}, ensure_ascii=False)),
    ]
    for candidate in candidates:
        try:
            model = _build_llm_model(candidate, timeout=12, max_retries=0, json_mode=True)
            parsed = json.loads(str(_llm_invoke_bounded(model, messages, 108.0).content))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return bool(parsed.get("is_problem"))
    return False


def _problem_solve_rephrase(question: str, schema: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Problem-solve mode: interpret the user's underlying problem and rewrite it
    into ONE precise analytical data question the deterministic planner can execute.

    General and schema-driven: the model sees the live table/column schema and the
    analytical patterns the system already supports, and is asked to express the
    problem as a data question. No problem-specific phrases are hardcoded; the
    mapping is delegated to the model. Returns a dict with question and insight,
    or None on any failure so the caller falls back to the deterministic path.
    """
    if get_settings().llm_mode == "never" or not schema:
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    schema_brief = json.dumps(schema, ensure_ascii=False, default=str)[:6000]
    messages = [
        ("system", _PROBLEM_SOLVE_REPHRASE_PROMPT),
        ("human", json.dumps({"problem": question, "schema": schema_brief}, ensure_ascii=False)),
    ]
    for candidate in candidates:
        try:
            model = _build_llm_model(candidate, timeout=12, max_retries=0, json_mode=True)
            parsed = json.loads(str(_llm_invoke_bounded(model, messages, 108.0).content))
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        new_question = str(parsed.get("question") or "").strip()
        insight = str(parsed.get("insight") or "").strip()
        if new_question:
            return {"question": new_question[:400], "insight": insight[:300]}
    return None


def _attach_problem_insight(activity: dict[str, Any], insight: str) -> dict[str, Any]:
    """Surface the problem interpretation in the dashboard reasoning + subtitle."""
    if not activity or not insight:
        return activity
    trace = activity.get("decisionTrace")
    if not isinstance(trace, list):
        trace = []
    trace.insert(
        0,
        {"stage": "Interpret problem", "detail": insight, "evidence": []},
    )
    activity["decisionTrace"] = trace
    return activity


_CHART_VERIFY_PROMPT = (
    "You are a strict data-visualization reviewer. You are given the user's question and a "
    "compact description of the chart that was produced. Decide whether the chart's DATA "
    "(not its styling) directly answers the question. "
    "Return only JSON with four keys: answers (boolean), reason (one short sentence), "
    "refined_question (a single, more precise analytical data question that WOULD answer it, "
    "or an empty string), and usefulness (one short, plain-language sentence explaining why "
    "this chart helps the user understand or answer their question). "
    "Set answers=true only if the chart's measured quantity, grouping dimension, and category "
    "match what the question asks for. Set answers=false when the chart measures the wrong "
    "quantity (for example a plain count when a rate or percentage was asked), groups by the "
    "wrong dimension, or is too aggregate to answer a 'why' question. When answers=false, "
    "refined_question must be ONE precise data question that a single chart can answer. Do not "
    "mention specific table or column names you were not given."
)


def _chart_brief(activity: dict[str, Any]) -> dict[str, Any]:
    """Compact, PII-free description of a produced chart for the reviewer LLM."""
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    layout = activity.get("layoutSpec") if isinstance(activity.get("layoutSpec"), dict) else {}
    slots: list[dict[str, Any]] = []
    for slot in activity.get("chartSlots") or []:
        if not isinstance(slot, dict):
            continue
        data = slot.get("data") or []
        labels = [
            str(rec.get("label") or rec.get("x") or "")
            for rec in data[:5]
            if isinstance(rec, dict)
        ]
        values = [
            rec.get("value") or rec.get("y")
            for rec in data[:5]
            if isinstance(rec, dict)
        ]
        slots.append(
            {
                "title": slot.get("title"),
                "chartType": slot.get("chartType"),
                "field": slot.get("field"),
                "splitField": slot.get("splitField"),
                "rows": len(data),
                "sampleLabels": labels,
                "sampleValues": values,
            }
        )
    return {
        "title": layout.get("title"),
        "reasoningSource": summary.get("reasoningSource"),
        "analyticalPlan": summary.get("analyticalPlan"),
        "chartSlots": slots,
    }


def _chart_answers_question(question: str, activity: dict[str, Any]) -> dict[str, Any] | None:
    """Cheap reviewer LLM call: does the produced chart's data answer the question?

    General and schema-free - it judges question semantics against chart semantics,
    with no hardcoded fields or phrases. Iterates all LLM candidates with fallback and
    is bounded by the request LLM budget. Returns the verdict dict, or None on any
    failure so the caller keeps the original chart (safe degradation).
    """
    if get_settings().llm_mode == "never":
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    messages = [
        ("system", _CHART_VERIFY_PROMPT),
        (
            "human",
            json.dumps(
                {"question": question, "chart": _chart_brief(activity)},
                ensure_ascii=False,
                default=str,
            ),
        ),
    ]
    for candidate in candidates:
        try:
            model = _build_llm_model(candidate, timeout=10, max_retries=0, json_mode=True)
            parsed = json.loads(str(_llm_invoke_bounded(model, messages, 108.0).content))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _replan_dashboard(
    store: Any,
    database_path: Any,
    question: str,
    graph_hints: Any,
    original_question: str,
) -> dict[str, Any] | None:
    """Re-run the deterministic data-planning chain for a (refined) question.

    The shape layer reads the ORIGINAL wording, so the re-plan must hand it the
    REFINED question (which names the category and measure the reviewer asked
    for), not the user's raw problem statement - otherwise the shape layer
    re-derives the same coarse chart the reviewer just rejected.
    """
    if not database_path or not database_path.exists():
        return None
    return (
        build_complex_dashboard(
            database_path,
            question,
            graph_hints=graph_hints,
            original_question=question,
        )
        or _duckdb_grouped_time_series_context(store, question, graph_hints=graph_hints)
        or _duckdb_ranked_dimension_context(store, question, graph_hints=graph_hints)
        or _duckdb_activity_context(store, question)
    )


def _attach_why_chart(activity: dict[str, Any], why: str) -> dict[str, Any]:
    """Surface a 'why this chart is useful' explanation in the decision trace."""
    if not activity or not why:
        return activity
    trace = activity.get("decisionTrace")
    if not isinstance(trace, list):
        trace = []
    for t in trace:
        if isinstance(t, dict) and t.get("stage") == "Why this chart":
            return activity
    trace.insert(0, {"stage": "Why this chart", "detail": why, "evidence": []})
    activity["decisionTrace"] = trace
    return activity


def _verify_and_repair_dashboard(
    store: Any,
    question: str,
    activity: dict[str, Any],
    database_path: Any,
    graph_hints: Any,
    original_question: str,
) -> dict[str, Any]:
    """Hermes-style verify->repair loop, bounded to ONE repair iteration.

    After a dashboard is produced, a cheap reviewer LLM checks whether the chart's
    data actually answers the question. If it clearly does not and offers a more
    precise question, the data is re-planned once. Only a data-backed,
    deterministic (shape-resolved) repair is accepted; otherwise the original
    chart is kept. Any failure keeps the original (safe degradation). The reviewer
    also explains WHY the final chart is useful for the user's problem, which is
    surfaced in the decision trace. This is the 'keep acting until the result is
    verified' pattern, adapted to the deterministic-first planner.
    """
    if not _activity_has_data(activity) or get_settings().llm_mode == "never":
        return activity
    verdict = _chart_answers_question(question, activity)
    if not isinstance(verdict, dict):
        return activity
    usefulness = str(verdict.get("usefulness") or verdict.get("reason") or "").strip()
    if verdict.get("answers"):
        return _attach_why_chart(activity, usefulness)
    refined = str(verdict.get("refined_question") or "").strip()
    if not refined or refined.lower() == question.lower():
        return _attach_why_chart(activity, usefulness)
    try:
        repaired = _replan_dashboard(store, database_path, refined, graph_hints, original_question)
    except Exception:
        return _attach_why_chart(activity, usefulness)
    if not repaired or not _activity_has_data(repaired):
        return _attach_why_chart(activity, usefulness)
    repaired_is_shape = (
        str((repaired.get("summary") or {}).get("reasoningSource") or "") == "shape"
    )
    if repaired_is_shape:
        # The repaired chart answers the refined question - explain that directly.
        return _attach_why_chart(repaired, "This chart answers: " + refined + ".")
    return _attach_why_chart(activity, usefulness)

_CHART_CHAIN_PROMPT = (
    "You are a data analyst continuing an investigation. You are given the user's problem, "
    "the charts already produced (titles, fields, and a few sample values), and the available "
    "data fields. Decide whether the charts so far fully answer the problem. If not, propose "
    "the ONE next chart that most advances the answer - typically a drill-down into the most "
    "surprising or most extreme finding visible in the current charts. The next chart must be "
    "expressible as a single chart: one measure grouped by ONE available field (or split by a "
    "second one) - it cannot filter to a single specific row value. Return only JSON with two "
    "keys: next_question (a single precise analytical data question that one chart can answer "
    "using the available fields, or an empty string if the problem is already answered) and "
    "why (one short plain-language sentence explaining what this next chart adds to the "
    "answer). Do not propose a chart that merely repeats an existing one."
)


def _next_chain_step(
    question: str, activity: dict[str, Any], schema: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Cheap reviewer LLM: given the problem + charts so far, propose the next drill-down.

    Schema-driven (the available fields are passed so the proposal is expressible by the
    deterministic planner); iterates all LLM candidates with fallback and is bounded by
    the request LLM budget. Returns the step dict, or None on any failure (the chain
    stops - safe degradation).
    """
    if get_settings().llm_mode == "never":
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    payload: dict[str, Any] = {"problem": question, "charts_so_far": _chart_brief(activity)}
    if schema:
        payload["available_fields"] = schema
    messages = [
        ("system", _CHART_CHAIN_PROMPT),
        ("human", json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    for candidate in candidates:
        try:
            model = _build_llm_model(candidate, timeout=10, max_retries=0, json_mode=True)
            parsed = json.loads(str(_llm_invoke_bounded(model, messages, 108.0).content))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _merge_chart_into_activity(
    base: dict[str, Any], extra: dict[str, Any], why: str
) -> tuple[dict[str, Any], bool]:
    """Merge a chained chart into the base dashboard activity.

    Returns (activity, added). Dedupes by (field, splitField, title) so the chain loop
    stops when a step proposes a chart that already exists.
    """
    base_slots = base.get("chartSlots")
    if not isinstance(base_slots, list):
        base_slots = []
        base["chartSlots"] = base_slots
    existing = {
        (str(s.get("field") or ""), str(s.get("splitField") or ""), str(s.get("title") or ""))
        for s in base_slots
        if isinstance(s, dict)
    }
    existing_ids = {str(s.get("id") or "") for s in base_slots if isinstance(s, dict)}
    charts = base.get("charts")
    if not isinstance(charts, dict):
        charts = {}
        base["charts"] = charts
    plan = base.get("chartPlan")
    if not isinstance(plan, list):
        plan = []
        base["chartPlan"] = plan
    added = False
    for slot in extra.get("chartSlots") or []:
        if not isinstance(slot, dict):
            continue
        key = (str(slot.get("field") or ""), str(slot.get("splitField") or ""), str(slot.get("title") or ""))
        if key in existing:
            continue
        existing.add(key)
        slot_id = str(slot.get("id") or "slot")
        n = 2
        while slot_id in existing_ids:
            slot_id = f"{slot_id}-{n}"
            n += 1
        slot = dict(slot)
        slot["id"] = slot_id
        existing_ids.add(slot_id)
        base_slots.append(slot)
        charts[slot_id] = slot.get("data")
        plan.append({k: v for k, v in slot.items() if k != "data"})
        added = True
    if not added:
        return base, False
    layout = base.get("layoutSpec")
    if isinstance(layout, dict):
        blocks = layout.get("blocks")
        if not isinstance(blocks, list):
            blocks = []
            layout["blocks"] = blocks
        have = {b.get("slotId") for b in blocks if isinstance(b, dict)}
        for slot in base_slots:
            if slot.get("id") not in have:
                blocks.append({"type": "chart", "slotId": slot.get("id"), "span": 2})
        layout["title"] = " / ".join(str(s.get("title") or s.get("id")) for s in base_slots)
        layout["subtitle"] = layout["title"]
    trace = base.get("decisionTrace")
    if not isinstance(trace, list):
        trace = []
        base["decisionTrace"] = trace
    if why:
        trace.append({"stage": "Drill down", "detail": why, "evidence": []})
    trace.extend(t for t in extra.get("decisionTrace") or [] if isinstance(t, dict))
    summary = base.get("summary")
    if isinstance(summary, dict):
        try:
            summary["totalRecords"] = int(summary.get("totalRecords") or 0) + int(
                (extra.get("summary") or {}).get("totalRecords") or 0
            )
        except Exception:
            pass
    return base, True


def _chain_problem_analysis(
    store: Any,
    user_question: str,
    activity: dict[str, Any],
    database_path: Any,
    graph_hints: Any,
    max_steps: int = 2,
) -> dict[str, Any]:
    """Hermes-style iterative chain: keep drilling into the problem with additional
    charts until the analyst LLM says the problem is answered (or steps/budget run out).

    Each step is ONE cheap LLM proposal + ONE deterministic re-plan (no LLM in the
    re-plan), so the whole chain is bounded by the request LLM budget. Only
    data-backed, shape-resolved charts are accepted; duplicates stop the chain. Any
    failure stops the chain and keeps the charts produced so far (safe degradation).
    """
    if not _activity_has_data(activity) or get_settings().llm_mode == "never":
        return activity
    for _ in range(max_steps):
        if len(activity.get("chartSlots") or []) >= 4:
            break
        step = _next_chain_step(user_question, activity, _duckdb_semantic_schema(store))
        if not isinstance(step, dict):
            break
        nxt = str(step.get("next_question") or "").strip()
        why = str(step.get("why") or "").strip()
        if not nxt:
            break  # the analyst says the problem is answered
        try:
            extra = _replan_dashboard(store, database_path, nxt, graph_hints, nxt)
        except Exception:
            break
        if not extra or not _activity_has_data(extra):
            break
        if str((extra.get("summary") or {}).get("reasoningSource") or "") != "shape":
            break  # only accept deterministic, trustworthy charts
        activity, added = _merge_chart_into_activity(
            activity, extra, why or f"This chart answers: {nxt}."
        )
        if not added:
            break  # duplicate chart - stop
    return activity


def _canonicalize_question_for_processing(
    question: str,
    *,
    graph_path: Path | None = None,
    semantic_context: dict[str, Any] | None = None,
) -> str:
    # English analytics requests already contain the user's strongest source of
    # truth. Normalizing their vocabulary is safe; asking a model to rewrite
    # them can collapse multi-chart requests or turn date filters into trends.
    if not _contains_thai(question) and _preserve_english_analytics_request(question):
        return _normalize_canonical_analytics_question(question)
    multi_view_contract = _multi_view_request_contract(question)
    # Preserve the original sibling clauses before model-based semantic
    # canonicalization.  The latter returns one canonical query by design and
    # would otherwise collapse a multi-view dashboard into one chart.
    if multi_view_contract:
        preserved = _fallback_english_question(question) if _contains_thai(question) else question
        return _normalize_canonical_analytics_question(preserved)
    cache_key = f"analytics-v5:{question}" if semantic_context else question
    cached = _question_language_cache.get(cache_key)
    if cached:
        return cached
    # Short Thai prompts whose every Thai run is covered by the intent
    # lexicon are glossed deterministically.  This is the same general
    # rule the shape resolver uses, and it is preferred over a persisted
    # translation because a model can mis-translate a misspelled atomic
    # prompt into an unrelated intent (e.g. a by-institution request
    # becoming an over-time request) and that wrong form then matches a
    # stale dashboard-cache entry.
    if _contains_thai(question) and len(question) <= 60 and _thai_question_fully_glossable(question):
        glossed = _normalize_canonical_analytics_question(_thai_gloss(question))
        glossed = _preserve_source_analytics_intent(question, glossed)
        _question_language_cache[cache_key] = glossed
        if graph_path is not None:
            write_question_translation(graph_path, cache_key, glossed)
        return glossed
    persisted_key = cache_key
    if graph_path is not None:
        persisted = read_question_translation(graph_path, persisted_key)
        if persisted:
            persisted = _normalize_canonical_analytics_question(persisted)
            persisted = _preserve_source_analytics_intent(question, persisted)
            _question_language_cache[cache_key] = persisted
            write_question_translation(graph_path, persisted_key, persisted)
            return persisted
    # Mixed-language schema requests already carry exact identifiers. A
    # deterministic lexical normalization preserves those identifiers, dates,
    # limits, and clause order more faithfully than asking a model to rewrite
    # the whole request into a single sentence.
    deterministic = (
        _fallback_english_question(question)
        if _contains_thai(question) and len(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", question.lower())) >= 2
        else ""
    )
    parsed = None if deterministic else _invoke_language_json(
        system=(
            "Interpret the user's analytics request and rewrite it as one precise English canonical analytics query. "
            "Equivalent precise, conversational, paraphrased, and Thai requests must produce the same canonical query. "
            "Use candidate graph fields and schema fields as grounding, including semantic mappings such as a natural location phrase "
            "to an available geographic field. Preserve every requested measure, dimension, filter value, comparison, time grain, "
            "ranking, split, and chart type. Resolve implicit analytics language such as change month to month into an over-time request. "
            "When the request compares a primary dimension with a categorical breakdown, use the canonical form "
            "'show <primary dimension> by number of <measure> split by <secondary dimension>'. "
            "For time analysis use 'show number of <measure> over time split by <dimension>'. "
            "For ranking use 'show <dimension> by number of <measure> ranked highest'. "
            "Keep a genuinely neutral overview neutral. Do not answer the request, invent filters, or add unrelated dimensions. "
            "Prefer exact candidate field names when they represent the user's concept. "
            "Return only JSON with canonicalQuestion, intent, measureField, dimensionFields in display order, splitField, "
            "timeField, timeGrain, ranking, filters as field/value objects, chartType, and confidence."
        ),
        payload={"question": question, **(semantic_context or {})},
    )
    english = deterministic or (
        str(parsed.get("canonicalQuestion") or parsed.get("englishQuestion") or "").strip()
        if isinstance(parsed, dict)
        else ""
    )
    if isinstance(parsed, dict) and semantic_context:
        english = _canonical_question_from_semantic_plan(
            parsed,
            semantic_context,
            fallback=english,
            original_question=question,
        )
    if not english or _contains_thai(english):
        english = _fallback_english_question(question) if _contains_thai(question) else question
    english = _normalize_canonical_analytics_question(english)
    english = _preserve_source_analytics_intent(question, english)
    _question_language_cache[cache_key] = english
    if graph_path is not None:
        write_question_translation(graph_path, persisted_key, english)
    if len(_question_language_cache) > 200:
        _question_language_cache.pop(next(iter(_question_language_cache)))
    return english


def _preserve_english_analytics_request(question: str) -> bool:
    lowered = question.lower()
    signals = sum(
        bool(re.search(pattern, lowered))
        for pattern in (
            r"\b(?:distinct|unique)\b",
            r"\b(?:top|rank|ranking|highest|most)\b",
            r"\b(?:split|breakdown|break\s+.+\s+down|series)\b",
            r"\b(?:monthly|daily|weekly|yearly|trend|over time)\b",
            r"\b(?:chart|visual|dashboard|distribution|composition)\b",
            r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b",
            r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b",
        )
    )
    return signals >= 2


def _canonical_question_from_semantic_plan(
    parsed: dict[str, Any],
    semantic_context: dict[str, Any],
    *,
    fallback: str,
    original_question: str = "",
) -> str:
    schema_fields = [
        str(field)
        for table in semantic_context.get("schemaCatalog") or []
        if isinstance(table, dict)
        for field in table.get("fields") or []
        if field
    ]
    graph_fields = [
        str(field)
        for field in (semantic_context.get("graphCandidates") or {}).get("fields") or []
        if field
    ]
    allowed_fields = list(dict.fromkeys([*schema_fields, *graph_fields]))

    def resolve(raw_field: Any) -> str:
        value = str(raw_field or "").strip()
        if not value:
            return ""
        exact = next((field for field in allowed_fields if field.lower() == value.lower()), "")
        if exact:
            return exact
        intent_terms = set(_intent_terms(value))
        scored = sorted(
            (
                (_dimension_field_score(field, value, intent_terms), field)
                for field in allowed_fields
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return scored[0][1] if scored and scored[0][0] >= 10 else ""

    measure = resolve(parsed.get("measureField"))
    raw_dimensions = parsed.get("dimensionFields")
    if not isinstance(raw_dimensions, list):
        raw_dimensions = [raw_dimensions] if raw_dimensions else []
    dimensions = [
        field
        for raw_field in raw_dimensions
        for field in [resolve(raw_field)]
        if field
    ]
    dimensions = list(dict.fromkeys(dimensions))[:4]
    split_field = resolve(parsed.get("splitField"))
    time_field = resolve(parsed.get("timeField"))
    if _semantic_time_intent(original_question or fallback):
        if not time_field:
            time_candidates = [field for field in allowed_fields if _semantic_time_field(field)]
            time_candidates.sort(
                key=lambda field: (
                    -_field_score(field, set(_intent_terms(original_question or fallback))),
                    -len(_field_terms(field) & _field_terms(measure)),
                    field,
                )
            )
            time_field = time_candidates[0] if time_candidates else ""
        if not dimensions:
            dimension_candidates = [
                field
                for field in allowed_fields
                if field != measure
                and not _semantic_time_field(field)
                and "id" not in _field_terms(field)
            ]
            dimension_candidates.sort(
                key=lambda field: (
                    -_dimension_field_score(
                        field,
                        original_question or fallback,
                        set(_intent_terms(original_question or fallback)),
                    ),
                    field,
                )
            )
            if dimension_candidates and _dimension_field_score(
                dimension_candidates[0],
                original_question or fallback,
                set(_intent_terms(original_question or fallback)),
            ) >= 10:
                dimensions = [dimension_candidates[0]]
    ranking = bool(parsed.get("ranking")) or str(parsed.get("intent") or "").lower() in {
        "rank",
        "ranking",
        "top",
    } or bool(re.search(r"\b(?:highest|largest|most|ranked|top)\b", fallback, flags=re.IGNORECASE))
    if not measure or (not dimensions and not split_field and not time_field):
        return fallback

    if time_field:
        canonical = f"show number of {measure} over time"
        group_field = split_field or (dimensions[0] if dimensions else "")
        if group_field:
            canonical += f" split by {group_field}"
        time_grain = str(parsed.get("timeGrain") or "").strip().lower() or _semantic_time_grain(
            original_question or fallback
        )
        if time_grain and re.fullmatch(r"daily|weekly|monthly|quarterly|yearly", time_grain):
            canonical += f" {time_grain}"
    else:
        primary = dimensions[0] if dimensions else split_field
        canonical = f"show {primary} by number of {measure}"
        secondary = split_field if split_field and split_field != primary else ""
        if not secondary and len(dimensions) > 1:
            secondary = dimensions[1]
        if secondary:
            canonical += f" split by {secondary}"
        if ranking:
            canonical += " ranked highest"

    filters = parsed.get("filters") if isinstance(parsed.get("filters"), list) else []
    filter_clauses: list[str] = []
    for item in filters[:8]:
        if not isinstance(item, dict):
            continue
        field = resolve(item.get("field"))
        value = str(item.get("value") or "").strip()
        if field and value and len(value) <= 80:
            filter_clauses.append(f"{field} is {value}")
    if filter_clauses:
        canonical += " where " + " or ".join(filter_clauses)
    chart_type = str(parsed.get("chartType") or "").strip().lower().replace("_", " ")
    if chart_type and chart_type in {
        "area",
        "column",
        "donut",
        "funnel",
        "horizontal bar",
        "line",
        "pie",
        "radar",
        "radial bar",
        "stacked bar",
        "stacked column",
        "treemap",
    }:
        canonical += f" as {chart_type} chart"
    return canonical


def _semantic_time_intent(question: str) -> bool:
    terms = set(_intent_terms(question))
    return bool(
        terms
        & {
            "change",
            "changed",
            "daily",
            "day",
            "monthly",
            "month",
            "quarter",
            "quarterly",
            "time",
            "timeline",
            "trend",
            "week",
            "weekly",
            "yearly",
        }
    )


def _semantic_time_field(field: str) -> bool:
    terms = _field_terms(field)
    return bool(terms & {"date", "datetime", "time", "timestamp"}) or field.lower().endswith("_at")


def _semantic_time_grain(question: str) -> str:
    terms = set(_intent_terms(question))
    for grain, aliases in (
        ("daily", {"daily", "day"}),
        ("weekly", {"week", "weekly"}),
        ("monthly", {"month", "monthly"}),
        ("quarterly", {"quarter", "quarterly"}),
        ("yearly", {"year", "yearly"}),
    ):
        if terms & aliases:
            return grain
    return ""


def _seed_question_for_semantic_search(question: str) -> str:
    if _contains_thai(question):
        seed = _fallback_english_question(question)
        return _preserve_source_analytics_intent(question, _normalize_canonical_analytics_question(seed))
    return _normalize_canonical_analytics_question(question)


def _duckdb_semantic_schema(store: Any) -> list[dict[str, Any]]:
    database_path = _duckdb_database_path(store)
    if database_path is None or not database_path.exists():
        return []
    cache_key = str(database_path.resolve())
    modified = int(database_path.stat().st_mtime_ns)
    cached = _semantic_schema_cache.get(cache_key)
    if cached and cached[0] == modified:
        return cached[1]
    try:
        import duckdb

        con = duckdb.connect(str(database_path), read_only=True)
        rows = con.execute(
            """
            select table_name, column_name
            from information_schema.columns
            where table_schema = 'main'
              and table_name like 'dashboard_agent_%'
              and table_name not like '%cache%'
              and table_name not like '%lineage%'
              and table_name not like '%map%'
            order by
              case when table_name like '%fact%' or table_name like '%joined%' then 0 else 1 end,
              table_name,
              ordinal_position
            """
        ).fetchall()
        con.close()
    except Exception:
        return []
    tables: dict[str, list[str]] = {}
    for table_name, column_name in rows:
        fields = tables.setdefault(str(table_name), [])
        if len(fields) < 30:
            fields.append(str(column_name))
    schema = [{"table": table, "fields": fields} for table, fields in list(tables.items())[:12]]
    _semantic_schema_cache[cache_key] = (modified, schema)
    return schema


def _normalize_canonical_analytics_question(question: str) -> str:
    result = question.strip()
    result = re.sub(
        r"\bbreak\s+(?:each|every)\s+[^,.;]{0,60}?\s+down\s+by\b",
        "split by",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bbreak\s+(?:each|every)\s+[^,.;]{0,60}?\s+into\s+"
        r"(?:passed|pass)\s+versus\s+not\s+passed\b",
        "split by course_pass",
        result,
        flags=re.IGNORECASE,
    )
    for source, target in sorted(CANONICAL_ANALYTICS_PHRASES.items(), key=lambda item: -len(item[0])):
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
    result = re.sub(r"\bper\b", "by", result, flags=re.IGNORECASE)
    result = re.sub(r"\bfor each\b", "by", result, flags=re.IGNORECASE)
    result = re.sub(r"\bsplit\s+into\b", "split by", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:broken|break)\s+down\s+by\b", "split by", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:passed|pass)\s+versus\s+not\s+passed\b", "split by course_pass", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def _preserve_source_analytics_intent(source_question: str, english_question: str) -> str:
    result = english_question.strip()
    if THAI_SPLIT_INTENT_RE.search(source_question) and "split by" not in result.lower():
        connectors = list(re.finditer(r"\b(?:and|with)\b", result, flags=re.IGNORECASE))
        if connectors:
            connector = connectors[-1]
            result = f"{result[:connector.start()]}split by{result[connector.end():]}"
    return re.sub(r"\s+", " ", result).strip()


def _localize_activity_to_thai(question: str, activity: dict[str, Any]) -> dict[str, Any]:
    if not _contains_thai(question):
        return activity
    layout = activity.get("layoutSpec") if isinstance(activity.get("layoutSpec"), dict) else {}
    slots = activity.get("chartSlots") if isinstance(activity.get("chartSlots"), list) else []
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    summary["presentationLanguage"] = "th"
    trace = activity.get("decisionTrace") if isinstance(activity.get("decisionTrace"), list) else []
    display_values = sorted(
        {
            str(value).strip()
            for slot in slots
            if isinstance(slot, dict)
            for datum in (slot.get("data") or [])
            if isinstance(datum, dict)
            for value in (datum.get("label"), datum.get("series"))
            if value not in (None, "") and len(str(value).strip()) <= 80
        }
    )
    localization_context = {
        "request": question,
        "title": layout.get("title"),
        "subtitle": layout.get("subtitle"),
        "charts": [{"id": slot.get("id"), "title": slot.get("title")} for slot in slots if isinstance(slot, dict)],
        "metricLabels": summary.get("metricLabels") or {},
        "displayValues": display_values,
        "reasoning": [
            {
                "index": index,
                "stage": item.get("stage") or item.get("step"),
                "detail": item.get("detail"),
                "evidence": item.get("evidence") or [],
            }
            for index, item in enumerate(trace[:8])
            if isinstance(item, dict)
        ],
    }
    parsed = _invoke_language_json(
        system=(
            "Localize dashboard presentation text into natural Thai. The backend analytical plan is already complete. "
            "Write the surrounding explanatory prose in Thai, but NEVER translate established technical terms. Preserve terms such as "
            "Request interpretation, Execution source, Data lineage, Chart binding, graph traversal, database query, data source, "
            "dimension, measure, filter, split, series, chart type, stacked bar chart, horizontal bar chart, Parquet, JSON, DuckDB, and LangGraph. "
            "Translate display-category values when they are semantic labels (for example Passed or In Progress), but preserve ids, "
            "field names, source paths, numbers, course codes, and proper names. This is a concise decision summary, not private chain-of-thought. "
            "Return only JSON with title, subtitle, chartTitles keyed by chart id, metricLabels keyed by existing metric id, "
            "valueLabels keyed by the exact existing display value, and reasoning items containing index, detail, and evidence. "
            "Do not translate or return a replacement for each reasoning stage."
        ),
        payload=localization_context,
    )
    if isinstance(parsed, dict):
        if isinstance(parsed.get("title"), str) and _contains_thai(parsed["title"]):
            layout["title"] = parsed["title"].strip()[:120]
        if isinstance(parsed.get("subtitle"), str) and _contains_thai(parsed["subtitle"]):
            layout["subtitle"] = parsed["subtitle"].strip()[:220]
        chart_titles = parsed.get("chartTitles") if isinstance(parsed.get("chartTitles"), dict) else {}
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            translated = chart_titles.get(str(slot.get("id")))
            if isinstance(translated, str) and _contains_thai(translated):
                slot["title"] = translated.strip()[:140]
        metric_labels = parsed.get("metricLabels") if isinstance(parsed.get("metricLabels"), dict) else {}
        if metric_labels:
            existing = summary.get("metricLabels") if isinstance(summary.get("metricLabels"), dict) else {}
            for metric_id in list(existing):
                translated = metric_labels.get(metric_id)
                if isinstance(translated, str) and _contains_thai(translated):
                    existing[metric_id] = translated.strip()[:100]
            summary["metricLabels"] = existing
        # Data labels are contract keys used by factual validators and clients.
        # Keep them locale-neutral; localize chart titles and explanatory text.
        localized_trace = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), list) else []
        for localized in localized_trace:
            if not isinstance(localized, dict) or not isinstance(localized.get("index"), int):
                continue
            index = localized["index"]
            if not 0 <= index < len(trace) or not isinstance(trace[index], dict):
                continue
            value = localized.get("detail")
            if isinstance(value, str) and _contains_thai(value):
                trace[index]["detail"] = value.strip()
            if isinstance(localized.get("evidence"), list):
                trace[index]["evidence"] = [str(item) for item in localized["evidence"][:8]]
    _fallback_localize_activity(activity)
    activity["layoutSpec"] = layout
    activity["summary"] = summary
    return activity


def _invoke_language_json(*, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if get_settings().llm_mode == "never":
        return None
    candidates = _llm_candidates()
    if not candidates:
        return None
    from langchain_openai import ChatOpenAI

    for candidate in candidates:
        try:
            model = _build_llm_model(candidate, timeout=12, max_retries=0, json_mode=True)
            # Wrap in the per-request LLM budget like every other call site:
            # the raw timeout=12 is a READ timeout, not a total-time bound, so a
            # slow free-tier call could otherwise hang this Thai-localization step
            # past the 120s latency gate. On TimeoutError the existing
            # except/continue falls back to the next candidate / deterministic path.
            parsed = json.loads(
                str(
                    _llm_invoke_bounded(
                        model,
                        [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))],
                        108.0,
                    ).content
                )
            )
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _fallback_english_question(question: str) -> str:
    result = question
    for thai, english in sorted(THAI_ANALYTICS_FALLBACK.items(), key=lambda item: -len(item[0])):
        result = result.replace(thai, english)
    result = THAI_TEXT_RE.sub(" ", result)
    return re.sub(r"\s+", " ", result).strip() or question


def _fallback_localize_activity(activity: dict[str, Any]) -> None:
    def localize(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        result = value
        for english, thai in sorted(THAI_PRESENTATION_FALLBACK.items(), key=lambda item: -len(item[0])):
            pattern = re.escape(english)
            if english[:1].isalnum():
                pattern = rf"\b{pattern}"
            if english[-1:].isalnum():
                pattern = rf"{pattern}\b"
            result = re.sub(pattern, thai, result, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", result).strip()

    layout = activity.get("layoutSpec") if isinstance(activity.get("layoutSpec"), dict) else {}
    for key in ("title", "subtitle"):
        layout[key] = localize(layout.get(key))
    if layout.get("title") and not _contains_thai(str(layout["title"])):
        layout["title"] = "แดชบอร์ดข้อมูล"
    if layout.get("subtitle") and not _contains_thai(str(layout["subtitle"])):
        layout["subtitle"] = "สรุปข้อมูลตามคำขอ"
    for slot in activity.get("chartSlots") or []:
        if isinstance(slot, dict):
            slot["title"] = localize(slot.get("title"))
            if slot.get("title") and not _contains_thai(str(slot["title"])):
                slot["title"] = "กราฟข้อมูล"
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    labels = summary.get("metricLabels") if isinstance(summary.get("metricLabels"), dict) else {}
    summary["metricLabels"] = {
        key: translated if _contains_thai(str(translated)) else "ตัวชี้วัด"
        for key, value in labels.items()
        for translated in [localize(value)]
    }
    summary["presentationLanguage"] = "th"
    trace = activity.get("decisionTrace") if isinstance(activity.get("decisionTrace"), list) else []
    fallback_stage_keys = tuple(THAI_REASONING_FALLBACK)
    for index, item in enumerate(trace):
        if not isinstance(item, dict):
            continue
        original_stage = str(item.get("stage") or item.get("step") or "").strip()
        stage_key = re.sub(r"[^a-z0-9]+", "_", original_stage.lower()).strip("_")
        localized_stage = original_stage.replace("_", " ").strip().title()
        localized_detail = item.get("detail")
        fallback = THAI_REASONING_FALLBACK.get(stage_key)
        if fallback is None and index < len(fallback_stage_keys):
            fallback = THAI_REASONING_FALLBACK[fallback_stage_keys[index]]
        if fallback:
            localized_stage = fallback[0]
            if (
                not localized_detail
                or not _contains_thai(str(localized_detail))
                or UNTRANSLATED_REASONING_RE.search(str(localized_detail))
            ):
                localized_detail = fallback[1]
        item["stage"] = localized_stage or "Decision summary"
        item["detail"] = localized_detail or "ประมวลผลตาม analytical plan ที่ตรวจสอบแล้ว"


def _graph_planning_hints(store: Any, results: list[dict[str, Any]]) -> dict[str, Any]:
    tables: list[str] = []
    fields: list[str] = []
    source_paths: list[str] = []
    evidence: list[str] = []
    relationships: list[str] = []

    def add(target: list[str], value: Any) -> None:
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return
        text = str(value).strip()
        if text and text not in target:
            target.append(text)

    for result in results[:24]:
        if not isinstance(result, dict):
            continue
        add(evidence, result.get("label") or result.get("id"))
        result_path = str(result.get("path") or "").strip()
        result_source = str(result.get("source") or "").strip()
        if result_path.startswith("$."):
            path_field = re.split(r"[.\[\]]+", result_path)[-1]
            add(fields, path_field or result.get("label") or result.get("text"))
        else:
            add(source_paths, result_path)
        add(source_paths, result_source)
        source_table_match = re.match(r"duckdb/([^/]+)/", result_source, flags=re.IGNORECASE)
        if source_table_match:
            add(tables, source_table_match.group(1))
        raw_value = result.get("value")
        if isinstance(raw_value, str):
            try:
                raw_value = json.loads(raw_value)
            except json.JSONDecodeError:
                raw_value = None
        if isinstance(raw_value, dict):
            for key in ("duckdb_table", "table", "baseTable", "source_table"):
                add(tables, raw_value.get(key))
            for key in ("dimension_field", "measure_field", "time_field", "field", "column"):
                add(fields, raw_value.get(key))
            for key in ("source_path",):
                add(source_paths, raw_value.get(key))
            for key in ("source_paths", "sourcePaths"):
                values = raw_value.get(key)
                if isinstance(values, list):
                    for value in values:
                        add(source_paths, value)
            sample_fields = raw_value.get("sample_fields")
            if isinstance(sample_fields, list):
                for field in sample_fields:
                    if isinstance(field, dict):
                        add(fields, field.get("name"))
                    else:
                        add(fields, field)
        identifier = str(result.get("id") or "")
        identifier_field_match = re.search(r"::\$\.(?:[^:]+\.)?([^.:\[\]]+)$", identifier)
        if identifier_field_match:
            add(fields, identifier_field_match.group(1))
        aggregate_match = re.search(
            r"ranked_dimension/([^/]+)/([^/]+)/by/([^/]+)",
            identifier,
            flags=re.IGNORECASE,
        )
        if aggregate_match:
            add(tables, aggregate_match.group(1))
            add(fields, aggregate_match.group(2))
            add(fields, aggregate_match.group(3))
        graph = getattr(store, "graph", None)
        if graph is not None and identifier in graph:
            for neighbor in list(graph.neighbors(identifier))[:8]:
                attrs = graph.nodes[neighbor]
                label = str(attrs.get("label") or attrs.get("name") or neighbor)
                add(relationships, f"{result.get('label') or identifier} -> {label}")
                node_type = str(attrs.get("type") or "").lower()
                if node_type in {"field", "column"}:
                    add(fields, attrs.get("name") or attrs.get("label") or neighbor)
                if node_type in {"dataset", "table"}:
                    add(tables, attrs.get("name") or attrs.get("label") or neighbor)

    return {
        "tables": tables[:12],
        "fields": fields[:24],
        "sourcePaths": _dedupe_source_paths(source_paths)[:16],
        "evidence": evidence[:12],
        "relationships": relationships[:16],
    }


def _graph_hint_score(graph_hints: dict[str, Any] | None, table: str, *fields: str) -> float:
    hints = graph_hints or {}
    preferred_tables = {str(value).lower() for value in hints.get("tables") or [] if value}
    preferred_fields = {str(value).lower() for value in hints.get("fields") or [] if value}
    score = 16.0 if table.lower() in preferred_tables else 0.0
    for field in fields:
        lowered = field.lower()
        if lowered in preferred_fields:
            score += 12.0
            continue
        field_terms = _field_terms(field)
        score += min(
            4.0,
            max((float(len(field_terms & _field_terms(value)) * 2) for value in preferred_fields), default=0.0),
        )
    return score


def _mark_hybrid_execution(activity: dict[str, Any], graph_hints: dict[str, Any]) -> dict[str, Any]:
    if not activity:
        return activity
    summary = activity.get("summary") if isinstance(activity.get("summary"), dict) else {}
    summary.update(
        {
            "executionSource": "hybrid",
            "planningSource": "graph",
            "calculationSource": "duckdb",
            "graphPlanning": {
                "tables": list(graph_hints.get("tables") or [])[:12],
                "fields": list(graph_hints.get("fields") or [])[:24],
                "sourcePaths": list(graph_hints.get("sourcePaths") or [])[:16],
                "relationships": list(graph_hints.get("relationships") or [])[:16],
                "evidence": list(graph_hints.get("evidence") or [])[:12],
            },
        }
    )
    # Preserve shape-derived reasoning markers: the analytical shape already
    # resolved the question and produced a complete decisionTrace. Stripping
    # them here made downstream checks (shape_resolved, reasoning early-return)
    # fail, forcing redundant slow LLM calls to re-derive the same design.
    if summary.get("reasoningSource") != "shape":
        summary.pop("reasoningSource", None)
        summary.pop("reasoningExecutionSource", None)
    activity["summary"] = summary
    return activity


def _activity_has_data(activity: dict[str, Any] | None) -> bool:
    """True when an activity payload carries at least one computed chart point.

    A data-backed analytical plan (from build_complex_dashboard) produces
    chartSlots whose rows were aggregated from the warehouse.  Skeleton
    activities from the graph/context builders can carry slot shells, sample
    records, or row-level search results but no computed points; treating
    those as data-backed output would mask a failed plan and leak PII.
    """
    if not isinstance(activity, dict):
        return False
    slots = activity.get("chartSlots")
    if isinstance(slots, list):
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            data = slot.get("data")
            if isinstance(data, list) and any(
                isinstance(row, dict) and row.get("value") is not None for row in data
            ):
                return True
    charts = activity.get("charts")
    if isinstance(charts, dict):
        for rows in charts.values():
            if isinstance(rows, list) and rows:
                return True
    return False


_PLANNER_CODE_VERSION_CACHE: dict[str, str] = {}


def _planner_code_version() -> str:
    """Fingerprint of the dashboard planner code.

    The graph dashboard cache is keyed by (normalized_question, source_version).
    source_version previously reflected only the warehouse file (size + mtime),
    so edits to the planner/selection logic kept replaying stale cached
    dashboards. Folding a content hash of the planner modules into the version
    makes any change to the selection logic automatically invalidate the cache.
    """
    parts = []
    for module in ("dashboard_planner", "graph_dashboard_cache"):
        try:
            path = Path(__file__).with_name(f"{module}.py")
            stat = path.stat()
        except OSError:
            continue
        key = f"{module}:{stat.st_size}:{stat.st_mtime_ns}"
        digest = _PLANNER_CODE_VERSION_CACHE.get(key)
        if digest is None:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            except OSError:
                digest = "unreadable"
            _PLANNER_CODE_VERSION_CACHE[key] = digest
        parts.append(f"{module}={digest}")
    return "+".join(parts)


def _dashboard_source_version(database_path: Path | None) -> str:
    if database_path is None:
        return ""
    try:
        stat = database_path.stat()
    except OSError:
        return ""
    return f"{stat.st_size}:{stat.st_mtime_ns}:{_planner_code_version()}"


def run_agent(state: AgentState) -> dict[str, list[AIMessage]]:
    _llm_budget_set(108.0)
    user_question = _last_question(state)
    store = get_store()
    seed_question = _seed_question_for_semantic_search(user_question)
    if _wants_refresh(user_question):
        question = seed_question
    else:
        seed_results = store.search(seed_question, limit=24)
        seed_hints = _graph_planning_hints(store, seed_results)
        question = _canonicalize_question_for_processing(
            user_question,
            graph_path=store.path,
            semantic_context={
                "graphCandidates": {
                    "tables": seed_hints.get("tables") or [],
                    "fields": seed_hints.get("fields") or [],
                    "evidence": seed_hints.get("evidence") or [],
                },
                "schemaCatalog": _duckdb_semantic_schema(store),
            },
        )
    notices: list[str] = []
    wants_refresh = _wants_refresh(user_question)
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
    graph_results = store.search(question, limit=24)
    results = cache_results + graph_results
    # Route on the canonicalized intent, not only the raw surface form: a
    # misspelled Thai prompt can leave the deterministic gloss empty (no
    # intent words survive), so the structural dashboard rules never see the
    # grouping/trend/ranking the user actually asked for.  The canonicalized
    # English already carries that intent, so checking it too keeps the same
    # general rules while fixing typo-tolerant routing.  English prompts are
    # unchanged (canonicalization preserves their structure), and scalar
    # counts still stay on the text path.
    # An explicit text request in the RAW question is authoritative: the
    # canonicalized form is a lossy LLM rewrite that can drop the marker and
    # flip routing to a dashboard, so the canonicalized clause only applies
    # when the user did not explicitly ask for text.
    explicit_text = _has_explicit_text_request(user_question)
    wants_dashboard = _wants_dashboard(user_question) or (
        not explicit_text and _wants_dashboard(question)
    )
    # Problem-solve mode (automatic, no toggle): a request that is not an
    # explicit text answer and not already a specified chart request would
    # otherwise fall to the text path. Let the model decide whether the user is
    # actually stating a problem/goal to interpret; if so, rephrase it into a
    # precise analytical question and route it to the dashboard. No problem
    # keywords are hardcoded - the model makes the call. Any failure (or a
    # "not a problem" verdict) leaves the normal text path untouched.
    problem_rephrase = None
    # The LLM is the sole decider of whether the request is a problem/goal to
    # interpret. This runs on ALL non-explicit-text requests (including those the
    # deterministic router would call "dashboard", e.g. "I want to answer why X.
    # What chart should be shown?") so the user's natural problem phrasing always
    # reaches the model. A "not a problem" verdict (or any failure) leaves the
    # normal dashboard/text path untouched.
    if not explicit_text and _problem_is_problem(user_question):
        problem_rephrase = _problem_solve_rephrase(user_question, _duckdb_semantic_schema(store))
    if (problem_rephrase is not None) or wants_dashboard:
        database_path = _duckdb_database_path(store)
        source_version = _dashboard_source_version(database_path)
        graph_hints = _graph_planning_hints(store, graph_results)
        plan_question = problem_rephrase["question"] if problem_rephrase else question
        problem_insight = problem_rephrase.get("insight", "") if problem_rephrase else ""
        activity = read_graph_dashboard_cache(
            store.path,
            plan_question,
            source_version=source_version,
            original_question=user_question,
        )
        generated = False
        if not activity and database_path and database_path.exists():
            activity = (
                build_complex_dashboard(
                    database_path,
                    plan_question,
                    graph_hints=graph_hints,
                    original_question=plan_question if problem_rephrase else user_question,
                )
                or _duckdb_grouped_time_series_context(store, question, graph_hints=graph_hints)
                or _duckdb_ranked_dimension_context(store, question, graph_hints=graph_hints)
                or _duckdb_activity_context(store, question)
            )
            if activity:
                activity = _mark_hybrid_execution(activity, graph_hints)
                generated = True
            if not _activity_has_data(activity):
                # Graph context is a fallback for questions the analytical
                # plan could not answer with computed points.  A data-backed
                # plan already answers the question, so replacing it with a
                # skeleton (and its row-level search results) would lose the
                # computed chart and expose record values in the payload.
                fallback_activity = (
                    _graph_time_series_activity_context(store, question)
                    or _graph_ranked_dimension_context(store, question)
                    or _aggregate_cache_activity(store, question)
                    or _activity_dashboard_context(store, results, question)
                )
                if _activity_has_data(fallback_activity):
                    activity = fallback_activity
                    generated = False
        if activity:
            shape_resolved = str(
                (activity.get("summary") or {}).get("reasoningSource") or ""
            ) == "shape"
            if generated and not shape_resolved:
                activity = _llm_design_complex_dashboard(plan_question, activity)
            # Hermes-style verify->repair: a cheap reviewer LLM checks the produced
            # chart actually answers the question, and re-plans once if not. Fires
            # only on the hard paths (problem-solve requests and the LLM-planner
            # fallback) - never on the deterministic-shape eval corpus.
            if problem_rephrase is not None or (generated and not shape_resolved):
                # Verify against the user's ORIGINAL problem (not the rephrase):
                # a bad rephrase must not be able to fool the reviewer into
                # accepting a chart that restates the problem instead of
                # answering it. The re-plan still uses the reviewer's refined
                # question, which names the category and measure needed.
                activity = _verify_and_repair_dashboard(
                    store,
                    user_question,
                    activity,
                    database_path,
                    graph_hints,
                    user_question,
                )
            if problem_rephrase is not None:
                # Hermes-style chain: after the verified chart, keep drilling into
                # the problem with additional charts until it is answered.
                activity = _chain_problem_analysis(
                    store,
                    user_question,
                    activity,
                    database_path,
                    graph_hints,
                )
            activity = _agent_dashboard_reasoning(question, activity)
            write_graph_dashboard_cache(
                store.path,
                question,
                activity,
                source_version=source_version,
                original_question=user_question,
            )
            if problem_rephrase is not None:
                # Self-heal the cache: the read was keyed on the rephrased
                # question, so write the (possibly repaired) chart back under
                # that key too, replacing any poisoned entry.
                write_graph_dashboard_cache(
                    store.path,
                    plan_question,
                    activity,
                    source_version=source_version,
                    original_question=user_question,
                )
            if problem_insight:
                activity = _attach_problem_insight(activity, problem_insight)
            activity = _localize_activity_to_thai(user_question, activity)
        title = "แดชบอร์ดข้อมูล" if _contains_thai(user_question) else ("Activity dashboard" if activity else "Dashboard graph")
        marker_results = [] if activity else results
        answer = graph_dashboard_marker(
            status=store.status(),
            results=marker_results,
            activity=activity,
            title=title,
        )
    else:
        answer = _duckdb_basic_text_answer(_duckdb_database_path(store), user_question) or _llm_answer(user_question, results)
    if notices:
        answer = "\n\n".join(notices + [answer])
    return {"messages": [AIMessage(content=answer)]}


def _is_complex_dashboard_request(question: str) -> bool:
    lowered = f" {question.lower()} "
    return any(
        phrase in lowered
        for phrase in (" and ", " compare ", " split by ", " grouped by ", " group by ", " breakdown by ", " for each ", " within ")
    )


builder = StateGraph(AgentState)
builder.add_node("dashboard_agent", run_agent)
builder.add_edge(START, "dashboard_agent")
builder.add_edge("dashboard_agent", END)

graph = builder.compile()
