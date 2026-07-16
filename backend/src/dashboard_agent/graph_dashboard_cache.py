from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


CACHE_VERSION = 1
MAX_CACHE_ENTRIES = 100
CACHE_LOCK = threading.Lock()
LANGUAGE_CACHE_LOCK = threading.Lock()
STOP_TERMS = {
    "a", "all", "and", "as", "at", "by", "can", "dashboard", "for", "from",
    "give", "have", "i", "in", "me", "of", "please", "show", "the", "to", "what", "which", "with",
}
ALIASES = {
    "institution": "institute",
    "school": "institute",
    "organization": "institute",
    "student": "user",
    "learner": "user",
    "users": "user",
    "students": "user",
    "learners": "user",
    "completed": "complete",
    "finished": "complete",
    "passed": "complete",
    "ids": "id",
}


def read_graph_dashboard_cache(graph_path: str | Path, question: str) -> dict[str, Any]:
    path = _cache_path(Path(graph_path))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    normalized = _normalize_question(question)
    terms = _semantic_terms(question)
    intent = _intent_shape(question)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("activity"), dict):
            continue
        if entry.get("normalized_question") == normalized:
            candidates.append((2.0, entry))
            continue
        if entry.get("intent") != intent:
            continue
        cached_terms = {str(term) for term in entry.get("semantic_terms") or []}
        if not terms or not cached_terms:
            continue
        intersection = terms & cached_terms
        containment = len(intersection) / max(min(len(terms), len(cached_terms)), 1)
        union_score = len(intersection) / max(len(terms | cached_terms), 1)
        if containment >= 0.9 and union_score >= 0.72:
            candidates.append((containment + union_score, entry))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (-item[0], -float(item[1].get("updated_at") or 0)))
    entry = candidates[0][1]
    activity = deepcopy(entry["activity"])
    _mark_graph_hit(activity, str(entry.get("question") or ""))
    return activity


def write_graph_dashboard_cache(graph_path: str | Path, question: str, activity: dict[str, Any]) -> bool:
    if not _is_duckdb_activity(activity):
        return False
    path = _cache_path(Path(graph_path))
    entry = {
        "id": hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()[:24],
        "question": question.strip(),
        "normalized_question": _normalize_question(question),
        "semantic_terms": sorted(_semantic_terms(question)),
        "intent": _intent_shape(question),
        "updated_at": time.time(),
        "activity": deepcopy(activity),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_LOCK:
        payload: dict[str, Any] = {"version": CACHE_VERSION, "entries": []}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                    payload = existing
            except (OSError, json.JSONDecodeError):
                pass
        entries = [item for item in payload.get("entries") or [] if isinstance(item, dict) and item.get("id") != entry["id"]]
        entries.insert(0, entry)
        payload = {"version": CACHE_VERSION, "entries": entries[:MAX_CACHE_ENTRIES]}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(path)
    return True


def read_question_translation(graph_path: str | Path, question: str) -> str:
    path = _language_cache_path(Path(graph_path))
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    translations = payload.get("translations") if isinstance(payload, dict) else None
    if not isinstance(translations, dict):
        return ""
    entry = translations.get(_translation_key(question))
    if not isinstance(entry, dict) or entry.get("question") != question.strip():
        return ""
    return str(entry.get("english_question") or "").strip()


def write_question_translation(graph_path: str | Path, question: str, english_question: str) -> bool:
    source = question.strip()
    translated = english_question.strip()
    if not source or not translated:
        return False
    path = _language_cache_path(Path(graph_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with LANGUAGE_CACHE_LOCK:
        payload: dict[str, Any] = {"version": CACHE_VERSION, "translations": {}}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("translations"), dict):
                    payload = existing
            except (OSError, json.JSONDecodeError):
                pass
        translations = dict(payload.get("translations") or {})
        translations[_translation_key(source)] = {
            "question": source,
            "english_question": translated,
            "updated_at": time.time(),
        }
        if len(translations) > 500:
            ordered = sorted(
                translations.items(),
                key=lambda item: float(item[1].get("updated_at") or 0) if isinstance(item[1], dict) else 0,
                reverse=True,
            )
            translations = dict(ordered[:500])
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": CACHE_VERSION, "translations": translations}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    return True


def _mark_graph_hit(activity: dict[str, Any], matched_question: str) -> None:
    summary = activity.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["executionSource"] = "graph"
        summary["graphCacheMatchedQuestion"] = matched_question
        reasoning_is_current = (
            summary.get("reasoningSource") == "agent"
            and summary.get("reasoningExecutionSource") == "graph"
        )
    else:
        reasoning_is_current = False
    datasets = activity.get("datasets")
    if isinstance(datasets, dict):
        for metadata in datasets.values():
            if isinstance(metadata, dict):
                metadata["object_type"] = "graph_dashboard_aggregate"
    if not reasoning_is_current:
        activity["decisionTrace"] = []


def _is_duckdb_activity(activity: dict[str, Any]) -> bool:
    datasets = activity.get("datasets")
    if not isinstance(datasets, dict):
        return False
    for metadata in datasets.values():
        if not isinstance(metadata, dict):
            continue
        object_type = str(metadata.get("object_type") or "").lower()
        if "duckdb" in object_type:
            return True
    return bool(activity.get("chartSlots") or activity.get("layoutSpec"))


def _cache_path(graph_path: Path) -> Path:
    return graph_path.with_name(f"{graph_path.name}.dashboards.json")


def _language_cache_path(graph_path: Path) -> Path:
    return graph_path.with_name(f"{graph_path.name}.languages.json")


def _translation_key(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _normalize_question(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question.lower()))


def _semantic_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for term in re.findall(r"[a-z0-9]+", question.lower()):
        if term in STOP_TERMS or len(term) < 2:
            continue
        if term.endswith("s") and len(term) > 3:
            term = term[:-1]
        terms.add(ALIASES.get(term, term))
    return terms


def _intent_shape(question: str) -> dict[str, bool]:
    lowered = question.lower()
    return {
        "time": any(term in lowered for term in ("over time", "trend", "timeline", "daily", "weekly", "monthly")),
        "rank": any(term in lowered for term in ("most", "top", "highest", "largest", "rank")),
        "split": any(term in lowered for term in ("split", "grouped", "breakdown", "composition")),
        "distribution": any(term in lowered for term in ("distribution", "composition", "share", "percentage")),
    }
