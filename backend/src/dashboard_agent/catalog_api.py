"""Field & Table Guide catalog API.

A tiny read-only FastAPI service that exposes the warehouse schema
(tables, columns, types, row counts, sample values) to the web UI.

It reuses the planner's cached schema catalog (_catalog), so the schema
introspection cost is paid once per warehouse file version and shared.
Row counts and sample rows are cached the same way.

Thai documentation (what each table/field is about, in Thai) is generated
from the live schema by the configured LLM -- schema-driven, never
hardcoded -- and cached on disk keyed by a schema hash. It regenerates
automatically when the schema changes. If the LLM is unavailable the UI
falls back to generic schema-derived descriptions.

Run:  uvicorn dashboard_agent.catalog_api:app --host 127.0.0.1 --port 8090
Env:  DASHBOARD_WAREHOUSE_PATH  (default: data/_warehouse/dashboard_agent.duckdb)
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard_agent.dashboard_planner import _catalog
from dashboard_agent.readonly_duckdb import connect_read_only

app = FastAPI(title="dashboard-agent-catalog", version="1.1")

# The UI is served from :3000 and proxies server-side, but allow direct
# localhost access for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_DEFAULT_WAREHOUSE = "/home/thacha/dashboard_agent/data/_warehouse/dashboard_agent.duckdb"
_MAX_SAMPLES_PER_COLUMN = 3
_SAMPLE_ROW_SCAN = 5
_MAX_SAMPLE_LEN = 60
_MAX_TERMS = 12
_THAI_LLM_TIMEOUT = 60.0

# ---------------------------------------------------------------------------
# env / settings
# ---------------------------------------------------------------------------

_env_loaded = False


def _load_env() -> None:
    """Load backend/.env once (the catalog service is not launched with it)."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception:
        pass


def _warehouse_path() -> Path:
    return Path(os.environ.get("DASHBOARD_WAREHOUSE_PATH", _DEFAULT_WAREHOUSE))


_payload_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_payload_lock = threading.Lock()
# Last successfully built payload, kept across warehouse file versions so a
# request that lands while the ingest holds the exclusive lock can serve a
# slightly-stale snapshot instead of a 500. Refreshed in the background.
_last_good_payload: dict[str, Any] | None = None
_refresh_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="catalog-refresh"
)
_refresh_lock = threading.Lock()
_refresh_inflight = False

_thai_lock = threading.Lock()
_thai_inflight = False
_source_thai_lock = threading.Lock()
_source_thai_inflight = False


def _file_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def _sample_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) > _MAX_SAMPLE_LEN:
        text = text[: _MAX_SAMPLE_LEN - 1] + "\u2026"
    return text


def _source_tables(con: Any) -> list[dict[str, Any]]:
    """Original source tables from the pipeline's object manifest.

    The S3->DuckDB pipeline records every ingested source dataset in
    `lead_etl_object_manifest` (source dataset path -> physical table, with
    row count, ingestion mode, and last-ingested time). Those `lead_etl_*`
    tables are the ORIGINAL data before the derived `dashboard_agent_*`
    tables are built, so the guide surfaces them here, straight from that
    metadata. Returns [] if the manifest is absent (older warehouse).
    """
    try:
        rows = con.execute(
            """
            select physical_table, dataset, row_count, ingestion_mode, ingested_at
            from lead_etl_object_manifest
            order by dataset
            """
        ).fetchall()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for physical, dataset, row_count, mode, ingested in rows:
        name = str(physical)
        try:
            col_rows = con.execute(f'pragma table_info("{name}")').fetchall()
            columns = [
                {"name": str(r[1]), "dataType": str(r[2])}
                for r in col_rows
            ]
        except Exception:
            columns = []
        ds = str(dataset or "")
        source_system = ds.split("/", 1)[0] if ds else ""
        ingested_iso = None
        if ingested is not None:
            try:
                ingested_iso = ingested.isoformat()
            except Exception:
                ingested_iso = str(ingested)
        out.append(
            {
                "name": name,
                "sourceSystem": source_system,
                "sourcePath": ds,
                "rowCount": int(row_count or 0),
                "ingestionMode": str(mode or ""),
                "ingestedAt": ingested_iso,
                "columns": columns,
            }
        )
    return out


def _build_payload(path: Path, *, attempts: int = 8, backoff_seconds: float = 1.0) -> dict[str, Any]:
    con = connect_read_only(str(path), attempts=attempts, backoff_seconds=backoff_seconds)
    try:
        catalog = _catalog(con, path)
        tables: list[dict[str, Any]] = []
        for name in sorted(catalog):
            profile = catalog[name]
            # Row count + sample rows: two cheap queries per table.
            try:
                row_count = int(con.execute(f'select count(*) from "{name}"').fetchone()[0])
            except Exception:
                row_count = 0
            try:
                sample_rows = con.execute(f'select * from "{name}" limit {_SAMPLE_ROW_SCAN}').fetchall()
                col_names = [d[0] for d in con.description]
            except Exception:
                sample_rows, col_names = [], []
            samples_by_col: dict[str, list[str]] = {}
            for row in sample_rows:
                for col, value in zip(col_names, row):
                    bucket = samples_by_col.setdefault(col, [])
                    text = _sample_value(value)
                    if text is not None and text not in bucket and len(bucket) < _MAX_SAMPLES_PER_COLUMN:
                        bucket.append(text)
            columns = []
            for col_name, col in profile.columns.items():
                columns.append(
                    {
                        "name": col.name,
                        "dataType": col.data_type,
                        "isTime": col.is_time,
                        "isIdentifier": col.is_identifier,
                        "terms": sorted(col.terms)[:_MAX_TERMS],
                        "samples": samples_by_col.get(col.name, []),
                    }
                )
            tables.append(
                {
                    "name": profile.name,
                    "rowCount": row_count,
                    "sourcePaths": list(profile.source_paths),
                    "columns": columns,
                }
            )
        return {
            "version": 2,
            "warehouse": path.name,
            "tableCount": len(tables),
            "tables": tables,
            "sourceTables": _source_tables(con),
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Thai documentation (LLM-generated from the live schema, cached on disk)
# ---------------------------------------------------------------------------


def _thai_cache_path() -> Path:
    return _warehouse_path().parent / "catalog_thai.json"


def _source_thai_cache_path() -> Path:
    return _warehouse_path().parent / "catalog_thai_source.json"


def _derived_hash(payload: dict[str, Any]) -> str:
    # Hash ONLY the derived tables' identity (names + column names + types).
    # Deliberately EXCLUDES rowCount (changes on every ingestion) and the
    # source tables (they have their own hash/cache), so the derived docs
    # stay valid across hourly ingestions and source-table changes.
    canonical = {
        t["name"]: {
            "cols": {c["name"]: c["dataType"] for c in t["columns"]},
        }
        for t in payload["tables"]
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# Bump when the source-doc prompt/output shape changes, so a stale cache
# (e.g. summaries-only) is regenerated with the new per-column descriptions.
_SOURCE_DOC_VERSION = 2

def _source_hash(payload: dict[str, Any]) -> str:
    canonical = {
        "v": _SOURCE_DOC_VERSION,
        "tables": {
            t["name"]: {
                "cols": {c["name"]: c["dataType"] for c in t["columns"]},
            }
            for t in payload.get("sourceTables") or []
        },
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _load_thai_cache(schema_hash: str) -> dict[str, Any] | None:
    """Derived tables: {'overview': str, 'tables': {name: {summary, columns}}}."""
    try:
        data = json.loads(_thai_cache_path().read_text(encoding="utf-8"))
        if data.get("schemaHash") == schema_hash and isinstance(data.get("tables"), dict):
            return {
                "overview": str(data.get("overview") or ""),
                "tables": data["tables"],
            }
    except Exception:
        pass
    return None


def _save_thai_cache(schema_hash: str, docs: dict[str, Any]) -> None:
    """docs = {'overview': str, 'tables': {name: {summary, columns}}}."""
    try:
        _thai_cache_path().write_text(
            json.dumps(
                {
                    "schemaHash": schema_hash,
                    "generatedAt": int(time.time()),
                    "overview": str(docs.get("overview") or ""),
                    "tables": docs.get("tables") or {},
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_source_thai_cache(schema_hash: str) -> dict[str, Any] | None:
    """Source tables: {'tables': {name: {summary}}}."""
    try:
        data = json.loads(_source_thai_cache_path().read_text(encoding="utf-8"))
        if data.get("schemaHash") == schema_hash and isinstance(data.get("tables"), dict):
            return {"tables": data["tables"]}
    except Exception:
        pass
    return None


def _save_source_thai_cache(schema_hash: str, docs: dict[str, Any]) -> None:
    try:
        _source_thai_cache_path().write_text(
            json.dumps(
                {
                    "schemaHash": schema_hash,
                    "generatedAt": int(time.time()),
                    "tables": docs.get("tables") or {},
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _schema_brief(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for t in payload["tables"]:
        lines.append(f"Table {t['name']} ({t['rowCount']} rows):")
        for c in t["columns"]:
            flags = []
            if c["isTime"]:
                flags.append("time")
            if c["isIdentifier"]:
                flags.append("identifier")
            flag_txt = f" [{', '.join(flags)}]" if flags else ""
            samples = ", ".join(c["samples"][:3])
            sample_txt = f" e.g. [{samples}]" if samples else ""
            lines.append(f"  - {c['name']} ({c['dataType']}){flag_txt}{sample_txt}")
    return "\n".join(lines)


def _source_brief(payload: dict[str, Any]) -> str:
    """Compact brief for the original source tables (summary + columns)."""
    lines: list[str] = []
    for t in payload.get("sourceTables") or []:
        src = t.get("sourcePath") or t.get("sourceSystem") or ""
        lines.append(f"Table {t['name']} ({t['rowCount']} rows) [source: {src}]:")
        for c in t["columns"]:
            lines.append(f"  - {c['name']} ({c['dataType']})")
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You document a data warehouse for end users who will ask questions about it in Thai. "
    "First write ONE overall overview (2-3 sentences) in natural, correct Thai explaining what "
    "this whole dataset is about at a high level (the business domain, who/what it tracks, and "
    "how the tables relate) -- this is the 'what is this data about' summary. "
    "Then for EACH table write a 1-2 sentence summary in natural, correct Thai explaining what the "
    "data in that table is about (the business meaning, not the technical layout). "
    "For EACH column write one short Thai phrase (max ~15 words) explaining what the field "
    "means and what kind of values it holds, based on the column name, its type, and its "
    "sample values. Keep technical identifiers (table names, column names) in English where "
    "natural, but explain their meaning in Thai. Do not invent facts not supported by the "
    "schema and samples. Respond with VALID JSON ONLY (no markdown, no commentary) in this "
    'exact shape: {"overview": "<Thai>", "tables": {"<table_name>": {"summary": "<Thai>", '
    '"columns": {"<column_name>": "<Thai>"}}}}'
)

_SOURCE_SYSTEM_PROMPT = (
    "You document ORIGINAL SOURCE tables of a data warehouse for end users who ask questions in Thai. "
    "These are raw tables ingested from source systems (the [source: ...] tag shows the source system and "
    "dataset path). For EACH table write ONE 1-2 sentence summary in natural, correct Thai explaining what "
    "the data in that table is about (the business meaning, not the technical layout). "
    "For EACH column write one short Thai phrase (max ~15 words) explaining what the field means and what "
    "kind of values it holds, based on the column name and its data type. Keep technical identifiers "
    "(table names, column names) in English where natural. Do not invent facts not supported by the table "
    "name, source path, and column names. Respond with VALID JSON ONLY (no markdown, no commentary) "
    'in this exact shape: {"tables": {"<table_name>": {"summary": "<Thai>", '
    '"columns": {"<column_name>": "<Thai>", ...}}}}'
)


def _call_llm_json(system: str, human: str) -> dict[str, Any]:
    _load_env()
    from langchain_openai import ChatOpenAI

    from dashboard_agent.config import Settings

    settings = Settings.from_env()

    # Candidate providers, tried in order: the GPT-OSS server (reached via the
    # local SSH tunnel on 127.0.0.1:18080) first, then the OpenRouter free
    # tier as a fallback. GPT-OSS is OpenAI-compatible, so the same client
    # works for both. Each candidate carries provider-specific kwargs.
    candidates: list[tuple[str, str, str, dict[str, str], dict[str, Any]]] = []
    gptoss_base = os.environ.get("GPTOSS_BASE_URL", "http://127.0.0.1:18080/v1")
    gptoss_key = os.environ.get("GPTOSS_API_KEY", "")
    gptoss_model = os.environ.get("GPTOSS_MODEL", "gpt-oss-120b")
    if gptoss_key:
        # GPT-OSS is a reasoning model: cap the reasoning effort and give it a
        # real token budget so the JSON answer isn't starved by its own
        # chain-of-thought.
        candidates.append((
            gptoss_model, gptoss_key, gptoss_base, {},
            {"max_tokens": int(os.environ.get("GPTOSS_MAX_TOKENS", "8000")),
             "extra_body": {"reasoning_effort": "low"}},
        ))

    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_title:
        headers["X-OpenRouter-Title"] = settings.openrouter_title
    if settings.openrouter_api_key:
        models: list[str] = [settings.openrouter_main_model]
        for extra in (settings.openrouter_reserve_model_1, settings.openrouter_reserve_model_2):
            if extra and extra not in models:
                models.append(extra)
        for model_name in models:
            candidates.append((model_name, settings.openrouter_api_key, settings.openrouter_base_url, headers, {}))

    if not candidates:
        raise RuntimeError("no LLM provider configured")

    last_err: Exception | None = None
    for model_name, api_key, base_url, hdrs, extra in candidates:
        for attempt in range(2):  # one retry per model for transient 429/5xx
            try:
                model = ChatOpenAI(
                    api_key=api_key,
                    model=model_name,
                    base_url=base_url,
                    default_headers=hdrs or None,
                    temperature=0,
                    **extra,
                )
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                fut = ex.submit(model.invoke, [("system", system), ("human", human)])
                try:
                    content = str(fut.result(timeout=_THAI_LLM_TIMEOUT).content)
                finally:
                    ex.shutdown(wait=False)
                text = content.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx == -1 or end_idx <= start_idx:
                    raise ValueError("no JSON object in LLM response")
                return json.loads(text[start_idx : end_idx + 1])
            except Exception as exc:  # noqa: BLE001 - retry, then next candidate
                last_err = exc
                if attempt == 0:
                    time.sleep(5.0)
    raise RuntimeError(f"Thai doc generation failed: {last_err}")

def _generate_thai_docs(payload: dict[str, Any]) -> dict[str, Any]:
    """Derived tables: overview + per-table summaries + column descriptions."""
    data = _call_llm_json(_SYSTEM_PROMPT, _schema_brief(payload))
    tables = data.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("LLM response missing 'tables' object")
    overview = str(data.get("overview") or "").strip()
    cleaned: dict[str, Any] = {}
    for t in payload["tables"]:
        entry = tables.get(t["name"])
        if not isinstance(entry, dict):
            continue
        summary = str(entry.get("summary") or "").strip()
        cols_in = entry.get("columns")
        cols_in = cols_in if isinstance(cols_in, dict) else {}
        columns: dict[str, str] = {}
        for c in t["columns"]:
            desc = str(cols_in.get(c["name"]) or "").strip()
            if desc:
                columns[c["name"]] = desc
        if summary or columns:
            cleaned[t["name"]] = {"summary": summary, "columns": columns}
    if not cleaned:
        raise ValueError("LLM response had no usable table docs")
    return {"overview": overview, "tables": cleaned}


# Keep each LLM call small: 475 source columns in one call would exceed the
# output token budget and truncate. Batch tables until the batch's column count
# approaches this ceiling, so every call fits comfortably.
_SOURCE_BATCH_MAX_COLS = 100

def _source_batches(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_cols = 0
    for t in payload.get("sourceTables") or []:
        n = len(t["columns"])
        if cur and cur_cols + n > _SOURCE_BATCH_MAX_COLS:
            batches.append(cur)
            cur = []
            cur_cols = 0
        cur.append(t)
        cur_cols += n
    if cur:
        batches.append(cur)
    return batches

def _generate_source_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """One LLM call for a batch of source tables -> {name: {summary, columns}}."""
    data = _call_llm_json(_SOURCE_SYSTEM_PROMPT, _source_brief({"sourceTables": batch}))
    tables = data.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("LLM response missing 'tables' object")
    cleaned: dict[str, Any] = {}
    for t in batch:
        entry = tables.get(t["name"])
        if not isinstance(entry, dict):
            continue
        summary = str(entry.get("summary") or "").strip()
        cols_in = entry.get("columns")
        cols_in = cols_in if isinstance(cols_in, dict) else {}
        columns: dict[str, str] = {}
        for c in t["columns"]:
            desc = str(cols_in.get(c["name"]) or "").strip()
            if desc:
                columns[c["name"]] = desc
        if summary or columns:
            cleaned[t["name"]] = {"summary": summary, "columns": columns}
    return cleaned


def _thai_worker(payload: dict[str, Any], schema_hash: str) -> None:
    global _thai_inflight
    try:
        for attempt in range(3):  # transient 429/5xx; ride it out
            try:
                docs = _generate_thai_docs(payload)
                _save_thai_cache(schema_hash, docs)
                return
            except Exception:
                if attempt < 2:
                    time.sleep(20.0)
    finally:
        with _thai_lock:
            _thai_inflight = False


def _source_thai_worker(payload: dict[str, Any], schema_hash: str) -> None:
    global _source_thai_inflight
    try:
        merged: dict[str, Any] = {}
        for batch in _source_batches(payload):
            ok = False
            for attempt in range(3):  # transient 429/5xx; ride it out
                try:
                    merged.update(_generate_source_batch(batch))
                    ok = True
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(20.0)
            if not ok:
                return  # leave cache stale; a later request retries the whole pass
        _save_source_thai_cache(schema_hash, {"tables": merged})
    finally:
        with _source_thai_lock:
            _source_thai_inflight = False


def _attach_thai(payload: dict[str, Any]) -> None:
    """Merge cached Thai docs into the payload; kick off generation if stale.

    Derived tables and original source tables use SEPARATE caches and
    SEPARATE hashes, so a stale/failed source pass never blocks the derived
    docs (and vice versa). If a payload is already pending, re-check the
    cache first so a finished generation is picked up on the next request
    instead of staying 'pending' forever.
    """
    global _thai_inflight, _source_thai_inflight
    derived_hash = _derived_hash(payload)
    docs = _load_thai_cache(derived_hash)
    if docs is None:
        with _thai_lock:
            if _load_thai_cache(derived_hash) is not None:
                docs = _load_thai_cache(derived_hash)
            elif not _thai_inflight:
                _thai_inflight = True
                threading.Thread(
                    target=_thai_worker, args=(payload, derived_hash), daemon=True
                ).start()
    if docs:
        payload["thaiOverview"] = str(docs.get("overview") or "") or None
        for t in payload["tables"]:
            entry = docs.get("tables", {}).get(t["name"]) or {}
            t["thaiSummary"] = entry.get("summary") or None
            col_docs = entry.get("columns") or {}
            for c in t["columns"]:
                desc = col_docs.get(c["name"])
                if desc:
                    c["thaiDescription"] = desc
        payload["thaiStatus"] = "ready"
    else:
        payload["thaiStatus"] = "pending"

    # Source tables: independent best-effort pass.
    source_tables = payload.get("sourceTables") or []
    if source_tables:
        source_hash = _source_hash(payload)
        sdocs = _load_source_thai_cache(source_hash)
        if sdocs is None:
            with _source_thai_lock:
                if _load_source_thai_cache(source_hash) is not None:
                    sdocs = _load_source_thai_cache(source_hash)
                elif not _source_thai_inflight:
                    _source_thai_inflight = True
                    threading.Thread(
                        target=_source_thai_worker,
                        args=(payload, source_hash),
                        daemon=True,
                    ).start()
        if sdocs:
            for t in source_tables:
                entry = sdocs.get("tables", {}).get(t["name"]) or {}
                t["thaiSummary"] = entry.get("summary") or None
                col_docs = entry.get("columns") or {}
                for c in t["columns"]:
                    desc = col_docs.get(c["name"])
                    if desc:
                        c["thaiDescription"] = desc


def _refresh_in_background(path: Path) -> None:
    """Rebuild the payload once the ingest releases the exclusive lock."""
    global _last_good_payload, _refresh_inflight
    with _refresh_lock:
        if _refresh_inflight:
            return
        _refresh_inflight = True

    def _run() -> None:
        global _last_good_payload
        try:
            # Retry long enough to outlast a full ingest (3-10 min):
            # ~27 attempts with linear backoff 2,4,...,54s ~= 12.5 min.
            payload = _build_payload(path, attempts=27, backoff_seconds=2.0)
            _attach_thai(payload)
            key = _file_key(path)
            with _payload_lock:
                _payload_cache.clear()
                _payload_cache[key] = payload
            _last_good_payload = payload
        except Exception:
            # Still locked (or a build error): keep serving the stale
            # snapshot; the next request retries the refresh.
            pass
        finally:
            with _refresh_lock:
                _refresh_inflight = False

    _refresh_executor.submit(_run)


def _get_payload() -> dict[str, Any]:
    global _last_good_payload
    path = _warehouse_path()
    if not path.exists():
        return {"version": 2, "warehouse": path.name, "tableCount": 0, "tables": [], "sourceTables": [], "error": "warehouse not found"}
    key = _file_key(path)
    cached = _payload_cache.get(key)
    if cached is not None:
        # Re-attach Thai docs on every request: the derived/source caches
        # may have been (re)generated since this payload was built, and the
        # attach is a cheap idempotent merge (two small JSON reads, no DB).
        # thaiStatus alone can't signal this -- it only tracks the derived
        # pass, so a freshly generated source pass would otherwise never
        # surface until the next agent restart.
        _attach_thai(cached)
        return cached
    with _payload_lock:
        cached = _payload_cache.get(key)
        if cached is not None:
            _attach_thai(cached)
            return cached
        try:
            # Foreground: fail fast (single attempt). If the ingest holds
            # the lock, serve the stale snapshot and refresh in the
            # background instead of blocking the request for the retry
            # window.
            payload = _build_payload(path, attempts=1)
        except Exception:
            # The hourly ingest holds the exclusive warehouse lock for
            # minutes, longer than the read-retry window. Serve the last
            # known-good snapshot instead of a 500 and refresh in the
            # background once the lock is released.
            if _last_good_payload is not None:
                stale = dict(_last_good_payload)
                stale["stale"] = True
                _refresh_in_background(path)
                return stale
            raise
        _attach_thai(payload)
        _payload_cache.clear()  # keep only the newest file version
        _payload_cache[key] = payload
        _last_good_payload = payload
        return payload


@app.get("/health")
def health() -> dict[str, Any]:
    path = _warehouse_path()
    return {"ok": True, "warehouse": path.name, "exists": path.exists()}


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    return _get_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CATALOG_PORT", "8090")))
