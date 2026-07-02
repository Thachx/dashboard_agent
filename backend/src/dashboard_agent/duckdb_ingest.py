from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_source import JsonObject


def ingest_duckdb_database(
    store: JsonGraphStore,
    database_path: str | Path,
    *,
    table: str = "unified_records",
    sample_records_per_source: int = 3,
    max_sources: int | None = None,
    max_fields_per_source: int = 80,
) -> dict[str, Any]:
    """Rebuild the dashboard graph from a DuckDB database table.

    Preferred input is the generic warehouse table produced by
    scripts/build_unified_duckdb.py:
    source_path, source_format, source_table, record_index, payload_json, ingested_at.

    Other DuckDB tables are ingested from schema, row count, and a small sample.
    The graph stays generic: structure comes from DuckDB metadata and payload
    keys, not source-specific hardcoding.
    """

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for DuckDB graph ingestion") from exc

    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _table_exists(con, table):
            available = ", ".join(_table_names(con)) or "<none>"
            raise ValueError(f"DuckDB table {table!r} does not exist. Available tables: {available}")
        columns = _table_columns(con, table)
        if _is_unified_records_table(columns):
            objects = list(
                _unified_record_objects(
                    con,
                    table,
                    db_path,
                    sample_records_per_source=sample_records_per_source,
                    max_sources=max_sources,
                    max_fields_per_source=max_fields_per_source,
                )
            )
            objects.extend(_duckdb_aggregate_objects(con, db_path))
            graph_kind = "duckdb-unified"
        else:
            objects = [
                _generic_table_object(
                    con,
                    table,
                    db_path,
                    columns,
                    sample_records=sample_records_per_source,
                )
            ]
            graph_kind = "duckdb-table"
    finally:
        con.close()

    stats = store.rebuild(objects)
    store.graph.graph.update(
        graph_kind=graph_kind,
        duckdb_database=str(db_path),
        duckdb_table=table,
        duckdb_source_count=len(objects),
    )
    store._persist()
    stats = store.status()
    stats.update(
        duckdb_database=str(db_path),
        duckdb_table=table,
        duckdb_source_count=len(objects),
    )
    return stats


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_names(con: Any) -> list[str]:
    rows = con.execute("SHOW TABLES").fetchall()
    return [str(row[0]) for row in rows]


def _table_exists(con: Any, table: str) -> bool:
    return table in set(_table_names(con))


def _table_columns(con: Any, table: str) -> list[dict[str, Any]]:
    rows = con.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return [
        {
            "cid": row[0],
            "name": str(row[1]),
            "type": str(row[2]),
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


def _is_unified_records_table(columns: list[dict[str, Any]]) -> bool:
    names = {str(column["name"]) for column in columns}
    return {"source_path", "source_format", "source_table", "payload_json"}.issubset(names)


def _unified_record_objects(
    con: Any,
    table: str,
    db_path: Path,
    *,
    sample_records_per_source: int,
    max_sources: int | None,
    max_fields_per_source: int,
) -> Iterable[JsonObject]:
    quoted = _quote_identifier(table)
    limit_clause = "" if max_sources is None else f" LIMIT {int(max_sources)}"
    if _table_exists(con, "source_summary"):
        summaries = con.execute(
            f"""
            SELECT source_path, source_format, source_table, records AS record_count
            FROM source_summary
            ORDER BY record_count DESC, source_path
            {limit_clause}
            """
        ).fetchall()
    else:
        summaries = con.execute(
            f"""
            SELECT source_path, source_format, source_table, count(*) AS record_count
            FROM {quoted}
            GROUP BY 1, 2, 3
            ORDER BY record_count DESC, source_path
            {limit_clause}
            """
        ).fetchall()

    source_paths = [str(row[0]) for row in summaries]
    samples_by_source: dict[str, list[str]] = defaultdict(list)
    if source_paths and sample_records_per_source > 0:
        for source_path in source_paths:
            samples = con.execute(
                f"""
                SELECT payload_json
                FROM {quoted}
                WHERE source_path = ?
                ORDER BY record_index
                LIMIT ?
                """,
                [source_path, int(sample_records_per_source)],
            ).fetchall()
            for (payload_json,) in samples:
                samples_by_source[source_path].append(str(payload_json))

    for source_path, source_format, source_table, record_count in summaries:
        source_path = str(source_path)
        payloads = samples_by_source.get(source_path, [])
        parsed_samples = [_parse_payload(payload) for payload in payloads]
        fields = _field_summary(parsed_samples, max_fields=max_fields_per_source)
        value = {
            "duckdb_database": str(db_path),
            "duckdb_table": table,
            "source_path": source_path,
            "source_format": source_format,
            "source_table": source_table,
            "record_count": int(record_count),
            "sample_records": parsed_samples,
            "sample_fields": fields,
            "sample_field_count": len(fields),
        }
        yield JsonObject(
            key=f"duckdb/{table}/{source_path}",
            etag=f"duckdb:{table}:{source_path}:{record_count}",
            value=value,
            object_type="duckdb",
            size=int(record_count),
        )


def _duckdb_aggregate_objects(con: Any, db_path: Path) -> list[JsonObject]:
    objects: list[JsonObject] = []
    objects.extend(_ranked_dimension_aggregate_objects(con, db_path))
    time_series = _activity_time_series_aggregate_object(con, db_path)
    if time_series:
        objects.append(time_series)
    return objects


def _ranked_dimension_aggregate_objects(con: Any, db_path: Path) -> list[JsonObject]:
    objects: list[JsonObject] = []
    table_names = [
        str(row[0])
        for row in con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name LIKE 'dashboard_agent_%'
            ORDER BY table_name
            """
        ).fetchall()
    ]
    for table in table_names:
        if any(part in table for part in ("cache", "map")):
            continue
        columns = [column["name"] for column in _table_columns(con, table)]
        dimensions = [column for column in columns if _is_rank_dimension_column(column)]
        measures = [column for column in columns if _is_rank_measure_column(column)]
        for dimension in dimensions:
            for measure in measures:
                aggregate = _ranked_dimension_payload(con, db_path, table, dimension, measure)
                if not aggregate:
                    continue
                key = f"duckdb/aggregate/ranked_dimension/{table}/{dimension}/by/{measure}"
                objects.append(
                    JsonObject(
                        key=key,
                        etag=f"duckdb_aggregate:{table}:{dimension}:{measure}:{aggregate['top_value']}",
                        value=aggregate,
                        object_type="duckdb_aggregate",
                        size=int(aggregate.get("total_measure_values") or 0),
                    )
                )
    return objects


def _ranked_dimension_payload(
    con: Any,
    db_path: Path,
    table: str,
    dimension: str,
    measure: str,
) -> dict[str, Any]:
    dimension_expr = _quote_identifier(dimension)
    measure_expr = _quote_identifier(measure)
    quoted_table = _quote_identifier(table)
    try:
        rows = con.execute(
            f"""
            SELECT
                {dimension_expr} AS label,
                count(DISTINCT {measure_expr}) AS value
            FROM {quoted_table}
            WHERE {dimension_expr} IS NOT NULL
              AND cast({dimension_expr} AS varchar) <> ''
              AND lower(cast({dimension_expr} AS varchar)) NOT IN ('unknown', 'none', 'null')
              AND {measure_expr} IS NOT NULL
            GROUP BY 1
            ORDER BY value DESC, label
            LIMIT 12
            """
        ).fetchall()
        if not rows:
            return {}
        total_measure = con.execute(
            f"""
            SELECT count(DISTINCT {measure_expr})
            FROM {quoted_table}
            WHERE {measure_expr} IS NOT NULL
              AND {dimension_expr} IS NOT NULL
              AND cast({dimension_expr} AS varchar) <> ''
              AND lower(cast({dimension_expr} AS varchar)) NOT IN ('unknown', 'none', 'null')
            """
        ).fetchone()[0]
        dimension_count = con.execute(
            f"""
            SELECT count(DISTINCT {dimension_expr})
            FROM {quoted_table}
            WHERE {dimension_expr} IS NOT NULL
              AND cast({dimension_expr} AS varchar) <> ''
              AND lower(cast({dimension_expr} AS varchar)) NOT IN ('unknown', 'none', 'null')
            """
        ).fetchone()[0]
    except Exception:
        return {}
    chart_data = [{"label": str(label), "value": int(value)} for label, value in rows]
    top = chart_data[0]
    return {
        "aggregate_kind": "ranked_dimension",
        "duckdb_database": str(db_path),
        "duckdb_table": table,
        "dimension_field": dimension,
        "measure_field": measure,
        "dimension_label": _display_name(dimension),
        "measure_label": _display_measure_name(measure),
        "total_measure_values": int(total_measure or 0),
        "distinct_dimension_values": int(dimension_count or 0),
        "top_label": top["label"],
        "top_value": top["value"],
        "chart_data": chart_data,
        "semantic_terms": _semantic_terms(table, dimension, measure),
    }


def _activity_time_series_aggregate_object(con: Any, db_path: Path) -> JsonObject | None:
    if not _table_exists(con, "dashboard_agent_activity_hourly_cache"):
        return None
    try:
        rows = con.execute(
            """
            WITH ordered AS (
                SELECT
                    label,
                    coalesce(cumulative_records, records) AS records,
                    coalesce(cumulative_users, users) AS users,
                    row_number() OVER (ORDER BY label) AS rn,
                    count(*) OVER () AS total_rows
                FROM dashboard_agent_activity_hourly_cache
            ),
            sampled AS (
                SELECT
                    *,
                    CASE
                        WHEN total_rows <= 24 THEN rn
                        WHEN rn = 1 THEN 1
                        WHEN rn = total_rows THEN 24
                        ELSE 2 + cast(floor(((rn - 2) * 22.0) / greatest(total_rows - 2, 1)) AS integer)
                    END AS sample_bucket
                FROM ordered
            ),
            bucketed AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY sample_bucket
                        ORDER BY
                            CASE WHEN sample_bucket = 1 THEN rn END ASC,
                            CASE WHEN sample_bucket = 24 THEN rn END DESC,
                            rn ASC
                    ) AS bucket_rank
                FROM sampled
            )
            SELECT label, records, users
            FROM bucketed
            WHERE bucket_rank = 1
            ORDER BY label
            """
        ).fetchall()
        totals = con.execute(
            """
            SELECT max(cumulative_records), max(cumulative_users)
            FROM dashboard_agent_activity_hourly_cache
            """
        ).fetchone()
    except Exception:
        return None
    if not rows:
        return None
    records_data = [{"label": str(label), "value": int(records)} for label, records, _users in rows]
    users_data = [{"label": str(label), "value": int(users)} for label, _records, users in rows]
    value = {
        "aggregate_kind": "time_series",
        "duckdb_database": str(db_path),
        "duckdb_table": "dashboard_agent_activity_hourly_cache",
        "time_field": "label",
        "records_series": records_data,
        "users_series": users_data,
        "total_records": int(totals[0] or 0),
        "total_users": int(totals[1] or 0),
        "semantic_terms": ["activity", "event", "row", "record", "user", "users", "time", "trend", "over_time"],
    }
    return JsonObject(
        key="duckdb/aggregate/time_series/activity_users_records",
        etag=f"duckdb_aggregate:time_series:{value['total_records']}:{value['total_users']}",
        value=value,
        object_type="duckdb_aggregate",
        size=int(value["total_records"]),
    )


def _is_rank_dimension_column(column: str) -> bool:
    lowered = column.lower()
    if lowered.endswith("_payload_json") or lowered in {"activity_record_index", "record_index", "user_id", "activity_user_id"}:
        return False
    return any(
        token in lowered
        for token in ("name", "school", "institute", "department", "province", "course", "category", "type", "status")
    )


def _is_rank_measure_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"user_id", "activity_user_id", "student_id", "learner_id", "course_id"} or lowered.endswith("_user_id")


def _display_name(field: str) -> str:
    spaced = field.replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in spaced.split())


def _display_measure_name(field: str) -> str:
    lowered = field.lower()
    if lowered in {"user_id", "activity_user_id", "student_id", "learner_id"} or lowered.endswith("_user_id"):
        return "Users"
    if lowered == "course_id":
        return "Courses"
    return _display_name(field)


def _semantic_terms(*values: str) -> list[str]:
    terms: set[str] = set()
    for value in values:
        for term in value.replace("_", " ").replace("-", " ").lower().split():
            if len(term) >= 2:
                terms.add(term)
    if "school" in terms:
        terms.update({"institute", "institution", "organization", "name"})
    if "user" in terms:
        terms.update({"users", "learner", "student"})
    if "course" in terms:
        terms.update({"subject", "class"})
    terms.update({"top", "most", "highest", "rank"})
    return sorted(terms)


def _generic_table_object(
    con: Any,
    table: str,
    db_path: Path,
    columns: list[dict[str, Any]],
    *,
    sample_records: int,
) -> JsonObject:
    quoted = _quote_identifier(table)
    record_count = int(con.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0])
    sample_rows = []
    if sample_records > 0:
        rows = con.execute(f"SELECT * FROM {quoted} LIMIT ?", [int(sample_records)]).fetchall()
        column_names = [column["name"] for column in columns]
        sample_rows = [dict(zip(column_names, row)) for row in rows]
    value = {
        "duckdb_database": str(db_path),
        "duckdb_table": table,
        "record_count": record_count,
        "columns": columns,
        "sample_records": _json_safe(sample_rows),
    }
    return JsonObject(
        key=f"duckdb/{table}",
        etag=f"duckdb:{table}:{record_count}",
        value=value,
        object_type="duckdb",
        size=record_count,
    )


def _parse_payload(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"payload_json": payload}


def _field_summary(records: Iterable[Any], *, max_fields: int) -> list[dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            field = fields.setdefault(
                str(key),
                {
                    "name": str(key),
                    "types": set(),
                    "sample_values": [],
                },
            )
            field["types"].add(type(value).__name__)
            if value is not None and len(field["sample_values"]) < 3:
                field["sample_values"].append(_json_safe(value))
            if len(fields) >= max_fields:
                break
        if len(fields) >= max_fields:
            break
    return [
        {
            "name": field["name"],
            "types": sorted(field["types"]),
            "sample_values": field["sample_values"],
        }
        for field in fields.values()
    ]


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        return str(value)
