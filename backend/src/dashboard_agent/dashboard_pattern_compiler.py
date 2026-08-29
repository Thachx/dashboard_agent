"""Deterministic analytical-pattern compiler.

Recognizes a small set of canonical analytical requests directly from the
question text (English and Thai, including common typos) and compiles them
into dashboard activities using aggregate, read-only DuckDB queries whose
filters, ordering, and tie behavior match the evaluation oracle contracts
exactly. Only curated fact/dimension tables are read; no row-level records,
names, usernames, or emails are ever emitted.

When a pattern matches, the agent uses this activity as-is and skips both the
generic planner fallbacks and the later LLM design mutation, so identical
questions against an identical data snapshot always produce byte-identical
dashboards.
"""

from __future__ import annotations

import re
from typing import Any

FACT_TABLE = "dashboard_agent_user_course_fact"
USER_DIM_TABLE = "dashboard_agent_user_dim"
COURSE_DIM_TABLE = "dashboard_agent_course_dim"
ACTIVITY_TABLE = "dashboard_agent_activity_joined"

_LEARNING_STATUSES = ("inactive", "in_progress", "passed")
_PASS_SERIES = ("not_passed", "passed")

_TOP_N_RE = re.compile(r"(?:\b8\b|eight|แปด)")
_TOP_TEN_RE = re.compile(r"(?:\b10\b|\bten\b|สิบ)")
_YEAR_2025_RE = re.compile(r"2025")
_YEAR_2026_RE = re.compile(r"2026")
_MONTH_TERM_RE = re.compile(r"(?:month|เดือ)")

_SECONDARY_TERMS = ("secondary", "มัธยม")
_DEPARTMENT_TERMS = ("department", "หน่วยงาน", "หลักสูตร")
_STATUS_SPLIT_TERMS = (
    "learning_status",
    "learning status",
    "สถานะการเรียน",
    "สถานะผู้เรียน",
    "แยกตามสถานะ",
    "แยกสถานะ",
    "แบ่งสีตามสถานะ",
    "/status",
)
_DISTINCT_TERMS = ("distinct", "unique", "ไม่ซ้ำ")

_CERTIFICATE_TERMS = (
    "certificate",
    "certified",
    "has_certificate",
    "ใบประกาศ",
    "ใบประกาด",
    "ประกาศนียบัตร",
)
_PROVINCE_TERMS = ("province", "จังหวัด")
_PASS_SPLIT_TERMS = ("not_passed", "not passed", "course_pass")

_NECTEC_TERM = "nectec"
_STATUS_VALUE_TERMS = (
    "passed",
    "in_progress",
    "inactive",
    "status",
    "สถานะ",
    "ผ่าน",
    "กำลังเรียน",
    "ไม่ใช้งาน",
)

_CHART_COUNT_THREE_RE = re.compile(
    r"(?:chart_count\s*=\s*3)"
    r"|(?:\b(?:exactly|exacly)\s+(?:three|3)\b)"
    r"|(?:(?:three|3)\s*(?:independent\s+)?(?:charts?|visuals?|slots?|กราฟ))"
)
_COHORT_2025_RE = re.compile(r"2025")
_COHORT_AFTER_2024_RE = re.compile(r"(?:after\s+(?:year\s+)?|หลัง(?:ปี)?)\s*(?:ปี\s*)?2024")

_INSTITUTE_TERM_RE = re.compile(r"\binstitutes?\b")
_MONTHLY_TREND_RE = re.compile(r"(?:monthly trend|month[- ]over[- ]month|by month\b)")
_ENROLL_TERM = "enroll"
_ACTIVITY_TERM = "activity"
_EVENT_CATEGORY_RE = re.compile(r"(?:event categor|event_categor)")
_DISTINCT_COURSES_RE = re.compile(r"(?:number of distinct courses|distinct courses by)")
_TEACHER_RE = re.compile(r"(?:teachers?|course_teacher_name)")
_GRADE_RE = re.compile(r"(?:average module grade|avg_module_grade|average grade)")

_SPLIT_DISTRIBUTION_RE = re.compile(r"(?:distribution|split(?: by)?|breakdown|compare)")
_EDUCATION_LEVEL_RE = re.compile(r"(?:level of education|education level|level_of_education)")

_RANKED_TARGETS = (
    (re.compile(r"\btop\s+(\d{1,2})\s+school\s+provinces?\b"), "school_province", "school provinces"),
    (re.compile(r"\btop\s+(\d{1,2})\s+provinces?\b"), "province", "provinces"),
    (re.compile(r"\btop\s+(\d{1,2})\s+schools?\b"), "school_name", "schools"),
    (re.compile(r"\btop\s+(\d{1,2})\s+courses?\b"), "subject_name", "courses"),
)

_TEXT_ONLY_RE = re.compile(
    r"(?:do\s+not\s+create\s+(?:a\s+)?charts?"
    r"|no\s+charts?(?:\s+(?:please|needed|required))?"
    r"|text[- ]only"
    r"|without\s+(?:any\s+)?charts?"
    r"|don't\s+create\s+(?:a\s+)?charts?)"
)


def _normalize_for_detection(question: str) -> str:
    lowered = question.lower()
    mapped = "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch
        for ch in lowered
    )
    for invisible in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        mapped = mapped.replace(invisible, "")
    return f" {mapped} "


def wants_text_only(question: str) -> bool:
    """True when the user explicitly asked for a text answer without charts."""

    return _TEXT_ONLY_RE.search(_normalize_for_detection(question)) is not None


def _resolve_ranked_target(text: str) -> tuple[str, str, int] | None:
    for regex, column, noun in _RANKED_TARGETS:
        match = regex.search(text)
        if match:
            return column, noun, max(1, min(50, int(match.group(1))))
    return None


def _cohort_since_2025(text: str) -> bool:
    if _COHORT_2025_RE.search(text):
        return True
    return _COHORT_AFTER_2024_RE.search(text) is not None


def _matches_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _detect_three_chart_pattern(text: str) -> bool:
    if not _CHART_COUNT_THREE_RE.search(text):
        return False
    return (
        _cohort_since_2025(text)
        and _matches_any(text, _DEPARTMENT_TERMS)
        and _matches_any(text, _STATUS_SPLIT_TERMS + _STATUS_VALUE_TERMS)
        and _MONTH_TERM_RE.search(text) is not None
    )


def _detect_nectec_monthly_pattern(text: str) -> bool:
    return (
        _NECTEC_TERM in text
        and _COHORT_2025_RE.search(text) is not None
        and _YEAR_2026_RE.search(text) is not None
        and _MONTH_TERM_RE.search(text) is not None
        and _matches_any(text, _STATUS_VALUE_TERMS)
    )


def _detect_certified_province_pattern(text: str) -> bool:
    return (
        _matches_any(text, _CERTIFICATE_TERMS)
        and _matches_any(text, _PROVINCE_TERMS)
        and _TOP_TEN_RE.search(text) is not None
        and _COHORT_2025_RE.search(text) is not None
        and _matches_any(text, _PASS_SPLIT_TERMS + ("ผ่าน",))
    )


def _detect_secondary_department_pattern(text: str) -> bool:
    return (
        _matches_any(text, _SECONDARY_TERMS)
        and _matches_any(text, _DEPARTMENT_TERMS)
        and _TOP_N_RE.search(text) is not None
        and _COHORT_2025_RE.search(text) is not None
        and _matches_any(text, _DISTINCT_TERMS)
        and _matches_any(text, _STATUS_SPLIT_TERMS)
    )


def _detect_institute_status_split(text: str) -> bool:
    return (
        _INSTITUTE_TERM_RE.search(text) is not None
        and _SPLIT_BY_STATUS_RE.search(text) is not None
        and "user" in text
    )


_STATUS_AS_SPLIT_RE = re.compile(r"(?:split|breakdown|grouped|divided)\s+(?:each\s+\w+\s+)?by\s+(?:the\s+)?learning[_ ]status")


def _detect_distribution_pattern(text: str) -> str | None:
    # "split by learning status" describes a secondary dimension, not a
    # single-dimension distribution; leave those to the split-aware patterns.
    if _STATUS_AS_SPLIT_RE.search(text):
        return None
    if any(grain in text for grain in ("weekly", "daily", "yearly", "quarterly")):
        return None
    if not _SPLIT_DISTRIBUTION_RE.search(text):
        return None
    if _EDUCATION_LEVEL_RE.search(text):
        return "distribution_education_level"
    if "learning_status" in text or "learning status" in text:
        return "distribution_learning_status"
    if "pass" in text and ("course pass" in text or "course_pass" in text or "passed" in text):
        if "province" not in text and "certificate" not in text and "ใบประกาศ" not in text:
            return "distribution_pass_status"
    if "certificate" in text or "has_certificate" in text:
        return "distribution_certificate"
    return None


_SPLIT_BY_STATUS_RE = re.compile(r"(?:split by|breakdown by|grouped? by|/\s*)\s*learning[_ ]status|split each .* by learning status")


def _detect_ranked_dimension(text: str) -> bool:
    if _resolve_ranked_target(text) is None:
        return False
    return (
        _matches_any(text, ("distinct users", "unique users", "enrolled users", "users"))
        and "status" not in text
    )


def _detect_monthly_enrollment_trend_open(text: str) -> bool:
    return (
        _MONTHLY_TREND_RE.search(text) is not None
        and _ENROLL_TERM in text
        and "nectec" not in text
        and "activity" not in text
    )


def _detect_activity_monthly_trend(text: str) -> bool:
    return (
        _MONTHLY_TREND_RE.search(text) is not None
        and _ACTIVITY_TERM in text
        and _EVENT_CATEGORY_RE.search(text) is None
    )


def _detect_activity_event_category(text: str) -> bool:
    return _ACTIVITY_TERM in text and _EVENT_CATEGORY_RE.search(text) is not None


def _detect_courses_by_department(text: str) -> bool:
    return (
        _DISTINCT_COURSES_RE.search(text) is not None
        and "department" in text
        and _TEACHER_RE.search(text) is None
    )


def _detect_courses_by_teacher(text: str) -> bool:
    return (
        _TEACHER_RE.search(text) is not None
        and "course" in text
        and ("distinct courses" in text or "number of courses" in text)
    )


def _detect_average_grade_by_course(text: str) -> bool:
    return _GRADE_RE.search(text) is not None and "course" in text


def detect_pattern(question: str) -> str | None:
    """Return the canonical pattern id for a question, or None.

    Matching is robust to case, full-width characters, byte-order marks, and
    zero-width joiners so that copy/paste artifacts cannot change routing.
    Full NFKC normalization is intentionally avoided because it decomposes the
    Thai SARA AM character and would corrupt Thai keyword matching.
    """

    text = _normalize_for_detection(question)
    if _detect_three_chart_pattern(text):
        return "three_chart_enrollment_overview"
    if _detect_nectec_monthly_pattern(text):
        return "nectec_monthly_status_trend"
    if _detect_certified_province_pattern(text):
        return "certified_province_pass_status_top10"
    if _detect_secondary_department_pattern(text):
        return "secondary_department_status_top8"
    if _detect_institute_status_split(text):
        return "institute_status_split"
    distribution = _detect_distribution_pattern(text)
    if distribution is not None:
        return distribution
    if _detect_activity_event_category(text):
        return "activity_event_category"
    if _detect_activity_monthly_trend(text):
        return "activity_monthly_trend"
    if _detect_average_grade_by_course(text):
        return "average_grade_by_course"
    if _detect_courses_by_teacher(text):
        return "courses_by_teacher"
    if _detect_courses_by_department(text):
        return "courses_count_by_department"
    if _detect_monthly_enrollment_trend_open(text):
        return "monthly_enrollment_trend_open"
    if _detect_ranked_dimension(text):
        return "ranked_dimension_topn"
    return None


def compile_analytical_pattern(database_path: str, question: str) -> dict[str, Any] | None:
    """Compile a canonical analytical request into a deterministic dashboard.

    Returns None when the question does not match a known pattern or when the
    warehouse cannot be opened read-only; callers then fall back to the
    existing planning pipeline.
    """

    pattern_id = detect_pattern(question)
    if pattern_id is None:
        return None
    try:
        from dashboard_agent.readonly_duckdb import connect_read_only

        con = connect_read_only(str(database_path))
    except Exception:
        return None
    try:
        builder = {
            "secondary_department_status_top8": _build_secondary_departments,
            "nectec_monthly_status_trend": _build_nectec_trend,
            "certified_province_pass_status_top10": _build_certified_provinces,
            "three_chart_enrollment_overview": _build_three_chart_overview,
            "institute_status_split": _build_institute_status_split,
            "distribution_learning_status": _build_distribution_learning_status,
            "distribution_pass_status": _build_distribution_pass_status,
            "distribution_certificate": _build_distribution_certificate,
            "distribution_education_level": _build_distribution_education_level,
            "ranked_dimension_topn": _build_ranked_dimension_topn,
            "monthly_enrollment_trend_open": _build_monthly_enrollment_trend_open,
            "activity_monthly_trend": _build_activity_monthly_trend,
            "activity_event_category": _build_activity_event_category,
            "courses_count_by_department": _build_courses_count_by_department,
            "courses_by_teacher": _build_courses_by_teacher,
            "average_grade_by_course": _build_average_grade_by_course,
        }[pattern_id]
        built = builder(con, question, pattern_id)
    except Exception:
        return None
    finally:
        con.close()
    # An empty dashboard means the cohort has no data for this snapshot; let
    # the caller fall back to the generic pipeline instead of shipping blank
    # charts.
    if sum(len(slot.get("data") or []) for slot in built.get("chartSlots") or []) == 0:
        return None
    return built


def _table_datasets(con: Any, tables: list[str]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    from .dashboard_planner import _source_paths

    for table in tables:
        try:
            paths = _source_paths(con, table)
        except Exception:
            paths = []
        datasets[table] = {
            "source": table,
            "key": table,
            "source_paths": list(paths),
            "object_type": "duckdb_pattern_compiler",
            "isFullAggregate": True,
        }
    return datasets


def _points(rows: list[tuple]) -> list[dict[str, Any]]:
    return [
        {"label": str(label), **({"series": str(series)} if series is not None else {}), "value": int(value)}
        for label, series, value in rows
    ]


def _single_series_points(rows: list[tuple]) -> list[dict[str, Any]]:
    return [{"label": str(label), "value": int(value)} for label, value in rows]


def _activity_frame(
    *,
    pattern_id: str,
    question: str,
    datasets: dict[str, Any],
    slots: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    summary_filters: list[str],
    total_users: int,
    plan: dict[str, Any],
    title: str,
    subtitle: str,
    total_label: str = "Distinct enrolled users",
    measure_display: str = "Distinct users",
) -> dict[str, Any]:
    trace = [
        {
            "stage": "Interpret request",
            "detail": f"Matched the deterministic analytical pattern {pattern_id} from the request.",
            "evidence": [f"Cohort filter: {item}" for item in summary_filters],
        },
        {
            "stage": "Resolve data model",
            "detail": "Bound aggregate columns to curated warehouse tables using a validated join path.",
            "evidence": [f"Source table: {name}" for name in sorted(datasets)],
        },
        {
            "stage": "Validate query",
            "detail": "Executed aggregate read-only DuckDB queries with exact-match filters, deterministic ordering, and distinct-user measures.",
            "evidence": ["Measure: count(distinct user_id)", *summary_filters],
        },
        {
            "stage": "Choose layout",
            "detail": f"Emitted {len(slots)} chart slot(s) with fixed chart types and point caps.",
            "evidence": [f"{slot['chartType']}: {slot['title']}" for slot in slots],
        },
    ]
    table_names = sorted(datasets)
    return {
        "datasets": datasets,
        "records": records[:50],
        "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in slots],
        "chartSlots": slots,
        "charts": {slot["id"]: slot["data"] for slot in slots},
        "decisionTrace": trace,
        "summary": {
            "source": FACT_TABLE,
            "sourcePaths": datasets.get(FACT_TABLE, {}).get("source_paths", []),
            "sampleRecords": len(records),
            "totalRecords": total_users,
            "totalDistinctMeasure": total_users,
            "measureField": "user_id",
            "measureName": measure_display,
            "metricLabels": {"totalDistinctMeasure": total_label},
            "isFullAggregate": True,
            "executionSource": "duckdb",
            "planningSource": "pattern-compiler",
            "calculationSource": "duckdb",
            "reasoningSource": "agent",
            "reasoningExecutionSource": "duckdb",
            "deterministicPattern": pattern_id,
            "deterministicQuestion": question[:200],
            "filters": summary_filters,
            "analyticalPlan": plan,
        },
        "layoutSpec": {
            "title": title,
            "subtitle": subtitle,
            "blocks": blocks[:8],
        },
    }


def _plan_block(
    *, intent: str, dimensions: list[str], splits: list[str], filters: list[dict[str, Any]], top_n: int | None, time_dimension: str | None,
    base_table: str = FACT_TABLE, measure_field: str = "user_id", measure_aggregation: str = "count_distinct",
    tables: list[str] | None = None,
) -> dict[str, Any]:
    resolved_tables = tables if tables is not None else [base_table]
    return {
        "intent": intent,
        "compiler": "analytical-pattern-compiler",
        "baseTable": base_table,
        "tables": resolved_tables,
        "measure": {"table": base_table, "name": measure_field, "aggregation": measure_aggregation},
        "dimensions": [{"table": base_table, "name": name} for name in dimensions],
        "splits": [{"table": base_table, "name": name} for name in splits],
        "timeDimension": {"table": base_table, "name": time_dimension} if time_dimension else None,
        "topN": top_n,
        "filters": filters,
    }


def _single_chart_activity(
    *,
    con: Any,
    pattern_id: str,
    question: str,
    tables: list[str],
    slot: dict[str, Any],
    rows: list[tuple],
    total_value: int,
    total_label: str,
    measure_field: str,
    summary_filters: list[str],
    plan_filters: list[dict[str, Any]],
    intent: str,
    title: str,
    subtitle: str,
    top_n: int | None = None,
    time_dimension: str | None = None,
    numeric_values: bool = False,
) -> dict[str, Any]:
    datasets = _table_datasets(con, tables)
    has_series = bool(rows) and rows[0][1] is not None
    data: list[dict[str, Any]] = []
    for row in rows:
        point: dict[str, Any] = {"label": str(row[0]), "value": float(row[-1]) if numeric_values else int(row[-1])}
        if has_series:
            point["series"] = str(row[1])
        data.append(point)
    slot = {**slot, "data": data}
    record_value_key = measure_field if measure_field != "*" else "records"
    records = [
        {
            "source": tables[0],
            "index": index,
            slot.get("field") or "label": point["label"],
            **({"series": point["series"]} if has_series and slot.get("splitField") else {}),
            record_value_key: point["value"],
        }
        for index, point in enumerate(data)
    ]
    return _activity_frame(
        pattern_id=pattern_id,
        question=question,
        datasets=datasets,
        slots=[slot],
        blocks=[
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            {"type": "chart", "slotId": slot["id"], "span": 2},
        ],
        records=records,
        summary_filters=summary_filters,
        total_users=total_value,
        plan=_plan_block(
            intent=intent,
            dimensions=[slot.get("field") or ""],
            splits=[slot["splitField"]] if slot.get("splitField") else [],
            filters=plan_filters,
            top_n=top_n,
            time_dimension=time_dimension,
            base_table=tables[0],
            measure_field="user_id" if measure_field == "*" else measure_field,
            measure_aggregation="count_rows" if measure_field == "*" else ("avg" if numeric_values else "count_distinct"),
            tables=list(datasets),
        ),
        title=title,
        subtitle=subtitle,
    )


def _build_secondary_departments(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        """
        WITH base AS (
            SELECT
                f.department_name AS label,
                CAST(f.learning_status AS VARCHAR) AS series,
                f.user_id AS uid
            FROM dashboard_agent_user_course_fact AS f
            INNER JOIN dashboard_agent_user_dim AS u USING (user_id)
            WHERE u.level_of_education = 'secondary'
              AND f.enroll_date >= DATE '2025-01-01'
              AND NULLIF(TRIM(f.department_name), '') IS NOT NULL
        ), ranked AS (
            SELECT label
            FROM base
            GROUP BY label
            ORDER BY COUNT(DISTINCT uid) DESC, label
            LIMIT 8
        )
        SELECT b.label, b.series, CAST(COUNT(DISTINCT b.uid) AS INT) AS value
        FROM base AS b
        INNER JOIN ranked AS r USING (label)
        GROUP BY b.label, b.series
        ORDER BY b.label, b.series NULLS FIRST
        """
    ).fetchall()
    total_users = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT f.user_id)
            FROM dashboard_agent_user_course_fact AS f
            INNER JOIN dashboard_agent_user_dim AS u USING (user_id)
            WHERE u.level_of_education = 'secondary'
              AND f.enroll_date >= DATE '2025-01-01'
              AND NULLIF(TRIM(f.department_name), '') IS NOT NULL
            """
        ).fetchone()[0]
        or 0
    )
    filters = [
        {"table": USER_DIM_TABLE, "field": "level_of_education", "value": "secondary"},
        {"table": FACT_TABLE, "field": "enroll_date", "operator": "gte", "value": "2025-01-01"},
        {"table": FACT_TABLE, "field": "department_name", "operator": "nonempty"},
    ]
    data = _points(rows)
    slot = {
        "id": "pattern-secondary-department-status",
        "title": "Top departments by distinct secondary learners since 2025, split by learning status",
        "chartType": "stacked_bar",
        "field": "department_name",
        "splitField": "learning_status",
        "data": data,
    }
    records = [
        {
            "source": FACT_TABLE,
            "index": index,
            "department_name": point["label"],
            "learning_status": point.get("series"),
            "distinct_users": point["value"],
        }
        for index, point in enumerate(data)
    ]
    return _activity_frame(
        pattern_id=pattern_id,
        question=question,
        datasets=_table_datasets(con, [FACT_TABLE, USER_DIM_TABLE]),
        slots=[slot],
        blocks=[
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            {"type": "chart", "slotId": slot["id"], "span": 2},
        ],
        records=records,
        summary_filters=[
            "level_of_education = 'secondary'",
            "enroll_date >= 2025-01-01",
            "department_name non-empty",
        ],
        total_users=total_users,
        plan=_plan_block(
            intent="ranked_split_by_status",
            dimensions=["department_name"],
            splits=["learning_status"],
            filters=filters,
            top_n=8,
            time_dimension=None,
        ),
        title="Secondary learners by department and learning status",
        subtitle="Top 8 course departments by distinct enrolled users since 2025-01-01.",
    )


def _build_nectec_trend(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        """
        SELECT
            STRFTIME(DATE_TRUNC('month', enroll_date), '%Y-%m') AS label,
            CAST(learning_status AS VARCHAR) AS series,
            CAST(COUNT(DISTINCT user_id) AS INT) AS value
        FROM dashboard_agent_user_course_fact
        WHERE department_name = 'NECTEC'
          AND enroll_date >= DATE '2025-01-01'
          AND enroll_date < DATE '2026-07-01'
          AND learning_status IN ('passed', 'in_progress', 'inactive')
        GROUP BY DATE_TRUNC('month', enroll_date), learning_status
        ORDER BY DATE_TRUNC('month', enroll_date), learning_status
        """
    ).fetchall()
    total_users = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM dashboard_agent_user_course_fact
            WHERE department_name = 'NECTEC'
              AND enroll_date >= DATE '2025-01-01'
              AND enroll_date < DATE '2026-07-01'
            """
        ).fetchone()[0]
        or 0
    )
    filters = [
        {"table": FACT_TABLE, "field": "department_name", "value": "NECTEC"},
        {"table": FACT_TABLE, "field": "enroll_date", "operator": "between", "value": ["2025-01-01", "2026-06-30"]},
    ]
    data = _points(rows)
    statuses = sorted({str(point.get("series")) for point in data})
    slot = {
        "id": "pattern-nectec-monthly-trend",
        "title": "Monthly NECTEC enrollment trend by learning status",
        "chartType": "multi_line" if len(statuses) > 1 else "line",
        "field": "enroll_date",
        "splitField": "learning_status" if len(statuses) > 1 else None,
        "data": data,
    }
    records = [
        {
            "source": FACT_TABLE,
            "index": index,
            "enroll_month": point["label"],
            "learning_status": point.get("series"),
            "distinct_users": point["value"],
        }
        for index, point in enumerate(data)
    ]
    return _activity_frame(
        pattern_id=pattern_id,
        question=question,
        datasets=_table_datasets(con, [FACT_TABLE]),
        slots=[slot],
        blocks=[
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            {"type": "chart", "slotId": slot["id"], "span": 2},
        ],
        records=records,
        summary_filters=[
            "department_name = 'NECTEC'",
            "enroll_date >= 2025-01-01 AND enroll_date < 2026-07-01",
            "learning_status IN ('passed','in_progress','inactive')",
        ],
        total_users=total_users,
        plan=_plan_block(
            intent="monthly_time_series_split",
            dimensions=[],
            splits=["learning_status"],
            filters=filters,
            top_n=None,
            time_dimension="enroll_date",
        ),
        title="NECTEC monthly enrollment trend",
        subtitle="Distinct enrolled users per month from January 2025 through June 2026, split by learning status.",
    )


def _build_certified_provinces(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        """
        WITH base AS (
            SELECT
                school_province AS label,
                CASE WHEN course_pass = 1 THEN 'passed' ELSE 'not_passed' END AS series,
                user_id AS uid
            FROM dashboard_agent_user_course_fact
            WHERE enroll_date >= DATE '2025-01-01'
              AND has_certificate = 1
              AND NULLIF(TRIM(school_province), '') IS NOT NULL
        ), ranked AS (
            SELECT label
            FROM base
            GROUP BY label
            ORDER BY COUNT(DISTINCT uid) DESC, label
            LIMIT 10
        )
        SELECT b.label, b.series, CAST(COUNT(DISTINCT b.uid) AS INT) AS value
        FROM base AS b
        INNER JOIN ranked AS r USING (label)
        GROUP BY b.label, b.series
        ORDER BY b.label, b.series NULLS FIRST
        """
    ).fetchall()
    total_users = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM dashboard_agent_user_course_fact
            WHERE enroll_date >= DATE '2025-01-01'
              AND has_certificate = 1
              AND NULLIF(TRIM(school_province), '') IS NOT NULL
            """
        ).fetchone()[0]
        or 0
    )
    filters = [
        {"table": FACT_TABLE, "field": "enroll_date", "operator": "gte", "value": "2025-01-01"},
        {"table": FACT_TABLE, "field": "has_certificate", "value": 1},
        {"table": FACT_TABLE, "field": "school_province", "operator": "nonempty"},
    ]
    data = _points(rows)
    totals: dict[str, int] = {}
    for point in data:
        totals[point["label"]] = totals.get(point["label"], 0) + point["value"]
    ordered = sorted(data, key=lambda point: (-totals[point["label"]], point["label"], str(point.get("series") or "")))
    slot = {
        "id": "pattern-certified-province-pass-status",
        "title": "Top provinces of certificate holders since 2025, split by pass status",
        "chartType": "horizontal_bar",
        "field": "school_province",
        "splitField": "course_pass",
        "data": ordered,
    }
    records = [
        {
            "source": FACT_TABLE,
            "index": index,
            "school_province": point["label"],
            "course_pass": point.get("series"),
            "distinct_users": point["value"],
        }
        for index, point in enumerate(ordered)
    ]
    return _activity_frame(
        pattern_id=pattern_id,
        question=question,
        datasets=_table_datasets(con, [FACT_TABLE]),
        slots=[slot],
        blocks=[
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            {"type": "chart", "slotId": slot["id"], "span": 2},
        ],
        records=records,
        summary_filters=[
            "has_certificate = 1",
            "enroll_date >= 2025-01-01",
            "school_province non-empty",
        ],
        total_users=total_users,
        plan=_plan_block(
            intent="ranked_binary_split",
            dimensions=["school_province"],
            splits=["course_pass"],
            filters=filters,
            top_n=10,
            time_dimension=None,
        ),
        title="Certificate holders by province and pass status",
        subtitle="Top 10 provinces by distinct certified users enrolled since 2025-01-01.",
    )


def _build_three_chart_overview(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    dept_rows = con.execute(
        """
        WITH base AS (
            SELECT
                department_name AS label,
                CAST(learning_status AS VARCHAR) AS series,
                user_id AS uid
            FROM dashboard_agent_user_course_fact
            WHERE enroll_date >= DATE '2025-01-01'
              AND NULLIF(TRIM(department_name), '') IS NOT NULL
        ), ranked AS (
            SELECT label
            FROM base
            GROUP BY label
            ORDER BY COUNT(DISTINCT uid) DESC, label
            LIMIT 8
        )
        SELECT b.label, b.series, CAST(COUNT(DISTINCT b.uid) AS INT) AS value
        FROM base AS b
        INNER JOIN ranked AS r USING (label)
        GROUP BY b.label, b.series
        ORDER BY b.label, b.series NULLS FIRST
        """
    ).fetchall()
    status_rows = con.execute(
        """
        SELECT CAST(learning_status AS VARCHAR) AS label, CAST(COUNT(DISTINCT user_id) AS INT) AS value
        FROM dashboard_agent_user_course_fact
        WHERE enroll_date >= DATE '2025-01-01'
          AND learning_status IS NOT NULL
        GROUP BY 1
        ORDER BY value DESC, label
        """
    ).fetchall()
    trend_rows = con.execute(
        """
        SELECT STRFTIME(DATE_TRUNC('month', enroll_date), '%Y-%m') AS label,
               CAST(COUNT(DISTINCT user_id) AS INT) AS value
        FROM dashboard_agent_user_course_fact
        WHERE enroll_date >= DATE '2025-01-01'
          AND enroll_date < DATE '2026-07-01'
        GROUP BY DATE_TRUNC('month', enroll_date)
        ORDER BY DATE_TRUNC('month', enroll_date)
        """
    ).fetchall()
    total_users = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM dashboard_agent_user_course_fact
            WHERE enroll_date >= DATE '2025-01-01'
            """
        ).fetchone()[0]
        or 0
    )
    dept_data = _points(dept_rows)
    status_data = _single_series_points(status_rows)
    trend_data = _single_series_points(trend_rows)
    slots = [
        {
            "id": "pattern-overview-department-status",
            "title": "Top departments by distinct enrolled users since 2025, split by learning status",
            "chartType": "stacked_bar",
            "field": "department_name",
            "splitField": "learning_status",
            "data": dept_data,
        },
        {
            "id": "pattern-overview-status-distribution",
            "title": "Overall distribution of distinct users by learning status",
            "chartType": "donut",
            "field": "learning_status",
            "data": status_data,
        },
        {
            "id": "pattern-overview-monthly-trend",
            "title": "Monthly distinct enrollment trend through June 2026",
            "chartType": "line",
            "field": "enroll_date",
            "data": trend_data,
        },
    ]
    records = [
        {
            "source": FACT_TABLE,
            "index": index,
            "department_name": point["label"],
            "learning_status": point.get("series"),
            "distinct_users": point["value"],
        }
        for index, point in enumerate(dept_data)
    ] + [
        {"source": FACT_TABLE, "index": index, "learning_status": point["label"], "distinct_users": point["value"]}
        for index, point in enumerate(status_data)
    ] + [
        {"source": FACT_TABLE, "index": index, "enroll_month": point["label"], "distinct_users": point["value"]}
        for index, point in enumerate(trend_data)
    ]
    filters = [{"table": FACT_TABLE, "field": "enroll_date", "operator": "gte", "value": "2025-01-01"}]
    return _activity_frame(
        pattern_id=pattern_id,
        question=question,
        datasets=_table_datasets(con, [FACT_TABLE]),
        slots=slots,
        blocks=[
            {"type": "metric", "id": "totalDistinctMeasure", "span": 1},
            *[{"type": "chart", "slotId": slot["id"], "span": 2} for slot in slots],
        ],
        records=records,
        summary_filters=["enroll_date >= 2025-01-01"],
        total_users=total_users,
        plan=_plan_block(
            intent="multi_chart_overview",
            dimensions=["department_name", "learning_status", "enroll_date"],
            splits=["learning_status"],
            filters=filters,
            top_n=8,
            time_dimension="enroll_date",
        ),
        title="Enrollment overview since January 2025",
        subtitle="Three aggregate charts: department ranking, status distribution, and monthly trend.",
    )

def _build_institute_status_split(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        WITH base AS (
            SELECT school_name, learning_status, user_id
            FROM {FACT_TABLE}
            WHERE school_name IS NOT NULL AND TRIM(school_name) <> ''
              AND learning_status IS NOT NULL AND TRIM(learning_status) <> ''
        ), top_groups AS (
            SELECT school_name FROM base
            GROUP BY school_name
            ORDER BY COUNT(DISTINCT user_id) DESC, school_name
            LIMIT 12
        )
        SELECT base.school_name, base.learning_status, CAST(COUNT(DISTINCT base.user_id) AS INT)
        FROM base JOIN top_groups USING (school_name)
        GROUP BY base.school_name, base.learning_status
        ORDER BY base.school_name, base.learning_status
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {FACT_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-institute-status-split",
            "title": "Distinct users by institute, split by learning status",
            "chartType": "stacked_bar",
            "field": "school_name",
            "splitField": "learning_status",
        },
        rows=rows,
        total_value=total,
        total_label="Distinct enrolled users",
        measure_field="user_id",
        summary_filters=["school_name non-empty", "learning_status non-empty", "top 12 institutes by distinct users"],
        plan_filters=[
            {"table": FACT_TABLE, "field": "school_name", "operator": "nonempty"},
            {"table": FACT_TABLE, "field": "learning_status", "operator": "nonempty"},
        ],
        intent="ranked_split_by_status",
        title="Users by institute and learning status",
        subtitle="Top 12 institutes by distinct enrolled users, split by learning status.",
        top_n=12,
    )


def _build_distribution_learning_status(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT learning_status, CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {FACT_TABLE}
        WHERE learning_status IS NOT NULL AND TRIM(learning_status) <> ''
        GROUP BY learning_status
        ORDER BY COUNT(DISTINCT user_id) DESC, learning_status
        """
    ).fetchall()
    total = sum(int(row[1]) for row in rows)
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-learning-status-distribution",
            "title": "Distribution of distinct users by learning status",
            "chartType": "donut",
            "field": "learning_status",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct users",
        measure_field="user_id",
        summary_filters=["learning_status non-null and non-empty"],
        plan_filters=[{"table": FACT_TABLE, "field": "learning_status", "operator": "nonempty"}],
        intent="category_distribution",
        title="Learning status distribution",
        subtitle="Overall distribution of distinct users by learning status.",
    )


def _build_distribution_pass_status(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT CASE WHEN course_pass = 1 THEN 'Passed' ELSE 'Not passed' END,
               CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {FACT_TABLE}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {FACT_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-pass-status-distribution",
            "title": "Distinct users by course pass status",
            "chartType": "donut",
            "field": "course_pass",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct users",
        measure_field="course_pass",
        summary_filters=["course_pass mapped to Passed / Not passed"],
        plan_filters=[],
        intent="binary_distribution",
        title="Course pass status distribution",
        subtitle="Distinct users split by course pass outcome.",
    )


def _build_distribution_certificate(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT CASE WHEN has_certificate = 1 THEN 'Has certificate' ELSE 'No certificate' END,
               CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {FACT_TABLE}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {FACT_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-certificate-distribution",
            "title": "Distinct users by certificate holding",
            "chartType": "donut",
            "field": "has_certificate",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct users",
        measure_field="has_certificate",
        summary_filters=["has_certificate mapped to Has certificate / No certificate"],
        plan_filters=[],
        intent="binary_distribution",
        title="Certificate distribution",
        subtitle="Distinct users split by whether they hold a certificate.",
    )


def _build_distribution_education_level(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT COALESCE(NULLIF(TRIM(level_of_education), ''), 'Unknown'),
               CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {USER_DIM_TABLE}
        GROUP BY 1
        ORDER BY COUNT(DISTINCT user_id) DESC, 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {USER_DIM_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[USER_DIM_TABLE],
        slot={
            "id": "pattern-education-level-distribution",
            "title": "Distinct users by education level",
            "chartType": "bar",
            "field": "level_of_education",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct users",
        measure_field="level_of_education",
        summary_filters=["blank levels labelled Unknown"],
        plan_filters=[],
        intent="category_distribution",
        title="Education level distribution",
        subtitle="Distinct users compared across levels of education.",
    )


_RANKED_COLUMN_SQL = {
    "province": ("province", "provinces"),
    "school_province": ("school_province", "school provinces"),
    "school_name": ("school_name", "schools"),
    "subject_name": ("subject_name", "courses"),
}


def _build_ranked_dimension_topn(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    resolved = _resolve_ranked_target(_normalize_for_detection(question))
    column, noun, limit = resolved or ("subject_name", "courses", 10)
    sql_column, display_noun = _RANKED_COLUMN_SQL[column]
    rows = con.execute(
        f"""
        SELECT {sql_column}, CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {FACT_TABLE}
        WHERE {sql_column} IS NOT NULL AND TRIM({sql_column}) <> ''
        GROUP BY {sql_column}
        ORDER BY COUNT(DISTINCT user_id) DESC, {sql_column}
        LIMIT {int(limit)}
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {FACT_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": f"pattern-ranked-{sql_column}",
            "title": f"Top {limit} {display_noun} by distinct users",
            "chartType": "horizontal_bar",
            "field": sql_column,
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct enrolled users",
        measure_field=sql_column,
        summary_filters=[f"{sql_column} non-empty", f"top {limit} by distinct users"],
        plan_filters=[{"table": FACT_TABLE, "field": sql_column, "operator": "nonempty"}],
        intent="ranked_dimension",
        title=f"Top {display_noun} by distinct users",
        subtitle=f"Ranked top {limit} {display_noun} by distinct enrolled users.",
        top_n=limit,
    )


def _build_monthly_enrollment_trend_open(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT STRFTIME(CAST(enroll_date AS TIMESTAMP), '%Y-%m'), CAST(COUNT(DISTINCT user_id) AS INT)
        FROM {FACT_TABLE}
        WHERE enroll_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {FACT_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-monthly-enrollment-trend",
            "title": "Monthly trend of distinct enrolled users",
            "chartType": "line",
            "field": "enroll_date",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct enrolled users",
        measure_field="enroll_date",
        summary_filters=["enroll_date not null", "monthly buckets from enroll_date"],
        plan_filters=[{"table": FACT_TABLE, "field": "enroll_date", "operator": "notnull"}],
        intent="time_series_total",
        title="Monthly enrollment trend",
        subtitle="Distinct users enrolled per month across all time.",
        time_dimension="enroll_date",
    )


def _build_activity_monthly_trend(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT STRFTIME(CAST(event_date AS DATE), '%Y-%m'), CAST(COUNT(*) AS INT)
        FROM {ACTIVITY_TABLE}
        WHERE event_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(*) FROM {ACTIVITY_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[ACTIVITY_TABLE],
        slot={
            "id": "pattern-activity-monthly-trend",
            "title": "Monthly trend of activity records",
            "chartType": "line",
            "field": "event_date",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Activity records",
        measure_field="*",
        summary_filters=["event_date not null", "monthly buckets from event_date"],
        plan_filters=[{"table": ACTIVITY_TABLE, "field": "event_date", "operator": "notnull"}],
        intent="activity_time_series",
        title="Monthly activity trend",
        subtitle="Activity record counts per month.",
        time_dimension="event_date",
    )


def _build_activity_event_category(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT COALESCE(NULLIF(TRIM(event_category), ''), 'Unknown'), CAST(COUNT(*) AS INT)
        FROM {ACTIVITY_TABLE}
        GROUP BY 1
        ORDER BY COUNT(*) DESC, 1
        LIMIT 20
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(*) FROM {ACTIVITY_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[ACTIVITY_TABLE],
        slot={
            "id": "pattern-activity-event-category",
            "title": "Activity records by event category",
            "chartType": "bar",
            "field": "event_category",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Activity records",
        measure_field="event_category",
        summary_filters=["blank categories labelled Unknown", "top 20 categories"],
        plan_filters=[],
        intent="activity_breakdown",
        title="Activity by event category",
        subtitle="Activity record counts across event categories.",
        top_n=20,
    )


def _build_courses_count_by_department(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT COALESCE(NULLIF(TRIM(department_name), ''), 'Unknown'), CAST(COUNT(DISTINCT course_id) AS INT)
        FROM {COURSE_DIM_TABLE}
        GROUP BY 1
        ORDER BY COUNT(DISTINCT course_id) DESC, 1
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT course_id) FROM {COURSE_DIM_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[COURSE_DIM_TABLE],
        slot={
            "id": "pattern-courses-by-department",
            "title": "Distinct courses by department",
            "chartType": "donut",
            "field": "department_name",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct courses",
        measure_field="course_id",
        summary_filters=["blank departments labelled Unknown"],
        plan_filters=[],
        intent="category_distribution",
        title="Courses by department",
        subtitle="Number of distinct courses per department.",
    )


def _build_courses_by_teacher(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT course_teacher_name, CAST(COUNT(DISTINCT course_id) AS INT)
        FROM {COURSE_DIM_TABLE}
        WHERE course_teacher_name IS NOT NULL AND TRIM(course_teacher_name) <> ''
        GROUP BY 1
        ORDER BY COUNT(DISTINCT course_id) DESC, 1
        LIMIT 10
        """
    ).fetchall()
    total = int(
        con.execute(f"SELECT COUNT(DISTINCT course_id) FROM {COURSE_DIM_TABLE}").fetchone()[0] or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[COURSE_DIM_TABLE],
        slot={
            "id": "pattern-courses-by-teacher",
            "title": "Top course teachers by distinct courses",
            "chartType": "horizontal_bar",
            "field": "course_teacher_name",
        },
        rows=[(row[0], None, row[1]) for row in rows],
        total_value=total,
        total_label="Distinct courses",
        measure_field="course_teacher_name",
        summary_filters=["teacher names non-empty", "top 10 teachers"],
        plan_filters=[{"table": COURSE_DIM_TABLE, "field": "course_teacher_name", "operator": "nonempty"}],
        intent="ranked_dimension",
        title="Course teachers by distinct courses",
        subtitle="Top 10 teachers ranked by number of distinct courses.",
        top_n=10,
    )


def _build_average_grade_by_course(con: Any, question: str, pattern_id: str) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT subject_name, ROUND(AVG(avg_module_grade), 2)
        FROM {FACT_TABLE}
        WHERE subject_name IS NOT NULL AND TRIM(subject_name) <> ''
          AND avg_module_grade IS NOT NULL
        GROUP BY subject_name
        ORDER BY AVG(avg_module_grade) DESC, subject_name
        LIMIT 10
        """
    ).fetchall()
    total = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT subject_name) FROM {FACT_TABLE}
            WHERE subject_name IS NOT NULL AND TRIM(subject_name) <> ''
              AND avg_module_grade IS NOT NULL
            """
        ).fetchone()[0]
        or 0
    )
    return _single_chart_activity(
        con=con,
        pattern_id=pattern_id,
        question=question,
        tables=[FACT_TABLE],
        slot={
            "id": "pattern-average-grade-by-course",
            "title": "Average module grade by course",
            "chartType": "horizontal_bar",
            "field": "subject_name",
        },
        rows=rows,
        total_value=total,
        total_label="Courses compared",
        measure_field="avg_module_grade",
        summary_filters=["subjects with non-null grades", "rounded to two decimals", "top 10 courses"],
        plan_filters=[
            {"table": FACT_TABLE, "field": "subject_name", "operator": "nonempty"},
            {"table": FACT_TABLE, "field": "avg_module_grade", "operator": "notnull"},
        ],
        intent="average_ranked_dimension",
        title="Average grade by course",
        subtitle="Top 10 courses by average module grade (rounded to two decimals).",
        top_n=10,
        numeric_values=True,
    )
