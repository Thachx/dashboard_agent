from __future__ import annotations

import calendar
import datetime as dt
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_TERMS = {
    "a",
    "all",
    "and",
    "as",
    "at",
    "by",
    "can",
    "compare",
    "comparison",
    "dashboard",
    "distribution",
    "for",
    "from",
    "give",
    "have",
    "how",
    "in",
    "me",
    "most",
    "number",
    "of",
    "over",
    "show",
    "split",
    "the",
    "time",
    "to",
    "top",
    "trend",
    "what",
    "which",
    "with",
}
ID_TERMS = {"id", "ids", "identifier", "identifiers"}
SEMANTIC_ALIASES = {
    "institute": {"institution", "school", "organization"},
    "institution": {"institute", "school", "organization"},
    "school": {"institute", "institution", "organization"},
    "student": {"user", "learner"},
    "learner": {"user", "student"},
    "user": {"student", "learner"},
    "finish": {"finished", "complete", "completed", "passed", "status"},
    "finished": {"finish", "complete", "completed", "passed", "status"},
    "complete": {"completed", "finish", "finished", "passed", "status"},
    "completed": {"complete", "finish", "finished", "passed", "status"},
    "pass": {"passed", "complete", "completed", "finish", "finished", "status"},
    "passed": {"pass", "complete", "completed", "finish", "finished", "status"},
    "status": {"state", "result"},
    "province": {"region", "location"},
    "enroll": {"enrolled", "enrollment"},
    "enrolled": {"enroll", "enrollment"},
    "enrollment": {"enroll", "enrolled"},
    "certificate": {"certified"},
    "certified": {"certificate"},
}
DIMENSION_ALIASES = {
    key: value
    for key, value in SEMANTIC_ALIASES.items()
    if key in {"institute", "institution", "school", "student", "learner", "user", "province"}
}
DIMENSION_ALIASES.update({
    "pass": {"passed", "not_passed", "status"},
    "status": {"state", "result", "pass", "passed"},
})
MEASURE_HINTS = {"user", "users", "student", "students", "learner", "learners", "course", "courses"}
TIME_HINTS = {"time", "timeline", "trend", "daily", "weekly", "monthly", "yearly", "date"}
NON_DIMENSION_PARTS = {
    "created",
    "updated",
    "modified",
    "payload",
    "record",
    "source",
    "timestamp",
}


@dataclass(frozen=True)
class ColumnProfile:
    table: str
    name: str
    data_type: str
    terms: frozenset[str]
    is_identifier: bool
    is_time: bool


@dataclass
class TableProfile:
    name: str
    columns: dict[str, ColumnProfile]
    row_count: int = 0
    source_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JoinStep:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    confidence: float


@dataclass(frozen=True)
class FilterSpec:
    column: ColumnProfile
    value: str
    confidence: float
    operator: str = "eq"


@dataclass
class AnalyticalPlan:
    intent: str
    base_table: str
    measure: ColumnProfile
    dimensions: list[ColumnProfile]
    time_dimension: ColumnProfile | None
    joins: list[JoinStep]
    requested_terms: list[str]
    filters: list[FilterSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    graph_hints: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "baseTable": self.base_table,
            "measure": _column_dict(self.measure),
            "dimensions": [_column_dict(column) for column in self.dimensions],
            "timeDimension": _column_dict(self.time_dimension) if self.time_dimension else None,
            "joins": [asdict(join) for join in self.joins],
            "filters": [
                {
                    "table": item.column.table,
                    "field": item.column.name,
                    "value": item.value,
                    "confidence": item.confidence,
                    **({"operator": item.operator} if item.operator != "eq" else {}),
                }
                for item in self.filters
            ],
            "requestedTerms": self.requested_terms,
            "warnings": self.warnings,
            "graphHints": self.graph_hints,
        }


def build_complex_dashboard(
    database_path: str | Path,
    question: str,
    *,
    graph_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan and execute a validated multi-dimension or multi-table dashboard."""

    try:
        import duckdb
    except ImportError:
        return {}
    path = Path(database_path)
    if not path.exists():
        return {}
    try:
        from dashboard_agent.readonly_duckdb import connect_read_only

        con = connect_read_only(str(path))
    except Exception:
        return {}
    try:
        catalog = _catalog(con)
        plan = plan_complex_dashboard(
            catalog,
            question,
            value_validator=lambda column: _column_has_value(con, column),
            dimension_profiler=lambda column: _column_value_profile(con, column),
            graph_hints=graph_hints,
        )
        if plan is None:
            return {}
        series_dimension = (
            _time_split_dimension(con, plan.dimensions, question)
            if plan.time_dimension is not None
            else None
        )
        value_filters = _resolve_value_filters(con, catalog, plan, question)
        filter_tables = {item.column.table for item in value_filters}
        current_targets = {
            plan.measure.table,
            *(column.table for column in plan.dimensions),
            *filter_tables,
        }
        if plan.time_dimension is not None:
            current_targets.add(plan.time_dimension.table)
        plan.joins = _resolve_join_tree(catalog, plan.base_table, current_targets)
        reachable_after_filters = {
            plan.base_table,
            *(join.left_table for join in plan.joins),
            *(join.right_table for join in plan.joins),
        }
        value_filters = [
            item for item in value_filters if item.column.table in reachable_after_filters
        ]
        if series_dimension is not None and not _has_explicit_field_filter(question, series_dimension.name):
            series_key = (series_dimension.table, series_dimension.name)
            requested_series_values = _requested_series_values(con, series_dimension, question)
            # Named series values describe the categories to compare. They are
            # not a request to filter either the selected dimension or another
            # field that happens to contain the same category label.
            value_filters = [
                item
                for item in value_filters
                if (item.column.table, item.column.name) != series_key
                and _normalized_category_value(item.value) not in requested_series_values
            ]
        plan.filters = [
            *value_filters,
            *_resolve_temporal_filters(catalog, plan, question),
        ]
        filter_value_counts: dict[tuple[str, str], set[str]] = {}
        for item in plan.filters:
            if item.operator == "eq":
                filter_value_counts.setdefault((item.column.table, item.column.name), set()).add(item.value)
        filtered_columns = {
            key for key, values in filter_value_counts.items() if len(values) == 1
        }
        plan.dimensions = [
            column
            for column in plan.dimensions
            if (column.table, column.name) not in filtered_columns
        ]
        plan.dimensions = _order_dimensions(plan.dimensions, question)
        return _execute_plan(con, plan, question)
    except Exception:
        return {}
    finally:
        con.close()


def plan_complex_dashboard(
    catalog: dict[str, TableProfile],
    question: str,
    *,
    value_validator: Callable[[ColumnProfile], bool] | None = None,
    dimension_profiler: Callable[[ColumnProfile], tuple[int, int]] | None = None,
    graph_hints: dict[str, Any] | None = None,
) -> AnalyticalPlan | None:
    excluded_terms = _excluded_query_terms(question)
    requested_source_terms = _raw_terms(question)
    source_domain_exclusions: set[str] = set()
    if requested_source_terms & {"enroll", "enrolled", "enrollment", "registration", "registered"}:
        if not requested_source_terms & {"activity", "activities", "event", "events", "session", "sessions", "log", "logs"}:
            source_domain_exclusions.update({"activity", "event", "session", "log"})
    eligible_catalog = {
        name: table
        for name, table in catalog.items()
        if _semantic_overlap(_terms(name), excluded_terms) == 0
        and not (_terms(name) & source_domain_exclusions)
    }
    if eligible_catalog:
        catalog = eligible_catalog
    query_terms = _expanded_terms(question) - excluded_terms
    raw_query_terms = _raw_terms(question) - excluded_terms
    requested_ids = bool(query_terms & ID_TERMS)
    graph_hints = graph_hints or {}
    measure = _resolve_measure(
        catalog,
        query_terms,
        _measure_query_terms(question),
        value_validator=value_validator,
        graph_hints=graph_hints,
    )
    if measure is None:
        return None

    # Treat monthly cadence language as an analytical time request, even when
    # it uses the singular unit ("each month").  A bare
    # date range remains a filter; it only becomes a time chart when the
    # question also asks for a cadence or temporal comparison.
    time_analysis_terms = TIME_HINTS - {"date"} | {"month"}
    requests_date_dimension = bool(
        re.search(r"\b(?:by|per|across|over)\s+(?:(?:calendar|each)\s+)?(?:[a-z0-9_]*date|day|week|month|year)\b", question.lower())
    )
    requests_temporal_cadence = bool(
        re.search(
            r"\b(?:(?:each|every|per)\s+month|monthly)\b",
            question.lower(),
        )
        or re.search(r"\bmonth\s+by\s+month\b|\bover\s+time\b", question.lower())
    )
    time_dimension = (
        _resolve_time(catalog, query_terms, value_validator=value_validator, graph_hints=graph_hints)
        if (query_terms & time_analysis_terms and requests_temporal_cadence)
        or requests_date_dimension
        or bool(query_terms & {"time", "timeline", "trend", "daily", "weekly", "monthly", "yearly"})
        else None
    )
    dimensions = _resolve_dimensions(
        catalog,
        query_terms,
        raw_query_terms=raw_query_terms,
        measure=measure,
        time_dimension=time_dimension,
        allow_identifiers=requested_ids,
        value_validator=value_validator,
        dimension_profiler=dimension_profiler,
        graph_hints=graph_hints,
    )
    neutral_overview = not _has_explicit_analytical_shape(question)
    if not dimensions and time_dimension is None and neutral_overview:
        dimensions = _resolve_overview_dimensions(
            catalog,
            measure=measure,
            value_validator=value_validator,
            dimension_profiler=dimension_profiler,
            graph_hints=graph_hints,
        )
    complex_language = any(
        phrase in f" {question.lower()} "
        for phrase in (" and ", " split by ", " split into ", " grouped by ", " group by ", " compare ", " within ", " for each ")
    )
    if len(dimensions) < 2 and not complex_language and not neutral_overview:
        return None
    if not dimensions and time_dimension is None:
        return None

    required = [measure, *dimensions]
    if time_dimension is not None:
        required.append(time_dimension)
    base_table = _select_base_table(
        catalog,
        required,
        query_terms=query_terms,
        excluded_terms=excluded_terms,
        graph_hints=graph_hints,
    )
    base_columns = catalog[base_table].columns
    measure = base_columns.get(measure.name, measure)
    dimensions = [base_columns.get(column.name, column) for column in dimensions]
    dimensions = list({(column.table, column.name): column for column in dimensions}.values())
    if time_dimension is not None:
        time_dimension = base_columns.get(time_dimension.name, time_dimension)
    required = [measure, *dimensions]
    if time_dimension is not None:
        required.append(time_dimension)
    joins = _resolve_join_tree(catalog, base_table, {column.table for column in required})
    reachable = {base_table, *(join.right_table for join in joins), *(join.left_table for join in joins)}
    dimensions = [column for column in dimensions if column.table in reachable]
    if time_dimension is not None and time_dimension.table not in reachable:
        time_dimension = None
    if measure.table not in reachable:
        return None

    unresolved_tables = sorted({column.table for column in required} - reachable)
    warnings = []
    if unresolved_tables:
        warnings.append("No validated join path was found for: " + ", ".join(unresolved_tables))
    if not dimensions and time_dimension is None:
        return None
    intent = (
        "neutral_overview"
        if neutral_overview
        else ("time_comparison" if time_dimension else "multi_dimension_comparison")
    )
    return AnalyticalPlan(
        intent=intent,
        base_table=base_table,
        measure=measure,
        dimensions=dimensions[:4],
        time_dimension=time_dimension,
        joins=joins,
        requested_terms=sorted(query_terms - STOP_TERMS),
        warnings=warnings,
        graph_hints={
            "tables": list(graph_hints.get("tables") or [])[:12],
            "fields": list(graph_hints.get("fields") or [])[:20],
            "sourcePaths": list(graph_hints.get("sourcePaths") or [])[:12],
            "evidence": list(graph_hints.get("evidence") or [])[:12],
        },
    )


def _catalog(con: Any) -> dict[str, TableProfile]:
    rows = con.execute(
        """
        select table_name
        from information_schema.tables
        where table_schema = 'main'
          and table_name like 'dashboard_agent_%'
        order by table_name
        """
    ).fetchall()
    profiles: dict[str, TableProfile] = {}
    for (raw_table,) in rows:
        table = str(raw_table)
        if any(part in table.lower() for part in ("cache", "lineage", "map")):
            continue
        column_rows = con.execute(f"pragma table_info({_quote(table)})").fetchall()
        columns: dict[str, ColumnProfile] = {}
        for row in column_rows:
            name = str(row[1])
            data_type = str(row[2])
            terms = frozenset(_terms(name))
            columns[name] = ColumnProfile(
                table=table,
                name=name,
                data_type=data_type,
                terms=terms,
                is_identifier=_is_identifier(name),
                is_time=_is_time_column(name, data_type),
            )
        profiles[table] = TableProfile(name=table, columns=columns, source_paths=_source_paths(con, table))
    return profiles


def _resolve_measure(
    catalog: dict[str, TableProfile],
    query_terms: set[str],
    measure_terms: set[str],
    *,
    value_validator: Callable[[ColumnProfile], bool] | None,
    graph_hints: dict[str, Any],
) -> ColumnProfile | None:
    requested = measure_terms or (query_terms & MEASURE_HINTS)
    candidates: list[tuple[float, ColumnProfile]] = []
    for table in catalog.values():
        for column in table.columns.values():
            if not column.is_identifier:
                continue
            # An explicit measure phrase (for example, "unique learners") is
            # stronger evidence than incidental table or graph-hint overlap.
            # Restrict the candidate set to identifiers that actually express
            # that measure, while retaining the broader scoring fallback for
            # requests that do not name one.
            if measure_terms and _semantic_overlap(column.terms, measure_terms) == 0:
                continue
            score = _semantic_overlap(column.terms, requested or query_terms) * 12.0
            direct_identity = (set(column.terms) - ID_TERMS) & requested
            score += len(direct_identity) * 20.0
            score += _semantic_overlap(_terms(table.name), query_terms) * 3.0
            score += _graph_column_hint_score(column, graph_hints)
            if column.name.lower() in {"user_id", "student_id", "learner_id", "course_id"}:
                score += 4.0
            if "fact" in table.name or "joined" in table.name:
                score += 2.0
            if score > 0:
                candidates.append((score, column))
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    return next(
        (
            column
            for _score, column in candidates
            if value_validator is None or value_validator(column)
        ),
        None,
    )


def _resolve_time(
    catalog: dict[str, TableProfile],
    query_terms: set[str],
    *,
    value_validator: Callable[[ColumnProfile], bool] | None,
    graph_hints: dict[str, Any],
) -> ColumnProfile | None:
    candidates: list[tuple[float, ColumnProfile]] = []
    for table in catalog.values():
        for column in table.columns.values():
            if not column.is_time:
                continue
            score = _semantic_overlap(column.terms, query_terms) * 8.0
            score += _semantic_overlap(_terms(table.name), query_terms) * 3.0
            score += _graph_column_hint_score(column, graph_hints)
            if "activity" in column.terms or "complete" in column.terms or "completed" in column.terms:
                score += 3.0
            candidates.append((score, column))
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    return next(
        (
            column
            for _score, column in candidates
            if value_validator is None or value_validator(column)
        ),
        None,
    )


def _resolve_dimensions(
    catalog: dict[str, TableProfile],
    query_terms: set[str],
    *,
    raw_query_terms: set[str],
    measure: ColumnProfile,
    time_dimension: ColumnProfile | None,
    allow_identifiers: bool,
    value_validator: Callable[[ColumnProfile], bool] | None,
    dimension_profiler: Callable[[ColumnProfile], tuple[int, int]] | None,
    graph_hints: dict[str, Any],
) -> list[ColumnProfile]:
    candidates: list[tuple[float, ColumnProfile]] = []
    for table in catalog.values():
        for column in table.columns.values():
            if column == measure or column == time_dimension or column.is_time:
                continue
            if _is_numeric_type(column.data_type) and not column.is_identifier:
                concept = _dimension_concept(column.name)
                explicit_binary_concept = concept in raw_query_terms or (
                    concept == "pass" and bool(raw_query_terms & {"passed", "not_passed"})
                ) or (
                    concept == "certificate" and bool(raw_query_terms & {"certificate", "certified"})
                )
                # Binary flags require direct outcome language. A generic word
                # such as "status" must not pull every semantically related
                # 0/1 column into the grouping plan.
                if not explicit_binary_concept:
                    continue
                if not (_raw_terms(column.name) & {"status", "pass", "certificate", "flag", "type", "category"}):
                    continue
                if dimension_profiler is not None:
                    sampled_rows, distinct_values = dimension_profiler(column)
                    if sampled_rows < 1 or distinct_values > 2:
                        continue
            if column.is_identifier and not allow_identifiers:
                continue
            if column.is_identifier:
                identity_terms = set(column.terms) - ID_TERMS
                measure_identity = set(measure.terms) - ID_TERMS
                if identity_terms & measure_identity:
                    continue
                if not identity_terms & raw_query_terms:
                    continue
            if column.terms & NON_DIMENSION_PARTS:
                continue
            concept = _dimension_concept(column.name)
            concept_matches = concept in raw_query_terms or bool(DIMENSION_ALIASES.get(concept, set()) & raw_query_terms)
            reverse_alias_match = any(concept in DIMENSION_ALIASES.get(term, set()) for term in raw_query_terms)
            if concept and not concept_matches and not reverse_alias_match:
                continue
            overlap = _semantic_overlap(column.terms, query_terms)
            if overlap <= 0:
                continue
            direct_overlap = len(_raw_terms(column.name) & raw_query_terms)
            specificity = max(len(_raw_terms(column.name) - {"id", "name"}), 1)
            score = overlap * 6.0 + direct_overlap * 14.0 + (4.0 / specificity)
            normalized_column = " ".join(TOKEN_RE.findall(column.name.lower().replace("_", " ")))
            if normalized_column and set(normalized_column.split()) <= raw_query_terms:
                score += len(normalized_column.split()) * 18.0
            score += _semantic_overlap(_terms(table.name), query_terms) * 2.0
            score += _graph_column_hint_score(column, graph_hints)
            if "name" in column.terms and column.terms & query_terms:
                score += 4.0
            if "fact" in table.name or "joined" in table.name:
                score += 1.0
            candidates.append((score, column))
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    selected: list[ColumnProfile] = []
    covered_terms: set[str] = set()
    for _, column in candidates:
        if value_validator is not None and not value_validator(column):
            continue
        concept = _dimension_concept(column.name)
        concepts = (_raw_terms(column.name) & raw_query_terms) - {"id", "ids", "name"}
        if not concepts and concept and concept in raw_query_terms:
            concepts = {concept}
        if concepts and concepts <= covered_terms:
            continue
        selected.append(column)
        covered_terms.update(concepts)
        if len(selected) >= 4:
            break
    return selected


def _resolve_overview_dimensions(
    catalog: dict[str, TableProfile],
    *,
    measure: ColumnProfile,
    value_validator: Callable[[ColumnProfile], bool] | None,
    dimension_profiler: Callable[[ColumnProfile], tuple[int, int]] | None,
    graph_hints: dict[str, Any],
) -> list[ColumnProfile]:
    preferred_fields = {str(value).lower() for value in graph_hints.get("fields") or [] if value}
    candidates: list[tuple[float, ColumnProfile]] = []
    for table in catalog.values():
        for column in table.columns.values():
            if column == measure or column.is_identifier or column.is_time:
                continue
            if _is_numeric_type(column.data_type) or column.terms & NON_DIMENSION_PARTS:
                continue
            if value_validator is not None and not value_validator(column):
                continue
            score = _graph_column_hint_score(column, graph_hints)
            if column.table == measure.table:
                score += 24.0
            if "fact" in column.table or "joined" in column.table:
                score += 3.0
            candidates.append((score, column))

    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    chartable: list[tuple[float, ColumnProfile]] = []
    for score, column in candidates[:32]:
        if dimension_profiler is None:
            chartable.append((score, column))
            continue
        sampled_rows, distinct_values = dimension_profiler(column)
        if sampled_rows < 2 or distinct_values < 2:
            continue
        max_distinct = 2000 if column.name.lower() in preferred_fields else 100
        if distinct_values > max_distinct or distinct_values / sampled_rows > 0.8:
            continue
        cardinality_score = max(0.0, 8.0 - distinct_values / 12.5)
        chartable.append((score + cardinality_score, column))

    chartable.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    return [column for _, column in chartable[:3]]


def _select_base_table(
    catalog: dict[str, TableProfile],
    required: list[ColumnProfile],
    *,
    query_terms: set[str],
    excluded_terms: set[str],
    graph_hints: dict[str, Any],
) -> str:
    required_names = {column.name for column in required}
    scores: list[tuple[float, str]] = []
    for table in catalog.values():
        coverage = len(required_names & set(table.columns))
        score = float(coverage * 20)
        if any(column.table == table.name for column in required):
            score += 5.0
        if "fact" in table.name or "joined" in table.name:
            score += 3.0
        table_terms = _terms(table.name)
        score += _semantic_overlap(table_terms, query_terms) * 4.0
        score -= _semantic_overlap(table_terms, excluded_terms) * 30.0
        score += sum(6.0 for column in required if column.table == table.name)
        score += _graph_table_hint_score(table.name, graph_hints)
        scores.append((score, table.name))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def _graph_table_hint_score(table: str, graph_hints: dict[str, Any]) -> float:
    preferred = {str(value).lower() for value in graph_hints.get("tables") or [] if value}
    if table.lower() in preferred:
        return 18.0
    table_terms = _terms(table)
    return min(
        8.0,
        max(
            (
                float(_semantic_overlap(table_terms, _terms(value)) * 4)
                for value in preferred
            ),
            default=0.0,
        ),
    )


def _graph_column_hint_score(column: ColumnProfile, graph_hints: dict[str, Any]) -> float:
    preferred_fields = {str(value).lower() for value in graph_hints.get("fields") or [] if value}
    exact = column.name.lower() in preferred_fields
    score = 14.0 if exact else 0.0
    if not exact:
        score += min(
            6.0,
            max(
                (
                    float(_semantic_overlap(column.terms, _terms(value)) * 3)
                    for value in preferred_fields
                ),
                default=0.0,
            ),
        )
    return score + _graph_table_hint_score(column.table, graph_hints)


def _resolve_join_tree(
    catalog: dict[str, TableProfile],
    base_table: str,
    target_tables: set[str],
) -> list[JoinStep]:
    adjacency: dict[str, list[JoinStep]] = {table: [] for table in catalog}
    names = sorted(catalog)
    for index, left_name in enumerate(names):
        left = catalog[left_name]
        for right_name in names[index + 1 :]:
            right = catalog[right_name]
            matches = _join_key_candidates(left, right)
            if not matches:
                continue
            left_key, right_key, confidence = matches[0]
            adjacency[left_name].append(JoinStep(left_name, right_name, left_key, right_key, confidence))
            adjacency[right_name].append(JoinStep(right_name, left_name, right_key, left_key, confidence))

    result: list[JoinStep] = []
    connected = {base_table}
    for target in sorted(target_tables - connected):
        path = _shortest_join_path(adjacency, connected, target)
        if not path:
            continue
        for step in path:
            if step.right_table not in connected:
                result.append(step)
                connected.add(step.right_table)
    return result


def _join_key_candidates(left: TableProfile, right: TableProfile) -> list[tuple[str, str, float]]:
    candidates: list[tuple[str, str, float]] = []
    for left_column in left.columns.values():
        if not left_column.is_identifier:
            continue
        for right_column in right.columns.values():
            if not right_column.is_identifier or not _compatible_types(left_column.data_type, right_column.data_type):
                continue
            if left_column.name.lower() == right_column.name.lower():
                confidence = 0.98
            else:
                overlap = _semantic_overlap(left_column.terms - {"id"}, right_column.terms - {"id"})
                if overlap <= 0:
                    continue
                confidence = min(0.9, 0.68 + overlap * 0.08)
            candidates.append((left_column.name, right_column.name, confidence))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    return candidates


def _shortest_join_path(
    adjacency: dict[str, list[JoinStep]],
    starts: set[str],
    target: str,
) -> list[JoinStep]:
    queue = deque((start, []) for start in starts)
    visited = set(starts)
    while queue:
        table, path = queue.popleft()
        for step in sorted(adjacency.get(table, []), key=lambda item: (-item.confidence, item.right_table)):
            if step.right_table in visited:
                continue
            next_path = [*path, step]
            if step.right_table == target:
                return next_path
            visited.add(step.right_table)
            queue.append((step.right_table, next_path))
    return []


def _execute_plan(con: Any, plan: AnalyticalPlan, question: str) -> dict[str, Any]:
    aliases, from_sql = _from_sql(plan)
    measure_sql = _column_sql(plan.measure, aliases)
    filter_sql, filter_params = _filter_sql(plan, aliases)
    slots: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    used_dimensions: set[tuple[str, str]] = set()
    total_measure = int(
        con.execute(
            f"select count(distinct {measure_sql}) {from_sql} where {measure_sql} is not null{filter_sql}",
            filter_params,
        ).fetchone()[0]
        or 0
    )
    blocks.append({"type": "metric", "id": "totalDistinctMeasure", "span": 1})

    if plan.time_dimension is not None:
        time_sql = _column_sql(plan.time_dimension, aliases)
        group = _time_split_dimension(con, plan.dimensions, question)
        if group is not None:
            group_sql = _column_sql(group, aliases)
            rows = con.execute(
                f"""
                select date_trunc('month', {time_sql}) as bucket,
                       cast({group_sql} as varchar) as series,
                       count(distinct {measure_sql}) as value
                {from_sql}
                where {time_sql} is not null and {group_sql} is not null and {measure_sql} is not null{filter_sql}
                group by 1, 2
                qualify dense_rank() over (order by count(distinct {measure_sql}) desc) <= 60
                order by 1, 2
                """
            , filter_params).fetchall()
            totals: dict[str, int] = {}
            for _bucket, series, value in rows:
                totals[str(series)] = totals.get(str(series), 0) + int(value)
            selected_series = {name for name, _value in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:6]}
            data = [
                {"label": _format_time_bucket(bucket), "series": _series_label(group, series, question), "value": int(value)}
                for bucket, series, value in rows
                if str(series) in selected_series
            ]
            slot_id = "plan-grouped-time"
            slots.append({
                "id": slot_id,
                "title": f"{_measure_display(plan.measure.name)} over time by {_display(group.name)}",
                "chartType": "multi_line",
                "field": plan.time_dimension.name,
                "splitField": group.name,
                "reason": f"Multi-series time comparison grouped by {_display(group.name)} from the validated analytical plan.",
                "data": data,
            })
            blocks.append({"type": "chart", "slotId": slot_id, "span": 2})
            used_dimensions.add((group.table, group.name))
        else:
            rows = con.execute(
                f"""
                select date_trunc('month', {time_sql}) as bucket, count(distinct {measure_sql}) as value
                {from_sql}
                where {time_sql} is not null and {measure_sql} is not null{filter_sql}
                group by 1 order by 1
                """
            , filter_params).fetchall()
            data = [{"label": _format_time_bucket(label), "value": int(value)} for label, value in rows]
            slots.append({"id": "plan-time", "title": f"{_measure_display(plan.measure.name)} over time", "chartType": "line", "field": plan.time_dimension.name, "data": data})
            blocks.append({"type": "chart", "slotId": "plan-time", "span": 2})

    if len(plan.dimensions) >= 2 and _requests_category_split(question):
        primary, split = plan.dimensions[:2]
        primary_sql = _column_sql(primary, aliases)
        split_sql = _column_sql(split, aliases)
        top_limit = _requested_top_limit(question)
        top_rows = con.execute(
            f"""
            select cast({primary_sql} as varchar) as label, count(distinct {measure_sql}) as value
            {from_sql}
            where {primary_sql} is not null and cast({primary_sql} as varchar) <> ''
              and {measure_sql} is not null{filter_sql}
            group by 1 order by value desc, label limit {top_limit}
            """,
            filter_params,
        ).fetchall()
        top_labels = [str(label) for label, _value in top_rows]
        if top_labels:
            placeholders = ", ".join("?" for _ in top_labels)
            rows = con.execute(
                f"""
                select cast({primary_sql} as varchar) as label,
                       cast({split_sql} as varchar) as series,
                       count(distinct {measure_sql}) as value
                {from_sql}
                where {primary_sql} is not null and cast({primary_sql} as varchar) <> ''
                  and {split_sql} is not null and {measure_sql} is not null{filter_sql}
                  and cast({primary_sql} as varchar) in ({placeholders})
                group by 1, 2
                """,
                [*filter_params, *top_labels],
            ).fetchall()
            rank = {label: index for index, label in enumerate(top_labels)}
            rows.sort(key=lambda row: (rank.get(str(row[0]), len(rank)), str(row[1])))
            data = [
                {
                    "label": str(label),
                    "series": _series_label(split, series, question),
                    "value": int(value),
                }
                for label, series, value in rows
            ]
            slot_id = f"plan-{_slug(primary.name)}-by-{_slug(split.name)}"
            slots.append({
                "id": slot_id,
                "title": f"{_measure_display(plan.measure.name)} by {_display(primary.name)} and {_display(split.name)}",
                "chartType": "stacked_bar",
                "field": primary.name,
                "splitField": split.name,
                "reason": f"Ranked {_display(primary.name)} and preserved the requested {_display(split.name)} split.",
                "data": data,
            })
            blocks.append({"type": "chart", "slotId": slot_id, "span": 2})
            records.extend(
                {
                    "source": primary.table,
                    "index": row_index,
                    primary.name: label,
                    split.name: series,
                    plan.measure.name: value,
                }
                for row_index, (label, series, value) in enumerate(rows)
            )
            used_dimensions.update({(primary.table, primary.name), (split.table, split.name)})

    used_chart_types = {str(slot.get("chartType") or "") for slot in slots}
    for index, dimension in enumerate(plan.dimensions):
        if (
            (dimension.table, dimension.name) in used_dimensions
            and not _requests_standalone_dimension(question, dimension)
        ):
            continue
        dimension_sql = _column_sql(dimension, aliases)
        rows = con.execute(
            f"""
            select cast({dimension_sql} as varchar) as label, count(distinct {measure_sql}) as value
            {from_sql}
            where {dimension_sql} is not null and cast({dimension_sql} as varchar) <> '' and {measure_sql} is not null{filter_sql}
            group by 1 order by value desc, label limit 12
            """
        , filter_params).fetchall()
        if not rows:
            continue
        data = [{"label": str(label), "value": int(value)} for label, value in rows]
        slot_id = f"plan-{_slug(dimension.name)}"
        chart_preferences = _categorical_chart_preferences(question, dimension.name, len(data))
        chart_type = chart_preferences[0]
        if not _requested_categorical_chart_type(question):
            chart_type = next(
                (candidate for candidate in chart_preferences if candidate not in used_chart_types),
                chart_type,
            )
        used_chart_types.add(chart_type)
        slots.append({
            "id": slot_id,
            "title": f"{_measure_display(plan.measure.name)} by {_display(dimension.name)}",
            "chartType": chart_type,
            "field": dimension.name,
            "reason": "Dimension and measure were bound from the validated analytical plan.",
            "data": data,
        })
        blocks.append({"type": "chart", "slotId": slot_id, "span": 2})
        records.extend({"source": dimension.table, "index": row_index, dimension.name: label, plan.measure.name: value} for row_index, (label, value) in enumerate(rows))

    if not slots:
        return {}
    _apply_requested_chart_contract(plan, slots, blocks, question)
    _apply_plan_presentation_to_slots(plan, slots)
    table_names = {plan.base_table, *(join.left_table for join in plan.joins), *(join.right_table for join in plan.joins)}
    source_paths = _dedupe(path for table in table_names for path in _source_paths(con, table))
    presentation = _plan_presentation_spec(plan, slots, question)
    trace = [
        {
            "stage": "Interpret request",
            "detail": f"Created a {plan.intent.replace('_', ' ')} analytical plan from the requested measures and dimensions.",
            "evidence": [f"Measure: {_measure_display(plan.measure.name)}", *[f"Dimension: {_display(item.name)}" for item in plan.dimensions]],
        },
        {
            "stage": "Resolve data model",
            "detail": f"Selected {plan.base_table} as the base table and resolved {len(plan.joins)} validated join step(s).",
            "evidence": [f"{step.left_table}.{step.left_key} = {step.right_table}.{step.right_key} ({step.confidence:.0%})" for step in plan.joins] or ["All requested fields are available in one table."],
        },
        {
            "stage": "Validate query",
            "detail": "Verified field ownership, compatible join-key types, and distinct-measure aggregation before building charts.",
            "evidence": [
                *([f"Filter: {_display(item.column.name)} = {item.value} ({item.confidence:.0%})" for item in plan.filters]),
                *(plan.warnings or ["No unresolved fields or join paths."]),
            ],
        },
        {
            "stage": "Choose layout",
            "detail": f"Built {len(slots)} chart(s) from the result shapes requested by the prompt.",
            "evidence": [f"{slot['chartType']}: {slot['title']}" for slot in slots],
        },
    ]
    summary = {
        "source": plan.base_table,
        "sourcePaths": source_paths,
        "sampleRecords": len(records),
        "totalRecords": total_measure,
        "isFullAggregate": True,
        "measureField": plan.measure.name,
        "measureName": _measure_display(plan.measure.name),
        "totalDistinctMeasure": total_measure,
        "filters": [f"{_display(item.column.name)} = {item.value}" for item in plan.filters],
        "metricLabels": {"totalDistinctMeasure": f"Distinct {_measure_display(plan.measure.name)}"},
        "analyticalPlan": plan.public_dict(),
    }
    return {
        "datasets": {
            table: {
                "source": table,
                "key": table,
                "source_paths": _source_paths(con, table),
                "object_type": "duckdb_join_plan",
                "totalRecords": total_measure,
                "isFullAggregate": True,
            }
            for table in sorted(table_names)
        },
        "records": records[:50],
        "chartPlan": [{key: value for key, value in slot.items() if key != "data"} for slot in slots],
        "chartSlots": slots,
        "charts": {slot["id"]: slot["data"] for slot in slots},
        "decisionTrace": trace,
        "summary": summary,
        "layoutSpec": {
            "title": presentation["title"],
            "subtitle": presentation["subtitle"],
            "blocks": blocks[:8],
        },
    }


def _plan_presentation_spec(
    plan: AnalyticalPlan,
    slots: list[dict[str, Any]],
    question: str,
) -> dict[str, str]:
    """Create concise, factual presentation text from the executed plan.

    This deliberately uses only field-level plan metadata.  It never includes a
    result label, source sample, identifier, or raw row value, so titles and
    descriptions remain useful without exposing personal data.
    """

    measure = _measure_display(plan.measure.name)
    primary = plan.dimensions[0] if plan.dimensions else plan.time_dimension
    title = f"{measure} by {_display(primary.name)}" if primary else measure
    terms = _raw_terms(question)
    details: list[str] = []

    if plan.time_dimension is not None:
        cadence = next((value for value in ("daily", "weekly", "monthly", "yearly") if value in terms), None)
        details.append(f"Shows {cadence + ' ' if cadence else ''}distinct {measure.lower()} over time")
    elif primary is not None:
        details.append(f"Shows distinct {measure.lower()} grouped by {_display(primary.name)}")
    else:
        details.append(f"Shows distinct {measure.lower()} from the validated analytical plan")

    split = next((slot.get("splitField") for slot in slots if isinstance(slot.get("splitField"), str)), None)
    if split:
        details.append(f"split by {_display(split)}")
    if "top" in terms or "rank" in terms or "ranked" in terms:
        details.append("ranked by the requested measure")

    filter_text = _presentation_filters(plan.filters)
    if filter_text:
        details.append(f"filtered to {filter_text}")

    subtitle = "; ".join(details).strip() + "."
    return {"title": title, "subtitle": subtitle[:220]}


def _apply_plan_presentation_to_slots(plan: AnalyticalPlan, slots: list[dict[str, Any]]) -> None:
    """Give each chart a verified, plan-backed description for UI consumers."""

    measure = _measure_display(plan.measure.name).lower()
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        field = str(slot.get("field") or "").strip()
        split = str(slot.get("splitField") or "").strip()
        chart_type = str(slot.get("chartType") or "chart").replace("_", " ")
        if field:
            description = f"Displays distinct {measure} by {_display(field)}"
        else:
            description = f"Displays distinct {measure}"
        if "time" in chart_type:
            description += " over time"
        if split:
            description += f" split by {_display(split)}"
        slot["description"] = description + "."
        # Keep the existing reason as a concise implementation note, but do not
        # let it become the user-facing data description.


def _presentation_filters(filters: list[FilterSpec]) -> str:
    """Render filter fields without leaking values from source records."""

    rendered: list[str] = []
    for item in filters[:3]:
        field = _display(item.column.name)
        value = str(item.value).strip()
        # Dates are analytic constraints, not source-record content.  All
        # other raw values stay out of presentation text to avoid PII leaks.
        safe_date = value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else "the requested date"
        if item.operator == "gte":
            rendered.append(f"{field} from {safe_date}")
        elif item.operator == "lte":
            rendered.append(f"{field} through {safe_date}")
        else:
            rendered.append(field)
    return ", ".join(rendered)


def _requested_chart_count(question: str) -> int | None:
    lowered = question.lower()
    contract_match = re.search(r"\bchart[_ ]count\s*[=:]\s*(\d{1,2})\b", lowered)
    if contract_match:
        return min(max(int(contract_match.group(1)), 1), 8)
    match = re.search(r"\b(?:exactly\s+)?(\d{1,2})\s+(?:charts?|views?|visuals?|กราฟ|แผนภูมิ)", lowered)
    if match:
        return min(max(int(match.group(1)), 1), 8)
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\b(?:exactly\s+)?(" + "|".join(words) + r")\s+(?:charts?|views?|visuals?)\b", lowered)
    return words[match.group(1)] if match else None


def _apply_requested_chart_contract(
    plan: AnalyticalPlan,
    slots: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    question: str,
    *,
    max_total_points: int = 50,
) -> None:
    """Honor an explicit multi-chart count and keep its views renderable."""

    requested_count = _requested_chart_count(question)
    if requested_count is None or requested_count <= 1:
        return

    split_slots = [
        slot for slot in slots
        if slot.get("splitField") and str(slot.get("chartType")) not in {"line", "multi_line"}
    ]
    standalone_fields = {
        dimension.name
        for dimension in plan.dimensions
        if _requests_standalone_dimension(question, dimension)
    }
    standalone_slots = [
        slot for slot in slots
        if not slot.get("splitField") and str(slot.get("field") or "") in standalone_fields
    ]
    split_fields = {str(slot.get("splitField") or "") for slot in split_slots}
    standalone_slots.sort(
        key=lambda slot: (
            0 if str(slot.get("field") or "") in split_fields else 1,
            str(slot.get("id") or ""),
        )
    )
    time_slots = [
        slot for slot in slots
        if str(slot.get("chartType")) in {"line", "multi_line", "area"}
        or (plan.time_dimension is not None and slot.get("field") == plan.time_dimension.name)
    ]
    ordered = [*split_slots, *standalone_slots, *time_slots, *slots]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot in ordered:
        slot_id = str(slot.get("id") or "")
        if not slot_id or slot_id in seen:
            continue
        selected.append(slot)
        seen.add(slot_id)
        if len(selected) >= requested_count:
            break

    # A multi-series time chart can satisfy both a trend and a requested
    # overall distribution. Derive the missing aggregate view from its own
    # aggregate rows instead of inventing or re-querying prompt-specific data.
    if len(selected) < requested_count and time_slots:
        grouped_time = next((slot for slot in time_slots if slot.get("splitField")), None)
        if grouped_time is not None:
            totals: dict[str, int] = {}
            for row in grouped_time.get("data") or []:
                series = str(row.get("series") or "")
                if series:
                    totals[series] = totals.get(series, 0) + int(row.get("value") or 0)
            if totals:
                field = str(grouped_time.get("splitField") or "series")
                derived = {
                    "id": f"plan-{_slug(field)}-overall",
                    "title": f"{_measure_display(plan.measure.name)} by {_display(field)}",
                    "chartType": "donut",
                    "field": field,
                    "reason": "Overall distribution aggregated from the validated time-series result.",
                    "data": [
                        {"label": label, "value": value}
                        for label, value in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
                    ],
                }
                selected.insert(max(len(selected) - 1, 0), derived)
                seen.add(derived["id"])
                blocks.append({"type": "chart", "slotId": derived["id"], "span": 2})

    remaining = max_total_points
    for slot in selected:
        data = slot.get("data")
        if not isinstance(data, list):
            continue
        keep = max(0, min(len(data), remaining))
        slot["data"] = data[:keep]
        remaining -= keep
    slots[:] = selected
    blocks[:] = [block for block in blocks if block.get("slotId") in seen or block.get("type") != "chart"]


def _requested_top_limit(question: str, default: int = 12) -> int:
    match = re.search(r"\btop\s+(\d{1,3})\b", question, flags=re.IGNORECASE)
    if match:
        return min(max(int(match.group(1)), 1), 50)
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    match = re.search(
        r"\b(?:top\s+)?(" + "|".join(word_numbers) + r")\b",
        question,
        flags=re.IGNORECASE,
    )
    return word_numbers[match.group(1).lower()] if match else default


def _analytical_clauses(question: str) -> list[str]:
    return [
        clause.strip(" ,:")
        for clause in re.split(r"[.;]|\b(?:first|second|third|fourth)\s*,?", question.lower())
        if clause.strip(" ,:")
    ]


_SERIES_INTENT_TERMS = {"split", "separate", "separately", "series", "line", "lines", "each"}


def _time_split_dimension(
    con: Any,
    dimensions: list[ColumnProfile],
    question: str,
) -> ColumnProfile | None:
    """Bind a categorical series to a time view from wording and value evidence.

    A request can say "separate lines" or name desired categories without the
    literal phrase "split by".  Candidate fields are ranked from their semantic
    overlap and sampled values that occur in the request, rather than from a
    prompt-specific mapping.
    """

    time_terms = {"time", "trend", "monthly", "daily", "weekly", "yearly", "month", "date"}
    for clause in _analytical_clauses(question):
        raw_clause_terms = _raw_terms(clause)
        if not (raw_clause_terms & time_terms) or not (raw_clause_terms & _SERIES_INTENT_TERMS):
            continue
        split_match = re.search(r"\bsplit(?:\s+each\s+[a-z0-9 _-]+)?\s+(?:by|into)\s+([^,.;]+)", clause)
        requested_terms = _terms(split_match.group(1)) if split_match else _terms(clause)
        candidates: list[tuple[float, ColumnProfile]] = []
        for column in dimensions:
            direct_semantic_score = float(
                len((_raw_terms(column.name) - STOP_TERMS) & raw_clause_terms)
            )
            alias_semantic_score = float(_semantic_overlap(column.terms, requested_terms))
            value_score = _series_value_evidence_score(con, column, raw_clause_terms)
            if direct_semantic_score <= 0 and alias_semantic_score <= 0 and value_score <= 0:
                continue
            candidates.append(
                (direct_semantic_score * 20.0 + alias_semantic_score * 2.0 + value_score, column)
            )
        candidates.sort(key=lambda item: (-item[0], item[1].name))
        if candidates and candidates[0][0] > 0:
            return candidates[0][1]
    return None


def _series_value_evidence_score(con: Any, column: ColumnProfile, question_terms: set[str]) -> float:
    """Return evidence that request tokens name categorical field values."""

    ignored = STOP_TERMS | TIME_HINTS | _SERIES_INTENT_TERMS | {"show"}
    requested = question_terms - ignored - _raw_terms(column.name)
    if not requested:
        return 0.0
    try:
        rows = con.execute(
            f"select distinct cast({_quote(column.name)} as varchar) from {_quote(column.table)} "
            f"where {_quote(column.name)} is not null and cast({_quote(column.name)} as varchar) <> '' limit 80"
        ).fetchall()
    except Exception:
        return 0.0
    score = 0.0
    values = {str(value).strip().lower() for (value,) in rows if value is not None}
    boolean_like = bool(values) and values <= {"0", "1", "true", "false"} and len(values) >= 2
    if boolean_like and _binary_series_kind(column) is not None:
        score += 12.0
    for (value,) in rows:
        raw_overlap = (_raw_terms(str(value)) - STOP_TERMS) & question_terms
        if raw_overlap:
            # Exact observed category names are stronger evidence than aliases
            # inferred from a boolean field's schema name.
            score += float(len(raw_overlap)) * 16.0
        else:
            score += float(len(_terms(str(value)) & requested)) * 4.0
    return score


def _normalized_category_value(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value).lower().replace("_", " ")))


def _requested_series_values(con: Any, column: ColumnProfile, question: str) -> set[str]:
    """Return observed or semantic category labels explicitly named as series."""

    normalized_question = _normalized_category_value(question)
    requested: set[str] = set()
    try:
        rows = con.execute(
            f"select distinct cast({_quote(column.name)} as varchar) from {_quote(column.table)} "
            f"where {_quote(column.name)} is not null and cast({_quote(column.name)} as varchar) <> '' limit 80"
        ).fetchall()
    except Exception:
        rows = []
    for (value,) in rows:
        normalized = _normalized_category_value(value)
        if normalized and re.search(rf"\b{re.escape(normalized)}\b", normalized_question):
            requested.add(normalized)
    kind = _binary_series_kind(column)
    if kind == "pass" and re.search(r"\b(?:pass|passed|not passed)\b", normalized_question):
        requested.update({"passed", "not passed"})
    elif kind == "certificate" and re.search(r"\b(?:certificate|certified|not certified)\b", normalized_question):
        requested.update({"certified", "not certified"})
    return requested


def _requests_standalone_dimension(question: str, dimension: ColumnProfile) -> bool:
    """Detect an explicitly separate categorical view for a reused split field."""

    view_terms = {"distribution", "breakdown", "composition", "overall"}
    return (_requested_chart_count(question) or 0) >= 3 or any(
        bool(_raw_terms(clause) & view_terms)
        and _semantic_overlap(dimension.terms, _terms(clause)) > 0
        for clause in _analytical_clauses(question)
    )


def _format_time_bucket(value: Any) -> str:
    text = str(value)
    return text[:7] if re.match(r"^\d{4}-\d{2}", text) else text


def _series_label(column: ColumnProfile, value: Any, question: str) -> str:
    text = str(value)
    kind = _binary_series_kind(column)
    if kind == "pass" and text.lower() in {"0", "1", "false", "true"}:
        return "passed" if text.lower() in {"1", "true"} else "not_passed"
    if kind == "certificate" and text.lower() in {"0", "1", "false", "true"}:
        return "certified" if text.lower() in {"1", "true"} else "not_certified"
    return text


def _binary_series_kind(column: ColumnProfile) -> str | None:
    """Classify only schema-semantic boolean outcome fields for labels."""

    terms = _raw_terms(column.name)
    if terms & {"pass", "passed", "outcome", "result", "complete", "completed", "finish", "finished"}:
        return "pass"
    if terms & {"certificate", "certified"}:
        return "certificate"
    return None


def _from_sql(plan: AnalyticalPlan) -> tuple[dict[str, str], str]:
    aliases = {plan.base_table: "t0"}
    sql = f"from {_quote(plan.base_table)} t0"
    pending = list(plan.joins)
    while pending:
        progressed = False
        for step in list(pending):
            if step.left_table not in aliases:
                continue
            alias = f"t{len(aliases)}"
            aliases[step.right_table] = alias
            sql += (
                f" left join {_quote(step.right_table)} {alias}"
                f" on {aliases[step.left_table]}.{_quote(step.left_key)} = {alias}.{_quote(step.right_key)}"
            )
            pending.remove(step)
            progressed = True
        if not progressed:
            break
    return aliases, sql


def _column_sql(column: ColumnProfile, aliases: dict[str, str]) -> str:
    alias = aliases.get(column.table, "t0")
    return f"{alias}.{_quote(column.name)}"


def _source_paths(con: Any, table: str) -> list[str]:
    try:
        exists = con.execute(
            "select count(*) from information_schema.tables where table_schema='main' and table_name='dashboard_agent_table_lineage'"
        ).fetchone()[0]
        if exists:
            rows = con.execute(
                "select distinct source_path from dashboard_agent_table_lineage where table_name=? and source_path is not null order by source_path",
                [table],
            ).fetchall()
            paths = _dedupe(_original_source_path(str(row[0])) for row in rows if row[0])
            if paths:
                return paths
        has_summary = con.execute(
            "select count(*) from information_schema.tables where table_schema='main' and table_name='source_summary'"
        ).fetchone()[0]
        if not has_summary:
            return []
        columns = [str(row[1]) for row in con.execute(f"pragma table_info({_quote(table)})").fetchall()]
        target_terms = _terms(" ".join([table, *columns])) - STOP_TERMS
        rows = con.execute(
            "select source_path, source_table, records from source_summary where records > 0 order by records desc, source_path"
        ).fetchall()
        scored: list[tuple[int, int, str]] = []
        for source_path, source_table, records in rows:
            path = _original_source_path(str(source_path))
            lowered = path.lower()
            if any(part in lowered for part in ("dashboard_agent", "unified_records", "_warehouse")):
                continue
            score = _semantic_overlap(_terms(f"{path} {source_table or ''}"), target_terms)
            if score > 0:
                scored.append((score, int(records or 0), path))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return _dedupe(item[2] for item in scored[:12])
    except Exception:
        return []


def _resolve_value_filters(
    con: Any,
    catalog: dict[str, TableProfile],
    plan: AnalyticalPlan,
    question: str,
) -> list[FilterSpec]:
    excluded_terms = _excluded_query_terms(question)
    query_terms = _expanded_terms(question) - STOP_TERMS - ID_TERMS - excluded_terms
    raw_query_terms = _raw_terms(question) - STOP_TERMS - ID_TERMS - excluded_terms
    lowered_question = question.lower().replace("_", " ")
    split_match = re.search(r"\bsplit(?:\s+each\s+[a-z0-9 _-]+)?\s+(?:by|into)\s+([^.;]+)", lowered_question)
    split_terms = _terms(split_match.group(1)) if split_match else set()
    split_raw_terms = _raw_terms(split_match.group(1)) if split_match else set()
    reachable = {plan.base_table, *(join.left_table for join in plan.joins), *(join.right_table for join in plan.joins)}
    # Value-only filters may live on a dimension table that was not required by
    # the initial measure/grouping plan. Inspect only additional tables whose
    # schema terms match the request; selected filter tables are joined later.
    candidate_tables = set(reachable)
    for table_name, table in catalog.items():
        if any(
            not column.is_identifier
            and not column.is_time
            and _semantic_overlap(column.terms, query_terms) > 0
            for column in table.columns.values()
        ):
            candidate_tables.add(table_name)
    excluded = {(plan.measure.table, plan.measure.name)}
    candidates: list[FilterSpec] = []
    for table_name in sorted(candidate_tables):
        for column in catalog[table_name].columns.values():
            if (column.table, column.name) in excluded or column.is_identifier or column.is_time:
                continue
            if column.terms & NON_DIMENSION_PARTS:
                continue
            column_overlap = _semantic_overlap(column.terms, query_terms)
            meaningful_column_overlap = (
                set(column.terms) - {"id", "name", "value", "date", "status"}
            ) & query_terms
            if _is_numeric_type(column.data_type):
                binary_value = _requested_binary_value(column, question)
                if binary_value is not None:
                    candidates.append(FilterSpec(column=column, value=binary_value, confidence=0.95))
                continue
            try:
                rows = con.execute(
                    f"select distinct cast({_quote(column.name)} as varchar) from {_quote(column.table)} "
                    f"where {_quote(column.name)} is not null and cast({_quote(column.name)} as varchar) <> '' limit 40"
                ).fetchall()
            except Exception:
                continue
            for (raw_value,) in rows:
                value = str(raw_value)
                if len(value) > 160 or value.lstrip().startswith(("{", "[")):
                    continue
                value_terms = _terms(value) - STOP_TERMS
                overlap = (_raw_terms(value) - STOP_TERMS) & raw_query_terms
                normalized_value = " ".join(TOKEN_RE.findall(value.lower().replace("_", " ")))
                exact_phrase = bool(normalized_value and normalized_value in lowered_question)
                if not overlap or not any(not term.isdigit() for term in overlap):
                    continue
                if column_overlap <= 0 and not exact_phrase:
                    continue
                if not exact_phrase and not meaningful_column_overlap:
                    continue
                if value_terms & split_terms and len(_raw_terms(column.name) & split_raw_terms) < 2:
                    continue
                confidence = min(0.99, 0.72 + len(overlap) * 0.09 + (0.15 if exact_phrase else 0.0))
                candidates.append(FilterSpec(column=column, value=value, confidence=confidence))
    candidates.sort(
        key=lambda item: (
            -item.confidence,
            item.column.table not in reachable,
            item.column.table,
            item.column.name,
            item.value,
        )
    )
    selected: list[FilterSpec] = []
    used_values: set[tuple[str, str, str]] = set()
    claimed_exact_values: set[str] = set()
    claimed_exact_terms: set[str] = set()
    for item in candidates:
        key = (item.column.table, item.column.name, item.value.lower())
        if key in used_values:
            continue
        normalized_value = " ".join(TOKEN_RE.findall(item.value.lower().replace("_", " ")))
        is_exact = bool(normalized_value and normalized_value in lowered_question)
        value_terms = _terms(item.value) - STOP_TERMS
        if is_exact and normalized_value in claimed_exact_values:
            continue
        if not is_exact and value_terms & claimed_exact_terms:
            continue
        selected.append(item)
        used_values.add(key)
        if is_exact:
            claimed_exact_values.add(normalized_value)
            claimed_exact_terms.update(value_terms)
        if len(selected) >= 8:
            break
    return selected


def _requested_binary_value(column: ColumnProfile, question: str) -> str | None:
    """Resolve explicit positive predicates for low-cardinality has_* fields."""

    lowered = question.lower().replace("_", " ")
    name = column.name.lower()
    if not name.startswith("has_"):
        return None
    concept = name.removeprefix("has_").replace("_", " ")
    concept_terms = _terms(concept)
    query_terms = _expanded_terms(question) - _excluded_query_terms(question)
    if not concept_terms & query_terms:
        return None
    if re.search(rf"\b(?:has|have|having|with)\b[^,.]{{0,30}}\b{re.escape(concept)}s?\b", lowered):
        return "1"
    if concept == "certificate" and "certified" in query_terms:
        return "1"
    return None


MONTH_NUMBERS = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
MONTH_NUMBERS.update(
    {name.lower(): index for index, name in enumerate(calendar.month_abbr) if name}
)
DATE_TEXT = (
    r"(?:\d{4}-\d{2}-\d{2}"
    r"|(?:\d{1,2}\s+)?[A-Za-z]+\s+\d{4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
)


def _has_explicit_temporal_filter(question: str) -> bool:
    """Whether a date bound explicitly names its temporal field.

    A bare ``since 2025-01-01`` remains compatible with the compact split
    executor. A request such as ``filter enroll_date from 2025-01-01`` needs
    the complex planner so the named field and operator remain traceable.
    """

    lowered = question.lower()
    temporal_field = r"[a-z][a-z0-9_]*(?:date|time|_at)"
    return bool(
        re.search(
            rf"\b{temporal_field}\s+(?:from|since|after|on\s+or\s+after)\s+{DATE_TEXT}",
            lowered,
        )
    )


def _has_explicit_field_filter(question: str, field_name: str) -> bool:
    """Return true when filter syntax explicitly constrains a named field."""

    normalized_question = question.lower().replace("_", " ")
    normalized_field = " ".join(TOKEN_RE.findall(field_name.lower().replace("_", " ")))
    if not normalized_field:
        return False
    field = re.escape(normalized_field)
    return bool(
        re.search(rf"\b(?:where|filter(?:ed)?(?:\s+to)?)\b[^.;]*\b{field}\b", normalized_question)
        or re.search(rf"\b{field}\b\s+(?:is|equals?|=|in)\b", normalized_question)
    )


def _parse_date_text(value: str) -> tuple[dt.date, str] | None:
    value = " ".join(value.strip().replace(",", "").split())
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return dt.date.fromisoformat(value), "day"
        parts = value.lower().split()
        if len(parts) == 2 and parts[0] in MONTH_NUMBERS:
            return dt.date(int(parts[1]), MONTH_NUMBERS[parts[0]], 1), "month"
        if len(parts) == 3 and parts[1] in MONTH_NUMBERS:
            return dt.date(int(parts[2]), MONTH_NUMBERS[parts[1]], int(parts[0])), "day"
        if len(parts) == 3 and parts[0] in MONTH_NUMBERS:
            return dt.date(int(parts[2]), MONTH_NUMBERS[parts[0]], int(parts[1])), "day"
    except (TypeError, ValueError):
        return None
    return None


def _next_month(value: dt.date) -> dt.date:
    return dt.date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def _resolve_temporal_filters(
    catalog: dict[str, TableProfile],
    plan: AnalyticalPlan,
    question: str,
) -> list[FilterSpec]:
    """Translate explicit English date bounds into typed planner filters."""

    lowered = question.lower()
    range_match = re.search(rf"\bfrom\s+({DATE_TEXT})\s+(?:through|to|until)\s+({DATE_TEXT})", lowered)
    lower_match = re.search(
        rf"\b(?:(?:since|on or after|after)\s+|from\s+)({DATE_TEXT})(?:\s+onward)?",
        lowered,
    )
    if not range_match and not lower_match:
        return []

    reachable = {plan.base_table, *(join.left_table for join in plan.joins), *(join.right_table for join in plan.joins)}
    query_terms = _expanded_terms(question) - _excluded_query_terms(question)
    candidates: list[tuple[float, ColumnProfile]] = []
    normalized_question = " ".join(TOKEN_RE.findall(lowered.replace("_", " ")))
    for table_name in reachable:
        for column in catalog[table_name].columns.values():
            if not column.is_time:
                continue
            normalized_name = " ".join(TOKEN_RE.findall(column.name.lower().replace("_", " ")))
            score = _semantic_overlap(column.terms, query_terms) * 10.0
            if normalized_name and normalized_name in normalized_question:
                score += 30.0
            score += _semantic_overlap(_terms(table_name), query_terms) * 2.0
            if score > 0:
                candidates.append((score, column))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    column = candidates[0][1]

    filters: list[FilterSpec] = []
    if range_match:
        start = _parse_date_text(range_match.group(1))
        end = _parse_date_text(range_match.group(2))
        if start and end:
            filters.append(FilterSpec(column, start[0].isoformat(), 0.99, "gte"))
            exclusive_end = _next_month(end[0]) if end[1] == "month" else end[0] + dt.timedelta(days=1)
            filters.append(FilterSpec(column, exclusive_end.isoformat(), 0.99, "lt"))
    elif lower_match:
        start = _parse_date_text(lower_match.group(1))
        if start:
            filters.append(FilterSpec(column, start[0].isoformat(), 0.99, "gte"))
    return filters


def _excluded_query_terms(question: str) -> set[str]:
    """Return concepts explicitly rejected by a negative constraint."""

    chunks: list[str] = []
    patterns = (
        r"\bdo\s+not\s+(?:include|use|expose|show|return)\s+([^.;]+)",
        r"\brather\s+than\s+([^,.;]+)",
        r"\binstead\s+of\s+([^,.;]+)",
        r"\bwithout\s+([^,.;]+)",
    )
    for pattern in patterns:
        chunks.extend(match.group(1) for match in re.finditer(pattern, question, flags=re.IGNORECASE))
    return _terms(" ".join(chunks)) if chunks else set()


def _requests_category_split(question: str) -> bool:
    """Recognize structural series requests without depending on one verb."""

    lowered = question.lower().replace("_", " ")
    return bool(
        re.search(r"\b(?:split|grouped|stacked|broken\s+down)\s+(?:each\s+[^,.;]+?\s+)?(?:by|into)\b", lowered)
        or re.search(r"\b(?:series|lines|colors?)\s+(?:by|for)\b", lowered)
    )


def _order_dimensions(dimensions: list[ColumnProfile], question: str) -> list[ColumnProfile]:
    if len(dimensions) < 2:
        return dimensions
    lowered = question.lower().replace("_", " ")
    split_match = re.search(
        r"\b(?:split(?:\s+each\s+[^,.;]+?)?|stacked|grouped|broken\s+down)\s+(?:by|into)\s+([^.;,]+)",
        lowered,
    )
    split_terms = _terms(split_match.group(1)) if split_match else set()

    def position(column: ColumnProfile) -> tuple[int, int, str]:
        normalized_name = " ".join(TOKEN_RE.findall(column.name.lower().replace("_", " ")))
        exact_position = lowered.find(normalized_name) if normalized_name else -1
        if exact_position >= 0:
            return (0, exact_position, column.name)
        positions = [lowered.find(term) for term in _raw_terms(column.name) if len(term) > 2 and lowered.find(term) >= 0]
        return (1, min(positions) if positions else len(lowered) + 1, column.name)

    split_candidates = [column for column in dimensions if _semantic_overlap(column.terms, split_terms) > 0]
    split = max(split_candidates, key=lambda column: _semantic_overlap(column.terms, split_terms), default=None)
    remaining = sorted((column for column in dimensions if column != split), key=position)
    return [*remaining[:1], *([split] if split else []), *remaining[1:]]


def _filter_sql(plan: AnalyticalPlan, aliases: dict[str, str]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    columns: dict[tuple[str, str], ColumnProfile] = {}
    for item in plan.filters:
        if item.operator != "eq":
            if item.column.table not in aliases:
                continue
            operator = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}.get(item.operator)
            if operator:
                clauses.append(f"{_column_sql(item.column, aliases)} {operator} ?")
                params.append(item.value)
            continue
        key = (item.column.table, item.column.name)
        grouped.setdefault(key, []).append(item.value)
        columns[key] = item.column
    for key, values in grouped.items():
        column = columns[key]
        if column.table not in aliases:
            continue
        placeholders = ", ".join("lower(?)" for _ in values)
        clauses.append(f"lower(cast({_column_sql(column, aliases)} as varchar)) in ({placeholders})")
        params.extend(values)
    return (" and " + " and ".join(clauses), params) if clauses else ("", [])


def _column_has_value(con: Any, column: ColumnProfile) -> bool:
    try:
        row = con.execute(
            f"select 1 from {_quote(column.table)} where {_quote(column.name)} is not null limit 1"
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _column_value_profile(con: Any, column: ColumnProfile, *, sample_size: int = 5000) -> tuple[int, int]:
    try:
        row = con.execute(
            f"""
            select count(*), count(distinct value)
            from (
                select cast({_quote(column.name)} as varchar) as value
                from {_quote(column.table)}
                where {_quote(column.name)} is not null
                  and cast({_quote(column.name)} as varchar) <> ''
                limit {int(sample_size)}
            ) sampled_values
            """
        ).fetchone()
        return int(row[0] or 0), int(row[1] or 0)
    except Exception:
        return 0, 0


def _has_explicit_analytical_shape(question: str) -> bool:
    raw_terms = _raw_terms(question)
    return bool(
        raw_terms
        & {
            "breakdown",
            "compare",
            "comparison",
            "composition",
            "distribution",
            "group",
            "grouped",
            "highest",
            "largest",
            "most",
            "rank",
            "ranking",
            "split",
            "time",
            "timeline",
            "top",
            "trend",
            "versus",
            "vs",
        }
    )


def _terms(value: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ").replace("-", " ").lower()
    terms = set(TOKEN_RE.findall(normalized))
    expanded = set(terms)
    for term in terms:
        if term.endswith("s") and len(term) > 3 and not term.endswith(("ss", "us", "is")):
            expanded.add(term[:-1])
        expanded.update(SEMANTIC_ALIASES.get(term, set()))
    return expanded


def _expanded_terms(question: str) -> set[str]:
    return _terms(question)


def _measure_query_terms(question: str) -> set[str]:
    lowered = question.lower()
    # Enrollment is a population event, not the entity being counted. Keep a
    # later scope noun such as "NECTEC courses" from changing the measure.
    if re.search(r"\b(?:distinct|unique)\s+(?:enrollments?|registrations?)\b", lowered):
        return {"user", "student", "learner"}
    count_phrases: list[str] = []
    for pattern in (
        r"\b(?:number|count|total)\s+of\s+([a-z0-9 _-]+?)(?:\s+by\b|\s+per\b|\s+split\b|\s+grouped\b|\s+over\b|$)",
        r"\bby\s+(?:the\s+)?(?:number|count|total)\s+of\s+([a-z0-9 _-]+?)(?:\s+and\b|\s+split\b|\s+grouped\b|$)",
    ):
        match = re.search(pattern, lowered)
        if match:
            count_phrases.append(match.group(1))
    terms: set[str] = set()
    for phrase in count_phrases:
        terms.update(_terms(phrase) & MEASURE_HINTS)
    if terms:
        return terms
    distinct_match = re.search(
        r"\b(?:distinct|unique)\s+(?:[a-z0-9_-]+\s+){0,4}"
        r"(users?|students?|learners?|courses?)\b",
        lowered,
    )
    if distinct_match:
        return _terms(distinct_match.group(1)) & MEASURE_HINTS
    match = re.search(
        r"\b(?:show|compare|rank|list|give me)\s+([a-z0-9 _-]+?)\s+(?:by|per|split by|grouped by|over time)\b",
        lowered,
    )
    if match:
        terms.update(_terms(match.group(1)) & MEASURE_HINTS)
    return terms


def _raw_terms(value: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ").replace("-", " ").lower()
    terms = set(TOKEN_RE.findall(normalized))
    return terms | {
        term[:-1]
        for term in terms
        if term.endswith("s") and len(term) > 3 and not term.endswith(("ss", "us", "is"))
    }


def _dimension_concept(field_name: str) -> str:
    ordered = TOKEN_RE.findall(field_name.replace("_", " ").replace("-", " ").lower())
    meaningful = [term for term in ordered if term not in {"id", "ids", "name", "label", "value"}]
    return meaningful[-1] if meaningful else ""


def _semantic_overlap(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> int:
    return len((set(left) - STOP_TERMS) & (set(right) - STOP_TERMS))


def _is_identifier(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.endswith("id")


def _is_time_column(name: str, data_type: str) -> bool:
    lowered_type = data_type.lower()
    lowered_name = name.lower()
    return any(token in lowered_type for token in ("date", "time")) or any(token in lowered_name for token in ("date", "time", "created_at", "updated_at"))


def _compatible_types(left: str, right: str) -> bool:
    left_type = left.lower()
    right_type = right.lower()
    if left_type == right_type:
        return True
    numeric = ("int", "decimal", "numeric", "double", "float")
    return any(token in left_type for token in numeric) and any(token in right_type for token in numeric)


def _is_numeric_type(data_type: str) -> bool:
    lowered = data_type.lower()
    return any(token in lowered for token in ("int", "decimal", "numeric", "double", "float", "hugeint"))


def _column_dict(column: ColumnProfile) -> dict[str, Any]:
    return {"table": column.table, "field": column.name, "type": column.data_type}


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _display(value: str) -> str:
    return " ".join(part.upper() if part.lower() == "id" else part.capitalize() for part in re.split(r"[_\-\s]+", value) if part)


def _measure_display(value: str) -> str:
    terms = _raw_terms(value)
    if terms & {"user", "student", "learner"}:
        return "Users"
    if "course" in terms:
        return "Courses"
    return _display(value)


def _categorical_chart_type(question: str, field: str, cardinality: int) -> str:
    return _categorical_chart_preferences(question, field, cardinality)[0]


def _requested_categorical_chart_type(question: str) -> str:
    lowered = question.lower()
    requested = {
        "donut": ("donut", "ring chart"),
        "pie": ("pie chart", "pie"),
        "treemap": ("treemap", "tree map", "hierarchy"),
        "radar": ("radar", "spider chart", "profile"),
        "radial_bar": ("radial", "circular bar"),
        "funnel": ("funnel", "conversion", "journey", "stage"),
        "column": ("column", "vertical bar"),
        "horizontal_bar": ("horizontal bar", "rank", "ranking", "top", "most"),
    }
    for chart_type, phrases in requested.items():
        if any(phrase in lowered for phrase in phrases):
            return chart_type
    return ""


def _categorical_chart_preferences(question: str, field: str, cardinality: int) -> list[str]:
    lowered = question.lower()
    explicitly_requested = _requested_categorical_chart_type(question)
    if explicitly_requested:
        if explicitly_requested in {"donut", "pie", "radar", "radial_bar"} and cardinality > 8:
            return ["horizontal_bar", "treemap", "column"]
        return [explicitly_requested]
    field_terms = _raw_terms(field)
    if field_terms & {"stage", "step", "funnel"} and 2 <= cardinality <= 8:
        return ["funnel", "column", "horizontal_bar"]
    if any(term in lowered for term in ("composition", "share", "percentage", "proportion")) and cardinality <= 8:
        return ["donut", "column", "horizontal_bar"]
    if any(term in lowered for term in ("rank", "ranking", "top", "most", "highest", "largest")):
        return ["horizontal_bar", "treemap", "column"]
    if field_terms & {"status", "state", "category", "type", "result"} and 2 <= cardinality <= 6:
        return ["donut", "column", "horizontal_bar"]
    if 9 <= cardinality <= 20:
        return ["treemap", "horizontal_bar", "column"]
    if 2 <= cardinality <= 8:
        return ["column", "donut", "horizontal_bar"]
    return ["horizontal_bar", "treemap", "column"]


def _slug(value: str) -> str:
    return "-".join(TOKEN_RE.findall(value.lower()))


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _original_source_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3 and parts[0] == "parquet":
        return f"parquet/{parts[1]}.parquet"
    if len(parts) == 3 and parts[2] == f"{parts[1]}.json":
        return f"{parts[0]}/{parts[2]}"
    return normalized
