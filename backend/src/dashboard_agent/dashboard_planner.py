from __future__ import annotations

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
}
DIMENSION_ALIASES = {
    key: value
    for key, value in SEMANTIC_ALIASES.items()
    if key in {"institute", "institution", "school", "student", "learner", "user", "province"}
}
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
                {"table": item.column.table, "field": item.column.name, "value": item.value, "confidence": item.confidence}
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
        con = duckdb.connect(str(path), read_only=True)
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
        plan.filters = _resolve_value_filters(con, catalog, plan, question)
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
    query_terms = _expanded_terms(question)
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

    time_dimension = (
        _resolve_time(catalog, query_terms, value_validator=value_validator, graph_hints=graph_hints)
        if query_terms & TIME_HINTS
        else None
    )
    dimensions = _resolve_dimensions(
        catalog,
        query_terms,
        raw_query_terms=_raw_terms(question),
        measure=measure,
        time_dimension=time_dimension,
        allow_identifiers=requested_ids,
        value_validator=value_validator,
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
        for phrase in (" and ", " split by ", " grouped by ", " group by ", " compare ", " within ", " for each ")
    )
    if len(dimensions) < 2 and not complex_language and not neutral_overview:
        return None
    if not dimensions and time_dimension is None:
        return None

    required = [measure, *dimensions]
    if time_dimension is not None:
        required.append(time_dimension)
    base_table = _select_base_table(catalog, required, graph_hints=graph_hints)
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
    if " split by " in f" {question.lower()} " and len(dimensions) == 2 and not joins and time_dimension is None:
        # The existing split-series planner has a richer stacked encoding for this compact shape.
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
            if value_validator is not None and not value_validator(column):
                continue
            score = _semantic_overlap(column.terms, requested or query_terms) * 12.0
            score += _graph_column_hint_score(column, graph_hints)
            if column.name.lower() in {"user_id", "student_id", "learner_id", "course_id"}:
                score += 4.0
            if "fact" in table.name or "joined" in table.name:
                score += 2.0
            if score > 0:
                candidates.append((score, column))
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    return candidates[0][1] if candidates else None


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
            if value_validator is not None and not value_validator(column):
                continue
            score = _semantic_overlap(column.terms, query_terms) * 8.0
            score += _graph_column_hint_score(column, graph_hints)
            if "activity" in column.terms or "complete" in column.terms or "completed" in column.terms:
                score += 3.0
            candidates.append((score, column))
    candidates.sort(key=lambda item: (-item[0], item[1].table, item[1].name))
    return candidates[0][1] if candidates else None


def _resolve_dimensions(
    catalog: dict[str, TableProfile],
    query_terms: set[str],
    *,
    raw_query_terms: set[str],
    measure: ColumnProfile,
    time_dimension: ColumnProfile | None,
    allow_identifiers: bool,
    value_validator: Callable[[ColumnProfile], bool] | None,
    graph_hints: dict[str, Any],
) -> list[ColumnProfile]:
    candidates: list[tuple[float, ColumnProfile]] = []
    for table in catalog.values():
        for column in table.columns.values():
            if column == measure or column == time_dimension or column.is_time:
                continue
            if value_validator is not None and not value_validator(column):
                continue
            if _is_numeric_type(column.data_type) and not column.is_identifier:
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
        concept = _dimension_concept(column.name)
        concepts = {concept} if concept and concept in raw_query_terms else ((_raw_terms(column.name) & raw_query_terms) - {"id", "ids", "name"})
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
        if plan.dimensions:
            group = plan.dimensions[0]
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
                {"label": str(bucket), "series": str(series), "value": int(value)}
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
        else:
            rows = con.execute(
                f"""
                select date_trunc('month', {time_sql}) as bucket, count(distinct {measure_sql}) as value
                {from_sql}
                where {time_sql} is not null and {measure_sql} is not null{filter_sql}
                group by 1 order by 1
                """
            , filter_params).fetchall()
            data = [{"label": str(label), "value": int(value)} for label, value in rows]
            slots.append({"id": "plan-time", "title": f"{_measure_display(plan.measure.name)} over time", "chartType": "line", "field": plan.time_dimension.name, "data": data})
            blocks.append({"type": "chart", "slotId": "plan-time", "span": 2})

    used_chart_types = {str(slot.get("chartType") or "") for slot in slots}
    for index, dimension in enumerate(plan.dimensions):
        if plan.time_dimension is not None and index == 0:
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
    table_names = {plan.base_table, *(join.left_table for join in plan.joins), *(join.right_table for join in plan.joins)}
    source_paths = _dedupe(path for table in table_names for path in _source_paths(con, table))
    title_dimension = plan.dimensions[0] if plan.dimensions else plan.time_dimension
    title = f"{_measure_display(plan.measure.name)} by {_display(title_dimension.name)}" if title_dimension else _measure_display(plan.measure.name)
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
            "title": title,
            "subtitle": "Dashboard generated from a validated multi-source analytical plan.",
            "blocks": blocks[:8],
        },
    }


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
    query_terms = _expanded_terms(question) - STOP_TERMS - ID_TERMS
    raw_query_terms = _raw_terms(question) - STOP_TERMS - ID_TERMS
    reachable = {plan.base_table, *(join.left_table for join in plan.joins), *(join.right_table for join in plan.joins)}
    excluded = {(plan.measure.table, plan.measure.name)}
    candidates: list[FilterSpec] = []
    for table_name in sorted(reachable):
        for column in catalog[table_name].columns.values():
            if (column.table, column.name) in excluded or column.is_identifier or column.is_time or _is_numeric_type(column.data_type):
                continue
            if _semantic_overlap(column.terms, query_terms) <= 0:
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
                value_terms = _raw_terms(value) - STOP_TERMS
                overlap = value_terms & raw_query_terms
                if not overlap:
                    continue
                confidence = min(0.99, 0.72 + len(overlap) * 0.09)
                candidates.append(FilterSpec(column=column, value=value, confidence=confidence))
    candidates.sort(key=lambda item: (-item.confidence, item.column.table, item.column.name, item.value))
    selected: list[FilterSpec] = []
    used_values: set[tuple[str, str, str]] = set()
    for item in candidates:
        key = (item.column.table, item.column.name, item.value.lower())
        if key in used_values:
            continue
        selected.append(item)
        used_values.add(key)
        if len(selected) >= 8:
            break
    return selected


def _filter_sql(plan: AnalyticalPlan, aliases: dict[str, str]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    columns: dict[tuple[str, str], ColumnProfile] = {}
    for item in plan.filters:
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
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        expanded.update(SEMANTIC_ALIASES.get(term, set()))
    return expanded


def _expanded_terms(question: str) -> set[str]:
    return _terms(question)


def _measure_query_terms(question: str) -> set[str]:
    lowered = question.lower()
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
    return terms | {term[:-1] for term in terms if term.endswith("s") and len(term) > 3}


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
