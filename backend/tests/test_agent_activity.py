import json

import pytest

from dashboard_agent import agent
from dashboard_agent.dashboard_planner import (
    AnalyticalPlan,
    ColumnProfile,
    FilterSpec,
    TableProfile,
    _apply_plan_presentation_to_slots,
    _catalog,
    _measure_query_terms,
    _plan_presentation_spec,
    _requested_top_limit,
    _resolve_measure,
    plan_complex_dashboard,
)
from dashboard_agent.graph_dashboard_cache import (
    read_graph_dashboard_cache,
    read_question_translation,
    write_graph_dashboard_cache,
    write_question_translation,
)


class FakeStore:
    def __init__(self, path):
        self.path = path


def test_plan_presentation_uses_plan_fields_and_never_filter_values():
    user_id = ColumnProfile(
        table="enrollment_fact",
        name="user_id",
        data_type="INTEGER",
        terms=frozenset({"user", "id"}),
        is_identifier=True,
        is_time=False,
    )
    department = ColumnProfile(
        table="enrollment_fact",
        name="department_name",
        data_type="VARCHAR",
        terms=frozenset({"department", "name"}),
        is_identifier=False,
        is_time=False,
    )
    status = ColumnProfile(
        table="enrollment_fact",
        name="learning_status",
        data_type="VARCHAR",
        terms=frozenset({"learning", "status"}),
        is_identifier=False,
        is_time=False,
    )
    plan = AnalyticalPlan(
        intent="ranked_comparison",
        base_table="enrollment_fact",
        measure=user_id,
        dimensions=[department, status],
        time_dimension=None,
        joins=[],
        requested_terms=["top", "department", "users", "split", "learning", "status"],
    )
    assert plan is not None
    # Simulate a filter with a value that must never become presentation text.
    plan.filters = [FilterSpec(column=department, value="private-person@example.com", confidence=1.0)]
    slots = [{"id": "departments", "field": "department_name", "splitField": "learning_status", "chartType": "stacked_bar"}]

    presentation = _plan_presentation_spec(plan, slots, "top departments by users split by learning status")
    _apply_plan_presentation_to_slots(plan, slots)

    assert presentation["title"] == "Users by Department Name"
    assert "Department Name" in presentation["subtitle"]
    assert "private-person@example.com" not in presentation["subtitle"]
    assert slots[0]["description"] == "Displays distinct users by Department Name split by Learning Status."


def test_plan_presentation_describes_time_filter_with_validated_date_range():
    user_id = ColumnProfile(
        table="enrollment_fact", name="user_id", data_type="INTEGER", terms=frozenset({"user", "id"}), is_identifier=True, is_time=False
    )
    enrolled_at = ColumnProfile(
        table="enrollment_fact", name="enrolled_at", data_type="TIMESTAMP", terms=frozenset({"enrolled", "at"}), is_identifier=False, is_time=True
    )
    plan = AnalyticalPlan(
        intent="time_series",
        base_table="enrollment_fact",
        measure=user_id,
        dimensions=[],
        time_dimension=enrolled_at,
        joins=[],
        requested_terms=["monthly", "users", "over", "time"],
    )
    assert plan is not None
    plan.filters = [FilterSpec(column=enrolled_at, value="2025-01-01", confidence=1.0, operator="gte")]

    presentation = _plan_presentation_spec(plan, [{"field": "enrolled_at", "chartType": "line"}], "monthly users over time")

    assert "monthly distinct users over time" in presentation["subtitle"]
    assert "from 2025-01-01" in presentation["subtitle"]


def test_thai_fallback_preserves_plan_based_topic_when_translation_is_unavailable(monkeypatch):
    monkeypatch.setattr(agent, "_invoke_language_json", lambda **_kwargs: None)
    activity = {
        "layoutSpec": {
            "title": "Users by Department",
            "subtitle": "Shows distinct users grouped by Department; filtered to Enrollment from 2025-01-01.",
        },
        "summary": {"metricLabels": {}},
        "chartSlots": [{"id": "departments", "title": "Users by Department", "data": []}],
    }

    localized = agent._localize_activity_to_thai("แสดงผู้ใช้ตามแผนก", activity)

    assert "Department" not in localized["layoutSpec"]["title"]
    assert agent._contains_thai(localized["layoutSpec"]["subtitle"])


def test_aggregate_cache_activity_binds_generic_distinct_time_buckets(tmp_path):
    graph_path = tmp_path / "s3-json-graph.json"
    cache_path = tmp_path / "full-scan-aggregates.json"
    cache_path.write_text(
        json.dumps(
            {
                "data/edx-elastic/ae-activity-data-stream.json": {
                    "full_scan_status": "ok",
                    "full_record_count": 300,
                    "full_counts_json": json.dumps(
                        {
                            "userID": [
                                {"label": "user-a", "value": 120},
                                {"label": "user-b", "value": 80},
                            ],
                            "event": [{"label": "page_view", "value": 200}],
                        }
                    ),
                    "full_time_buckets_json": json.dumps(
                        [{"label": "2026-06-24T10", "value": 180}]
                    ),
                    "full_distinct_time_buckets_json": json.dumps(
                        {
                            "customer_id": [
                                {"label": "2026-06-24T10", "value": 42},
                                {"label": "2026-06-24T11", "value": 58},
                            ]
                        }
                    ),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    activity = agent._aggregate_cache_activity(FakeStore(graph_path), "customer number over time")

    timeline = next(slot for slot in activity["chartSlots"] if slot["id"] == "distinctTime-customer-id")
    assert timeline["title"] == "Customer ID over time"
    assert timeline["sourceField"] == "customer_id"
    assert timeline["data"] == [
        {"label": "2026-06-24T10", "value": 42},
        {"label": "2026-06-24T11", "value": 58},
    ]


def test_grouped_time_series_plan_uses_requested_group_and_valid_time_column():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table dashboard_agent_user_course_fact (
            user_id integer,
            learning_status varchar,
            last_activity_date timestamp,
            empty_activity_date timestamp,
            course_type varchar
        )
        """
    )
    con.execute(
        """
        insert into dashboard_agent_user_course_fact
        values
          (1, 'passed', '2026-01-01', null, 'self-paced'),
          (2, 'passed', '2026-02-01', null, 'self-paced'),
          (3, 'in_progress', '2026-02-01', null, 'instructor-led')
        """
    )

    plan = agent._grouped_time_series_plan(
        con,
        "show users over time by learning status",
    )

    assert plan == {
        "table": "dashboard_agent_user_course_fact",
        "time": "last_activity_date",
        "dimension": "learning_status",
        "measure": "user_id",
    }


def test_companion_dimension_slots_follow_secondary_prompt_dimensions():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table dashboard_agent_user_course_fact (
            user_id integer,
            province varchar,
            learning_status varchar,
            course_id varchar
        )
        """
    )
    con.execute(
        """
        insert into dashboard_agent_user_course_fact
        values
          (1, 'Bangkok', 'passed', 'course-a'),
          (2, 'Bangkok', 'in_progress', 'course-b'),
          (3, 'Chiang Mai', 'passed', 'course-a'),
          (4, 'Bangkok', 'passed', 'course-c')
        """
    )

    slots = agent._duckdb_companion_dimension_slots(
        con,
        "dashboard_agent_user_course_fact",
        "which province has the most users and show learning status distribution",
        measure="user_id",
        excluded_dimensions={"province"},
        primary_dimension="province",
        primary_value="Bangkok",
    )

    assert slots
    assert slots[0]["field"] == "learning_status"
    assert slots[0]["data"] == [
        {"label": "passed", "value": 2},
        {"label": "in_progress", "value": 1},
    ]


def test_complex_planner_resolves_validated_join_path():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            course_id varchar,
            learning_status varchar,
            completed_at timestamp
        );
        create table dashboard_agent_user_dimension (
            user_id integer,
            province varchar,
            institute_id integer
        );
        create table dashboard_agent_institute_dimension (
            institute_id integer,
            institute_name varchar
        );
        """
    )

    plan = plan_complex_dashboard(
        _catalog(con),
        "compare users by province and institute name over time",
    )

    assert plan is not None
    assert plan.measure.name == "user_id"
    assert {dimension.name for dimension in plan.dimensions} == {"province", "institute_name"}
    assert plan.time_dimension is not None
    assert plan.time_dimension.name == "completed_at"
    assert len(plan.joins) == 2
    assert {(join.left_key, join.right_key) for join in plan.joins} == {
        ("user_id", "user_id"),
        ("institute_id", "institute_id"),
    }


def test_date_filter_field_does_not_create_unrequested_time_chart():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        "create table dashboard_agent_enrollment_fact "
        "(user_id integer, department_name varchar, learning_status varchar, enroll_date timestamp)"
    )
    plan = plan_complex_dashboard(
        _catalog(con),
        "rank top 8 department_name by distinct users split by learning_status "
        "and filter enroll_date from 2025-01-01",
    )
    assert plan is not None
    assert plan.time_dimension is None


def test_complex_planner_does_not_invent_unrequested_filters(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "planner.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            course_id varchar,
            learning_status varchar
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'course-a', 'passed'),
            (2, 'course-a', 'in_progress'),
            (3, 'course-b', 'passed');
        """
    )
    con.close()

    activity = agent.build_complex_dashboard(
        db_path,
        "compare users by course ID and learning status",
    )

    assert activity
    public_plan = activity["summary"]["analyticalPlan"]
    assert public_plan["filters"] == []
    assert {item["field"] for item in public_plan["dimensions"]} == {"course_id", "learning_status"}


def test_complex_planner_uses_graph_hints_to_choose_duckdb_table():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    for table in ("dashboard_agent_alpha_fact", "dashboard_agent_beta_fact"):
        con.execute(
            f"""
            create table {table} (
                user_id integer,
                province varchar,
                learning_status varchar
            )
            """
        )

    plan = plan_complex_dashboard(
        _catalog(con),
        "compare users by province and learning status",
        graph_hints={
            "tables": ["dashboard_agent_beta_fact"],
            "fields": ["user_id", "province", "learning_status"],
            "sourcePaths": ["data/beta.json"],
            "evidence": ["Beta enrollment dataset"],
        },
    )

    assert plan is not None
    assert plan.base_table == "dashboard_agent_beta_fact"
    assert plan.measure.table == "dashboard_agent_beta_fact"
    assert {dimension.table for dimension in plan.dimensions} == {"dashboard_agent_beta_fact"}
    assert plan.public_dict()["graphHints"]["sourcePaths"] == ["data/beta.json"]


def test_neutral_prompt_builds_schema_driven_overview(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "neutral.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            learning_status varchar,
            province varchar,
            course_type varchar,
            free_text varchar
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'passed', 'Bangkok', 'self-paced', 'unique note 1'),
            (2, 'in_progress', 'Bangkok', 'self-paced', 'unique note 2'),
            (3, 'passed', 'Chiang Mai', 'instructor-led', 'unique note 3'),
            (4, 'inactive', 'Phuket', 'self-paced', 'unique note 4');
        """
    )
    con.close()

    activity = agent.build_complex_dashboard(
        db_path,
        "tell me about users",
        graph_hints={
            "tables": ["dashboard_agent_enrollment_fact"],
            "fields": ["user_id", "learning_status", "province", "course_type"],
            "sourcePaths": ["data/enrollments.json"],
        },
    )

    assert activity
    plan = activity["summary"]["analyticalPlan"]
    assert plan["intent"] == "neutral_overview"
    assert plan["measure"]["field"] == "user_id"
    assert {item["field"] for item in plan["dimensions"]} == {
        "course_type",
        "learning_status",
        "province",
    }
    assert len(activity["chartSlots"]) == 3
    assert {slot["chartType"] for slot in activity["chartSlots"]} == {
        "column",
        "donut",
        "horizontal_bar",
    }


def test_explicit_rank_prompt_does_not_fall_back_to_neutral_overview():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            learning_status varchar
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'passed'),
            (2, 'in_progress');
        """
    )

    plan = plan_complex_dashboard(
        _catalog(con),
        "which province has the most users",
        dimension_profiler=lambda _column: (2, 2),
    )

    assert plan is None


def test_hybrid_execution_records_graph_planning_and_duckdb_calculation():
    activity = {
        "datasets": {"fact": {"object_type": "duckdb_join_plan"}},
        "summary": {"totalRecords": 10},
        "chartSlots": [{"id": "users", "data": [{"label": "A", "value": 10}]}],
    }
    hints = {
        "tables": ["fact"],
        "fields": ["user_id", "province"],
        "sourcePaths": ["data/users.json"],
        "relationships": ["Users -> Province"],
        "evidence": ["User dataset"],
    }

    hybrid = agent._mark_hybrid_execution(activity, hints)

    assert hybrid["summary"]["executionSource"] == "hybrid"
    assert hybrid["summary"]["planningSource"] == "graph"
    assert hybrid["summary"]["calculationSource"] == "duckdb"
    assert hybrid["summary"]["graphPlanning"]["relationships"] == ["Users -> Province"]


def test_graph_planning_hints_separate_json_field_paths_from_sources(tmp_path):
    hints = agent._graph_planning_hints(
        FakeStore(tmp_path / "graph.json"),
        [
            {
                "id": "duckdb/unified_records/parquet/fact_student_course/part.parquet::$.sample_records[0].province",
                "label": "Province",
                "path": "$.sample_records[0].province",
                "source": "duckdb/unified_records/parquet/fact_student_course/part.parquet",
                "text": "province",
            }
        ],
    )

    assert "province" in hints["fields"]
    assert "unified_records" in hints["tables"]
    assert "$.sample_records[0].province" not in hints["sourcePaths"]
    assert hints["sourcePaths"] == ["duckdb/unified_records/parquet/fact_student_course/part.parquet"]


def test_chart_type_choices_only_accept_compatible_renderers():
    activity = {
        "chartSlots": [
            {
                "id": "ranking",
                "title": "Users by province",
                "chartType": "horizontal_bar",
                "field": "province",
                "data": [{"label": "A", "value": 10}, {"label": "B", "value": 8}, {"label": "C", "value": 3}],
            },
            {
                "id": "timeline",
                "title": "Users over time",
                "chartType": "multi_line",
                "field": "activity_date",
                "data": [
                    {"label": "2026-01", "series": "passed", "value": 10},
                    {"label": "2026-01", "series": "in_progress", "value": 20},
                ],
            },
        ]
    }

    agent._apply_chart_type_choices(activity, {"ranking": "radar", "timeline": "pie"})

    assert activity["chartSlots"][0]["chartType"] == "radar"
    assert activity["chartSlots"][1]["chartType"] == "multi_line"
    assert agent._prompt_chart_type_choice("show the composition as a pie chart") == "pie"


def test_llm_chart_choices_do_not_collapse_meaningful_variety():
    slots = [
        {
            "id": "status",
            "chartType": "donut",
            "data": [{"label": "passed", "value": 10}, {"label": "active", "value": 8}],
        },
        {
            "id": "department",
            "chartType": "column",
            "data": [{"label": "A", "value": 10}, {"label": "B", "value": 8}],
        },
        {
            "id": "teacher",
            "chartType": "treemap",
            "data": [{"label": "One", "value": 10}, {"label": "Two", "value": 8}],
        },
    ]

    choices = agent._preserve_meaningful_chart_variety(
        slots,
        {"status": "horizontal_bar", "department": "horizontal_bar", "teacher": "horizontal_bar"},
    )

    assert choices == {
        "status": "horizontal_bar",
        "department": "column",
        "teacher": "treemap",
    }


def test_duckdb_fallback_is_persisted_and_reused_for_exact_normalized_question(tmp_path):
    graph_path = tmp_path / "graph.json"
    activity = {
        "datasets": {
            "fact": {
                "source": "fact",
                "object_type": "duckdb_join_plan",
            }
        },
        "summary": {"totalDistinctMeasure": 42},
        "decisionTrace": [
            {
                "stage": "Select data",
                "detail": "Selected dashboard_agent_user_course_fact from duckdb context because it contains the ranked dimension and measure.",
            },
            {
                "stage": "Collect lineage",
                "detail": "Resolved dashboard sources from DuckDB lineage metadata, source path columns, or source_summary fallback.",
            },
        ],
        "chartSlots": [{"id": "users", "chartType": "column", "data": [{"label": "A", "value": 42}]}],
    }

    assert write_graph_dashboard_cache(graph_path, "compare users by province and institute", activity)
    cached = read_graph_dashboard_cache(graph_path, "  COMPARE users by province and institute  ")

    assert cached["summary"]["executionSource"] == "graph"
    assert cached["summary"]["reasoningSource"] == "cache"
    assert cached["datasets"]["fact"]["object_type"] == "graph_dashboard_aggregate"
    assert cached["chartSlots"] == activity["chartSlots"]
    details = " ".join(str(item.get("detail") or "") for item in cached["decisionTrace"])
    assert "Reused a matching precomputed dashboard" in details
    assert "from duckdb context" not in details.lower()


def test_dashboard_cache_rejects_stale_source_version(tmp_path):
    graph_path = tmp_path / "graph.json"
    activity = {
        "datasets": {"fact": {"object_type": "duckdb_join_plan"}},
        "chartSlots": [{"id": "users", "data": [{"label": "A", "value": 42}]}],
    }

    assert write_graph_dashboard_cache(
        graph_path,
        "show users by province",
        activity,
        source_version="warehouse-v1",
    )

    assert read_graph_dashboard_cache(
        graph_path,
        "show users by province",
        source_version="warehouse-v2",
    ) == {}
    cached = read_graph_dashboard_cache(
        graph_path,
        "show users by province",
        source_version="warehouse-v1",
    )
    assert cached["summary"]["reasoningSource"] == "cache"
    assert cached["decisionTrace"]


def test_thai_fallback_normalizes_processing_and_localizes_presentation(monkeypatch):
    monkeypatch.setattr(agent, "_invoke_language_json", lambda **_kwargs: None)
    question = "แสดงจำนวนผู้ใช้งานแยกตามจังหวัดมากที่สุด"

    canonical = agent._canonicalize_question_for_processing(question)
    activity = {
        "layoutSpec": {"title": "Users by Province", "subtitle": "Activity dashboard"},
        "summary": {"metricLabels": {"total": "Total Users"}},
        "chartSlots": [{"id": "province", "title": "Users by Province", "data": []}],
    }
    localized = agent._localize_activity_to_thai(question, activity)

    assert not agent._contains_thai(canonical)
    assert "users" in canonical.lower()
    assert agent._contains_thai(localized["layoutSpec"]["title"])
    assert localized["summary"]["metricLabels"]["total"] == "ทั้งหมด ผู้ใช้งาน"


def test_thai_fallback_preserves_complex_analytical_structure(monkeypatch):
    monkeypatch.setattr(agent, "_invoke_language_json", lambda **_kwargs: None)

    canonical = agent._canonicalize_question_for_processing(
        "แสดงแนวโน้มจำนวนผู้ใช้งานรายเดือนแยกตามจังหวัดและสถานศึกษา"
    )

    assert "trend" in canonical
    assert "monthly" in canonical
    assert "users" in canonical
    assert "province" in canonical
    assert "split by" in canonical
    assert "institute" in canonical


def test_english_canonicalization_preserves_full_multi_chart_request(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: pytest.fail("English requests must not be rewritten by an LLM"),
    )
    question = (
        "Build exactly three charts since 2025-01-01: top departments split by status; "
        "overall status distribution; and monthly enrollment through June 2026."
    )

    canonical = agent._canonicalize_question_for_processing(question)

    assert canonical == question
    assert "top departments split by status" in canonical
    assert "monthly enrollment" in canonical


def test_date_filter_alone_is_not_time_series_intent():
    assert not agent._semantic_time_intent(
        "Rank the top provinces for enrollments since 2025-01-01"
    )
    assert agent._semantic_time_intent(
        "Show monthly enrollments from January 2025 through June 2026"
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Break each province down by learning status", "split by learning status"),
        ("Break every bar down by status", "split by status"),
        ("Show passed versus not passed", "split by course_pass"),
        ("Break each province into passed versus not passed", "split by course_pass"),
        ("Split into passed and not_passed", "split by passed and not_passed"),
    ],
)
def test_canonical_normalization_recognizes_split_language(question, expected):
    assert expected in agent._normalize_canonical_analytics_question(question).lower()


def test_planner_recognizes_split_into_pass_status():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        "create table dashboard_agent_user_course_fact "
        "(user_id integer, school_province varchar, course_pass integer, "
        "has_certificate integer, enroll_date timestamp)"
    )
    plan = plan_complex_dashboard(
        _catalog(con),
        "Show top school provinces by distinct certified users since 2025-01-01 "
        "split into passed and not_passed",
    )
    assert plan is not None
    assert [column.name for column in plan.dimensions[:2]] == ["school_province", "course_pass"]
    assert plan.time_dimension is None


def test_measure_terms_and_word_top_limit_are_generic():
    assert "user" in _measure_query_terms("top eight provinces by unique users")
    assert "user" in _measure_query_terms(
        "top departments by unique secondary-education enrolled users"
    )
    assert _requested_top_limit("which ten provinces have the most users") == 10


def test_mixed_thai_schema_request_preserves_structured_clauses():
    question = (
        "เชื่อมตาราง enrollment fact กับ user dimension กรอง level_of_education เป็น secondary "
        "และ enroll_date ตั้งแต่ 2025-01-01 ตัด department_name ที่ว่าง จัดอันดับ 8 อันดับด้วย "
        "distinct users และแยก series ตาม learning_status โดยคืนเฉพาะข้อมูลรวม"
    )
    canonical = agent._canonicalize_question_for_processing(question, semantic_context={})
    for term in (
        "level_of_education",
        "secondary",
        "enroll_date",
        "2025-01-01",
        "department_name",
        "distinct users",
        "split by learning_status",
    ):
        assert term in canonical


def test_explicit_measure_identity_beats_incidental_context():
    fact = TableProfile(
        name="dashboard_agent_user_course_fact",
        columns={
            name: ColumnProfile(
                table="dashboard_agent_user_course_fact",
                name=name,
                data_type="VARCHAR",
                terms=frozenset(name.removesuffix("_id").split("_")) | {"id"},
                is_identifier=True,
                is_time=False,
            )
            for name in ("user_id", "course_id")
        },
    )
    chosen = _resolve_measure(
        {fact.name: fact},
        {"user", "course", "province"},
        {"user"},
        value_validator=None,
        graph_hints={"fields": ["course_id"]},
    )
    assert chosen is not None
    assert chosen.name == "user_id"


def test_thai_localization_rejects_untranslated_model_output(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: {
            "title": "Course and State Distribution Overview",
            "subtitle": "Counts of courses and states",
            "chartTitles": {"courses": "Course ID Distribution"},
            "metricLabels": {"rows": "Activity rows"},
        },
    )
    activity = {
        "layoutSpec": {
            "title": "Course and State Distribution Overview",
            "subtitle": "Counts of courses and states",
        },
        "summary": {"metricLabels": {"rows": "Activity rows"}},
        "chartSlots": [
            {"id": "courses", "title": "Course ID Distribution", "data": []},
        ],
    }

    localized = agent._localize_activity_to_thai("แสดงการกระจายหลักสูตร", activity)

    assert agent._contains_thai(localized["layoutSpec"]["title"])
    assert agent._contains_thai(localized["layoutSpec"]["subtitle"])
    assert agent._contains_thai(localized["chartSlots"][0]["title"])
    assert agent._contains_thai(localized["summary"]["metricLabels"]["rows"])


def test_thai_localization_covers_mixed_labels_reasoning_and_category_values(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: {
            "title": "ผู้ใช้งาน by Institute Name",
            "subtitle": "ราชวินิตบางแก้ว leads this comparison.",
            "chartTitles": {"institutes": "ผู้ใช้งาน by Institute Name split by Learning Status"},
            "metricLabels": {
                "topDimensionValue": "Leading ผู้ใช้งาน",
                "totalDistinctMeasure": "Matched ผู้ใช้งาน",
            },
            "reasoning": [
                {
                    "index": 0,
                    "stage": "request_interpretation",
                    "detail": "Identify institute names and user counts categorized by learning status.",
                    "evidence": [],
                }
            ],
        },
    )
    activity = {
        "layoutSpec": {
            "title": "Users by Institute Name",
            "subtitle": "ราชวินิตบางแก้ว leads this comparison.",
        },
        "summary": {
            "metricLabels": {
                "topDimensionValue": "Leading Users",
                "totalDistinctMeasure": "Matched Users",
            }
        },
        "chartSlots": [
            {
                "id": "institutes",
                "title": "Users by Institute Name split by Learning Status",
                "data": [
                    {"label": "ราชวินิตบางแก้ว", "series": "Passed", "value": 10},
                    {"label": "ราชวินิตบางแก้ว", "series": "In Progress", "value": 8},
                    {"label": "ราชวินิตบางแก้ว", "series": "Inactive", "value": 2},
                ],
            }
        ],
        "decisionTrace": [
            {
                "stage": "request_interpretation",
                "detail": "Identify institute names and user counts categorized by learning status.",
            }
        ],
    }

    localized = agent._localize_activity_to_thai("จำนวนผู้ใช้แยกตามสถาบันและสถานะการเรียน", activity)

    assert localized["summary"]["presentationLanguage"] == "th"
    assert "Institute" not in localized["layoutSpec"]["title"]
    assert "leads this comparison" not in localized["layoutSpec"]["subtitle"]
    assert "Learning Status" not in localized["chartSlots"][0]["title"]
    assert [row["series"] for row in localized["chartSlots"][0]["data"]] == [
        "Passed",
        "In Progress",
        "Inactive",
    ]
    assert localized["decisionTrace"][0]["stage"] == "Request interpretation"
    assert "Identify" not in localized["decisionTrace"][0]["detail"]
    assert "dimension" in localized["decisionTrace"][0]["detail"]


def test_thai_reasoning_preserves_technical_terms(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: {
            "title": "ผู้ใช้งานตามสถาบัน",
            "subtitle": "แสดงผลตามคำขอ",
            "reasoning": [
                {
                    "index": 0,
                    "detail": "ดึงข้อมูลด้วย graph traversal แทน direct database query",
                    "evidence": [],
                },
                {
                    "index": 1,
                    "detail": "แสดงผลด้วย stacked bar chart และ horizontal bar chart",
                    "evidence": [],
                },
            ],
        },
    )
    activity = {
        "layoutSpec": {"title": "Users by Institute", "subtitle": "Dashboard"},
        "summary": {"metricLabels": {}},
        "chartSlots": [],
        "decisionTrace": [
            {"stage": "execution_source", "detail": "Data retrieved from graph."},
            {"stage": "chart_binding", "detail": "Mapped to charts."},
        ],
    }

    localized = agent._localize_activity_to_thai("แสดงผู้ใช้ตามสถาบัน", activity)

    assert localized["decisionTrace"][0] == {
        "stage": "Execution source",
        "detail": "ดึงข้อมูลด้วย graph traversal แทน direct database query",
        "evidence": [],
    }
    assert localized["decisionTrace"][1] == {
        "stage": "Chart binding",
        "detail": "แสดงผลด้วย stacked bar chart และ horizontal bar chart",
        "evidence": [],
    }


def test_thai_canonical_translation_persists_across_memory_cache(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    question = "แสดงจำนวนผู้ใช้แยกตามจังหวัด"
    calls = 0

    def translate(**_kwargs):
        nonlocal calls
        calls += 1
        return {"englishQuestion": "show users by province"}

    monkeypatch.setattr(agent, "_invoke_language_json", translate)
    agent._question_language_cache.clear()
    first = agent._canonicalize_question_for_processing(question, graph_path=graph_path)
    agent._question_language_cache.clear()
    second = agent._canonicalize_question_for_processing(question, graph_path=graph_path)

    assert first == second == "show users by province"
    assert calls == 1
    assert read_question_translation(graph_path, question) == first


def test_shared_semantic_interpreter_normalizes_natural_prompt_with_schema_context(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    question = "where do most learners live?"
    captured = {}

    def interpret(*, system, payload):
        captured["system"] = system
        captured["payload"] = payload
        return {
            "canonicalQuestion": "show province by number of users ranked highest",
            "measureField": "user_id",
            "dimensionFields": ["province"],
            "timeField": None,
            "filters": [],
            "confidence": 0.96,
        }

    monkeypatch.setattr(agent, "_invoke_language_json", interpret)
    agent._question_language_cache.clear()
    canonical = agent._canonicalize_question_for_processing(
        question,
        graph_path=graph_path,
        semantic_context={
            "graphCandidates": {"fields": ["province", "user_id"]},
            "schemaCatalog": [
                {"table": "dashboard_agent_user_fact", "fields": ["user_id", "province"]}
            ],
        },
    )

    assert canonical == "show province by number of user_id ranked highest"
    assert captured["payload"]["schemaCatalog"][0]["fields"] == ["user_id", "province"]
    assert "Equivalent precise" in captured["system"]
    assert read_question_translation(graph_path, f"analytics-v5:{question}") == canonical


def test_shared_semantic_interpreter_has_deterministic_english_fallback(monkeypatch):
    monkeypatch.setattr(agent, "_invoke_language_json", lambda **_kwargs: None)
    agent._question_language_cache.clear()

    canonical = agent._canonicalize_question_for_processing(
        "compare user counts across schools, broken down by study status",
        semantic_context={"graphCandidates": {}, "schemaCatalog": []},
    )

    assert canonical == "compare user counts across schools, split by learning status"


def test_structured_semantic_plan_generates_fixed_quality_canonical_query():
    canonical = agent._canonical_question_from_semantic_plan(
        {
            "intent": "ranking",
            "measureField": "user_id",
            "dimensionFields": ["province"],
            "splitField": None,
            "timeField": None,
            "ranking": True,
            "filters": [],
        },
        {
            "graphCandidates": {"fields": ["province", "user_id"]},
            "schemaCatalog": [
                {"table": "dashboard_agent_user_fact", "fields": ["user_id", "province"]}
            ],
        },
        fallback="where do most learners live?",
    )

    assert canonical == "show province by number of user_id ranked highest"


def test_structured_semantic_plan_recovers_implicit_time_shape_from_schema():
    canonical = agent._canonical_question_from_semantic_plan(
        {
            "intent": "comparison",
            "measureField": "user_id",
            "dimensionFields": [],
            "timeField": None,
            "filters": [],
        },
        {
            "graphCandidates": {"fields": ["learning_status", "last_activity_date"]},
            "schemaCatalog": [
                {
                    "table": "dashboard_agent_user_fact",
                    "fields": ["user_id", "learning_status", "last_activity_date"],
                }
            ],
        },
        fallback="show learning state changes",
        original_question="how has each learning state changed month to month?",
    )

    assert canonical == "show number of user_id over time split by learning_status monthly"


def test_ranked_prompt_does_not_add_unrequested_companion_charts(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    graph_path = tmp_path / "graph.json"
    db_path = tmp_path / "dashboard_agent.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table dashboard_agent_user_fact (
            user_id integer,
            province varchar,
            learning_status varchar,
            course_id varchar
        );
        insert into dashboard_agent_user_fact values
            (1, 'Bangkok', 'passed', 'course-a'),
            (2, 'Bangkok', 'in_progress', 'course-b'),
            (3, 'Chiang Mai', 'passed', 'course-a');
        """
    )
    con.close()

    activity = agent._duckdb_ranked_dimension_context(
        FakeStore(graph_path),
        "show province by number of user_id ranked highest",
    )

    assert activity
    assert [slot["field"] for slot in activity["chartSlots"]] == ["province"]


def test_complex_planner_supports_multiple_values_for_one_filter(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "multi-filter.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            learning_status varchar,
            last_activity_date timestamp
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'passed', '2026-01-01'),
            (2, 'in_progress', '2026-01-01'),
            (3, 'inactive', '2026-01-01');
        """
    )
    con.close()

    activity = agent.build_complex_dashboard(
        db_path,
        "show number of user_id over time split by learning_status "
        "where learning_status is passed or learning_status is in_progress",
    )

    assert activity
    filters = activity["summary"]["analyticalPlan"]["filters"]
    assert {item["value"] for item in filters} == {"passed", "in_progress"}
    series = {
        row["series"]
        for slot in activity["chartSlots"]
        for row in slot["data"]
        if row.get("series")
    }
    assert series == {"passed", "in_progress"}


def test_question_translation_cache_round_trip(tmp_path):
    graph_path = tmp_path / "graph.json"
    assert write_question_translation(graph_path, "คำถาม", "question")
    assert read_question_translation(graph_path, "คำถาม") == "question"


def test_canonical_analytics_synonyms_preserve_split_semantics():
    normalized = agent._normalize_canonical_analytics_question(
        "Number of students per institution broken down by enrollment status for each institution"
    )

    assert "users by institute split by learning status" in normalized.lower()
    assert "broken down" not in normalized.lower()
    assert "enrollment status" not in normalized.lower()


def test_thai_split_intent_repairs_weakened_translation(tmp_path):
    graph_path = tmp_path / "graph.json"
    question = "จำนวนผู้ใช้ในแต่ละสถาบันโดยแบ่งด้วยสถานะการเรียน"
    write_question_translation(
        graph_path,
        question,
        "Number of users by institute and learning status",
    )
    agent._question_language_cache.clear()

    canonical = agent._canonicalize_question_for_processing(question, graph_path=graph_path)

    assert canonical == "Number of users by institute split by learning status"
    assert read_question_translation(graph_path, question) == canonical


def test_agent_reasoning_uses_verified_graph_execution_context(monkeypatch):
    captured = {}

    def explain(*, system, payload):
        captured.update(payload)
        return {
            "reasoning": [
                {
                    "stage": "Execute plan",
                    "detail": "The persisted graph aggregate supplied the selected dimensions and measure.",
                    "evidence": ["executionSource: graph"],
                },
                {
                    "stage": "Bind chart",
                    "detail": "School Name is the category and Learning Status is the split series.",
                    "evidence": ["chartType: stacked_bar"],
                },
            ]
        }

    monkeypatch.setattr(agent, "_invoke_agent_json", explain)
    activity = {
        "datasets": {"fact": {"object_type": "graph_dashboard_aggregate", "source_paths": ["fact.parquet"]}},
        "summary": {"executionSource": "graph", "totalRecords": 10},
        "chartSlots": [
            {
                "id": "users",
                "chartType": "stacked_bar",
                "field": "school_name",
                "splitField": "learning_status",
                "data": [{"label": "A", "series": "passed", "value": 1}],
            }
        ],
    }

    result = agent._agent_dashboard_reasoning("users by school split by status", activity)

    assert captured["executionSource"] == "graph"
    assert result["summary"]["reasoningSource"] == "agent"
    assert result["summary"]["reasoningExecutionSource"] == "graph"
    assert len(result["decisionTrace"]) == 2
