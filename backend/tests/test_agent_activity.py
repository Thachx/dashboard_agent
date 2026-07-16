import json

import pytest

from dashboard_agent import agent
from dashboard_agent.dashboard_planner import _catalog, plan_complex_dashboard
from dashboard_agent.graph_dashboard_cache import (
    read_graph_dashboard_cache,
    read_question_translation,
    write_graph_dashboard_cache,
    write_question_translation,
)


class FakeStore:
    def __init__(self, path):
        self.path = path


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


def test_duckdb_fallback_is_persisted_and_reused_as_graph_aggregate(tmp_path):
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
    cached = read_graph_dashboard_cache(graph_path, "compare user by institution and province")

    assert cached["summary"]["executionSource"] == "graph"
    assert cached["datasets"]["fact"]["object_type"] == "graph_dashboard_aggregate"
    assert cached["chartSlots"] == activity["chartSlots"]
    details = " ".join(str(item.get("detail") or "") for item in cached["decisionTrace"])
    assert details == ""
    assert "from duckdb context" not in details.lower()


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
        "ผ่าน",
        "กำลังเรียน",
        "ไม่ได้ใช้งาน",
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
    assert read_question_translation(graph_path, f"analytics-v4:{question}") == canonical


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
