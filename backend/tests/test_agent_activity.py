import json

import pytest

from dashboard_agent import agent
from dashboard_agent.dashboard_planner import _catalog, plan_complex_dashboard


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
