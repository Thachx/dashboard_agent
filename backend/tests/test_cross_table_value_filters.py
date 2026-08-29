import pytest

from dashboard_agent.dashboard_planner import build_complex_dashboard


def test_value_only_dimension_filter_extends_validated_join_plan(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "cross-table-filter.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table dashboard_agent_user_course_fact (
            user_id integer,
            department_name varchar,
            learning_status varchar,
            enroll_date timestamp
        );
        create table dashboard_agent_user_dim (
            user_id integer,
            level_of_education varchar
        );
        insert into dashboard_agent_user_course_fact values
            (1, 'A', 'passed', '2025-01-02'),
            (2, 'A', 'inactive', '2025-01-03'),
            (3, 'B', 'passed', '2025-01-04'),
            (4, 'B', 'passed', '2024-12-31');
        insert into dashboard_agent_user_dim values
            (1, 'secondary'),
            (2, 'primary'),
            (3, 'secondary'),
            (4, 'secondary');
        """
    )
    con.close()

    dashboard = build_complex_dashboard(
        path,
        "Show top 8 populated course departments by distinct secondary-level learners "
        "enrolled since 2025-01-01, split by learning status.",
    )

    plan = dashboard["summary"]["analyticalPlan"]
    assert any(join["right_table"] == "dashboard_agent_user_dim" for join in plan["joins"])
    assert any(
        item["field"] == "level_of_education" and item["value"] == "secondary"
        for item in plan["filters"]
    )
    slot = next(item for item in dashboard["chartSlots"] if item.get("splitField") == "learning_status")
    assert {(row["label"], row["series"], row["value"]) for row in slot["data"]} == {
        ("A", "passed", 1),
        ("B", "passed", 1),
    }
