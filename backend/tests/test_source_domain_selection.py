import pytest

from dashboard_agent.dashboard_planner import build_complex_dashboard


def test_enrollment_request_avoids_activity_derived_source(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "source-domain.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table dashboard_agent_user_course_fact (
            user_id integer,
            department_name varchar,
            learning_status varchar,
            enroll_date timestamp
        );
        create table dashboard_agent_activity_joined (
            user_id integer,
            department_name varchar,
            learning_status varchar,
            enroll_date timestamp,
            event_date timestamp
        );
        insert into dashboard_agent_user_course_fact values
            (1, 'A', 'passed', '2025-01-02'),
            (2, 'A', 'inactive', '2025-01-03'),
            (3, 'B', 'passed', '2025-02-03');
        insert into dashboard_agent_activity_joined values
            (99, 'Wrong', 'passed', '2025-01-02', '2025-01-03');
        """
    )
    con.close()

    dashboard = build_complex_dashboard(
        path,
        "Show enrolled users by department, stacked by learning status, using enrollment data.",
    )

    plan = dashboard["summary"]["analyticalPlan"]
    assert plan["baseTable"] == "dashboard_agent_user_course_fact"
    assert "dashboard_agent_activity_joined" not in dashboard["summary"]["source"]
    slot = next(item for item in dashboard["chartSlots"] if item.get("splitField") == "learning_status")
    assert {row["label"] for row in slot["data"]} == {"A", "B"}
