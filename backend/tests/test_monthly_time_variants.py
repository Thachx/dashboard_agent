import pytest

from dashboard_agent.dashboard_planner import build_complex_dashboard


def _make_database(path):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table dashboard_agent_activity_fact (
            user_id integer,
            enrollment_status varchar,
            activity_at timestamp
        );
        insert into dashboard_agent_activity_fact values
            (1, 'active', '2025-01-05'),
            (2, 'inactive', '2025-01-18'),
            (3, 'active', '2025-02-03'),
            (4, 'inactive', '2025-03-12');
        """
    )
    con.close()


def test_singular_month_cadence_resolves_time_and_status_series(tmp_path):
    path = tmp_path / "each-month.duckdb"
    _make_database(path)

    dashboard = build_complex_dashboard(
        path,
        "Show distinct users for each month between January 2025 and March 2025 "
        "with separate active and inactive enrollment status series.",
    )

    slots = [item for item in dashboard.get("chartSlots", []) if item.get("chartType") == "multi_line"]
    assert slots, (dashboard.get("summary"), dashboard.get("chartSlots"))
    assert slots[0]["splitField"] == "enrollment_status"


def test_date_range_filter_without_cadence_does_not_create_time_chart(tmp_path):
    path = tmp_path / "date-filter.duckdb"
    _make_database(path)

    dashboard = build_complex_dashboard(path, "Show distinct users since January 2025.")

    assert not any(item.get("chartType") == "multi_line" for item in dashboard.get("chartSlots", []))
