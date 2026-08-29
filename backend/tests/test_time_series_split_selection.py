import pytest

from dashboard_agent.dashboard_planner import (
    _catalog,
    _column_has_value,
    _column_value_profile,
    build_complex_dashboard,
    plan_complex_dashboard,
)


def _make_database(path):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            enrollment_status varchar,
            learning_status varchar,
            course_pass integer,
            enrolled_at timestamp
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'active', 'passed', 1, '2025-01-05'),
            (2, 'inactive', 'in_progress', 0, '2025-01-18'),
            (3, 'active', 'inactive', 1, '2025-02-03'),
            (4, 'inactive', 'passed', 0, '2025-02-12');
        """
    )
    con.close()


def test_time_series_accepts_separate_series_wording_without_split(tmp_path):
    path = tmp_path / "series.duckdb"
    _make_database(path)

    dashboard = build_complex_dashboard(
        path,
        "Show monthly distinct users as separate series for active and inactive enrollment status.",
    )

    matching = [item for item in dashboard["chartSlots"] if item["chartType"] == "multi_line"]
    assert matching, (dashboard.get("summary"), dashboard.get("chartSlots"))
    slot = matching[0]
    assert slot["splitField"] == "enrollment_status"
    assert {row["series"] for row in slot["data"]} == {"active", "inactive"}


def test_time_series_uses_named_category_values_to_choose_binary_series(tmp_path):
    path = tmp_path / "binary-series.duckdb"
    _make_database(path)

    question = "Show monthly distinct users as separate lines for passed and not passed course outcomes."
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(str(path), read_only=True)
    plan = plan_complex_dashboard(
        _catalog(con),
        question,
        value_validator=lambda column: _column_has_value(con, column),
        dimension_profiler=lambda column: _column_value_profile(con, column),
    )
    assert plan is not None
    assert "course_pass" in {column.name for column in plan.dimensions}, plan.public_dict()
    con.close()

    dashboard = build_complex_dashboard(path, question)

    matching = [item for item in dashboard["chartSlots"] if item["chartType"] == "multi_line"]
    assert matching, (dashboard.get("summary"), dashboard.get("chartSlots"))
    slot = matching[0]
    assert slot["splitField"] == "course_pass"
    assert {row["series"] for row in slot["data"]} == {"passed", "not_passed"}


def test_time_series_preserves_all_named_status_series(tmp_path):
    path = tmp_path / "named-status-series.duckdb"
    _make_database(path)

    dashboard = build_complex_dashboard(
        path,
        "Chart monthly distinct users with separate passed, in_progress, and inactive series based on enrolled_at.",
    )

    slot = next(item for item in dashboard["chartSlots"] if item["chartType"] == "multi_line")
    assert slot["splitField"] == "learning_status"
    assert {row["series"] for row in slot["data"]} == {"passed", "in_progress", "inactive"}


def test_ranked_split_keeps_complex_plan_when_temporal_filter_field_is_named(tmp_path):
    path = tmp_path / "ranked-split.duckdb"
    _make_database(path)

    dashboard = build_complex_dashboard(
        path,
        "Show top users split by enrollment status and course pass; filter enrolled_at from January 2025.",
    )

    assert dashboard
    assert dashboard["chartSlots"]
    assert any(slot.get("splitField") == "course_pass" for slot in dashboard["chartSlots"])
