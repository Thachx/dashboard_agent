import pytest

from dashboard_agent.dashboard_planner import (
    AnalyticalPlan,
    ColumnProfile,
    FilterSpec,
    JoinStep,
    TableProfile,
    _catalog,
    _measure_query_terms,
    _resolve_measure,
    _resolve_value_filters,
)


def test_unique_enrollment_phrase_binds_to_people_not_enrollment_event_id():
    """Counting unique enrollments must still use the learner population measure."""

    query = "show unique enrollment by department"
    measure_terms = _measure_query_terms(query)

    assert measure_terms == {"user", "student", "learner"}

    user_id = ColumnProfile(
        table="dashboard_agent_enrollment_fact",
        name="user_id",
        data_type="INTEGER",
        terms=frozenset({"user", "student", "learner", "id"}),
        is_identifier=True,
        is_time=False,
    )
    enrollment_id = ColumnProfile(
        table="dashboard_agent_enrollment_fact",
        name="enrollment_id",
        data_type="INTEGER",
        terms=frozenset({"enrollment", "id"}),
        is_identifier=True,
        is_time=False,
    )
    catalog = {
        "dashboard_agent_enrollment_fact": TableProfile(
            name="dashboard_agent_enrollment_fact",
            columns={column.name: column for column in (user_id, enrollment_id)},
        )
    }

    selected = _resolve_measure(
        catalog,
        {"unique", "enrollment", "department", "user"},
        measure_terms,
        value_validator=None,
        graph_hints={},
    )

    assert selected is user_id


def test_exact_value_filter_is_not_duplicated_across_same_named_join_columns():
    """An unqualified exact value should produce one filter, not one per joined table."""

    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table dashboard_agent_enrollment_fact (
            user_id integer,
            department varchar,
            status varchar,
            profile_id integer
        );
        create table dashboard_agent_user_dimension (
            profile_id integer,
            status varchar
        );
        insert into dashboard_agent_enrollment_fact values
            (1, 'A', 'active', 10),
            (2, 'B', 'inactive', 20);
        insert into dashboard_agent_user_dimension values
            (10, 'active'),
            (20, 'active');
        """
    )
    catalog = _catalog(con)
    fact = catalog["dashboard_agent_enrollment_fact"]
    dimension = catalog["dashboard_agent_user_dimension"]
    plan = AnalyticalPlan(
        intent="multi_dimension_comparison",
        base_table=fact.name,
        measure=fact.columns["user_id"],
        dimensions=[fact.columns["department"]],
        time_dimension=None,
        joins=[
            JoinStep(
                left_table=fact.name,
                right_table=dimension.name,
                left_key="profile_id",
                right_key="profile_id",
                confidence=1.0,
            )
        ],
        requested_terms=["users", "department", "status", "active"],
    )

    filters = _resolve_value_filters(con, catalog, plan, "show users by department where status is active")

    status_filters = [item for item in filters if item.column.name == "status" and item.value == "active"]
    assert len(status_filters) == 1

    con.close()
