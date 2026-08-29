from dashboard_agent.graph_dashboard_cache import (
    read_graph_dashboard_cache,
    write_graph_dashboard_cache,
)


def _activity():
    return {
        "datasets": {"fact": {"object_type": "duckdb_join_plan"}},
        "chartSlots": [{"id": "users", "data": [{"label": "A", "value": 1}]}],
    }


def test_dashboard_cache_reuses_exact_normalized_question(tmp_path):
    graph_path = tmp_path / "graph.json"
    assert write_graph_dashboard_cache(graph_path, "Show Users by Province", _activity())

    cached = read_graph_dashboard_cache(graph_path, "  show users by province  ")

    assert cached["summary"]["reasoningSource"] == "cache"


def test_dashboard_cache_does_not_reuse_near_similar_dashboard_requests(tmp_path):
    graph_path = tmp_path / "graph.json"
    question = "show users by province"
    assert write_graph_dashboard_cache(graph_path, question, _activity())

    near_similar_questions = (
        "show active users by province",       # filter/constraint differs
        "show users by province and institute",  # series differs
        "show users by province with courses",   # join differs
        "show users by province as two charts",  # requested chart count differs
        "แสดงจำนวนผู้ใช้งานแยกตามจังหวัด",       # locale differs
    )
    for near_similar in near_similar_questions:
        assert read_graph_dashboard_cache(graph_path, near_similar) == {}
