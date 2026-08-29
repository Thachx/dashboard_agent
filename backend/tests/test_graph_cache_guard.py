from __future__ import annotations

from pathlib import Path

from dashboard_agent.graph_dashboard_cache import (
    read_graph_dashboard_cache,
    write_graph_dashboard_cache,
)


def _activity(title: str) -> dict:
    return {
        "datasets": {"t": {"source": "t"}},
        "chartSlots": [{"id": "s", "data": [{"label": title, "value": 1}]}],
        "layoutSpec": {"title": title},
    }


def test_thai_only_questions_bypass_cache(tmp_path):
    graph = tmp_path / "g.json"
    written = write_graph_dashboard_cache(
        graph,
        "แสดงสถานะการเรียนของผู้ใช้ทั้งหมด",
        _activity("A"),
        source_version="v1",
    )
    assert written is False
    assert not graph.exists()


def test_distinct_thai_prompts_cannot_collide(tmp_path):
    graph = tmp_path / "g.json"
    first = "สรุปยอดผู้ใช้ตามหลักสูตร"
    second = "แสดงจำนวนผู้ลงทะเบียนรายเดือน"
    for prompt in (first, second):
        assert write_graph_dashboard_cache(graph, prompt, _activity(prompt), source_version="v1") is False


def test_ascii_question_roundtrip(tmp_path):
    graph = tmp_path / "g.json"
    question = "Show the top 8 departments by distinct users"
    assert write_graph_dashboard_cache(graph, question, _activity("T"), source_version="v1") is True
    hit = read_graph_dashboard_cache(graph, question, source_version="v1")
    assert hit is not None
    assert hit["layoutSpec"]["title"] == "T"
    miss = read_graph_dashboard_cache(graph, question + " now", source_version="v1")
    assert miss == {}
