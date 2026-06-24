import json

from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_source import JsonObject, parse_s3_uri


def test_parse_s3_uri():
    assert parse_s3_uri("s3://edx-nectec-demo/data") == ("edx-nectec-demo", "data")


def test_arbitrary_json_is_persisted_as_graph(tmp_path):
    path = tmp_path / "graph.json"
    store = JsonGraphStore(path)
    stats = store.rebuild(
        [JsonObject("data/sales.json", "abc", {"region": "Bangkok", "sales": 42, "items": [{"sku": "A1"}]})]
    )

    assert stats["objects"] == 1
    assert stats["nodes"] >= 6
    assert store.search("Bangkok")[0]["value"] == "Bangkok"
    assert store.search("sku A1")
    assert json.loads(path.read_text(encoding="utf-8"))["graph"]["format"] == "dashboard-agent-json-graph-v1"


def test_existing_graph_reopens(tmp_path):
    path = tmp_path / "graph.json"
    JsonGraphStore(path).rebuild([JsonObject("data/a.json", "1", {"metric": "retention"})])
    reopened = JsonGraphStore(path)
    assert reopened.status()["objects"] == 1
    assert reopened.search("retention")
