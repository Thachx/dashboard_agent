import json

from dashboard_agent.config import _split_extensions
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_source import JsonObject, parse_s3_uri


def test_parse_s3_uri():
    assert parse_s3_uri("s3://edx-nectec-demo") == ("edx-nectec-demo", "")
    assert parse_s3_uri("s3://edx-nectec-demo/data") == ("edx-nectec-demo", "data")


def test_split_extensions_normalizes_config():
    assert _split_extensions("json, .parquet,,CSV") == (".json", ".parquet", ".csv")


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


def test_parquet_metadata_is_persisted_as_graph(tmp_path):
    path = tmp_path / "graph.json"
    store = JsonGraphStore(path)
    stats = store.rebuild(
        [
            JsonObject(
                "parquet/fact_student_course.parquet",
                "etag",
                {
                    "s3_uri": "s3://edx-nectec-demo/parquet/fact_student_course.parquet",
                    "object_type": "parquet",
                    "size_bytes": 2048,
                },
                object_type="parquet",
                size=2048,
            )
        ]
    )

    assert stats["objects"] == 1
    assert store.search("fact_student_course parquet")
    assert store.graph.nodes["object::parquet/fact_student_course.parquet"]["object_type"] == "parquet"


def test_existing_graph_reopens(tmp_path):
    path = tmp_path / "graph.json"
    JsonGraphStore(path).rebuild([JsonObject("data/a.json", "1", {"metric": "retention"})])
    reopened = JsonGraphStore(path)
    assert reopened.status()["objects"] == 1
    assert reopened.search("retention")


def test_graphify_links_graph_can_replace_store(tmp_path):
    graphify_path = tmp_path / "graphify.json"
    graphify_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "dataset::fact_student_course",
                        "label": "fact_student_course",
                        "type": "dataset",
                        "summary": "Student course dashboard data",
                    },
                    {"id": "column::school_name", "label": "school_name", "type": "column"},
                ],
                "links": [
                    {
                        "source": "dataset::fact_student_course",
                        "target": "column::school_name",
                        "relation": "has_column",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = JsonGraphStore(tmp_path / "app-graph.json")
    stats = store.replace_from_graph_json(graphify_path)

    assert stats["graph_kind"] == "graphify"
    assert stats["nodes"] == 2
    assert store.search("student dashboard")[0]["id"] == "dataset::fact_student_course"
    assert store.neighbors("dataset::fact_student_course") == ["school_name"]


def test_graphify_search_uses_source_file_and_split_identifiers(tmp_path):
    graphify_path = tmp_path / "graphify.json"
    graphify_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "edx_mysql_assessment_criterion",
                        "label": "assessment_criterion.json",
                        "file_type": "code",
                        "source_file": "data/edx-mysql/assessment_criterion.json",
                        "source_location": "L1",
                    }
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    store = JsonGraphStore(tmp_path / "app-graph.json")
    store.replace_from_graph_json(graphify_path)
    result = store.search("show dashboard graph for edx mysql assessment criterion", 1)[0]

    assert result["id"] == "edx_mysql_assessment_criterion"
    assert result["source"] == "data/edx-mysql/assessment_criterion.json"
    assert "assessment_criterion.json" in result["text"]
