import json
import subprocess

from dashboard_agent.graphify_ingest import ingest_s3_with_graphify
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_source import JsonObject


class FakeSource:
    uri = "s3://edx-nectec-demo"

    def load(self):
        return [
            JsonObject(
                "data/sales.json",
                "etag",
                {"region": "Bangkok", "sales": 42},
            )
        ]


def test_ingest_s3_with_graphify_uses_graphify_graph_as_store(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, cwd, check, capture_output, text, timeout):
        commands.append(command)
        graphify_out = cwd / "graphify"
        graphify_out.mkdir()
        (graphify_out / "graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "graphify::dataset",
                            "label": "Sales Graphify Dataset",
                            "type": "dataset",
                            "summary": "Bangkok sales from Graphify graph",
                        }
                    ],
                    "links": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dashboard_agent.graphify_ingest.subprocess.run", fake_run)
    monkeypatch.setattr("dashboard_agent.graphify_ingest.shutil.which", lambda command: command)

    store = JsonGraphStore(tmp_path / "app-graph.json")
    stats = ingest_s3_with_graphify(
        FakeSource(),
        store,
        graphify_output_dir=tmp_path / "graphify-out",
    )

    assert stats["graphify_status"] == "ok"
    assert stats["graph_kind"] == "graphify"
    assert stats["nodes"] == 1
    assert (tmp_path / "graphify-out" / "graph.json").exists()
    assert commands[0][:2] == ["graphify", "extract"]
    assert commands[0][-1] == "--no-cluster"
    assert store.search("Bangkok")[0]["id"] == "graphify::dataset"


def test_ingest_s3_with_graphify_falls_back_to_update_without_semantic_key(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, cwd, check, capture_output, text, timeout):
        commands.append(command)
        if command[1] == "extract":
            return subprocess.CompletedProcess(command, 1, "", "No LLM API key found for semantic extraction")
        graphify_out = cwd / "graphify"
        graphify_out.mkdir()
        (graphify_out / "graph.json").write_text(
            json.dumps({"nodes": [{"id": "fallback", "label": "Fallback Graph", "type": "dataset"}], "links": []}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dashboard_agent.graphify_ingest.subprocess.run", fake_run)
    monkeypatch.setattr("dashboard_agent.graphify_ingest.shutil.which", lambda command: command)

    stats = ingest_s3_with_graphify(
        FakeSource(),
        JsonGraphStore(tmp_path / "app-graph.json"),
        graphify_output_dir=tmp_path / "graphify-out",
    )

    assert stats["graphify_status"] == "ok"
    assert [command[1] for command in commands] == ["extract", "update"]


def test_ingest_s3_with_graphify_can_disable_graphify(tmp_path, monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("graphify should not run when disabled")

    monkeypatch.setattr("dashboard_agent.graphify_ingest.subprocess.run", fail_run)

    stats = ingest_s3_with_graphify(
        FakeSource(),
        JsonGraphStore(tmp_path / "app-graph.json"),
        graphify_output_dir=tmp_path / "graphify-out",
        graphify_enabled=False,
    )

    assert stats["graphify_status"] == "disabled"
    assert stats["objects"] == 1
    assert stats["graph_kind"] == "fallback"
