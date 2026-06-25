import json
import re

from dashboard_agent.dashboard_widget import graph_dashboard_marker


def test_graph_dashboard_widget_payload_marker():
    marker = graph_dashboard_marker(
        status={"objects": 2, "nodes": 10, "edges": 9, "updated_at": 123.0},
        results=[{"path": "$.sales", "value": 42}],
        title="Graph",
    )
    payload = json.loads(
        re.search(
            r"<<<GRAPH_DASHBOARD_WIDGET>>>\n(.*)\n<<<END_GRAPH_DASHBOARD_WIDGET>>>",
            marker,
        ).group(1)
    )
    assert payload["kind"] == "graph-dashboard-widget"
    assert payload["status"]["nodes"] == 10
    assert payload["results"][0]["path"] == "$.sales"
    assert payload["charts"]["status"][1] == {"label": "Nodes", "value": 10}
    assert payload["charts"]["results"][0]["label"] == "$.sales"
