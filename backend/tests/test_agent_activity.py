import json

from dashboard_agent import agent


class FakeStore:
    def __init__(self, path):
        self.path = path


def test_aggregate_cache_activity_binds_generic_distinct_time_buckets(tmp_path):
    graph_path = tmp_path / "s3-json-graph.json"
    cache_path = tmp_path / "full-scan-aggregates.json"
    cache_path.write_text(
        json.dumps(
            {
                "data/edx-elastic/ae-activity-data-stream.json": {
                    "full_scan_status": "ok",
                    "full_record_count": 300,
                    "full_counts_json": json.dumps(
                        {
                            "userID": [
                                {"label": "user-a", "value": 120},
                                {"label": "user-b", "value": 80},
                            ],
                            "event": [{"label": "page_view", "value": 200}],
                        }
                    ),
                    "full_time_buckets_json": json.dumps(
                        [{"label": "2026-06-24T10", "value": 180}]
                    ),
                    "full_distinct_time_buckets_json": json.dumps(
                        {
                            "customer_id": [
                                {"label": "2026-06-24T10", "value": 42},
                                {"label": "2026-06-24T11", "value": 58},
                            ]
                        }
                    ),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    activity = agent._aggregate_cache_activity(FakeStore(graph_path), "customer number over time")

    timeline = next(slot for slot in activity["chartSlots"] if slot["id"] == "distinctTime-customer-id")
    assert timeline["title"] == "Customer Id over time"
    assert timeline["sourceField"] == "customer_id"
    assert timeline["data"] == [
        {"label": "2026-06-24T10", "value": 42},
        {"label": "2026-06-24T11", "value": 58},
    ]
