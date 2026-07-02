import json

import duckdb

from dashboard_agent.duckdb_ingest import ingest_duckdb_database
from dashboard_agent.graph_store import JsonGraphStore


def test_ingest_unified_duckdb_database_builds_source_graph(tmp_path):
    db_path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE unified_records (
            source_path VARCHAR,
            source_format VARCHAR,
            source_table VARCHAR,
            record_index BIGINT,
            payload_json VARCHAR,
            ingested_at TIMESTAMP
        )
        """
    )
    con.executemany(
        "INSERT INTO unified_records VALUES (?, ?, ?, ?, ?, now())",
        [
            (
                "edx-mysql/auth_user.json",
                "json",
                "edx_mysql_auth_user",
                0,
                json.dumps({"id": 1, "username": "alice", "is_active": True}),
            ),
            (
                "edx-mysql/auth_user.json",
                "json",
                "edx_mysql_auth_user",
                1,
                json.dumps({"id": 2, "username": "bob", "is_active": False}),
            ),
            (
                "raw-parquet/dim_user.parquet",
                "parquet",
                "raw_parquet_dim_user",
                0,
                json.dumps({"user_id": 1, "name": "Alice"}),
            ),
        ],
    )
    con.close()

    store = JsonGraphStore(tmp_path / "graph.json")
    stats = ingest_duckdb_database(store, db_path, sample_records_per_source=2)

    assert stats["graph_kind"] == "duckdb-unified"
    assert stats["duckdb_source_count"] == 2
    assert stats["objects"] == 2
    assert store.search("auth_user username alice")
    assert store.search("dim_user parquet")


def test_ingest_generic_duckdb_table_builds_schema_graph(tmp_path):
    db_path = tmp_path / "generic.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE metrics (metric VARCHAR, value INTEGER)")
    con.execute("INSERT INTO metrics VALUES ('retention', 42)")
    con.close()

    store = JsonGraphStore(tmp_path / "graph.json")
    stats = ingest_duckdb_database(store, db_path, table="metrics")

    assert stats["graph_kind"] == "duckdb-table"
    assert stats["duckdb_source_count"] == 1
    assert store.search("retention metric")
    assert store.search("columns value")


def test_ingest_unified_duckdb_prefers_source_summary_view(tmp_path):
    db_path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE unified_records (
            source_path VARCHAR,
            source_format VARCHAR,
            source_table VARCHAR,
            record_index BIGINT,
            payload_json VARCHAR,
            ingested_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO unified_records VALUES
        ('large/source.json', 'json', 'large_source', 0, '{"field":"sample"}', now())
        """
    )
    con.execute(
        """
        CREATE VIEW source_summary AS
        SELECT 'large/source.json' AS source_path,
               'json' AS source_format,
               'large_source' AS source_table,
               999999 AS records
        """
    )
    con.close()

    store = JsonGraphStore(tmp_path / "graph.json")
    stats = ingest_duckdb_database(store, db_path)

    assert stats["duckdb_source_count"] == 1
    node = store.graph.nodes["object::duckdb/unified_records/large/source.json"]
    assert node["size"] == 999999
