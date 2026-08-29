# Dashboard Graph Agent

This repository provides a LangGraph chat agent that:

- reads arbitrary JSON objects and Parquet metadata from `s3://edx-nectec-demo` by default;
- embeds JSON structure into a persisted NetworkX node-link graph and exports a Graphify graph when `graphify` is available;
- retrieves graph context and optionally sends it through an OpenAI-compatible LLM;
- renders dashboard widgets from graph status and matching graph nodes through the Power BI widget component; and
- uses the existing guest chat session flow without Google OAuth.

## Configure

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item agent-chat-ui/.env.example agent-chat-ui/.env.local
```

Set AWS credentials through the normal boto3 chain: environment variables, AWS profile, container credentials, or an IAM role. Do not put credentials in the repository.

No external dashboard embed configuration is required. The Power BI widget component is populated from the embedded JSON graph.

LLM synthesis is optional. `LLM_PROVIDER` in `backend/.env` chooses the provider: `openrouter` uses the OpenRouter main/reserve model chain, `openai` uses direct OpenAI-compatible settings only, and `auto` tries OpenRouter first then OpenAI. `LLM_MODE` controls when the chosen provider is called: `auto` uses the LLM only when graph context is found, `always` uses it whenever a model is configured, and `never` disables LLM calls.

For OpenRouter, set `OPENROUTER_API_KEY`, `OPENROUTER_MAIN_MODEL`, `OPENROUTER_RESERVE_MODEL_1`, and `OPENROUTER_RESERVE_MODEL_2`. The agent tries the main OpenRouter model first, then reserve model 1, then reserve model 2.

## Run

```powershell
cd backend
python -m pip install -e .
langgraph dev --port 2024
```

In another terminal:

```powershell
cd agent-chat-ui
npm install
npm run dev
```

Open `http://localhost:3000`. Ask the agent to `refresh S3 data`, query a field or value, or `show dashboard graph`.

## Hourly lead ETL pipeline

The production warehouse is populated from `s3://lead-etl` by selecting the
newest Parquet snapshot for every discovered dataset. Dataset names and schemas
come from the S3 paths and Parquet files; the loader does not keep a hardcoded
table list. Each dataset is published as a typed `lead_etl_*` DuckDB table and
also appears in the generic `unified_records` contract used by the Agent.

The first successful run preserves an existing unmanaged warehouse as a
timestamped `.legacy-*.bak` file. Later runs compare S3 key, ETag, and size with
`lead_etl_object_manifest`, replace only changed datasets in a temporary
database, and atomically publish the completed warehouse. A failed download or
Parquet read leaves the prior warehouse in place. The graph search index is
then rebuilt from the same warehouse snapshot.

Historical partitions can be backfilled once with the same schema contract:

```bash
.venv/bin/python run_s3_duckdb_pipeline.py --backfill-history
```

The backfill detects the source behavior from consecutive-file overlap. Full
snapshots continue to replace their current `lead_etl_*` table. Incremental
datasets are reconstructed across all partitions using inferred, validated
business keys, and future hourly deltas merge on those same keys. Exact row
versions are retained separately in `lead_etl_history_*` tables and the generic
`lead_etl_history_records` table. Object-level provenance is recorded in
`lead_etl_history_objects`; current analytical queries continue to use
`unified_records`, so historical versions do not double-count current metrics.

```bash
cd /home/thacha/dashboard_agent/backend
.venv/bin/python run_s3_duckdb_pipeline.py --s3-uri s3://lead-etl
```

Install the user timer to run at minute 25 of every hour, after the observed
Open edX, Bookroll, and Simulator exports have landed:

```bash
cd /home/thacha/dashboard_agent
./deploy/install-ingest-timer.sh --enable
systemctl --user list-timers dashboard-agent-ingest.timer
journalctl --user -u dashboard-agent-ingest.service -n 100 --no-pager
```

Set `S3_DATA_URI`, `DUCKDB_PATH`, `AWS_REGION`, and `GRAPH_PATH` in
`backend/.env` when deployment paths differ. AWS credentials continue to use
the normal boto3 provider chain.

## Data Model

Each S3 object becomes a `dataset` node. JSON objects become `field` nodes, arrays become ordered `record` nodes, and scalar values are stored on the owning field or record nodes. `has_field` and `has_item` edges preserve the original hierarchy.

The app graph is stored at `backend/data/s3-json-graph.json` by default and refreshed using `GRAPH_REFRESH_SECONDS`.

`dashboard_agent.graphify_ingest.ingest_s3_with_graphify(...)` is the callable ingestion function. It downloads JSON from the configured S3 URI, rebuilds the app graph database, writes the downloaded JSON corpus to a temporary directory, runs `graphify <corpus> --no-viz`, and copies Graphify's `graph.json` to `GRAPHIFY_OUTPUT_DIR` (`backend/data/graphify-out` by default). Set `GRAPHIFY_ENABLED=false` to rebuild only the app graph.

## Verify

```powershell
cd backend
python -m pytest -q
python -m compileall src
cd ../agent-chat-ui
npm run build
```
