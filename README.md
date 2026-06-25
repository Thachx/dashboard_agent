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
