# Power BI Dashboard Agent

This repository provides a LangGraph chat agent that:

- reads arbitrary JSON objects from `s3://edx-nectec-demo/data` by default;
- embeds their structure into a persisted NetworkX node-link graph;
- retrieves graph context and optionally sends it through an OpenAI-compatible LLM;
- renders a configured Power BI report directly inside chat; and
- uses the existing guest chat session flow without Google OAuth.

## Configure

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item agent-chat-ui/.env.example agent-chat-ui/.env.local
```

Set AWS credentials through the normal boto3 chain: environment variables, an AWS profile, container credentials, or an IAM role. Do not put credentials in the repository.

For Power BI, set `POWER_BI_EMBED_URL`. A public/organization embed URL is rendered in an iframe. For app-owns-data embedding, also set `POWER_BI_REPORT_ID` and a short-lived `POWER_BI_ACCESS_TOKEN`; the UI uses Microsoft's Power BI JavaScript SDK.

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

Open `http://localhost:3000`. Ask the agent to `refresh the S3 data`, query a field/value, or `show the Power BI dashboard`.

## Data model

Each S3 object becomes a `dataset` node. JSON objects become `field` nodes, arrays become ordered `record` nodes, and scalar values are stored on their owning field/record nodes. `has_field` and `has_item` edges preserve the original hierarchy. The graph is stored at `backend/data/s3-json-graph.json` by default and refreshed using `GRAPH_REFRESH_SECONDS`.

## Verify

```powershell
cd backend
python -m pytest -q
python -m compileall src

cd ../agent-chat-ui
npm run build
```
