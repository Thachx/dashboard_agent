# Dashboard Agent Repository Guidance

## Working rules

- Use `C:\Users\manag\.codex\RTK.md`; prefix shell commands with `rtk`.
- Keep the dashboard data model generic. Do not hardcode assumptions about a specific JSON schema.
- Treat `s3://edx-nectec-demo` as the default source, configurable through environment variables.
- Never commit AWS, Azure, OpenAI, or other service credentials.
- Run backend tests and the frontend production build after behavior changes.

## Architecture

- `backend/` is a LangGraph agent that imports JSON objects from S3 into a persisted NetworkX node-link graph.
- `agent-chat-ui/` is the Next.js chat client. It renders graph dashboard widget payloads through the Power BI widget component.

## Work log

- 2026-06-22: Created the repository guidance file.
- 2026-06-22: Added the S3 JSON graph ingestion, graph retrieval agent, Power BI embed payload, dashboard chat renderer, environment examples, and tests.
- 2026-06-22: Removed Google OAuth controls and request metadata from the chat UI.
- 2026-06-22: Verified 5 backend tests, Python compilation, the Next.js production build, LangGraph `/info`, and a live `/runs/wait` request.
- 2026-06-22: Verified S3 listing access; ingestion is currently blocked because IAM user `lead` lacks `s3:GetObject` on the bucket data objects.
- 2026-06-23: Re-tested S3 JSON downloads from `s3://edx-nectec-demo/data`; listing still works, but `GetObject` is denied for MongoDB and MySQL JSON samples, so database JSON files cannot be downloaded until IAM grants object read access.
- 2026-06-24: Changed the Power BI chat renderer to a simple iframe widget, removed the unused `powerbi-client` frontend dependency, and updated setup docs to describe widget-only embedding.
- 2026-06-24: Simplified Power BI configuration to `POWER_BI_EMBED_URL` only; removed unused report ID and access token fields from backend settings and marker payloads.
- 2026-06-24: Made Power BI embedding optional; the agent no longer requires `POWER_BI_EMBED_URL` or emits a missing-configuration warning when it is absent.
- 2026-06-24: Replaced the remaining Power BI widget path with a graph dashboard widget generated from graph status and graph search results; dashboard graph is now the required dashboard surface.
- 2026-06-24: Switched the graph dashboard renderer to the Power BI widget component while keeping the payload backed by graph status and graph search results.
- 2026-06-24: Added OpenRouter LLM synthesis with one main model and two reserve models, plus optional direct OpenAI-compatible fallback after OpenRouter attempts fail.
- 2026-06-24: Added `LLM_MODE=auto|always|never` so `.env` controls when the agent calls the configured LLM.
- 2026-06-24: Added `LLM_PROVIDER=openrouter|openai|auto` so `.env` chooses OpenRouter, direct OpenAI-compatible models, or OpenRouter-then-OpenAI fallback.
- 2026-06-24: Added `C:\Users\manag\AppData\Roaming\Python\Python314\Scripts` to the user PATH so `langgraph` resolves from PowerShell.
- 2026-06-24: Added `C:\Users\manag\.headroom\bin\langgraph.cmd` shim that calls the LangGraph CLI with UTF-8 environment variables, because the Python 3.14 `langgraph.exe` launcher hit a cp1252 help-output crash.
- 2026-06-24: Repaired the frontend install to use `next@15.4.10` for the `src/app` router, added the required LangGraph thread `state_updated_at` placeholder field, and cast the external UI stream prop for current SDK types.
- 2026-06-24: Changed S3 ingestion to default to `s3://edx-nectec-demo`, added `ingest_s3_with_graphify(...)` for rebuilding the app graph and exporting Graphify's `graph.json`, and added Graphify configuration/test coverage.
- 2026-06-24: Reviewed sample workspace `C:\aws_download\project\workspaces`; applied the reusable generic part by indexing `.parquet` S3 objects as graph metadata alongside JSON, controlled by `S3_INCLUDE_EXTENSIONS`, without copying hardcoded credentials or schema-specific dashboard SQL.
- 2026-06-24: Live-smoked metadata-only ingest for `s3://edx-nectec-demo/parquet/fact_student_course` with Graphify disabled and a temp graph path; 136 Parquet objects produced 1,088 graph nodes and searchable metadata.
- 2026-06-24: Adapted the sample dashboard widget idea into the chat renderer by adding chart data to graph widget payloads and rendering graph size plus result relevance charts with existing `recharts`.
- 2026-06-24: Added temporary `backend/temp_ingest_graph.py` runner for direct S3-to-graph ingestion with CLI overrides for S3 URI, extensions, graph path, Graphify output, Graphify disablement, and post-ingest search.
- 2026-06-24: Fixed LangGraph slow-import warning by removing import-time graph store loading from `agent.py`; settings and `JsonGraphStore` now initialize lazily on first request/refresh while preserving test override compatibility.
- 2026-06-24: Aligned graph behavior with `C:\Program Files (x86)\Git\qa_agent`: Graphify `graph.json` is now loaded as the primary searchable graph after successful ingest, with the generic S3 JSON/metadata graph retained as fallback; Graphify uses `extract --out --no-cluster` and missing-semantic-key fallback to `update --no-cluster --force`.
- 2026-06-24: Fixed graph dashboard widget noise by splitting Graphify identifiers on underscores/dots, returning `source_file` and node text in graph search results, abbreviating large chart values, and truncating long matching-node text in the chat widget.
- 2026-06-24: Fixed dashboard prompt SSE stalls by adding fast exact-node graph search, capping fallback graph scans, skipping LLM synthesis for dashboard/widget requests, allowing forced/temp ingest to overwrite huge graph files without loading them, and replacing the 1.38GB local app graph with a 191KB assessment-criterion fallback graph.
- 2026-06-24: Fixed remaining chat SSE reconnect by disabling automatic S3 refresh on normal prompts; only explicit refresh/reload/rebuild prompts ingest S3 now. Verified `run_agent()` returns the assessment-criterion dashboard response in 0.8s.
- 2026-06-24: Added compact `s3-json-graph.json.search.json` sidecar index for fast request-time graph search; chat startup now loads the search index instead of deserializing graph JSON when current, reducing the assessment-criterion dashboard path to about 0.39s.
- 2026-06-24: Confirmed stale LangGraph server on port 2024 was still serving old 2.8M-node graph responses; started fresh backend on port 2025, pointed `agent-chat-ui/.env.local` at it, restarted Next on port 3000, and verified `/api/runs/wait` responds successfully in about 3.3s.
- 2026-06-25: Diagnosed `ConnectionError: Unable to connect to LangGraph server` from the Next `/api` passthrough; `agent-chat-ui/.env.local` pointed at dead port 2025 while the active `langgraph dev` process was listening on 2024, so `LANGGRAPH_API_URL` was changed back to `http://127.0.0.1:2024`.
- 2026-06-25: Reconnected the chat UI by launching Next dev on port 3000 against the live LangGraph backend on port 2024; verified `/api/info`, `/api/assistants/search`, and `/api/runs/wait` all return HTTP 200 through the Next passthrough.
