# Dashboard Agent Repository Guidance

## Working rules

- Use `C:\Users\manag\.codex\RTK.md`; prefix shell commands with `rtk`.
- Keep the dashboard data model generic. Do not hardcode assumptions about a specific JSON schema.
- Treat `s3://edx-nectec-demo/data` as the default source, configurable through environment variables.
- Never commit AWS, Azure, OpenAI, or Power BI credentials.
- Run backend tests and the frontend production build after behavior changes.

## Architecture

- `backend/` is a LangGraph agent that imports JSON objects from S3 into a persisted NetworkX node-link graph.
- `agent-chat-ui/` is the Next.js chat client. It renders Power BI payloads emitted by the agent.

## Work log

- 2026-06-22: Created the repository guidance file.
- 2026-06-22: Added the S3 JSON graph ingestion, graph retrieval agent, Power BI embed payload, dashboard chat renderer, environment examples, and tests.
- 2026-06-22: Removed Google OAuth controls and request metadata from the chat UI.
- 2026-06-22: Verified 5 backend tests, Python compilation, the Next.js production build, LangGraph `/info`, and a live `/runs/wait` request.
- 2026-06-22: Verified S3 listing access; ingestion is currently blocked because IAM user `lead` lacks `s3:GetObject` on the bucket data objects.
- 2026-06-23: Re-tested S3 JSON downloads from `s3://edx-nectec-demo/data`; listing still works, but `GetObject` is denied for MongoDB and MySQL JSON samples, so database JSON files cannot be downloaded until IAM grants object read access.
