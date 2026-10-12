# Dashboard Agent MCP Guidance

## Purpose

This repository exposes the existing Dashboard Graph Agent as an MCP Apps server. Keep analytics in the upstream Dashboard Agent and keep presentation inside the MCP App resource.

## Rules

- Do not copy database credentials, cloud credentials, or private data into this repository.
- Keep `DashboardDocument` generic and versioned. Never encode prompt-specific dashboards in the renderer.
- The model-facing MCP surface should remain small. Prefer one high-level dashboard tool over many schema-specific tools.
- MCP App rendering is progressive enhancement. Preserve useful text and structured-data responses for non-UI clients.
- Keep the UI resource self-contained and do not load scripts from public CDNs.
- Run `npm test` and `npm run build` after behavior changes.
- Browser-check the standalone fixture view at desktop and mobile widths after chart or layout changes.

## Integration boundary

- Upstream backend: configured by `DASHBOARD_AGENT_URL`.
- MCP transport: Streamable HTTP at `/mcp`, plus optional stdio.
- UI resource: `ui://dashboard-agent/dashboard-app.html`.
- Canonical result: `DashboardDocument` from `src/shared/dashboard-contract.ts`.

## Work log
- 2026-07-23: Built the Dashboard Agent MCP app server (Streamable HTTP at /mcp + optional stdio) with one high-level create_dashboard tool, the DashboardDocument contract, the self-contained ui:// app resource, configurable MCP host headers, a portable dashboard prompt, and an image fallback for clients without the UI app.
- 2026-07-23: Embedded Thai fonts (Sarabun via @expo-google-fonts) into dashboard previews, rendered previews with bundled multilingual fonts, embedded preview images directly in model responses, simplified the prompt/final-response flow, and kept dashboard routing instructions hidden from users.
- 2026-08-03: Rendered dashboard preview PNGs with @napi-rs/canvas (dashboard-image.ts) so charts work for any MCP client, not only the UI app.
- 2026-08-05: Extended the contract with dashboardFromTextResponse (upstream answers without an unsupported chart) and tuned the marker parser for LangGraph assistant text.
- 2026-08-13: Added dashboard-adapter.ts (LangGraph calls, timeout, fixture mode) plus its test suite.
- 2026-08-24: Reworked server.ts around @modelcontextprotocol/sdk + ext-apps: markdown image lines in tool text, preview URLs via a publishPreview hook, _meta.previewUrl(s) for the UI app, and bounded inline image content (max 6).
- 2026-08-25: Added the disk-backed preview store (data/previews, TTL sweep) so pasted chat history keeps rendering chart images after service restarts.
- 2026-09-12: Added the vitest setup/tests for the preview store (tests/setup.ts, dashboard-preview-store.test.ts).
NOTE: the 2026-08-03 to 2026-08-25 entries above were committed from the local development clone as "Improve dashboard rendering and response modes" (2d238e3) and "Persist previews and cancel timed-out agent runs" (fdc7efe).
