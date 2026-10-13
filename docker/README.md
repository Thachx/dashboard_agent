# Docker deployment

Runs the full Dashboard Agent stack as Docker services with
`restart: unless-stopped`, so it comes back automatically after reboots
(24/7).

## Services

| Service  | Port | Replaces host unit                     |
|----------|------|----------------------------------------|
| backend  | 2024 | manual `langgraph dev` nohup process   |
| catalog  | 8090 | `dashboard-agent-catalog.service`      |
| ui       | 3000 | manual `next start` nohup process      |
| mcp      | 3001 | `dashboard-agent-mcp.service`          |
| ingest   | —    | `dashboard-agent-ingest.timer` (hourly :25) |

The **GPT-OSS SSH tunnel** (`gptoss-tunnel.service`) is intentionally not a
container — it needs a private SSH key and a private-network target. Keep it
running on the host; the backend reaches it via `host.docker.internal:18080`
(override with `GPTOSS_BASE_URL` in the environment).

## Prerequisites

- Docker + compose on the host
- The user running compose must be able to talk to the docker daemon
  (`sudo usermod -aG docker $USER`, then re-login)
- `backend/.env` in the repo root's `backend/` (S3 + LLM keys)
- `~/.aws` credentials for S3 ingestion
- `data/_warehouse/dashboard_agent.duckdb` present (the 66GB warehouse is
  bind-mounted, never copied into an image)

## Migrate (recommended)

```bash
bash docker/migrate-to-docker.sh          # build, stop host services, start Docker
bash docker/migrate-to-docker.sh --rollback
```

The script stops the host systemd units and the manual nohup processes,
starts the compose stack, and verifies the health endpoints. All state
(warehouse, langgraph store, graph JSON) is bind-mounted, so nothing is lost.

## Manual

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f backend
```

After migration, disable the replaced host units so they don't fight the
containers for the same ports:

```bash
systemctl --user disable --now dashboard-agent-catalog.service \
  dashboard-agent-mcp.service dashboard-agent-ingest.timer
# keep: gptoss-tunnel.service (SSH tunnel)
```

## Notes

- Containers run as uid 1006 (thacha) so the bind-mounted data, langgraph
  state, and AWS credentials are readable. If you run this as a different
  user, set `HOST_UID` in the environment (and rebuild is not needed —
  `user:` is applied at run time).
- The `ingest` service runs `docker/ingest_scheduler.py`, which fires the
  same pipeline at minute 25 of every hour (±2 min jitter) and does a
  catch-up run after a missed window — the Docker equivalent of the
  systemd timer's `Persistent=true`.
- Ports are bound to `127.0.0.1`; nginx (ask.lead.in.th) proxies to them as
  before. The MCP is reachable at `127.0.0.1:3001/mcp`.
- `MCP_PUBLIC_BASE_URL` controls the URL embedded in Markdown image results;
  set it to the public URL when exposing the MCP beyond localhost.
