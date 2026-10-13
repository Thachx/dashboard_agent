#!/usr/bin/env bash
# Migrate the host systemd deployment to the Dockerized stack.
#
# Run from the repo root:  bash docker/migrate-to-docker.sh
#
# Steps:
#   1. Preflight: docker available, ports free-able, data present.
#   2. Build the images.
#   3. Stop the host services (systemd + manual nohup processes) so the
#      ports are free. State (data/, .langgraph_api/) is bind-mounted, so
#      nothing is lost.
#   4. Start the compose stack (restart: unless-stopped -> 24/7).
#   5. Verify health endpoints.
#
# The GPT-OSS SSH tunnel (gptoss-tunnel.service) is left RUNNING: it needs a
# private key + private network and is reached via host.docker.internal.
#
# Rollback:  bash docker/migrate-to-docker.sh --rollback
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="docker compose -f ${REPO_ROOT}/docker/docker-compose.yml"

rollback() {
  echo ">> Rolling back: stopping Docker stack, restoring host services..."
  ${COMPOSE} down
  # langgraph + UI were manual nohup processes; restart them the same way.
  (cd "${REPO_ROOT}/backend" && setsid nohup .venv/bin/python .venv/bin/langgraph dev \
      --port 2024 --no-browser --no-reload --n-jobs-per-worker 50 \
      > /tmp/lg_2024.log 2>&1 < /dev/null &)
  (cd "${REPO_ROOT}/agent-chat-ui" && setsid nohup npm run start \
      > /tmp/next_start_docker_rollback.log 2>&1 < /dev/null &)
  systemctl --user start dashboard-agent-catalog.service dashboard-agent-mcp.service
  systemctl --user start dashboard-agent-ingest.timer
  echo ">> Rollback done. Host services restored (verify ports 2024/3000/8090/3001)."
  exit 0
}

if [[ "${1:-}" == "--rollback" ]]; then rollback; fi

echo ">> Preflight"
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker info >/dev/null 2>&1 || { echo "cannot talk to the docker daemon (are you in the docker group?)"; exit 1; }
[[ -f "${REPO_ROOT}/backend/.env" ]] || { echo "backend/.env missing"; exit 1; }
[[ -f "${REPO_ROOT}/data/_warehouse/dashboard_agent.duckdb" ]] || { echo "warehouse duckdb missing"; exit 1; }

echo ">> Building images"
${COMPOSE} build

echo ">> Stopping host services (state is bind-mounted; nothing is lost)"
systemctl --user stop dashboard-agent-catalog.service dashboard-agent-mcp.service 2>/dev/null || true
systemctl --user stop dashboard-agent-ingest.timer 2>/dev/null || true
# Manual nohup processes (langgraph, next) — patterns that do not match this script.
pkill -f 'langgraph dev --port 202[4]' 2>/dev/null || true
pkill -f 'next [s]tart' 2>/dev/null || true
pkill -f 'next-[s]erver' 2>/dev/null || true
sleep 5
for port in 2024 3000 8090 3001; do
  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "port ${port} still busy — aborting (find and stop the process first)"
    exit 1
  fi
done
echo ">> Ports free"

echo ">> Starting Docker stack"
${COMPOSE} up -d

echo ">> Waiting for health checks"
for i in $(seq 1 30); do
  ok=1
  curl -sf http://127.0.0.1:2024/ok >/dev/null 2>&1 || ok=0
  curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 || ok=0
  curl -sf http://127.0.0.1:3000/api/health >/dev/null 2>&1 || ok=0
  curl -sf http://127.0.0.1:3001/health >/dev/null 2>&1 || ok=0
  [[ $ok -eq 1 ]] && break
  sleep 5
done
${COMPOSE} ps

echo ">> Verifying"
curl -sf http://127.0.0.1:8090/health && echo
curl -sf http://127.0.0.1:3001/health && echo
curl -sf http://127.0.0.1:3000/api/health && echo
echo ">> Done. Stack is up with restart: unless-stopped (24/7)."
echo ">> GPT-OSS tunnel: confirm 'systemctl --user is-active gptoss-tunnel' == active."
