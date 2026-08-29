#!/usr/bin/env bash
set -euo pipefail

repo_dir="${DASHBOARD_AGENT_DIR:-$HOME/dashboard_agent}"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$unit_dir"
install -m 0644 "$repo_dir/deploy/systemd/dashboard-agent-ingest.service" "$unit_dir/"
install -m 0644 "$repo_dir/deploy/systemd/dashboard-agent-ingest.timer" "$unit_dir/"
systemctl --user daemon-reload

if [[ "${1:-}" == "--enable" ]]; then
    systemctl --user enable --now dashboard-agent-ingest.timer
fi

systemctl --user status dashboard-agent-ingest.timer --no-pager || true
