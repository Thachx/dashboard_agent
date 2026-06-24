from __future__ import annotations

import json
from dataclasses import dataclass


START = "<<<POWER_BI_DASHBOARD>>>"
END = "<<<END_POWER_BI_DASHBOARD>>>"


@dataclass(frozen=True)
class PowerBiEmbed:
    embed_url: str | None
    report_id: str | None = None
    access_token: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.embed_url)

    def marker(self, title: str = "Power BI dashboard") -> str:
        if not self.embed_url:
            return "Power BI is not configured. Set POWER_BI_EMBED_URL in the backend environment."
        payload = {
            "version": 1,
            "kind": "power-bi-dashboard",
            "title": title,
            "embedUrl": self.embed_url,
            "reportId": self.report_id,
            "accessToken": self.access_token,
        }
        return f"{START}\n{json.dumps(payload)}\n{END}"
