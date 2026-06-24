import json
import re

from dashboard_agent.powerbi import PowerBiEmbed


def test_power_bi_payload_marker():
    marker = PowerBiEmbed("https://app.powerbi.com/reportEmbed", "report-1", "token").marker("Sales")
    payload = json.loads(re.search(r"<<<POWER_BI_DASHBOARD>>>\n(.*)\n<<<END_POWER_BI_DASHBOARD>>>", marker).group(1))
    assert payload["kind"] == "power-bi-dashboard"
    assert payload["reportId"] == "report-1"


def test_missing_power_bi_configuration_is_actionable():
    assert "POWER_BI_EMBED_URL" in PowerBiEmbed(None).marker()
