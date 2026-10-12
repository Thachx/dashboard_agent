import { demoDashboard } from "../shared/demo-dashboard.js";
import {
  dashboardFromGraphPayload,
  dashboardFromTextResponse,
  type DashboardDocument,
} from "../shared/dashboard-contract.js";
import { assistantTextFromLangGraph, tryParseGraphDashboardMarker } from "./marker-parser.js";

export interface DashboardRequest {
  question: string;
  locale?: string;
  timezone?: string;
  maxCharts?: number;
}

interface AdapterConfig {
  baseUrl: string;
  assistantId: string;
  timeoutMs: number;
  fixture: boolean;
}

function config(): AdapterConfig {
  return {
    baseUrl: (process.env.DASHBOARD_AGENT_URL ?? "http://127.0.0.1:2024").replace(/\/$/, ""),
    assistantId: process.env.DASHBOARD_AGENT_ASSISTANT_ID ?? "dashboard_agent",
    timeoutMs: Number(process.env.DASHBOARD_AGENT_TIMEOUT_MS ?? 120000),
    fixture: /^(1|true|yes)$/i.test(process.env.DASHBOARD_AGENT_FIXTURE ?? ""),
  };
}

async function jsonRequest(url: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
  const response = await fetch(url, {
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
    signal: AbortSignal.timeout(timeoutMs),
  });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Dashboard Agent request failed (${response.status}): ${body.slice(0, 500)}`);
  }
  return body ? JSON.parse(body) as unknown : {};
}

async function cancelRun(baseUrl: string, threadId: string, runId: string): Promise<void> {
  try {
    await jsonRequest(
      `${baseUrl}/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST", body: "{}" },
      5_000,
    );
  } catch {
    // Preserve the original request failure; cancellation is best-effort cleanup.
  }
}

export async function createDashboard(request: DashboardRequest): Promise<DashboardDocument> {
  const settings = config();
  if (settings.fixture) {
    return {
      ...demoDashboard,
      dashboardId: crypto.randomUUID(),
      query: request.question,
      locale: request.locale ?? demoDashboard.locale,
      timezone: request.timezone ?? demoDashboard.timezone,
      generatedAt: new Date().toISOString(),
      charts: demoDashboard.charts.slice(0, request.maxCharts ?? demoDashboard.charts.length),
      execution: { source: "fixture", graphUsed: true, duckdbUsed: true },
    };
  }

  const threadResponse = await jsonRequest(
    `${settings.baseUrl}/threads`,
    { method: "POST", body: "{}" },
    settings.timeoutMs,
  );
  const threadId = String((threadResponse as Record<string, unknown>).thread_id ?? "");
  if (!threadId) throw new Error("Dashboard Agent did not return a thread_id.");

  const createdRun = await jsonRequest(
    `${settings.baseUrl}/threads/${encodeURIComponent(threadId)}/runs`,
    {
      method: "POST",
      body: JSON.stringify({
        assistant_id: settings.assistantId,
        input: { messages: [{ role: "user", content: request.question }] },
      }),
    },
    settings.timeoutMs,
  );
  const runId = String((createdRun as Record<string, unknown>).run_id ?? "");
  if (!runId) throw new Error("Dashboard Agent did not return a run_id.");

  let runResponse: unknown;
  try {
    runResponse = await jsonRequest(
      `${settings.baseUrl}/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/join`,
      { method: "GET" },
      settings.timeoutMs,
    );
  } catch (error) {
    await cancelRun(settings.baseUrl, threadId, runId);
    throw error;
  }
  const assistantText = assistantTextFromLangGraph(runResponse);
  const graphPayload = tryParseGraphDashboardMarker(assistantText);
  if (graphPayload === undefined) {
    return dashboardFromTextResponse(assistantText, {
      query: request.question,
      locale: request.locale,
      timezone: request.timezone,
    });
  }
  const dashboard = dashboardFromGraphPayload(graphPayload, {
    query: request.question,
    locale: request.locale,
    timezone: request.timezone,
  });
  return {
    ...dashboard,
    charts: dashboard.charts.slice(0, request.maxCharts ?? dashboard.charts.length),
  };
}
