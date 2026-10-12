import { demoDashboard } from "../shared/demo-dashboard.js";
import {
  DashboardDocumentSchema,
  dashboardFromTextResponse,
  type DashboardDocument,
} from "../shared/dashboard-contract.js";
import { assistantTextFromLangGraph, dashboardDocumentFromLangGraph } from "./marker-parser.js";

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

function applyLimits(
  document: DashboardDocument,
  request: DashboardRequest,
): DashboardDocument {
  const charts = document.charts.slice(0, request.maxCharts ?? document.charts.length);
  if (charts.length === document.charts.length) return document;
  const visible = new Set(charts.map((chart) => chart.id));
  return {
    ...document,
    charts,
    layout: document.layout.filter((block) => {
      if (block.type === "chart") return visible.has(block.slotId);
      return true;
    }),
  };
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

  // The Dashboard Agent backend is the single source of truth for the
  // dashboard document: it builds the document from its activity context and
  // attaches it to the assistant message (additional_kwargs.dashboard_document).
  // This server only validates and forwards it; it does not re-derive charts,
  // metrics, layout, or reasoning from the response text.
  const upstreamDocument = dashboardDocumentFromLangGraph(runResponse);
  if (upstreamDocument) {
    const document = DashboardDocumentSchema.parse(upstreamDocument);
    return applyLimits(document, request);
  }

  // Non-dashboard answers (plain text, no structured document) still get a
  // chart-free document so the MCP client receives a valid result.
  const assistantText = assistantTextFromLangGraph(runResponse);
  const textDocument = dashboardFromTextResponse(assistantText, {
    query: request.question,
    locale: request.locale,
    timezone: request.timezone,
  });
  return applyLimits(textDocument, request);
}
