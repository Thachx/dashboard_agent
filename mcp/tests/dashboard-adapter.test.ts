import { afterEach, describe, expect, it, vi } from "vitest";

import { createDashboard } from "../src/server/dashboard-adapter";

describe("dashboard adapter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.DASHBOARD_AGENT_FIXTURE;
    delete process.env.DASHBOARD_AGENT_URL;
    delete process.env.DASHBOARD_AGENT_TIMEOUT_MS;
  });

  it("cancels an upstream run when waiting times out", async () => {
    process.env.DASHBOARD_AGENT_TIMEOUT_MS = "5";
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/threads")) {
        return new Response(JSON.stringify({ thread_id: "thread-2" }), { status: 200 });
      }
      if (url.endsWith("/threads/thread-2/runs")) {
        return new Response(JSON.stringify({ run_id: "run-2" }), { status: 200 });
      }
      if (url.endsWith("/join")) {
        await new Promise((_resolve, reject) => setTimeout(() => reject(new DOMException("timed out", "TimeoutError")), 10));
      }
      return new Response("{}", { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(createDashboard({ question: "slow request" })).rejects.toThrow();

    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/runs/run-2/cancel"))).toBe(true);
  });

  it("returns a chart-free document when the upstream answer has no dashboard marker", async () => {
    const responses = [
      new Response(JSON.stringify({ thread_id: "thread-1" }), { status: 200 }),
      new Response(JSON.stringify({ run_id: "run-1" }), { status: 200 }),
      new Response(JSON.stringify({
        values: {
          messages: [
            { type: "human", content: "How many active users are there?" },
            { type: "ai", content: "There are 42 active users." },
          ],
        },
      }), { status: 200 }),
    ];
    const fetchMock = vi.fn(async () => responses.shift() ?? new Response("unexpected request", { status: 500 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createDashboard({ question: "How many active users are there?" });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(String(fetchMock.mock.calls[1])).toContain("/threads/thread-1/runs");
    expect(String(fetchMock.mock.calls[2])).toContain("/threads/thread-1/runs/run-1/join");
    expect(result).toMatchObject({
      title: "Analysis result",
      summary: "There are 42 active users.",
      charts: [],
      metrics: [],
      execution: { source: "unknown" },
    });
  });
  it("forwards the structured dashboard document built by the backend", async () => {
    const upstreamDocument = {
      schemaVersion: "1.0",
      dashboardId: "doc-1",
      query: "Show users by status",
      title: "User status",
      locale: "en",
      timezone: "Asia/Bangkok",
      generatedAt: "2026-10-12T00:00:00.000Z",
      summary: "Status with 2 plotted values.",
      metrics: [{ id: "totalRecords", label: "Activity rows", value: 100 }],
      charts: [
        { id: "status", title: "Status", chartType: "donut", field: "value", data: [{ label: "Active", value: 42 }] },
        { id: "trend", title: "Trend", chartType: "line", field: "value", data: [{ label: "A", value: 1 }] },
      ],
      layout: [
        { type: "metric", id: "totalRecords" },
        { type: "chart", slotId: "status" },
        { type: "chart", slotId: "trend" },
      ],
      sources: [{ id: "lead_etl_users", label: "Users" }],
      reasoningSummary: [{ stage: "Plan", detail: "Matched a categorical distribution." }],
      execution: { source: "graph", graphUsed: true, duckdbUsed: false },
    };
    const responses = [
      new Response(JSON.stringify({ thread_id: "thread-3" }), { status: 200 }),
      new Response(JSON.stringify({ run_id: "run-3" }), { status: 200 }),
      new Response(JSON.stringify({
        values: {
          messages: [
            { type: "human", content: "Show users by status" },
            {
              type: "ai",
              content: "<<<GRAPH_DASHBOARD_WIDGET>>>{}",
              additional_kwargs: { dashboard_document: upstreamDocument },
            },
          ],
        },
      }), { status: 200 }),
    ];
    const fetchMock = vi.fn(async () => responses.shift() ?? new Response("unexpected request", { status: 500 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createDashboard({ question: "Show users by status", maxCharts: 1 });

    expect(result.dashboardId).toBe("doc-1");
    expect(result.title).toBe("User status");
    expect(result.execution.source).toBe("graph");
    expect(result.charts).toHaveLength(1);
    expect(result.layout).toEqual([
      { type: "metric", id: "totalRecords" },
      { type: "chart", slotId: "status" },
    ]);
  });

});
