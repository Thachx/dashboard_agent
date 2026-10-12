import { describe, expect, it } from "vitest";

import {
  dashboardFromGraphPayload,
  dashboardFromTextResponse,
} from "../src/shared/dashboard-contract";

describe("dashboard contract adapter", () => {
  it("normalizes graph widget payloads without schema-specific field rules", () => {
    const dashboard = dashboardFromGraphPayload({
      title: "User overview",
      datasets: [{ key: "users", label: "Users", path: "parquet/users.parquet" }],
      activity: {
        summary: { totalRecords: 100, distinctUsers: 42, executionSource: "graph with duckdb fallback" },
        chartSlots: [{ id: "status", title: "Status", chartType: "donut", data: [{ label: "Active", value: 42 }] }],
        decisionTrace: [{ stage: "Plan", detail: "Matched a categorical distribution." }],
      },
    }, { query: "Show user status" });

    expect(dashboard.metrics).toHaveLength(2);
    expect(dashboard.charts[0]).toMatchObject({ id: "status", chartType: "donut" });
    expect(dashboard.sources[0].path).toBe("parquet/users.parquet");
    expect(dashboard.execution.source).toBe("hybrid");
  });

  it("normalizes non-chart analytical responses without inventing a chart", () => {
    const dashboard = dashboardFromTextResponse(
      "## User count\nThere are **42 active users**.",
      { query: "How many active users are there?", locale: "en", timezone: "Asia/Bangkok" },
    );

    expect(dashboard).toMatchObject({
      title: "User count",
      summary: "There are **42 active users**.",
      query: "How many active users are there?",
      locale: "en",
      timezone: "Asia/Bangkok",
      metrics: [],
      charts: [],
      layout: [],
    });
  });
});
